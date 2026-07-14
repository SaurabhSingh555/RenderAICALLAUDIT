# -*- coding: utf-8 -*-
"""
CallAI Analytics - Streamlit App (SaaS Edition)
================================================
Flow:
  1. Silent login to CRM (fixed credentials, no login screen)
  2. Pick client, date range
  3. Fetch CDR report (auto-filtered by company_id)
  4. REMOVE VDCL AGENT CALLS (abandoned calls) - automatically filtered out
  5. Show ONE filtered data table with duration first, S.No, other columns
  6. Pick call-type and count
  7. Choose sort order
  8. Run VAD (Silero) to get Talk Time / Silence / Dead Air / Longest Silence
  9. Transcribe each call with Groq Whisper → RoBERTa sentiment analysis → add Sentiment column
 10. Download final Excel report with two sheets: Call Report + Agent Analytics
 11. Agent-wise analysis sheet included (updated categories: Short<2min, Medium 2-5min, Large>5min)
"""

import os
import re
import io
import time
import shutil
import subprocess
import tempfile
from datetime import date, timedelta
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import soundfile as sf
import librosa
from bs4 import BeautifulSoup
import streamlit as st
import torch
import groq

# ============================================================
# PAGE CONFIG + SAAS-STYLE THEME
# ============================================================
st.set_page_config(
    page_title="CallAI · Talk-Time + Sentiment",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", Inter, Roboto, Arial, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    .callai-hero {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .callai-hero h1 {
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 4px 0;
        color: white;
    }
    .callai-hero p {
        font-size: 15px;
        margin: 0;
        opacity: 0.9;
    }

    .step-card {
        background: #FFFFFF;
        border: 1px solid #ECECF4;
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(20, 20, 43, 0.04);
    }
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #4F46E5;
        color: white;
        font-weight: 700;
        font-size: 14px;
        margin-right: 10px;
    }
    .step-title {
        font-size: 17px;
        font-weight: 700;
        color: #14142B;
        display: inline-flex;
        align-items: center;
        margin-bottom: 6px;
    }
    .step-subtitle {
        color: #6E7191;
        font-size: 13.5px;
        margin: 0 0 16px 40px;
    }

    .metric-pill {
        background: #F5F4FF;
        border: 1px solid #E4E1FF;
        border-radius: 14px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-pill .value {
        font-size: 24px;
        font-weight: 800;
        color: #4F46E5;
    }
    .metric-pill .label {
        font-size: 12.5px;
        color: #6E7191;
        margin-top: 2px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        border: none;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    }
    div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        background: linear-gradient(135deg, #16A34A 0%, #22C55E 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.4rem;
    }

    div[role="radiogroup"] label {
        border: 1px solid #E4E1FF;
        padding: 6px 14px;
        border-radius: 20px;
        margin-right: 6px;
    }

    .status-banner-ok {
        background: #ECFDF5;
        border: 1px solid #6EE7B7;
        color: #065F46;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 13.5px;
    }
    .status-banner-warning {
        background: #FEF3C7;
        border: 1px solid #FCD34D;
        color: #92400E;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 13.5px;
    }
    
    .agent-card {
        background: #F8F7FF;
        border-left: 4px solid #4F46E5;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .agent-card .agent-name {
        font-weight: 700;
        color: #14142B;
        font-size: 15px;
    }
    .agent-card .agent-stats {
        color: #6E7191;
        font-size: 13px;
        margin-top: 4px;
    }
    .agent-card .highlight {
        color: #4F46E5;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="callai-hero">
    <h1>📞 CallAI · Talk-Time + Sentiment</h1>
    <p>Pick a client, fetch calls, filter, get Talk-Time / Silence / Dead-Air, and now also analyse sentiment with Groq Whisper + RoBERTa.</p>
</div>
""", unsafe_allow_html=True)

CRM_BASE = "https://crmapi.dialdesk.in"
LOGIN_URL = f"{CRM_BASE}/auth/login"
CDR_URL = f"{CRM_BASE}/report/cdr_report"

# ============================================================
# ⚠️ FIXED CRM CREDENTIALS - Loaded from Streamlit Secrets (with fallback)
# ============================================================
try:
    CRM_EMAIL = st.secrets["CRM_EMAIL"]
    CRM_PASSWORD = st.secrets["CRM_PASSWORD"]
except:
    CRM_EMAIL = "ispark@dialdesk.in"
    CRM_PASSWORD = "1234"

# ============================================================
# GROQ API KEY - load from secrets
# ============================================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = ""

# ============================================================
# ⚠️ CLIENTS - name -> company_id (edit this dict to add/remove clients)
# ============================================================
CLIENTS = {
    "Weebo": "687",
    "Hari Om Pvt Ltd": "689",
    "F1 INFO SOLUTION": "609",
    "Saatvik": "663",
    "Fortum Charge": "395",
    "Alphanso": "629",
}

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
defaults = {
    "token": None,
    "cdr_df": None,
    "cdr_client": None,
    "final_df": None,
    "agent_analytics_df": None,
    "vdcl_removed": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# CRM FUNCTIONS
# ============================================================
def do_login():
    resp = requests.post(
        LOGIN_URL,
        json={"email": CRM_EMAIL, "password": CRM_PASSWORD},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30,
        proxies=None,
    )
    resp.raise_for_status()
    data = resp.json()
    token = (
        data.get("token")
        or data.get("access_token")
        or (data.get("data", {}) or {}).get("token")
    )
    if not token:
        raise RuntimeError(f"Login response had no token field: {data}")
    st.session_state["token"] = token
    return token

def get_valid_token():
    if not st.session_state.get("token"):
        with st.spinner("Signing in..."):
            do_login()
    return st.session_state["token"]

def fetch_cdr(payload, retry_on_401=True):
    token = get_valid_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    resp = requests.post(CDR_URL, json=payload, headers=headers, timeout=120, proxies=None)
    if resp.status_code == 401 and retry_on_401:
        do_login()
        return fetch_cdr(payload, retry_on_401=False)
    return resp

# ============================================================
# RECORDING DOWNLOAD FUNCTIONS (UPDATED)
# ============================================================
def html_recording_to_direct_url(webform_url, retries=3):
    """
    Enhanced version: checks Content-Type for audio/video.
    If the URL directly returns an audio file (even without extension), it returns the final URL.
    Otherwise, parses HTML for audio tags, iframes, etc.
    """
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    audio_exts = (".mp3", ".wav", ".m4a", ".mp4")
    for attempt in range(retries):
        try:
            # Use stream=True so we can check headers without downloading full body
            resp = session.get(webform_url, headers=headers, timeout=30, stream=True)
            resp.raise_for_status()
            
            # 1. If Content-Type indicates audio/video, return the final URL
            content_type = resp.headers.get('content-type', '').lower()
            if 'audio' in content_type or 'video' in content_type:
                return resp.url
            
            # 2. If final URL ends with an audio extension, accept it
            if resp.url.lower().endswith(audio_exts):
                return resp.url
            
            # 3. Otherwise, assume it's HTML – we need to read the body
            # Because we used stream=True, we can still access resp.text
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            
            # ----- HTML parsing (same as before) -----
            for tag in soup.find_all(["audio", "video"]):
                src = tag.get("src")
                if src and any(ext in src.lower() for ext in audio_exts):
                    return urljoin(webform_url, src)
            for tag in soup.find_all("source"):
                src = tag.get("src")
                if src and any(ext in src.lower() for ext in audio_exts):
                    return urljoin(webform_url, src)
            for div in soup.find_all(attrs={"data-recording": True}):
                for attr in ["data-recording", "data-url", "data-src", "data-file"]:
                    url = div.get(attr)
                    if url:
                        return urljoin(webform_url, url)
            patterns = [
                r'https?://[^\s"\']+\.(?:mp3|wav|m4a)',
                r'//[^\s"\']+\.(?:mp3|wav|m4a)',
                r'/[^\s"\']+\.(?:mp3|wav|m4a)',
            ]
            for pattern in patterns:
                m = re.search(pattern, html, re.IGNORECASE)
                if m:
                    match = m.group()
                    if match.startswith("//"):
                        return "https:" + match
                    if match.startswith("/"):
                        return urljoin(webform_url, match)
                    return match
            js_patterns = [
                r'recordingUrl\s*[:=]\s*["\']([^"\']+)["\']',
                r'audioUrl\s*[:=]\s*["\']([^"\']+)["\']',
                r'fileUrl\s*[:=]\s*["\']([^"\']+)["\']',
                r'src\s*[:=]\s*["\']([^"\']+\.(?:mp3|wav|m4a))["\']',
            ]
            for pattern in js_patterns:
                m = re.search(pattern, html, re.IGNORECASE)
                if m:
                    return urljoin(webform_url, m.group(1))
            iframe = soup.find("iframe")
            if iframe and iframe.get("src"):
                iframe_src = urljoin(webform_url, iframe.get("src"))
                return html_recording_to_direct_url(iframe_src, retries=retries - 1)
            meta_refresh = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
            if meta_refresh and meta_refresh.get("content"):
                m = re.search(r"url=([^;]+)", meta_refresh.get("content"), re.IGNORECASE)
                if m:
                    return html_recording_to_direct_url(urljoin(webform_url, m.group(1)), retries=retries - 1)
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                if any(ext in href.lower() for ext in audio_exts):
                    return urljoin(webform_url, href)
            # Nothing found
            return None
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1)
    return None

def resolve_audio_url(recording_url):
    if not isinstance(recording_url, str) or not recording_url.strip():
        return None
    recording_url = recording_url.strip()
    if recording_url.lower().endswith((".mp3", ".wav", ".m4a", ".mp4")):
        return recording_url
    return html_recording_to_direct_url(recording_url)

# ============================================================
# FLEXIBLE COLUMN MAPPING
# ============================================================
COLUMN_CANDIDATES = {
    "date": ["call_date", "CallDate", "Date"],
    "time": ["start_time", "Time", "StartTime"],
    "agent_name": ["full_name", "AgentName", "agent", "Agent Name", "agent_name"],
    "call_from": ["phone_number", "PhoneNumber", "Call From"],
    "recording": ["Recording", "RecordingUrl", "RecordingURL", "recording_url"],
}

def find_column(df, keys):
    lower_map = {c.lower(): c for c in df.columns}
    for key in keys:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None

def parse_duration_series_to_seconds(series):
    s = series.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    needs_time_parse = numeric.isna() & s.str.contains(":", na=False)
    if needs_time_parse.any():
        def to_seconds(val):
            parts = val.split(":")
            try:
                parts = [float(p) for p in parts]
            except ValueError:
                return np.nan
            if len(parts) == 3:
                h, m, sec = parts
                return h * 3600 + m * 60 + sec
            elif len(parts) == 2:
                m, sec = parts
                return m * 60 + sec
            return np.nan
        numeric.loc[needs_time_parse] = s.loc[needs_time_parse].apply(to_seconds)
    return numeric

def resolve_duration_column(df):
    candidates_in_order = [
        ("call_duration", "sec"),
        ("call_duration1", "sec"),
        ("CallDurationSecond", "sec"),
        ("Talkduration", "sec"),
        ("CallDurationMinute", "min"),
    ]
    best_col, best_seconds, best_score = None, None, -1
    for name, unit in candidates_in_order:
        col = find_column(df, [name])
        if not col:
            continue
        seconds = parse_duration_series_to_seconds(df[col])
        if unit == "min":
            seconds = seconds * 60
        non_null = seconds.notna().sum()
        non_zero = (seconds.fillna(0) > 0).sum()
        score = non_zero
        if non_null > 0 and score > best_score:
            best_col, best_seconds, best_score = col, seconds.fillna(0), score
    return best_col, best_seconds

def fmt_hms(total_seconds):
    if total_seconds is None or (isinstance(total_seconds, float) and np.isnan(total_seconds)):
        return "-"
    total_seconds = int(round(total_seconds))
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def filter_out_vdcl_calls(df):
    if df is None or len(df) == 0:
        return df, 0
    
    agent_col = None
    if 'Agent Name' in df.columns:
        agent_col = 'Agent Name'
    elif 'AgentName' in df.columns:
        agent_col = 'AgentName'
    elif 'agent' in df.columns:
        agent_col = 'agent'
    elif 'full_name' in df.columns:
        agent_col = 'full_name'
    elif 'agent_name' in df.columns:
        agent_col = 'agent_name'
    else:
        for col in df.columns:
            col_lower = col.lower()
            if 'agent name' in col_lower or 'agentname' in col_lower or 'full_name' in col_lower:
                agent_col = col
                break
    
    if agent_col is None:
        return df, 0
    
    agent_values = df[agent_col].fillna('').astype(str)
    mask = agent_values.str.contains('VDCL', case=False, na=False, regex=False)
    removed_count = mask.sum()
    
    if removed_count > 0:
        removed_samples = df[mask][agent_col].head(5).tolist()
        st.session_state['vdcl_removed'] = removed_count
        sample_str = ', '.join(str(x) for x in removed_samples[:3])
        st.markdown(
            f'<div class="status-banner-warning">🗑️ Removed {removed_count} VDCL (abandoned) call(s) from Agent column "{agent_col}". '
            f'Samples: {sample_str}</div>',
            unsafe_allow_html=True
        )
    
    filtered_df = df[~mask].copy()
    return filtered_df, removed_count

# ============================================================
# AGENT ANALYTICS FUNCTION (UPDATED CATEGORIES)
# ============================================================
def generate_agent_analytics(df, duration_col='_duration_sec'):
    """
    Generate comprehensive agent-wise analytics with updated categories:
    - Short: < 2 min
    - Medium: 2 to 5 min
    - Large: > 5 min
    """
    if df is None or len(df) == 0:
        return None
    
    # Find agent column
    agent_col = None
    for col in ['full_name', 'agent', 'Agent Name', 'AgentName', 'agent_name']:
        if col in df.columns:
            agent_col = col
            break
    
    if agent_col is None:
        return None
    
    # Create a copy with the duration column
    df_copy = df.copy()
    if duration_col not in df_copy.columns:
        return None
    
    # Categorize calls (updated)
    def categorize_call(duration):
        if duration < 120:          # < 2 min
            return 'Short'
        elif duration <= 300:       # 2 to 5 min
            return 'Medium'
        else:                       # > 5 min
            return 'Large'
    
    df_copy['Call_Category'] = df_copy[duration_col].apply(categorize_call)
    
    # Group by agent
    agent_stats = df_copy.groupby(agent_col).agg({
        duration_col: ['count', 'mean', 'sum'],
        'Call_Category': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    # Flatten column names
    agent_stats.columns = ['Agent', 'Total_Calls', 'Avg_Duration', 'Total_Duration', 'Category_Counts']
    
    # Extract category counts
    def extract_category_counts(category_dict, category):
        return category_dict.get(category, 0)
    
    agent_stats['Short_Calls'] = agent_stats['Category_Counts'].apply(
        lambda x: extract_category_counts(x, 'Short')
    )
    agent_stats['Medium_Calls'] = agent_stats['Category_Counts'].apply(
        lambda x: extract_category_counts(x, 'Medium')
    )
    agent_stats['Large_Calls'] = agent_stats['Category_Counts'].apply(
        lambda x: extract_category_counts(x, 'Large')
    )
    
    # Calculate percentages
    agent_stats['Short_%'] = (agent_stats['Short_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    agent_stats['Medium_%'] = (agent_stats['Medium_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    agent_stats['Large_%'] = (agent_stats['Large_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    
    # Format duration columns
    agent_stats['Avg_Duration_Formatted'] = agent_stats['Avg_Duration'].apply(fmt_hms)
    agent_stats['Total_Duration_Formatted'] = agent_stats['Total_Duration'].apply(fmt_hms)
    
    # Drop the Category_Counts column
    agent_stats = agent_stats.drop('Category_Counts', axis=1)
    
    # Sort by total calls (descending)
    agent_stats = agent_stats.sort_values('Total_Calls', ascending=False)
    
    # Add ranking
    agent_stats['Rank'] = range(1, len(agent_stats) + 1)
    
    return agent_stats

# ============================================================
# 🆕 GROQ WHISPER + SENTIMENT FUNCTIONS - FIXED (Lazy Loading)
# ============================================================

@st.cache_resource(show_spinner="Loading RoBERTa sentiment model (first run only)...")
def load_sentiment_pipeline():
    """
    Load a RoBERTa-based sentiment analysis pipeline from Hugging Face.
    Model: cardiffnlp/twitter-roberta-base-sentiment (supports negative/neutral/positive)
    """
    try:
        # Lazy import - only loads transformers when this function is called
        from transformers import pipeline
        return pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=-1,  # CPU
            top_k=None
        )
    except Exception as e:
        st.warning(f"⚠️ Sentiment model not available: {e}")
        return None

def groq_transcribe(audio_file_path, api_key):
    """
    Send audio file to Groq Whisper API and return the transcribed text.
    """
    if not api_key:
        return ""
    try:
        client = groq.Groq(api_key=api_key)
        with open(audio_file_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), f.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
                language="en"
            )
        return transcription
    except Exception as e:
        return ""

def analyze_sentiment(text, pipeline):
    """
    Use the loaded RoBERTa pipeline to get sentiment label.
    Returns one of: 'Positive', 'Negative', 'Neutral'.
    """
    if not text or not text.strip():
        return "Neutral"
    if pipeline is None:
        return "Neutral"
    try:
        results = pipeline(text)
        if results and isinstance(results, list) and len(results) > 0:
            best = max(results[0], key=lambda x: x['score'])
            label = best['label']
            if label == 'LABEL_2':
                return "Positive"
            elif label == 'LABEL_0':
                return "Negative"
            else:
                return "Neutral"
    except Exception as e:
        return "Neutral"
    return "Neutral"

# ============================================================
# 🆕 CUSTOM FILTER FUNCTION - Parse user filter expression
# ============================================================
def apply_custom_filter(df, filter_expr):
    """
    Apply custom filter expression on duration column.
    Supports: <, >, <=, >=, ==, !=, and, or
    Example: duration < 120  (less than 2 min)
             duration > 480  (greater than 8 min)
             duration >= 300 and duration <= 600
    """
    if df is None or len(df) == 0:
        return df
    
    if not filter_expr or not filter_expr.strip():
        return df
    
    # Clean the expression
    expr = filter_expr.strip()
    
    # Replace 'duration' with actual column name
    expr = expr.replace('duration', 'df["_duration_sec"]')
    
    # Safety: only allow safe operations
    safe_chars = set('0123456789.() +-*/<>=andornotdf["_duration_sec"]')
    if not all(c in safe_chars or c.isspace() for c in expr):
        st.error("⚠️ Invalid characters in filter expression. Use only numbers, operators, and 'duration'.")
        return df
    
    try:
        # Evaluate the expression
        filtered_df = df[eval(expr)]
        return filtered_df
    except Exception as e:
        st.error(f"⚠️ Error in filter expression: {e}")
        return df

# ============================================================
# STEP 1 — CLIENT + DATE RANGE + FETCH
# ============================================================
st.markdown('<div class="step-card">', unsafe_allow_html=True)
st.markdown('<div class="step-title"><span class="step-badge">1</span>Choose Client & Date Range</div>', unsafe_allow_html=True)
st.markdown('<p class="step-subtitle">Only calls belonging to the selected client will be fetched.</p>', unsafe_allow_html=True)

c1, c2 = st.columns([1.2, 1.8])
with c1:
    client_name = st.selectbox("Client", options=list(CLIENTS.keys()))
    company_id = CLIENTS[client_name]
with c2:
    dc1, dc2 = st.columns(2)
    with dc1:
        from_date = st.date_input("Start Date", value=date.today() - timedelta(days=1), max_value=date.today())
    with dc2:
        to_date = st.date_input("End Date", value=date.today(), max_value=date.today())
    if from_date > to_date:
        st.error("End Date must be on or after Start Date.")

# ---- Clear old data if client changed ----
if st.session_state.get("cdr_client") is not None and st.session_state["cdr_client"] != client_name:
    st.session_state["cdr_df"] = None
    st.session_state["cdr_client"] = None
    st.session_state["final_df"] = None
    st.session_state["agent_analytics_df"] = None

fetch_clicked = st.button("📥  Fetch Calls", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FETCH CDR REPORT
# ============================================================
if fetch_clicked:
    if from_date > to_date:
        st.error("Please correct the date range before fetching.")
    else:
        try:
            payload = {
                "from_date": from_date.strftime("%Y-%m-%d"),
                "to_date": to_date.strftime("%Y-%m-%d"),
                "company_id": str(company_id),
            }
            with st.spinner(f"Fetching calls for {client_name}..."):
                resp = fetch_cdr(payload)
            if resp.status_code == 200:
                data = resp.json()
                records = data.get("data", data) if isinstance(data, dict) else data
                if isinstance(records, dict):
                    for v in records.values():
                        if isinstance(v, list):
                            records = v
                            break
                cdr_df = pd.DataFrame(records)
                
                # Auto-filter VDCL (abandoned) calls
                cdr_df, removed_count = filter_out_vdcl_calls(cdr_df)
                
                # Add _duration_sec for internal use
                dur_source_col, duration_seconds = resolve_duration_column(cdr_df)
                cdr_df["_duration_sec"] = duration_seconds

                st.session_state["cdr_df"] = cdr_df
                st.session_state["cdr_client"] = client_name
                st.session_state["final_df"] = None
                st.session_state["agent_analytics_df"] = None
                
                if len(cdr_df) == 0:
                    st.warning(f"No valid calls found for **{client_name}** in this date range (VDCL calls auto-removed).")
                else:
                    vdcl_msg = f" (removed {removed_count} VDCL abandoned calls)" if removed_count > 0 else " (no VDCL calls found)"
                    st.markdown(
                        f'<span class="status-banner-ok">✅ Fetched {len(cdr_df)} valid calls for {client_name}{vdcl_msg}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.error(f"Fetch failed: HTTP {resp.status_code} — {resp.text[:300]}")
        except Exception as e:
            st.error(f"Fetch error: {e}")

# ============================================================
# STEP 2 — FILTER & DISPLAY (single table)
# ============================================================
have_data = (
    st.session_state["cdr_df"] is not None
    and len(st.session_state["cdr_df"]) > 0
    and st.session_state.get("cdr_client") == client_name
)

if have_data:
    cdr_df = st.session_state["cdr_df"].copy()
    col_date = find_column(cdr_df, COLUMN_CANDIDATES["date"])
    col_time = find_column(cdr_df, COLUMN_CANDIDATES["time"])
    col_agent = find_column(cdr_df, COLUMN_CANDIDATES["agent_name"])
    col_phone = find_column(cdr_df, COLUMN_CANDIDATES["call_from"])
    col_recording = find_column(cdr_df, COLUMN_CANDIDATES["recording"])
    dur_source_col, duration_seconds = resolve_duration_column(cdr_df)

    # Double-check VDCL
    cdr_df, _ = filter_out_vdcl_calls(cdr_df)

    if dur_source_col is None:
        st.error("Could not find a usable call-duration column.")
        st.stop()

    cdr_df["_duration_sec"] = duration_seconds

    # ---- Step 2 card ----
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">2</span>Pick Call Type & Count</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-subtitle">Choose which calls you want in the report.</p>', unsafe_allow_html=True)

    # Summary pills
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f'<div class="metric-pill"><div class="value">{len(cdr_df)}</div><div class="label">Total valid calls</div></div>', unsafe_allow_html=True)
    with p2:
        avg_dur = cdr_df["_duration_sec"].mean() if len(cdr_df) else 0
        st.markdown(f'<div class="metric-pill"><div class="value">{fmt_hms(avg_dur)}</div><div class="label">Average call duration</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown(f'<div class="metric-pill"><div class="value">{client_name}</div><div class="label">Client</div></div>', unsafe_allow_html=True)

    # Filter controls - UPDATED with Custom Filter option
    bcol, ccol = st.columns([2, 1.2])
    with bcol:
        bucket = st.radio(
            "Call type",
            [
                "All calls",
                "Short (< 2 min)",
                "Medium (2 – 5 min)",
                "Large (> 5 min)",
                "Custom Filter",  # NEW option
            ],
            horizontal=True,
        )
        
        # Custom filter input - shown only when Custom Filter is selected
        custom_filter_expr = ""
        if bucket == "Custom Filter":
            st.markdown("""
            **📝 Enter filter expression:**
            - Use `duration` as the variable name
            - Examples:
                - `duration < 120` (less than 2 min)
                - `duration > 480` (greater than 8 min)
                - `duration >= 300 and duration <= 600` (between 5-10 min)
                - `duration == 0` (zero duration calls)
            """)
            custom_filter_expr = st.text_input(
                "Filter expression",
                value="duration < 120",
                help="Use 'duration' as variable name. Example: duration < 120 (less than 2 min)",
                key="custom_filter_input"
            )
            # Show quick preset buttons
            col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
            with col_preset1:
                if st.button("⬇️ < 2 min", use_container_width=True):
                    st.session_state.custom_filter_input = "duration < 120"
                    st.rerun()
            with col_preset2:
                if st.button("⬆️ > 8 min", use_container_width=True):
                    st.session_state.custom_filter_input = "duration > 480"
                    st.rerun()
            with col_preset3:
                if st.button("📊 2-5 min", use_container_width=True):
                    st.session_state.custom_filter_input = "duration >= 120 and duration <= 300"
                    st.rerun()
            with col_preset4:
                if st.button("📊 5-10 min", use_container_width=True):
                    st.session_state.custom_filter_input = "duration >= 300 and duration <= 600"
                    st.rerun()
    
    with ccol:
        count_mode = st.radio("How many calls?", ["All matching", "Manual number"], horizontal=True)

    # Sort order
    sort_order = st.radio(
        "Sort by Duration",
        ["Descending (longest first)", "Ascending (shortest first)"],
        horizontal=True,
        index=0,
        help="Choose how the table is sorted."
    )
    ascending_sort = sort_order.startswith("Ascending")

    # --- Determine selected data based on filters ---
    if bucket == "Short (< 2 min)":
        matched = cdr_df[cdr_df["_duration_sec"] < 120]
    elif bucket == "Medium (2 – 5 min)":
        matched = cdr_df[(cdr_df["_duration_sec"] >= 120) & (cdr_df["_duration_sec"] <= 300)]
    elif bucket == "Large (> 5 min)":
        matched = cdr_df[cdr_df["_duration_sec"] > 300]
    elif bucket == "Custom Filter":
        matched = apply_custom_filter(cdr_df, custom_filter_expr)
        if len(matched) == 0:
            st.warning("No calls match your filter expression. Please check the syntax.")
    else:  # All calls
        matched = cdr_df

    available = len(matched)
    if count_mode == "Manual number":
        manual_n = st.number_input("Number of calls", min_value=1, value=min(50, available) if available else 1, step=1)
        selected_df = matched.head(int(manual_n))
        if available < manual_n:
            st.warning(f"Only {available} call(s) match – showing all.")
        else:
            st.info(f"Showing {int(manual_n)} of {available} matching calls.")
    else:
        selected_df = matched
        if bucket == "Custom Filter":
            st.info(f"Showing all {available} matching calls for custom filter: `{custom_filter_expr}`")
        else:
            st.info(f"Showing all {available} matching calls for **{bucket}**.")

    # --- Build the SINGLE filtered table (duration first, S.No, sorted) ---
    display_cols = [
        "campaign_id", "agent", "full_name", "leadid", "phone_number",
        "call_date", "start_time", "end_time", "call_duration1", "Recording", "_duration_sec"
    ]
    available_display_cols = [c for c in display_cols if c in selected_df.columns]
    table_df = selected_df[available_display_cols].copy()
    table_df.sort_values("_duration_sec", ascending=ascending_sort, inplace=True)
    table_df.insert(1, "S.No", range(1, len(table_df) + 1))
    final_display_cols = ["_duration_sec", "S.No"] + [c for c in available_display_cols if c != "_duration_sec"]
    table_df = table_df[final_display_cols]

    st.markdown(f"### Filtered Data – Sorted by Duration ({'ascending' if ascending_sort else 'descending'})")
    st.dataframe(table_df, use_container_width=True, height=350)

    st.markdown('</div>', unsafe_allow_html=True)  # end step-card

    # ============================================================
    # STEP 3 — RUN VAD + SENTIMENT ANALYSIS
    # ============================================================
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">3</span>Run Talk-Time & Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-subtitle">Downloads recordings, measures speech/silence, transcribes with Groq Whisper, and performs RoBERTa sentiment analysis.</p>', unsafe_allow_html=True)

    with st.expander("⚙️ Fine-tune detection accuracy (optional)"):
        st.caption(
            "If Talk Time is coming out too low / Silence too high, move the slider "
            "towards 'Detect more speech'. Default is fine for most calls."
        )
        sensitivity = st.slider(
            "Detection sensitivity",
            min_value=1, max_value=9, value=5,
            help="Lower = detect more speech (fixes low Talk Time). Higher = stricter, only counts confident speech.",
        )
        vad_threshold = round(0.15 + (sensitivity - 1) * (0.45 - 0.15) / 8, 3)
        dead_air_secs = st.number_input(
            "Count a pause as 'Dead Air' only if longer than (sec)",
            min_value=1, value=5, step=1,
        )

    run_vad_clicked = st.button("▶️  Run Analysis & Build Report", type="primary")

    if run_vad_clicked:
        if not col_recording:
            st.error("No recording-URL column found in CDR data — cannot fetch recordings.")
        elif len(selected_df) == 0:
            st.warning("No calls selected — nothing to process.")
        else:
            # ---------- Load sentiment pipeline ----------
            sentiment_pipeline = load_sentiment_pipeline()

            @st.cache_resource(show_spinner="Loading voice-detection model (first run only)...")
            def load_vad_model():
                hub_dir = os.path.expanduser("~/.cache/torch/hub")
                try:
                    os.makedirs(hub_dir, exist_ok=True)
                except Exception:
                    hub_dir = os.path.join(tempfile.gettempdir(), "torch_hub")
                    os.makedirs(hub_dir, exist_ok=True)
                
                torch.hub.set_dir(hub_dir)
                try:
                    model, utils = torch.hub.load(
                        "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
                    )
                except Exception:
                    model, utils = torch.hub.load(
                        "snakers4/silero-vad", "silero_vad", force_reload=False
                    )
                return model, utils

            model, utils = load_vad_model()
            get_speech_timestamps = utils[0]

            VAD_CFG = {
                "threshold": vad_threshold,
                "min_speech_duration_ms": 100,
                "min_silence_duration_ms": 200,
                "speech_pad_ms": 300,
                "window_size_samples": 512,
                "dead_air_threshold_sec": dead_air_secs,
            }

            def robust_normalize(audio):
                rms = np.sqrt(np.mean(np.square(audio)))
                if rms > 1e-4:
                    target_rms = 0.1
                    gain = target_rms / rms
                    gain = min(gain, 20.0)
                    audio = audio * gain
                return np.clip(audio, -1.0, 1.0)

            def load_channel_16k(data, sr, channel_idx=None):
                if data.ndim > 1:
                    chan = data[:, channel_idx] if channel_idx is not None else np.mean(data, axis=1)
                else:
                    chan = data
                chan = robust_normalize(chan.astype(np.float32))
                if sr != 16000:
                    chan = librosa.resample(chan, orig_sr=sr, target_sr=16000)
                return torch.from_numpy(chan).float()

            def run_vad(audio_tensor):
                return get_speech_timestamps(
                    audio_tensor, model, sampling_rate=16000,
                    threshold=VAD_CFG["threshold"],
                    min_speech_duration_ms=VAD_CFG["min_speech_duration_ms"],
                    min_silence_duration_ms=VAD_CFG["min_silence_duration_ms"],
                    speech_pad_ms=VAD_CFG["speech_pad_ms"],
                    window_size_samples=VAD_CFG["window_size_samples"],
                )

            def merge_intervals(intervals):
                if not intervals:
                    return []
                intervals = sorted(intervals, key=lambda x: x[0])
                merged = [list(intervals[0])]
                for s, e in intervals[1:]:
                    if s <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], e)
                    else:
                        merged.append([s, e])
                return merged

            def compute_metrics(intervals, total_duration):
                if not intervals:
                    return {
                        "talk_time": 0.0,
                        "silence_time": round(total_duration, 2),
                        "dead_air": round(total_duration, 2) if total_duration > VAD_CFG["dead_air_threshold_sec"] else 0.0,
                        "longest_silence": round(total_duration, 2),
                    }
                speech_time, longest_silence, dead_air, prev_end = 0.0, 0.0, 0.0, 0.0
                for s, e in intervals:
                    speech_time += (e - s)
                    silence = max(0.0, s - prev_end)
                    longest_silence = max(longest_silence, silence)
                    if silence > VAD_CFG["dead_air_threshold_sec"]:
                        dead_air += silence
                    prev_end = e
                ending_silence = max(0.0, total_duration - prev_end)
                longest_silence = max(longest_silence, ending_silence)
                if ending_silence > VAD_CFG["dead_air_threshold_sec"]:
                    dead_air += ending_silence
                silence_time = max(0.0, total_duration - speech_time)
                return {
                    "talk_time": round(speech_time, 2),
                    "silence_time": round(silence_time, 2),
                    "dead_air": round(dead_air, 2),
                    "longest_silence": round(longest_silence, 2),
                }

            results = []
            progress = st.progress(0)
            status = st.empty()
            total_rows = len(selected_df)

            with tempfile.TemporaryDirectory() as tmpdir:
                for i, (_, row) in enumerate(selected_df.iterrows()):
                    status.text(f"Processing call {i+1}/{total_rows}...")
                    rec_url = row.get(col_recording)
                    metrics = {
                        "talk_time": None, "silence_time": None,
                        "dead_air": None, "longest_silence": None, "duration": None,
                    }
                    sentiment = "N/A"
                    transcript = None
                    debug_status = "OK"
                    actual_mp3 = None
                    mp3_path = os.path.join(tmpdir, f"{i}.mp3")
                    wav_path = os.path.join(tmpdir, f"{i}.wav")

                    try:
                        if not rec_url:
                            debug_status = "No recording URL in this row"
                        else:
                            actual_mp3 = resolve_audio_url(rec_url)
                            if not actual_mp3:
                                debug_status = "Could not resolve a direct audio URL from the recording link"
                            else:
                                r = requests.get(actual_mp3, timeout=60, stream=True)
                                if r.status_code != 200:
                                    debug_status = f"Download failed: HTTP {r.status_code}"
                                else:
                                    with open(mp3_path, "wb") as f:
                                        for chunk in r.iter_content(8192):
                                            if chunk:
                                                f.write(chunk)
                                    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
                                        debug_status = "Downloaded file is empty"
                                    else:
                                        # ---- Convert to WAV for VAD ----
                                        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
                                        ff = subprocess.run(
                                            [ffmpeg_cmd, "-y", "-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "16000", wav_path],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        )
                                        if ff.returncode != 0 or not os.path.exists(wav_path):
                                            err_tail = ff.stderr.decode(errors="ignore")[-300:]
                                            debug_status = f"FFmpeg conversion failed: {err_tail.strip()}"
                                        else:
                                            # ---- VAD Analysis ----
                                            data, sr = sf.read(wav_path)
                                            total_duration = len(data) / sr
                                            is_stereo = data.ndim > 1 and data.shape[1] > 1
                                            if is_stereo:
                                                all_intervals = []
                                                for ch in range(data.shape[1]):
                                                    tensor = load_channel_16k(data, sr, channel_idx=ch)
                                                    ts = run_vad(tensor)
                                                    all_intervals.extend([(s["start"] / 16000, s["end"] / 16000) for s in ts])
                                                merged = merge_intervals(all_intervals)
                                            else:
                                                tensor = load_channel_16k(data, sr)
                                                ts = run_vad(tensor)
                                                merged = merge_intervals([(s["start"] / 16000, s["end"] / 16000) for s in ts])
                                            metrics = compute_metrics(merged, total_duration)
                                            metrics["duration"] = round(total_duration, 2)
                                            
                                            # ---- NEW: Groq Whisper transcription ----
                                            if GROQ_API_KEY:
                                                try:
                                                    transcript = groq_transcribe(mp3_path, GROQ_API_KEY)
                                                    # Sentiment analysis on transcript
                                                    sentiment = analyze_sentiment(transcript, sentiment_pipeline)
                                                except Exception as e:
                                                    debug_status = f"Groq/Sentiment error: {str(e)[:100]}"
                                                    transcript = None
                                                    sentiment = "Error"
                                                else:
                                                    debug_status = "OK"
                                            else:
                                                sentiment = "No API Key"
                                                debug_status = "No API Key"
                    except requests.exceptions.RequestException as e:
                        debug_status = f"Network/download error: {str(e)[:100]}"
                    except Exception as e:
                        debug_status = f"Processing error: {str(e)[:100]}"
                    finally:
                        if os.path.exists(mp3_path):
                            try: os.remove(mp3_path)
                            except Exception: pass
                        if os.path.exists(wav_path):
                            try: os.remove(wav_path)
                            except Exception: pass

                    if debug_status != "OK" and debug_status != "No API Key" and "Groq" not in debug_status:
                        st.warning(f"Row {i+1} ({row.get(col_agent) if col_agent else ''}): {debug_status}")

                    crm_duration = row.get("_duration_sec")

                    results.append({
                        "Date": row.get(col_date) if col_date else None,
                        "Time": row.get(col_time) if col_time else None,
                        "Agent Name": row.get(col_agent) if col_agent else None,
                        "Call From": row.get(col_phone) if col_phone else None,
                        "Actual MP3": actual_mp3,
                        "Audio Duration(sec)": metrics.get("duration"),
                        "Audio Call Duration": crm_duration,
                        "AI Tools Talk time": metrics.get("talk_time"),
                        "Silence Time": metrics.get("silence_time"),
                        "Dead Air(included in Silence time)": metrics.get("dead_air"),
                        "Longest Silence": metrics.get("longest_silence"),
                        "Sentiment": sentiment,
                        "_debug_status": debug_status,
                    })
                    progress.progress((i + 1) / total_rows)

            status.text("Done ✅")
            final_df = pd.DataFrame(results)

            # Format dates & times
            if col_date:
                final_df["Date"] = pd.to_datetime(final_df["Date"], errors="coerce").dt.strftime("%d/%m/%Y")
            if col_time:
                final_df["Time"] = pd.to_datetime(final_df["Time"], errors="coerce").dt.strftime("%H:%M:%S")

            # Sort by Audio Call Duration descending (default for report)
            final_df.sort_values("Audio Call Duration", ascending=False, inplace=True)

            # Reorder columns for final display / export
            REORDERED_COLUMNS = [
                "Audio Call Duration",
                "Date",
                "Time",
                "Agent Name",
                "Call From",
                "AI Tools Talk time",
                "Silence Time",
                "Dead Air(included in Silence time)",
                "Longest Silence",
                "Sentiment",
                "Audio Duration(sec)",
                "Actual MP3",
            ]
            final_df = final_df[REORDERED_COLUMNS + ["_debug_status"]]

            # Generate Agent Analytics (with updated categories)
            agent_analytics_df = generate_agent_analytics(selected_df, '_duration_sec')
            
            st.session_state["final_df"] = final_df
            st.session_state["agent_analytics_df"] = agent_analytics_df

            failed_count = (final_df["_debug_status"] != "OK").sum()
            if failed_count > 0:
                st.error(f"⚠️ {failed_count} of {len(final_df)} call(s) failed to process — see warnings above for the reason.")
            else:
                st.success("✅ All calls processed successfully.")

            st.dataframe(final_df.drop(columns=["_debug_status"]), use_container_width=True, height=380)
            
            # Display Agent Analytics if available
            if agent_analytics_df is not None and len(agent_analytics_df) > 0:
                st.markdown("### 📊 Agent-Wise Analytics")
                st.info("Comprehensive breakdown of call performance by agent.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🏆 Top Agents by Large Calls**")
                    top_large = agent_analytics_df.nlargest(3, 'Large_Calls')[['Agent', 'Large_Calls', 'Large_%']]
                    for _, row in top_large.iterrows():
                        st.markdown(f"""
                        <div class="agent-card">
                            <div class="agent-name">{row['Agent']}</div>
                            <div class="agent-stats">
                                {row['Large_Calls']} large calls ({row['Large_%']:.1f}% of total)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**📉 Agents with Most Short Calls**")
                    top_short = agent_analytics_df.nlargest(3, 'Short_Calls')[['Agent', 'Short_Calls', 'Short_%']]
                    for _, row in top_short.iterrows():
                        st.markdown(f"""
                        <div class="agent-card">
                            <div class="agent-name">{row['Agent']}</div>
                            <div class="agent-stats">
                                {row['Short_Calls']} short calls ({row['Short_%']:.1f}% of total)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show summary metrics
                st.markdown("**📈 Performance Summary**")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Agents", len(agent_analytics_df))
                with m2:
                    avg_calls = agent_analytics_df['Total_Calls'].mean()
                    st.metric("Avg Calls per Agent", f"{avg_calls:.1f}")
                with m3:
                    best_agent = agent_analytics_df.iloc[0]['Agent'] if len(agent_analytics_df) > 0 else "N/A"
                    st.metric("Most Active Agent", best_agent)
                with m4:
                    best_large = agent_analytics_df.nlargest(1, 'Large_Calls')['Agent'].iloc[0] if len(agent_analytics_df) > 0 else "N/A"
                    st.metric("Most Large Calls", best_large)
                
                # Show detailed table
                st.dataframe(
                    agent_analytics_df[[
                        'Rank', 'Agent', 'Total_Calls', 'Short_Calls', 'Short_%',
                        'Medium_Calls', 'Medium_%', 'Large_Calls', 'Large_%',
                        'Avg_Duration_Formatted', 'Total_Duration_Formatted'
                    ]],
                    use_container_width=True,
                    height=300
                )

    st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # STEP 4 — DOWNLOAD FINAL REPORT
    # ============================================================
    if st.session_state.get("final_df") is not None:
        EXPORT_COLUMNS = [
            "Audio Call Duration",
            "Date",
            "Time",
            "Agent Name",
            "Call From",
            "AI Tools Talk time",
            "Silence Time",
            "Dead Air(included in Silence time)",
            "Longest Silence",
            "Sentiment",
            "Audio Duration(sec)",
            "Actual MP3",
        ]
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title"><span class="step-badge">4</span>Download Report</div>', unsafe_allow_html=True)
        st.markdown('<p class="step-subtitle">Excel file with two sheets: Detailed Call Report (now with Sentiment) and Agent-Wise Analytics (Short<2min, Medium 2-5min, Large>5min).</p>', unsafe_allow_html=True)

        export_df = st.session_state["final_df"][EXPORT_COLUMNS]
        agent_export_df = st.session_state.get("agent_analytics_df")
        
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Call Report")
            if agent_export_df is not None and len(agent_export_df) > 0:
                agent_sheet = agent_export_df[[
                    'Rank', 'Agent', 'Total_Calls', 'Short_Calls', 'Short_%',
                    'Medium_Calls', 'Medium_%', 'Large_Calls', 'Large_%',
                    'Avg_Duration_Formatted', 'Total_Duration_Formatted'
                ]].copy()
                agent_sheet.columns = [
                    'Rank', 'Agent', 'Total Calls', 'Short Calls', 'Short %',
                    'Medium Calls', 'Medium %', 'Large Calls', 'Large %',
                    'Avg Duration', 'Total Duration'
                ]
                agent_sheet.to_excel(writer, index=False, sheet_name="Agent Analytics")
        buf.seek(0)
        st.download_button(
            "⬇️  Download Excel Report",
            data=buf,
            file_name=f"CallAI_Talk_Time_Report_{client_name.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👆 Pick a client and date range above, then click **Fetch Calls** to get started.")