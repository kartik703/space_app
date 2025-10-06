# utils.py
import base64
from pathlib import Path
import pandas as pd
import streamlit as st
import datetime as dt

# -------------------------------
# VIDEO BACKGROUND
# -------------------------------
def set_background(video_path: str = "docs/bg.mp4"):
    """Set a looping video as background for the entire app."""
    try:
        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background: transparent !important;
            }}
            [data-testid="stHeader"] {{
                background: rgba(0,0,0,0.3) !important;
            }}
            [data-testid="stSidebar"] {{
                background: rgba(0,0,0,0.6) !important;
            }}

            video#bgvid {{
                position: fixed;
                top: 0; left: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                z-index: -2;
            }}

            .overlay {{
                position: fixed;
                top: 0; left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                z-index: -1;
            }}
            </style>

            <video autoplay muted loop id="bgvid">
              <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            <div class="overlay"></div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"⚠️ Background video not found: {video_path}")

# -------------------------------
# DATA HELPERS
# -------------------------------
def read_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    """Safely read CSV file, return empty DataFrame if missing or broken."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=parse_dates)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

def badge(ok: bool, ok_text="Ready", fail_text="Missing"):
    """Return ✅ or ❌ badge string."""
    return f"✅ {ok_text}" if ok else f"❌ {fail_text}"

def human_ts(path: str) -> str:
    """Get human-readable last modified timestamp for a file."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return "—"
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
