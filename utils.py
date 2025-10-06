# utils.py
import base64
from pathlib import Path
import pandas as pd
import streamlit as st
import datetime as dt

# -------------------------------
# VIDEO BACKGROUND
# -------------------------------
def set_background(theme: str = "space"):
    """Set animated background for the Space AI Platform"""
    
    if theme == "space":
        st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, 
                #0c0c0c 0%, 
                #1a1a2e 25%, 
                #16213e 50%, 
                #0f3460 75%, 
                #533483 100%
            );
            background-size: 400% 400%;
            animation: spaceGradient 20s ease infinite;
        }
        
        @keyframes spaceGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        [data-testid="stHeader"] {
            background: rgba(12, 12, 12, 0.8) !important;
            backdrop-filter: blur(10px);
        }
        
        [data-testid="stSidebar"] {
            background: rgba(12, 12, 12, 0.9) !important;
            backdrop-filter: blur(15px);
            border-right: 2px solid rgba(83, 52, 131, 0.5);
        }
        
        .main .block-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem;
            box-shadow: 0 8px 32px rgba(83, 52, 131, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .metric-card {
            background: linear-gradient(135deg, 
                rgba(255,255,255,0.9) 0%, 
                rgba(255,255,255,0.7) 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Floating particles effect */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(2px 2px at 20px 30px, #ffffff, transparent),
                radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
                radial-gradient(1px 1px at 90px 40px, rgba(255,255,255,0.6), transparent),
                radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.4), transparent);
            background-repeat: repeat;
            background-size: 200px 150px;
            animation: float 30s linear infinite;
            pointer-events: none;
            z-index: -1;
        }
        
        @keyframes float {
            0% { transform: translateY(0px) translateX(0px); }
            100% { transform: translateY(-100px) translateX(-100px); }
        }
        </style>
        """, unsafe_allow_html=True)

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
