from pathlib import Path
import pandas as pd
import streamlit as st

def read_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=parse_dates)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

def badge(ok: bool, ok_text="Ready", fail_text="Missing"):
    return f"✅ {ok_text}" if ok else "❌ {fail_text}"

def human_ts(path: str) -> str:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return "—"
    import datetime as dt
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
