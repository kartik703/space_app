import streamlit as st
import pandas as pd
from pathlib import Path
from utils import set_background, read_csv_safe, badge, human_ts

# Always set video background
set_background("docs/bg.mp4")


st.title("📡 Space Tracker — Upcoming & History")

def read_csv_safe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)  # parse_dates ignored by global patch
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

def to_dt_if_present(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

# Load & normalize
up   = to_dt_if_present(read_csv_safe("data/launches.csv"),
                        ["window_start", "window_end", "net", "t0", "date"])
hist = to_dt_if_present(read_csv_safe("data/launches_history.csv"),
                        ["window_end", "window_start", "net", "t0", "date"])

query = st.text_input("Search mission/provider/pad", "")

def filt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not query:
        return df
    q = query.lower()
    cols = [c for c in ["name", "mission", "provider", "pad", "location", "vehicle"] if c in df.columns]
    if not cols:
        return df
    m = False
    for c in cols:
        m = m | df[c].astype(str).str.lower().str.contains(q, na=False)
    return df[m]

# Upcoming
st.subheader("Upcoming launches")
if not up.empty:
    dcol = next((c for c in ["window_start", "net", "t0", "date"] if c in up.columns), None)
    cols_show = [c for c in ["window_start", "name", "provider", "vehicle", "pad", "location", "mission"] if c in up.columns]
    dfu = filt(up)
    if dcol: dfu = dfu.sort_values(dcol)
    st.dataframe(dfu[cols_show].head(60), use_container_width=True, height=360)
else:
    st.info("No upcoming launches found.")

# History / reliability
st.markdown("### Provider reliability (history) ↪︎")
if not hist.empty:
    dcol_h = next((c for c in ["window_end", "window_start", "net", "t0", "date"] if c in hist.columns), None)
    cols_h = [c for c in ["window_end", "window_start", "name", "provider", "status", "success", "vehicle"] if c in hist.columns]
    dfh = filt(hist)
    if dcol_h: dfh = dfh.sort_values(dcol_h, ascending=False)
    st.dataframe(dfh[cols_h].head(200), use_container_width=True, height=300)
else:
    st.info("No historical dataset yet.")


