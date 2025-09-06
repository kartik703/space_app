# pages/5_Space_Tracker.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Space Tracker", page_icon="📡", layout="wide")
st.title("📡 Space Tracker — Upcoming & History")

def read_csv_safe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

def parse_dates_if_present(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

up = read_csv_safe("data/launches.csv")
hist = read_csv_safe("data/launches_history.csv")

# Normalize dates if present
date_candidates = ["window_start","window_end","net","t0","date"]
up = parse_dates_if_present(up, date_candidates)
hist = parse_dates_if_present(hist, date_candidates)

query = st.text_input("Search mission/provider/pad", "")
def filt(df):
    if df.empty or not query:
        return df
    q = query.lower()
    cols = [c for c in ["name","mission","provider","pad","location"] if c in df.columns]
    if not cols:
        return df
    m = False
    for c in cols:
        m = m | df[c].astype(str).str.lower().str.contains(q, na=False)
    return df[m]

st.subheader("Upcoming launches")
if not up.empty:
    date_col = next((c for c in ["window_start","net","t0","date"] if c in up.columns), None)
    cols = [c for c in ["name","provider",date_col,"pad","location"] if c]
    df_show = filt(up)
    if date_col:
        df_show = df_show.sort_values(date_col)
    st.dataframe(df_show[cols].head(50), use_container_width=True, height=300)
else:
    st.info("No upcoming launches found.")

st.subheader("Historical launches")
if not hist.empty:
    date_col_h = next((c for c in ["window_end","window_start","net","t0","date"] if c in hist.columns), None)
    cols_h = [c for c in ["name","provider",date_col_h,"pad","status","success","vehicle"] if c]
    dfh = filt(hist)
    if date_col_h:
        dfh = dfh.sort_values(date_col_h, ascending=False)
    st.dataframe(dfh[cols_h].head(200), use_container_width=True, height=380)
else:
    st.info("No historical data found.")
