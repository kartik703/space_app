# pages/5_Space_Tracker.py
import streamlit as st, pandas as pd
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

def parse_dates_if_present(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

up   = parse_dates_if_present(read_csv_safe("data/launches.csv"),
                              ["window_start","window_end","net","t0","date"])
hist = parse_dates_if_present(read_csv_safe("data/launches_history.csv"),
                              ["window_start","window_end","net","t0","date"])

query = st.text_input("Search mission/provider/pad", "")

def filt(df):
    if df.empty or not query: return df
    q = query.lower()
    cols = [c for c in ["name","mission","provider","pad","location","vehicle"] if c in df.columns]
    if not cols: return df
    m = False
    for c in cols:
        m = m | df[c].astype(str).str.lower().str.contains(q, na=False)
    return df[m]

st.subheader("Upcoming launches")
if not up.empty:
    dcol = next((c for c in ["window_start","net","t0","date"] if c in up.columns), None)
    cols = [c for c in ["window_start","name","provider","vehicle","pad","location","mission"] if c in up.columns]
    df_show = filt(up)
    if dcol: df_show = df_show.sort_values(dcol)
    st.dataframe(df_show[cols].head(50), use_container_width=True, height=360)
else:
    st.info("No upcoming launches found.")

st.markdown("### Provider reliability (history) ↪︎")
if not hist.empty:
    dcol = next((c for c in ["window_end","window_start","net","t0","date"] if c in hist.columns), None)
    cols = [c for c in ["window_end","window_start","name","provider","status","success","vehicle"] if c in hist.columns]
    dfh = filt(hist)
    if dcol: dfh = dfh.sort_values(dcol, ascending=False)
    st.dataframe(dfh[cols].head(200), use_container_width=True, height=260)
else:
    st.info("No historical dataset yet.")
