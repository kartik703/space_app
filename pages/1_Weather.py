# pages/1_Weather.py
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Space Weather", page_icon="🌞", layout="wide")
st.title("🌞 Space Weather")

def read_csv_safe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

# Current Kp
kp = read_csv_safe("data/kp_latest.csv")
kp_val = float(kp["Kp"].tail(1).values[0]) if not kp.empty and "Kp" in kp.columns else None
c1, c2 = st.columns([1,3])
with c1:
    st.metric("Current Kp", f"{kp_val:.1f}" if kp_val is not None else "—")
    st.caption("Source: NOAA SWPC")

# Forecast
with c2:
    horizon = st.slider("Forecast horizon (hours)", 12, 48, 12)
    kpf = read_csv_safe("data/kp_forecast.csv")
    if not kpf.empty:
        time_col = next((c for c in ["time","timestamp","datetime","ts"] if c in kpf.columns), None)
        kp_col   = next((c for c in ["kp","Kp"] if c in kpf.columns), None)
        if time_col and kp_col:
            kpf[time_col] = pd.to_datetime(kpf[time_col], errors="coerce", utc=True)
            kpf = kpf.dropna(subset=[time_col]).sort_values(time_col).tail(horizon)
            chart = alt.Chart(kpf).mark_line().encode(
                x=alt.X(f"{time_col}:T", title="Time"),
                y=alt.Y(f"{kp_col}:Q", title="Kp"),
                tooltip=[time_col, kp_col]
            ).properties(height=220)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No forecast columns (time/Kp) found yet.")
    else:
        st.info("No forecast yet. Refresh data from sidebar on Home.")
