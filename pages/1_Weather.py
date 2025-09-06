import streamlit as st, pandas as pd, altair as alt
from utils import read_csv_safe

st.set_page_config(page_title="Space Weather", page_icon="🌞", layout="wide")
st.title("🌞 Space Weather")

kp = read_csv_safe("data/kp_latest.csv", parse_dates=["time_tag"])
kpf = read_csv_safe("data/kp_forecast.csv", parse_dates=["time"])

col1, col2 = st.columns(2)
with col1:
    if not kp.empty:
        st.metric("Current Kp", f"{kp['Kp'].iloc[-1]:.1f}")
    st.caption("Source: NOAA SWPC")

with col2:
    horizon = st.slider("Forecast horizon (hours)", 12, 48, 48, 6)

if not kpf.empty:
    kpf = kpf.sort_values("time").tail(horizon)
    chart = alt.Chart(kpf).mark_area(opacity=0.6).encode(
        x="time:T", y=alt.Y("kp:Q", title="Kp"),
        tooltip=["time:T","kp:Q"]
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No forecast yet. Refresh data from sidebar on Home.")
