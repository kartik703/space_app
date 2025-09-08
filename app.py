# app.py
import streamlit as st
import altair as alt
from pathlib import Path
from utils import set_background, read_csv_safe, badge, human_ts

# -----------------------------
# CONFIG & BACKGROUND
# -----------------------------
st.set_page_config(
    page_title="🌌 Space Intelligence Super App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Always show video background
set_background("docs/bg.mp4")

# App title
st.title("🚀 Space Intelligence Super App")
st.caption("AI-powered dashboard for space weather, asteroid mining, orbital safety, and launch insights.")

DATA = Path("data")

# -----------------------------
# SIDEBAR MODULES
# -----------------------------
st.sidebar.title("📊 Modules")
choice = st.sidebar.radio(
    "Select Module",
    ["🌞 Space Weather", "🪨 Asteroid Mining", "🛰 Orbital Congestion",
     "🚀 Launch Success", "📡 Launch Tracker", "⚠️ Anomalies"]
)

# -----------------------------
# SPACE WEATHER
# -----------------------------
if choice == "🌞 Space Weather":
    st.subheader("🌞 Space Weather (NOAA + AI Forecast)")
    kp = read_csv_safe(DATA / "kp_latest.csv", parse_dates=["time_tag"])
    fc = read_csv_safe(DATA / "kp_forecast.csv", parse_dates=["time"])

    if not kp.empty:
        st.metric("Current Kp", kp["Kp"].iloc[-1])
        st.caption("Source: NOAA SWPC")

    if not fc.empty and {"time", "forecast"}.issubset(fc.columns):
        horizon = st.slider("Forecast horizon (hours)", 12, 48, 24)
        subset = fc.head(horizon)

        chart = alt.Chart(subset).mark_line(color="cyan").encode(
            x="time:T",
            y=alt.Y("forecast:Q", title="Kp Index"),
            tooltip=["time", "forecast"]
        ).properties(title=f"{horizon}h Forecast")

        st.altair_chart(chart, use_container_width=True)
    elif not fc.empty:
        st.warning("Forecast present but missing time/forecast columns.")
    else:
        st.info("No forecast data available.")

# -----------------------------
# ASTEROID MINING
# -----------------------------
elif choice == "🪨 Asteroid Mining":
    st.subheader("🪨 Asteroid Profitability Leaderboard")
    ast = read_csv_safe(DATA / "asteroids_scored.csv")

    if not ast.empty and "profit_index" in ast.columns:
        top = ast.sort_values("profit_index", ascending=False).head(10)
        st.dataframe(top[["object","profit_index","dv_kms","est_value_usd"]])

        chart = alt.Chart(top).mark_bar(color="orange").encode(
            x="profit_index:Q",
            y=alt.Y("object:N", sort="-x"),
            tooltip=["profit_index","dv_kms","est_value_usd"]
        ).properties(title="Top 10 Asteroids")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No asteroid scores available yet.")

# -----------------------------
# ORBITAL CONGESTION
# -----------------------------
elif choice == "🛰 Orbital Congestion":
    st.subheader("🛰 Orbital Congestion Heatmap")
    cong = read_csv_safe(DATA / "congestion_bins.csv")

    if not cong.empty and {"alt_bin_km","inc_bin_deg","count"}.issubset(cong.columns):
        chart = alt.Chart(cong).mark_rect().encode(
            x=alt.X("inc_bin_deg:O", title="Inclination Bin (°)"),
            y=alt.Y("alt_bin_km:O", title="Altitude Bin (km)"),
            color="count:Q",
            tooltip=["alt_bin_km","inc_bin_deg","count"]
        ).properties(title="Orbital Congestion Map")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Congestion data invalid or missing. Run scripts/congestion_map.py.")

# -----------------------------
# LAUNCH SUCCESS
# -----------------------------
elif choice == "🚀 Launch Success":
    st.subheader("🚀 Hybrid Launch Success Predictor")
    ls = read_csv_safe(DATA / "launch_success_scores.csv")

    if not ls.empty:
        st.dataframe(ls)

        chart = alt.Chart(ls).mark_bar(color="lime").encode(
            x="go_score:Q",
            y=alt.Y("name:N", sort="-x"),
            tooltip=["window_start","factors"]
        ).properties(title="Launch GO Scores")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No launch success scores computed yet.")

# -----------------------------
# LAUNCH TRACKER
# -----------------------------
elif choice == "📡 Launch Tracker":
    st.subheader("📡 Upcoming Launches")
    launches = read_csv_safe(DATA / "launches.csv", parse_dates=["window_start"])

    if not launches.empty:
        st.dataframe(launches[["name","window_start","provider","pad"]].head(15))
    else:
        st.warning("No upcoming launches available.")

# -----------------------------
# ANOMALIES
# -----------------------------
elif choice == "⚠️ Anomalies":
    st.subheader("⚠️ Orbital Anomalies")
    anom = read_csv_safe(DATA / "anomalies.csv")

    if not anom.empty:
        st.dataframe(anom.head(20))
    else:
        st.info("No anomalies detected yet.")

# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.success("✅ Data auto-refreshed via run_all.py nightly")
