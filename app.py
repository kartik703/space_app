

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="🌌 Space Intelligence Super App", layout="wide")
st.title("🌌 Space Intelligence Super App")
st.caption("Real data. Nightly auto-refresh via GitHub Actions.")

st.markdown("""
**Modules**
1) 🌞 Space Weather — NOAA Kp + 48h forecast  
2) 🪨 Asteroid Mining — JPL SBDB + Δv cost + commodity pricing  
3) 🛰 Collisions — sgp4 propagation + close-approach screening (24h)  
4) 🚀 Launch Window — Weather + target-orbit feasibility  
5) 📡 Space Tracker — Missions, reliability, delays
""")

data_dir = Path("data")
st.subheader("Data snapshots")
for name in [
    "kp_latest.csv","kp_forecast.csv",
    "asteroids.csv","commodities.csv","asteroids_scored.csv",
    "tle_small.csv","conjunctions.csv",
    "launch_weather.csv","launches.csv","launches_history.csv"
]:
    st.write(f"- {name}: {'✅' if (data_dir/name).exists() else '❌'}")
