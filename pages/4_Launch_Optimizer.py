import streamlit as st, pandas as pd, numpy as np, pathlib, altair as alt

st.set_page_config(page_title="Launch Window", page_icon="🚀", layout="wide")
st.title("🚀 Launch Window — Weather + Target Orbit Feasibility")

site_lat = 28.5  # KSC
target_i = st.slider("Target inclination (deg)", 28.5, 98.0, 51.6, 0.1)
target_h = st.slider("Target circular altitude (km)", 200, 800, 400, 10)

if target_i < abs(site_lat):
    st.error("Direct injection infeasible from this latitude (without dogleg). Increase inclination.")
else:
    st.success("Direct injection feasible.")

mu = 398600.4418; Re = 6378.137
r = Re + target_h
v_orbit = np.sqrt(mu / r)
dv_inject = v_orbit + 1.8
st.caption(f"Approx orbital velocity: **{v_orbit:.2f} km/s**, Δv to inject (rough): **{dv_inject:.2f} km/s**")

p = pathlib.Path("data/launch_weather.csv")
if p.exists():
    df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    df["weather_score"] = (10 - df["wind_speed_10m"]).clip(lower=0) + (10 - df["precip_mm"].clip(0,10))
    feas = (target_i - abs(site_lat)) / (98 - abs(site_lat))
    alt_pen = 1 - (target_h - 200) / (800 - 200)
    feas_scalar = np.clip(0.6*feas + 0.4*alt_pen, 0, 1) * 10

    df["feasibility"] = feas_scalar
    df["score"] = (0.6*df["weather_score"] + 0.4*df["feasibility"]).round(2)
    df["window"] = df["score"].apply(lambda s: "✅ Good" if s >= 14 else ("⚠️ Ok" if s >= 12 else "❌ Poor"))
    st.dataframe(df, use_container_width=True)

    chart = alt.Chart(df).mark_line(point=True).encode(
        x="date:T", y=alt.Y("score:Q", title="Launch score (0–20)"),
        color="window:N", tooltip=["date:T","score:Q","weather_score:Q","feasibility:Q"]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
else:
    st.warning("No launch_weather.csv yet.")
