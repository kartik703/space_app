import streamlit as st, pandas as pd, pathlib, subprocess, sys, datetime as dt, altair as alt
from utils import read_csv_safe
from utils import set_background, read_csv_safe, badge, human_ts

# Always set video background
set_background("docs/bg.mp4")

st.title("🛰 Conjunction Candidates (sgp4)")

p_tle = pathlib.Path("data/tle_small.csv")
p_conj = pathlib.Path("data/conjunctions.csv")

colA, colB, colC = st.columns(3)
with colA:
    only_leo = st.checkbox("LEO filter (≤1200/2000 km)", True)
with colB:
    threshold_km = st.slider("Threshold (km)", 5, 50, 20, 1)
with colC:
    horizon_h = st.slider("Horizon (hours)", 12, 72, 48, 6)

colD, colE = st.columns(2)
with colD:
    step_s = st.selectbox("Step (sec)", [30, 60, 120, 300], index=1)
with colE:
    max_sats = st.slider("Max satellites", 50, 300, 200, 10)

colX, colY = st.columns(2)
if colX.button("🔁 Retry TLE fetch"):
    subprocess.run([sys.executable, "scripts/fetch_tle.py"], check=False)
if colY.button("🧮 Recompute with settings"):
    cmd = [sys.executable, "scripts/conjunctions.py",
           "--threshold_km", str(threshold_km),
           "--horizon_h", str(horizon_h),
           "--step_s", str(step_s),
           "--max_sats", str(max_sats),
           "--top_n", "300"]
    if only_leo: cmd.append("--only_leo")
    with st.spinner("Propagating…"):
        subprocess.run(cmd, check=False)

# Display
df = read_csv_safe(str(p_conj), parse_dates=["time"])
st.caption(f"Updated: {dt.datetime.fromtimestamp(p_conj.stat().st_mtime):%Y-%m-%d %H:%M UTC}" if p_conj.exists() else "—")
if df.empty:
    st.info("No conjunction data yet. Try Retry TLE, then Recompute.")
else:
    st.dataframe(df.sort_values(["time","sep_km"]).head(300), use_container_width=True)
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X("sep_km:Q", bin=alt.Bin(step=5), title="Separation (km)"),
        y="count()"
    ).properties(height=200)
    st.altair_chart(hist, use_container_width=True)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "conjunctions.csv", "text/csv")

