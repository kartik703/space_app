from pathlib import Path
import subprocess, sys, os, datetime as dt
import streamlit as st
import altair as alt
import pandas as pd
from utils import read_csv_safe, badge, human_ts

st.set_page_config(page_title="Space Intelligence Super App", page_icon="🛰️", layout="wide")

# ---------- helpers ----------
def run_job(cmd: list[str]) -> None:
    env = os.environ.copy()
    if "spacetrack" in st.secrets:
        u = st.secrets["spacetrack"].get("username", "")
        p = st.secrets["spacetrack"].get("password", "")
        if u and p:
            env["SPACETRACK_USERNAME"] = u
            env["SPACETRACK_PASSWORD"] = p
    subprocess.run([sys.executable, *cmd], check=False, env=env)

def ensure_data():
    Path("data").mkdir(parents=True, exist_ok=True)
    base = [
        ("data/kp_latest.csv",        ["scripts/fetch_space_weather.py"]),
        ("data/kp_forecast.csv",      ["scripts/forecast_kp.py"]),
        ("data/asteroids.csv",        ["scripts/fetch_asteroids.py"]),
        ("data/commodities.csv",      ["scripts/fetch_commodities.py"]),
        ("data/asteroids_scored.csv", ["scripts/compute_asteroid_profit.py"]),
        ("data/launch_weather.csv",   ["scripts/fetch_launch_weather.py"]),
        ("data/launches.csv",         ["scripts/fetch_launches.py"]),
        ("data/launches_history.csv", ["scripts/fetch_launches_history.py"]),
        ("data/tle_small.csv",        ["scripts/fetch_tle.py"]),
    ]
    for target, cmd in base:
        p = Path(target)
        if not p.exists() or p.stat().st_size == 0:
            with st.spinner(f"Fetching {p.name}…"):
                run_job(cmd)
    # conjunctions depend on TLEs
    if Path("data/tle_small.csv").exists() and Path("data/tle_small.csv").stat().st_size > 0:
        p = Path("data/conjunctions.csv")
        if not p.exists() or p.stat().st_size == 0:
            with st.spinner("Propagating conjunctions (sgp4)…"):
                run_job([
                    "scripts/conjunctions.py",
                    "--only_leo", "--threshold_km", "20",
                    "--horizon_h", "24", "--step_s", "60",
                    "--max_sats", "120", "--top_n", "200",
                ])

# ---------- sidebar ----------
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh all data now"):
        steps = [
            ("Kp latest", ["scripts/fetch_space_weather.py"]),
            ("Kp forecast", ["scripts/forecast_kp.py"]),
            ("Asteroids", ["scripts/fetch_asteroids.py"]),
            ("Commodities", ["scripts/fetch_commodities.py"]),
            ("Asteroid scores", ["scripts/compute_asteroid_profit.py"]),
            ("Launch weather", ["scripts/fetch_launch_weather.py"]),
            ("Upcoming launches", ["scripts/fetch_launches.py"]),
            ("Launch history", ["scripts/fetch_launches_history.py"]),
            ("TLEs", ["scripts/fetch_tle.py"]),
        ]
        for _, cmd in steps:
            run_job(cmd)
        if Path("data/tle_small.csv").exists() and Path("data/tle_small.csv").stat().st_size > 0:
            run_job([
                "scripts/conjunctions.py","--only_leo","--threshold_km","20",
                "--horizon_h","24","--step_s","60","--max_sats","120","--top_n","200"
            ])
        st.success("Refreshed. Reload the page.")
    st.markdown("---")
    st.page_link("pages/1_Weather.py", label="🌞 Space Weather")
    st.page_link("pages/2_Asteroid_Mining.py", label="🪨 Asteroid Mining")
    st.page_link("pages/3_Collisions.py", label="🛰️ Conjunctions")
    st.page_link("pages/4_Launch_Optimizer.py", label="🚀 Launch Window")
    st.page_link("pages/5_Space_Tracker.py", label="📡 Space Tracker")

# first-load populate
ensure_data()

# ---------- hero ----------
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;">
  <img src="https://em-content.zobj.net/source/microsoft-teams/363/satellite_1f6f0-fe0f.png" width="42"/>
  <div>
    <h1 style="margin:0;">Space Intelligence Super App</h1>
    <p style="margin:0;color:#9CA3AF;">AI-powered mission intelligence: weather • mining • collisions • launch • tracking</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- KPI cards ----------
col1, col2, col3, col4 = st.columns(4)
kp = read_csv_safe("data/kp_latest.csv", parse_dates=["time_tag"])
kpv = float(kp["Kp"].tail(1).values[0]) if not kp.empty else None
with col1:
    st.metric("Current Kp", f"{kpv:.1f}" if kpv is not None else "—", help="NOAA planetary K-index")

ast = read_csv_safe("data/asteroids_scored.csv")
with col2:
    st.metric("Asteroids scored", f"{len(ast):,}" if not ast.empty else "—")

tle = read_csv_safe("data/tle_small.csv")
conj = read_csv_safe("data/conjunctions.csv", parse_dates=["time"])
with col3:
    st.metric("Conjunction rows", f"{len(conj):,}" if not conj.empty else "0")

ll2 = read_csv_safe("data/launches.csv", parse_dates=["window_start"])
with col4:
    st.metric("Upcoming launches", f"{len(ll2):,}" if not ll2.empty else "—")

st.divider()

# ---------- quick previews ----------
c1, c2, c3 = st.columns(3)

# Kp preview
with c1:
    st.subheader("Kp forecast (48h)")
    kpf = read_csv_safe("data/kp_forecast.csv", parse_dates=["time"])
    if not kpf.empty:
        chart = alt.Chart(kpf.tail(48)).mark_line().encode(
            x="time:T", y=alt.Y("kp:Q", title="Kp"),
            tooltip=["time:T","kp:Q"]
        ).properties(height=200)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No forecast available yet.")

# Asteroid profit preview
with c2:
    st.subheader("Top asteroid profitability")
    if not ast.empty:
        top = ast.sort_values("profit_index", ascending=False).head(15)
        chart = alt.Chart(top).mark_bar().encode(
            x=alt.X("profit_index:Q", title="Profit index"),
            y=alt.Y("object:N", sort="-x", title="Asteroid"),
            tooltip=["object","profit_index","dv_kms","est_value_usd"]
        ).properties(height=200)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No asteroid scores yet.")

# Launch preview
with c3:
    st.subheader("Next launches (D-7)")
    if not ll2.empty:
        soon = ll2.sort_values("window_start").head(10)[["name","provider","window_start","pad"]]
        st.dataframe(soon, use_container_width=True, height=220)
    else:
        st.info("No launch data yet.")

st.divider()

# ---------- data snapshot table ----------
st.subheader("Data snapshots")
files = [
    "kp_latest.csv","kp_forecast.csv","asteroids.csv","commodities.csv",
    "asteroids_scored.csv","tle_small.csv","conjunctions.csv",
    "launch_weather.csv","launches.csv","launches_history.csv",
]
rows = []
for f in files:
    p = Path("data")/f
    ok = p.exists() and p.stat().st_size>0
    rows.append({"file": f, "status": badge(ok), "updated": human_ts(str(p))})
st.dataframe(pd.DataFrame(rows), use_container_width=True)
