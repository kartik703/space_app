# app.py
from pathlib import Path
import subprocess
import sys
import os
import datetime as dt
import streamlit as st

# ──────────────────────────
# Page setup
# ──────────────────────────
st.set_page_config(
    page_title="Space Intelligence Super App",
    page_icon="🛰️",
    layout="wide",
)
# NOTE: Disable file-watcher in cloud via .streamlit/config.toml:
# [server]
# headless = true
# fileWatcherType = "none"
# runOnSave = false

# ──────────────────────────
# Helpers: run jobs (with Space-Track secrets) + ensure data
# ──────────────────────────
def run_job(cmd: list[str]) -> None:
    """Run a Python script (with args). Never hard-fail the UI."""
    try:
        env = os.environ.copy()
        # Wire Space-Track credentials from Streamlit secrets -> environment
        if "spacetrack" in st.secrets:
            st_user = st.secrets["spacetrack"].get("username", "")
            st_pass = st.secrets["spacetrack"].get("password", "")
            if st_user and st_pass:
                env["SPACETRACK_USERNAME"] = st_user
                env["SPACETRACK_PASSWORD"] = st_pass
        subprocess.run([sys.executable, *cmd], check=False, env=env)
    except Exception as e:
        st.warning(f"⚠️ Job failed: {' '.join(cmd)} → {e}")

def ensure_data() -> None:
    """
    Populate any missing CSVs on-demand.
    Conjunctions depend on TLEs; only compute if TLE fetch succeeded.
    """
    Path("data").mkdir(parents=True, exist_ok=True)

    # 1) Independent datasets (or light deps)
    base_jobs = [
        ("data/kp_latest.csv",        ["scripts/fetch_space_weather.py"]),
        ("data/kp_forecast.csv",      ["scripts/forecast_kp.py"]),
        ("data/asteroids.csv",        ["scripts/fetch_asteroids.py"]),
        ("data/commodities.csv",      ["scripts/fetch_commodities.py"]),
        ("data/asteroids_scored.csv", ["scripts/compute_asteroid_profit.py"]),
        ("data/launch_weather.csv",   ["scripts/fetch_launch_weather.py"]),
        ("data/launches.csv",         ["scripts/fetch_launches.py"]),
        ("data/launches_history.csv", ["scripts/fetch_launches_history.py"]),
        # TLEs (now robust: Space-Track primary, CelesTrak fallback inside script)
        ("data/tle_small.csv",        ["scripts/fetch_tle.py"]),
    ]
    for target, cmd in base_jobs:
        p = Path(target)
        if not p.exists() or p.stat().st_size == 0:
            with st.spinner(f"Fetching {p.name} …"):
                run_job(cmd)

    # 2) Dependent dataset: conjunctions (requires TLEs)
    tle_path = Path("data/tle_small.csv")
    conj_path = Path("data/conjunctions.csv")
    if tle_path.exists() and tle_path.stat().st_size > 0:
        if not conj_path.exists() or conj_path.stat().st_size == 0:
            with st.spinner("Propagating conjunctions (sgp4) …"):
                run_job([
                    "scripts/conjunctions.py",
                    "--only_leo", "--threshold_km", "20",
                    "--horizon_h", "24", "--step_s", "60",
                    "--max_sats", "120", "--top_n", "200",
                ])
    else:
        st.info("TLEs not available yet; conjunctions will generate automatically after TLE fetch succeeds.")

# ──────────────────────────
# Sidebar controls
# ──────────────────────────
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh all data now"):
        # Re-run the base jobs (order matters: TLEs before conjunctions)
        for _, cmd in [
            ("data/kp_latest.csv",        ["scripts/fetch_space_weather.py"]),
            ("data/kp_forecast.csv",      ["scripts/forecast_kp.py"]),
            ("data/asteroids.csv",        ["scripts/fetch_asteroids.py"]),
            ("data/commodities.csv",      ["scripts/fetch_commodities.py"]),
            ("data/asteroids_scored.csv", ["scripts/compute_asteroid_profit.py"]),
            ("data/launch_weather.csv",   ["scripts/fetch_launch_weather.py"]),
            ("data/launches.csv",         ["scripts/fetch_launches.py"]),
            ("data/launches_history.csv", ["scripts/fetch_launches_history.py"]),
            ("data/tle_small.csv",        ["scripts/fetch_tle.py"]),
        ]:
            run_job(cmd)

        # Try conjunctions only if TLEs present
        if Path("data/tle_small.csv").exists() and Path("data/tle_small.csv").stat().st_size > 0:
            run_job([
                "scripts/conjunctions.py",
                "--only_leo", "--threshold_km", "20",
                "--horizon_h", "24", "--step_s", "60",
                "--max_sats", "120", "--top_n", "200",
            ])
        st.success("Refreshed. Reload the page to see updates.")

# ──────────────────────────
# First-load populate (no CSVs committed to Git)
# ──────────────────────────
ensure_data()

# ──────────────────────────
# UI — Home
# ──────────────────────────
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;">
      <img src="https://em-content.zobj.net/source/microsoft-teams/363/satellite_1f6f0-fe0f.png" width="36"/>
      <h1 style="margin:0;">Space Intelligence Super App</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Real data. Auto-fetched on demand; nightly refresh via GitHub Actions.")

st.markdown(
    """
**Modules**
1. 🌞 Space Weather — NOAA Kp + 48h forecast  
2. 🪨 Asteroid Mining — JPL SBDB + Δv cost + commodity pricing  
3. 🛰️ Conjunctions — sgp4 propagation + close-approach screening  
4. 🚀 Launch Window — Weather + target-orbit feasibility  
5. 📡 Space Tracker — Missions, reliability, delays
"""
)

# Quick links to pages
links = [
    ("pages/1_Weather.py", "🌞 Space Weather"),
    ("pages/2_Asteroid_Mining.py", "🪨 Asteroid Mining"),
    ("pages/3_Collisions.py", "🛰️ Conjunctions"),
    ("pages/4_Launch_Optimizer.py", "🚀 Launch Window"),
    ("pages/5_Space_Tracker.py", "📡 Space Tracker"),
]
cols = st.columns(len(links))
for col, (path, label) in zip(cols, links):
    with col:
        try:
            st.page_link(path, label=label, icon="➡️")
        except Exception:
            st.markdown(f"- [{label}]({path})")

st.divider()

# Data snapshots
st.subheader("Data snapshots")
def file_status(p: Path) -> str:
    if p.exists() and p.stat().st_size > 0:
        ts = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return f"✅ *(updated {ts})*"
    return "❌"

for name in [
    "kp_latest.csv",
    "kp_forecast.csv",
    "asteroids.csv",
    "commodities.csv",
    "asteroids_scored.csv",
    "tle_small.csv",
    "conjunctions.csv",
    "launch_weather.csv",
    "launches.csv",
    "launches_history.csv",
]:
    p = Path("data") / name
    st.write(f"- `{name}`: {file_status(p)}")
