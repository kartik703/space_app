# app.py
from pathlib import Path
import subprocess
import sys
import os
import datetime as dt
import streamlit as st
import altair as alt
import pandas as pd

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
# Helpers
# ──────────────────────────
def read_csv_safe(path: str, parse_dates=None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=parse_dates)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

def pick_name_col(df: pd.DataFrame) -> str | None:
    """Return the best asteroid display-name column present."""
    for c in ["object", "full_name", "designation", "name", "Object", "OBJECT"]:
        if c in df.columns:
            return c
    return None

def run_job(cmd: list[str]) -> None:
    """Run a Python script (with args). Never hard-fail the UI."""
    try:
        env = os.environ.copy()
        # pass Space-Track creds from Streamlit secrets -> env for subprocess
        if "spacetrack" in st.secrets:
            u = st.secrets["spacetrack"].get("username", "")
            p = st.secrets["spacetrack"].get("password", "")
            if u and p:
                env["SPACETRACK_USERNAME"] = u
                env["SPACETRACK_PASSWORD"] = p
        subprocess.run([sys.executable, *cmd], check=False, env=env)
    except Exception as e:
        st.warning(f"⚠️ Job failed: {' '.join(cmd)} → {e}")

def ensure_data() -> None:
    """
    Populate any missing CSVs on-demand.
    Conjunctions depend on TLEs; only compute if TLE fetch succeeded.
    """
    Path("data").mkdir(parents=True, exist_ok=True)

    # Base datasets
    base_jobs = [
        ("data/kp_latest.csv",        ["scripts/fetch_space_weather.py"]),
        ("data/kp_forecast.csv",      ["scripts/forecast_kp.py"]),
        ("data/asteroids.csv",        ["scripts/fetch_asteroids.py"]),
        ("data/commodities.csv",      ["scripts/fetch_commodities.py"]),
        ("data/asteroids_scored.csv", ["scripts/compute_asteroid_profit.py"]),
        ("data/launch_weather.csv",   ["scripts/fetch_launch_weather.py"]),
        ("data/launches.csv",         ["scripts/fetch_launches.py"]),
        ("data/launches_history.csv", ["scripts/fetch_launches_history.py"]),
        ("data/tle_small.csv",        ["scripts/fetch_tle.py"]),  # prerequisite for conjunctions
    ]
    for target, cmd in base_jobs:
        p = Path(target)
        if not p.exists() or p.stat().st_size == 0:
            with st.spinner(f"Fetching {p.name} …"):
                run_job(cmd)

    # Conjunctions only if we have TLEs
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
        # Try conjunctions only if TLEs present
        if Path("data/tle_small.csv").exists() and Path("data/tle_small.csv").stat().st_size > 0:
            run_job([
                "scripts/conjunctions.py",
                "--only_leo", "--threshold_km", "20",
                "--horizon_h", "24", "--step_s", "60",
                "--max_sats", "120", "--top_n", "200",
            ])
        st.success("Refreshed. Reload the page to see updates.")
    st.markdown("---")
    st.page_link("pages/1_Weather.py", label="🌞 Space Weather")
    st.page_link("pages/2_Asteroid_Mining.py", label="🪨 Asteroid Mining")
    st.page_link("pages/3_Collisions.py", label="🛰️ Conjunctions")
    st.page_link("pages/4_Launch_Optimizer.py", label="🚀 Launch Window")
    st.page_link("pages/5_Space_Tracker.py", label="📡 Space Tracker")

# ──────────────────────────
# First-load populate
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

# ──────────────────────────
# KPI cards
# ──────────────────────────
col1, col2, col3, col4 = st.columns(4)

kp = read_csv_safe("data/kp_latest.csv", parse_dates=["time_tag"])
kpv = float(kp["Kp"].tail(1).values[0]) if not kp.empty and "Kp" in kp.columns else None
with col1:
    st.metric("Current Kp", f"{kpv:.1f}" if kpv is not None else "—", help="NOAA planetary K-index")

ast = read_csv_safe("data/asteroids_scored.csv")
with col2:
    st.metric("Asteroids scored", f"{len(ast):,}" if not ast.empty else "—")

conj = read_csv_safe("data/conjunctions.csv", parse_dates=["time"])
with col3:
    st.metric("Conjunction rows", f"{len(conj):,}" if not conj.empty else "0")

ll2 = read_csv_safe("data/launches.csv", parse_dates=["window_start"])
with col4:
    st.metric("Upcoming launches", f"{len(ll2):,}" if not ll2.empty else "—")

st.divider()

# ──────────────────────────
# Quick previews
# ──────────────────────────
c1, c2, c3 = st.columns(3)

# Kp forecast
with c1:
    st.subheader("Kp forecast (48h)")
    kpf = read_csv_safe("data/kp_forecast.csv", parse_dates=["time"])
    if not kpf.empty and "kp" in {c.lower() for c in kpf.columns}:
        # normalize kp column name (kp or Kp)
        if "kp" in kpf.columns:
            col_kp = "kp"
        elif "Kp" in kpf.columns:
            kpf = kpf.rename(columns={"Kp": "kp"})
            col_kp = "kp"
        else:
            col_kp = None
        if col_kp:
            chart = alt.Chart(kpf.tail(48)).mark_line().encode(
                x="time:T", y=alt.Y(f"{col_kp}:Q", title="Kp"),
                tooltip=["time:T", alt.Tooltip(f"{col_kp}:Q", title="Kp")]
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Forecast present but missing Kp column.")
    else:
        st.info("No forecast available yet.")

# Asteroid profitability — robust to name column differences
with c2:
    st.subheader("Top asteroid profitability")
    if not ast.empty:
        # pick a display name column
        name_col = pick_name_col(ast)
        # coerce numerics safely
        for col in ["profit_index", "dv_kms", "est_value_usd", "diameter_km"]:
            if col in ast.columns:
                ast[col] = pd.to_numeric(ast[col], errors="coerce")
        if name_col and "profit_index" in ast.columns:
            top = (
                ast[[name_col, "profit_index", "dv_kms", "est_value_usd"]]
                .dropna(subset=["profit_index"])
                .sort_values("profit_index", ascending=False)
                .head(15)
            )
            if not top.empty:
                chart = alt.Chart(top).mark_bar().encode(
                    x=alt.X("profit_index:Q", title="Profit index"),
                    y=alt.Y(f"{name_col}:N", sort="-x", title="Asteroid"),
                    tooltip=[name_col, "profit_index", "dv_kms", "est_value_usd"]
                ).properties(height=200)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No ranked asteroids available yet (profit_index empty after filtering).")
        else:
            missing = "name column" if not name_col else "profit_index"
            st.info(f"Asteroid table missing required field: {missing}.")
    else:
        st.info("No asteroid scores yet.")

# Launch preview
with c3:
    st.subheader("Next launches (soon)")
    if not ll2.empty and "window_start" in ll2.columns:
        soon = ll2.sort_values("window_start").head(10)[
            [c for c in ["name", "provider", "window_start", "pad"] if c in ll2.columns]
        ]
        st.dataframe(soon, use_container_width=True, height=220)
    else:
        st.info("No launch data yet.")

st.divider()

# ──────────────────────────
# Data snapshots
# ──────────────────────────
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
