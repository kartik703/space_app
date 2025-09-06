# app.py — Space Intelligence Super App (robust, no parse_dates warnings)

from pathlib import Path
import os, sys, subprocess, datetime as dt
import pandas as pd
import streamlit as st
import altair as alt

# ─────────────────────────────────────────────────────────────────────────────
# Global CSV safety patch: ignore parse_dates everywhere (in our code or libs)
# ─────────────────────────────────────────────────────────────────────────────
_pd_read_csv_real = pd.read_csv
def _pd_read_csv_no_parse_dates(*args, **kwargs):
    kwargs.pop("parse_dates", None)  # silently drop if present
    return _pd_read_csv_real(*args, **kwargs)
pd.read_csv = _pd_read_csv_no_parse_dates

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Space Intelligence Super App",
                   page_icon="🛰️", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def read_csv_safe(path: str) -> pd.DataFrame:
    """Read CSV; return empty DataFrame on any problem."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

def to_datetime_if_present(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce listed columns to UTC datetimes if they exist."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

def pick_name_col(df: pd.DataFrame) -> str | None:
    for c in ["object", "full_name", "designation", "name", "Object", "OBJECT"]:
        if c in df.columns:
            return c
    return None

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def run_job(cmd: list[str]) -> None:
    """Run a Python script with optional Space-Track credentials from secrets."""
    env = os.environ.copy()
    if "spacetrack" in st.secrets:
        u = st.secrets["spacetrack"].get("username", "")
        p = st.secrets["spacetrack"].get("password", "")
        if u and p:
            env["SPACETRACK_USERNAME"] = u
            env["SPACETRACK_PASSWORD"] = p
    subprocess.run([sys.executable, *cmd], check=False, env=env)

def ensure_data() -> None:
    """If any required snapshot is missing/empty, fetch it. Then build conjunctions."""
    steps = [
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
    for target, cmd in steps:
        p = Path(target)
        if not p.exists() or p.stat().st_size == 0:
            with st.spinner(f"Fetching {p.name} …"):
                run_job(cmd)

    # Build conjunctions only if TLEs exist
    tle = Path("data/tle_small.csv")
    if tle.exists() and tle.stat().st_size > 0:
        conj = Path("data/conjunctions.csv")
        if not conj.exists() or conj.stat().st_size == 0:
            with st.spinner("Propagating conjunctions (sgp4) …"):
                run_job([
                    "scripts/conjunctions.py",
                    "--only_leo", "--threshold_km", "20",
                    "--horizon_h", "24", "--step_s", "60",
                    "--max_sats", "120", "--top_n", "200",
                ])
    else:
        st.info("TLEs not available yet; conjunctions will run after TLE fetch succeeds.")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh all data now"):
        for cmd in [
            ["scripts/fetch_space_weather.py"],
            ["scripts/forecast_kp.py"],
            ["scripts/fetch_asteroids.py"],
            ["scripts/fetch_commodities.py"],
            ["scripts/compute_asteroid_profit.py"],
            ["scripts/fetch_launch_weather.py"],
            ["scripts/fetch_launches.py"],
            ["scripts/fetch_launches_history.py"],
            ["scripts/fetch_tle.py"],
        ]:
            run_job(cmd)
        if (DATA_DIR/"tle_small.csv").exists() and (DATA_DIR/"tle_small.csv").stat().st_size > 0:
            run_job([
                "scripts/conjunctions.py",
                "--only_leo","--threshold_km","20",
                "--horizon_h","24","--step_s","60",
                "--max_sats","120","--top_n","200",
            ])
        st.success("Refreshed. Reload the page to see the latest data.")

    st.markdown("---")
    st.page_link("pages/1_Weather.py", label="🌞 Space Weather")
    st.page_link("pages/2_Asteroid_Mining.py", label="🪨 Asteroid Mining")
    st.page_link("pages/3_Collisions.py", label="🛰️ Conjunctions")
    st.page_link("pages/4_Launch_Optimizer.py", label="🚀 Launch Window")
    st.page_link("pages/5_Space_Tracker.py", label="📡 Space Tracker")

# Ensure snapshots exist (first run bootstrap)
ensure_data()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;">
  <img src="https://em-content.zobj.net/source/microsoft-teams/363/satellite_1f6f0-fe0f.png" width="36"/>
  <h1 style="margin:0;">Space Intelligence Super App</h1>
</div>
""", unsafe_allow_html=True)
st.caption("Real data. Auto-fetched on demand; nightly refresh via GitHub Actions.")

st.markdown("""
**Modules**
1. 🌞 Space Weather — NOAA Kp + 48h forecast  
2. 🪨 Asteroid Mining — JPL SBDB + Δv cost + commodity pricing  
3. 🛰️ Conjunctions — sgp4 propagation + close-approach screening  
4. 🚀 Launch Window — Weather + target-orbit feasibility  
5. 📡 Space Tracker — Missions, reliability, delays
""")

cols = st.columns(5)
for c, (p, label) in zip(cols, [
    ("pages/1_Weather.py","🌞 Space Weather"),
    ("pages/2_Asteroid_Mining.py","🪨 Asteroid Mining"),
    ("pages/3_Collisions.py","🛰️ Conjunctions"),
    ("pages/4_Launch_Optimizer.py","🚀 Launch Window"),
    ("pages/5_Space_Tracker.py","📡 Space Tracker"),
]):
    with c:
        try:
            st.page_link(p, label=label, icon="➡️")
        except Exception:
            st.markdown(f"- [{label}]({p})")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPI strip
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

kp_latest = read_csv_safe("data/kp_latest.csv")
to_datetime_if_present(kp_latest, ["time_tag", "time"])
kp_val = float(kp_latest["Kp"].tail(1).values[0]) if not kp_latest.empty and "Kp" in kp_latest.columns else None
with k1:
    st.metric("Current Kp", f"{kp_val:.1f}" if kp_val is not None else "—")

asteroids_scored = read_csv_safe("data/asteroids_scored.csv")
with k2:
    st.metric("Asteroids scored", f"{len(asteroids_scored):,}" if not asteroids_scored.empty else "—")

conjunctions = read_csv_safe("data/conjunctions.csv")
to_datetime_if_present(conjunctions, ["time"])
with k3:
    st.metric("Conjunction rows", f"{len(conjunctions):,}" if not conjunctions.empty else "0")

launches = read_csv_safe("data/launches.csv")
to_datetime_if_present(launches, ["window_start", "net", "t0", "date"])
with k4:
    st.metric("Upcoming launches", f"{len(launches):,}" if not launches.empty else "—")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Previews (Kp forecast • Asteroid profitability • Launches)
# ─────────────────────────────────────────────────────────────────────────────
cA, cB, cC = st.columns(3)

# Kp forecast preview (auto-detect time/field names)
with cA:
    st.subheader("Kp forecast (48h)")
    kpf = read_csv_safe("data/kp_forecast.csv")
    if not kpf.empty:
        tcol = next((c for c in ["time","time_tag","timestamp","datetime","ts"] if c in kpf.columns), None)
        kcol = next((c for c in ["kp","Kp","kp_pred","kp_value"] if c in kpf.columns), None)
        if tcol and kcol:
            kpf[tcol] = pd.to_datetime(kpf[tcol], errors="coerce", utc=True)
            kpf = kpf.dropna(subset=[tcol]).sort_values(tcol).tail(48)
            chart = alt.Chart(kpf).mark_line().encode(
                x=alt.X(f"{tcol}:T", title="Time"),
                y=alt.Y(f"{kcol}:Q", title="Kp"),
                tooltip=[tcol, kcol],
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Forecast present but missing time/Kp columns.")
    else:
        st.info("No forecast available yet.")

# Asteroid profitability preview (detect columns and coerce numeric)
with cB:
    st.subheader("Top asteroid profitability")
    if not asteroids_scored.empty:
        name_col   = pick_name_col(asteroids_scored)
        profit_col = pick_col(asteroids_scored, ["profit_index","profit","score"])
        dv_col     = pick_col(asteroids_scored, ["dv_kms","dv_km_s","delta_v_kms","delta_v","dv"])
        value_col  = pick_col(asteroids_scored, ["est_value_usd","estimated_value_usd","value_usd","usd_value","est_value"])
        coerce_numeric(asteroids_scored, [profit_col, dv_col, value_col])
        if name_col and profit_col:
            cols_keep = [c for c in [name_col, profit_col, dv_col, value_col] if c]
            top = (asteroids_scored[cols_keep]
                   .dropna(subset=[profit_col])
                   .sort_values(profit_col, ascending=False)
                   .head(15))
            if not top.empty:
                tooltips = [name_col, profit_col] + [c for c in [dv_col, value_col] if c]
                chart = alt.Chart(top).mark_bar().encode(
                    x=alt.X(f"{profit_col}:Q", title="Profit index"),
                    y=alt.Y(f"{name_col}:N", sort="-x", title="Asteroid"),
                    tooltip=tooltips,
                ).properties(height=200)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No ranked asteroids after filtering.")
        else:
            st.info("Asteroid table missing name or profit field.")
    else:
        st.info("No asteroid scores yet.")

# Next launches preview
with cC:
    st.subheader("Next launches (soon)")
    if not launches.empty:
        dcol = next((c for c in ["window_start","net","t0","date"] if c in launches.columns), None)
        if dcol:
            launches[dcol] = pd.to_datetime(launches[dcol], errors="coerce", utc=True)
            cols_show = [c for c in ["name","provider", dcol, "pad"] if c in launches.columns]
            st.dataframe(launches.sort_values(dcol).head(10)[cols_show],
                         use_container_width=True, height=220)
        else:
            st.info("Launch data missing a datetime column.")
    else:
        st.info("No launch data yet.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot table
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Data snapshots")
def file_status(p: Path) -> str:
    if p.exists() and p.stat().st_size > 0:
        ts = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return f"✅ *(updated {ts})*"
    return "❌"

for name in [
    "kp_latest.csv", "kp_forecast.csv", "asteroids.csv", "commodities.csv",
    "asteroids_scored.csv", "tle_small.csv", "conjunctions.csv",
    "launch_weather.csv", "launches.csv", "launches_history.csv",
]:
    p = DATA_DIR / name
    st.write(f"- `{name}`: {file_status(p)}")
