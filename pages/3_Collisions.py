import streamlit as st, pandas as pd, subprocess, sys, pathlib, datetime as dt

st.title("🛰 Conjunction Candidates — Interactive (sgp4)")

# Controls
colA, colB, colC = st.columns(3)
with colA:
    only_leo = st.checkbox("Filter to LEO (≤1200/2000 km)", value=True)
    threshold_km = st.slider("Threshold (km)", 1, 50, 20, 1)
with colB:
    horizon_h = st.slider("Horizon (hours)", 6, 72, 48, 6)
    step_s = st.selectbox("Time step", [30, 60, 120, 300], index=1)
with colC:
    max_sats = st.slider("Max satellites", 50, 300, 200, 10)
    top_n = st.slider("Keep closest N pairs", 50, 500, 200, 50)

run_btn = st.button("Recompute (sgp4)")

# Run propagation (calls the script)
if run_btn:
    cmd = [
        sys.executable, "scripts/conjunctions.py",
        "--threshold_km", str(threshold_km),
        "--horizon_h", str(horizon_h),
        "--step_s", str(step_s),
        "--max_sats", str(max_sats),
        "--top_n", str(top_n),
    ]
    if only_leo:
        cmd.append("--only_leo")
    with st.spinner("Propagating… this can take ~10–60s depending on settings"):
        subprocess.run(cmd, check=False)

# Show latest results
p = pathlib.Path("data/conjunctions.csv")
if not p.exists():
    st.info("No conjunctions.csv yet. Click Recompute.")
else:
    df = pd.read_csv(p, parse_dates=["time"]) if p.stat().st_size > 0 else pd.DataFrame()
    if df.empty:
        st.info("No pairs found. Try increasing threshold, horizon, or sats.")
    else:
        st.caption(f"Rows: {len(df)} | Last updated: {dt.datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
        st.dataframe(df.sort_values(["time","sep_km"]).head(300), use_container_width=True)
        st.download_button("⬇️ Download CSV", data=p.read_bytes(), file_name="conjunctions.csv", mime="text/csv")
st.caption("Notes: Distances are 3D ECI norms (sgp4). This is a demo screening, not operational CDM.")
