# pages/7_Anomalies.py
import streamlit as st, pandas as pd, subprocess, sys
from pathlib import Path
from utils import set_background, read_csv_safe, badge, human_ts

# Always set video background
set_background("docs/bg.mp4")

st.title("🛑 Satellite Anomaly Screening")

CSV = Path("data/anomalies.csv")

def read_csv(p: Path):
    if not p.exists() or p.stat().st_size == 0: return pd.DataFrame()
    try: return pd.read_csv(p)
    except: return pd.DataFrame()

c1,c2 = st.columns([1,3])
with c1:
    if st.button("🔁 Recompute anomalies"):
        subprocess.run([sys.executable, "scripts/anomaly_orbit.py"], check=False)

df = read_csv(CSV)
if df.empty:
    st.info("No anomalies table yet. Click **Recompute anomalies** first.")
else:
    status = st.multiselect("Status filter", ["ALERT","WARN","OK"], default=["ALERT","WARN"])
    q = st.text_input("Search name / NORAD")
    dd = df[df["status"].isin(status)]
    if q:
        ql = q.lower()
        dd = dd[dd.astype(str).apply(lambda r: ql in " ".join(r.values).lower(), axis=1)]
    st.dataframe(dd, use_container_width=True, height=500)


