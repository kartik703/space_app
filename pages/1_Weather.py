# pages/1_Weather.py
import streamlit as st, pandas as pd, altair as alt
from pathlib import Path

st.set_page_config(page_title="Space Weather", page_icon="🌞", layout="wide")
st.title("🌞 Space Weather")

def read_csv_safe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

kp = read_csv_safe("data/kp_latest.csv")
kp_val = float(kp["Kp"].tail(1).values[0]) if not kp.empty and "Kp" in kp.columns else None

c1,c2 = st.columns([1,3])
with c1:
    st.metric("Current Kp", f"{kp_val:.1f}" if kp_val is not None else "—")
    st.caption("Source: NOAA SWPC")

with c2:
    horizon = st.slider("Forecast horizon (hours)", 12, 48, 12)
    kpf = read_csv_safe("data/kp_forecast.csv")
    if not kpf.empty:
        tcol = next((c for c in ["time","time_tag","timestamp","datetime","ts"] if c in kpf.columns), None)
        kcol = next((c for c in ["kp","Kp"] if c in kpf.columns), None)
        if tcol and kcol:
            kpf[tcol] = pd.to_datetime(kpf[tcol], errors="coerce", utc=True)
            kpf = kpf.dropna(subset=[tcol]).sort_values(tcol).tail(horizon)
            st.altair_chart(
                alt.Chart(kpf).mark_line().encode(
                    x=alt.X(f"{tcol}:T", title="Time"),
                    y=alt.Y(f"{kcol}:Q", title="Kp"),
                    tooltip=[tcol,kcol]
                ).properties(height=220),
                use_container_width=True
            )
        else:
            st.info("Forecast present but missing time/Kp columns.")
    else:
        st.info("No forecast yet. Refresh from Home → sidebar.")
