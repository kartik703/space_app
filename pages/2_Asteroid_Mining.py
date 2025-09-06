# pages/2_Asteroid_Mining.py
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Asteroid Mining", page_icon="🪨", layout="wide")
st.title("🪨 Asteroid Mining — Profitability Explorer")

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
    for c in ["object", "full_name", "designation", "name", "Object", "OBJECT"]:
        if c in df.columns:
            return c
    return None

df = read_csv_safe("data/asteroids_scored.csv")
if df.empty:
    st.warning("No asteroid scores yet. Use Home → Refresh.")
    st.stop()

# coerce numerics
for col in ["profit_index", "dv_kms", "est_value_usd", "diameter_km", "albedo"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    dv_max = st.slider("Max Δv (km/s)", 3.0, 15.0, float(df["dv_kms"].quantile(0.8) if "dv_kms" in df else 8.0), 0.5)
with col2:
    size_min = st.slider("Min diameter (km)", 0.0, 50.0, float((df["diameter_km"].median() if "diameter_km" in df else 0.0)), 0.5)
with col3:
    top_n = st.slider("Top N", 10, 200, 50, 10)

name_col = pick_name_col(df)
q = df.copy()
if "dv_kms" in q.columns:
    q = q[q["dv_kms"] <= dv_max]
if "diameter_km" in q.columns:
    q = q[q["diameter_km"] >= size_min]

q = q.sort_values("profit_index", ascending=False) if "profit_index" in q.columns else q
q = q.head(top_n)

st.caption(f"Filtered: {len(q)} asteroids")

# Table + chart
tab1, tab2 = st.tabs(["Table", "Chart"])

with tab1:
    display_cols = [c for c in [name_col, "dv_kms", "diameter_km", "profit_index", "est_value_usd", "albedo"] if c and c in q.columns]
    if display_cols:
        st.dataframe(q[display_cols], use_container_width=True)
    else:
        st.info("No suitable columns to display. Check your CSV headers.")

with tab2:
    if name_col and all(c in q.columns for c in ["dv_kms", "profit_index"]):
        chart = alt.Chart(q).mark_circle(size=120).encode(
            x=alt.X("dv_kms:Q", title="Δv (km/s)"),
            y=alt.Y("profit_index:Q", title="Profit index"),
            color=alt.Color("diameter_km:Q", title="Diameter (km)", scale=alt.Scale(scheme="blues")),
            tooltip=[name_col, "dv_kms", "diameter_km", "profit_index", "est_value_usd"]
        ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Need name, Δv, and profit_index columns to plot.")

# Download
st.download_button(
    "⬇️ Download filtered CSV",
    q.to_csv(index=False).encode(),
    "asteroids_filtered.csv",
    "text/csv",
)
