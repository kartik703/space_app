import streamlit as st, pandas as pd, altair as alt
from utils import read_csv_safe

st.set_page_config(page_title="Asteroid Mining", page_icon="🪨", layout="wide")
st.title("🪨 Asteroid Mining — Profitability Explorer")

df = read_csv_safe("data/asteroids_scored.csv")
if df.empty:
    st.warning("No asteroid scores yet. Use Home → Refresh.")
    st.stop()

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    dv_max = st.slider("Max Δv (km/s)", 3.0, 15.0, 8.0, 0.5)
with col2:
    size_min = st.slider("Min diameter (km)", 0.0, 50.0, 0.0, 0.5)
with col3:
    top_n = st.slider("Top N", 10, 200, 50, 10)

q = df.copy()
for col in ["dv_kms","diameter_km"]:
    if col not in q.columns: q[col] = 0.0
q = q[(q["dv_kms"] <= dv_max) & (q["diameter_km"] >= size_min)]

q = q.sort_values("profit_index", ascending=False).head(top_n)
st.caption(f"Filtered: {len(q)} asteroids")

# Table + chart
tab1, tab2 = st.tabs(["Table", "Chart"])
with tab1:
    st.dataframe(q[["object","dv_kms","diameter_km","profit_index","est_value_usd"]], use_container_width=True)
with tab2:
    chart = alt.Chart(q).mark_circle(size=120).encode(
        x=alt.X("dv_kms:Q", title="Δv (km/s)"),
        y=alt.Y("profit_index:Q", title="Profit index"),
        color=alt.Color("diameter_km:Q", title="Diameter (km)"),
        tooltip=["object","dv_kms","diameter_km","profit_index","est_value_usd"]
    ).properties(height=420)
    st.altair_chart(chart, use_container_width=True)

st.download_button("⬇️ Download filtered CSV", q.to_csv(index=False).encode(), "asteroids_filtered.csv", "text/csv")
