import streamlit as st, pandas as pd, pathlib

st.title("🪨 Asteroid Mining — Δv + Commodity Pricing")
p = pathlib.Path("data/asteroids_scored.csv")
if not p.exists():
    st.warning("No scored file yet. Wait for scheduler or run compute_asteroid_profit.py.")
else:
    df = pd.read_csv(p)
    st.metric("Asteroids ranked", len(df))
    st.dataframe(
        df[["spkid","name","class","diameter_km","delta_v_kms","revenue_usd","mission_cost_usd","profit_index"]]
        .sort_values("profit_index", ascending=False).head(30),
        use_container_width=True
    )
    st.caption("Δv proxy = base + plane-change + a/e penalties. Prices: latest from fetch_commodities.py.")
