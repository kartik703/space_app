import sys, subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import pydeck as pdk
from utils import set_background, read_csv_safe, badge, human_ts

# Always set video background
set_background("docs/bg.mp4")

# ---------------------------
# Page setup
# ---------------------------

st.title("🧭 Orbital Congestion Explorer")

DATA = Path("data")
RAW  = DATA / "tle_altinc.csv"        # propagated satellites
BINS = DATA / "congestion_bins.csv"   # pre-binned congestion snapshot

# ---------------------------
# Utilities
# ---------------------------
def recompute():
    """Recompute congestion dataset by running script."""
    subprocess.run([sys.executable, "scripts/congestion_map.py"], check=False)

@st.cache_data(show_spinner=False)
def read_df(p: Path) -> pd.DataFrame:
    """Safe CSV reader."""
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

raw = read_df(RAW)
bins = read_df(BINS)

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.subheader("Controls")
    if st.button("🔄 Recompute datasets"):
        recompute(); st.cache_data.clear(); st.rerun()

    use_raw = not raw.empty
    st.toggle("Interactive mode (row-level)", value=use_raw, key="use_raw")

    view = st.radio("View mode", ["2D Heatmap", "3D Globe"])

# ---------------------------
# 2D HEATMAP MODE
# ---------------------------
if view == "2D Heatmap":
    tab1, tab2 = st.tabs(["Heatmap", "Hotspots"])

    if st.session_state.use_raw and not raw.empty:
        df = raw.copy()

        # Bin sizes
        step_a, step_i = 25, 2
        grouped = (
            df.assign(
                ax=(df["alt_km"] // step_a * step_a),
                iy=(df["inc_deg"] // step_i * step_i),
            )
            .groupby(["ax", "iy"])
            .size()
            .reset_index(name="count")
        )

        piv = grouped.pivot(index="ax", columns="iy", values="count").fillna(0)

        # Smooth safely
        piv = piv.rolling(3, min_periods=1).mean()
        piv = piv.T.rolling(3, min_periods=1).mean().T

        heat_df = (
            piv.reset_index()
            .melt("ax", var_name="iy", value_name="count")
            .dropna()
            .rename(columns={"ax": "alt_bin_km", "iy": "inc_bin_deg"})
        )

        with tab1:
            chart = (
                alt.Chart(heat_df)
                .mark_rect()
                .encode(
                    x=alt.X("inc_bin_deg:O", title="Inclination Bin (°)"),
                    y=alt.Y("alt_bin_km:O", title="Altitude Bin (km)"),
                    color=alt.Color("count:Q", scale=alt.Scale(scheme="plasma"), title="Satellites"),
                    tooltip=["alt_bin_km", "inc_bin_deg", "count"]
                )
                .properties(height=600)
            )
            st.altair_chart(chart, use_container_width=True)

        with tab2:
            hot = heat_df.sort_values("count", ascending=False).head(20)
            st.subheader("Top Congestion Bins")
            st.dataframe(hot, use_container_width=True)

    else:
        if bins.empty:
            st.info("No snapshot available. Click **Recompute** in the sidebar.")
            st.stop()

        piv = bins.pivot(index="alt_bin_km", columns="inc_bin_deg", values="count").fillna(0)
        piv = piv.rolling(3, min_periods=1).mean()
        piv = piv.T.rolling(3, min_periods=1).mean().T

        heat_df = (
            piv.reset_index()
            .melt("alt_bin_km", var_name="inc_bin_deg", value_name="count")
            .dropna()
        )

        with tab1:
            chart = (
                alt.Chart(heat_df)
                .mark_rect()
                .encode(
                    x=alt.X("inc_bin_deg:O", title="Inclination Bin (°)"),
                    y=alt.Y("alt_bin_km:O", title="Altitude Bin (km)"),
                    color=alt.Color("count:Q", scale=alt.Scale(scheme="inferno"), title="Satellites"),
                    tooltip=["alt_bin_km", "inc_bin_deg", "count"]
                )
                .properties(height=600)
            )
            st.altair_chart(chart, use_container_width=True)

        with tab2:
            hot = bins.sort_values("count", ascending=False).head(20)
            st.subheader("Top Congestion Bins")
            st.dataframe(hot, use_container_width=True)

# ---------------------------
# 3D GLOBE MODE
# ---------------------------
else:
    if raw.empty:
        st.info("No propagated dataset (`data/tle_altinc.csv`). Run `scripts/propagate_tle.py` first.")
        st.stop()

    st.subheader("🌍 3D Orbital Visualization")

    df = raw.copy()
    df = df.head(2000)  # limit for performance

    # Approximate lat/lon positions for visualization
    df["lat"] = np.sin(np.radians(df["inc_deg"])) * 90
    df["lon"] = (df.index % 360) - 180

    # PyDeck globe view
    view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1, pitch=30)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_color=[200, 30, 0, 160],
        get_radius=10000,
        pickable=True,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=None,
        tooltip={"text": "Satellite: {satname}\nAlt: {alt_km} km\nInc: {inc_deg}°"}
    )

    st.pydeck_chart(deck)


