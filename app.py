# app.py
import streamlit as st
import altair as alt
from pathlib import Path
import subprocess
from utils import set_background, read_csv_safe, badge, human_ts

# -----------------------------
# CONFIG & BACKGROUND
# -----------------------------
st.set_page_config(
    page_title="🌌 Space Intelligence Super App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Always show video background
set_background("docs/bg.mp4")

# App title
st.title("🚀 Space Intelligence Super App")
st.caption("AI-powered dashboard for space weather, asteroid mining, orbital safety, and launch insights.")

DATA = Path("data")

# -----------------------------
# SIDEBAR MODULES
# -----------------------------
st.sidebar.title("📊 Modules")
choice = st.sidebar.radio(
    "Select Module",
    ["🌞 Space Weather", "🪨 Asteroid Mining", "🛰 Orbital Congestion",
     "🚀 Launch Success", "📡 Launch Tracker", "⚠️ Anomalies"]
)

# -----------------------------
# 🌞 SPACE WEATHER
# -----------------------------
if choice == "🌞 Space Weather":
    st.subheader("🌞 Space Weather (NOAA + AI Forecast)")

    kp = read_csv_safe(DATA / "kp_latest.csv")
    fc = read_csv_safe(DATA / "kp_forecast.csv")

    if not kp.empty and "Kp" in kp.columns:
        st.metric("Current Kp", kp["Kp"].iloc[-1])
        st.caption("Source: NOAA SWPC")

    if not fc.empty:
        # Normalize time column
        if "time" not in fc.columns:
            for alt_name in ["time_tag", "timestamp", "date"]:
                if alt_name in fc.columns:
                    fc = fc.rename(columns={alt_name: "time"})
                    break

        if {"time", "forecast"}.issubset(fc.columns):
            horizon = st.slider("Forecast horizon (hours)", 12, 48, 24)
            subset = fc.head(horizon)

            chart = alt.Chart(subset).mark_line(color="cyan").encode(
                x="time:T",
                y=alt.Y("forecast:Q", title="Kp Index"),
                tooltip=["time", "forecast"]
            ).properties(title=f"{horizon}h Forecast")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning(f"⚠️ Forecast file missing expected columns. Found: {list(fc.columns)}")
    else:
        st.info("No forecast data available.")

# -----------------------------
# 🪨 ASTEROID MINING
# -----------------------------
elif choice == "🪨 Asteroid Mining":
    st.subheader("🪨 Asteroid Mining Opportunities")

    ast = read_csv_safe(DATA / "asteroids_scored.csv")

    if not ast.empty and "profit_index" in ast.columns:
        top = ast.sort_values("profit_index", ascending=False).head(10)

        # Show available columns for debugging
        st.caption(f"Available columns: {list(ast.columns)}")

        # Define possible column mappings
        column_aliases = {
            "object": ["object", "name", "id"],
            "dv_kms": ["dv_kms", "delta_v", "delta_v_kms"],
            "est_value_usd": ["est_value_usd", "value_usd", "est_value"],
            "profit_index": ["profit_index"]
        }

        # Build safe list of columns that actually exist
        selected_cols = []
        for canonical, aliases in column_aliases.items():
            for alias in aliases:
                if alias in top.columns:
                    selected_cols.append(alias)
                    break

        if selected_cols:
            st.dataframe(top[selected_cols])
        else:
            st.warning("⚠️ Could not find expected asteroid mining columns, showing raw data instead.")
            st.dataframe(top.head(10))

        # Chart: always plot profit_index
        if "profit_index" in top.columns:
            chart = alt.Chart(top.reset_index()).mark_bar(color="orange").encode(
                x="profit_index:Q",
                y=alt.Y(selected_cols[0] if selected_cols else top.columns[0], sort="-x"),
                tooltip=selected_cols
            ).properties(width=700, height=400, title="Top 10 Asteroid Mining Opportunities")
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No asteroid mining dataset available. Run the pipeline to generate `asteroids_scored.csv`.")

# -----------------------------
# 🛰 ORBITAL CONGESTION
# -----------------------------
elif choice == "🛰 Orbital Congestion":
    st.subheader("🛰 Orbital Congestion Heatmap")
    cong = read_csv_safe(DATA / "congestion_bins.csv")

    if not cong.empty and {"alt_bin_km","inc_bin_deg","count"}.issubset(cong.columns):
        chart = alt.Chart(cong).mark_rect().encode(
            x=alt.X("inc_bin_deg:O", title="Inclination Bin (°)"),
            y=alt.Y("alt_bin_km:O", title="Altitude Bin (km)"),
            color="count:Q",
            tooltip=["alt_bin_km","inc_bin_deg","count"]
        ).properties(title="Orbital Congestion Map")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Congestion data invalid or missing. Run scripts/congestion_map.py.")

# -----------------------------
# 🚀 LAUNCH SUCCESS
# -----------------------------
elif choice == "🚀 Launch Success":
    st.subheader("🚀 Hybrid Launch Success Predictor")
    ls = read_csv_safe(DATA / "launch_success_scores.csv")

    if not ls.empty:
        st.dataframe(ls)

        chart = alt.Chart(ls).mark_bar(color="lime").encode(
            x="go_score:Q",
            y=alt.Y("name:N", sort="-x"),
            tooltip=["window_start","factors"]
        ).properties(title="Launch GO Scores")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No launch success scores computed yet.")

# -----------------------------
# 📡 LAUNCH TRACKER
# -----------------------------
elif choice == "📡 Launch Tracker":
    st.subheader("📡 Upcoming Launches")
    launches = read_csv_safe(DATA / "launches.csv")

    if not launches.empty and {"name","window_start","provider","pad"}.issubset(launches.columns):
        st.dataframe(launches[["name","window_start","provider","pad"]].head(15))
    else:
        st.warning("No upcoming launches available.")

# -----------------------------
# ⚠️ ANOMALIES
# -----------------------------
elif choice == "⚠️ Anomalies":
    st.subheader("⚠️ Orbital Anomalies")

    # 🔄 Auto-refresh anomalies by running pipeline
    try:
        subprocess.run(["python", "scripts/run_all.py"], check=True)
    except Exception as e:
        st.warning(f"⚠️ Could not refresh anomalies automatically: {e}")

    anom = read_csv_safe(DATA / "anomalies.csv")

    if not anom.empty:
        st.dataframe(anom.head(20))
    else:
        st.info("No anomalies detected yet.")

# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.success("✅ Data auto-refreshed via run_all.py")
