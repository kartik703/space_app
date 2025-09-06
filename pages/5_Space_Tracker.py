import streamlit as st, pandas as pd, pathlib

st.title("📡 Space Launch Tracker — Missions, Reliability, Delays")

p_up = pathlib.Path("data/launches.csv")
p_hist = pathlib.Path("data/launches_history.csv")

col1, col2 = st.columns(2)

if p_up.exists():
    up = pd.read_csv(p_up, parse_dates=["window_start"]).sort_values("window_start")
    with col1:
        st.metric("Upcoming launches", len(up))
        st.dataframe(up[["window_start","name","provider","vehicle","pad","location","mission"]].head(50),
                     use_container_width=True)
else:
    st.warning("No upcoming launches data.")

if p_hist.exists():
    hist = pd.read_csv(p_hist, parse_dates=["window_start","net"])
    rel = (hist.assign(success=lambda d: (d["status"]=="Success").astype(int))
                .groupby("provider")["success"].mean().sort_values(ascending=False))
    delays = (hist.groupby("provider")["delay_hours"].mean().sort_values())
    with col2:
        st.subheader("Provider reliability (success rate)")
        st.dataframe(rel.rename("success_rate").reset_index(), use_container_width=True)
        st.subheader("Average schedule slip (hours)")
        st.dataframe(delays.rename("avg_delay_h").reset_index(), use_container_width=True)
    st.caption("Source: Launch Library 2. Rates over the latest ~100 launches.")
else:
    st.info("No historical launches yet.")
