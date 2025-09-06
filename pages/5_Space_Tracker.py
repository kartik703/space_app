import streamlit as st, pandas as pd, altair as alt, pathlib
from utils import read_csv_safe

st.set_page_config(page_title="Space Tracker", page_icon="📡", layout="wide")
st.title("📡 Space Tracker — Upcoming & History")

up = read_csv_safe("data/launches.csv", parse_dates=["window_start"])
hist = read_csv_safe("data/launches_history.csv", parse_dates=["window_start","window_end"])

q = st.text_input("Search mission/provider/pad", "")

if not up.empty:
    f = up.copy()
    if q:
        ql = q.lower()
        f = f[f.apply(lambda r: ql in str(r.values).lower(), axis=1)]
    st.subheader("Upcoming launches")
    st.dataframe(f.sort_values("window_start").head(200), use_container_width=True)
else:
    st.info("No upcoming launches yet.")

st.markdown("---")
st.subheader("Provider reliability (history)")

if not hist.empty:
    grp = hist.groupby("provider").agg(
        total=("status","count"),
        success=("status", lambda s: (s=="Success").sum()),
        fail=("status", lambda s: (s!="Success").sum())
    ).reset_index()
    grp["success_rate"] = (grp["success"] / grp["total"] * 100).round(1)

    chart = alt.Chart(grp.sort_values("success_rate", ascending=False)).mark_bar().encode(
        x=alt.X("success_rate:Q", title="Success rate (%)"),
        y=alt.Y("provider:N", sort="-x"),
        tooltip=["provider","success_rate","total","success","fail"]
    ).properties(height=340)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No historical dataset yet.")
