import streamlit as st, pandas as pd, pathlib

st.title("🌞 Space Weather (NOAA Kp) + 48h Forecast")
p_hist = pathlib.Path("data/kp_latest.csv")
p_fc   = pathlib.Path("data/kp_forecast.csv")

if not p_hist.exists():
    st.warning("No Kp history yet. Run ingestion.")
else:
    hist = pd.read_csv(p_hist, parse_dates=["time_tag"]).sort_values("time_tag")
    st.metric("Latest Kp", f"{hist.iloc[-1]['Kp']:.1f}")
    st.line_chart(hist.set_index("time_tag")["Kp"], height=220)

    if p_fc.exists():
        fc = pd.read_csv(p_fc, parse_dates=["time_tag"])
        st.area_chart(fc.set_index("time_tag")[["kp_lo","kp_hi"]], height=140, use_container_width=True)
        st.line_chart(fc.set_index("time_tag")["kp_pred"], height=220, use_container_width=True)
        st.caption("Model: SARIMAX (daily seasonality). 80% band shown.")
    else:
        st.info("No forecast yet. It will appear after the scheduler runs forecast_kp.py.")
