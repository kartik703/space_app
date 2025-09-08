# pages/2_Asteroid_Mining.py
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from utils import set_background, read_csv_safe, badge, human_ts

# Always set video background
set_background("docs/bg.mp4")

st.title(" Asteroid Mining — Profitability Explorer")

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

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

df = read_csv_safe("data/asteroids_scored.csv")
if df.empty:
    st.warning("No asteroid scores yet. Use Home → Refresh.")
    st.stop()

# detect key columns
name_col   = pick_name_col(df)
profit_col = pick_col(df, ["profit_index", "profit", "score"])
dv_col     = pick_col(df, ["dv_kms", "dv_km_s", "delta_v_kms", "delta_v", "dv"])
diam_col   = pick_col(df, ["diameter_km", "diameter", "diam_km"])
value_col  = pick_col(df, ["est_value_usd", "estimated_value_usd", "value_usd", "usd_value", "est_value"])
albedo_col = pick_col(df, ["albedo", "geom_albedo"])

# numeric coercion
coerce_numeric(df, [profit_col, dv_col, diam_col, value_col, albedo_col])

# Filters (use detected fields, fall back to defaults)
col1, col2, col3 = st.columns(3)
with col1:
    default_dv = float(df[dv_col].quantile(0.8)) if dv_col and dv_col in df.columns else 8.0
    dv_max = st.slider("Max Δv (km/s)", 3.0, 15.0, default_dv, 0.5)
with col2:
    default_diam = float(df[diam_col].median()) if diam_col and diam_col in df.columns else 0.0
    size_min = st.slider("Min diameter (km)", 0.0, 50.0, default_diam, 0.5)
with col3:
    top_n = st.slider("Top N", 10, 200, 50, 10)

q = df.copy()
if dv_col:   q = q[q[dv_col] <= dv_max]
if diam_col: q = q[q[diam_col] >= size_min]
q = q.sort_values(profit_col, ascending=False) if profit_col else q
q = q.head(top_n)
st.caption(f"Filtered: {len(q)} asteroids")

# Table + Chart
tab1, tab2 = st.tabs(["Table", "Chart"])

with tab1:
    display_cols = [c for c in [name_col, dv_col, diam_col, profit_col, value_col, albedo_col] if c]
    if display_cols:
        st.dataframe(q[display_cols], use_container_width=True)
    else:
        st.info("No suitable columns to display. Check CSV headers.")

with tab2:
    if name_col and dv_col and profit_col:
        tooltips = [name_col, profit_col, dv_col] + [c for c in [diam_col, value_col] if c]
        color_enc = alt.Color(f"{diam_col}:Q", title="Diameter (km)") if diam_col else alt.value("steelblue")
        chart = alt.Chart(q).mark_circle(size=120).encode(
            x=alt.X(f"{dv_col}:Q", title="Δv (km/s)"),
            y=alt.Y(f"{profit_col}:Q", title="Profit index"),
            color=color_enc,
            tooltip=tooltips
        ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Need name, Δv, and profit columns to plot.")

st.download_button(
    "⬇️ Download filtered CSV",
    q.to_csv(index=False).encode(),
    "asteroids_filtered.csv",
    "text/csv",
)

