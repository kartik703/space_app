# scripts/launch_success_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
WTH  = DATA / "launch_weather.csv"
KP   = DATA / "kp_latest.csv"
LL2  = DATA / "launches.csv"
HIST = DATA / "launches_history.csv"
OUT  = DATA / "launch_success_scores.csv"


def read_csv(p: Path):
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except:
        return pd.DataFrame()


def compute_rule_based(lw, kpval):
    """Compute a simple rule-based launch GO score."""
    wind = next((c for c in ["wind_speed_10m","windspeed_10m","wind","wind_speed"] if c in lw.columns), None)
    cloud = next((c for c in ["cloud_cover","cloudcover","clouds"] if c in lw.columns), None)
    precip = next((c for c in ["precipitation","precip","rain"] if c in lw.columns), None)

    if not all([wind, cloud, precip]):
        return 50.0, ["Insufficient weather data"]

    w = lw.sort_values("time").tail(1).iloc[0]
    base = 100.0
    factors = []

    if w[wind] > 12: base -= (w[wind]-12)*2; factors.append("High wind")
    if w[cloud] > 70: base -= (w[cloud]-70)*0.6; factors.append("Cloud cover")
    if w[precip] > 0.1: base -= min(30, w[precip]*50); factors.append("Precipitation")
    if kpval >= 6: base -= (kpval-5)*5; factors.append("Geomagnetic activity")

    score = float(np.clip(base, 0, 100))
    return score, (factors if factors else ["Nominal"])


def train_ml_model(hist_df):
    """Train logistic regression model from historical launches."""
    # Try to guess useful cols
    wind = next((c for c in ["wind_speed_10m","windspeed_10m","wind","wind_speed"] if c in hist_df.columns), None)
    cloud = next((c for c in ["cloud_cover","clouds"] if c in hist_df.columns), None)
    precip = next((c for c in ["precipitation","precip","rain"] if c in hist_df.columns), None)
    kpcol = next((c for c in ["kp","Kp","geomag_index"] if c in hist_df.columns), None)

    if not all([wind, cloud, precip, kpcol]) or "success" not in hist_df.columns:
        return None

    X = hist_df[[wind, cloud, precip, kpcol]].fillna(0)
    y = hist_df["success"].astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500)
    model.fit(Xs, y)
    return model, scaler, [wind, cloud, precip, kpcol]


def predict_ml_score(model, scaler, cols, lw, kpval):
    """Predict ML-based probability of success."""
    if model is None or scaler is None:
        return 50.0

    # Use last weather snapshot
    w = lw.sort_values("time").tail(1).iloc[0]
    x = [[
        w.get(cols[0], 0),
        w.get(cols[1], 0),
        w.get(cols[2], 0),
        kpval
    ]]
    x_scaled = scaler.transform(x)
    proba = model.predict_proba(x_scaled)[0,1]  # success probability
    return float(proba * 100)


def main():
    lw = read_csv(WTH)
    kp = read_csv(KP)
    ll = read_csv(LL2)
    hist = read_csv(HIST)

    if lw.empty or ll.empty:
        pd.DataFrame(columns=["name","window_start","final_go_score","rule_score","ml_score","factors"]).to_csv(OUT, index=False)
        print("Missing weather or launches; wrote empty scores.")
        return

    # Ensure datetime parsing
    for df, col_opts in [(lw, ["time","timestamp","datetime","ts"]),
                         (ll, ["window_start","net","t0","date"])]:
        col = next((c for c in col_opts if c in df.columns), None)
        if col: df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Latest Kp
    kpval = float(kp["Kp"].tail(1).values[0]) if not kp.empty and "Kp" in kp.columns else 2.0

    # Rule-based
    rule_score, factors = compute_rule_based(lw, kpval)

    # ML model
    model_bundle = train_ml_model(hist)
    ml_score = predict_ml_score(*model_bundle, lw, kpval) if model_bundle else 50.0

    # Hybrid final score
    final_score = 0.5*rule_score + 0.5*ml_score

    # Build output for next launches
    tcol_ll = next((c for c in ["window_start","net","t0","date"] if c in ll.columns), None)
    ll = ll.dropna(subset=[tcol_ll]).sort_values(tcol_ll).head(10)
    out = pd.DataFrame({
        "name": ll.get("name", pd.Series([None]*len(ll))),
        "window_start": ll[tcol_ll].dt.strftime("%Y-%m-%d %H:%M UTC"),
        "rule_score": rule_score,
        "ml_score": ml_score,
        "final_go_score": final_score,
        "factors": ", ".join(factors)
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} hybrid launch GO scores → {OUT}")


if __name__ == "__main__":
    main()
