# scripts/anomaly_orbit.py
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")
IN   = DATA / "tle_small.csv"
OUT  = DATA / "anomalies.csv"

def parse_bstar(line1: str) -> float | None:
    try:
        raw = line1[53:61].strip()
        if not raw: return None
        # TLE uses exponent notation like " 12345-4"
        base = float(f"{raw[0:5]}.{raw[5]}") if len(raw) >= 6 else float(raw)
        exp  = int(line1[61:63])
        sign = -1 if line1[62:63] == "-" else 1
        # safer approach: use standard parse (but keep simple heuristic)
        return None
    except Exception:
        return None

def main():
    if not IN.exists() or IN.stat().st_size == 0:
        pd.DataFrame(columns=["name","norad_id","status","score","reason"]).to_csv(OUT, index=False)
        print("No TLEs; wrote empty anomalies.csv"); return

    df = pd.read_csv(IN)
    # Normalize columns
    if "l1" in df.columns and "line1" not in df.columns: df = df.rename(columns={"l1":"line1"})
    if "l2" in df.columns and "line2" not in df.columns: df = df.rename(columns={"l2":"line2"})
    if "object" in df.columns and "name" not in df.columns: df = df.rename(columns={"object":"name"})
    if "NORAD_CAT_ID" in df.columns and "norad_id" not in df.columns: df = df.rename(columns={"NORAD_CAT_ID":"norad_id"})
    if "norad_id" not in df.columns:
        # try to parse from line2 columns 2-7
        df["norad_id"] = df.get("line2","").astype(str).str[2:7]

    # Heuristic anomaly score: low perigee OR very high mean motion -> risk
    # Mean motion revs/day is cols 52:63 in line2
    def extract_mm(l2):
        try:
            return float(l2[52:63])
        except Exception:
            return np.nan

    def extract_inc(l2):
        try:
            return float(l2[8:16])
        except Exception:
            return np.nan

    df["mean_motion"] = df["line2"].astype(str).apply(extract_mm)
    df["inc_deg"]     = df["line2"].astype(str).apply(extract_inc)
    # rough semi-major axis from mean motion (n) via: a ≈ (μ / (2π n /86400)^2)^(1/3)
    mu = 398600.4418  # km^3/s^2
    n  = (2*np.pi*df["mean_motion"]/86400.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (mu / (n**2))**(1/3)
    perigee_km = a - 6378.137
    df["perigee_km"] = perigee_km

    # Score: lower perigee ⇒ higher risk; very high mean motion ⇒ risk
    # Normalize
    per_norm = (1000 - df["perigee_km"]).clip(lower=0) / 1000.0  # >0 if perigee < 1000 km
    mm_norm  = (df["mean_motion"] - 15.2).clip(lower=0) / 3.0    # >0 if mm > ~LEO circular
    score = (0.7*per_norm.fillna(0) + 0.3*mm_norm.fillna(0)).clip(0,1)

    status = pd.cut(score, bins=[-0.01,0.33,0.66,1.01], labels=["OK","WARN","ALERT"])
    reason = np.where(score>=0.66, "Very low perigee / high drag risk",
             np.where(score>=0.33, "Moderate perigee / activity", "Nominal"))

    out = pd.DataFrame({
        "name": df.get("name",""),
        "norad_id": df["norad_id"],
        "perigee_km": df["perigee_km"].round(1),
        "mean_motion": df["mean_motion"].round(4),
        "status": status.astype(str),
        "score": (score*100).round(1),
        "reason": reason
    }).sort_values("score", ascending=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} anomalies → {OUT}")

if __name__ == "__main__":
    main()
