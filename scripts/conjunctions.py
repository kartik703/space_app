import pandas as pd, numpy as np, argparse
from pathlib import Path
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
from itertools import combinations
from scipy.spatial.distance import cdist

IN  = Path("data/tle_small.csv")
OUT = Path("data/conjunctions.csv")

def eci_series(l1, l2, times):
    sat = Satrec.twoline2rv(l1, l2)
    pts = []
    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond/1e6)
        e, r, _ = sat.sgp4(jd, fr)
        pts.append([np.nan, np.nan, np.nan] if e != 0 else r)
    return np.array(pts, dtype=float)  # (T,3)

def main(threshold_km: float, horizon_h: int, step_s: int, max_sats: int, top_n: int, only_leo: bool):
    df = pd.read_csv(IN)
    assert {"l1","l2","perigee_km","apogee_km","satname","norad_id"}.issubset(df.columns), "TLE CSV missing columns"

    # Optionally restrict to LEO (helps find closer passes)
    if only_leo:
        df = df[(df["perigee_km"] <= 1200) & (df["apogee_km"] <= 2000)]

    df = df.head(max_sats).reset_index(drop=True)
    if df.empty:
        pd.DataFrame(columns=["time","sat_a","id_a","sat_b","id_b","sep_km","note"]).to_csv(OUT, index=False)
        print("No sats after filtering; wrote empty file.")
        return

    start = datetime.now(timezone.utc)
    steps = max(1, int((horizon_h*3600)//step_s))
    times = [start + timedelta(seconds=step_s*i) for i in range(steps)]

    # Propagate all sats
    tracks = [eci_series(str(r["l1"]), str(r["l2"]), times) for _, r in df.iterrows()]  # list of (T,3)
    tracks = np.stack(tracks, axis=0)  # (N,T,3)

    ids   = df["norad_id"].astype(str).tolist()
    names = df["satname"].astype(str).tolist()

    # For each pair, compute min separation & the time it occurs
    results = []
    N = len(ids)
    for i, j in combinations(range(N), 2):
        pts_i = tracks[i]; pts_j = tracks[j]
        mask = np.isfinite(pts_i).all(axis=1) & np.isfinite(pts_j).all(axis=1)
        if not mask.any(): 
            continue
        d = np.linalg.norm(pts_i[mask] - pts_j[mask], axis=1)  # km
        k = int(np.argmin(d))
        sep = float(d[k])
        tmin = [t for m, t in zip(mask, times) if m][k]
        results.append({
            "time": tmin,
            "sat_a": names[i], "id_a": ids[i],
            "sat_b": names[j], "id_b": ids[j],
            "sep_km": sep,
            "note": "under_threshold" if sep < threshold_km else "closest_pair"
        })

    out = pd.DataFrame(results).sort_values("sep_km")
    # Keep top N closest pairs so you always see something
    if len(out) > top_n:
        out = out.head(top_n)
    out.to_csv(OUT, index=False)
    # Summary
    n_under = int((out["sep_km"] < threshold_km).sum()) if not out.empty else 0
    print(f"Pairs analyzed: {N*(N-1)//2}, wrote {len(out)} rows "
          f"(closest {top_n}); under {threshold_km} km: {n_under} → {OUT}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold_km", type=float, default=20.0)
    ap.add_argument("--horizon_h",   type=int,   default=24)
    ap.add_argument("--step_s",      type=int,   default=60)   # finer resolution (1 min)
    ap.add_argument("--max_sats",    type=int,   default=120)  # more sats -> more chances
    ap.add_argument("--top_n",       type=int,   default=200)
    ap.add_argument("--only_leo",    action="store_true")
    args = ap.parse_args()
    main(**vars(args))
