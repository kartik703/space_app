# scripts/propagate_tle.py
import pandas as pd
from pathlib import Path
from sgp4.api import Satrec, jday
import datetime as dt

INFILE  = Path("data/tle_small.csv")
OUTFILE = Path("data/tle_altinc.csv")

def propagate(tle_df: pd.DataFrame, days: int = 50, step_hours: int = 12) -> pd.DataFrame:
    rows = []
    now = dt.datetime.utcnow()
    for _, row in tle_df.iterrows():
        try:
            sat = Satrec.twoline2rv(row["l1"], row["l2"])
        except Exception:
            continue
        for d in range(0, days*24, step_hours):
            ts = now + dt.timedelta(hours=d)
            jd, fr = jday(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
            e, r, v = sat.sgp4(jd, fr)
            if e == 0:
                alt = (sum(x*x for x in r)**0.5) - 6378.137
                inc = sat.inclo * 180.0/3.14159
                rows.append({
                    "norad_id": row["norad_id"],
                    "satname": row["satname"],
                    "datetime": ts.isoformat(),
                    "alt_km": alt,
                    "inc_deg": inc
                })
    return pd.DataFrame(rows)

def main():
    if not INFILE.exists() or INFILE.stat().st_size == 0:
        print("⚠️ No input TLEs found.")
        return

    tle_df = pd.read_csv(INFILE)
    if tle_df.empty:
        print("⚠️ Empty TLE dataframe.")
        return

    df = propagate(tle_df)

    if df.empty:
        print("⚠️ No propagated satellites written.")
        return

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)

    # ✅ append mode with header only if file doesn’t exist
    header = not OUTFILE.exists()
    df.to_csv(OUTFILE, mode="a", index=False, header=header)
    print(f"✅ Propagated {tle_df.shape[0]} sats → {len(df)} rows appended → {OUTFILE}")

if __name__ == "__main__":
    main()
