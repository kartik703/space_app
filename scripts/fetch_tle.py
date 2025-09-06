# scripts/fetch_tle.py
import requests, pandas as pd, time
from math import pi
from pathlib import Path

OUT = Path("data/tle_small.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Try both mirrors + groups (Starlink gives more close passes; fall back to active)
ENDPOINTS = [
    ("https://celestrak.org/NORAD/elements/gp.php", {"GROUP":"starlink","FORMAT":"tle"}),
    ("https://celestrak.com/NORAD/elements/gp.php", {"GROUP":"starlink","FORMAT":"tle"}),
    ("https://celestrak.org/NORAD/elements/gp.php", {"GROUP":"active","FORMAT":"tle"}),
    ("https://celestrak.com/NORAD/elements/gp.php", {"GROUP":"active","FORMAT":"tle"}),
]

MU = 398600.4418  # km^3/s^2
RE = 6378.137     # km

def parse_record(name_line, l1, l2):
    name = (name_line or "").strip()
    norad = l1[2:7].strip()
    inc = float(l2[8:16])
    ecc = float(f"0.{l2[26:33].strip()}") if l2[26:33].strip() else 0.0
    mm = float(l2[52:63])  # rev/day
    n = mm * 2*pi / (24*3600)
    a = (MU / (n*n))**(1/3)
    perigee = max(a*(1-ecc) - RE, 0.0)
    apogee  = max(a*(1+ecc) - RE, 0.0)
    return {
        "satname": name if name else f"NORAD {norad}",
        "norad_id": norad,
        "inclination_deg": inc,
        "eccentricity": ecc,
        "mean_motion_rev_per_day": mm,
        "perigee_km": perigee,
        "apogee_km": apogee,
        "l1": l1.strip(),
        "l2": l2.strip()
    }

def try_fetch():
    s = requests.Session()
    s.headers.update({"User-Agent":"space-intel-app/1.0"})
    for url, params in ENDPOINTS:
        # retry with backoff
        for attempt in range(3):
            try:
                r = s.get(url, params=params, timeout=20)
                r.raise_for_status()
                if "1 " in r.text and "2 " in r.text:
                    return r.text
            except Exception:
                time.sleep(1.5 * (attempt+1))
    return None

def main():
    txt = try_fetch()
    if not txt:
        # If we already have a previous CSV, keep it; else write a safe empty file
        if OUT.exists() and OUT.stat().st_size > 0:
            print("Fetch TLE failed; keeping previous tle_small.csv")
            return
        pd.DataFrame(columns=[
            "satname","norad_id","inclination_deg","eccentricity",
            "mean_motion_rev_per_day","perigee_km","apogee_km","l1","l2"
        ]).to_csv(OUT, index=False)
        print("Fetch TLE failed; wrote empty tle_small.csv to avoid downstream crashes")
        return

    lines = txt.splitlines()
    rows = []
    i = 0
    while i < len(lines)-1:
        cur = lines[i]; nxt = lines[i+1]
        if cur.startswith("1 ") and nxt.startswith("2 "):
            name_line = None if i==0 else (lines[i-1] if not lines[i-1].startswith(("1 ","2 ")) else None)
            try: rows.append(parse_record(name_line, cur, nxt))
            except Exception: pass
            i += 2; continue
        if (not cur.startswith(("1 ","2 "))) and (i+2 < len(lines)) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            try: rows.append(parse_record(cur, lines[i+1], lines[i+2]))
            except Exception: pass
            i += 3; continue
        i += 1

    df = pd.DataFrame(rows)
    if len(df) > 200: df = df.head(200)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} → {OUT}")

if __name__ == "__main__":
    main()
