import requests, pandas as pd
from math import pi
from pathlib import Path

OUT = Path("data/tle_small.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

URL = "https://celestrak.org/NORAD/elements/gp.php"
PARAMS = {"GROUP":"starlink","FORMAT":"tle"}

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

def main():
    r = requests.get(URL, params=PARAMS, headers={"User-Agent":"space-intel-app/1.0"}, timeout=90)
    r.raise_for_status()
    lines = r.text.splitlines()

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
    if len(df)>200: df = df.head(200)
    df.to_csv(OUT, index=False)
    print("Saved", len(df), "→", OUT)

if __name__ == "__main__":
    main()
