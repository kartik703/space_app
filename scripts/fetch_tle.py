# scripts/fetch_tle.py
import os, time, requests, pandas as pd
from math import pi
from pathlib import Path

OUT = Path("data/tle_small.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

MU = 398600.4418  # km^3/s^2
RE = 6378.137     # km

def parse_triplets(text: str) -> pd.DataFrame:
    """
    Parse TLE text that may include a leading '0 NAME' line per object (Space-Track)
    or plain 3-line blocks (name optional).
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    rows, i = [], 0
    def rec(name_line, l1, l2):
        name = (name_line or "").strip()
        name = name[1:].strip() if name.startswith("0 ") else name
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

    while i < len(lines):
        ln = lines[i]
        # patterns:
        # 0 NAME
        # 1 ...
        # 2 ...
        if ln.startswith("0 ") and i+2 < len(lines) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            try: rows.append(rec(ln, lines[i+1], lines[i+2]))
            except Exception: pass
            i += 3; continue
        # NAME (no leading 0), then 1/2
        if (not ln.startswith(("0 ","1 ","2 "))) and i+2 < len(lines) and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
            try: rows.append(rec(ln, lines[i+1], lines[i+2]))
            except Exception: pass
            i += 3; continue
        # plain 1/2 (no name)
        if ln.startswith("1 ") and i+1 < len(lines) and lines[i+1].startswith("2 "):
            name_line = lines[i-1] if i>0 and not lines[i-1].startswith(("0 ","1 ","2 ")) else None
            try: rows.append(rec(name_line, ln, lines[i+1]))
            except Exception: pass
            i += 2; continue
        i += 1

    df = pd.DataFrame(rows)
    # keep it light
    if len(df) > 200: df = df.head(200)
    return df

def fetch_spacetrack(group: str, max_attempts: int = 2, timeout: int = 25) -> str | None:
    """
    Fetch latest TLEs from Space-Track for a group.
    Requires env vars: SPACETRACK_USERNAME, SPACETRACK_PASSWORD.
    """
    user = os.environ.get("SPACETRACK_USERNAME")
    pwd  = os.environ.get("SPACETRACK_PASSWORD")
    if not user or not pwd:
        return None

    s = requests.Session()
    s.headers.update({"User-Agent": "space-intel-app/1.0"})
    login_url = "https://www.space-track.org/ajaxauth/login"
    q_starlink = "https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/OBJECT_NAME/STARLINK%25/format/tle"
    q_active   = "https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/DECAYED/false/OBJECT_TYPE/PAYLOAD/format/tle"

    query_url = q_starlink if group == "starlink" else q_active

    for attempt in range(max_attempts):
        try:
            # login
            r = s.post(login_url, data={"identity": user, "password": pwd}, timeout=timeout)
            r.raise_for_status()
            # fetch
            r2 = s.get(query_url, timeout=timeout)
            r2.raise_for_status()
            txt = r2.text.strip()
            if "1 " in txt and "2 " in txt:
                return txt
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None

def fetch_celestrak(group: str, max_attempts: int = 3, timeout: int = 15) -> str | None:
    mirrors = [
        "https://celestrak.org/NORAD/elements/gp.php",
        "https://celestrak.com/NORAD/elements/gp.php",
    ]
    params = {"GROUP": group, "FORMAT": "tle"}
    s = requests.Session()
    s.headers.update({"User-Agent":"space-intel-app/1.0"})
    for url in mirrors:
        for attempt in range(max_attempts):
            try:
                r = s.get(url, params=params, timeout=timeout)
                r.raise_for_status()
                txt = r.text.strip()
                if "1 " in txt and "2 " in txt:
                    return txt
            except Exception:
                time.sleep(1.2 * (attempt + 1))
    return None

def main():
    # 1) Space-Track Starlink, then Active
    for group in ("starlink", "active"):
        txt = fetch_spacetrack(group)
        if txt:
            df = parse_triplets(txt)
            if not df.empty:
                df.to_csv(OUT, index=False); print(f"[Space-Track:{group}] Saved {len(df)} → {OUT}"); return

    # 2) Fallback CelesTrak Starlink, then Active (mirrors)
    for group in ("starlink", "active"):
        txt = fetch_celestrak(group)
        if txt:
            df = parse_triplets(txt)
            if not df.empty:
                df.to_csv(OUT, index=False); print(f"[CelesTrak:{group}] Saved {len(df)} → {OUT}"); return

    # 3) Absolute fallback: keep previous file if any; else create safe empty
    if OUT.exists() and OUT.stat().st_size > 0:
        print("TLE fetch failed; keeping previous tle_small.csv")
        return
    pd.DataFrame(columns=[
        "satname","norad_id","inclination_deg","eccentricity",
        "mean_motion_rev_per_day","perigee_km","apogee_km","l1","l2"
    ]).to_csv(OUT, index=False)
    print("TLE fetch failed; wrote empty tle_small.csv")

if __name__ == "__main__":
    main()
