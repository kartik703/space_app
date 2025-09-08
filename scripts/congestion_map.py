# scripts/congestion_map.py
"""
Builds congestion datasets from TLEs or element tables.

Outputs:
  - data/tle_altinc.csv         (alt_km, inc_deg, satname, provider, constellation)
  - data/congestion_bins.csv    (alt_bin_km, inc_bin_deg, count)

Autodetects input schema:
  A) elements table: columns ["inclination_deg","perigee_km"] (optional "apogee_km")
  B) raw TLEs: ["name","line1","line2"] (or ["object","l1","l2"]) -> sgp4 single-epoch

Binning (snapshot):
  altitude 25 km, inclination 2 deg
"""
from __future__ import annotations
import math, re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except Exception:
    SGP4_AVAILABLE = False

DATA = Path("data")
IN   = DATA / "tle_small.csv"
RAW  = DATA / "tle_altinc.csv"
OUT  = DATA / "congestion_bins.csv"

R_EARTH_KM = 6378.137

PROVIDER_MAP = [
    (r"STARLINK|SPACEX",        ("SpaceX",        "Starlink")),
    (r"ONEWEB",                  ("OneWeb",        "OneWeb")),
    (r"IRIDIUM",                 ("Iridium",       "Iridium")),
    (r"GLONASS|KOSMOS|COSMOS",   ("ROSCOSMOS",     "GLONASS/Cosmos")),
    (r"BEIDOU|BDS",              ("CNSA",          "BeiDou")),
    (r"GPS|NAVSTAR",             ("USSF",          "GPS")),
    (r"GALILEO",                 ("ESA",           "Galileo")),
    (r"PLANET|DOVE|FLOCK",       ("Planet",        "Flock")),
    (r"ONEWEB|UTELSAT",          ("Eutelsat",      "OneWeb/Eutelsat")),
]

def guess_provider(name: str) -> tuple[str, str]:
    n = (name or "").upper()
    for pat, (prov, const) in PROVIDER_MAP:
        if re.search(pat, n):
            return prov, const
    return "Unknown", "Unknown"

def _load_input() -> pd.DataFrame:
    if not IN.exists() or IN.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(IN)
    except Exception:
        return pd.DataFrame()
    ren = {}
    if "object" in df.columns and "name" not in df.columns: ren["object"] = "name"
    if "l1" in df.columns and "line1" not in df.columns: ren["l1"] = "line1"
    if "l2" in df.columns and "line2" not in df.columns: ren["l2"] = "line2"
    if ren:
        df = df.rename(columns=ren)
    return df

def _rows_from_elements(df: pd.DataFrame) -> pd.DataFrame:
    satname = (df["satname"] if "satname" in df.columns else df.get("name", pd.Series(["Unknown"]*len(df)))).astype(str)
    alt = pd.to_numeric(df["perigee_km"], errors="coerce").clip(0, 2000)
    if "apogee_km" in df.columns:
        apo = pd.to_numeric(df["apogee_km"], errors="coerce")
        good = apo.between(0, 2000)
        alt = np.where(good, (alt + apo)/2.0, alt)
    inc = pd.to_numeric(df["inclination_deg"], errors="coerce").clip(0, 180)
    prov, const = zip(*[guess_provider(s) for s in satname])
    rows = pd.DataFrame({"alt_km": alt, "inc_deg": inc, "satname": satname, "provider": prov, "constellation": const})
    rows = rows.dropna(subset=["alt_km","inc_deg"])
    return rows

def _rows_from_tle(df: pd.DataFrame, sample:int|None=1500) -> pd.DataFrame:
    if not SGP4_AVAILABLE:
        return pd.DataFrame()
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    when = datetime.now(timezone.utc)
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute, when.second + when.microsecond/1e6)

    rows = []
    for _, r in df.iterrows():
        name = str(r.get("name", r.get("satname", "Unknown")))
        l1   = str(r.get("line1","")).strip()
        l2   = str(r.get("line2","")).strip()
        if len(l1) < 60 or len(l2) < 60:
            continue
        try:
            sat = Satrec.twoline2rv(l1, l2)
            e, pos, vel = sat.sgp4(jd, fr)
            if e != 0:
                continue
            rmag = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            alt = rmag - R_EARTH_KM
            inc = math.degrees(sat.inclo)
            if not np.isfinite(alt) or not np.isfinite(inc): continue
            prov, const = guess_provider(name)
            rows.append((alt, inc, name, prov, const))
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["alt_km","inc_deg","satname","provider","constellation"]).assign(
        alt_km=lambda d: d["alt_km"].clip(0, 2000),
        inc_deg=lambda d: d["inc_deg"].clip(0, 180),
    )

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    df_in = _load_input()
    if df_in.empty:
        pd.DataFrame(columns=["alt_km","inc_deg","satname","provider","constellation"]).to_csv(RAW, index=False)
        pd.DataFrame(columns=["alt_bin_km","inc_bin_deg","count"]).to_csv(OUT, index=False)
        print("congestion_map: no input, wrote empty outputs.")
        return

    cols = set(df_in.columns)
    if {"inclination_deg","perigee_km"}.issubset(cols):
        pts = _rows_from_elements(df_in)
        source = "elements"
    elif {"line1","line2"}.issubset(cols):
        pts = _rows_from_tle(df_in)
        source = "tle"
    else:
        pts = pd.DataFrame()
        source = "unsupported"

    if pts.empty:
        pts.to_csv(RAW, index=False)
        pd.DataFrame(columns=["alt_bin_km","inc_bin_deg","count"]).to_csv(OUT, index=False)
        print(f"congestion_map: {source} produced 0 rows; wrote empty outputs.")
        return

    # save raw for interactive binning
    pts.to_csv(RAW, index=False)

    # default snapshot bins
    bins = (pts.assign(
                alt_bin_km=lambda d: (d["alt_km"] // 25 * 25).clip(0, 2000),
                inc_bin_deg=lambda d: (d["inc_deg"] // 2 * 2).clip(0, 180),
           ).groupby(["alt_bin_km","inc_bin_deg"]).size().reset_index(name="count")
            .sort_values(["alt_bin_km","inc_bin_deg"]))

    bins.to_csv(OUT, index=False)
    print(f"congestion_map: source={source}, raw={len(pts)}, bins={len(bins)} → {OUT} & {RAW}")

if __name__ == "__main__":
    main()
