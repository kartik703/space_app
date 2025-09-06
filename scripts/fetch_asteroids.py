import requests, pandas as pd
from pathlib import Path

OUT = Path("data/asteroids.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

BASE = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
FIELDS = "full_name,spkid,albedo,diameter,e,i,a"

def parse_columns(js):
    f = js.get("fields", [])
    if not f: return []
    if isinstance(f[0], dict) and "name" in f[0]:
        return [c["name"] for c in f]
    return list(f)

def fetch(limit: int) -> pd.DataFrame:
    r = requests.get(BASE, params={"fields":FIELDS,"limit":limit,"full-prec":"true"},
                     headers={"User-Agent":"space-intel-app/1.0"}, timeout=120)
    r.raise_for_status()
    js = r.json()
    cols = parse_columns(js)
    data = js.get("data", [])
    df = pd.DataFrame(data, columns=cols) if cols else pd.DataFrame(data)
    df.rename(columns={"full_name":"name","i":"i_deg","a":"a_au","diameter":"diameter_km","e":"e"}, inplace=True)
    for c in ["albedo","diameter_km","i_deg","a_au","e"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "albedo" not in df.columns: df["albedo"] = pd.NA
    if "name" not in df.columns: df["name"] = pd.NA
    def guess_class(alb):
        try:
            a = float(alb)
            if a >= 0.30: return "S"
            if a >= 0.15: return "M"
            return "C"
        except Exception:
            return "C"
    df["class"] = df["albedo"].apply(guess_class)
    return df

def main():
    for lim in (5000, 2000, 1000, 500):
        try:
            df = fetch(lim)
            if not df.empty:
                df.to_csv(OUT, index=False); print("Saved", len(df), "→", OUT); return
        except Exception as e:
            print("[warn]", e)
    raise SystemExit("Failed to fetch asteroids")

if __name__ == "__main__":
    main()
