import requests, pandas as pd
from pathlib import Path

OUT = Path("data/kp_latest.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    r = requests.get(url, headers={"User-Agent":"space-intel-app/1.0"}, timeout=60)
    r.raise_for_status()
    js = r.json()
    cols, rows = js[0], js[1:]
    df = pd.DataFrame(rows, columns=cols)
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
    df["Kp"] = pd.to_numeric(df["Kp"], errors="coerce")
    df = df.dropna(subset=["time_tag","Kp"]).sort_values("time_tag")
    df.to_csv(OUT, index=False)
    print("Saved", OUT)

if __name__ == "__main__":
    main()
