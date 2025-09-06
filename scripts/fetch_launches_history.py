import requests, pandas as pd
from pathlib import Path

OUT = Path("data/launches_history.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    r = requests.get("https://ll.thespacedevs.com/2.2.0/launch/previous/?limit=100",
                     headers={"User-Agent":"space-intel-app/1.0"}, timeout=60)
    r.raise_for_status()
    js = r.json()
    rows = []
    for x in js.get("results", []):
        rows.append({
            "provider": (x.get("launch_service_provider") or {}).get("name"),
            "status": (x.get("status") or {}).get("name"),
            "window_start": x.get("window_start"),
            "net": x.get("net"),
            "name": x.get("name")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
        df["net"] = pd.to_datetime(df["net"], utc=True, errors="coerce")
        df["delay_hours"] = (df["net"] - df["window_start"]).dt.total_seconds()/3600
    df.to_csv(OUT, index=False); print("Saved", OUT)

if __name__ == "__main__":
    main()
