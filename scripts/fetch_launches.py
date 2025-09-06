import requests, pandas as pd
from pathlib import Path

OUT = Path("data/launches.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    r = requests.get("https://ll.thespacedevs.com/2.2.0/launch/upcoming/?limit=50",
                     headers={"User-Agent":"space-intel-app/1.0"}, timeout=60)
    r.raise_for_status()
    js = r.json()
    items = []
    for x in js.get("results", []):
        items.append({
            "window_start": x.get("window_start"),
            "name": x.get("name"),
            "provider": (x.get("launch_service_provider") or {}).get("name"),
            "vehicle": (x.get("rocket") or {}).get("configuration", {}).get("full_name"),
            "pad": (x.get("pad") or {}).get("name"),
            "location": (x.get("pad") or {}).get("location", {}).get("name"),
            "mission": (x.get("mission") or {}).get("name"),
        })
    df = pd.DataFrame(items)
    if not df.empty:
        df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    df.to_csv(OUT, index=False); print("Saved", OUT)

if __name__ == "__main__":
    main()
