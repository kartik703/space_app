import requests, pandas as pd
from datetime import date, timedelta
from pathlib import Path

OUT = Path("data/launch_weather.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

LAT, LON = 28.396837, -80.605659  # Cape

def main():
    start = date.today(); end = start + timedelta(days=6)
    r = requests.get("https://api.open-meteo.com/v1/forecast",
                     params={"latitude":LAT,"longitude":LON,"hourly":"windspeed_10m,precipitation",
                             "start_date":start.isoformat(),"end_date":end.isoformat(),"timezone":"UTC"},
                     timeout=60)
    r.raise_for_status()
    js = r.json()
    hourly = pd.DataFrame(js["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"], utc=True)
    daily = (hourly.assign(date=lambda d: d["time"].dt.date)
             .groupby("date")
             .agg(wind_speed_10m=("windspeed_10m","mean"),
                  precip_mm=("precipitation","sum"))
             .reset_index())
    daily.to_csv(OUT, index=False); print("Saved", OUT)

if __name__ == "__main__":
    main()
