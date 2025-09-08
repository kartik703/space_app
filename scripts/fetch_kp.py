# scripts/fetch_kp.py
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA = Path("data")
DATA.mkdir(exist_ok=True)

LATEST = DATA / "kp_latest.csv"
FORECAST = DATA / "kp_forecast.csv"

def fetch_noaa_latest():
    url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js)
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    df["Kp"] = pd.to_numeric(df["kp_index"], errors="coerce")
    return df

def fetch_noaa_forecast():
    url = "https://services.swpc.noaa.gov/json/planetary_k_index_forecast.json"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js)

    # Handle possible key names
    if "forecastTime" in df.columns:
        df.rename(columns={"forecastTime": "time"}, inplace=True)
    elif "validtime" in df.columns:
        df.rename(columns={"validtime": "time"}, inplace=True)

    if "kpIndex" in df.columns:
        df.rename(columns={"kpIndex": "forecast"}, inplace=True)
    elif "kp" in df.columns:
        df.rename(columns={"kp": "forecast"}, inplace=True)

    # Standardize formats
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["forecast"] = pd.to_numeric(df["forecast"], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["time", "forecast"])
    return df

def main():
    try:
        df_latest = fetch_noaa_latest()
        if not df_latest.empty:
            df_latest.to_csv(LATEST, index=False)
            print(f"✅ Saved latest Kp → {LATEST}")
    except Exception as e:
        print(f"⚠️ Latest Kp fetch failed: {e}")

    try:
        df_fc = fetch_noaa_forecast()
        if not df_fc.empty:
            df_fc.to_csv(FORECAST, index=False)
            print(f"✅ Saved forecast (48h) → {FORECAST}")
    except Exception as e:
        print(f"⚠️ Forecast fetch failed: {e}")

if __name__ == "__main__":
    main()
