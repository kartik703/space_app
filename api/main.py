# api/main.py
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import pandas as pd

API_KEY = os.getenv("API_KEY", "")  # set in environment/Streamlit secrets
DATA = Path("data")

app = FastAPI(title="Space Intel API", version="0.1.0")

def require_key(request: Request):
    key = request.headers.get("x-api-key", "")
    if not API_KEY or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

def read_csv_safe(name: str) -> pd.DataFrame:
    p = DATA / name
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except:
        return pd.DataFrame()

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/kp_forecast")
def kp_forecast(request: Request):
    require_key(request)
    df = read_csv_safe("kp_forecast.csv")
    return df.to_dict(orient="records")

@app.get("/api/conjunctions")
def conjunctions(request: Request, threshold_km: float | None = None):
    require_key(request)
    df = read_csv_safe("conjunctions.csv")
    if threshold_km and "min_sep_km" in df.columns:
        df = df[df["min_sep_km"] <= threshold_km]
    return df.to_dict(orient="records")

@app.get("/api/anomalies")
def anomalies(request: Request):
    require_key(request)
    df = read_csv_safe("anomalies.csv")
    return df.to_dict(orient="records")

@app.get("/api/launch_go_scores")
def launch_go_scores(request: Request):
    require_key(request)
    df = read_csv_safe("launch_success_scores.csv")
    return df.to_dict(orient="records")

@app.get("/api/asteroids/top")
def asteroids_top(request: Request, n: int = 50):
    require_key(request)
    df = read_csv_safe("asteroids_scored.csv")
    if "profit_index" in df.columns:
        df = df.sort_values("profit_index", ascending=False).head(n)
    return df.to_dict(orient="records")
