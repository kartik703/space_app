# scripts/fetch_commodities.py
from __future__ import annotations
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

DATA = Path("data")
OUT = DATA / "commodities.csv"

# Practical defaults if no recent price is available
# Units:
#   - pt_usd: USD per troy ounce (XPTUSD)
#   - ni_usd: USD per metric tonne
#   - co_usd: USD per metric tonne
DEFAULTS = {
    "pt_usd": 950.0,
    "ni_usd": 18500.0,
    "co_usd": 32000.0,
}

# Optional environment overrides for demos (set env vars before running)
OVERRIDE = {
    "pt_usd": os.getenv("OVERRIDE_PT_USD"),
    "ni_usd": os.getenv("OVERRIDE_NI_USD"),
    "co_usd": os.getenv("OVERRIDE_CO_USD"),
}

# Candidate tickers to try in order (some can be flaky)
CANDIDATES = {
    "pt_usd": ["XPTUSD=X", "PL=F"],  # spot / futures
    "ni_usd": [],                    # no reliable free Yahoo symbols; rely on last_known/default/override
    "co_usd": [],                    # same
}

def _read_last_known() -> dict:
    if OUT.exists() and OUT.stat().st_size > 0:
        try:
            df = pd.read_csv(OUT)
            if not df.empty:
                row = df.iloc[-1].to_dict()
                return {
                    "pt_usd": pd.to_numeric(row.get("pt_usd", np.nan), errors="coerce"),
                    "ni_usd": pd.to_numeric(row.get("ni_usd", np.nan), errors="coerce"),
                    "co_usd": pd.to_numeric(row.get("co_usd", np.nan), errors="coerce"),
                }
        except Exception:
            pass
    return {"pt_usd": np.nan, "ni_usd": np.nan, "co_usd": np.nan}

def _fetch_yf_last_close(ticker: str) -> float | None:
    try:
        import yfinance as yf
        df = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        close = df["Close"].dropna().iloc[-1]
        return float(close) if np.isfinite(close) else None
    except Exception:
        return None

def _resolve_price(key: str, candidates: list[str], last_known: float | None, default_val: float) -> tuple[float, str]:
    # 0) Manual override wins
    ov = OVERRIDE.get(key)
    if ov is not None:
        try:
            return float(ov), "override"
        except Exception:
            pass
    # 1) Live candidates
    for tkr in candidates:
        px = _fetch_yf_last_close(tkr)
        if px is not None:
            return px, f"yahoo:{tkr}"
    # 2) Last-known
    if last_known is not None and np.isfinite(last_known):
        return float(last_known), "last_known"
    # 3) Default
    return float(default_val), "default"

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    last = _read_last_known()

    pt_val, pt_src = _resolve_price("pt_usd", CANDIDATES["pt_usd"], last.get("pt_usd"), DEFAULTS["pt_usd"])
    ni_val, ni_src = _resolve_price("ni_usd", CANDIDATES["ni_usd"], last.get("ni_usd"), DEFAULTS["ni_usd"])
    co_val, co_src = _resolve_price("co_usd", CANDIDATES["co_usd"], last.get("co_usd"), DEFAULTS["co_usd"])

    now = datetime.now(timezone.utc).isoformat()
    row = pd.DataFrame([{
        "asof": now,
        "pt_usd": round(pt_val, 4),
        "ni_usd": round(ni_val, 2),
        "co_usd": round(co_val, 2),
        "pt_src": pt_src,
        "ni_src": ni_src,
        "co_src": co_src,
    }])

    if OUT.exists() and OUT.stat().st_size > 0:
        try:
            prev = pd.read_csv(OUT)
            df = pd.concat([prev, row], ignore_index=True)
        except Exception:
            df = row
    else:
        df = row

    # Keep only latest 60 rows
    if len(df) > 60:
        df = df.iloc[-60:].reset_index(drop=True)

    df.to_csv(OUT, index=False)
    print(
        f"Commodities → platinum={df.iloc[-1]['pt_usd']} ({pt_src}), "
        f"nickel={df.iloc[-1]['ni_usd']} ({ni_src}), "
        f"cobalt={df.iloc[-1]['co_usd']} ({co_src}) → {OUT}"
    )

if __name__ == "__main__":
    main()
