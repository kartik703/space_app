# scripts/compute_asteroid_profit.py
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
AST = DATA / "asteroids.csv"
COM = DATA / "commodities.csv"
OUT = DATA / "asteroids_scored.csv"

# Defaults if commodities file is missing/NaN
DEFAULTS = {"pt_usd": 950.0, "ni_usd": 18500.0, "co_usd": 32000.0}

def _safe_series(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")

def _load_prices() -> dict:
    if COM.exists() and COM.stat().st_size > 0:
        try:
            c = pd.read_csv(COM).iloc[-1]
            pt = pd.to_numeric(c.get("pt_usd"), errors="coerce")
            ni = pd.to_numeric(c.get("ni_usd"), errors="coerce")
            co = pd.to_numeric(c.get("co_usd"), errors="coerce")
            return {
                "pt_usd": float(pt) if np.isfinite(pt) else DEFAULTS["pt_usd"],
                "ni_usd": float(ni) if np.isfinite(ni) else DEFAULTS["ni_usd"],
                "co_usd": float(co) if np.isfinite(co) else DEFAULTS["co_usd"],
            }
        except Exception:
            pass
    return DEFAULTS.copy()

def _delta_v_cost_kms(a_au: float, i_deg: float, e: float) -> float:
    a = float(a_au) if np.isfinite(a_au) else 2.0
    i = abs(float(i_deg)) if np.isfinite(i_deg) else 10.0
    e = abs(float(e)) if np.isfinite(e) else 0.1
    dv = 2.0 + 3.0 * (a - 1.0) + 0.03 * i + 1.5 * e
    return max(0.1, dv)

def main():
    if not (AST.exists() and AST.stat().st_size > 0):
        pd.DataFrame().to_csv(OUT, index=False)
        print(f"[warn] {AST} not found or empty → wrote empty {OUT}")
        return

    df = pd.read_csv(AST)

    # Normalize object name
    if "object" not in df.columns:
        if "full_name" in df.columns:
            df = df.rename(columns={"full_name": "object"})
        else:
            df["object"] = np.arange(len(df))

    # Inputs (safe)
    a_au = _safe_series(df, "a")
    e = _safe_series(df, "e")
    i = _safe_series(df, "i")
    albedo = _safe_series(df, "albedo", default=0.12).fillna(0.12)
    diameter = _safe_series(df, "diameter", default=0.05).fillna(0.05)  # km

    # Rough density proxy from albedo
    dens_gcc = np.where(albedo > 0.2, 2.5, 3.5)

    # Mass estimate
    vol_km3 = (4.0 / 3.0) * math.pi * (diameter / 2.0) ** 3
    mass_tonnes = vol_km3 * 1e12 * (dens_gcc / 1000.0)  # km^3→cm^3; /1000 to tonnes

    # Prices (no network)
    prices = _load_prices()
    pt_usd, ni_usd, co_usd = prices["pt_usd"], prices["ni_usd"], prices["co_usd"]

    # Value mix (demo)
    PT_OZ_TO_KG = 32.1507  # demo scaling factor
    est_value_usd = (
        mass_tonnes * 0.005 * ni_usd +
        mass_tonnes * 0.0005 * co_usd +
        mass_tonnes * 0.000001 * (pt_usd * PT_OZ_TO_KG)
    )

    dv_kms = [_delta_v_cost_kms(a, ii, ee) for a, ii, ee in zip(a_au, i, e)]
    profit_index = est_value_usd / (1.0 + np.array(dv_kms))

    out = pd.DataFrame({
        "object": df["object"],
        "a": a_au,
        "e": e,
        "i": i,
        "albedo": albedo,
        "diameter": diameter,
        "dv_kms": dv_kms,
        "est_value_usd": est_value_usd,
        "profit_index": profit_index,
        "price_pt_usd": pt_usd,
        "price_ni_usd": ni_usd,
        "price_co_usd": co_usd,
    }).replace([np.inf, -np.inf], np.nan).dropna(subset=["profit_index"])

    out.to_csv(OUT, index=False)
    print(f"Scored {len(out)} asteroids → {OUT} (pt={pt_usd} ni={ni_usd} co={co_usd})")

if __name__ == "__main__":
    main()
