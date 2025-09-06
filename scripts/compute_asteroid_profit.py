import pandas as pd, numpy as np
from pathlib import Path

AST = Path("data/asteroids.csv")
COM = Path("data/commodities.csv")
OUT = Path("data/asteroids_scored.csv")

def delta_v_proxy(row):
    i = np.nan_to_num(row.get("i_deg", 0.0))
    e = np.nan_to_num(row.get("e", 0.0))
    a = np.nan_to_num(row.get("a_au", 1.0))
    plane = 2.0 * np.sin(np.deg2rad(abs(i))/2)
    a_term = 2.0 * min(abs(a-1.0), 1.0)
    e_term = 1.5 * min(e, 0.6)
    return 6.0 + plane + a_term + e_term

def comp_mult(c): return {"M":1.0, "S":0.6, "C":0.4}.get(str(c), 0.5)

def main():
    df = pd.read_csv(AST)
    prices = pd.read_csv(COM).iloc[-1].to_dict()

    df["delta_v_kms"] = df.apply(delta_v_proxy, axis=1)
    df["comp_mult"] = df["class"].map(comp_mult).fillna(0.5)

    dkm = df["diameter_km"].fillna(0.2)
    r_m = dkm * 500
    vol = (4/3)*np.pi*(r_m**3)
    density = 3500
    recovery = 1e-7 * df["comp_mult"] * density * vol

    weights = {"pt_usd":0.4, "ni_usd":0.3, "co_usd":0.3}
    vals, wts = [], []
    for k,w in weights.items():
        v = prices.get(k, np.nan)
        if v==v:
            vals.append(w*float(v)); wts.append(w)
    basket = sum(vals)/max(sum(wts), 1e-9)

    revenue = recovery * basket
    cost = 60_000_000 * (1 + (df["delta_v_kms"]/6.0)**2)

    df["revenue_usd"] = revenue
    df["mission_cost_usd"] = cost
    df["profit_index"] = (revenue - cost) / (cost + 1e-9)

    df.sort_values("profit_index", ascending=False).to_csv(OUT, index=False)
    print("Saved", OUT)

if __name__ == "__main__":
    main()
