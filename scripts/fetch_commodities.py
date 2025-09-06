import yfinance as yf, pandas as pd, math
from datetime import datetime
from pathlib import Path

OUT = Path("data/commodities.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def last_close(tkr):
    try:
        df = yf.download(tkr, period="10d", interval="1d", progress=False, auto_adjust=True)
        if df.empty: return math.nan
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return math.nan

def main():
    pt = last_close("PL=F")  # Platinum futures (USD/oz)
    ni = last_close("NI=F")  # Nickel futures (may be missing)
    co = math.nan            # Yahoo cobalt unreliable

    if math.isnan(ni):
        jjn = last_close("JJN")  # Nickel ETN proxy
        if not math.isnan(jjn): ni = jjn * 1000
    if math.isnan(co):
        coba = last_close("COBA.L")  # Cobalt proxy ETF (if available)
        if not math.isnan(coba): co = coba * 1000

    if math.isnan(ni): ni = 17000.0
    if math.isnan(co): co = 32000.0
    if math.isnan(pt): pt = 950.0

    pd.DataFrame([{"asof": datetime.utcnow(), "ni_usd": ni, "co_usd": co, "pt_usd": pt}]) \
      .to_csv(OUT, index=False)
    print("Saved", OUT)

if __name__ == "__main__":
    main()
