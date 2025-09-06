import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

IN  = Path("data/kp_latest.csv")
OUT = Path("data/kp_forecast.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(IN, parse_dates=["time_tag"]).sort_values("time_tag")
    s = (df.set_index("time_tag")["Kp"].resample("1h").ffill().asfreq("1h"))
    model = SARIMAX(s, order=(2,1,2), seasonal_order=(1,0,1,24),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    f = res.get_forecast(steps=48)
    pred = f.predicted_mean.clip(lower=0)
    ci = f.conf_int(alpha=0.2)
    out = pd.DataFrame({
        "time_tag": pred.index, "kp_pred": pred.values,
        "kp_lo": ci.iloc[:,0].clip(lower=0).values,
        "kp_hi": ci.iloc[:,1].clip(lower=0).values
    })
    out.to_csv(OUT, index=False)
    print("Saved", OUT)

if __name__ == "__main__":
    main()
