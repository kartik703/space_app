# scripts/compute_asteroid_leaderboard.py
import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
IN_FILE = os.path.join(DATA, "asteroids_scored.csv")
OUT_FILE = os.path.join(DATA, "asteroids_leaderboard.csv")

def main():
    if not os.path.exists(IN_FILE):
        print(f"[ERROR] Input file missing: {IN_FILE}")
        return

    df = pd.read_csv(IN_FILE)

    # Ensure required columns exist
    required_cols = ["object", "profit_index", "dv_kms", "est_value_usd"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Missing column: {col}")
            return

    # Clean + numeric
    for col in required_cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort and take top 10
    leaderboard = (
        df.dropna(subset=["profit_index"])
          .sort_values("profit_index", ascending=False)
          .head(10)
          .reset_index(drop=True)
    )

    # Select only key fields
    leaderboard = leaderboard[required_cols]

    # Save
    leaderboard.to_csv(OUT_FILE, index=False)
    print(f"Saved Top 10 leaderboard → {OUT_FILE}")
    print(leaderboard)

if __name__ == "__main__":
    main()
