# scripts/run_all.py
import subprocess, sys
from datetime import datetime

steps = [
    "scripts/fetch_tle.py",
    "scripts/propagate_tle.py",
    "scripts/congestion_map.py",
    "scripts/fetch_kp.py",              # <-- NEW: Space Weather data
    "scripts/compute_asteroid_profit.py",
    "scripts/launch_success_model.py",
]

def run_step(script):
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] ▶️ Running {script} ...")
    try:
        subprocess.check_call([sys.executable, script])
        print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Finished {script}")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Skipped {script}: {e}")

def main():
    for step in steps:
        run_step(step)
    print("✅ All pipeline steps finished. Data refreshed in /data/")

if __name__ == "__main__":
    main()
