import subprocess, sys, pathlib

def run(cmd):
    print(">>", " ".join(cmd)); subprocess.check_call(cmd)

if __name__ == "__main__":
    pathlib.Path("data").mkdir(exist_ok=True, parents=True)
    run([sys.executable, "scripts/fetch_space_weather.py"])
    run([sys.executable, "scripts/forecast_kp.py"])

    run([sys.executable, "scripts/fetch_asteroids.py"])
    run([sys.executable, "scripts/fetch_commodities.py"])
    run([sys.executable, "scripts/compute_asteroid_profit.py"])

    run([sys.executable, "scripts/fetch_tle.py"])
    run([sys.executable, "scripts/conjunctions.py"])

    run([sys.executable, "scripts/fetch_launch_weather.py"])
    run([sys.executable, "scripts/fetch_launches.py"])
    run([sys.executable, "scripts/fetch_launches_history.py"])
    print("All data refreshed.")
