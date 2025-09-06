# 🌌 Space Intelligence Super App

![Banner](docs\Banner.png)

> **AI-powered dashboard for space intelligence** — live space weather, asteroid mining economics, satellite conjunction alerts, launch feasibility, and mission tracking.  
> Built with **Python + Streamlit**, auto-refreshed nightly via **GitHub Actions**.

---

## ✨ Features

### 1. 🌞 Space Weather
- Real-time NOAA **Kp index**  
- 48h AI forecast (SARIMAX)

### 2. 🪨 Asteroid Mining
- NASA JPL SBDB asteroid catalog  
- Δv cost model + commodity pricing (Nickel, Cobalt, Platinum)  
- Profitability index ranking

### 3. 🛰 Conjunctions
- **sgp4 propagation** from CelesTrak TLEs  
- Screens for close approaches (configurable thresholds, default 20 km)  
- Demo: LEO filter + 48h horizon

### 4. 🚀 Launch Window
- Cape Canaveral weather forecast (Open-Meteo API)  
- Target orbit feasibility (inclination + altitude Δv)  
- Scoring: 60% weather + 40% orbital feasibility

### 5. 📡 Space Tracker
- Upcoming launches (Launch Library 2)  
- Historical launch stats (success rates, average delays)  
- Provider reliability leaderboard

---

## 📊 Data Snapshots

All data lives in `/data` and is refreshed nightly by [GitHub Actions](.github/workflows/ingest.yml).

- `kp_latest.csv` – NOAA Kp  
- `kp_forecast.csv` – AI forecast  
- `asteroids.csv` – JPL SBDB  
- `commodities.csv` – market prices  
- `asteroids_scored.csv` – profitability index  
- `tle_small.csv` – CelesTrak TLEs  
- `conjunctions.csv` – close approach screening  
- `launch_weather.csv` – Cape Canaveral forecasts  
- `launches.csv` – upcoming launches  
- `launches_history.csv` – historical launches  

---

## 🖥️ Screenshots

Home dashboard:

![Home](docs\Home.png)

Asteroid Mining:

![Asteroids](docs\mining.png)

Conjunctions:

![Conjunctions](docs\conjuction.png)

---

## ⚡ Quickstart

```bash
git clone https://github.com/kartik703/space_app.git
cd space-intel-app

# create venv
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.\.venv\Scripts\Activate.ps1 # (Windows PowerShell)

# install deps
pip install -r requirements.txt

# fetch data
python scripts/run_all.py

# run app
streamlit run app.py
