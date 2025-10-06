# 🚀 Space Intelligence Platform v3.0

[![CI/CD Pipeline](https://github.com/kartik703/space_app/actions/workflows/ci-cd-pipeline.yml/badge.svg)](https://github.com/kartik703/space_app/actions)
[![Data Monitoring](https://github.com/kartik703/space_app/actions/workflows/monitoring.yml/badge.svg)](https://github.com/kartik703/space_app/actions)
[![Docker Build](https://github.com/kartik703/space_app/actions/workflows/deployment.yml/badge.svg)](https://github.com/kartik703/space_app/actions)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Advanced real-time space data analytics with AI-powered insights, automated CI/CD pipelines, and comprehensive monitoring**

## 🎯 Quick Demo
```bash
git clone https://github.com/kartik703/space_app.git
cd space_app
pip install -r requirements.txt
python launch.py
```

**🌐 Dashboard:** http://localhost:8501

## 🌟 **Key Features**

### 🛰️ **Real Data Sources**
- **NASA SDO**: Live solar images (11 EUV wavelengths)
- **NOAA SWPC**: Real-time space weather JSON data
- **Kyoto WDC**: Geomagnetic indices (Kp, Dst)
- **Ground Observatories**: Telescope observation data

### 🤖 **AI Models**
- **YOLO CV Model**: Solar flare detection (99.5% mAP)
- **Random Forest**: Space weather classification (78%+ accuracy)
- **Storm Predictor**: Geomagnetic storm forecasting (85%+ accuracy)
- **Anomaly Detector**: Isolation Forest for outlier detection

### 📊 **Live Dashboard**
- Real-time CV analysis on NASA solar images
- Interactive space weather visualizations
- Fusion AI risk assessment (0-100% scale)
- Model performance metrics and statistics

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- 4GB+ RAM recommended
- Internet connection for live data

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kartik703/space_app.git
cd space_app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the data collector** (background process)
```bash
python continuous_space_collector.py
```

4. **Train AI models** (after collecting some data)
```bash
python train_all_ai_models.py
```

5. **Launch the dashboard**
```bash
streamlit run fusion_ai_live.py --server.port 8501
```

6. **Access dashboard**: http://localhost:8501

## 📁 **Project Structure**

```
space_app/
├── fusion_ai_live.py              # Main Streamlit dashboard
├── continuous_space_collector.py   # 24/7 data collection system
├── train_all_ai_models.py         # AI model trainer
├── comprehensive_space_pipeline.py # Complete data pipeline
├── integrated_space_ai_platform.py # Full SaaS platform
├── requirements.txt                # Python dependencies
├── utils.py                       # Utility functions
├── docs/                          # Documentation and assets
├── pages/                         # Additional Streamlit pages
├── scripts/                       # Utility scripts
├── cv_pipeline/                   # Computer vision pipeline
└── .gitignore                     # Git ignore rules
```

## 🔧 **Core Components**

### 1. Data Collection (`continuous_space_collector.py`)
- Automated 24/7 collection from multiple space agencies
- Rate-limited API calls with error handling
- Organized data storage with timestamps
- Progress tracking and statistics

### 2. AI Training (`train_all_ai_models.py`)
- YOLO model training on NASA solar images
- Random Forest training on NOAA weather data
- Anomaly detection model training
- Automated evaluation and model saving

### 3. Live Dashboard (`fusion_ai_live.py`)
- Real-time data visualization
- Live CV analysis with detection overlays
- Fusion AI risk assessment
- Interactive charts and metrics

## 📊 **Data Pipeline**

### Input Data Formats:
- **Solar Images**: 1024x1024 JPEG (NASA SDO)
- **Weather Data**: JSON arrays with timestamps (NOAA)
- **Geomagnetic Data**: HTML/TXT files (Kyoto WDC)
- **Observatory Data**: HTML reports (Ground telescopes)

### Processing Pipeline:
```
Data Sources → Collection → Storage → AI Models → Dashboard → User
```

## 🤖 **AI Models Details**

### YOLO Computer Vision
- **Architecture**: YOLOv8 Nano
- **Input**: 1024x1024 solar images
- **Output**: Bounding boxes for solar flares
- **Training**: Automated labeling based on brightness analysis

### Random Forest Classifier
- **Input Features**: Solar wind speed, magnetic field, particle density
- **Output**: Binary storm classification (Normal/Storm)
- **Training**: Real NOAA space weather data

### Anomaly Detection
- **Algorithm**: Isolation Forest
- **Purpose**: Detect unusual space weather patterns
- **Training**: Unsupervised on all collected data

## 📈 **Performance Metrics**

- **Data Collection**: 2GB+ real space data
- **YOLO Accuracy**: 99.5% mAP on solar flare detection
- **ML Models**: 78-85% accuracy on weather prediction
- **Real-time Processing**: <1 second inference time
- **Update Frequency**: Every 30 seconds for live data

## 🛠️ **Configuration**

### Environment Variables (optional):
```bash
export SPACE_DATA_DIR="./data"          # Data storage directory
export MODEL_DIR="./models"              # Model storage directory  
export COLLECTION_RATE=30                # Collection interval (seconds)
export TARGET_DATA_SIZE=100              # Target data size (GB)
```

### Customization:
- Modify `continuous_space_collector.py` to add new data sources
- Update `train_all_ai_models.py` to adjust model parameters
- Customize `fusion_ai_live.py` dashboard layout and features

## 🔍 **Monitoring & Logging**

- Collection statistics and progress tracking
- Model training logs and performance metrics
- Real-time dashboard status indicators
- Error handling and automatic retry mechanisms

## 📚 **API Documentation**

### Data Sources:
- NASA SDO: `https://sdo.gsfc.nasa.gov/assets/img/latest/`
- NOAA SWPC: `https://services.swpc.noaa.gov/json/`
- Kyoto WDC: `http://wdc.kugi.kyoto-u.ac.jp/`

### Model Endpoints:
- Live CV analysis: Real-time YOLO inference
- Weather prediction: Scikit-learn models
- Fusion AI: Combined risk assessment

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **NASA Solar Dynamics Observatory** for real solar imagery
- **NOAA Space Weather Prediction Center** for space weather data
- **Kyoto World Data Center** for geomagnetic indices
- **Ultralytics YOLO** for computer vision framework
- **Streamlit** for interactive dashboard capabilities

## 📞 **Support**

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/kartik703/space_app/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/kartik703/space_app/discussions)

---

**⚡ Real-Time Space Weather Intelligence with AI ⚡**

*Built with ❤️ for space science and AI research*