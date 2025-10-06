# Solar CV Pipeline Setup Guide

This guide will help you set up the complete Solar Computer Vision pipeline for detecting and classifying solar events using YOLOv8 and Vision Transformers.

## 📋 Prerequisites

### 1. Google Cloud SDK (CLI)
Download & install: https://cloud.google.com/sdk/docs/install

Authenticate your account:
```bash
gcloud auth login
gcloud auth application-default login
```

Set your project:
```bash
gcloud config set project YOUR_PROJECT_ID
```

### 2. Python Dependencies
The required dependencies are already in `requirements.txt`. Install them:
```bash
pip install -r requirements.txt
```

## 🌞 Step-by-Step Setup

### Step 1. Setup Cloud Resources

Create storage buckets and BigQuery dataset:

```bash
# Create storage buckets
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://solar-raw/
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://solar-yolo/
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://solar-tfrecords/

# Create BigQuery dataset
bq --location=US mk -d space_weather
```

### Step 2. Test Cloud Ingestion

Set your Google Cloud Project ID:
```bash
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
```

Test fetching 1 day of solar data:
```bash
python cv_pipeline/cloud_ingest/gcs_ingestor.py --start 2012-03-04T00:00:00 --end 2012-03-04T23:59:59 --project-id YOUR_PROJECT_ID
```

Expected results:
- Files in `gs://solar-raw/aia/193/...`
- Metadata in BigQuery → `space_weather.solar_events`

Check results:
```bash
gsutil ls gs://solar-raw/aia/193/
bq query --use_legacy_sql=false 'SELECT * FROM `YOUR_PROJECT_ID.space_weather.solar_events` LIMIT 10'
```

### Step 3. Prepare YOLO Dataset

Convert raw images to YOLO format:
```bash
python cv_pipeline/preprocess/prepare_yolo_dataset.py
```

This will:
- Find all solar images in the data directory
- Create YOLO format dataset structure
- Generate placeholder annotations (replace with real NOAA event data)
- Split into train/val/test sets

### Step 4. Train YOLOv8 Model

Train the YOLO model for object detection:
```bash
python cv_pipeline/train/train_yolo.py
```

Optional parameters:
```bash
python cv_pipeline/train/train_yolo.py --epochs 50 --batch 8 --device 0
```

### Step 5. Train Vision Transformer

Train ViT for storm/quiet classification:
```bash
python cv_pipeline/train/train_vit.py
```

Optional parameters:
```bash
python cv_pipeline/train/train_vit.py --epochs 25 --batch 16 --lr 0.0001
```

### Step 6. Run the Streamlit App

Launch the web interface:
```bash
streamlit run app.py
```

Navigate to the Solar CV Detector page to:
- Upload solar images
- Get YOLO detections (flares, CMEs, sunspots, debris)
- Get ViT classification (storm vs quiet)

## 📁 Project Structure

```
cv_pipeline/
├── __init__.py
├── config.py                    # Configuration constants
├── cloud_ingest/
│   ├── __init__.py
│   └── gcs_ingestor.py          # Helioviewer -> GCS ingestion
├── preprocess/
│   ├── __init__.py
│   └── prepare_yolo_dataset.py  # YOLO dataset preparation
└── train/
    ├── __init__.py
    ├── solar.yaml               # YOLO config
    ├── train_yolo.py            # YOLOv8 training
    └── train_vit.py             # Vision Transformer training

data/
├── yolo_dataset/                # YOLO format dataset
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── solar_raw/                   # Raw solar images

models/
├── yolo/                        # Trained YOLO models
└── vit/                         # Trained ViT models
```

## 🔧 Configuration

Key configuration options in `cv_pipeline/config.py`:

```python
# Model sizes
YOLO_MODEL_SIZE = "yolov8s"      # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
VIT_MODEL_NAME = "vit_base_patch16_224"

# Training parameters
YOLO_EPOCHS = 100
VIT_EPOCHS = 50
YOLO_BATCH_SIZE = 16
VIT_BATCH_SIZE = 32
```

## 🚀 Quick Start (No Cloud)

If you want to test locally without Google Cloud:

1. **Get sample images:**
   ```bash
   python cv_pipeline/preprocess/prepare_yolo_dataset.py
   ```
   This will download sample solar images from SOHO.

2. **Train models:**
   ```bash
   python cv_pipeline/train/train_yolo.py --epochs 10
   python cv_pipeline/train/train_vit.py --epochs 5
   ```

3. **Run app:**
   ```bash
   streamlit run app.py
   ```

## 📊 Model Classes

### YOLO Detection Classes:
- **flare**: Solar flares
- **cme**: Coronal mass ejections  
- **sunspot**: Sunspots and active regions
- **debris**: Space debris or artifacts

### ViT Classification Classes:
- **quiet**: Normal solar conditions
- **storm**: Active/stormy solar conditions

## 🔍 Monitoring Training

Both training scripts save models and metrics:

**YOLO:**
- Best model: `models/yolo/solar_yolo_yolov8s_best.pt`
- Tensorboard logs in `runs/detect/`

**ViT:**
- Best model: `models/vit/solar_vit_best.pth`
- Training history: `models/vit/training_history.json`
- Test results: `models/vit/test_results.json`

## 🛠️ Troubleshooting

### Common Issues:

1. **CUDA out of memory:**
   - Reduce batch size: `--batch 8` or `--batch 4`
   - Use smaller model: `--model yolov8n`

2. **No images found:**
   - Run the preprocessing script first
   - Check that images exist in `data/solar_raw/`

3. **Import errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path includes the project directory

4. **Google Cloud authentication:**
   - Run `gcloud auth application-default login`
   - Set `GOOGLE_CLOUD_PROJECT` environment variable

## 📈 Next Steps

1. **Real annotations:** Replace placeholder YOLO labels with actual solar event data from NOAA/SWPC
2. **More data:** Ingest more historical data using the GCS ingestor
3. **Model tuning:** Experiment with different model architectures and hyperparameters
4. **Deployment:** Deploy models using Google Cloud AI Platform or other cloud services
5. **Real-time inference:** Set up streaming pipeline for real-time solar monitoring

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Timm Models](https://github.com/rwightman/pytorch-image-models)
- [Helioviewer API](https://api.helioviewer.org/)
- [NOAA Space Weather](https://www.swpc.noaa.gov/)
- [SDO Mission](https://sdo.gsfc.nasa.gov/)