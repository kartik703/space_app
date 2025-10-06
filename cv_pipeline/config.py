"""
Configuration constants for the Solar CV Pipeline
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Local data directories
LOCAL_RAW_DIR = os.path.join(DATA_DIR, "solar_raw")
YOLO_DATASET_DIR = os.path.join(DATA_DIR, "yolo_dataset")
VIT_DATASET_DIR = os.path.join(DATA_DIR, "vit_dataset")

# GCS Configuration
GCS_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
GCS_RAW_BUCKET = "solar-raw"
GCS_YOLO_BUCKET = "solar-yolo"
GCS_TFRECORDS_BUCKET = "solar-tfrecords"

# BigQuery Configuration
BQ_DATASET = "space_weather"
BQ_SOLAR_EVENTS_TABLE = "solar_events"

# Model Configuration
YOLO_MODEL_SIZE = "yolov8s"  # small model for faster training
YOLO_IMAGE_SIZE = 640
YOLO_EPOCHS = 100
YOLO_BATCH_SIZE = 16

# ViT Configuration
VIT_MODEL_NAME = "vit_base_patch16_224"
VIT_IMAGE_SIZE = 224
VIT_EPOCHS = 50
VIT_BATCH_SIZE = 32
VIT_LEARNING_RATE = 1e-4

# Solar data configuration
INSTRUMENTS = ["AIA", "HMI"]
AIA_WAVELENGTHS = [94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500]
HMI_WAVELENGTHS = [6173]

# Class labels for detection
YOLO_CLASSES = {
    0: "flare",
    1: "cme", 
    2: "sunspot",
    3: "debris"
}

# Class labels for classification
VIT_CLASSES = {
    0: "quiet",
    1: "storm"
}

# Create directories if they don't exist
os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
os.makedirs(YOLO_DATASET_DIR, exist_ok=True)
os.makedirs(VIT_DATASET_DIR, exist_ok=True)