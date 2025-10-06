#!/usr/bin/env python3
"""
Simple YOLO Training Script
Trains YOLO model without MLflow integration issues.
"""

import os
import sys
from pathlib import Path

# Disable MLflow integration
os.environ['MLFLOW_TRACKING_URI'] = ''
os.environ['MLFLOW_DISABLE_ENV_CREATION'] = 'true'

# Add current directory to Python path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

import torch
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_simple_yolo():
    """Train YOLO model with simple configuration."""
    
    # Configuration
    dataset_yaml = "data/yolo_dataset/dataset.yaml"
    model_name = "yolov8n.pt"  # Nano model for faster training
    epochs = 5
    img_size = 640
    batch_size = 2
    
    logger.info("🚀 Starting simple YOLO training")
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_yaml}")
    logger.info(f"Epochs: {epochs}, Batch: {batch_size}, Image Size: {img_size}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    try:
        # Initialize model
        model = YOLO(model_name)
        
        # Train with minimal configuration
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=50,
            save=True,
            verbose=True,
            device='cpu',  # Use CPU to avoid CUDA issues
            project='models',
            name='solar_storm_detector',
            exist_ok=True,
            # Disable problematic integrations
            plots=False,
            save_period=-1
        )
        
        logger.info("✅ Training completed successfully!")
        
        # Get the trained model path
        model_path = results.save_dir / "weights" / "best.pt"
        logger.info(f"Best model saved at: {model_path}")
        
        # Test the model with a sample prediction
        logger.info("Testing trained model...")
        test_results = model.val()
        logger.info(f"Validation results: mAP50={test_results.box.map50:.3f}")
        
        return str(model_path)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    try:
        model_path = train_simple_yolo()
        print(f"\n🎉 Success! Trained model available at: {model_path}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)