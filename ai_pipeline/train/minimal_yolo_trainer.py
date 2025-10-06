#!/usr/bin/env python3
"""
Minimal YOLO Training Without MLflow
A stripped-down version that avoids MLflow completely.
"""

import os
import sys
import logging
from pathlib import Path

# Completely disable MLflow and other integrations
os.environ['MLFLOW_TRACKING_URI'] = ''
os.environ['MLFLOW_DISABLE_ENV_CREATION'] = 'true'
os.environ['DISABLE_MLFLOW'] = 'true'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['CLEARML_DISABLE'] = 'true'

# Set ultralytics settings to avoid integrations
os.environ['ULTRALYTICS_SETTINGS'] = '{"mlflow": false, "wandb": false, "clearml": false}'

import torch
import yaml
from ultralytics.utils import LOGGER

# Disable all loggers except basic ones
LOGGER.disabled = True

def minimal_yolo_train():
    """Train YOLO with absolutely minimal configuration."""
    
    print("🚀 Starting minimal YOLO training...")
    
    try:
        # Import here to avoid early MLflow initialization
        from ultralytics import YOLO
        
        # Configuration
        dataset_yaml = "data/yolo_dataset/dataset.yaml"
        model_path = "yolov8n.pt"
        output_dir = Path("models/minimal_solar")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Dataset: {dataset_yaml}")
        print(f"🏠 Output: {output_dir}")
        print(f"💻 Device: CPU")
        print(f"🔧 Torch version: {torch.__version__}")
        
        # Load model
        model = YOLO(model_path, verbose=False)
        
        # Override trainer settings to disable integrations
        trainer_args = {
            'data': dataset_yaml,
            'epochs': 3,
            'imgsz': 416,  # Smaller for faster training
            'batch': 1,    # Minimal batch
            'device': 'cpu',
            'verbose': False,
            'save': True,
            'project': str(output_dir.parent),
            'name': output_dir.name,
            'exist_ok': True,
            'plots': False,
            'save_period': -1,
            'patience': 10,
            'workers': 1,
            'single_cls': False,
            'optimizer': 'SGD',  # Simple optimizer
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'warmup_epochs': 0,  # No warmup
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'val': True,
            'fraction': 1.0
        }
        
        print("🏃 Starting training...")
        
        # Start training
        results = model.train(**trainer_args)
        
        print("✅ Training completed!")
        
        # Find the best model
        weights_dir = output_dir / "weights"
        best_model = weights_dir / "best.pt"
        last_model = weights_dir / "last.pt"
        
        if best_model.exists():
            print(f"🏆 Best model: {best_model}")
            return str(best_model)
        elif last_model.exists():
            print(f"📄 Last model: {last_model}")
            return str(last_model)
        else:
            print("⚠️ No model weights found")
            return None
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model_path = minimal_yolo_train()
    
    if model_path:
        print(f"\n🎉 Success! Model saved at: {model_path}")
        
        # Test the model
        try:
            print("🧪 Testing model...")
            from ultralytics import YOLO
            test_model = YOLO(model_path, verbose=False)
            print(f"Model loaded successfully: {test_model.model}")
        except Exception as e:
            print(f"⚠️ Model test failed: {e}")
    else:
        print("\n💥 Training failed completely")
        sys.exit(1)