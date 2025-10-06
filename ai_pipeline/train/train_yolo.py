#!/usr/bin/env python3
"""
Train YOLOv8 model for solar event detection
"""

import os
import logging
from pathlib import Path
from ultralytics import YOLO
import torch
from cv_pipeline.config import (YOLO_DATASET_DIR, YOLO_MODEL_SIZE, YOLO_IMAGE_SIZE, 
                                YOLO_EPOCHS, YOLO_BATCH_SIZE, YOLO_CLASSES)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolarYOLOTrainer:
    """Train YOLO model for solar event detection"""
    
    def __init__(self, model_size: str = YOLO_MODEL_SIZE):
        self.model_size = model_size
        self.model = None
        self.dataset_yaml = os.path.join(YOLO_DATASET_DIR, "dataset.yaml")
        
        # Create output directory for models
        self.output_dir = os.path.join("models", "yolo")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized YOLO trainer with model size: {model_size}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    def check_dataset(self) -> bool:
        """Check if dataset is ready for training"""
        if not os.path.exists(self.dataset_yaml):
            logger.error(f"Dataset configuration not found: {self.dataset_yaml}")
            logger.error("Please run: python cv_pipeline/preprocess/prepare_yolo_dataset.py")
            return False
        
        # Check if train images exist
        train_images_dir = os.path.join(YOLO_DATASET_DIR, "images", "train")
        if not os.path.exists(train_images_dir) or not os.listdir(train_images_dir):
            logger.error(f"No training images found in: {train_images_dir}")
            return False
        
        # Check if train labels exist
        train_labels_dir = os.path.join(YOLO_DATASET_DIR, "labels", "train")
        if not os.path.exists(train_labels_dir) or not os.listdir(train_labels_dir):
            logger.error(f"No training labels found in: {train_labels_dir}")
            return False
        
        logger.info("Dataset validation passed")
        return True
    
    def load_model(self, pretrained: bool = True):
        """Load YOLO model"""
        if pretrained:
            # Load pretrained model
            model_name = f"{self.model_size}.pt"
            logger.info(f"Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)
        else:
            # Load architecture only
            model_name = f"{self.model_size}.yaml"
            logger.info(f"Loading model architecture: {model_name}")
            self.model = YOLO(model_name)
        
        logger.info(f"Model loaded successfully")
    
    def train(self, epochs: int = YOLO_EPOCHS, imgsz: int = YOLO_IMAGE_SIZE, 
              batch: int = YOLO_BATCH_SIZE, device: str = "auto", 
              workers: int = 8, patience: int = 50):
        """Train the YOLO model"""
        
        if not self.check_dataset():
            return None
        
        if self.model is None:
            self.load_model(pretrained=True)
        
        logger.info("Starting YOLO training...")
        logger.info(f"Epochs: {epochs}, Image size: {imgsz}, Batch size: {batch}")
        logger.info(f"Dataset: {self.dataset_yaml}")
        
        # Training arguments
        train_args = {
            "data": self.dataset_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": workers,
            "patience": patience,
            "save": True,
            "save_period": 10,  # Save checkpoint every 10 epochs
            "cache": False,     # Don't cache images (saves memory)
            "rect": False,      # Rectangular training
            "cos_lr": True,     # Cosine learning rate scheduler
            "close_mosaic": 10, # Close mosaic augmentation for last N epochs
            "resume": False,    # Resume from last checkpoint
            "amp": True,        # Automatic Mixed Precision
            "fraction": 1.0,    # Dataset fraction to use
            "profile": False,   # Profile ONNX and TensorRT speeds
            "freeze": None,     # Freeze layers: backbone=10, first3=0,1,2
            "multi_scale": False, # Multi-scale training ±50% image size
            "overlap_mask": True, # Masks should overlap during training
            "mask_ratio": 4,    # Mask downsample ratio
            "dropout": 0.0,     # Use dropout regularization
            "val": True,        # Validate/test during training
            "project": None,    # Custom project name (disable default)
        }
        
        try:
            # Train the model
            results = self.model.train(**train_args)
            
            # Save the trained model
            model_path = os.path.join(self.output_dir, f"solar_yolo_{self.model_size}_best.pt")
            self.model.save(model_path)
            logger.info(f"Model saved to: {model_path}")
            
            logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def validate(self, model_path: str = None):
        """Validate the trained model"""
        if model_path:
            self.model = YOLO(model_path)
        elif self.model is None:
            logger.error("No model loaded for validation")
            return None
        
        logger.info("Running validation...")
        try:
            results = self.model.val(data=self.dataset_yaml)
            logger.info("Validation completed")
            return results
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return None
    
    def export_model(self, model_path: str = None, formats: list = None):
        """Export model to different formats"""
        if formats is None:
            formats = ["onnx", "torchscript"]
        
        if model_path:
            self.model = YOLO(model_path)
        elif self.model is None:
            logger.error("No model loaded for export")
            return
        
        logger.info(f"Exporting model to formats: {formats}")
        
        for format_name in formats:
            try:
                export_path = self.model.export(format=format_name, imgsz=YOLO_IMAGE_SIZE)
                logger.info(f"Exported {format_name}: {export_path}")
            except Exception as e:
                logger.error(f"Export to {format_name} failed: {e}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO model for solar event detection")
    parser.add_argument("--model", default=YOLO_MODEL_SIZE, help=f"Model size (default: {YOLO_MODEL_SIZE})")
    parser.add_argument("--epochs", type=int, default=YOLO_EPOCHS, help=f"Number of epochs (default: {YOLO_EPOCHS})")
    parser.add_argument("--batch", type=int, default=YOLO_BATCH_SIZE, help=f"Batch size (default: {YOLO_BATCH_SIZE})")
    parser.add_argument("--imgsz", type=int, default=YOLO_IMAGE_SIZE, help=f"Image size (default: {YOLO_IMAGE_SIZE})")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, 0, 1, etc.)")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--validate", action="store_true", help="Run validation after training")
    parser.add_argument("--export", action="store_true", help="Export model after training")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SolarYOLOTrainer(model_size=args.model)
    
    # Train model
    results = trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience
    )
    
    if results is None:
        logger.error("Training failed")
        return 1
    
    # Validate if requested
    if args.validate:
        trainer.validate()
    
    # Export if requested
    if args.export:
        trainer.export_model()
    
    logger.info("YOLO training pipeline completed!")
    return 0


if __name__ == "__main__":
    exit(main())