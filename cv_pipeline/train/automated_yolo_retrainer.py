#!/usr/bin/env python3
"""
Automated YOLO Model Retraining Pipeline
Automatically retrains YOLO models with performance tracking and comparison.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import json
import os
import shutil
from datetime import datetime
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('automated_retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedYOLORetrainer:
    """Automated YOLO model retraining system"""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.trainer_script = self.base_dir / "cv_pipeline" / "train" / "minimal_yolo_trainer.py"
        self.test_script = self.base_dir / "cv_pipeline" / "train" / "test_trained_model.py"
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.yolo_dataset_dir = self.data_dir / "yolo_dataset"
        
        # Training configurations for different scenarios
        self.training_configs = {
            "quick": {"epochs": 10, "batch_size": 4, "imgsz": 416},
            "standard": {"epochs": 30, "batch_size": 8, "imgsz": 640},
            "extended": {"epochs": 50, "batch_size": 4, "imgsz": 640},
            "production": {"epochs": 100, "batch_size": 2, "imgsz": 640}
        }
        
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Trainer script: {self.trainer_script}")
    
    def validate_dataset(self):
        """Validate that YOLO dataset is ready for training"""
        yaml_file = self.yolo_dataset_dir / "dataset.yaml"
        
        if not yaml_file.exists():
            logger.error("❌ Dataset YAML file not found!")
            return False
        
        # Check train and val directories
        train_dir = self.yolo_dataset_dir / "images" / "train"
        val_dir = self.yolo_dataset_dir / "images" / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            logger.error("❌ Required dataset directories not found!")
            return False
        
        train_images = len(list(train_dir.glob("*.jpg")))
        val_images = len(list(val_dir.glob("*.jpg")))
        
        if train_images == 0 or val_images == 0:
            logger.error("❌ No images found in dataset directories!")
            return False
        
        logger.info(f"✅ Dataset validation passed: {train_images} train, {val_images} val images")
        return True
    
    def backup_previous_model(self):
        """Backup previous model if it exists"""
        current_model_dir = self.models_dir / "minimal_solar"
        
        if current_model_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.models_dir / f"minimal_solar_backup_{timestamp}"
            
            logger.info(f"📦 Backing up previous model to: {backup_dir}")
            shutil.copytree(current_model_dir, backup_dir)
            return backup_dir
        
        return None
    
    def train_model(self, config_name="standard", custom_config=None):
        """Train YOLO model with specified configuration"""
        
        if custom_config:
            config = custom_config
        else:
            config = self.training_configs.get(config_name, self.training_configs["standard"])
        
        logger.info(f"🚀 Starting YOLO training with {config_name} configuration")
        logger.info(f"Config: {config}")
        
        # Build training command
        dataset_yaml = self.yolo_dataset_dir / "dataset.yaml"
        
        cmd = [
            sys.executable,
            str(self.trainer_script),
            "--data", str(dataset_yaml),
            "--epochs", str(config["epochs"]),
            "--batch-size", str(config["batch_size"]),
            "--imgsz", str(config["imgsz"])
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            
            if result.returncode == 0:
                logger.info("✅ Model training completed successfully!")
                logger.info(f"Training output: {result.stdout[-1000:]}")  # Last 1000 chars
                return True
            else:
                logger.error("❌ Model training failed!")
                logger.error(f"STDERR: {result.stderr}")
                logger.error(f"STDOUT: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ Model training timed out!")
            return False
        except Exception as e:
            logger.error(f"💥 Exception during model training: {e}")
            return False
    
    def test_trained_model(self, num_samples=10):
        """Test the newly trained model"""
        model_path = self.models_dir / "minimal_solar" / "weights" / "best.pt"
        
        if not model_path.exists():
            logger.error("❌ Trained model not found!")
            return False
        
        val_images_dir = self.yolo_dataset_dir / "images" / "val"
        
        cmd = [
            sys.executable,
            str(self.test_script),
            "--model", str(model_path),
            "--images", str(val_images_dir),
            "--num-samples", str(num_samples)
        ]
        
        logger.info(f"🧪 Testing trained model with {num_samples} samples")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
            
            if result.returncode == 0:
                logger.info("✅ Model testing completed successfully!")
                logger.info(f"Test output: {result.stdout}")
                return True
            else:
                logger.error("❌ Model testing failed!")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ Model testing timed out!")
            return False
        except Exception as e:
            logger.error(f"💥 Exception during model testing: {e}")
            return False
    
    def save_training_metadata(self, config_name, config, success=True):
        """Save metadata about the training session"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config_name": config_name,
            "config": config,
            "success": success,
            "model_path": str(self.models_dir / "minimal_solar" / "weights" / "best.pt"),
            "dataset_path": str(self.yolo_dataset_dir / "dataset.yaml")
        }
        
        # Count dataset images
        train_dir = self.yolo_dataset_dir / "images" / "train"
        val_dir = self.yolo_dataset_dir / "images" / "val"
        metadata["dataset_stats"] = {
            "train_images": len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0,
            "val_images": len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
        }
        
        # Save metadata
        metadata_file = self.models_dir / "training_metadata.json"
        
        # Load existing metadata if available
        all_metadata = []
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    all_metadata = json.load(f)
            except:
                pass
        
        all_metadata.append(metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        logger.info(f"📝 Training metadata saved to: {metadata_file}")
        
        return metadata
    
    def get_model_info(self):
        """Get information about the current trained model"""
        model_path = self.models_dir / "minimal_solar" / "weights" / "best.pt"
        
        if not model_path.exists():
            return None
        
        info = {
            "model_path": str(model_path),
            "model_size": model_path.stat().st_size,
            "created": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        }
        
        return info

def main():
    parser = argparse.ArgumentParser(description="Automated YOLO Retraining Pipeline")
    parser.add_argument("--config", choices=["quick", "standard", "extended", "production"], 
                      default="standard", help="Training configuration preset")
    parser.add_argument("--epochs", type=int, help="Custom number of epochs")
    parser.add_argument("--batch-size", type=int, help="Custom batch size")
    parser.add_argument("--imgsz", type=int, help="Custom image size")
    parser.add_argument("--test-samples", type=int, default=10, help="Number of samples for testing")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backing up previous model")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory path")
    
    args = parser.parse_args()
    
    logger.info("🤖 Automated YOLO Retraining Pipeline")
    logger.info(f"Training config: {args.config}")
    
    retrainer = AutomatedYOLORetrainer(args.base_dir)
    
    # Build training configuration
    if any([args.epochs, args.batch_size, args.imgsz]):
        # Use custom configuration
        base_config = retrainer.training_configs[args.config]
        custom_config = {
            "epochs": args.epochs or base_config["epochs"],
            "batch_size": args.batch_size or base_config["batch_size"],
            "imgsz": args.imgsz or base_config["imgsz"]
        }
        config_name = "custom"
        config = custom_config
    else:
        config_name = args.config
        config = None
    
    # Validate dataset
    if not retrainer.validate_dataset():
        logger.error("❌ Dataset validation failed!")
        return False
    
    # Backup previous model
    if not args.skip_backup:
        backup_path = retrainer.backup_previous_model()
        if backup_path:
            logger.info(f"📦 Previous model backed up to: {backup_path}")
    
    # Train model
    training_success = retrainer.train_model(config_name, config)
    
    # Save training metadata
    metadata = retrainer.save_training_metadata(config_name, config or retrainer.training_configs[config_name], training_success)
    
    if not training_success:
        logger.error("❌ Training failed!")
        return False
    
    # Test model
    testing_success = retrainer.test_trained_model(args.test_samples)
    
    if not testing_success:
        logger.warning("⚠️ Model testing failed, but training succeeded")
    
    # Get final model info
    model_info = retrainer.get_model_info()
    
    logger.info("🎯 Retraining Complete!")
    logger.info(f"   Training: {'✅ Success' if training_success else '❌ Failed'}")
    logger.info(f"   Testing: {'✅ Success' if testing_success else '❌ Failed'}")
    logger.info(f"   Model size: {model_info['model_size'] / 1024 / 1024:.1f} MB" if model_info else "Model info unavailable")
    logger.info(f"   Dataset: {metadata['dataset_stats']['train_images']} train, {metadata['dataset_stats']['val_images']} val")
    logger.info("✅ New model ready for deployment!")
    
    return training_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)