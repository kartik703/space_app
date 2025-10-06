#!/usr/bin/env python3
"""
Automated YOLO Dataset Preprocessing Pipeline
Automatically prepares YOLO datasets from collected solar images.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import json
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('automated_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedYOLOPreprocessor:
    """Automated YOLO dataset preparation system"""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.preparer_script = self.base_dir / "cv_pipeline" / "preprocess" / "local_yolo_preparer.py"
        self.data_dir = self.base_dir / "data"
        self.yolo_dataset_dir = self.data_dir / "yolo_dataset"
        
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Preparer script: {self.preparer_script}")
    
    def count_available_images(self):
        """Count available solar images for processing"""
        image_dir = self.data_dir / "solar_images" / "2012"
        if not image_dir.exists():
            logger.warning(f"Image directory not found: {image_dir}")
            return 0
        
        jpeg_files = list(image_dir.rglob("*.jpg"))
        logger.info(f"📁 Available solar images: {len(jpeg_files)}")
        return len(jpeg_files)
    
    def count_existing_yolo_images(self):
        """Count existing YOLO dataset images"""
        train_dir = self.yolo_dataset_dir / "images" / "train"
        val_dir = self.yolo_dataset_dir / "images" / "val"
        
        train_count = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
        val_count = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
        
        total_count = train_count + val_count
        logger.info(f"📊 Existing YOLO dataset: {total_count} images ({train_count} train, {val_count} val)")
        return total_count, train_count, val_count
    
    def prepare_yolo_dataset(self, max_images=None, force_recreate=False):
        """Prepare YOLO dataset from available images"""
        
        # Count available images
        available_images = self.count_available_images()
        if available_images == 0:
            logger.error("❌ No solar images available for processing!")
            return False
        
        # Check existing dataset
        existing_total, existing_train, existing_val = self.count_existing_yolo_images()
        
        if existing_total > 0 and not force_recreate:
            logger.info(f"📋 Existing YOLO dataset found: {existing_total} images")
            
            # Determine if we should expand the dataset
            if max_images is None:
                max_images = min(available_images, available_images)  # Use all available
            
            if max_images <= existing_total:
                logger.info(f"✅ Existing dataset ({existing_total}) is sufficient for target ({max_images})")
                return True
            else:
                logger.info(f"📈 Expanding dataset from {existing_total} to {max_images} images")
        
        # Determine max images to use
        if max_images is None:
            max_images = min(available_images, 200)  # Default cap at 200 images
        
        max_images = min(max_images, available_images)
        
        logger.info(f"🚀 Preparing YOLO dataset with {max_images} images")
        
        # Build command
        cmd = [
            sys.executable,
            str(self.preparer_script),
            "--max-images", str(max_images)
        ]
        
        if force_recreate:
            cmd.append("--force-recreate")
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                logger.info("✅ YOLO dataset preparation completed successfully!")
                
                # Count final dataset
                final_total, final_train, final_val = self.count_existing_yolo_images()
                logger.info(f"📊 Final dataset: {final_total} images ({final_train} train, {final_val} val)")
                
                return True
            else:
                logger.error("❌ YOLO dataset preparation failed!")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ YOLO dataset preparation timed out!")
            return False
        except Exception as e:
            logger.error(f"💥 Exception during YOLO dataset preparation: {e}")
            return False
    
    def validate_dataset(self):
        """Validate the prepared YOLO dataset"""
        yaml_file = self.yolo_dataset_dir / "dataset.yaml"
        
        if not yaml_file.exists():
            logger.error("❌ Dataset YAML file not found!")
            return False
        
        # Check train and val directories
        train_dir = self.yolo_dataset_dir / "images" / "train"
        val_dir = self.yolo_dataset_dir / "images" / "val"
        train_labels_dir = self.yolo_dataset_dir / "labels" / "train"
        val_labels_dir = self.yolo_dataset_dir / "labels" / "val"
        
        if not all([train_dir.exists(), val_dir.exists(), train_labels_dir.exists(), val_labels_dir.exists()]):
            logger.error("❌ Required dataset directories not found!")
            return False
        
        # Count images and labels
        train_images = len(list(train_dir.glob("*.jpg")))
        val_images = len(list(val_dir.glob("*.jpg")))
        train_labels = len(list(train_labels_dir.glob("*.txt")))
        val_labels = len(list(val_labels_dir.glob("*.txt")))
        
        logger.info(f"📊 Dataset validation:")
        logger.info(f"   Train: {train_images} images, {train_labels} labels")
        logger.info(f"   Val: {val_images} images, {val_labels} labels")
        
        if train_images > 0 and val_images > 0:
            logger.info("✅ Dataset validation passed!")
            return True
        else:
            logger.error("❌ Dataset validation failed - insufficient images!")
            return False
    
    def get_dataset_summary(self):
        """Get summary of the prepared dataset"""
        summary_file = self.yolo_dataset_dir / "dataset_summary.json"
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                logger.info(f"📋 Dataset Summary: {summary}")
                return summary
            except Exception as e:
                logger.warning(f"Could not read dataset summary: {e}")
        
        # Manual count if summary not available
        final_total, final_train, final_val = self.count_existing_yolo_images()
        
        summary = {
            "total_images": final_total,
            "train_images": final_train,
            "val_images": final_val,
            "created": datetime.now().isoformat(),
            "status": "ready"
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Automated YOLO Preprocessing Pipeline")
    parser.add_argument("--max-images", type=int, default=None,
                      help="Maximum images to include in dataset (default: use all available)")
    parser.add_argument("--force-recreate", action="store_true",
                      help="Force recreation of existing dataset")
    parser.add_argument("--base-dir", type=str, default=None,
                      help="Base directory path")
    
    args = parser.parse_args()
    
    logger.info("🔧 Automated YOLO Preprocessing Pipeline")
    logger.info(f"Max images: {args.max_images or 'all available'}")
    logger.info(f"Force recreate: {args.force_recreate}")
    
    preprocessor = AutomatedYOLOPreprocessor(args.base_dir)
    
    # Check available images
    available = preprocessor.count_available_images()
    if available == 0:
        logger.error("❌ No solar images available for processing!")
        return False
    
    # Prepare dataset
    success = preprocessor.prepare_yolo_dataset(args.max_images, args.force_recreate)
    
    if not success:
        logger.error("❌ Dataset preparation failed!")
        return False
    
    # Validate dataset
    if not preprocessor.validate_dataset():
        logger.error("❌ Dataset validation failed!")
        return False
    
    # Get summary
    summary = preprocessor.get_dataset_summary()
    
    logger.info("🎯 Preprocessing Complete!")
    logger.info(f"   Dataset ready: {summary.get('total_images', 0)} images")
    logger.info(f"   Training split: {summary.get('train_images', 0)} images")
    logger.info(f"   Validation split: {summary.get('val_images', 0)} images")
    logger.info("✅ Ready for model training!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)