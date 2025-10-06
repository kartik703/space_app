#!/usr/bin/env python3
"""
Prepare YOLO Dataset from Solar Images
Converts raw solar images to YOLO format with annotations
"""

import os
import glob
import shutil
import logging
from pathlib import Path
import json
from typing import List, Dict, Tuple
from cv_pipeline.config import LOCAL_RAW_DIR, YOLO_DATASET_DIR, YOLO_CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODatasetPreparer:
    """Prepares YOLO format dataset from solar images"""
    
    def __init__(self, raw_dir: str = LOCAL_RAW_DIR, output_dir: str = YOLO_DATASET_DIR):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        
        # Create YOLO directory structure
        self.setup_directories()
    
    def setup_directories(self):
        """Create YOLO dataset directory structure"""
        subdirs = [
            "images/train",
            "images/val", 
            "images/test",
            "labels/train",
            "labels/val",
            "labels/test"
        ]
        
        for subdir in subdirs:
            dir_path = os.path.join(self.output_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def find_solar_images(self) -> List[str]:
        """Find all solar images in the raw directory"""
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif"]
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(self.raw_dir, "**", ext)
            image_files.extend(glob.glob(pattern, recursive=True))
        
        logger.info(f"Found {len(image_files)} solar images")
        return image_files
    
    def create_placeholder_annotation(self, image_path: str, output_path: str):
        """Create placeholder YOLO annotation for image"""
        # For now, create placeholder annotations
        # In a real scenario, you'd parse actual solar event data from NOAA/SWPC
        
        with open(output_path, "w") as f:
            # Placeholder: small flare in center of image
            # Format: class_id center_x center_y width height (normalized 0-1)
            f.write("0 0.5 0.5 0.2 0.2\n")  # flare class
            
            # Add occasional additional features
            basename = os.path.basename(image_path).lower()
            if "193" in basename:  # AIA 193 often shows coronal holes
                f.write("2 0.3 0.7 0.15 0.15\n")  # sunspot class
            elif "304" in basename:  # AIA 304 good for prominences/CMEs
                f.write("1 0.8 0.2 0.25 0.3\n")   # CME class
    
    def split_dataset(self, image_files: List[str], 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.2, 
                     test_ratio: float = 0.1) -> Dict[str, List[str]]:
        """Split dataset into train/val/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        total_images = len(image_files)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        
        # Sort for consistent splits
        image_files.sort()
        
        split = {
            "train": image_files[:train_count],
            "val": image_files[train_count:train_count + val_count],
            "test": image_files[train_count + val_count:]
        }
        
        logger.info(f"Dataset split - Train: {len(split['train'])}, "
                   f"Val: {len(split['val'])}, Test: {len(split['test'])}")
        
        return split
    
    def copy_images_and_labels(self, split_data: Dict[str, List[str]]):
        """Copy images and create corresponding label files"""
        
        for split_name, image_files in split_data.items():
            logger.info(f"Processing {split_name} set with {len(image_files)} images")
            
            images_dir = os.path.join(self.output_dir, "images", split_name)
            labels_dir = os.path.join(self.output_dir, "labels", split_name)
            
            for image_path in image_files:
                # Copy image
                image_filename = os.path.basename(image_path)
                dest_image_path = os.path.join(images_dir, image_filename)
                
                try:
                    shutil.copy2(image_path, dest_image_path)
                    
                    # Create corresponding label file
                    label_filename = os.path.splitext(image_filename)[0] + ".txt"
                    label_path = os.path.join(labels_dir, label_filename)
                    
                    self.create_placeholder_annotation(image_path, label_path)
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        yaml_content = f"""# Solar YOLO Dataset Configuration
path: {self.output_dir}
train: images/train
val: images/val
test: images/test

# Classes
nc: {len(YOLO_CLASSES)}  # number of classes
names: {list(YOLO_CLASSES.values())}
"""
        
        yaml_path = os.path.join(self.output_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        
        logger.info(f"Created dataset configuration: {yaml_path}")
        return yaml_path
    
    def prepare_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Main method to prepare the complete YOLO dataset"""
        logger.info("Starting YOLO dataset preparation")
        
        # Find all images
        image_files = self.find_solar_images()
        
        if not image_files:
            logger.warning(f"No images found in {self.raw_dir}")
            logger.info("You may need to run the GCS ingestor first or download sample images")
            return
        
        # Split dataset
        split_data = self.split_dataset(image_files, train_ratio, val_ratio, test_ratio)
        
        # Copy images and create labels
        self.copy_images_and_labels(split_data)
        
        # Create dataset YAML
        yaml_path = self.create_dataset_yaml()
        
        logger.info("YOLO dataset preparation complete!")
        logger.info(f"Dataset location: {self.output_dir}")
        logger.info(f"Configuration file: {yaml_path}")
        
        return yaml_path


def download_sample_images():
    """Download a few sample solar images for testing if no local images exist"""
    import requests
    
    sample_urls = [
        "https://soho.nascom.nasa.gov/data/realtime/eit_195/512/latest.jpg",
        "https://soho.nascom.nasa.gov/data/realtime/eit_284/512/latest.jpg",
        "https://soho.nascom.nasa.gov/data/realtime/eit_304/512/latest.jpg"
    ]
    
    os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
    
    logger.info("Downloading sample solar images...")
    for i, url in enumerate(sample_urls):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filename = f"sample_solar_{i+1:03d}.jpg"
            filepath = os.path.join(LOCAL_RAW_DIR, filename)
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {filename}")
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")


def main():
    """Main function to prepare YOLO dataset"""
    preparer = YOLODatasetPreparer()
    
    # Check if we have any images
    image_files = preparer.find_solar_images()
    
    if not image_files:
        logger.info("No local images found. Downloading sample images...")
        download_sample_images()
    
    # Prepare the dataset
    yaml_path = preparer.prepare_dataset()
    
    if yaml_path:
        logger.info(f"Dataset ready! You can now train YOLO with: {yaml_path}")
        logger.info("Next steps:")
        logger.info("1. python cv_pipeline/train/train_yolo.py")
        logger.info("2. python cv_pipeline/train/train_vit.py")


if __name__ == "__main__":
    main()