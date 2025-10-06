#!/usr/bin/env python3
"""
Local YOLO Dataset Preparation
Creates YOLO training datasets from locally downloaded solar images.
Prepares datasets for both solar storm detection and normal solar activity.
"""

import argparse
import sqlite3
import os
import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class LocalYOLODatasetPreparer:
    """Local YOLO dataset preparation from downloaded solar images."""
    
    def __init__(self, 
                 solar_images_dir: str = "data/solar_images",
                 output_dir: str = "data/yolo_dataset"):
        self.solar_images_dir = Path(solar_images_dir)
        self.output_dir = Path(output_dir)
        self.metadata_db = self.solar_images_dir / "metadata.db"
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        for split in ['train', 'val']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Load storm events for labeling
        self.storm_events = self.load_storm_events()
        
    def load_storm_events(self) -> List[Dict]:
        """Load storm events from the storm database."""
        storm_db_path = Path("solar_storm_data.db")
        if not storm_db_path.exists():
            logger.warning(f"Storm database not found at {storm_db_path}")
            return []
        
        try:
            with sqlite3.connect(storm_db_path) as conn:
                cursor = conn.execute("""
                    SELECT event_id, start_time, peak_time, end_time, 
                           class_type, class_magnitude, source_region,
                           description
                    FROM storm_events 
                    WHERE peak_time IS NOT NULL
                    ORDER BY start_time
                """)
                
                events = []
                for row in cursor.fetchall():
                    events.append({
                        'event_id': row[0],
                        'start_time': datetime.fromisoformat(row[1]) if row[1] else None,
                        'peak_time': datetime.fromisoformat(row[2]) if row[2] else None,
                        'end_time': datetime.fromisoformat(row[3]) if row[3] else None,
                        'class_type': row[4],
                        'class_magnitude': row[5],
                        'source_region': row[6],
                        'description': row[7]
                    })
                
                logger.info(f"Loaded {len(events)} storm events")
                return events
                
        except Exception as e:
            logger.error(f"Error loading storm events: {e}")
            return []
    
    def get_available_images(self) -> List[Dict]:
        """Get list of available images from metadata database."""
        if not self.metadata_db.exists():
            logger.error(f"Metadata database not found: {self.metadata_db}")
            return []
        
        try:
            with sqlite3.connect(self.metadata_db) as conn:
                cursor = conn.execute("""
                    SELECT image_id, observation_time, wavelength, 
                           year, month, day, local_path, file_size
                    FROM solar_images 
                    WHERE status = 'active'
                    ORDER BY observation_time
                """)
                
                images = []
                for row in cursor.fetchall():
                    image_path = Path(row[6])
                    if image_path.exists():
                        images.append({
                            'image_id': row[0],
                            'observation_time': datetime.fromisoformat(row[1]),
                            'wavelength': row[2],
                            'year': row[3],
                            'month': row[4],
                            'day': row[5],
                            'local_path': image_path,
                            'file_size': row[7]
                        })
                
                logger.info(f"Found {len(images)} available images")
                return images
                
        except Exception as e:
            logger.error(f"Error loading image metadata: {e}")
            return []
    
    def classify_image(self, image_time: datetime) -> Tuple[str, Optional[Dict]]:
        """Classify an image as storm, pre_storm, or normal based on timing."""
        
        # Check if image is during or near a storm event
        for event in self.storm_events:
            if not event['peak_time']:
                continue
            
            # Calculate time differences
            time_to_peak = (event['peak_time'] - image_time).total_seconds() / 3600  # hours
            
            # During storm (±2 hours from peak)
            if abs(time_to_peak) <= 2:
                return 'storm', event
            
            # Pre-storm (2-12 hours before peak)
            elif 2 < time_to_peak <= 12:
                return 'pre_storm', event
        
        # If no storm association, it's normal
        return 'normal', None
    
    def generate_yolo_labels(self, image_path: Path, classification: str, event: Optional[Dict] = None) -> List[str]:
        """Generate YOLO format labels for an image."""
        labels = []
        
        # Read image to get dimensions
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return []
            
            height, width, _ = img.shape
            
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            return []
        
        if classification == 'storm':
            # For storm images, create labels based on typical active region locations
            # This is a simplified approach - in practice, you'd use real storm location data
            
            # Create multiple storm region annotations
            num_regions = random.randint(1, 3)  # 1-3 active regions
            
            for i in range(num_regions):
                # Random active region position (more likely in center)
                center_x = random.uniform(0.3, 0.7)
                center_y = random.uniform(0.3, 0.7)
                
                # Size based on storm magnitude if available
                if event and event.get('class_magnitude'):
                    try:
                        magnitude = float(event['class_magnitude'])
                        base_size = min(0.1 + magnitude * 0.05, 0.3)  # Scale with magnitude
                    except:
                        base_size = 0.15
                else:
                    base_size = 0.15
                
                box_w = random.uniform(base_size * 0.8, base_size * 1.2)
                box_h = random.uniform(base_size * 0.8, base_size * 1.2)
                
                # Ensure bounding box is within image
                center_x = max(box_w/2, min(1 - box_w/2, center_x))
                center_y = max(box_h/2, min(1 - box_h/2, center_y))
                
                # YOLO format: class_id center_x center_y width height (normalized)
                # Class 0 = active_region/storm
                labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
        
        elif classification == 'pre_storm':
            # For pre-storm images, smaller/fewer regions
            if random.random() < 0.7:  # 70% chance of having an emerging active region
                center_x = random.uniform(0.3, 0.7)
                center_y = random.uniform(0.3, 0.7)
                
                # Smaller regions for pre-storm
                box_w = random.uniform(0.05, 0.12)
                box_h = random.uniform(0.05, 0.12)
                
                # Ensure bounding box is within image
                center_x = max(box_w/2, min(1 - box_w/2, center_x))
                center_y = max(box_h/2, min(1 - box_h/2, center_y))
                
                # Class 1 = emerging_region/pre_storm  
                labels.append(f"1 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
        
        # For 'normal' classification, return empty list (no objects to detect)
        
        return labels
    
    def copy_and_label_image(self, image_info: Dict, split: str, index: int) -> bool:
        """Copy image to dataset and create corresponding label file."""
        try:
            # Generate new filename
            new_filename = f"solar_{image_info['wavelength']}_{index:06d}.jpg"
            
            # Copy image
            src_path = image_info['local_path']
            dst_path = self.images_dir / split / new_filename
            shutil.copy2(src_path, dst_path)
            
            # Classify image and generate labels
            classification, event = self.classify_image(image_info['observation_time'])
            labels = self.generate_yolo_labels(src_path, classification, event)
            
            # Write label file
            label_path = self.labels_dir / split / f"solar_{image_info['wavelength']}_{index:06d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
            
            logger.debug(f"Processed {new_filename}: {classification} ({len(labels)} objects)")
            return True
            
        except Exception as e:
            logger.error(f"Error processing image {image_info['image_id']}: {e}")
            return False
    
    def create_dataset_yaml(self, num_classes: int = 2):
        """Create YOLO dataset configuration file."""
        yaml_content = f"""# Solar Storm Detection Dataset
train: {self.images_dir}/train
val: {self.images_dir}/val

nc: {num_classes}  # number of classes
names: ['active_region', 'emerging_region']  # class names

# Dataset info
description: "Solar storm detection dataset from SDO/AIA observations"
license: "Research use only"
created: "{datetime.now().isoformat()}"
"""
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Created dataset YAML: {yaml_path}")
    
    def create_dataset(self, train_ratio: float = 0.8, max_images: Optional[int] = None):
        """Create YOLO dataset from available images."""
        logger.info("Starting YOLO dataset creation")
        
        # Get available images
        images = self.get_available_images()
        
        if not images:
            logger.error("No images available for dataset creation")
            return False
        
        # Limit dataset size if specified
        if max_images and len(images) > max_images:
            images = random.sample(images, max_images)
            logger.info(f"Limited dataset to {max_images} images")
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        logger.info(f"Dataset split: {len(train_images)} train, {len(val_images)} val")
        
        # Process training images
        train_success = 0
        for i, image_info in enumerate(train_images):
            if self.copy_and_label_image(image_info, 'train', i):
                train_success += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(train_images)} training images")
        
        # Process validation images
        val_success = 0
        for i, image_info in enumerate(val_images):
            if self.copy_and_label_image(image_info, 'val', i):
                val_success += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(val_images)} validation images")
        
        # Create dataset YAML
        self.create_dataset_yaml()
        
        # Summary
        total_success = train_success + val_success
        logger.info(f"Dataset creation completed: {total_success}/{len(images)} images processed")
        logger.info(f"Training: {train_success} images")
        logger.info(f"Validation: {val_success} images")
        
        # Create summary statistics
        self.create_dataset_summary(train_success, val_success, train_images + val_images)
        
        return total_success > 0
    
    def create_dataset_summary(self, train_count: int, val_count: int, all_images: List[Dict]):
        """Create a summary report of the dataset."""
        
        # Classification statistics
        classifications = {'storm': 0, 'pre_storm': 0, 'normal': 0}
        
        for image_info in all_images:
            classification, _ = self.classify_image(image_info['observation_time'])
            classifications[classification] += 1
        
        summary = {
            'dataset_info': {
                'total_images': len(all_images),
                'train_images': train_count,
                'val_images': val_count,
                'created_at': datetime.now().isoformat()
            },
            'classifications': classifications,
            'time_range': {
                'start': min(img['observation_time'] for img in all_images).isoformat(),
                'end': max(img['observation_time'] for img in all_images).isoformat()
            },
            'wavelengths': list(set(img['wavelength'] for img in all_images))
        }
        
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dataset summary saved to: {summary_path}")
        logger.info(f"Classifications: Storm={classifications['storm']}, "
                   f"Pre-storm={classifications['pre_storm']}, "
                   f"Normal={classifications['normal']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Local YOLO Dataset Preparation')
    parser.add_argument('--solar-images-dir', default='data/solar_images',
                       help='Directory containing downloaded solar images')
    parser.add_argument('--output-dir', default='data/yolo_dataset',
                       help='Output directory for YOLO dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of images for training (default: 0.8)')
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to include in dataset')
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging()
    
    try:
        # Initialize dataset preparer
        preparer = LocalYOLODatasetPreparer(args.solar_images_dir, args.output_dir)
        
        # Create dataset
        success = preparer.create_dataset(args.train_ratio, args.max_images)
        
        if success:
            logger.info("✅ YOLO dataset creation completed successfully!")
            logger.info(f"Dataset available at: {args.output_dir}")
            logger.info(f"YAML config: {args.output_dir}/dataset.yaml")
        else:
            logger.error("❌ Dataset creation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()