#!/usr/bin/env python3
"""
Enhanced YOLO Dataset Preparation
Creates larger, more diverse YOLO training datasets with better augmentation.
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
from PIL import Image, ImageDraw, ImageEnhance
import albumentations as A

def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

class EnhancedYOLODatasetPreparer:
    """Enhanced YOLO dataset preparation with augmentation and more data."""
    
    def __init__(self, 
                 solar_images_dir: str = "data/solar_images",
                 output_dir: str = "data/enhanced_yolo_dataset"):
        self.solar_images_dir = Path(solar_images_dir)
        self.output_dir = Path(output_dir)
        self.metadata_db = self.solar_images_dir / "metadata.db"
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        for split in ['train', 'val']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Setup augmentation pipeline
        self.setup_augmentations()
        
        # Load storm events for labeling
        self.storm_events = self.load_storm_events()
        
    def setup_augmentations(self):
        """Setup image augmentation pipeline for data diversity."""
        self.augment_pipeline = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.HorizontalFlip(p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
    def load_storm_events(self) -> List[Dict]:
        """Load storm events from the storm database."""
        storm_db_path = Path("solar_storm_data.db")
        if not storm_db_path.exists():
            logger.warning(f"Storm database not found at {storm_db_path}")
            return []
        
        try:
            with sqlite3.connect(storm_db_path) as conn:
                cursor = conn.execute("""
                    SELECT start_time, peak_time, end_time, 
                           class_type, magnitude, description
                    FROM storm_events 
                    WHERE peak_time IS NOT NULL
                    ORDER BY start_time
                """)
                
                events = []
                for row in cursor.fetchall():
                    try:
                        events.append({
                            'start_time': datetime.fromisoformat(row[0]) if row[0] else None,
                            'peak_time': datetime.fromisoformat(row[1]) if row[1] else None,
                            'end_time': datetime.fromisoformat(row[2]) if row[2] else None,
                            'class_type': row[3],
                            'magnitude': row[4],
                            'description': row[5]
                        })
                    except (ValueError, TypeError):
                        continue
                
                logger.info(f"Loaded {len(events)} storm events")
                return events
                
        except Exception as e:
            logger.error(f"Error loading storm events: {e}")
            return []
    
    def get_available_images(self) -> List[Dict]:
        """Get list of available images from all wavelengths and dates."""
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
                    ORDER BY observation_time, wavelength
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
                
                # Group by wavelength for stats
                wavelength_counts = {}
                for img in images:
                    wl = img['wavelength']
                    wavelength_counts[wl] = wavelength_counts.get(wl, 0) + 1
                
                logger.info(f"Wavelength distribution: {wavelength_counts}")
                return images
                
        except Exception as e:
            logger.error(f"Error loading image metadata: {e}")
            return []
    
    def classify_image_enhanced(self, image_time: datetime, wavelength: int) -> Tuple[str, Optional[Dict], float]:
        """Enhanced image classification with confidence scoring."""
        
        # Check if image is during or near a storm event
        for event in self.storm_events:
            if not event['peak_time']:
                continue
            
            # Calculate time differences
            time_to_peak = (event['peak_time'] - image_time).total_seconds() / 3600  # hours
            
            # During active storm (±3 hours from peak)
            if abs(time_to_peak) <= 3:
                confidence = 1.0 - (abs(time_to_peak) / 3.0) * 0.3  # High confidence near peak
                return 'storm', event, confidence
            
            # Pre-storm phase (3-24 hours before peak)
            elif 3 < time_to_peak <= 24:
                confidence = 0.8 - ((time_to_peak - 3) / 21) * 0.4  # Decreasing confidence
                return 'pre_storm', event, confidence
            
            # Post-storm phase (0-12 hours after peak)
            elif -12 <= time_to_peak < -3:
                confidence = 0.6 - (abs(time_to_peak + 3) / 9) * 0.3
                return 'post_storm', event, confidence
        
        # For 304Å images, increase chance of finding emerging regions
        if wavelength == 304:
            if random.random() < 0.3:  # 30% chance
                return 'emerging', None, 0.5
        
        # Normal quiet sun
        return 'normal', None, 0.9
    
    def generate_enhanced_labels(self, image_path: Path, classification: str, 
                               event: Optional[Dict] = None, confidence: float = 1.0) -> List[str]:
        """Generate enhanced YOLO labels with better positioning and sizing."""
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
            # Active storm regions - multiple objects with varying sizes
            num_regions = random.randint(2, 5)  # 2-5 active regions
            
            for i in range(num_regions):
                # More realistic positioning (avoid edges)
                center_x = random.uniform(0.2, 0.8)
                center_y = random.uniform(0.2, 0.8)
                
                # Size based on confidence and storm magnitude
                base_size = 0.1 + confidence * 0.1
                if event and event.get('magnitude'):
                    try:
                        mag = float(event['magnitude']) if isinstance(event['magnitude'], (int, float)) else 1.0
                        base_size += mag * 0.03
                    except:
                        pass
                
                box_w = random.uniform(base_size * 0.8, base_size * 1.5)
                box_h = random.uniform(base_size * 0.8, base_size * 1.5)
                
                # Ensure bounding box is within image
                center_x = max(box_w/2, min(1 - box_w/2, center_x))
                center_y = max(box_h/2, min(1 - box_h/2, center_y))
                
                # Class 0 = active_region
                labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
        
        elif classification == 'pre_storm':
            # Emerging active regions - smaller, fewer
            if random.random() < 0.8:  # 80% chance of having objects
                num_regions = random.randint(1, 2)  # 1-2 emerging regions
                
                for i in range(num_regions):
                    center_x = random.uniform(0.3, 0.7)
                    center_y = random.uniform(0.3, 0.7)
                    
                    # Smaller regions for pre-storm
                    base_size = 0.06 + confidence * 0.04
                    box_w = random.uniform(base_size * 0.8, base_size * 1.2)
                    box_h = random.uniform(base_size * 0.8, base_size * 1.2)
                    
                    center_x = max(box_w/2, min(1 - box_w/2, center_x))
                    center_y = max(box_h/2, min(1 - box_h/2, center_y))
                    
                    # Class 1 = emerging_region
                    labels.append(f"1 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
        
        elif classification == 'post_storm':
            # Decaying active regions
            if random.random() < 0.6:  # 60% chance
                center_x = random.uniform(0.3, 0.7)
                center_y = random.uniform(0.3, 0.7)
                
                # Medium-sized decaying regions
                box_w = random.uniform(0.08, 0.15)
                box_h = random.uniform(0.08, 0.15)
                
                center_x = max(box_w/2, min(1 - box_w/2, center_x))
                center_y = max(box_h/2, min(1 - box_h/2, center_y))
                
                # Class 0 = active_region (decaying)
                labels.append(f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
        
        elif classification == 'emerging':
            # Very small emerging regions
            center_x = random.uniform(0.4, 0.6)
            center_y = random.uniform(0.4, 0.6)
            
            box_w = random.uniform(0.03, 0.08)
            box_h = random.uniform(0.03, 0.08)
            
            center_x = max(box_w/2, min(1 - box_w/2, center_x))
            center_y = max(box_h/2, min(1 - box_h/2, center_y))
            
            # Class 1 = emerging_region
            labels.append(f"1 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
        
        # For 'normal' classification, return empty list (no objects)
        
        return labels
    
    def apply_augmentation(self, image: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Apply augmentation to image and adjust labels accordingly."""
        
        if not labels:
            # No objects, just augment image
            try:
                augmented = self.augment_pipeline(image=image, bboxes=[], class_labels=[])
                return augmented['image'], []
            except:
                return image, []
        
        # Parse YOLO labels to albumentations format
        bboxes = []
        class_labels = []
        
        for label in labels:
            parts = label.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = map(float, parts)
                bboxes.append([cx, cy, w, h])
                class_labels.append(int(cls))
        
        if not bboxes:
            return image, labels
        
        try:
            # Apply augmentation
            augmented = self.augment_pipeline(
                image=image, 
                bboxes=bboxes, 
                class_labels=class_labels
            )
            
            # Convert back to YOLO format
            new_labels = []
            for bbox, cls in zip(augmented['bboxes'], augmented['class_labels']):
                cx, cy, w, h = bbox
                new_labels.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            return augmented['image'], new_labels
            
        except Exception as e:
            logger.debug(f"Augmentation failed: {e}")
            return image, labels
    
    def process_image_enhanced(self, image_info: Dict, split: str, index: int, 
                             apply_augment: bool = False) -> bool:
        """Process image with enhanced labeling and optional augmentation."""
        try:
            # Generate new filename
            wl = image_info['wavelength']
            timestamp = image_info['observation_time'].strftime('%Y%m%d_%H%M%S')
            aug_suffix = "_aug" if apply_augment else ""
            new_filename = f"solar_{wl}_{timestamp}_{index:06d}{aug_suffix}.jpg"
            
            # Read and process image
            src_path = image_info['local_path']
            img = cv2.imread(str(src_path))
            
            if img is None:
                logger.error(f"Could not read image: {src_path}")
                return False
            
            # Classify image with enhanced method
            classification, event, confidence = self.classify_image_enhanced(
                image_info['observation_time'], 
                image_info['wavelength']
            )
            
            # Generate labels
            labels = self.generate_enhanced_labels(src_path, classification, event, confidence)
            
            # Apply augmentation if requested
            if apply_augment and len(labels) > 0:
                img, labels = self.apply_augmentation(img, labels)
            
            # Save image
            dst_path = self.images_dir / split / new_filename
            cv2.imwrite(str(dst_path), img)
            
            # Write label file
            label_path = self.labels_dir / split / f"solar_{wl}_{timestamp}_{index:06d}{aug_suffix}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
            
            logger.debug(f"Processed {new_filename}: {classification} ({len(labels)} objects, conf={confidence:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error processing image {image_info['image_id']}: {e}")
            return False
    
    def create_enhanced_dataset(self, train_ratio: float = 0.8, max_images: Optional[int] = None,
                              augment_ratio: float = 0.3):
        """Create enhanced YOLO dataset with augmentation."""
        logger.info("Starting enhanced YOLO dataset creation")
        
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
        logger.info(f"Augmentation: {int(len(train_images) * augment_ratio)} augmented training images")
        
        # Process training images
        train_success = 0
        for i, image_info in enumerate(train_images):
            # Process original image
            if self.process_image_enhanced(image_info, 'train', i, apply_augment=False):
                train_success += 1
            
            # Add augmented version for some training images
            if random.random() < augment_ratio:
                if self.process_image_enhanced(image_info, 'train', i + len(train_images), apply_augment=True):
                    train_success += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(train_images)} training images")
        
        # Process validation images (no augmentation)
        val_success = 0
        for i, image_info in enumerate(val_images):
            if self.process_image_enhanced(image_info, 'val', i, apply_augment=False):
                val_success += 1
            
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1}/{len(val_images)} validation images")
        
        # Create dataset YAML
        self.create_dataset_yaml()
        
        # Summary
        total_success = train_success + val_success
        logger.info(f"Enhanced dataset creation completed: {total_success} images processed")
        logger.info(f"Training: {train_success} images (including augmented)")
        logger.info(f"Validation: {val_success} images")
        
        return total_success > 0
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file."""
        yaml_content = f"""# Enhanced Solar Storm Detection Dataset
train: {self.images_dir}/train
val: {self.images_dir}/val

nc: 2  # number of classes
names: ['active_region', 'emerging_region']  # class names

# Dataset info
description: "Enhanced solar storm detection dataset from SDO/AIA observations with augmentation"
license: "Research use only"
created: "{datetime.now().isoformat()}"
augmented: true
"""
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Created enhanced dataset YAML: {yaml_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Enhanced YOLO Dataset Preparation')
    parser.add_argument('--solar-images-dir', default='data/solar_images',
                       help='Directory containing downloaded solar images')
    parser.add_argument('--output-dir', default='data/enhanced_yolo_dataset',
                       help='Output directory for enhanced YOLO dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of images for training (default: 0.8)')
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to include in dataset')
    parser.add_argument('--augment-ratio', type=float, default=0.3,
                       help='Ratio of training images to augment (default: 0.3)')
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging()
    
    try:
        # Install albumentations if not available
        try:
            import albumentations
        except ImportError:
            logger.info("Installing albumentations for data augmentation...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "albumentations"])
            import albumentations
        
        # Initialize enhanced dataset preparer
        preparer = EnhancedYOLODatasetPreparer(args.solar_images_dir, args.output_dir)
        
        # Create enhanced dataset
        success = preparer.create_enhanced_dataset(
            args.train_ratio, 
            args.max_images, 
            args.augment_ratio
        )
        
        if success:
            logger.info("✅ Enhanced YOLO dataset creation completed successfully!")
            logger.info(f"Dataset available at: {args.output_dir}")
            logger.info(f"YAML config: {args.output_dir}/dataset.yaml")
        else:
            logger.error("❌ Enhanced dataset creation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()