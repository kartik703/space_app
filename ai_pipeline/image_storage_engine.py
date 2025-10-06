"""
Live Solar Image Storage Engine
Saves live solar images to database for future CV training
"""

import sqlite3
import os
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class SolarImageStorageEngine:
    """Stores live solar images with metadata for CV training"""
    
    def __init__(self, db_path: str = "data/solar_images.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
        
    def init_database(self):
        """Initialize the image storage database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create images table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS solar_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL,
                    wavelength TEXT,
                    timestamp DATETIME NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    cv_detections TEXT,
                    quality_score REAL,
                    is_training_ready BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create detection results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cv_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x REAL NOT NULL,
                    bbox_y REAL NOT NULL,
                    bbox_width REAL NOT NULL,
                    bbox_height REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES solar_images (id)
                )
            """)
            
            # Create training datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    description TEXT,
                    image_count INTEGER NOT NULL,
                    class_distribution TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'ready'
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Image storage database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            
    def calculate_image_hash(self, image_array: np.ndarray) -> str:
        """Calculate unique hash for image deduplication"""
        return hashlib.md5(image_array.tobytes()).hexdigest()
        
    def assess_image_quality(self, image_array: np.ndarray) -> float:
        """Assess image quality for training suitability"""
        try:
            # Convert to grayscale for quality assessment
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
                
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate contrast using standard deviation
            contrast = float(np.std(gray.astype(np.float64)))
            
            # Calculate brightness distribution
            brightness = float(np.mean(gray.astype(np.float64)))
            
            # Combine metrics (normalized to 0-1 scale)
            quality_score = min(1.0, (laplacian_var / 1000 + contrast / 128 + 
                                    (1 - abs(brightness - 128) / 128)) / 3)
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default quality score
            
    def store_image(self, 
                   image_array: np.ndarray,
                   source: str,
                   wavelength: Optional[str] = None,
                   cv_detections: Optional[List[Dict]] = None,
                   timestamp: Optional[datetime] = None) -> int:
        """Store solar image with metadata"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            # Calculate image hash for deduplication
            image_hash = self.calculate_image_hash(image_array)
            
            # Create file path
            date_str = timestamp.strftime("%Y%m%d")
            time_str = timestamp.strftime("%H%M%S")
            filename = f"{source}_{date_str}_{time_str}_{image_hash[:8]}.jpg"
            
            # Ensure storage directory exists
            storage_dir = f"data/stored_solar_images/{date_str}"
            os.makedirs(storage_dir, exist_ok=True)
            file_path = os.path.join(storage_dir, filename)
            
            # Save image to disk
            if len(image_array.shape) == 3:
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, image_bgr)
            else:
                cv2.imwrite(file_path, image_array)
                
            file_size = os.path.getsize(file_path)
            
            # Assess image quality
            quality_score = self.assess_image_quality(image_array)
            
            # Get image dimensions
            height, width = image_array.shape[:2]
            channels = image_array.shape[2] if len(image_array.shape) == 3 else 1
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO solar_images 
                    (image_hash, source, wavelength, timestamp, width, height, channels,
                     file_path, file_size, cv_detections, quality_score, is_training_ready)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_hash, source, wavelength, timestamp, width, height, channels,
                    file_path, file_size, json.dumps(cv_detections) if cv_detections else None,
                    quality_score, 1 if quality_score > 0.6 else 0
                ))
                
                image_id = cursor.lastrowid
                
                # Store detection results
                if cv_detections:
                    for detection in cv_detections:
                        cursor.execute("""
                            INSERT INTO cv_detections 
                            (image_id, class_name, confidence, bbox_x, bbox_y, bbox_width, bbox_height)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            image_id,
                            detection.get('class', 'unknown'),
                            detection.get('confidence', 0.0),
                            detection.get('bbox', [0, 0, 0, 0])[0],
                            detection.get('bbox', [0, 0, 0, 0])[1],
                            detection.get('bbox', [0, 0, 0, 0])[2],
                            detection.get('bbox', [0, 0, 0, 0])[3]
                        ))
                
                conn.commit()
                logger.info(f"Stored image {image_id}: {source} ({quality_score:.2f} quality)")
                return image_id or 0
                
            except sqlite3.IntegrityError:
                # Image already exists (duplicate hash)
                logger.debug(f"Duplicate image detected: {image_hash}")
                return -1
                
        except Exception as e:
            logger.error(f"Failed to store image: {e}")
            return -1
            
        finally:
            if 'conn' in locals():
                conn.close()
                
    def get_training_ready_images(self, min_quality: float = 0.6) -> List[Dict]:
        """Get images ready for training"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, file_path, source, wavelength, cv_detections, quality_score
                FROM solar_images 
                WHERE is_training_ready = 1 AND quality_score >= ?
                ORDER BY timestamp DESC
            """, (min_quality,))
            
            results = []
            for row in cursor.fetchall():
                image_id, file_path, source, wavelength, detections_json, quality = row
                detections = json.loads(detections_json) if detections_json else []
                
                results.append({
                    'id': image_id,
                    'file_path': file_path,
                    'source': source,
                    'wavelength': wavelength,
                    'detections': detections,
                    'quality_score': quality
                })
                
            conn.close()
            logger.info(f"Found {len(results)} training-ready images")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get training images: {e}")
            return []
            
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total images
            cursor.execute("SELECT COUNT(*) FROM solar_images")
            total_images = cursor.fetchone()[0]
            
            # Training ready images
            cursor.execute("SELECT COUNT(*) FROM solar_images WHERE is_training_ready = 1")
            training_ready = cursor.fetchone()[0]
            
            # Images by source
            cursor.execute("""
                SELECT source, COUNT(*) as count, AVG(quality_score) as avg_quality
                FROM solar_images 
                GROUP BY source
                ORDER BY count DESC
            """)
            by_source = cursor.fetchall()
            
            # Detections by class
            cursor.execute("""
                SELECT class_name, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM cv_detections 
                GROUP BY class_name
                ORDER BY count DESC
            """)
            by_class = cursor.fetchall()
            
            # Recent activity
            cursor.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM solar_images 
                WHERE created_at >= datetime('now', '-7 days')
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """)
            recent_activity = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_images': total_images,
                'training_ready': training_ready,
                'by_source': by_source,
                'by_class': by_class,
                'recent_activity': recent_activity,
                'training_readiness': f"{training_ready}/{total_images}" if total_images > 0 else "0/0"
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
            
    def create_training_dataset(self, 
                               dataset_name: str,
                               min_quality: float = 0.7,
                               max_images_per_class: int = 1000) -> bool:
        """Create a balanced training dataset"""
        try:
            images = self.get_training_ready_images(min_quality)
            
            if not images:
                logger.warning("No training-ready images found")
                return False
                
            # Group by detected classes
            class_groups = {}
            for image in images:
                for detection in image['detections']:
                    class_name = detection.get('class', 'unknown')
                    if class_name not in class_groups:
                        class_groups[class_name] = []
                    class_groups[class_name].append(image)
                    
            # Balance classes
            balanced_images = []
            for class_name, class_images in class_groups.items():
                # Take up to max_images_per_class from each class
                selected = class_images[:max_images_per_class]
                balanced_images.extend(selected)
                logger.info(f"Selected {len(selected)} images for class {class_name}")
                
            # Store dataset info
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            class_distribution = {cls: len(imgs) for cls, imgs in class_groups.items()}
            
            cursor.execute("""
                INSERT INTO training_datasets 
                (dataset_name, description, image_count, class_distribution)
                VALUES (?, ?, ?, ?)
            """, (
                dataset_name,
                f"Auto-generated dataset with min quality {min_quality}",
                len(balanced_images),
                json.dumps(class_distribution)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created training dataset '{dataset_name}' with {len(balanced_images)} images")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create training dataset: {e}")
            return False