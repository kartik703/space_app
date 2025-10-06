#!/usr/bin/env python3
"""
Enhanced YOLO Dataset Preparation with GCS and NOAA Event Integration
Creates YOLO annotations based on actual NOAA solar event data
"""

import os
import logging
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from google.cloud import storage, bigquery
from PIL import Image, ImageDraw
import io
import tempfile

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_DATASET_DIR, YOLO_CLASSES
from preprocess.prepare_yolo_dataset import YOLODatasetPreparer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GCSYOLODatasetPreparer(YOLODatasetPreparer):
    """Enhanced YOLO dataset preparer that works with GCS and NOAA events"""
    
    def __init__(self, project_id: str, raw_bucket: str = "solar-raw", 
                 output_dir: str = YOLO_DATASET_DIR):
        super().__init__(output_dir=output_dir)
        self.project_id = project_id
        self.raw_bucket = raw_bucket
        
        # Initialize Google Cloud clients
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        
        logger.info(f"Initialized GCS YOLO preparer for project: {project_id}")
    
    def list_gcs_images(self, prefix: str = "", limit: int = None) -> List[Dict]:
        """List solar images in GCS bucket"""
        logger.info(f"Listing images in gs://{self.raw_bucket}/{prefix}")
        
        bucket = self.storage_client.bucket(self.raw_bucket)
        blobs = bucket.list_blobs(prefix=prefix)
        
        images = []
        count = 0
        
        for blob in blobs:
            if blob.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Extract metadata from blob
                image_info = {
                    'gcs_path': f"gs://{self.raw_bucket}/{blob.name}",
                    'blob_name': blob.name,
                    'size': blob.size,
                    'updated': blob.updated,
                    'metadata': blob.metadata or {}
                }
                
                # Try to parse timestamp from filename or metadata
                if 'timestamp' in blob.metadata:
                    try:
                        image_info['timestamp'] = datetime.fromisoformat(blob.metadata['timestamp'])
                    except:
                        image_info['timestamp'] = None
                else:
                    # Try to extract from filename
                    image_info['timestamp'] = self._extract_timestamp_from_filename(blob.name)
                
                images.append(image_info)
                count += 1
                
                if limit and count >= limit:
                    break
        
        logger.info(f"Found {len(images)} images in GCS")
        return images
    
    def _extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract timestamp from filename patterns"""
        import re
        
        # Pattern: YYYYMMDD_HHMMSS
        pattern1 = r'(\d{8}_\d{6})'
        match = re.search(pattern1, filename)
        if match:
            try:
                return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
            except:
                pass
        
        # Pattern: YYYY/MM/DD structure in path
        pattern2 = r'(\d{4})/(\d{2})/(\d{2})'
        match = re.search(pattern2, filename)
        if match:
            try:
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))
            except:
                pass
        
        return None
    
    def fetch_noaa_events_for_timerange(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Fetch NOAA events from BigQuery for a time range"""
        logger.info(f"Fetching NOAA events from {start_time} to {end_time}")
        
        query = f"""
        SELECT 
            event_id,
            start_time,
            peak_time,
            end_time,
            flare_class as event_class,
            location,
            source_region,
            event_type,
            remarks as particulars
        FROM `{self.project_id}.space_weather.real_noaa_events`
        WHERE start_time >= @start_time
        AND start_time <= @end_time
        ORDER BY start_time
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", start_time),
                bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", end_time),
            ]
        )
        
        try:
            query_job = self.bq_client.query(query, job_config=job_config)
            results = query_job.result()
            
            events = []
            for row in results:
                events.append({
                    'event_id': row.event_id,
                    'start_time': row.start_time,
                    'peak_time': row.peak_time,
                    'end_time': row.end_time,
                    'event_class': row.event_class,
                    'location': row.location,
                    'source_region': row.source_region,
                    'event_type': row.event_type,
                    'particulars': row.particulars
                })
            
            logger.info(f"Found {len(events)} NOAA events")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching NOAA events: {e}")
            return []
    
    def fetch_soho_cme_events_for_timerange(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Fetch SOHO CME events from BigQuery for a time range"""
        logger.info(f"Fetching SOHO CME events from {start_time} to {end_time}")
        
        query = f"""
        SELECT 
            event_id,
            datetime,
            cme_type,
            width,
            speed,
            source_location,
            gcs_image_path
        FROM `{self.project_id}.space_weather.soho_cme_events`
        WHERE datetime >= @start_time
        AND datetime <= @end_time
        AND gcs_image_path IS NOT NULL
        ORDER BY datetime
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", start_time),
                bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", end_time),
            ]
        )
        
        try:
            query_job = self.bq_client.query(query, job_config=job_config)
            results = query_job.result()
            
            events = []
            for row in results:
                events.append({
                    'event_id': row.event_id,
                    'datetime': row.datetime,
                    'cme_type': row.cme_type,
                    'width': row.width,
                    'speed': row.speed,
                    'source_location': row.source_location,
                    'gcs_image_path': row.gcs_image_path
                })
            
            logger.info(f"Found {len(events)} SOHO CME events with images")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching SOHO CME events: {e}")
            return []
    
    def match_events_to_images(self, images: List[Dict], events: List[Dict], 
                              time_tolerance_minutes: int = 30) -> Dict[str, List[Dict]]:
        """Match NOAA/SOHO events to solar images based on timestamps"""
        logger.info(f"Matching events to images with {time_tolerance_minutes} minute tolerance")
        
        image_events = {}
        tolerance = timedelta(minutes=time_tolerance_minutes)
        
        for image in images:
            if not image.get('timestamp'):
                continue
            
            image_time = image['timestamp']
            matched_events = []
            
            for event in events:
                # Get event time (use peak_time if available, otherwise start_time)
                if 'peak_time' in event and event['peak_time']:
                    event_time = event['peak_time']
                elif 'datetime' in event:
                    event_time = event['datetime']
                else:
                    event_time = event['start_time']
                
                # Check if event is within tolerance of image time
                time_diff = abs(image_time - event_time)
                if time_diff <= tolerance:
                    matched_events.append(event)
            
            if matched_events:
                image_events[image['gcs_path']] = matched_events
        
        logger.info(f"Matched {len(image_events)} images with events")
        return image_events
    
    def create_yolo_annotation_from_events(self, events: List[Dict], image_width: int = 1024, 
                                          image_height: int = 1024) -> List[str]:
        """Create YOLO format annotations from NOAA/SOHO events"""
        annotations = []
        
        for event in events:
            class_id = self._map_event_to_class(event)
            if class_id is None:
                continue
            
            # Generate bounding box based on event type and location
            bbox = self._generate_bbox_from_event(event, image_width, image_height)
            if bbox:
                # YOLO format: class_id center_x center_y width height (normalized)
                annotations.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        
        return annotations
    
    def _map_event_to_class(self, event: Dict) -> Optional[int]:
        """Map event type to YOLO class ID"""
        event_type = event.get('event_type', '').lower()
        event_class = event.get('event_class', '').lower()
        
        # Map based on event characteristics
        if 'flare' in event_type or event_class.startswith(('a', 'b', 'c', 'm', 'x')):
            return 0  # flare
        elif 'cme' in event_type or event.get('cme_type'):
            return 1  # cme
        elif 'sunspot' in event_type or event.get('source_region'):
            return 2  # sunspot
        elif 'particle' in event_type or 'radiation' in event_type:
            return 3  # debris/particle
        
        return None
    
    def _generate_bbox_from_event(self, event: Dict, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
        """Generate bounding box from event location information"""
        location = event.get('location', '')
        
        # Default center position
        center_x = 0.5
        center_y = 0.5
        
        # Try to parse location (e.g., "S15E10", "N23W45")
        if location and len(location) >= 4:
            try:
                # Parse heliographic coordinates
                ns = location[0].upper()  # N or S
                lat_str = location[1:3]
                ew = location[3].upper()  # E or W
                lon_str = location[4:6] if len(location) >= 6 else location[4:]
                
                lat = int(lat_str)
                lon = int(lon_str)
                
                # Convert to image coordinates (approximate)
                # Sun center is at (0.5, 0.5), coordinates range from -90 to +90
                if ns == 'S':
                    lat = -lat
                if ew == 'W':
                    lon = -lon
                
                # Map to image coordinates (very approximate)
                center_x = 0.5 + (lon / 180.0) * 0.4  # Scale factor for solar disk
                center_y = 0.5 - (lat / 180.0) * 0.4  # Y axis inverted
                
                # Clamp to valid range
                center_x = max(0.1, min(0.9, center_x))
                center_y = max(0.1, min(0.9, center_y))
                
            except (ValueError, IndexError):
                # Fall back to center if parsing fails
                pass
        
        # Set bounding box size based on event type
        event_type = event.get('event_type', '').lower()
        if 'flare' in event_type:
            bbox_width = 0.1
            bbox_height = 0.1
        elif 'cme' in event_type:
            bbox_width = 0.3
            bbox_height = 0.3
        else:
            bbox_width = 0.15
            bbox_height = 0.15
        
        return (center_x, center_y, bbox_width, bbox_height)
    
    def download_gcs_image(self, gcs_path: str, local_path: str):
        """Download image from GCS to local file"""
        # Parse GCS path
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        path_parts = gcs_path[5:].split('/', 1)  # Remove 'gs://' and split
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.download_to_filename(local_path)
        logger.debug(f"Downloaded {gcs_path} to {local_path}")
    
    def prepare_gcs_dataset(self, start_date: str, end_date: str, 
                           max_images: int = 1000, time_tolerance_minutes: int = 30):
        """Prepare YOLO dataset from GCS images and NOAA events"""
        logger.info("Starting GCS-based YOLO dataset preparation")
        
        # Parse dates
        start_time = datetime.fromisoformat(start_date)
        end_time = datetime.fromisoformat(end_date)
        
        # Fetch images from GCS
        images = self.list_gcs_images(limit=max_images)
        
        # Filter images by date range
        filtered_images = []
        for img in images:
            if img.get('timestamp') and start_time <= img['timestamp'] <= end_time:
                filtered_images.append(img)
        
        logger.info(f"Found {len(filtered_images)} images in date range")
        
        # Fetch events
        noaa_events = self.fetch_noaa_events_for_timerange(start_time, end_time)
        soho_events = self.fetch_soho_cme_events_for_timerange(start_time, end_time)
        all_events = noaa_events + soho_events
        
        # Match events to images
        image_events = self.match_events_to_images(filtered_images, all_events, time_tolerance_minutes)
        
        # Prepare dataset
        self.setup_directories()
        
        # Split images
        split_data = self._split_images_with_events(filtered_images, image_events)
        
        # Process each split
        for split_name, split_images in split_data.items():
            logger.info(f"Processing {split_name} set with {len(split_images)} images")
            
            self._process_split_with_gcs(split_name, split_images, image_events)
        
        # Create dataset YAML
        yaml_path = self.create_dataset_yaml()
        
        logger.info("GCS dataset preparation complete!")
        return yaml_path
    
    def _split_images_with_events(self, images: List[Dict], image_events: Dict) -> Dict[str, List[Dict]]:
        """Split images ensuring events are distributed across splits"""
        # Separate images with and without events
        images_with_events = [img for img in images if img['gcs_path'] in image_events]
        images_without_events = [img for img in images if img['gcs_path'] not in image_events]
        
        logger.info(f"Images with events: {len(images_with_events)}")
        logger.info(f"Images without events: {len(images_without_events)}")
        
        # Split ratios
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        
        # Split images with events
        event_count = len(images_with_events)
        event_train_count = int(event_count * train_ratio)
        event_val_count = int(event_count * val_ratio)
        
        event_train = images_with_events[:event_train_count]
        event_val = images_with_events[event_train_count:event_train_count + event_val_count]
        event_test = images_with_events[event_train_count + event_val_count:]
        
        # Split images without events
        no_event_count = len(images_without_events)
        no_event_train_count = int(no_event_count * train_ratio)
        no_event_val_count = int(no_event_count * val_ratio)
        
        no_event_train = images_without_events[:no_event_train_count]
        no_event_val = images_without_events[no_event_train_count:no_event_train_count + no_event_val_count]
        no_event_test = images_without_events[no_event_train_count + no_event_val_count:]
        
        return {
            'train': event_train + no_event_train,
            'val': event_val + no_event_val,
            'test': event_test + no_event_test
        }
    
    def _process_split_with_gcs(self, split_name: str, images: List[Dict], image_events: Dict):
        """Process a data split by downloading images and creating annotations"""
        images_dir = os.path.join(self.output_dir, "images", split_name)
        labels_dir = os.path.join(self.output_dir, "labels", split_name)
        
        for image_info in images:
            gcs_path = image_info['gcs_path']
            
            try:
                # Generate local filename
                timestamp = image_info.get('timestamp')
                if timestamp:
                    local_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                else:
                    local_filename = os.path.basename(image_info['blob_name'])
                
                # Download image
                local_image_path = os.path.join(images_dir, local_filename)
                self.download_gcs_image(gcs_path, local_image_path)
                
                # Create annotation
                label_filename = os.path.splitext(local_filename)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_filename)
                
                if gcs_path in image_events:
                    # Create annotation from events
                    events = image_events[gcs_path]
                    annotations = self.create_yolo_annotation_from_events(events)
                else:
                    # Create empty annotation (negative example)
                    annotations = []
                
                # Write annotation file
                with open(label_path, 'w') as f:
                    for annotation in annotations:
                        f.write(annotation + '\n')
                
                logger.debug(f"Processed {local_filename} with {len(annotations)} annotations")
                
            except Exception as e:
                logger.error(f"Error processing {gcs_path}: {e}")
                continue


def main():
    """Main function for GCS-based dataset preparation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from GCS with NOAA events")
    parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--bucket", default="solar-raw", help="GCS bucket name")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-images", type=int, default=1000, help="Maximum images to process")
    parser.add_argument("--tolerance", type=int, default=30, help="Time tolerance in minutes")
    
    args = parser.parse_args()
    
    # Create preparer
    preparer = GCSYOLODatasetPreparer(args.project_id, args.bucket)
    
    # Prepare dataset
    try:
        yaml_path = preparer.prepare_gcs_dataset(
            start_date=args.start_date,
            end_date=args.end_date,
            max_images=args.max_images,
            time_tolerance_minutes=args.tolerance
        )
        
        logger.info(f"Dataset ready! Configuration: {yaml_path}")
        logger.info("Next step: python cv_pipeline/train/train_yolo.py")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())