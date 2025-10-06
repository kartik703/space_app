#!/usr/bin/env python3
"""
Mock Solar Image Generator for Testing GCS Pipeline
Creates synthetic solar images for testing when external APIs are unavailable
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from google.cloud import storage, bigquery
from google.cloud.exceptions import NotFound
import json
import time
import numpy as np
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockSolarImageGenerator:
    """Generate mock solar images for testing GCS pipeline"""
    
    def __init__(self, project_id: str, raw_bucket: str = "solar-raw"):
        self.project_id = project_id
        self.raw_bucket = raw_bucket
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        
    def create_synthetic_solar_image(self, timestamp: datetime, wavelength: int = 193) -> bytes:
        """Create a synthetic solar image"""
        # Create a 1024x1024 image
        size = (1024, 1024)
        width, height = size
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        
        # Create base image
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add solar disk with wavelength-dependent color
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        
        # Different colors for different wavelengths
        if wavelength == 193:
            color = [255, 200, 50]  # Orange for 193Å
        elif wavelength == 171:
            color = [255, 255, 100]  # Yellow for 171Å
        elif wavelength == 211:
            color = [200, 100, 255]  # Purple for 211Å
        else:
            color = [255, 255, 255]  # White for unknown
            
        img_array[mask] = color
        
        # Add some noise for realism
        noise = np.random.randint(-20, 20, (height, width, 3))
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add some "solar features" based on timestamp (creates time variation)
        num_features = 3 + int(timestamp.hour / 4)  # More features later in day
        for i in range(num_features):
            # Use timestamp to create reproducible but varied features
            seed = int(timestamp.timestamp()) + i * 1000
            np.random.seed(seed)
            
            spot_x = np.random.randint(center_x - radius//2, center_x + radius//2)
            spot_y = np.random.randint(center_y - radius//2, center_y + radius//2)
            spot_radius = np.random.randint(8, 25)
            
            y_spot, x_spot = np.ogrid[:height, :width]
            spot_mask = (x_spot - spot_x) ** 2 + (y_spot - spot_y) ** 2 <= spot_radius ** 2
            img_array[spot_mask] = [max(0, c - 100) for c in color]  # Darker spots
        
        # Convert to PIL Image and save as JPEG bytes
        image = Image.fromarray(img_array)
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='JPEG', quality=95)
        return byte_arr.getvalue()
    
    def create_mock_image_data(self, timestamp: datetime, instrument: str = "AIA", 
                              wavelength: int = 193) -> Dict:
        """Create mock image data similar to real API response"""
        image_bytes = self.create_synthetic_solar_image(timestamp, wavelength)
        
        return {
            "image_data": image_bytes,
            "timestamp": timestamp,
            "instrument": instrument,
            "wavelength": wavelength,
            "file_size": len(image_bytes),
            "width": 1024,
            "height": 1024,
            "metadata": {
                "source": "mock_generator",
                "date": timestamp.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "imageScale": 2.4,
                "layers": f"[{instrument},{wavelength},1,100]",
                "width": 1024,
                "height": 1024
            }
        }
    
    def upload_to_gcs(self, image_data: Dict) -> str:
        """Upload image to GCS and return the GCS path"""
        timestamp = image_data["timestamp"]
        instrument = image_data["instrument"].lower()
        wavelength = image_data["wavelength"]
        
        # Create GCS path: instrument/wavelength/year/month/day/filename
        gcs_path = (f"{instrument}/{wavelength}/"
                   f"{timestamp.year:04d}/{timestamp.month:02d}/{timestamp.day:02d}/"
                   f"{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg")
        
        bucket = self.storage_client.bucket(self.raw_bucket)
        blob = bucket.blob(gcs_path)
        
        # Upload with metadata
        blob.upload_from_string(
            image_data["image_data"],
            content_type="image/jpeg"
        )
        
        # Set metadata
        blob.metadata = {
            "instrument": image_data["instrument"],
            "wavelength": str(image_data["wavelength"]),
            "timestamp": image_data["timestamp"].isoformat(),
            "width": str(image_data["width"]),
            "height": str(image_data["height"]),
            "source": "mock_generator"
        }
        blob.patch()
        
        logger.info(f"Uploaded mock image to gs://{self.raw_bucket}/{gcs_path}")
        return f"gs://{self.raw_bucket}/{gcs_path}"
    
    def log_to_bigquery(self, image_data: Dict, gcs_path: str):
        """Log image metadata to BigQuery"""
        timestamp = image_data["timestamp"]
        event_id = f"MOCK_{image_data['instrument']}_{image_data['wavelength']}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        row = {
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "instrument": image_data["instrument"],
            "wavelength": image_data["wavelength"],
            "gcs_path": gcs_path,
            "file_size_bytes": image_data["file_size"],
            "image_width": image_data["width"],
            "image_height": image_data["height"],
            "metadata": json.dumps(image_data["metadata"]),
            "ingestion_time": datetime.utcnow().isoformat(),
        }
        
        table_ref = self.bq_client.dataset("space_weather").table("solar_events")
        errors = self.bq_client.insert_rows_json(table_ref, [row])
        
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
        else:
            logger.info(f"Logged mock event {event_id} to BigQuery")
    
    def generate_test_data(self, start_time: datetime, end_time: datetime, 
                          interval_minutes: int = 60, instrument: str = "AIA", 
                          wavelength: int = 193):
        """Generate test solar images for a time range"""
        current_time = start_time
        total_processed = 0
        successful_uploads = 0
        
        logger.info(f"Starting mock data generation from {start_time} to {end_time}")
        logger.info(f"Instrument: {instrument}, Wavelength: {wavelength}")
        logger.info(f"Interval: {interval_minutes} minutes")
        
        while current_time <= end_time:
            logger.info(f"Processing {current_time}")
            
            try:
                # Create mock image data
                image_data = self.create_mock_image_data(current_time, instrument, wavelength)
                
                # Upload to GCS
                gcs_path = self.upload_to_gcs(image_data)
                
                # Log to BigQuery
                self.log_to_bigquery(image_data, gcs_path)
                
                successful_uploads += 1
                
            except Exception as e:
                logger.error(f"Error processing {current_time}: {e}")
            
            total_processed += 1
            current_time += timedelta(minutes=interval_minutes)
            
            # Small delay to avoid overwhelming services
            time.sleep(0.5)
        
        logger.info(f"Mock data generation complete. Processed {total_processed} timestamps, "
                   f"successful uploads: {successful_uploads}")


def main():
    parser = argparse.ArgumentParser(description="Generate mock solar images for testing")
    parser.add_argument("--start", required=True, help="Start time (ISO format: 2012-03-04T00:00:00)")
    parser.add_argument("--end", required=True, help="End time (ISO format: 2012-03-04T23:59:59)")
    parser.add_argument("--project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT"), 
                       help="Google Cloud Project ID")
    parser.add_argument("--bucket", default="solar-raw", help="GCS bucket for raw images")
    parser.add_argument("--instrument", default="AIA", help="Solar instrument (default: AIA)")
    parser.add_argument("--wavelength", type=int, default=193, help="Wavelength (default: 193)")
    parser.add_argument("--interval", type=int, default=60, help="Interval in minutes (default: 60)")
    
    args = parser.parse_args()
    
    if not args.project_id:
        logger.error("Project ID is required. Set GOOGLE_CLOUD_PROJECT environment variable or use --project-id")
        return 1
    
    # Parse timestamps
    try:
        start_time = datetime.fromisoformat(args.start.replace('Z', '+00:00')).replace(tzinfo=None)
        end_time = datetime.fromisoformat(args.end.replace('Z', '+00:00')).replace(tzinfo=None)
    except ValueError as e:
        logger.error(f"Invalid timestamp format: {e}")
        return 1
    
    # Create generator and run
    generator = MockSolarImageGenerator(args.project_id, args.bucket)
    generator.generate_test_data(
        start_time=start_time,
        end_time=end_time,
        interval_minutes=args.interval,
        instrument=args.instrument,
        wavelength=args.wavelength
    )
    
    return 0


if __name__ == "__main__":
    exit(main())