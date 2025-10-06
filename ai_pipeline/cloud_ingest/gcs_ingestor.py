#!/usr/bin/env python3
"""
Solar Image Ingestor for Google Cloud Storage
Fetches solar images from Helioviewer API and uploads to GCS with BigQuery metadata
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx
import os
from google.cloud import storage, bigquery
from google.cloud.exceptions import NotFound
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolarImageIngestor:
    """Fetches solar images from Helioviewer and uploads to GCS with BigQuery metadata"""
    
    def __init__(self, project_id: str, raw_bucket: str = "solar-raw"):
        self.project_id = project_id
        self.raw_bucket = raw_bucket
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        self.helioviewer_base_url = "https://api.helioviewer.org/v2"
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        self._ensure_bigquery_table_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            self.storage_client.get_bucket(self.raw_bucket)
            logger.info(f"Bucket {self.raw_bucket} already exists")
        except NotFound:
            bucket = self.storage_client.bucket(self.raw_bucket)
            bucket = self.storage_client.create_bucket(bucket, location="us-central1")
            logger.info(f"Created bucket {self.raw_bucket}")
    
    def _ensure_bigquery_table_exists(self):
        """Create BigQuery table if it doesn't exist"""
        dataset_id = "space_weather"
        table_id = "solar_events"
        
        # Create dataset if it doesn't exist
        dataset_ref = self.bq_client.dataset(dataset_id)
        try:
            self.bq_client.get_dataset(dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            dataset = self.bq_client.create_dataset(dataset)
            logger.info(f"Created dataset {dataset_id}")
        
        # Create table if it doesn't exist
        table_ref = dataset_ref.table(table_id)
        try:
            self.bq_client.get_table(table_ref)
        except NotFound:
            schema = [
                bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("instrument", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("wavelength", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("gcs_path", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("file_size_bytes", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("image_width", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("image_height", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("ingestion_time", "TIMESTAMP", mode="REQUIRED"),
            ]
            table = bigquery.Table(table_ref, schema=schema)
            table = self.bq_client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    def fetch_solar_image(self, timestamp: datetime, instrument: str = "AIA", 
                         wavelength: int = 193) -> Optional[Dict]:
        """Fetch a single solar image from Helioviewer API"""
        
        # Format timestamp for API
        time_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        params = {
            "date": time_str,
            "imageScale": 2.4,  # arcsec per pixel
            "layers": f"[{instrument},{wavelength},1,100]",
            "events": "",
            "eventLabels": "false",
            "scale": "true",
            "scaleType": "earth",
            "scaleX": -1,
            "scaleY": -1,
            "width": 1024,
            "height": 1024,
            "x0": 0,
            "y0": 0,
            "display": "true",
            "watermark": "false"
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                # Get image metadata first
                response = client.get(f"{self.helioviewer_base_url}/takeScreenshot/", params=params)
                response.raise_for_status()
                
                image_data = response.content
                if len(image_data) == 0:
                    logger.warning(f"Empty image data for {time_str}")
                    return None
                
                return {
                    "image_data": image_data,
                    "timestamp": timestamp,
                    "instrument": instrument,
                    "wavelength": wavelength,
                    "file_size": len(image_data),
                    "width": params["width"],
                    "height": params["height"],
                    "metadata": params
                }
                
        except Exception as e:
            logger.error(f"Error fetching image for {time_str}: {e}")
            return None
    
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
            "height": str(image_data["height"])
        }
        blob.patch()
        
        logger.info(f"Uploaded to gs://{self.raw_bucket}/{gcs_path}")
        return f"gs://{self.raw_bucket}/{gcs_path}"
    
    def log_to_bigquery(self, image_data: Dict, gcs_path: str):
        """Log image metadata to BigQuery"""
        timestamp = image_data["timestamp"]
        event_id = f"{image_data['instrument']}_{image_data['wavelength']}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
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
            logger.info(f"Logged event {event_id} to BigQuery")
    
    def ingest_time_range(self, start_time: datetime, end_time: datetime, 
                         interval_minutes: int = 60, instrument: str = "AIA", 
                         wavelength: int = 193):
        """Ingest solar images for a time range"""
        current_time = start_time
        total_processed = 0
        successful_uploads = 0
        
        logger.info(f"Starting ingestion from {start_time} to {end_time}")
        logger.info(f"Instrument: {instrument}, Wavelength: {wavelength}")
        logger.info(f"Interval: {interval_minutes} minutes")
        
        while current_time <= end_time:
            logger.info(f"Processing {current_time}")
            
            # Fetch image
            image_data = self.fetch_solar_image(current_time, instrument, wavelength)
            if image_data:
                try:
                    # Upload to GCS
                    gcs_path = self.upload_to_gcs(image_data)
                    
                    # Log to BigQuery
                    self.log_to_bigquery(image_data, gcs_path)
                    
                    successful_uploads += 1
                    logger.info(f"Successfully processed {current_time}")
                    
                except Exception as e:
                    logger.error(f"Error processing {current_time}: {e}")
            
            total_processed += 1
            current_time += timedelta(minutes=interval_minutes)
            
            # Small delay to be respectful to the API
            time.sleep(1)
        
        logger.info(f"Ingestion complete. Processed {total_processed} timestamps, "
                   f"successful uploads: {successful_uploads}")


def main():
    parser = argparse.ArgumentParser(description="Ingest solar images to GCS")
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
    
    # Create ingestor and run
    ingestor = SolarImageIngestor(args.project_id, args.bucket)
    ingestor.ingest_time_range(
        start_time=start_time,
        end_time=end_time,
        interval_minutes=args.interval,
        instrument=args.instrument,
        wavelength=args.wavelength
    )
    
    return 0


if __name__ == "__main__":
    exit(main())