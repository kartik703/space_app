#!/usr/bin/env python3
"""
Enhanced JSOC Data Ingestor for Large-Scale Solar Data Collection
Handles months of data efficiently with JPEG conversion and GCS upload
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
import drms
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
import numpy as np
from PIL import Image
import io
from google.cloud import storage, bigquery
import tempfile
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BulkJSOCIngestor:
    """Enhanced JSOC ingestor for large-scale data collection"""
    
    def __init__(self, project_id: str, bucket_name: str = "solar-raw", 
                 bq_dataset: str = "space_weather"):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.bq_dataset = bq_dataset
        
        # Initialize clients
        self.storage_client = storage.Client(project=project_id)
        self.bq_client = bigquery.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # DRMS client
        self.drms_client = drms.Client()
        
        # Ensure BigQuery table exists
        self._setup_bigquery_table()
        
    def _setup_bigquery_table(self):
        """Setup BigQuery table for solar image metadata"""
        
        # Create dataset if it doesn't exist
        dataset_id = f"{self.project_id}.{self.bq_dataset}"
        try:
            self.bq_client.get_dataset(dataset_id)
        except:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            self.bq_client.create_dataset(dataset)
            logger.info(f"Created dataset {dataset_id}")
        
        # Define table schema
        schema = [
            bigquery.SchemaField("image_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("observation_time", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("wavelength", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("gcs_path", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("exposure_time", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("quality", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("image_scale", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("solar_b0", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("solar_l0", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("solar_r", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("processing_date", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("has_flares", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("flare_labels", "STRING", mode="NULLABLE")
        ]
        
        table_id = f"{dataset_id}.solar_images"
        try:
            table = self.bq_client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except:
            table = bigquery.Table(table_id, schema=schema)
            table = self.bq_client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    def query_jsoc_series(self, start_time: str, end_time: str, wavelength: int, 
                         cadence: int = 600) -> list:
        """Query JSOC for AIA data in specified time range"""
        
        series = f'aia.lev1_euv_12s[{start_time}-{end_time}@{cadence}s][{wavelength}]'
        
        logger.info(f"Querying JSOC series: {series}")
        
        try:
            # Query the series
            result = self.drms_client.query(series, key=['T_OBS', 'WAVELNTH', 'EXPTIME', 
                                                       'QUALITY', 'CDELT1', 'CROTA2',
                                                       'RSUN_REF', 'DSUN_OBS'])
            
            logger.info(f"Found {len(result)} records")
            return result.to_dict('records')
            
        except Exception as e:
            logger.error(f"JSOC query failed: {e}")
            return []
    
    def fits_to_jpeg(self, fits_data: np.ndarray, output_size: tuple = (1024, 1024)) -> bytes:
        """Convert FITS data to JPEG bytes"""
        
        # Normalize data to 0-255 range
        data = fits_data.copy()
        
        # Handle bad pixels and extreme values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use percentile-based scaling for better contrast
        p1, p99 = np.percentile(data[data > 0], [1, 99])
        data = np.clip(data, p1, p99)
        
        # Scale to 0-255
        if p99 > p1:
            data = 255 * (data - p1) / (p99 - p1)
        else:
            data = np.zeros_like(data)
        
        # Convert to uint8
        data = data.astype(np.uint8)
        
        # Create PIL Image and resize
        image = Image.fromarray(data)
        image = image.resize(output_size, Image.Resampling.LANCZOS)
        
        # Convert to JPEG bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90, optimize=True)
        
        return buffer.getvalue()
    
    def upload_to_gcs(self, jpeg_data: bytes, gcs_path: str) -> bool:
        """Upload JPEG data to Google Cloud Storage"""
        
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_string(jpeg_data, content_type='image/jpeg')
            logger.debug(f"Uploaded {gcs_path}")
            return True
            
        except Exception as e:
            logger.error(f"GCS upload failed for {gcs_path}: {e}")
            return False
    
    def save_metadata_to_bq(self, metadata_records: list):
        """Save image metadata to BigQuery"""
        
        if not metadata_records:
            return
        
        table_id = f"{self.project_id}.{self.bq_dataset}.solar_images"
        
        try:
            errors = self.bq_client.insert_rows_json(
                self.bq_client.get_table(table_id), 
                metadata_records
            )
            
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.info(f"Inserted {len(metadata_records)} records to BigQuery")
                
        except Exception as e:
            logger.error(f"BigQuery insert failed: {e}")
    
    def process_month(self, year: int, month: int, wavelength: int, 
                     cadence: int = 600, max_images: int = None) -> dict:
        """Process one month of solar data"""
        
        # Calculate start and end times
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        
        start_str = start_date.strftime('%Y.%m.%d_%H:%M:%S_TAI')
        end_str = end_date.strftime('%Y.%m.%d_%H:%M:%S_TAI')
        
        logger.info(f"Processing {year}-{month:02d} for {wavelength}Å")
        logger.info(f"Time range: {start_str} to {end_str}")
        
        # Query JSOC
        records = self.query_jsoc_series(start_str, end_str, wavelength, cadence)
        
        if not records:
            logger.warning(f"No records found for {year}-{month:02d} {wavelength}Å")
            return {"processed": 0, "uploaded": 0, "errors": 0}
        
        # Limit records if specified
        if max_images:
            records = records[:max_images]
        
        processed_count = 0
        uploaded_count = 0
        error_count = 0
        metadata_records = []
        
        for i, record in enumerate(records):
            try:
                # Parse observation time
                obs_time = record.get('T_OBS')
                if not obs_time:
                    continue
                
                # Parse observation time - handle both ISO and JSOC formats
                obs_time = record.get('T_OBS')
                if not obs_time:
                    continue
                
                # Try different datetime formats
                try:
                    if 'T' in obs_time and ('Z' in obs_time or '+' in obs_time):
                        # ISO format: "2012-01-01T00:09:32.84Z"
                        if obs_time.endswith('Z'):
                            obs_time = obs_time[:-1]  # Remove Z
                        # Handle fractional seconds
                        if '.' in obs_time:
                            obs_time = obs_time.split('.')[0]  # Remove fractional seconds
                        obs_datetime = datetime.fromisoformat(obs_time)
                    else:
                        # JSOC format: "2012.01.01_00:09:32_TAI"
                        obs_datetime = datetime.strptime(obs_time.replace('_TAI', ''), '%Y.%m.%d_%H:%M:%S')
                except (ValueError, TypeError) as e:
                    logger.error(f"Error parsing datetime '{obs_time}': {e}")
                    continue
                
                # Generate image ID and GCS path
                image_id = f"aia_{wavelength}_{obs_datetime.strftime('%Y%m%d_%H%M%S')}"
                gcs_path = f"aia/{wavelength}/{year}/{month:02d}/{obs_datetime.strftime('%d')}/{image_id}.jpg"
                
                # Check if already exists
                blob = self.bucket.blob(gcs_path)
                if blob.exists():
                    logger.debug(f"Skipping existing image: {gcs_path}")
                    continue
                
                # Export FITS data from JSOC
                export_request = self.drms_client.export(
                    f'aia.lev1_euv_12s[{obs_time}][{wavelength}]',
                    method='url',
                    protocol='fits'
                )
                
                if not export_request.urls:
                    logger.warning(f"No export URLs for {obs_time}")
                    error_count += 1
                    continue
                
                # Download and process FITS
                fits_url = export_request.urls[0]
                
                with tempfile.NamedTemporaryFile(suffix='.fits') as tmp_file:
                    # Download FITS
                    import requests
                    response = requests.get(fits_url, timeout=60)
                    response.raise_for_status()
                    
                    tmp_file.write(response.content)
                    tmp_file.flush()
                    
                    # Read FITS data
                    with fits.open(tmp_file.name) as hdul:
                        fits_data = hdul[1].data  # AIA data is in HDU 1
                        header = hdul[1].header
                
                # Convert to JPEG
                jpeg_data = self.fits_to_jpeg(fits_data)
                
                # Upload to GCS
                if self.upload_to_gcs(jpeg_data, gcs_path):
                    uploaded_count += 1
                    
                    # Prepare metadata
                    metadata = {
                        "image_id": image_id,
                        "observation_time": obs_datetime.isoformat(),
                        "wavelength": wavelength,
                        "gcs_path": f"gs://{self.bucket_name}/{gcs_path}",
                        "exposure_time": record.get('EXPTIME'),
                        "quality": record.get('QUALITY'),
                        "image_scale": record.get('CDELT1'),
                        "solar_b0": header.get('CRLT_OBS'),
                        "solar_l0": header.get('CRLN_OBS'),
                        "solar_r": record.get('RSUN_REF'),
                        "processing_date": datetime.now().isoformat(),
                        "has_flares": None,  # To be filled later
                        "flare_labels": None
                    }
                    
                    metadata_records.append(metadata)
                    
                    # Batch insert to BigQuery every 50 records
                    if len(metadata_records) >= 50:
                        self.save_metadata_to_bq(metadata_records)
                        metadata_records = []
                
                processed_count += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(records)} images "
                              f"(uploaded: {uploaded_count}, errors: {error_count})")
                
            except Exception as e:
                logger.error(f"Error processing record {i}: {e}")
                error_count += 1
                continue
        
        # Insert remaining metadata
        if metadata_records:
            self.save_metadata_to_bq(metadata_records)
        
        result = {
            "processed": processed_count,
            "uploaded": uploaded_count, 
            "errors": error_count
        }
        
        logger.info(f"Month {year}-{month:02d} {wavelength}Å completed: {result}")
        return result

def main():
    parser = argparse.ArgumentParser(description='Bulk JSOC Solar Data Ingestor')
    parser.add_argument('--project-id', required=True, help='GCP project ID')
    parser.add_argument('--bucket', default='solar-raw', help='GCS bucket name')
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--month', type=int, required=True, help='Month to process')
    parser.add_argument('--wavelength', type=int, required=True, 
                       choices=[94, 131, 171, 193, 211, 304, 335], help='AIA wavelength')
    parser.add_argument('--cadence', type=int, default=600, help='Cadence in seconds')
    parser.add_argument('--max-images', type=int, help='Maximum images to process')
    
    args = parser.parse_args()
    
    # Initialize ingestor
    ingestor = BulkJSOCIngestor(
        project_id=args.project_id,
        bucket_name=args.bucket
    )
    
    # Process the month
    result = ingestor.process_month(
        year=args.year,
        month=args.month, 
        wavelength=args.wavelength,
        cadence=args.cadence,
        max_images=args.max_images
    )
    
    logger.info(f"Final result: {result}")

if __name__ == "__main__":
    main()