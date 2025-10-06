#!/usr/bin/env python3
"""
Local JSOC Solar Data Ingestor
Downloads solar images from JSOC and saves locally when GCS billing is disabled.
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sqlite3

import requests
import drms
import numpy as np
from astropy.io import fits
import cv2


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class LocalJSOCIngestor:
    """Local JSOC data ingestion class."""
    
    def __init__(self, output_dir: str = "data/solar_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata database
        self.db_path = self.output_dir / "metadata.db"
        self.init_database()
        
        # Initialize DRMS client with email for exports
        email = "goswamikartik429@gmail.com"  # Required for JSOC exports
        try:
            self.drms_client = drms.Client(email=email)
            logger.info(f"DRMS client initialized successfully with email: {email}")
        except Exception as e:
            logger.error(f"Failed to initialize DRMS client: {e}")
            raise
    
    def init_database(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS solar_images (
                    image_id TEXT PRIMARY KEY,
                    observation_time TEXT,
                    wavelength INTEGER,
                    year INTEGER,
                    month INTEGER,
                    day INTEGER,
                    local_path TEXT,
                    file_size INTEGER,
                    solar_x0 REAL,
                    solar_y0 REAL,
                    solar_radius REAL,
                    solar_b0 REAL,
                    solar_l0 REAL,
                    solar_p REAL,
                    exposure_time REAL,
                    created_at TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            conn.commit()
    
    def insert_metadata(self, metadata: Dict[str, Any]):
        """Insert metadata into database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO solar_images 
                (image_id, observation_time, wavelength, year, month, day,
                 local_path, file_size, solar_x0, solar_y0, solar_radius,
                 solar_b0, solar_l0, solar_p, exposure_time, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata['image_id'], metadata['observation_time'], metadata['wavelength'],
                metadata['year'], metadata['month'], metadata['day'],
                metadata['local_path'], metadata['file_size'], metadata['solar_x0'],
                metadata['solar_y0'], metadata['solar_radius'], metadata['solar_b0'],
                metadata['solar_l0'], metadata['solar_p'], metadata['exposure_time'],
                metadata['created_at'], metadata['status']
            ))
            conn.commit()
    
    def query_month_data(self, year: int, month: int, wavelength: int, cadence: int = 600):
        """Query JSOC for a month of data."""
        start_time = f"{year}.{month:02d}.01_00:00:00_TAI"
        end_time = f"{year}.{month:02d}.{28 if month == 2 else 31}_23:59:59_TAI"
        
        series = f"aia.lev1_euv_12s[{start_time}-{end_time}@{cadence}s][{wavelength}]"
        
        logger.info(f"Processing {year}-{month:02d} for {wavelength}Å")
        logger.info(f"Time range: {start_time} to {end_time}")
        logger.info(f"Querying JSOC series: {series}")
        
        try:
            result = self.drms_client.query(series, key=['T_OBS', 'WAVELNTH', 'EXPTIME', 
                                                       'CRPIX1', 'CRPIX2', 'RSUN_OBS',
                                                       'CRLT_OBS', 'CRLN_OBS', 'CROTA2'])
            
            if result.empty:
                logger.warning(f"No data found for {year}-{month:02d} {wavelength}Å")
                return []
            
            logger.info(f"Found {len(result)} records")
            return result.to_dict('records')
            
        except Exception as e:
            logger.error(f"JSOC query failed: {e}")
            return []
    
    def process_records(self, records: list, year: int, month: int, wavelength: int, max_images: Optional[int] = None):
        """Process JSOC records and download images."""
        stats = {'processed': 0, 'uploaded': 0, 'errors': 0}
        
        if max_images:
            records = records[:max_images]
            logger.info(f"Limited to {max_images} images for testing")
        
        # Create month directory
        month_dir = self.output_dir / f"{year}" / f"{month:02d}" / f"{wavelength}"
        month_dir.mkdir(parents=True, exist_ok=True)
        
        for i, record in enumerate(records):
            try:
                # Parse observation time
                obs_time = record.get('T_OBS')
                if not obs_time:
                    continue
                
                # Handle different datetime formats
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
                    stats['errors'] += 1
                    continue
                
                # Generate image ID and local path
                image_id = f"aia_{wavelength}_{obs_datetime.strftime('%Y%m%d_%H%M%S')}"
                local_path = month_dir / f"{image_id}.jpg"
                
                # Check if already exists
                if local_path.exists():
                    logger.debug(f"Image already exists: {local_path}")
                    stats['processed'] += 1
                    continue
                
                # Export FITS data from JSOC
                export_request = self.drms_client.export(f"aia.lev1_euv_12s[{record['T_OBS']}][{wavelength}]")
                
                if len(export_request.urls) == 0:
                    logger.warning(f"No export URLs for record {i}")
                    stats['errors'] += 1
                    continue
                
                # Download and process first URL
                fits_url = export_request.urls.url.iloc[0]  # Extract URL from pandas Series
                logger.info(f"Downloading: {fits_url}")
                
                response = requests.get(fits_url, timeout=120)
                response.raise_for_status()
                
                # Process FITS data
                fits_data, header = self.process_fits_data(response.content)
                
                if fits_data is None:
                    logger.error(f"Failed to process FITS data for record {i}")
                    stats['errors'] += 1
                    continue
                
                # Convert to JPEG and save
                jpeg_data = self.fits_to_jpeg(fits_data)
                
                with open(local_path, 'wb') as f:
                    f.write(jpeg_data)
                
                file_size = len(jpeg_data)
                
                # Save metadata
                metadata = {
                    'image_id': image_id,
                    'observation_time': obs_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'wavelength': wavelength,
                    'year': year,
                    'month': month,
                    'day': obs_datetime.day,
                    'local_path': str(local_path),
                    'file_size': file_size,
                    'solar_x0': header.get('CRPIX1', 0),
                    'solar_y0': header.get('CRPIX2', 0),
                    'solar_radius': header.get('RSUN_OBS', 0),
                    'solar_b0': header.get('CRLT_OBS', 0),
                    'solar_l0': header.get('CRLN_OBS', 0),
                    'solar_p': header.get('CROTA2', 0),
                    'exposure_time': header.get('EXPTIME', 0),
                    'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'active'
                }
                
                self.insert_metadata(metadata)
                
                stats['uploaded'] += 1
                stats['processed'] += 1
                
                logger.info(f"Processed {i+1}/{len(records)}: {image_id}")
                
                # Brief pause to avoid overwhelming JSOC
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing record {i}: {e}")
                stats['errors'] += 1
                continue
        
        return stats
    
    def process_fits_data(self, fits_content: bytes):
        """Process FITS file content and extract data and header."""
        try:
            from io import BytesIO
            
            with fits.open(BytesIO(fits_content)) as hdul:
                # AIA data is typically in HDU 1
                fits_data = hdul[1].data
                header = hdul[1].header
                
                return fits_data, header
                
        except Exception as e:
            logger.error(f"Error processing FITS data: {e}")
            return None, {}
    
    def fits_to_jpeg(self, fits_data: np.ndarray, quality: int = 90) -> bytes:
        """Convert FITS data to JPEG format."""
        try:
            # Handle different data types and normalize
            if fits_data.dtype in [np.float32, np.float64]:
                # Clip extreme values and normalize to 0-255
                data_clipped = np.clip(fits_data, np.percentile(fits_data, 1), 
                                     np.percentile(fits_data, 99))
                data_normalized = ((data_clipped - data_clipped.min()) / 
                                 (data_clipped.max() - data_clipped.min()) * 255).astype(np.uint8)
            else:
                # For integer data, scale to 0-255 range
                data_normalized = ((fits_data.astype(np.float32) / fits_data.max()) * 255).astype(np.uint8)
            
            # Apply gamma correction for better visibility
            gamma = 0.5
            data_gamma = np.power(data_normalized / 255.0, gamma) * 255
            data_gamma = data_gamma.astype(np.uint8)
            
            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, encoded_img = cv2.imencode('.jpg', data_gamma, encode_param)
            
            if not success:
                raise ValueError("Failed to encode image as JPEG")
            
            return encoded_img.tobytes()
            
        except Exception as e:
            logger.error(f"Error converting FITS to JPEG: {e}")
            raise
    
    def ingest_month(self, year: int, month: int, wavelength: int, cadence: int = 600, max_images: Optional[int] = None):
        """Ingest a full month of data."""
        logger.info(f"Starting ingestion for {year}-{month:02d} wavelength {wavelength}Å")
        
        # Query JSOC for the month
        records = self.query_month_data(year, month, wavelength, cadence)
        
        if not records:
            logger.warning(f"No records found for {year}-{month:02d} {wavelength}Å")
            return {'processed': 0, 'uploaded': 0, 'errors': 0}
        
        # Process the records
        stats = self.process_records(records, year, month, wavelength, max_images)
        
        logger.info(f"Month {year}-{month:02d} {wavelength}Å completed: {stats}")
        return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Local JSOC Solar Data Ingestor')
    parser.add_argument('--output-dir', default='data/solar_images', 
                       help='Local output directory')
    parser.add_argument('--year', type=int, required=True, 
                       help='Year to download (e.g., 2012)')
    parser.add_argument('--month', type=int, required=True, 
                       help='Month to download (1-12)')
    parser.add_argument('--wavelength', type=int, required=True,
                       choices=[94, 131, 171, 193, 211, 304, 335],
                       help='AIA wavelength channel')
    parser.add_argument('--cadence', type=int, default=600,
                       help='Cadence in seconds (default: 600)')
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to download (for testing)')
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging()
    
    try:
        # Initialize ingestor
        ingestor = LocalJSOCIngestor(args.output_dir)
        
        # Ingest the month
        result = ingestor.ingest_month(
            args.year, args.month, args.wavelength, 
            args.cadence, args.max_images
        )
        
        logger.info(f"Final result: {result}")
        
        if result['uploaded'] > 0:
            logger.info("✅ Local ingestion completed successfully!")
            
            # Show summary
            db_path = Path(args.output_dir) / "metadata.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT wavelength, COUNT(*) as count, 
                           MIN(observation_time) as start_time,
                           MAX(observation_time) as end_time,
                           SUM(file_size) as total_size
                    FROM solar_images 
                    WHERE year = ? AND month = ?
                    GROUP BY wavelength
                """, (args.year, args.month))
                
                results = cursor.fetchall()
                for row in results:
                    wl, count, start, end, size = row
                    logger.info(f"Wavelength {wl}Å: {count} images, {start} to {end}, {size/1024/1024:.1f} MB")
        else:
            logger.warning("No images were successfully processed")
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()