#!/usr/bin/env python3
"""
JSOC Solar Data Ingestor for Real SDO AIA Images
Fetches science-grade EUV images from Stanford JSOC using drms client
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
from pathlib import Path
import json
import time
import warnings

# Scientific libraries
import drms
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import cv2
from PIL import Image
import sunpy.map

# Google Cloud
from google.cloud import storage, bigquery
from google.cloud.exceptions import NotFound

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from gcs_ingestor import SolarImageIngestor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress astropy warnings
warnings.filterwarnings('ignore', category=UserWarning, module='astropy')

class JSOCIngestor(SolarImageIngestor):
    """Real JSOC data ingestor for SDO AIA images"""
    
    def __init__(self, project_id: str, raw_bucket: str = "solar-raw", 
                 jsoc_email: str = None):
        super().__init__(project_id, raw_bucket)
        
        # Initialize JSOC client with user's email
        self.jsoc_email = jsoc_email or os.environ.get("JSOC_EMAIL") or "goswamikartik429@gmail.com"
        if not self.jsoc_email:
            logger.warning("JSOC_EMAIL not provided. Some features may be limited.")
        
        try:
            self.drms_client = drms.Client(email=self.jsoc_email)
            logger.info("Successfully connected to JSOC/DRMS")
        except Exception as e:
            logger.error(f"Failed to connect to JSOC: {e}")
            self.drms_client = None
        
        # AIA wavelengths and their properties
        self.aia_wavelengths = {
            94: {"temp": "6.3 MK", "primary": "flare plasma"},
            131: {"temp": "0.4/11 MK", "primary": "transition region/flare"},
            171: {"temp": "0.6 MK", "primary": "quiet corona/upper transition"},
            193: {"temp": "1.2/20 MK", "primary": "corona/flare plasma"},
            211: {"temp": "2.0 MK", "primary": "active regions"},
            304: {"temp": "0.05 MK", "primary": "chromosphere/transition"},
            335: {"temp": "2.5 MK", "primary": "active regions"}
        }
        
        logger.info(f"Initialized JSOC ingestor with {len(self.aia_wavelengths)} wavelengths")
    
    def query_aia_data(self, start_time: datetime, end_time: datetime, 
                       wavelength: int = 193, cadence: int = 600) -> List[Dict]:
        """Query AIA data from JSOC"""
        if not self.drms_client:
            logger.error("DRMS client not available")
            return []
        
        # Convert to JSOC time format
        start_str = start_time.strftime("%Y.%m.%d_%H:%M:%S_TAI")
        end_str = end_time.strftime("%Y.%m.%d_%H:%M:%S_TAI")
        
        # JSOC series for AIA Level 1.5 data (processed, suitable for science)
        series = "aia.lev1_euv_12s"
        
        # Build query
        query = f"{series}[{start_str}-{end_str}@{cadence}s][{wavelength}]"
        
        try:
            logger.info(f"Querying JSOC: {query}")
            
            # Query metadata first - only request fields that work reliably
            keys = self.drms_client.query(query, key='T_REC,WAVELNTH,EXPTIME,QUALITY')
            
            if keys.empty:
                logger.warning(f"No data found for query: {query}")
                return []
            
            logger.info(f"Found {len(keys)} AIA images")
            
            # Convert to list of dicts with error handling
            results = []
            for idx, row in keys.iterrows():
                try:
                    # Skip rows with invalid data
                    if 'Invalid' in str(row['WAVELNTH']) or 'Invalid' in str(row['QUALITY']):
                        logger.warning(f"Skipping row {idx} with invalid KeyLink data")
                        continue
                    
                    results.append({
                        'time_rec': row['T_REC'],
                        'wavelength': int(row['WAVELNTH']),
                        'exptime': float(row['EXPTIME']),
                        'quality': int(row['QUALITY']),
                        'series': series,
                        'jsoc_query': query
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping row {idx} due to data conversion error: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"JSOC query failed: {e}")
            return []
    
    def download_aia_image(self, metadata: Dict, output_dir: str = None) -> Optional[str]:
        """Download AIA FITS file from JSOC"""
        if not self.drms_client:
            return None
        
        try:
            # Create query for this specific image
            time_rec = metadata['time_rec']
            wavelength = metadata['wavelength']
            series = metadata['series']
            
            query = f"{series}[{time_rec}][{wavelength}]"
            
            # Request export
            logger.info(f"Requesting FITS export for {time_rec} {wavelength}Å")
            
            export_request = self.drms_client.export(query, method='url', protocol='fits')
            
            if not export_request.has_succeeded():
                logger.error(f"Export request failed for {query}")
                return None
            
            # Download URLs
            download_urls = export_request.urls
            if not download_urls:
                logger.error(f"No download URLs for {query}")
                return None
            
            # Create output directory
            if output_dir is None:
                output_dir = os.path.join("temp_fits")
            os.makedirs(output_dir, exist_ok=True)
            
            # Download first (and usually only) file
            url = download_urls.url.iloc[0]
            filename = download_urls.filename.iloc[0]
            
            output_path = os.path.join(output_dir, filename)
            
            logger.info(f"Downloading {filename} from JSOC...")
            
            # Use drms download functionality
            export_request.download(output_dir)
            
            if os.path.exists(output_path):
                logger.info(f"Successfully downloaded: {output_path}")
                return output_path
            else:
                logger.error(f"Download failed: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading AIA image: {e}")
            return None
    
    def fits_to_jpeg(self, fits_path: str, output_path: str = None, 
                     target_size: Tuple[int, int] = (1024, 1024)) -> Optional[str]:
        """Convert FITS image to JPEG for YOLO training"""
        try:
            # Read FITS file
            with fits.open(fits_path) as hdul:
                image_data = hdul[1].data  # AIA data is in extension 1
                header = hdul[1].header
            
            # Handle NaN values
            image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize to 8-bit
            # Use log scaling for solar data (high dynamic range)
            image_data = np.where(image_data > 0, image_data, 1)  # Avoid log(0)
            log_data = np.log10(image_data)
            
            # Scale to 0-255
            vmin, vmax = np.percentile(log_data[log_data > -np.inf], [1, 99])
            normalized = np.clip((log_data - vmin) / (vmax - vmin) * 255, 0, 255)
            image_8bit = normalized.astype(np.uint8)
            
            # Resize if needed
            if image_8bit.shape != target_size:
                image_8bit = cv2.resize(image_8bit, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to RGB (grayscale -> RGB for YOLO)
            image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
            
            # Create output path
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(fits_path))[0]
                output_path = f"{base_name}.jpg"
            
            # Save as JPEG
            image_pil = Image.fromarray(image_rgb)
            image_pil.save(output_path, "JPEG", quality=95)
            
            logger.info(f"Converted FITS to JPEG: {output_path}")
            
            # Add metadata
            metadata = {
                'fits_file': fits_path,
                'jpeg_file': output_path,
                'original_shape': image_data.shape,
                'target_shape': target_size,
                'vmin': float(vmin),
                'vmax': float(vmax),
                'header_info': {
                    'date_obs': header.get('DATE-OBS', ''),
                    'wavelnth': header.get('WAVELNTH', 0),
                    'exptime': header.get('EXPTIME', 0),
                    'cdelt1': header.get('CDELT1', 0),
                    'cdelt2': header.get('CDELT2', 0)
                }
            }
            
            return output_path, metadata
            
        except Exception as e:
            logger.error(f"Error converting FITS to JPEG: {e}")
            return None
    
    def process_aia_image(self, metadata: Dict, timestamp: datetime) -> Optional[Dict]:
        """Download, process, and prepare AIA image for upload"""
        try:
            # Download FITS file
            fits_path = self.download_aia_image(metadata)
            if not fits_path:
                return None
            
            # Convert to JPEG
            result = self.fits_to_jpeg(fits_path)
            if not result:
                return None
            
            jpeg_path, conversion_metadata = result
            
            # Read JPEG data for upload
            with open(jpeg_path, 'rb') as f:
                jpeg_data = f.read()
            
            # Clean up temporary files
            try:
                os.remove(fits_path)
                os.remove(jpeg_path)
            except:
                pass
            
            # Create image data dict compatible with existing ingestor
            image_data = {
                "image_data": jpeg_data,
                "timestamp": timestamp,
                "instrument": "AIA",
                "wavelength": metadata['wavelength'],
                "file_size": len(jpeg_data),
                "width": 1024,
                "height": 1024,
                "metadata": {
                    "source": "JSOC/SDO",
                    "series": metadata['series'],
                    "time_rec": metadata['time_rec'],
                    "quality": metadata['quality'],
                    "exptime": metadata['exptime'],
                    "conversion": conversion_metadata['header_info'],
                    "processing": {
                        "log_scaling": True,
                        "vmin": conversion_metadata['vmin'],
                        "vmax": conversion_metadata['vmax'],
                        "target_size": [1024, 1024]
                    }
                }
            }
            
            return image_data
            
        except Exception as e:
            logger.error(f"Error processing AIA image: {e}")
            return None
    
    def ingest_jsoc_data(self, start_time: datetime, end_time: datetime,
                        wavelength: int = 193, cadence: int = 600):
        """Ingest real JSOC data for specified time range"""
        
        logger.info(f"Starting JSOC ingestion from {start_time} to {end_time}")
        logger.info(f"Wavelength: {wavelength}Å, Cadence: {cadence}s")
        
        if wavelength in self.aia_wavelengths:
            props = self.aia_wavelengths[wavelength]
            logger.info(f"AIA {wavelength}Å: {props['temp']}, {props['primary']}")
        
        # Query JSOC for metadata
        metadata_list = self.query_aia_data(start_time, end_time, wavelength, cadence)
        
        if not metadata_list:
            logger.error("No data found in JSOC query")
            return
        
        logger.info(f"Processing {len(metadata_list)} AIA images...")
        
        total_processed = 0
        successful_uploads = 0
        
        for metadata in metadata_list:
            try:
                # Convert T_REC to datetime - handle both TAI and ISO formats
                time_rec_str = metadata['time_rec']
                try:
                    # Try TAI format first
                    timestamp = datetime.strptime(time_rec_str, "%Y.%m.%d_%H:%M:%S_TAI")
                except ValueError:
                    try:
                        # Try ISO format (2014-01-01T11:59:25Z)
                        timestamp = datetime.strptime(time_rec_str, "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        # Try ISO format without Z
                        timestamp = datetime.strptime(time_rec_str, "%Y-%m-%dT%H:%M:%S")
                
                logger.info(f"Processing {time_rec_str} ({wavelength}Å)")
                
                # Download and process image
                image_data = self.process_aia_image(metadata, timestamp)
                
                if image_data:
                    # Upload to GCS using parent class method
                    gcs_path = self.upload_to_gcs(image_data)
                    
                    # Log to BigQuery
                    self.log_to_bigquery(image_data, gcs_path)
                    
                    successful_uploads += 1
                else:
                    logger.warning(f"Failed to process {time_rec_str}")
                
            except Exception as e:
                logger.error(f"Error processing {metadata.get('time_rec', 'unknown')}: {e}")
            
            total_processed += 1
            
            # Small delay to avoid overwhelming JSOC
            time.sleep(1)
        
        logger.info(f"JSOC ingestion complete. Processed {total_processed}, "
                   f"successful uploads: {successful_uploads}")


def main():
    parser = argparse.ArgumentParser(description="Ingest real SDO AIA images from JSOC")
    parser.add_argument("--start", required=True, help="Start time (ISO format: 2012-03-04T00:00:00)")
    parser.add_argument("--end", required=True, help="End time (ISO format: 2012-03-04T23:59:59)")
    parser.add_argument("--project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT"), 
                       help="Google Cloud Project ID")
    parser.add_argument("--bucket", default="solar-raw", help="GCS bucket for raw images")
    parser.add_argument("--wavelength", type=int, default=193, 
                       choices=[94, 131, 171, 193, 211, 304, 335],
                       help="AIA wavelength (default: 193)")
    parser.add_argument("--cadence", type=int, default=600, 
                       help="Cadence in seconds (default: 600 = 10 minutes)")
    parser.add_argument("--jsoc-email", help="JSOC registered email address (default: goswamikartik429@gmail.com)")
    
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
    ingestor = JSOCIngestor(args.project_id, args.bucket, args.jsoc_email)
    ingestor.ingest_jsoc_data(
        start_time=start_time,
        end_time=end_time,
        wavelength=args.wavelength,
        cadence=args.cadence
    )
    
    return 0


if __name__ == "__main__":
    exit(main())