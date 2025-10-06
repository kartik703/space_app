#!/usr/bin/env python3
"""
Massive Solar Data Collector for 100GB+ Dataset
Collects real solar data with proper timeframes for storm prediction.

Strategy:
- Target: 100GB+ of real solar images
- Time period: 2010-2024 (covers multiple solar cycles)
- Wavelengths: 193Å, 304Å, 171Å, 211Å (multi-physics view)
- Storm periods: Pre-storm, during storm, post-storm sequences
- Normal periods: Quiet sun for baseline comparison
- Cadence: High-resolution temporal sequences for prediction
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import sqlite3
import concurrent.futures
from threading import Lock

import requests
import drms
import numpy as np
from astropy.io import fits
import cv2
import pandas as pd


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class MassiveSolarDataCollector:
    """Massive solar data collection for 100GB+ dataset."""
    
    def __init__(self, output_dir: str = "data/massive_solar_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        self.storm_dir = self.output_dir / "storm_periods"
        self.pre_storm_dir = self.output_dir / "pre_storm_periods"
        self.normal_dir = self.output_dir / "normal_periods"
        self.post_storm_dir = self.output_dir / "post_storm_periods"
        
        for dir_path in [self.storm_dir, self.pre_storm_dir, self.normal_dir, self.post_storm_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Database for metadata tracking
        self.db_path = self.output_dir / "massive_dataset_metadata.db"
        self.db_lock = Lock()
        self.init_database()
        
        # Initialize DRMS client
        email = "goswamikartik429@gmail.com"
        try:
            self.drms_client = drms.Client(email=email)
            logger.info(f"DRMS client initialized for massive data collection")
        except Exception as e:
            logger.error(f"Failed to initialize DRMS client: {e}")
            raise
            
        # Data collection statistics
        self.total_downloaded = 0
        self.total_size_bytes = 0
        self.target_size_gb = 100
        
    def init_database(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS solar_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE,
                    observation_time TEXT,
                    wavelength INTEGER,
                    data_type TEXT,  -- storm, pre_storm, normal, post_storm
                    file_size_bytes INTEGER,
                    width INTEGER,
                    height INTEGER,
                    quality_score REAL,
                    storm_event_id TEXT,
                    download_time TEXT,
                    file_path TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    total_images INTEGER,
                    total_size_gb REAL,
                    storm_images INTEGER,
                    pre_storm_images INTEGER,
                    normal_images INTEGER,
                    post_storm_images INTEGER
                )
            """)
            conn.commit()
    
    def get_major_storm_events(self) -> List[Dict]:
        """Get list of major solar storm events for targeted data collection."""
        # Major solar storm events with significant space weather impact
        storm_events = [
            # Solar Cycle 24 (2008-2019)
            {"start": "2012-03-07", "peak": "2012-03-08", "end": "2012-03-10", "class": "X5.4", "description": "Major X-class flare"},
            {"start": "2012-07-12", "peak": "2012-07-12", "end": "2012-07-14", "class": "X1.4", "description": "X-class CME event"},
            {"start": "2013-05-13", "peak": "2013-05-13", "end": "2013-05-15", "class": "X2.8", "description": "Active region flare"},
            {"start": "2014-02-25", "peak": "2014-02-25", "end": "2014-02-27", "class": "X4.9", "description": "Strong X-class flare"},
            {"start": "2014-09-10", "peak": "2014-09-10", "end": "2014-09-12", "class": "X1.6", "description": "Recurring active region"},
            {"start": "2015-03-11", "peak": "2015-03-11", "end": "2015-03-13", "class": "X2.2", "description": "Strong geomagnetic storm"},
            {"start": "2017-09-06", "peak": "2017-09-06", "end": "2017-09-08", "class": "X9.3", "description": "Largest flare of cycle 24"},
            {"start": "2017-09-10", "peak": "2017-09-10", "end": "2017-09-12", "class": "X8.2", "description": "Second major flare"},
            
            # Solar Cycle 25 (2019-present)
            {"start": "2021-10-28", "peak": "2021-10-28", "end": "2021-10-30", "class": "X1.0", "description": "First X-class of cycle 25"},
            {"start": "2022-04-30", "peak": "2022-04-30", "end": "2022-05-02", "class": "X1.1", "description": "Active region 13006"},
            {"start": "2023-02-11", "peak": "2023-02-11", "end": "2023-02-13", "class": "X2.2", "description": "Strong flare event"},
            {"start": "2024-05-08", "peak": "2024-05-08", "end": "2024-05-12", "class": "X2.9", "description": "Recent major storm"},
            
            # Additional significant events for prediction training
            {"start": "2011-02-15", "peak": "2011-02-15", "end": "2011-02-17", "class": "X2.2", "description": "Cycle 24 rising phase"},
            {"start": "2011-08-04", "peak": "2011-08-04", "end": "2011-08-06", "class": "M9.3", "description": "Near X-class event"},
            {"start": "2012-01-23", "peak": "2012-01-23", "end": "2012-01-25", "class": "M8.7", "description": "Strong M-class"},
            {"start": "2013-10-11", "peak": "2013-10-11", "end": "2013-10-13", "class": "M1.5", "description": "Medium activity"},
            {"start": "2016-07-23", "peak": "2016-07-23", "end": "2016-07-25", "class": "M7.6", "description": "Cycle declining"},
            {"start": "2020-05-29", "peak": "2020-05-29", "end": "2020-05-31", "class": "M1.1", "description": "Cycle 25 beginning"},
        ]
        
        return storm_events
    
    def get_normal_periods(self) -> List[Dict]:
        """Get quiet sun periods for baseline data."""
        # Periods of low solar activity for normal/quiet baseline
        normal_periods = [
            {"start": "2008-12-01", "end": "2008-12-31", "description": "Solar minimum"},
            {"start": "2009-06-01", "end": "2009-06-30", "description": "Deep minimum"},
            {"start": "2010-01-01", "end": "2010-01-31", "description": "Quiet period"},
            {"start": "2011-01-01", "end": "2011-01-15", "description": "Low activity"},
            {"start": "2018-01-01", "end": "2018-01-31", "description": "Declining phase"},
            {"start": "2018-12-01", "end": "2018-12-31", "description": "Minimum approaching"},
            {"start": "2019-06-01", "end": "2019-06-30", "description": "Solar minimum"},
            {"start": "2020-01-01", "end": "2020-01-31", "description": "Very quiet"},
            {"start": "2021-01-01", "end": "2021-01-15", "description": "Cycle 25 beginning"},
            {"start": "2023-07-01", "end": "2023-07-15", "description": "Moderate activity"},
        ]
        
        return normal_periods
    
    def calculate_data_requirements(self) -> Dict:
        """Calculate how much data we need to collect."""
        target_gb = self.target_size_gb
        estimated_image_size_mb = 3.0  # Average JPEG size in MB
        total_images_needed = int((target_gb * 1024) / estimated_image_size_mb)
        
        # Distribution strategy
        storm_images = int(total_images_needed * 0.30)  # 30% storm events
        pre_storm_images = int(total_images_needed * 0.25)  # 25% pre-storm
        post_storm_images = int(total_images_needed * 0.20)  # 20% post-storm
        normal_images = int(total_images_needed * 0.25)  # 25% normal/quiet
        
        requirements = {
            'total_images': total_images_needed,
            'storm_images': storm_images,
            'pre_storm_images': pre_storm_images,
            'post_storm_images': post_storm_images,
            'normal_images': normal_images,
            'target_gb': target_gb,
            'wavelengths': [193, 304, 171, 211],  # Multi-physics view
            'cadence_minutes': [12, 24, 60]  # Different temporal resolutions
        }
        
        logger.info(f"Data Collection Requirements:")
        logger.info(f"  Total images needed: {total_images_needed:,}")
        logger.info(f"  Storm period images: {storm_images:,}")
        logger.info(f"  Pre-storm images: {pre_storm_images:,}")
        logger.info(f"  Post-storm images: {post_storm_images:,}")
        logger.info(f"  Normal period images: {normal_images:,}")
        logger.info(f"  Target size: {target_gb}GB")
        
        return requirements
    
    def collect_storm_period_data(self, storm_event: Dict, wavelengths: List[int], 
                                 images_per_event: int) -> int:
        """Collect data for a specific storm event."""
        logger.info(f"Collecting storm data for {storm_event['description']} ({storm_event['class']})")
        
        start_time = datetime.strptime(storm_event['start'], '%Y-%m-%d')
        end_time = datetime.strptime(storm_event['end'], '%Y-%m-%d')
        
        collected_count = 0
        
        for wavelength in wavelengths:
            # During storm (peak activity)
            storm_count = self.collect_period_data(
                start_time, end_time, wavelength, 
                cadence_minutes=12,  # High cadence during storm
                max_images=images_per_event // len(wavelengths),
                data_type="storm",
                event_id=storm_event['class']
            )
            
            # Pre-storm (24 hours before)
            pre_start = start_time - timedelta(days=1)
            pre_count = self.collect_period_data(
                pre_start, start_time, wavelength,
                cadence_minutes=24,  # Medium cadence pre-storm
                max_images=images_per_event // (len(wavelengths) * 2),
                data_type="pre_storm",
                event_id=storm_event['class']
            )
            
            # Post-storm (24 hours after)
            post_end = end_time + timedelta(days=1)
            post_count = self.collect_period_data(
                end_time, post_end, wavelength,
                cadence_minutes=60,  # Lower cadence post-storm
                max_images=images_per_event // (len(wavelengths) * 2),
                data_type="post_storm",
                event_id=storm_event['class']
            )
            
            collected_count += storm_count + pre_count + post_count
            
        return collected_count
    
    def collect_period_data(self, start_time: datetime, end_time: datetime, 
                           wavelength: int, cadence_minutes: int, max_images: int,
                           data_type: str, event_id: str = None) -> int:
        """Collect data for a specific time period."""
        
        # Build DRMS query
        time_str = f"{start_time.strftime('%Y.%m.%d_%H:%M:%S')}_TAI-{end_time.strftime('%Y.%m.%d_%H:%M:%S')}_TAI"
        query = f"aia.lev1_euv_12s[{time_str}][{wavelength}]"
        
        try:
            # Query available data
            logger.info(f"Querying JSOC for {data_type} data: {wavelength}Å, {start_time} to {end_time}")
            
            # Use cadence to sample the data
            cadence_str = f"{cadence_minutes}m"
            query_with_cadence = f"aia.lev1_euv_12s[{time_str}@{cadence_str}][{wavelength}]"
            
            result = self.drms_client.query(query_with_cadence, key='T_REC, QUALITY, EXPTIME')
            
            if result.empty:
                logger.warning(f"No data found for {data_type} period {start_time} - {end_time}")
                return 0
                
            # Limit to max_images
            if len(result) > max_images:
                result = result.head(max_images)
                
            logger.info(f"Found {len(result)} images for {data_type} collection")
            
            # Download images
            downloaded_count = 0
            
            for idx, row in result.iterrows():
                try:
                    # Request export
                    export_request = self.drms_client.export(
                        query_with_cadence + f"[{idx}]",
                        method='url_quick',
                        protocol='fits'
                    )
                    
                    if export_request.status == 0:  # Success
                        for url in export_request.urls:
                            if self.download_and_process_image(url, wavelength, data_type, event_id, row):
                                downloaded_count += 1
                                
                                # Check if we've reached our target
                                if self.total_size_bytes >= (self.target_size_gb * 1024 * 1024 * 1024):
                                    logger.info(f"🎯 TARGET REACHED: {self.target_size_gb}GB collected!")
                                    return downloaded_count
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error downloading image {idx}: {e}")
                    continue
                    
            return downloaded_count
            
        except Exception as e:
            logger.error(f"Error in period data collection: {e}")
            return 0
    
    def download_and_process_image(self, url: str, wavelength: int, data_type: str, 
                                  event_id: str, metadata_row) -> bool:
        """Download and process a single FITS image."""
        try:
            # Download FITS file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Load FITS data
            fits_data = fits.open(response.content)
            image_data = fits_data[1].data if len(fits_data) > 1 else fits_data[0].data
            header = fits_data[1].header if len(fits_data) > 1 else fits_data[0].header
            fits_data.close()
            
            if image_data is None:
                return False
                
            # Generate filename
            obs_time = header.get('T_REC', datetime.now().isoformat())
            clean_time = obs_time.replace(':', '-').replace('_', '-')
            filename = f"{data_type}_{wavelength}A_{clean_time}.jpg"
            
            # Determine output directory
            if data_type == "storm":
                output_path = self.storm_dir / filename
            elif data_type == "pre_storm":
                output_path = self.pre_storm_dir / filename
            elif data_type == "post_storm":
                output_path = self.post_storm_dir / filename
            else:
                output_path = self.normal_dir / filename
                
            # Process and save image
            processed_image = self.process_solar_image(image_data)
            
            # Save as JPEG
            cv2.imwrite(str(output_path), processed_image)
            
            # Get file size
            file_size = output_path.stat().st_size
            self.total_size_bytes += file_size
            self.total_downloaded += 1
            
            # Save metadata
            self.save_image_metadata(
                filename=filename,
                observation_time=obs_time,
                wavelength=wavelength,
                data_type=data_type,
                file_size_bytes=file_size,
                width=processed_image.shape[1],
                height=processed_image.shape[0],
                storm_event_id=event_id,
                file_path=str(output_path)
            )
            
            # Progress logging
            if self.total_downloaded % 100 == 0:
                size_gb = self.total_size_bytes / (1024**3)
                logger.info(f"Progress: {self.total_downloaded:,} images, {size_gb:.2f}GB collected")
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing image from {url}: {e}")
            return False
    
    def process_solar_image(self, fits_data: np.ndarray) -> np.ndarray:
        """Process FITS data into JPEG-ready format."""
        # Handle NaN values
        fits_data = np.nan_to_num(fits_data, nan=0.0)
        
        # Robust scaling
        p1, p99 = np.percentile(fits_data[fits_data > 0], [1, 99])
        fits_data = np.clip(fits_data, p1, p99)
        
        # Normalize to 0-255
        if p99 > p1:
            fits_data = ((fits_data - p1) / (p99 - p1) * 255).astype(np.uint8)
        else:
            fits_data = np.zeros_like(fits_data, dtype=np.uint8)
            
        # Resize to standard size for consistency
        fits_data = cv2.resize(fits_data, (1024, 1024))
        
        return fits_data
    
    def save_image_metadata(self, **kwargs):
        """Save image metadata to database."""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                kwargs['download_time'] = datetime.now().isoformat()
                
                columns = ', '.join(kwargs.keys())
                placeholders = ', '.join(['?' for _ in kwargs])
                query = f"INSERT OR REPLACE INTO solar_images ({columns}) VALUES ({placeholders})"
                
                conn.execute(query, list(kwargs.values()))
                conn.commit()
    
    def update_collection_stats(self):
        """Update collection statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Count images by type
            stats = conn.execute("""
                SELECT 
                    data_type,
                    COUNT(*) as count,
                    SUM(file_size_bytes) as total_size
                FROM solar_images 
                GROUP BY data_type
            """).fetchall()
            
            storm_count = pre_storm_count = normal_count = post_storm_count = 0
            
            for data_type, count, size in stats:
                if data_type == "storm":
                    storm_count = count
                elif data_type == "pre_storm":
                    pre_storm_count = count
                elif data_type == "post_storm":
                    post_storm_count = count
                elif data_type == "normal":
                    normal_count = count
            
            total_size_gb = self.total_size_bytes / (1024**3)
            
            conn.execute("""
                INSERT INTO collection_stats 
                (timestamp, total_images, total_size_gb, storm_images, pre_storm_images, normal_images, post_storm_images)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                self.total_downloaded,
                total_size_gb,
                storm_count,
                pre_storm_count,
                normal_count,
                post_storm_count
            ))
            conn.commit()
    
    def run_massive_collection(self):
        """Run the massive data collection process."""
        logger.info("🚀 Starting Massive Solar Data Collection for 100GB+ Dataset")
        
        # Calculate requirements
        requirements = self.calculate_data_requirements()
        
        # Get storm events and normal periods
        storm_events = self.get_major_storm_events()
        normal_periods = self.get_normal_periods()
        
        logger.info(f"📊 Collection Plan:")
        logger.info(f"  {len(storm_events)} storm events to collect")
        logger.info(f"  {len(normal_periods)} normal periods to collect")
        logger.info(f"  Wavelengths: {requirements['wavelengths']}")
        
        # Start collection
        start_time = time.time()
        
        # Collect storm period data
        images_per_storm = requirements['storm_images'] // len(storm_events)
        logger.info(f"🌪️ Collecting storm data ({images_per_storm} images per event)...")
        
        for storm_event in storm_events:
            if self.total_size_bytes >= (self.target_size_gb * 1024 * 1024 * 1024):
                break
                
            collected = self.collect_storm_period_data(
                storm_event, 
                requirements['wavelengths'], 
                images_per_storm
            )
            logger.info(f"Collected {collected} images for {storm_event['description']}")
        
        # Collect normal period data if we haven't reached target
        if self.total_size_bytes < (self.target_size_gb * 1024 * 1024 * 1024):
            logger.info(f"🌞 Collecting normal/quiet period data...")
            images_per_normal = requirements['normal_images'] // len(normal_periods)
            
            for normal_period in normal_periods:
                if self.total_size_bytes >= (self.target_size_gb * 1024 * 1024 * 1024):
                    break
                    
                start_time_period = datetime.strptime(normal_period['start'], '%Y-%m-%d')
                end_time_period = datetime.strptime(normal_period['end'], '%Y-%m-%d')
                
                for wavelength in requirements['wavelengths']:
                    collected = self.collect_period_data(
                        start_time_period, end_time_period, wavelength,
                        cadence_minutes=60,  # Lower cadence for normal periods
                        max_images=images_per_normal // len(requirements['wavelengths']),
                        data_type="normal"
                    )
        
        # Final statistics
        end_time = time.time()
        duration_hours = (end_time - start_time) / 3600
        final_size_gb = self.total_size_bytes / (1024**3)
        
        logger.info("🎉 MASSIVE DATA COLLECTION COMPLETED!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"  Total images collected: {self.total_downloaded:,}")
        logger.info(f"  Total data size: {final_size_gb:.2f} GB")
        logger.info(f"  Collection time: {duration_hours:.1f} hours")
        logger.info(f"  Average speed: {self.total_downloaded/duration_hours:.0f} images/hour")
        
        # Update final stats
        self.update_collection_stats()
        
        return {
            'total_images': self.total_downloaded,
            'total_size_gb': final_size_gb,
            'duration_hours': duration_hours,
            'target_reached': final_size_gb >= self.target_size_gb
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Collect 100GB+ of real solar data for storm prediction")
    parser.add_argument("--target-gb", type=int, default=100, help="Target dataset size in GB")
    parser.add_argument("--output-dir", default="data/massive_solar_dataset", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel workers")
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = MassiveSolarDataCollector(output_dir=args.output_dir)
    collector.target_size_gb = args.target_gb
    
    try:
        # Run massive collection
        results = collector.run_massive_collection()
        
        # Print final summary
        print("\n" + "="*60)
        print("🌞 MASSIVE SOLAR DATA COLLECTION SUMMARY")
        print("="*60)
        print(f"Images Collected: {results['total_images']:,}")
        print(f"Dataset Size: {results['total_size_gb']:.2f} GB")
        print(f"Target Reached: {'✅ YES' if results['target_reached'] else '❌ NO'}")
        print(f"Collection Time: {results['duration_hours']:.1f} hours")
        print("="*60)
        
        if results['target_reached']:
            print("🎯 SUCCESS: Ready for YOLO training on 100GB+ real solar data!")
        else:
            print("⚠️  Partial collection completed. Consider running again to reach target.")
            
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()