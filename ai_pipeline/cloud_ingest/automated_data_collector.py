#!/usr/bin/env python3
"""
Automated Solar Data Collection Script
Collects solar images from multiple months and wavelengths automatically.
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import os
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('automated_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedDataCollector:
    """Automated solar data collection system"""
    
    def __init__(self, base_year=2012, max_images_per_month=30):
        self.base_year = base_year
        self.max_images_per_month = max_images_per_month
        self.script_path = Path(__file__).parent / "cloud_ingest" / "local_jsoc_ingestor.py"
        self.target_wavelengths = [193, 304, 171, 211]  # Multiple EUV channels
        self.cadence_map = {
            193: 1800,  # 30 minutes
            304: 3600,  # 1 hour  
            171: 2400,  # 40 minutes
            211: 3600   # 1 hour
        }
        
    def collect_month_data(self, year, month, wavelength, max_images=None):
        """Collect data for a specific month and wavelength"""
        if max_images is None:
            max_images = self.max_images_per_month
            
        cadence = self.cadence_map.get(wavelength, 3600)
        
        cmd = [
            sys.executable,
            str(self.script_path),
            "--year", str(year),
            "--month", str(month),
            "--wavelength", str(wavelength),
            "--cadence", str(cadence),
            "--max-images", str(max_images)
        ]
        
        logger.info(f"🌞 Collecting {wavelength}Å data for {year}-{month:02d}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info(f"✅ Successfully collected {wavelength}Å data for {year}-{month:02d}")
                return True
            else:
                logger.error(f"❌ Failed to collect {wavelength}Å data for {year}-{month:02d}")
                logger.error(f"Error: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Timeout collecting {wavelength}Å data for {year}-{month:02d}")
            return False
        except Exception as e:
            logger.error(f"💥 Exception collecting {wavelength}Å data for {year}-{month:02d}: {e}")
            return False
    
    def collect_solar_maximum_period(self, start_month=1, end_month=6):
        """Collect data for the 2012 solar maximum period"""
        logger.info(f"🚀 Starting automated data collection for {self.base_year}-{start_month:02d} to {self.base_year}-{end_month:02d}")
        
        total_attempts = 0
        successful_downloads = 0
        
        for month in range(start_month, end_month + 1):
            for wavelength in self.target_wavelengths:
                total_attempts += 1
                
                # Add delay between downloads to be respectful to JSOC
                if total_attempts > 1:
                    logger.info("⏳ Waiting 5 seconds between downloads...")
                    time.sleep(5)
                
                success = self.collect_month_data(self.base_year, month, wavelength)
                if success:
                    successful_downloads += 1
        
        logger.info(f"📊 Collection Summary:")
        logger.info(f"   Total attempts: {total_attempts}")
        logger.info(f"   Successful downloads: {successful_downloads}")
        logger.info(f"   Success rate: {successful_downloads/total_attempts*100:.1f}%")
        
        return successful_downloads, total_attempts
    
    def collect_extended_period(self, months_range):
        """Collect data for extended period with custom months"""
        logger.info(f"🚀 Starting extended data collection for months: {months_range}")
        
        total_attempts = 0
        successful_downloads = 0
        
        for month in months_range:
            for wavelength in self.target_wavelengths:
                total_attempts += 1
                
                # Add delay between downloads
                if total_attempts > 1:
                    logger.info("⏳ Waiting 5 seconds between downloads...")
                    time.sleep(5)
                
                success = self.collect_month_data(self.base_year, month, wavelength)
                if success:
                    successful_downloads += 1
        
        logger.info(f"📊 Extended Collection Summary:")
        logger.info(f"   Total attempts: {total_attempts}")
        logger.info(f"   Successful downloads: {successful_downloads}")
        logger.info(f"   Success rate: {successful_downloads/total_attempts*100:.1f}%")
        
        return successful_downloads, total_attempts
    
    def count_collected_images(self):
        """Count total images collected"""
        image_dir = Path("data/solar_images/2012")
        if not image_dir.exists():
            return 0
        
        jpeg_files = list(image_dir.rglob("*.jpg"))
        logger.info(f"📁 Total images collected: {len(jpeg_files)}")
        return len(jpeg_files)

def main():
    parser = argparse.ArgumentParser(description="Automated Solar Data Collection")
    parser.add_argument("--mode", choices=["standard", "extended", "quick"], default="standard",
                      help="Collection mode: standard (6 months), extended (12 months), quick (2 months)")
    parser.add_argument("--max-images", type=int, default=30,
                      help="Maximum images per month per wavelength")
    parser.add_argument("--start-month", type=int, default=1,
                      help="Starting month for collection")
    parser.add_argument("--end-month", type=int, default=6,
                      help="Ending month for collection")
    
    args = parser.parse_args()
    
    logger.info("🌞 Automated Solar Data Collection System")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Max images per month: {args.max_images}")
    
    collector = AutomatedDataCollector(max_images_per_month=args.max_images)
    
    # Count existing images
    initial_count = collector.count_collected_images()
    logger.info(f"📊 Initial image count: {initial_count}")
    
    # Run collection based on mode
    if args.mode == "standard":
        successful, total = collector.collect_solar_maximum_period(args.start_month, args.end_month)
    elif args.mode == "extended":
        months = list(range(1, 13))  # Full year
        successful, total = collector.collect_extended_period(months)
    elif args.mode == "quick":
        successful, total = collector.collect_solar_maximum_period(1, 2)  # Just Jan-Feb
    
    # Final count
    final_count = collector.count_collected_images()
    new_images = final_count - initial_count
    
    logger.info("🎯 Collection Complete!")
    logger.info(f"   Images before: {initial_count}")
    logger.info(f"   Images after: {final_count}")
    logger.info(f"   New images: {new_images}")
    logger.info(f"   Download success: {successful}/{total}")
    
    if new_images > 0:
        logger.info("✅ Ready for dataset preparation and retraining!")
        return True
    else:
        logger.warning("⚠️ No new images collected. Check logs for issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)