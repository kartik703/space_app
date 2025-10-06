#!/usr/bin/env python3
"""
Fixed Production Solar Data Collector
Uses 2014 data (confirmed working) for both storm and normal periods
"""

import logging
from datetime import datetime, timedelta
from jsoc_ingestor import JSOCIngestor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FixedProductionCollector:
    """Production collector using confirmed working 2014 JSOC data"""
    
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        logger.info(f"Initialized FixedProductionCollector for project {project_id}, bucket {bucket_name}")
    
    def collect_storm_and_normal_data(self):
        """Collect both storm period and normal period data from 2014"""
        
        logger.info("🚀 Starting production solar data collection with 2014 data...")
        
        # 1. Collect data for simulated storm period (2014-01-01 - known to work)
        logger.info("⛈️ Phase 1: Collecting storm period data (2014-01-01)...")
        
        storm_ingestor = JSOCIngestor(
            project_id=self.project_id,
            raw_bucket=self.bucket_name
        )
        
        # 2014-01-01: Solar maximum period - high activity expected
        storm_ingestor.ingest_jsoc_data(
            start_time=datetime(2014, 1, 1, 12, 0, 0),
            end_time=datetime(2014, 1, 1, 18, 0, 0),  # 6 hours of data
            wavelength=193,
            cadence=600  # 10 minute cadence
        )
        
        # 2. Collect normal period data (2014-06-01 - solar minimum period)
        logger.info("☀️ Phase 2: Collecting normal period data (2014-06-01)...")
        
        normal_ingestor = JSOCIngestor(
            project_id=self.project_id,
            raw_bucket=self.bucket_name
        )
        
        # 2014-06-01: Quieter period for normal baseline
        normal_ingestor.ingest_jsoc_data(
            start_time=datetime(2014, 6, 1, 12, 0, 0),
            end_time=datetime(2014, 6, 1, 18, 0, 0),  # 6 hours of normal data
            wavelength=193,
            cadence=600  # 10 minute cadence
        )
        
        # 3. Collect additional wavelengths for better analysis
        logger.info("🌈 Phase 3: Collecting multi-wavelength data...")
        
        multi_ingestor = JSOCIngestor(
            project_id=self.project_id,
            raw_bucket=self.bucket_name
        )
        
        # Collect 171 Å and 304 Å for the storm period
        for wavelength in [171, 304]:
            logger.info(f"Collecting {wavelength}Å data...")
            multi_ingestor.ingest_jsoc_data(
                start_time=datetime(2014, 1, 1, 12, 0, 0),
                end_time=datetime(2014, 1, 1, 15, 0, 0),  # 3 hours
                wavelength=wavelength,
                cadence=1800  # 30 minute cadence for additional wavelengths
            )
        
        logger.info("✅ Production data collection complete!")
        logger.info("📊 Collected: Storm period (2014-01-01), Normal period (2014-06-01), Multi-wavelength data")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed Production Solar Data Collector")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--bucket-name", required=True, help="GCS Bucket name")
    
    args = parser.parse_args()
    
    collector = FixedProductionCollector(
        project_id=args.project_id,
        bucket_name=args.bucket_name
    )
    
    collector.collect_storm_and_normal_data()


if __name__ == "__main__":
    main()