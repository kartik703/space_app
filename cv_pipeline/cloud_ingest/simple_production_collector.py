#!/usr/bin/env python3
"""
Simple Production Solar Data Collector
Collects recent normal data and historical storm data using proven JSOC method
"""

import logging
from datetime import datetime, timedelta
from jsoc_ingestor import JSOCIngestor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleProductionCollector:
    """Simple collector using proven JSOC ingestion for storm and normal data"""
    
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        logger.info(f"Initialized SimpleProductionCollector for project {project_id}, bucket {bucket_name}")
    
    def collect_storm_and_normal_data(self):
        """Collect both storm period and normal period data"""
        
        logger.info("🚀 Starting production solar data collection...")
        
        # 1. Collect data for known storm day (2012-03-07) - proven to work
        logger.info("⛈️ Phase 1: Collecting storm period data (2012-03-07)...")
        
        storm_ingestor = JSOCIngestor(
            project_id=self.project_id,
            raw_bucket=self.bucket_name
        )
        
        storm_ingestor.ingest_jsoc_data(
            start_time=datetime(2012, 3, 7),
            end_time=datetime(2012, 3, 7, 23, 59, 59),
            wavelength=193,
            cadence=3600  # 1 hour cadence for storm data
        )
        
        # 2. Collect normal period data (last week)
        logger.info("☀️ Phase 2: Collecting recent normal period data...")
        
        # Get last week's data for normal baseline
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        
        normal_ingestor = JSOCIngestor(
            project_id=self.project_id,
            raw_bucket=self.bucket_name
        )
        
        # Collect one day worth of normal data
        normal_day = start_time.replace(hour=12, minute=0, second=0, microsecond=0)
        
        normal_ingestor.ingest_jsoc_data(
            start_time=normal_day,
            end_time=normal_day + timedelta(hours=6),  # 6 hours of normal data
            wavelength=193,
            cadence=3600  # 1 hour cadence for normal data
        )
        
        logger.info("✅ Production data collection complete!")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Production Solar Data Collector")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--bucket-name", required=True, help="GCS Bucket name")
    
    args = parser.parse_args()
    
    collector = SimpleProductionCollector(
        project_id=args.project_id,
        bucket_name=args.bucket_name
    )
    
    collector.collect_storm_and_normal_data()


if __name__ == "__main__":
    main()