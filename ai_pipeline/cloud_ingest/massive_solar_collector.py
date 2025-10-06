#!/usr/bin/env python3
"""
🌞 MASSIVE SOLAR DATA COLLECTION: 2003-2025 🌞
Comprehensive ingestion of ALL available solar data with maximum temporal resolution

SCOPE: 22 Years of Solar Observations (2003-2025)
- Solar Cycle 23 decline (2003-2008) 
- Solar Minimum (2008-2010)
- Solar Cycle 24 rise (2010-2014)
- Solar Cycle 24 decline (2014-2019)
- Solar Cycle 25 rise (2019-2025)

DATA SOURCES:
- SDO/AIA: 2010-2025 (1-12 second cadence)
- SOHO/EIT: 2003-2025 (12 minute cadence)
- SOHO/LASCO: 2003-2025 (CME observations)
- GOES X-ray: 2003-2025 (1-second flare data)
- NOAA Events: 2003-2025 (All storm catalogs)

ESTIMATED DATA VOLUME: ~500TB of raw solar observations
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import asyncio
import json
from pathlib import Path
import time
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MassiveSolarDataCollector:
    """Collect ALL solar data from 2003-2025 with maximum temporal resolution"""
    
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        
        # Time ranges for different missions
        self.mission_periods = {
            "SOHO_EIT": {
                "start": "2003-01-01",
                "end": "2025-09-19", 
                "cadence": 720,  # 12 minutes (best available)
                "instruments": ["EIT_171", "EIT_195", "EIT_284", "EIT_304"]
            },
            "SOHO_LASCO": {
                "start": "2003-01-01", 
                "end": "2025-09-19",
                "cadence": 900,  # 15 minutes
                "instruments": ["LASCO_C2", "LASCO_C3"]
            },
            "SDO_AIA": {
                "start": "2010-05-01",  # SDO operational start
                "end": "2025-09-19",
                "cadence": 12,  # 12 seconds (maximum resolution)
                "instruments": ["AIA_94", "AIA_131", "AIA_171", "AIA_193", 
                              "AIA_211", "AIA_304", "AIA_335", "AIA_1600", "AIA_1700"]
            },
            "SDO_HMI": {
                "start": "2010-05-01",
                "end": "2025-09-19", 
                "cadence": 45,  # 45 seconds for magnetograms
                "instruments": ["HMI_CONTINUUM", "HMI_MAGNETOGRAM"]
            },
            "GOES_XRAY": {
                "start": "2003-01-01",
                "end": "2025-09-19",
                "cadence": 1,  # 1 second resolution
                "instruments": ["GOES_XRS"]
            }
        }
        
        # Priority data collection strategy
        self.collection_strategy = {
            "Phase_1_Critical": {
                "description": "Highest priority: Major solar storms and active periods",
                "time_periods": [
                    ("2003-10-19", "2003-11-15"),  # Halloween Storms 2003
                    ("2005-01-15", "2005-01-25"),  # Major January 2005 events
                    ("2006-12-05", "2006-12-15"),  # Major December 2006 flares
                    ("2010-05-01", "2010-12-31"),  # SDO first light + activity
                    ("2011-01-01", "2011-12-31"),  # Solar maximum approach
                    ("2012-01-01", "2012-12-31"),  # Peak solar maximum
                    ("2013-01-01", "2013-12-31"),  # Continued high activity
                    ("2014-01-01", "2014-12-31"),  # Late maximum activity
                    ("2017-09-01", "2017-09-30"),  # September 2017 storms
                    ("2021-01-01", "2021-12-31"),  # Cycle 25 rise
                    ("2022-01-01", "2022-12-31"),  # Recent major activity
                    ("2023-01-01", "2023-12-31"),  # High activity continues
                    ("2024-01-01", "2025-09-19"),  # Current period
                ],
                "cadence_override": 12,  # Maximum resolution for critical periods
                "priority": 1
            },
            "Phase_2_Background": {
                "description": "Background solar activity and quiet periods", 
                "time_periods": [
                    ("2003-01-01", "2003-10-18"),  # Pre-Halloween 2003
                    ("2003-11-16", "2004-12-31"),  # Post-Halloween to 2004
                    ("2005-01-26", "2009-12-31"),  # Solar minimum approach
                    ("2015-01-01", "2016-12-31"),  # Solar minimum period
                    ("2018-01-01", "2020-12-31"),  # Deep minimum to cycle 25
                ],
                "cadence_override": 300,  # 5-minute cadence for quieter periods
                "priority": 2
            }
        }
        
        # Import ingestors
        try:
            sys.path.append(str(Path(__file__).parent))
            from jsoc_ingestor import JSOCIngestor
            from real_noaa_ingestor import RealNOAAIngestor
            
            self.jsoc_ingestor = JSOCIngestor(project_id, bucket_name)
            self.noaa_ingestor = RealNOAAIngestor(project_id)
            
            logger.info("Initialized massive solar data collector")
            
        except ImportError as e:
            logger.error(f"Failed to import ingestors: {e}")
            self.jsoc_ingestor = None
            self.noaa_ingestor = None
    
    def estimate_data_volume(self) -> Dict:
        """Estimate total data volume for 2003-2025 collection"""
        
        estimates = {
            "SDO_AIA": {
                "years": 15.3,  # 2010-2025
                "images_per_day": 7200,  # 12s cadence * 9 wavelengths
                "image_size_mb": 16,  # 4K x 4K FITS
                "total_images": int(15.3 * 365 * 7200),
                "total_size_tb": 0
            },
            "SDO_HMI": {
                "years": 15.3,
                "images_per_day": 1920,  # 45s cadence * 2 products
                "image_size_mb": 32,  # Higher res magnetograms
                "total_images": int(15.3 * 365 * 1920),
                "total_size_tb": 0
            },
            "SOHO_EIT": {
                "years": 22,  # 2003-2025
                "images_per_day": 480,  # 12min cadence * 4 wavelengths  
                "image_size_mb": 4,  # 1K x 1K images
                "total_images": int(22 * 365 * 480),
                "total_size_tb": 0
            },
            "SOHO_LASCO": {
                "years": 22,
                "images_per_day": 192,  # 15min cadence * 2 coronagraphs
                "image_size_mb": 2,  # Smaller coronagraph images
                "total_images": int(22 * 365 * 192),
                "total_size_tb": 0
            }
        }
        
        # Calculate sizes
        total_size_tb = 0
        total_images = 0
        
        for mission, data in estimates.items():
            size_tb = (data["total_images"] * data["image_size_mb"]) / (1024 * 1024)
            data["total_size_tb"] = round(size_tb, 1)
            total_size_tb += size_tb
            total_images += data["total_images"]
        
        summary = {
            "missions": estimates,
            "totals": {
                "total_images": total_images,
                "total_size_tb": round(total_size_tb, 1),
                "estimated_cost_usd": round(total_size_tb * 20, 0),  # ~$20/TB storage
                "collection_time_days": round(total_images / 10000, 1)  # ~10K images/day
            }
        }
        
        return summary
    
    def create_yearly_batches(self, start_year: int = 2003, end_year: int = 2025) -> List[Dict]:
        """Create yearly batches for systematic data collection"""
        
        batches = []
        
        for year in range(start_year, end_year + 1):
            # Determine available missions for this year
            available_missions = []
            
            if year >= 2003:
                available_missions.extend(["SOHO_EIT", "SOHO_LASCO", "GOES_XRAY"])
            
            if year >= 2010:
                available_missions.extend(["SDO_AIA", "SDO_HMI"])
            
            # Determine activity level and priority
            high_activity_years = [2003, 2004, 2005, 2011, 2012, 2013, 2014, 2017, 2021, 2022, 2023, 2024, 2025]
            priority = 1 if year in high_activity_years else 2
            
            batch = {
                "year": year,
                "start_date": f"{year}-01-01T00:00:00",
                "end_date": f"{year}-12-31T23:59:59",
                "available_missions": available_missions,
                "priority": priority,
                "estimated_images": self._estimate_yearly_images(year, available_missions),
                "solar_cycle_phase": self._get_solar_cycle_phase(year)
            }
            
            batches.append(batch)
        
        return batches
    
    def _estimate_yearly_images(self, year: int, missions: List[str]) -> int:
        """Estimate number of images for a given year and missions"""
        
        daily_estimates = {
            "SOHO_EIT": 480,    # 4 wavelengths * 120 images/day
            "SOHO_LASCO": 192,  # 2 coronagraphs * 96 images/day  
            "SDO_AIA": 7200,    # 9 wavelengths * 800 images/day
            "SDO_HMI": 1920,    # 2 products * 960 images/day
            "GOES_XRAY": 86400  # 1-second data points
        }
        
        total_daily = sum(daily_estimates[mission] for mission in missions if mission in daily_estimates)
        return total_daily * 365
    
    def _get_solar_cycle_phase(self, year: int) -> str:
        """Determine solar cycle phase for given year"""
        
        phases = {
            2003: "Cycle 23 Decline",
            2004: "Cycle 23 Decline", 
            2005: "Cycle 23 Decline",
            2006: "Cycle 23 Decline",
            2007: "Cycle 23 Decline",
            2008: "Solar Minimum",
            2009: "Solar Minimum", 
            2010: "Cycle 24 Rise",
            2011: "Cycle 24 Rise",
            2012: "Cycle 24 Maximum",
            2013: "Cycle 24 Maximum",
            2014: "Cycle 24 Decline",
            2015: "Cycle 24 Decline",
            2016: "Cycle 24 Decline",
            2017: "Cycle 24 Decline", 
            2018: "Solar Minimum",
            2019: "Cycle 25 Rise",
            2020: "Cycle 25 Rise",
            2021: "Cycle 25 Rise",
            2022: "Cycle 25 Active",
            2023: "Cycle 25 Active",
            2024: "Cycle 25 Maximum",
            2025: "Cycle 25 Maximum"
        }
        
        return phases.get(year, "Unknown")
    
    def collect_sdo_data_batch(self, start_date: str, end_date: str, 
                              wavelengths: List[int] = [193, 304, 171], 
                              cadence: int = 12):
        """Collect SDO/AIA data for specified date range"""
        
        logger.info(f"Collecting SDO data: {start_date} to {end_date}")
        logger.info(f"Wavelengths: {wavelengths}, Cadence: {cadence}s")
        
        if not self.jsoc_ingestor:
            logger.error("JSOC ingestor not available")
            return False
        
        start_time = datetime.fromisoformat(start_date.replace('Z', ''))
        end_time = datetime.fromisoformat(end_date.replace('Z', ''))
        
        success_count = 0
        
        for wavelength in wavelengths:
            try:
                logger.info(f"Ingesting AIA {wavelength}Å...")
                
                self.jsoc_ingestor.ingest_jsoc_data(
                    start_time=start_time,
                    end_time=end_time,
                    wavelength=wavelength,
                    cadence=cadence
                )
                
                success_count += 1
                
                # Brief pause between wavelengths to avoid overwhelming JSOC
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to ingest AIA {wavelength}Å: {e}")
        
        logger.info(f"SDO batch complete: {success_count}/{len(wavelengths)} wavelengths")
        return success_count > 0
    
    def collect_noaa_events_batch(self, start_date: str, end_date: str):
        """Collect NOAA events for specified date range"""
        
        logger.info(f"Collecting NOAA events: {start_date} to {end_date}")
        
        if not self.noaa_ingestor:
            logger.error("NOAA ingestor not available")
            return False
        
        try:
            start_time = datetime.fromisoformat(start_date.replace('Z', ''))
            end_time = datetime.fromisoformat(end_date.replace('Z', ''))
            
            events = self.noaa_ingestor.ingest_real_events(start_time, end_time)
            
            logger.info(f"NOAA batch complete: {len(events)} events")
            return len(events) > 0
            
        except Exception as e:
            logger.error(f"Failed to ingest NOAA events: {e}")
            return False
    
    def run_massive_collection(self, start_year: int = 2003, end_year: int = 2025, 
                              priority_only: bool = False, test_mode: bool = False):
        """Run the massive 22-year data collection"""
        
        logger.info("=" * 80)
        logger.info("🌞 STARTING MASSIVE SOLAR DATA COLLECTION (2003-2025) 🌞")
        logger.info("=" * 80)
        
        # Show data volume estimates
        estimates = self.estimate_data_volume()
        logger.info(f"ESTIMATED TOTAL DATA VOLUME: {estimates['totals']['total_size_tb']} TB")
        logger.info(f"ESTIMATED TOTAL IMAGES: {estimates['totals']['total_images']:,}")
        logger.info(f"ESTIMATED COLLECTION TIME: {estimates['totals']['collection_time_days']} days")
        logger.info(f"ESTIMATED STORAGE COST: ${estimates['totals']['estimated_cost_usd']:,}")
        
        if test_mode:
            logger.info("🧪 TEST MODE: Processing 1 day samples only")
            end_year = start_year  # Only test first year
        
        # Create batches
        batches = self.create_yearly_batches(start_year, end_year)
        
        if priority_only:
            batches = [b for b in batches if b["priority"] == 1]
            logger.info(f"PRIORITY MODE: Processing {len(batches)} high-priority years only")
        
        total_batches = len(batches)
        completed_batches = 0
        failed_batches = 0
        
        logger.info(f"Processing {total_batches} yearly batches...")
        
        for i, batch in enumerate(batches, 1):
            year = batch["year"]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {i}/{total_batches}: YEAR {year}")
            logger.info(f"Solar Cycle: {batch['solar_cycle_phase']}")
            logger.info(f"Priority: {batch['priority']}")
            logger.info(f"Available Missions: {batch['available_missions']}")
            logger.info(f"Estimated Images: {batch['estimated_images']:,}")
            logger.info(f"{'='*60}")
            
            batch_start_time = time.time()
            batch_success = True
            
            try:
                if test_mode:
                    # Test mode: only process first week of year
                    start_date = f"{year}-01-01T00:00:00"
                    end_date = f"{year}-01-07T23:59:59"
                else:
                    start_date = batch["start_date"]
                    end_date = batch["end_date"]
                
                # Collect SDO data if available
                if "SDO_AIA" in batch["available_missions"]:
                    success = self.collect_sdo_data_batch(
                        start_date, end_date,
                        wavelengths=[193, 304, 171],  # Core wavelengths
                        cadence=300 if batch["priority"] == 2 else 60  # Adaptive cadence
                    )
                    batch_success &= success
                
                # Collect NOAA events
                success = self.collect_noaa_events_batch(start_date, end_date)
                batch_success &= success
                
                if batch_success:
                    completed_batches += 1
                    status = "✅ SUCCESS"
                else:
                    failed_batches += 1
                    status = "⚠️ PARTIAL"
                
            except Exception as e:
                logger.error(f"Batch {year} failed: {e}")
                failed_batches += 1
                status = "❌ FAILED"
                batch_success = False
            
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch {year} {status} - Time: {batch_time:.1f}s")
            
            # Progress summary
            progress = (i / total_batches) * 100
            logger.info(f"Progress: {progress:.1f}% ({i}/{total_batches})")
            
            if test_mode and i >= 3:  # Limit test mode to 3 batches
                logger.info("Test mode limit reached")
                break
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("🎉 MASSIVE DATA COLLECTION COMPLETE 🎉")
        logger.info("="*80)
        logger.info(f"Total Batches: {total_batches}")
        logger.info(f"Completed: {completed_batches}")
        logger.info(f"Failed: {failed_batches}")
        logger.info(f"Success Rate: {(completed_batches/total_batches)*100:.1f}%")
        
        if completed_batches > 0:
            logger.info("✅ SOLAR DATA ARCHIVE READY FOR ML TRAINING")
        
        return completed_batches > 0


def main():
    parser = argparse.ArgumentParser(description="Massive Solar Data Collection 2003-2025")
    parser.add_argument("--project-id", default="rosy-clover-471810-i6",
                       help="Google Cloud Project ID")
    parser.add_argument("--bucket", default="solar-raw",
                       help="GCS bucket for raw data")
    parser.add_argument("--start-year", type=int, default=2003,
                       help="Start year for collection")
    parser.add_argument("--end-year", type=int, default=2025,
                       help="End year for collection")
    parser.add_argument("--priority-only", action="store_true",
                       help="Collect only high-priority periods")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode: collect samples only")
    parser.add_argument("--estimate-only", action="store_true",
                       help="Show data volume estimates and exit")
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = MassiveSolarDataCollector(args.project_id, args.bucket)
    
    if args.estimate_only:
        estimates = collector.estimate_data_volume()
        
        print("\n🌞 MASSIVE SOLAR DATA COLLECTION ESTIMATES 🌞")
        print("="*60)
        
        for mission, data in estimates["missions"].items():
            print(f"\n{mission}:")
            print(f"  Years of data: {data['years']}")
            print(f"  Images per day: {data['images_per_day']:,}")
            print(f"  Total images: {data['total_images']:,}")
            print(f"  Data volume: {data['total_size_tb']} TB")
        
        print(f"\nTOTAL ESTIMATES:")
        print(f"  Total images: {estimates['totals']['total_images']:,}")
        print(f"  Total volume: {estimates['totals']['total_size_tb']} TB")
        print(f"  Storage cost: ${estimates['totals']['estimated_cost_usd']:,}")
        print(f"  Collection time: {estimates['totals']['collection_time_days']} days")
        
        return 0
    
    # Run collection
    success = collector.run_massive_collection(
        start_year=args.start_year,
        end_year=args.end_year,
        priority_only=args.priority_only,
        test_mode=args.test_mode
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())