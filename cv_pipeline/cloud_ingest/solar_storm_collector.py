#!/usr/bin/env python3
"""
Comprehensive Solar Storm Data Collector
Ingests data from major historical solar storms for robust ML training

Target Events:
- X5.4 Flare (March 7, 2012) - X1.3 follow-up
- Halloween Storms (October 2003) - X28, X17, X11 flares  
- Bastille Day Event (July 14, 2000) - X5.7 flare
- Recent Solar Maximum (2011-2014) - Multiple X-class events
- Current Solar Cycle (2019-2025) - Recent major events
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import asyncio
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolarStormDatabase:
    """Database of major solar storm events with available data"""
    
    @staticmethod
    def get_major_storm_events() -> List[Dict]:
        """Return list of major solar storm events with data availability"""
        
        return [
            {
                "name": "X5.4 Flare Event",
                "date": "2012-03-07",
                "start_time": "2012-03-07T00:00:00",
                "end_time": "2012-03-07T23:59:59",
                "description": "X5.4 flare at 00:02 UT, followed by X1.3 at 01:05 UT",
                "flare_classes": ["X5.4", "X1.3"],
                "active_region": "AR1429",
                "location": "N17E27",
                "cme_speed": "2684 km/s",
                "data_sources": ["SDO/AIA", "SDO/HMI", "SOHO/LASCO", "GOES"],
                "priority": 1,
                "data_available": True
            },
            {
                "name": "March 4 CME Event", 
                "date": "2012-03-04",
                "start_time": "2012-03-04T00:00:00",
                "end_time": "2012-03-04T23:59:59", 
                "description": "M2.3 and M1.5 flares with Earth-directed CME",
                "flare_classes": ["M2.3", "M1.5"],
                "active_region": "AR1429",
                "location": "S22E45",
                "cme_speed": "1200 km/s",
                "data_sources": ["SDO/AIA", "SDO/HMI", "SOHO/LASCO"],
                "priority": 2,
                "data_available": True
            },
            {
                "name": "X6.9 Flare Event",
                "date": "2011-08-09",
                "start_time": "2011-08-09T06:00:00",
                "end_time": "2011-08-09T18:00:00",
                "description": "X6.9 flare - largest of solar cycle 24",
                "flare_classes": ["X6.9"],
                "active_region": "AR1263",
                "location": "N16W69",
                "cme_speed": "1600 km/s",
                "data_sources": ["SDO/AIA", "SDO/HMI", "SOHO/LASCO"],
                "priority": 1,
                "data_available": True
            },
            {
                "name": "Halloween Storm X28",
                "date": "2003-10-28",
                "start_time": "2003-10-28T09:00:00",
                "end_time": "2003-10-28T15:00:00",
                "description": "X28+ flare - one of largest recorded",
                "flare_classes": ["X28+"],
                "active_region": "AR486",
                "location": "S16E08",
                "cme_speed": "2400 km/s",
                "data_sources": ["SOHO/EIT", "SOHO/LASCO", "GOES"],
                "priority": 1,
                "data_available": True  # SOHO data available
            },
            {
                "name": "Halloween Storm X17",
                "date": "2003-10-29",
                "start_time": "2003-10-29T20:00:00", 
                "end_time": "2003-10-29T23:59:59",
                "description": "X17.2 flare with major geomagnetic storm",
                "flare_classes": ["X17.2"],
                "active_region": "AR486",
                "location": "S15W02",
                "cme_speed": "2000 km/s",
                "data_sources": ["SOHO/EIT", "SOHO/LASCO", "GOES"],
                "priority": 1,
                "data_available": True
            },
            {
                "name": "Bastille Day Event",
                "date": "2000-07-14",
                "start_time": "2000-07-14T10:00:00",
                "end_time": "2000-07-14T14:00:00",
                "description": "X5.7 flare with full-halo CME",
                "flare_classes": ["X5.7"],
                "active_region": "AR9077",
                "location": "N22W07",
                "cme_speed": "1674 km/s",
                "data_sources": ["SOHO/EIT", "SOHO/LASCO", "GOES"],
                "priority": 2,
                "data_available": True
            },
            {
                "name": "X1.6 Recent Event",
                "date": "2014-10-22",
                "start_time": "2014-10-22T14:00:00",
                "end_time": "2014-10-22T18:00:00",
                "description": "X1.6 flare with good SDO coverage",
                "flare_classes": ["X1.6"],
                "active_region": "AR2192",
                "location": "S12E21", 
                "cme_speed": "900 km/s",
                "data_sources": ["SDO/AIA", "SDO/HMI", "SOHO/LASCO"],
                "priority": 2,
                "data_available": True
            },
            {
                "name": "X2.1 Recent Event",
                "date": "2014-03-29",
                "start_time": "2014-03-29T17:00:00",
                "end_time": "2014-03-29T20:00:00",
                "description": "X2.1 flare with Earth impact",
                "flare_classes": ["X2.1"],
                "active_region": "AR2017",
                "location": "N11E32",
                "cme_speed": "800 km/s",
                "data_sources": ["SDO/AIA", "SDO/HMI"],
                "priority": 2,
                "data_available": True
            }
        ]
    
    @staticmethod
    def get_priority_events(max_events: int = 5) -> List[Dict]:
        """Get highest priority storm events"""
        events = SolarStormDatabase.get_major_storm_events()
        
        # Sort by priority (1 = highest), then by flare magnitude
        def get_flare_magnitude(event):
            max_class = 0
            for flare in event['flare_classes']:
                if flare.startswith('X'):
                    magnitude = float(flare[1:].split('+')[0])
                    max_class = max(max_class, magnitude)
                elif flare.startswith('M'):
                    magnitude = float(flare[1:]) / 10  # Convert M-class to X equivalent
                    max_class = max(max_class, magnitude)
            return max_class
        
        sorted_events = sorted(events, 
                             key=lambda x: (x['priority'], -get_flare_magnitude(x)))
        
        return sorted_events[:max_events]

class SolarStormIngestor:
    """Comprehensive solar storm data ingestor"""
    
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        
        # Import ingestors
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent))
            
            from jsoc_ingestor import JSOCIngestor
            from real_noaa_ingestor import RealNOAAIngestor
            
            self.jsoc_ingestor = JSOCIngestor(project_id, bucket_name)
            self.noaa_ingestor = RealNOAAIngestor(project_id)
            
            logger.info("Initialized solar storm ingestor")
            
        except ImportError as e:
            logger.error(f"Failed to import ingestors: {e}")
            self.jsoc_ingestor = None
            self.noaa_ingestor = None
    
    def ingest_storm_event(self, event: Dict, wavelengths: List[int] = [193, 304, 171]):
        """Ingest data for a single storm event"""
        logger.info(f"Ingesting storm event: {event['name']}")
        logger.info(f"Date: {event['date']}, Flares: {event['flare_classes']}")
        
        start_time = datetime.fromisoformat(event['start_time'])
        end_time = datetime.fromisoformat(event['end_time'])
        
        # Ingest AIA data for multiple wavelengths
        if self.jsoc_ingestor and 'SDO/AIA' in event['data_sources']:
            for wavelength in wavelengths:
                logger.info(f"Ingesting AIA {wavelength}Å data...")
                try:
                    self.jsoc_ingestor.ingest_jsoc_data(
                        start_time=start_time,
                        end_time=end_time,
                        wavelength=wavelength,
                        cadence=1800  # 30-minute cadence for storm events
                    )
                except Exception as e:
                    logger.error(f"Failed to ingest AIA {wavelength}Å: {e}")
        
        # Ingest NOAA event data
        if self.noaa_ingestor:
            logger.info("Ingesting NOAA event data...")
            try:
                self.noaa_ingestor.ingest_real_events(
                    start_time.replace(hour=0, minute=0, second=0),
                    end_time.replace(hour=23, minute=59, second=59)
                )
            except Exception as e:
                logger.error(f"Failed to ingest NOAA events: {e}")
        
        logger.info(f"Completed ingestion for {event['name']}")
    
    def ingest_all_major_storms(self, max_events: int = 3):
        """Ingest data for all major storm events"""
        events = SolarStormDatabase.get_priority_events(max_events)
        
        logger.info(f"Starting ingestion for {len(events)} major storm events")
        
        total_ingested = 0
        
        for i, event in enumerate(events, 1):
            logger.info(f"\n=== Storm Event {i}/{len(events)} ===")
            
            try:
                self.ingest_storm_event(event)
                total_ingested += 1
                
            except Exception as e:
                logger.error(f"Failed to ingest {event['name']}: {e}")
                continue
        
        logger.info(f"\n=== Ingestion Complete ===")
        logger.info(f"Successfully ingested {total_ingested}/{len(events)} storm events")
        
        return total_ingested

def main():
    parser = argparse.ArgumentParser(description="Ingest comprehensive solar storm data")
    parser.add_argument("--project-id", default="rosy-clover-471810-i6",
                       help="Google Cloud Project ID")
    parser.add_argument("--bucket", default="solar-raw", 
                       help="GCS bucket for raw data")
    parser.add_argument("--max-events", type=int, default=3,
                       help="Maximum number of storm events to ingest")
    parser.add_argument("--list-events", action="store_true",
                       help="List available storm events and exit")
    parser.add_argument("--event-name", help="Ingest specific event by name")
    
    args = parser.parse_args()
    
    # List events if requested
    if args.list_events:
        events = SolarStormDatabase.get_major_storm_events()
        print(f"\nAvailable Solar Storm Events ({len(events)} total):")
        print("=" * 60)
        
        for i, event in enumerate(events, 1):
            print(f"{i}. {event['name']} ({event['date']})")
            print(f"   Flares: {', '.join(event['flare_classes'])}")
            print(f"   Active Region: {event['active_region']}")
            print(f"   CME Speed: {event['cme_speed']}")
            print(f"   Priority: {event['priority']}")
            print(f"   Description: {event['description']}")
            print()
        
        return 0
    
    # Initialize ingestor
    ingestor = SolarStormIngestor(args.project_id, args.bucket)
    
    if not ingestor.jsoc_ingestor:
        logger.error("Failed to initialize ingestors")
        return 1
    
    # Ingest specific event or all major events
    if args.event_name:
        events = SolarStormDatabase.get_major_storm_events()
        target_event = None
        
        for event in events:
            if args.event_name.lower() in event['name'].lower():
                target_event = event
                break
        
        if not target_event:
            logger.error(f"Event '{args.event_name}' not found")
            return 1
        
        ingestor.ingest_storm_event(target_event)
    else:
        ingestor.ingest_all_major_storms(args.max_events)
    
    return 0

if __name__ == "__main__":
    exit(main())