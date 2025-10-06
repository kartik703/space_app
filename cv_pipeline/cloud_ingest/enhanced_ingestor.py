#!/usr/bin/env python3
"""
Enhanced Solar Data Ingestor with NOAA Events and SOHO CME data
Fetches solar images, NOAA event lists, and SOHO CME images for labeled training data
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import httpx
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from google.cloud import storage, bigquery
from google.cloud.exceptions import NotFound

from gcs_ingestor import SolarImageIngestor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NOAAEvent:
    """NOAA Solar Event data structure"""
    event_id: str
    start_time: datetime
    peak_time: Optional[datetime]
    end_time: Optional[datetime]
    event_class: str
    location: str
    source_region: str
    event_type: str
    particulars: str

@dataclass
class SOHOCMEEvent:
    """SOHO CME Event data structure"""
    event_id: str
    datetime: datetime
    cme_type: str
    width: float
    speed: float
    acceleration: float
    source_location: str
    remarks: str

class EnhancedSolarIngestor(SolarImageIngestor):
    """Enhanced ingestor with NOAA events and SOHO CME data"""
    
    def __init__(self, project_id: str, raw_bucket: str = "solar-raw"):
        super().__init__(project_id, raw_bucket)
        
        # NOAA and SOHO APIs
        self.noaa_base_url = "https://services.swpc.noaa.gov/json"
        self.soho_lasco_base_url = "https://cdaw.gsfc.nasa.gov/CME_list"
        
        # Ensure additional BigQuery tables exist
        self._ensure_noaa_events_table()
        self._ensure_soho_cme_table()
    
    def _ensure_noaa_events_table(self):
        """Create NOAA events table in BigQuery"""
        dataset_id = "space_weather"
        table_id = "noaa_events"
        
        dataset_ref = self.bq_client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        
        try:
            self.bq_client.get_table(table_ref)
            logger.info(f"Table {table_id} already exists")
        except NotFound:
            schema = [
                bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("start_time", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("peak_time", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("end_time", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("event_class", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("location", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("source_region", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("particulars", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("ingestion_time", "TIMESTAMP", mode="REQUIRED"),
            ]
            table = bigquery.Table(table_ref, schema=schema)
            table = self.bq_client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    def _ensure_soho_cme_table(self):
        """Create SOHO CME events table in BigQuery"""
        dataset_id = "space_weather"
        table_id = "soho_cme_events"
        
        dataset_ref = self.bq_client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        
        try:
            self.bq_client.get_table(table_ref)
            logger.info(f"Table {table_id} already exists")
        except NotFound:
            schema = [
                bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("datetime", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("cme_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("width", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("speed", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("acceleration", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("source_location", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("remarks", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("lasco_image_url", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("gcs_image_path", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("ingestion_time", "TIMESTAMP", mode="REQUIRED"),
            ]
            table = bigquery.Table(table_ref, schema=schema)
            table = self.bq_client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    def fetch_noaa_events(self, start_time: datetime, end_time: datetime) -> List[NOAAEvent]:
        """Fetch NOAA solar events for the given time range"""
        logger.info(f"Fetching NOAA events from {start_time} to {end_time}")
        
        events = []
        
        # NOAA SWPC provides different event types
        event_types = [
            "solar-flares",
            "solar-energetic-particle", 
            "geomagnetic-storms",
            "solar-radiation-storms"
        ]
        
        try:
            with httpx.Client(timeout=30.0) as client:
                for event_type in event_types:
                    try:
                        url = f"{self.noaa_base_url}/{event_type}.json"
                        response = client.get(url)
                        response.raise_for_status()
                        
                        event_data = response.json()
                        
                        for event in event_data:
                            # Parse event timestamps
                            try:
                                start_dt = datetime.fromisoformat(event.get('begin_time', '').replace('Z', '+00:00')).replace(tzinfo=None)
                                
                                # Only include events in our time range
                                if start_time <= start_dt <= end_time:
                                    peak_dt = None
                                    end_dt = None
                                    
                                    if event.get('max_time'):
                                        peak_dt = datetime.fromisoformat(event.get('max_time', '').replace('Z', '+00:00')).replace(tzinfo=None)
                                    
                                    if event.get('end_time'):
                                        end_dt = datetime.fromisoformat(event.get('end_time', '').replace('Z', '+00:00')).replace(tzinfo=None)
                                    
                                    noaa_event = NOAAEvent(
                                        event_id=f"{event_type}_{start_dt.strftime('%Y%m%d_%H%M%S')}",
                                        start_time=start_dt,
                                        peak_time=peak_dt,
                                        end_time=end_dt,
                                        event_class=event.get('class', ''),
                                        location=event.get('location', ''),
                                        source_region=event.get('source_region', ''),
                                        event_type=event_type,
                                        particulars=event.get('particulars', '')
                                    )
                                    
                                    events.append(noaa_event)
                                    
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Error parsing event: {e}")
                                continue
                        
                        logger.info(f"Found {len([e for e in events if e.event_type == event_type])} {event_type} events")
                        
                    except Exception as e:
                        logger.error(f"Error fetching {event_type}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error fetching NOAA events: {e}")
        
        logger.info(f"Total NOAA events found: {len(events)}")
        return events
    
    def fetch_soho_cme_catalog(self, start_time: datetime, end_time: datetime) -> List[SOHOCMEEvent]:
        """Fetch SOHO LASCO CME catalog data"""
        logger.info(f"Fetching SOHO CME events from {start_time} to {end_time}")
        
        # SOHO CME catalog is organized by year
        events = []
        
        for year in range(start_time.year, end_time.year + 1):
            try:
                # SOHO CME catalog URL format
                catalog_url = f"{self.soho_lasco_base_url}/UNIVERSAL/text_ver/{year}/univ{year}.txt"
                
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(catalog_url)
                    response.raise_for_status()
                    
                    # Parse the text-based catalog
                    lines = response.text.strip().split('\n')
                    
                    for line in lines:
                        if line.startswith('#') or len(line.strip()) == 0:
                            continue
                        
                        try:
                            # Parse CME catalog format
                            parts = line.split()
                            if len(parts) < 10:
                                continue
                            
                            # Extract date/time
                            date_str = parts[0]
                            time_str = parts[1]
                            datetime_str = f"{date_str} {time_str}"
                            event_dt = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")
                            
                            # Only include events in our time range
                            if start_time <= event_dt <= end_time:
                                cme_event = SOHOCMEEvent(
                                    event_id=f"cme_{event_dt.strftime('%Y%m%d_%H%M%S')}",
                                    datetime=event_dt,
                                    cme_type=parts[2] if len(parts) > 2 else '',
                                    width=float(parts[3]) if len(parts) > 3 and parts[3] != '---' else 0.0,
                                    speed=float(parts[4]) if len(parts) > 4 and parts[4] != '---' else 0.0,
                                    acceleration=float(parts[5]) if len(parts) > 5 and parts[5] != '---' else 0.0,
                                    source_location=parts[6] if len(parts) > 6 else '',
                                    remarks=' '.join(parts[7:]) if len(parts) > 7 else ''
                                )
                                
                                events.append(cme_event)
                                
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing CME line '{line}': {e}")
                            continue
                    
                    logger.info(f"Found {len([e for e in events if e.datetime.year == year])} CME events for {year}")
                    
            except Exception as e:
                logger.error(f"Error fetching SOHO CME catalog for {year}: {e}")
                continue
        
        logger.info(f"Total SOHO CME events found: {len(events)}")
        return events
    
    def fetch_soho_lasco_image(self, cme_event: SOHOCMEEvent) -> Optional[Dict]:
        """Fetch SOHO LASCO image for a CME event"""
        try:
            # SOHO LASCO image URL format
            dt = cme_event.datetime
            image_url = (f"https://soho.esac.esa.int/data/REPROCESSING/Completed/"
                        f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/lasco_c2_1024/"
                        f"{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}_lasco_c2_1024.jpg")
            
            with httpx.Client(timeout=30.0) as client:
                response = client.get(image_url)
                response.raise_for_status()
                
                image_data = response.content
                if len(image_data) == 0:
                    logger.warning(f"Empty LASCO image for {cme_event.event_id}")
                    return None
                
                return {
                    "image_data": image_data,
                    "image_url": image_url,
                    "event_id": cme_event.event_id,
                    "timestamp": cme_event.datetime,
                    "instrument": "LASCO",
                    "detector": "C2",
                    "file_size": len(image_data),
                    "width": 1024,
                    "height": 1024
                }
                
        except Exception as e:
            logger.warning(f"Could not fetch LASCO image for {cme_event.event_id}: {e}")
            return None
    
    def store_noaa_events(self, events: List[NOAAEvent]):
        """Store NOAA events in BigQuery"""
        if not events:
            return
        
        logger.info(f"Storing {len(events)} NOAA events in BigQuery")
        
        rows = []
        for event in events:
            row = {
                "event_id": event.event_id,
                "start_time": event.start_time.isoformat(),
                "peak_time": event.peak_time.isoformat() if event.peak_time else None,
                "end_time": event.end_time.isoformat() if event.end_time else None,
                "event_class": event.event_class,
                "location": event.location,
                "source_region": event.source_region,
                "event_type": event.event_type,
                "particulars": event.particulars,
                "ingestion_time": datetime.utcnow().isoformat(),
            }
            rows.append(row)
        
        table_ref = self.bq_client.dataset("space_weather").table("noaa_events")
        errors = self.bq_client.insert_rows_json(table_ref, rows)
        
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
        else:
            logger.info(f"Stored {len(rows)} NOAA events in BigQuery")
    
    def store_soho_cme_events(self, events: List[SOHOCMEEvent], image_paths: Dict[str, str] = None):
        """Store SOHO CME events in BigQuery"""
        if not events:
            return
        
        logger.info(f"Storing {len(events)} SOHO CME events in BigQuery")
        
        rows = []
        for event in events:
            gcs_path = image_paths.get(event.event_id) if image_paths else None
            
            row = {
                "event_id": event.event_id,
                "datetime": event.datetime.isoformat(),
                "cme_type": event.cme_type,
                "width": event.width,
                "speed": event.speed,
                "acceleration": event.acceleration,
                "source_location": event.source_location,
                "remarks": event.remarks,
                "lasco_image_url": f"https://soho.esac.esa.int/data/REPROCESSING/Completed/{event.datetime.year:04d}/{event.datetime.month:02d}/{event.datetime.day:02d}/lasco_c2_1024/{event.datetime.year:04d}{event.datetime.month:02d}{event.datetime.day:02d}_{event.datetime.hour:02d}{event.datetime.minute:02d}_lasco_c2_1024.jpg",
                "gcs_image_path": gcs_path,
                "ingestion_time": datetime.utcnow().isoformat(),
            }
            rows.append(row)
        
        table_ref = self.bq_client.dataset("space_weather").table("soho_cme_events")
        errors = self.bq_client.insert_rows_json(table_ref, rows)
        
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
        else:
            logger.info(f"Stored {len(rows)} SOHO CME events in BigQuery")
    
    def upload_lasco_image_to_gcs(self, image_data: Dict) -> str:
        """Upload LASCO image to GCS"""
        timestamp = image_data["timestamp"]
        instrument = image_data["instrument"].lower()
        detector = image_data["detector"].lower()
        
        # Create GCS path: lasco/detector/year/month/day/filename
        gcs_path = (f"{instrument}/{detector}/"
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
            "detector": image_data["detector"],
            "timestamp": image_data["timestamp"].isoformat(),
            "width": str(image_data["width"]),
            "height": str(image_data["height"]),
            "event_id": image_data["event_id"],
            "image_url": image_data["image_url"]
        }
        blob.patch()
        
        gcs_full_path = f"gs://{self.raw_bucket}/{gcs_path}"
        logger.info(f"Uploaded LASCO image to {gcs_full_path}")
        return gcs_full_path
    
    def ingest_enhanced_data(self, start_time: datetime, end_time: datetime,
                           fetch_sdo: bool = True, fetch_noaa: bool = True, 
                           fetch_soho: bool = True, interval_minutes: int = 60):
        """Enhanced ingestion with SDO, NOAA events, and SOHO CME data"""
        
        logger.info(f"Starting enhanced ingestion from {start_time} to {end_time}")
        
        # 1. Fetch NOAA events
        noaa_events = []
        if fetch_noaa:
            noaa_events = self.fetch_noaa_events(start_time, end_time)
            self.store_noaa_events(noaa_events)
        
        # 2. Fetch SOHO CME events
        soho_events = []
        soho_image_paths = {}
        if fetch_soho:
            soho_events = self.fetch_soho_cme_catalog(start_time, end_time)
            
            # Fetch LASCO images for CME events
            logger.info(f"Fetching LASCO images for {len(soho_events)} CME events")
            for event in soho_events[:10]:  # Limit to first 10 for testing
                image_data = self.fetch_soho_lasco_image(event)
                if image_data:
                    gcs_path = self.upload_lasco_image_to_gcs(image_data)
                    soho_image_paths[event.event_id] = gcs_path
                    time.sleep(1)  # Be respectful to SOHO servers
            
            self.store_soho_cme_events(soho_events, soho_image_paths)
        
        # 3. Fetch SDO/AIA images
        if fetch_sdo:
            self.ingest_time_range(
                start_time=start_time,
                end_time=end_time,
                interval_minutes=interval_minutes,
                instrument="AIA",
                wavelength=193
            )
        
        # Summary
        logger.info("Enhanced ingestion completed!")
        logger.info(f"NOAA events: {len(noaa_events)}")
        logger.info(f"SOHO CME events: {len(soho_events)}")
        logger.info(f"LASCO images uploaded: {len(soho_image_paths)}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced solar data ingestion with NOAA and SOHO data")
    parser.add_argument("--start", required=True, help="Start time (ISO format: 2012-03-04T00:00:00)")
    parser.add_argument("--end", required=True, help="End time (ISO format: 2012-03-04T23:59:59)")
    parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--bucket", default="solar-raw", help="GCS bucket for raw images")
    parser.add_argument("--no-sdo", action="store_true", help="Skip SDO/AIA image ingestion")
    parser.add_argument("--no-noaa", action="store_true", help="Skip NOAA event ingestion")
    parser.add_argument("--no-soho", action="store_true", help="Skip SOHO CME ingestion")
    parser.add_argument("--interval", type=int, default=60, help="SDO image interval in minutes")
    
    args = parser.parse_args()
    
    # Parse timestamps
    try:
        start_time = datetime.fromisoformat(args.start.replace('Z', '+00:00')).replace(tzinfo=None)
        end_time = datetime.fromisoformat(args.end.replace('Z', '+00:00')).replace(tzinfo=None)
    except ValueError as e:
        logger.error(f"Invalid timestamp format: {e}")
        return 1
    
    # Create enhanced ingestor
    ingestor = EnhancedSolarIngestor(args.project_id, args.bucket)
    
    # Run enhanced ingestion
    ingestor.ingest_enhanced_data(
        start_time=start_time,
        end_time=end_time,
        fetch_sdo=not args.no_sdo,
        fetch_noaa=not args.no_noaa,
        fetch_soho=not args.no_soho,
        interval_minutes=args.interval
    )
    
    return 0


if __name__ == "__main__":
    exit(main())