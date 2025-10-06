#!/usr/bin/env python3
"""
Production Solar Storm Event Collector
Collects real historical solar storm data for prediction modeling
"""

import httpx
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import asyncio
import time
from google.cloud import bigquery
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolarStormEventCollector:
    """Collects real solar storm events from multiple sources"""
    
    def __init__(self, project_id: str, dataset_name: str = "space_weather"):
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.client = bigquery.Client(project=project_id)
        
        # NOAA SWPC data sources - updated URLs
        self.noaa_events_url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
        self.noaa_kp_url = "https://services.swpc.noaa.gov/json/geomag/kp-index/kp-3-day.json"
        self.noaa_proton_url = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-3-day.json"
        
        # Historical data sources
        self.historical_flare_url = "https://www.spaceweatherlive.com/api/v1/flares"
        
        # Initialize BigQuery tables
        self._setup_tables()
    
    def _setup_tables(self):
        """Setup BigQuery tables for storm events"""
        
        # Create dataset if not exists
        dataset_id = f"{self.project_id}.{self.dataset_name}"
        try:
            self.client.get_dataset(dataset_id)
            logger.info(f"Dataset {dataset_id} already exists")
        except:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset_id}")
        
        # Storm events table schema
        storm_events_schema = [
            bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),  # FLARE, CME, GEOMAG, PROTON
            bigquery.SchemaField("event_class", "STRING", mode="NULLABLE"),  # X1.2, M5.4, etc.
            bigquery.SchemaField("start_time", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("peak_time", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("end_time", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("location", "STRING", mode="NULLABLE"),  # Solar coordinates
            bigquery.SchemaField("intensity", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("raw_data", "STRING", mode="NULLABLE"),  # Changed from JSON to STRING
            bigquery.SchemaField("collection_time", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("severity_score", "INTEGER", mode="NULLABLE"),  # 1-10 scale
        ]
        
        table_id = f"{dataset_id}.solar_storm_events_v2"  # New table name
        try:
            self.client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except:
            table = bigquery.Table(table_id, schema=storm_events_schema)
            self.client.create_table(table)
            logger.info(f"Created table {table_id}")
        
        self.storm_events_table = table_id
    
    async def collect_current_noaa_events(self) -> List[Dict]:
        """Collect current solar events from NOAA SWPC"""
        events = []
        
        urls = [
            ("XRAY", self.noaa_events_url),
            ("PROTON", self.noaa_proton_url), 
            ("GEOMAG", self.noaa_kp_url)
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for event_type, url in urls:
                try:
                    logger.info(f"Fetching {event_type} events from NOAA...")
                    response = await client.get(url)
                    response.raise_for_status()
                    
                    data = response.json()
                    for event in data:
                        processed_event = self._process_noaa_event(event, event_type)
                        if processed_event:
                            events.append(processed_event)
                    
                    logger.info(f"Collected {len([e for e in events if e['event_type'] == event_type])} {event_type} events")
                    
                except Exception as e:
                    logger.error(f"Error fetching {event_type} events: {e}")
                    continue
        
        return events
    
    def _process_noaa_event(self, event: Dict, event_type: str) -> Optional[Dict]:
        """Process NOAA event data into standardized format"""
        try:
            if event_type == "XRAY":
                # Filter for significant X-ray events only (M-class and above)
                flux_val = event.get('flux', 0)
                if isinstance(flux_val, str):
                    try:
                        flux_val = float(flux_val)
                    except:
                        flux_val = 0
                
                # Only process M-class (1e-5) and above events
                if flux_val < 1e-5:
                    return None
                
                # Classify the event
                if flux_val >= 1e-4:
                    event_class = f"X{flux_val/1e-4:.1f}"
                elif flux_val >= 1e-5:
                    event_class = f"M{flux_val/1e-5:.1f}"
                else:
                    return None  # Skip C-class and below
                
                intensity = flux_val
                severity_score = self._calculate_severity_score(event_class, event_type)
                
            elif event_type == "PROTON":
                # Filter for significant proton events (>10 pfu)
                flux_val = event.get('flux', 0)
                if isinstance(flux_val, str):
                    try:
                        flux_val = float(flux_val)
                    except:
                        flux_val = 0
                
                if flux_val < 10:  # Only process significant proton events
                    return None
                
                event_class = f"P{flux_val:.0f}"
                intensity = flux_val
                severity_score = min(10, max(1, int(flux_val / 100)))
                
            else:
                return None  # Skip other event types for now
            
            # Extract times
            event_time = event.get('time_tag', event.get('time'))
            if not event_time:
                return None
            
            start_time = self._parse_timestamp(event_time)
            if not start_time:
                return None
            
            # Generate unique event ID
            event_id = f"NOAA_{event_type}_{start_time.strftime('%Y%m%d_%H%M%S')}_{event_class}"
            
            return {
                'event_id': event_id,
                'event_type': event_type,
                'event_class': event_class,
                'start_time': start_time,
                'peak_time': start_time + timedelta(minutes=30),  # Estimate peak
                'end_time': start_time + timedelta(hours=2),      # Estimate end
                'location': None,
                'intensity': intensity,
                'source': f"NOAA_SWPC_{event_type}",
                'raw_data': json.dumps(event),
                'collection_time': datetime.utcnow(),
                'severity_score': severity_score
            }
            
        except Exception as e:
            logger.error(f"Error processing NOAA event: {e}")
            return None
    
    def collect_historical_major_storms(self, start_year: int = 2003, end_year: int = 2025) -> List[Dict]:
        """Collect major historical solar storms"""
        major_storms = []
        
        # Known major solar storm events (X-class flares and major geomagnetic storms)
        historical_events = [
            # Halloween Storm 2003
            {"date": "2003-10-28", "type": "FLARE", "class": "X17.2", "description": "Halloween Storm - strongest flare recorded"},
            {"date": "2003-11-04", "type": "FLARE", "class": "X28", "description": "Halloween Storm series"},
            
            # 2005 January events
            {"date": "2005-01-15", "type": "FLARE", "class": "X2.6", "description": "Major flare event"},
            {"date": "2005-01-17", "type": "FLARE", "class": "X3.8", "description": "Strong flare"},
            
            # 2006 December
            {"date": "2006-12-05", "type": "FLARE", "class": "X9.0", "description": "Major X-class flare"},
            {"date": "2006-12-13", "type": "FLARE", "class": "X3.4", "description": "Strong flare"},
            
            # 2011-2014 Solar Maximum
            {"date": "2011-02-15", "type": "FLARE", "class": "X2.2", "description": "Solar cycle 24 ramp up"},
            {"date": "2011-08-09", "type": "FLARE", "class": "X6.9", "description": "Major flare"},
            {"date": "2011-09-22", "type": "FLARE", "class": "X1.4", "description": "Equinox storm"},
            
            {"date": "2012-01-23", "type": "FLARE", "class": "M8.7", "description": "Strong M-class"},
            {"date": "2012-03-07", "type": "FLARE", "class": "X5.4", "description": "Major flare - good data available"},
            {"date": "2012-07-12", "type": "FLARE", "class": "X1.4", "description": "Summer storm"},
            
            {"date": "2013-05-13", "type": "FLARE", "class": "X2.8", "description": "Solar max period"},
            {"date": "2013-11-05", "type": "FLARE", "class": "X3.3", "description": "Strong flare"},
            
            {"date": "2014-02-25", "type": "FLARE", "class": "X4.9", "description": "Major February flare"},
            {"date": "2014-09-10", "type": "FLARE", "class": "X1.6", "description": "September event"},
            
            # Recent events
            {"date": "2017-09-06", "type": "FLARE", "class": "X9.3", "description": "Strong solar cycle 24 event"},
            {"date": "2017-09-10", "type": "FLARE", "class": "X8.2", "description": "September 2017 series"},
            
            {"date": "2021-10-28", "type": "FLARE", "class": "X1.0", "description": "Solar cycle 25 begins"},
            {"date": "2022-04-20", "type": "FLARE", "class": "M9.6", "description": "Strong M-class"},
            
            {"date": "2023-02-17", "type": "FLARE", "class": "M6.8", "description": "Cycle 25 activity"},
            {"date": "2024-02-22", "type": "FLARE", "class": "X6.3", "description": "Major recent flare"},
            {"date": "2024-05-11", "type": "FLARE", "class": "X5.8", "description": "Strong May 2024 event"},
        ]
        
        for event in historical_events:
            try:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
                
                # Calculate severity score
                severity_score = self._calculate_severity_score(event["class"], event["type"])
                
                # Parse intensity
                intensity = self._parse_xray_intensity(event["class"]) if event["type"] == "FLARE" else None
                
                storm_event = {
                    'event_id': f"HIST_{event['type']}_{event_date.strftime('%Y%m%d')}",
                    'event_type': event["type"],
                    'event_class': event["class"],
                    'start_time': event_date,
                    'peak_time': event_date + timedelta(hours=1),  # Estimate peak 1 hour after start
                    'end_time': event_date + timedelta(hours=3),   # Estimate 3 hour duration
                    'location': None,
                    'intensity': intensity,
                    'source': "HISTORICAL_CURATED",
                    'raw_data': json.dumps(event),
                    'collection_time': datetime.utcnow(),
                    'severity_score': severity_score
                }
                
                major_storms.append(storm_event)
                
            except Exception as e:
                logger.error(f"Error processing historical event {event}: {e}")
                continue
        
        logger.info(f"Collected {len(major_storms)} historical major storm events")
        return major_storms
    
    def _parse_timestamp(self, time_str: Optional[str]) -> Optional[datetime]:
        """Parse various timestamp formats"""
        if not time_str:
            return None
        
        time_formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y%m%d %H%M",
            "%Y-%m-%d",
        ]
        
        for fmt in time_formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse timestamp: {time_str}")
        return None
    
    def _parse_xray_intensity(self, xray_class: str) -> Optional[float]:
        """Parse X-ray flare class to numerical intensity"""
        if not xray_class:
            return None
        
        try:
            if xray_class.startswith('X'):
                return float(xray_class[1:]) * 1e-4
            elif xray_class.startswith('M'):
                return float(xray_class[1:]) * 1e-5
            elif xray_class.startswith('C'):
                return float(xray_class[1:]) * 1e-6
            elif xray_class.startswith('B'):
                return float(xray_class[1:]) * 1e-7
            elif xray_class.startswith('A'):
                return float(xray_class[1:]) * 1e-8
        except:
            pass
        
        return None
    
    def _calculate_severity_score(self, event_class: str, event_type: str) -> int:
        """Calculate severity score (1-10) for storm event"""
        if not event_class:
            return 1
        
        try:
            if event_type == "FLARE":
                if event_class.startswith('X'):
                    magnitude = float(event_class[1:])
                    if magnitude >= 10:
                        return 10
                    elif magnitude >= 5:
                        return 9
                    elif magnitude >= 2:
                        return 8
                    else:
                        return 7
                elif event_class.startswith('M'):
                    magnitude = float(event_class[1:])
                    if magnitude >= 8:
                        return 6
                    elif magnitude >= 5:
                        return 5
                    else:
                        return 4
                elif event_class.startswith('C'):
                    return 3
                else:
                    return 2
            else:
                return 5  # Default for other event types
        except:
            return 1
    
    def save_events_to_bigquery(self, events: List[Dict]):
        """Save storm events to BigQuery"""
        if not events:
            logger.warning("No events to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(events)
        
        # Ensure proper data types
        for col in ['start_time', 'peak_time', 'end_time', 'collection_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Configure job
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_IF_NEEDED"
        )
        
        # Load to BigQuery
        job = self.client.load_table_from_dataframe(
            df, self.storm_events_table, job_config=job_config
        )
        job.result()  # Wait for job to complete
        
        logger.info(f"Saved {len(events)} storm events to BigQuery")
    
    def get_recent_normal_period(self, days: int = 5) -> tuple:
        """Get recent period for normal solar activity data collection"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        logger.info(f"Recent normal period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
        return start_time, end_time
    
    async def run_collection(self, include_historical: bool = True, include_current: bool = True):
        """Run complete storm event collection"""
        all_events = []
        
        if include_current:
            logger.info("🌟 Collecting current NOAA storm events...")
            current_events = await self.collect_current_noaa_events()
            all_events.extend(current_events)
        
        if include_historical:
            logger.info("📚 Collecting historical major storms...")
            historical_events = self.collect_historical_major_storms()
            all_events.extend(historical_events)
        
        if all_events:
            logger.info(f"💾 Saving {len(all_events)} total storm events to BigQuery...")
            self.save_events_to_bigquery(all_events)
        
        # Get recent normal period info
        start_normal, end_normal = self.get_recent_normal_period()
        
        logger.info("✅ Storm event collection completed!")
        logger.info(f"📊 Total events collected: {len(all_events)}")
        logger.info(f"🕐 Normal data period: {start_normal.strftime('%Y-%m-%d')} to {end_normal.strftime('%Y-%m-%d')}")
        
        return all_events


async def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Solar Storm Event Collector")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--current-only", action="store_true", help="Collect only current events")
    parser.add_argument("--historical-only", action="store_true", help="Collect only historical events")
    
    args = parser.parse_args()
    
    collector = SolarStormEventCollector(args.project_id)
    
    include_current = not args.historical_only
    include_historical = not args.current_only
    
    await collector.run_collection(
        include_historical=include_historical,
        include_current=include_current
    )


if __name__ == "__main__":
    asyncio.run(main())