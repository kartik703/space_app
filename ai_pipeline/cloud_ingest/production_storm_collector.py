#!/usr/bin/env python3
"""
Production Solar Storm Data Collector
Collects solar images for storm events from our database
"""

import drms
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path
import os
import asyncio
import aiofiles
from google.cloud import bigquery, storage
import numpy as np
from PIL import Image
import io
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionStormDataCollector:
    """Collects solar images for storm events"""
    
    def __init__(self, project_id: str, bucket_name: str, dataset_name: str = "space_weather"):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.dataset_name = dataset_name
        
        # Initialize clients
        self.bq_client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # DRMS client for SDO data
        try:
            self.drms_client = drms.Client()
            logger.info("✅ DRMS client initialized")
        except Exception as e:
            logger.error(f"❌ DRMS client initialization failed: {e}")
            self.drms_client = None
        
        # Data collection parameters
        self.instruments = {
            'AIA_171': 'aia.lev1_euv_12s[{time}][171]{/}',
            'AIA_304': 'aia.lev1_euv_12s[{time}][304]{/}',
            'HMI_MAGNETOGRAM': 'hmi.M_45s[{time}]{/}',
        }
        
        self.collection_stats = {
            'storm_images': 0,
            'normal_images': 0,
            'failed_downloads': 0,
            'total_events_processed': 0
        }
    
    def get_storm_events(self, limit: int = 50) -> pd.DataFrame:
        """Get storm events from BigQuery"""
        query = f"""
        SELECT 
            event_id,
            event_type,
            event_class,
            start_time,
            peak_time,
            end_time,
            severity_score,
            source
        FROM `{self.project_id}.{self.dataset_name}.solar_storm_events_v2`
        WHERE severity_score >= 7  -- Only get major storms
        ORDER BY start_time DESC
        LIMIT {limit}
        """
        
        # Use standard BigQuery to avoid permission issues
        job_config = bigquery.QueryJobConfig(use_query_cache=True)
        query_job = self.bq_client.query(query, job_config=job_config)
        results = query_job.result()
        
        # Convert to pandas manually to avoid BigQuery Storage issues
        rows = []
        for row in results:
            rows.append({
                'event_id': row.event_id,
                'event_type': row.event_type,
                'event_class': row.event_class,
                'start_time': row.start_time,
                'peak_time': row.peak_time,
                'end_time': row.end_time,
                'severity_score': row.severity_score,
                'source': row.source
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"📊 Retrieved {len(df)} major storm events from BigQuery")
        return df
    
    def get_normal_period_timeranges(self, days: int = 5) -> List[Tuple[datetime, datetime]]:
        """Get time ranges for normal solar activity (last 5 days)"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Split into 6-hour chunks for manageable data collection
        timeranges = []
        current = start_time
        while current < end_time:
            chunk_end = min(current + timedelta(hours=6), end_time)
            timeranges.append((current, chunk_end))
            current = chunk_end
        
        logger.info(f"📅 Generated {len(timeranges)} normal data time ranges")
        return timeranges
    
    async def collect_storm_images(self, event: pd.Series, images_per_event: int = 3):
        """Collect solar images for a specific storm event"""
        if not self.drms_client:
            logger.warning("DRMS client not available")
            return []
        
        try:
            event_id = event['event_id']
            start_time = pd.to_datetime(event['start_time'])
            peak_time = pd.to_datetime(event['peak_time']) if pd.notna(event['peak_time']) else start_time + timedelta(hours=1)
            
            logger.info(f"🌩️ Collecting images for storm event: {event_id}")
            
            collected_images = []
            
            # Collect images around peak time
            time_offsets = [-timedelta(hours=1), timedelta(0), timedelta(hours=1)][:images_per_event]
            
            for i, offset in enumerate(time_offsets):
                target_time = peak_time + offset
                
                for instrument_name, series_template in self.instruments.items():
                    try:
                        # Query JSOC for data
                        time_str = target_time.strftime('%Y.%m.%d_%H:%M:%S_TAI')
                        series = series_template.format(time=time_str)
                        
                        logger.info(f"Querying JSOC: {series}")
                        
                        # Search for available data
                        result = self.drms_client.query(series, key=['T_REC', 'DATAMIN', 'DATAMAX'][:1])
                        
                        if len(result) == 0:
                            logger.warning(f"No data found for {instrument_name} at {time_str}")
                            continue
                        
                        # Download the image
                        urls = self.drms_client.export(series)
                        if not urls.urls:
                            logger.warning(f"No download URLs for {instrument_name}")
                            continue
                        
                        # Process and upload first available image
                        url = urls.urls[0]
                        image_data = await self._download_and_process_fits(url)
                        
                        if image_data:
                            # Upload to GCS
                            blob_name = f"storm_data/{event_id}/{instrument_name}_{i:02d}_{target_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                            blob_url = await self._upload_to_gcs(image_data, blob_name)
                            
                            collected_images.append({
                                'event_id': event_id,
                                'event_type': 'STORM',
                                'instrument': instrument_name,
                                'timestamp': target_time,
                                'blob_name': blob_name,
                                'blob_url': blob_url,
                                'severity_score': event['severity_score']
                            })
                            
                            self.collection_stats['storm_images'] += 1
                            logger.info(f"✅ Collected {instrument_name} image for {event_id}")
                    
                    except Exception as e:
                        logger.error(f"❌ Error collecting {instrument_name} for {event_id}: {e}")
                        self.collection_stats['failed_downloads'] += 1
                        continue
            
            self.collection_stats['total_events_processed'] += 1
            return collected_images
            
        except Exception as e:
            logger.error(f"❌ Error processing storm event {event.get('event_id', 'unknown')}: {e}")
            return []
    
    async def collect_normal_images(self, time_range: Tuple[datetime, datetime], images_per_chunk: int = 2):
        """Collect normal solar activity images"""
        if not self.drms_client:
            logger.warning("DRMS client not available")
            return []
        
        start_time, end_time = time_range
        logger.info(f"☀️ Collecting normal images from {start_time} to {end_time}")
        
        collected_images = []
        
        # Sample times evenly across the chunk
        time_step = (end_time - start_time) / images_per_chunk
        
        for i in range(images_per_chunk):
            target_time = start_time + (i * time_step)
            
            for instrument_name, series_template in self.instruments.items():
                try:
                    # Query JSOC for data
                    time_str = target_time.strftime('%Y.%m.%d_%H:%M:%S_TAI')
                    series = series_template.format(time=time_str)
                    
                    # Search for available data
                    result = self.drms_client.query(series, key=['T_REC'][:1])
                    
                    if len(result) == 0:
                        continue
                    
                    # Download the image
                    urls = self.drms_client.export(series)
                    if not urls.urls:
                        continue
                    
                    # Process and upload first available image
                    url = urls.urls[0]
                    image_data = await self._download_and_process_fits(url)
                    
                    if image_data:
                        # Upload to GCS
                        blob_name = f"normal_data/{target_time.strftime('%Y%m%d')}/{instrument_name}_{target_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                        blob_url = await self._upload_to_gcs(image_data, blob_name)
                        
                        collected_images.append({
                            'event_id': f"NORMAL_{target_time.strftime('%Y%m%d_%H%M%S')}",
                            'event_type': 'NORMAL',
                            'instrument': instrument_name,
                            'timestamp': target_time,
                            'blob_name': blob_name,
                            'blob_url': blob_url,
                            'severity_score': 0
                        })
                        
                        self.collection_stats['normal_images'] += 1
                        logger.info(f"✅ Collected normal {instrument_name} image")
                        break  # Only get one instrument per time for normal data
                
                except Exception as e:
                    logger.error(f"❌ Error collecting normal {instrument_name}: {e}")
                    self.collection_stats['failed_downloads'] += 1
                    continue
        
        return collected_images
    
    async def _download_and_process_fits(self, url: str) -> Optional[bytes]:
        """Download FITS file and convert to JPEG"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Load FITS data
                fits_data = fits.open(io.BytesIO(response.content))
                image_data = fits_data[1].data if len(fits_data) > 1 else fits_data[0].data
                fits_data.close()
                
                # Normalize and convert to uint8
                if image_data is not None:
                    # Handle different data ranges
                    data_min, data_max = np.nanpercentile(image_data, [1, 99])
                    normalized = np.clip((image_data - data_min) / (data_max - data_min), 0, 1)
                    uint8_data = (normalized * 255).astype(np.uint8)
                    
                    # Convert to PIL Image and save as JPEG
                    pil_image = Image.fromarray(uint8_data)
                    
                    # Resize to standard size
                    pil_image = pil_image.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    # Convert to JPEG bytes
                    jpeg_buffer = io.BytesIO()
                    pil_image.save(jpeg_buffer, format='JPEG', quality=85)
                    return jpeg_buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Error processing FITS from {url}: {e}")
            return None
    
    async def _upload_to_gcs(self, image_data: bytes, blob_name: str) -> str:
        """Upload image data to Google Cloud Storage"""
        try:
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(image_data, content_type='image/jpeg')
            
            # Make blob publicly readable (optional)
            blob.make_public()
            
            return f"gs://{self.bucket_name}/{blob_name}"
            
        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
            return ""
    
    def save_metadata_to_bigquery(self, images_metadata: List[Dict]):
        """Save image metadata to BigQuery"""
        if not images_metadata:
            return
        
        df = pd.DataFrame(images_metadata)
        
        # Table schema for image metadata
        table_id = f"{self.project_id}.{self.dataset_name}.solar_images_metadata"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_IF_NEEDED"
        )
        
        job = self.bq_client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        
        logger.info(f"💾 Saved {len(images_metadata)} image metadata records to BigQuery")
    
    async def run_production_collection(self, max_storm_events: int = 20, normal_chunks: int = 5):
        """Run complete production data collection"""
        logger.info("🚀 Starting production solar storm data collection...")
        
        all_images_metadata = []
        
        # 1. Collect storm event images
        logger.info("⛈️ Phase 1: Collecting storm event images...")
        storm_events = self.get_storm_events(limit=max_storm_events)
        
        for _, event in storm_events.iterrows():
            storm_images = await self.collect_storm_images(event, images_per_event=2)
            all_images_metadata.extend(storm_images)
            
            # Small delay to be respectful to JSOC
            await asyncio.sleep(2)
        
        # 2. Collect normal period images
        logger.info("☀️ Phase 2: Collecting normal activity images...")
        normal_timeranges = self.get_normal_period_timeranges(days=5)[:normal_chunks]
        
        for time_range in normal_timeranges:
            normal_images = await self.collect_normal_images(time_range, images_per_chunk=1)
            all_images_metadata.extend(normal_images)
            
            # Small delay
            await asyncio.sleep(2)
        
        # 3. Save all metadata
        if all_images_metadata:
            self.save_metadata_to_bigquery(all_images_metadata)
        
        # 4. Print summary
        logger.info("✅ Production data collection completed!")
        logger.info(f"📊 Collection Summary:")
        logger.info(f"   🌩️ Storm images: {self.collection_stats['storm_images']}")
        logger.info(f"   ☀️ Normal images: {self.collection_stats['normal_images']}")
        logger.info(f"   ❌ Failed downloads: {self.collection_stats['failed_downloads']}")
        logger.info(f"   📋 Total events processed: {self.collection_stats['total_events_processed']}")
        logger.info(f"   🗂️ Total metadata records: {len(all_images_metadata)}")
        
        return all_images_metadata


async def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Solar Storm Data Collector")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--bucket-name", required=True, help="GCS Bucket name")
    parser.add_argument("--max-storms", type=int, default=10, help="Maximum storm events to process")
    parser.add_argument("--normal-chunks", type=int, default=5, help="Number of normal data chunks")
    
    args = parser.parse_args()
    
    collector = ProductionStormDataCollector(
        project_id=args.project_id,
        bucket_name=args.bucket_name
    )
    
    await collector.run_production_collection(
        max_storm_events=args.max_storms,
        normal_chunks=args.normal_chunks
    )


if __name__ == "__main__":
    asyncio.run(main())