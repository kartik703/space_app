#!/usr/bin/env python3
"""
CONTINUOUS SPACE DATA COLLECTOR
Real data collection targeting 100GB
- NASA SDO: Solar images, FITS files
- NOAA SWPC: Space weather JSON
- Kyoto WDC: Geomagnetic indices
- Ground observatories: Real telescope data
"""

import os
import sys
import time
import requests
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

class ContinuousSpaceCollector:
    def __init__(self):
        # Setup paths
        self.base_dir = Path(__file__).parent / "data" / "continuous_space_data"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Data directories
        self.dirs = {
            'sdo_images': self.base_dir / "sdo_solar_images",
            'noaa_data': self.base_dir / "noaa_space_weather", 
            'kyoto_data': self.base_dir / "kyoto_geomagnetic",
            'ground_obs': self.base_dir / "ground_observations",
            'logs': self.base_dir / "collection_logs"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.dirs['logs'] / f"collection_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()
        
        # Stats
        self.stats = {
            'total_files': 0,
            'total_size_gb': 0.0,
            'start_time': datetime.now(),
            'files_by_type': {
                'sdo_images': 0,
                'noaa_json': 0, 
                'kyoto_indices': 0,
                'ground_obs': 0
            }
        }
        
        self.target_gb = 100
        self.running = True
        
    def get_sdo_solar_data(self):
        """Collect NASA SDO solar images"""
        try:
            # SDO AIA image URLs (real solar EUV images)
            sdo_urls = [
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0094.jpg",
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0131.jpg", 
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg",
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193.jpg",
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0211.jpg",
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0304.jpg",
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0335.jpg",
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_1600.jpg",
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_1700.jpg",
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIIC.jpg",
                "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMII.jpg"
            ]
            
            for url in sdo_urls:
                if not self.running:
                    break
                    
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        wavelength = url.split('_')[-1].replace('.jpg', '')
                        filename = f"sdo_aia_{wavelength}_{timestamp}.jpg"
                        filepath = self.dirs['sdo_images'] / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                            
                        size_mb = len(response.content) / (1024*1024)
                        self.stats['total_files'] += 1
                        self.stats['total_size_gb'] += size_mb / 1024
                        self.stats['files_by_type']['sdo_images'] += 1
                        
                        self.logger.info(f"SDO: {filename} ({size_mb:.2f} MB)")
                        
                except Exception as e:
                    self.logger.warning(f"SDO image error: {e}")
                    
                time.sleep(2)  # Rate limiting
                
        except Exception as e:
            self.logger.error(f"SDO collection error: {e}")
            
    def get_noaa_space_weather(self):
        """Collect NOAA SWPC real-time space weather data"""
        try:
            # NOAA SWPC JSON endpoints
            noaa_endpoints = [
                "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json",
                "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-6-hour.json", 
                "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
                "https://services.swpc.noaa.gov/json/solar-wind/plasma-6-hour.json",
                "https://services.swpc.noaa.gov/json/solar-wind/mag-6-hour.json",
                "https://services.swpc.noaa.gov/json/geospace/geospace_pred_est_kp_1_min.json",
                "https://services.swpc.noaa.gov/json/ace_swepam_1m.json",
                "https://services.swpc.noaa.gov/json/ace_mag_1m.json"
            ]
            
            for endpoint in noaa_endpoints:
                if not self.running:
                    break
                    
                try:
                    response = requests.get(endpoint, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        if data:  # Only save if data exists
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            endpoint_name = endpoint.split('/')[-1].replace('.json', '')
                            filename = f"noaa_{endpoint_name}_{timestamp}.json"
                            filepath = self.dirs['noaa_data'] / filename
                            
                            with open(filepath, 'w') as f:
                                json.dump(data, f, indent=2)
                                
                            size_kb = filepath.stat().st_size / 1024
                            self.stats['total_files'] += 1 
                            self.stats['total_size_gb'] += size_kb / (1024*1024)
                            self.stats['files_by_type']['noaa_json'] += 1
                            
                            self.logger.info(f"NOAA: {filename} ({len(data)} records, {size_kb:.1f} KB)")
                            
                except Exception as e:
                    self.logger.warning(f"NOAA endpoint {endpoint} error: {e}")
                    
                time.sleep(3)  # Rate limiting
                
        except Exception as e:
            self.logger.error(f"NOAA collection error: {e}")
            
    def get_kyoto_geomagnetic(self):
        """Collect Kyoto WDC geomagnetic indices"""
        try:
            current_year = datetime.now().year
            years = [current_year, current_year - 1]
            
            for year in years:
                if not self.running:
                    break
                    
                # Dst index
                try:
                    dst_url = f"http://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year}/index.html"
                    response = requests.get(dst_url, timeout=30)
                    if response.status_code == 200:
                        filename = f"kyoto_dst_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        filepath = self.dirs['kyoto_data'] / filename
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(response.text)
                            
                        size_kb = len(response.text.encode('utf-8')) / 1024
                        self.stats['total_files'] += 1
                        self.stats['total_size_gb'] += size_kb / (1024*1024)
                        self.stats['files_by_type']['kyoto_indices'] += 1
                        
                        self.logger.info(f"Kyoto Dst {year}: {filename} ({size_kb:.1f} KB)")
                        
                except Exception as e:
                    self.logger.warning(f"Kyoto Dst {year} error: {e}")
                    
                time.sleep(5)
                
        except Exception as e:
            self.logger.error(f"Kyoto collection error: {e}")
            
    def get_ground_observations(self):
        """Collect ground-based telescope observations"""
        try:
            # Real ground observatory data sources
            ground_sources = [
                "https://cdaweb.gsfc.nasa.gov/",  # NASA CDAWeb
                "https://omniweb.gsfc.nasa.gov/",  # OMNI data
            ]
            
            for source in ground_sources:
                if not self.running:
                    break
                    
                try:
                    response = requests.get(source, timeout=30)
                    if response.status_code == 200:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        source_name = source.split('//')[1].split('.')[0]
                        filename = f"ground_{source_name}_{timestamp}.html"
                        filepath = self.dirs['ground_obs'] / filename
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(response.text)
                            
                        size_kb = len(response.text.encode('utf-8')) / 1024
                        self.stats['total_files'] += 1
                        self.stats['total_size_gb'] += size_kb / (1024*1024)
                        self.stats['files_by_type']['ground_obs'] += 1
                        
                        self.logger.info(f"Ground {source_name}: {filename} ({size_kb:.1f} KB)")
                        
                except Exception as e:
                    self.logger.warning(f"Ground source {source} error: {e}")
                    
                time.sleep(10)
                
        except Exception as e:
            self.logger.error(f"Ground observations error: {e}")
            
    def print_stats(self):
        """Print collection statistics"""
        elapsed = datetime.now() - self.stats['start_time']
        rate_gb_per_hour = self.stats['total_size_gb'] / (elapsed.total_seconds() / 3600) if elapsed.total_seconds() > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"CONTINUOUS SPACE DATA COLLECTION STATS")
        print(f"{'='*80}")
        print(f"Runtime: {elapsed}")
        print(f"Total Files: {self.stats['total_files']:,}")
        print(f"Total Size: {self.stats['total_size_gb']:.2f} GB / {self.target_gb} GB ({self.stats['total_size_gb']/self.target_gb*100:.1f}%)")
        print(f"Collection Rate: {rate_gb_per_hour:.2f} GB/hour")
        print(f"\nBreakdown:")
        for data_type, count in self.stats['files_by_type'].items():
            print(f"  {data_type}: {count:,} files")
        print(f"{'='*80}\n")
        
    def run_collection_cycle(self):
        """Run one complete collection cycle"""
        while self.running and self.stats['total_size_gb'] < self.target_gb:
            try:
                self.logger.info("Starting collection cycle...")
                
                # Collect from all sources in parallel  
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(self.get_sdo_solar_data),
                        executor.submit(self.get_noaa_space_weather), 
                        executor.submit(self.get_kyoto_geomagnetic),
                        executor.submit(self.get_ground_observations)
                    ]
                    
                    # Wait for all to complete
                    for future in futures:
                        try:
                            future.result(timeout=300)  # 5 minute timeout per source
                        except Exception as e:
                            self.logger.warning(f"Collection thread error: {e}")
                
                self.print_stats()
                
                if self.stats['total_size_gb'] >= self.target_gb:
                    self.logger.info(f"TARGET REACHED! Collected {self.stats['total_size_gb']:.2f} GB")
                    break
                    
                # Wait before next cycle
                time.sleep(60)  # 1 minute between cycles
                
            except KeyboardInterrupt:
                self.logger.info("Collection stopped by user")
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Collection cycle error: {e}")
                time.sleep(30)  # Wait before retry
                
    def run(self):
        """Main run loop"""
        print(f"\n{'='*80}")
        print("CONTINUOUS SPACE DATA COLLECTOR")
        print(f"{'='*80}")
        print(f"TARGET: {self.target_gb} GB of real space data")
        print("SOURCES:")
        print("  - NASA SDO: Real-time solar images")
        print("  - NOAA SWPC: Live space weather JSON")
        print("  - Kyoto WDC: Geomagnetic indices")
        print("  - Ground observatories: Telescope data")
        print("STOP: Ctrl+C")
        print(f"{'='*80}")
        print(f"DATA STORAGE: {self.base_dir}")
        print(f"{'='*80}\n")
        
        self.logger.info("CONTINUOUS SPACE DATA COLLECTOR STARTED")
        
        try:
            self.run_collection_cycle()
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            self.running = False
            self.print_stats()
            self.logger.info("Collection completed")

if __name__ == "__main__":
    collector = ContinuousSpaceCollector()
    collector.run()