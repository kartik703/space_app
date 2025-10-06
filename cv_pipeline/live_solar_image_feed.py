#!/usr/bin/env python3
"""
Live Solar Image Feed Manager
Downloads and processes real-time solar images from multiple observatories
"""

import requests
import numpy as np
import cv2
from PIL import Image
import io
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import json
import time
from astropy.io import fits
import warnings

# Suppress FITS warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveSolarImageFeed:
    """Manager for live solar image feeds from multiple observatories"""
    
    def __init__(self, cache_dir: str = "data/live_solar_images"):
        """Initialize live solar image feed"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Solar observatory data sources
        self.sources = {
            'soho_lasco_c2': {
                'name': 'SOHO LASCO C2',
                'url': 'https://soho.nascom.nasa.gov/data/LATEST/current_c2.gif',
                'description': 'Coronagraph - CME detection',
                'format': 'gif'
            },
            'soho_lasco_c3': {
                'name': 'SOHO LASCO C3',
                'url': 'https://soho.nascom.nasa.gov/data/LATEST/current_c3.gif',
                'description': 'Wide coronagraph - Large CMEs',
                'format': 'gif'
            },
            'soho_eit_195': {
                'name': 'SOHO EIT 195Å',
                'url': 'https://soho.nascom.nasa.gov/data/LATEST/current_eit_195.gif',
                'description': 'Extreme UV - Active regions',
                'format': 'gif'
            },
            'soho_eit_304': {
                'name': 'SOHO EIT 304Å',
                'url': 'https://soho.nascom.nasa.gov/data/LATEST/current_eit_304.gif',
                'description': 'Extreme UV - Prominence/filaments',
                'format': 'gif'
            },
            'sdo_aia_171': {
                'name': 'SDO AIA 171Å',
                'url': 'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg',
                'description': 'Extreme UV - Corona loops',
                'format': 'jpg'
            },
            'sdo_aia_193': {
                'name': 'SDO AIA 193Å',
                'url': 'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193.jpg',
                'description': 'Extreme UV - Hot corona',
                'format': 'jpg'
            },
            'sdo_aia_211': {
                'name': 'SDO AIA 211Å',
                'url': 'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0211.jpg',
                'description': 'Extreme UV - Active regions',
                'format': 'jpg'
            },
            'sdo_aia_304': {
                'name': 'SDO AIA 304Å',
                'url': 'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0304.jpg',
                'description': 'Extreme UV - Prominence/filaments',
                'format': 'jpg'
            },
            'sdo_hmi_continuum': {
                'name': 'SDO HMI Continuum',
                'url': 'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIC.jpg',
                'description': 'Visible light - Sunspots',
                'format': 'jpg'
            },
            'sdo_hmi_magnetogram': {
                'name': 'SDO HMI Magnetogram',
                'url': 'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIM.jpg',
                'description': 'Magnetic field map',
                'format': 'jpg'
            }
        }
        
        # Fallback images for when live feeds fail
        self.fallback_images = []
        self._initialize_fallback_images()
    
    def _initialize_fallback_images(self):
        """Initialize fallback images from local dataset"""
        try:
            fallback_patterns = [
                "data/comprehensive_dataset/val/images/*.jpg",
                "data/enhanced_distinct_dataset/val/images/*.jpg"
            ]
            
            import glob
            for pattern in fallback_patterns:
                images = glob.glob(pattern)
                self.fallback_images.extend(images[:10])  # Limit fallbacks
            
            logger.info(f"Initialized {len(self.fallback_images)} fallback images")
            
        except Exception as e:
            logger.warning(f"Failed to initialize fallback images: {e}")
    
    def download_live_image(self, source_key: str, timeout: int = 10) -> Optional[Tuple[np.ndarray, Dict]]:
        """Download a live solar image from specified source"""
        try:
            if source_key not in self.sources:
                logger.error(f"Unknown source: {source_key}")
                return None
            
            source = self.sources[source_key]
            logger.info(f"Downloading live image from {source['name']}")
            
            # Download image
            headers = {
                'User-Agent': 'Space Weather Monitor/1.0 (Live Solar Feed)'
            }
            
            response = requests.get(source['url'], headers=headers, timeout=timeout)
            response.raise_for_status()
            
            # Convert to image
            image_data = io.BytesIO(response.content)
            pil_image = Image.open(image_data)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Metadata
            metadata = {
                'source': source['name'],
                'description': source['description'],
                'download_time': datetime.now(),
                'url': source['url'],
                'format': source['format'],
                'shape': image_array.shape,
                'live': True
            }
            
            # Cache the image
            cache_filename = f"{source_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cache_path = self.cache_dir / cache_filename
            
            pil_image.save(cache_path, 'JPEG', quality=90)
            metadata['cache_path'] = str(cache_path)
            
            logger.info(f"Successfully downloaded {source['name']} image: {image_array.shape}")
            return image_array, metadata
            
        except Exception as e:
            logger.error(f"Failed to download live image from {source_key}: {e}")
            return None
    
    def get_live_solar_images(self, max_images: int = 5, include_fallback: bool = True) -> List[Tuple[np.ndarray, Dict]]:
        """Get multiple live solar images from different sources"""
        images = []
        
        # Priority sources for live monitoring
        priority_sources = [
            'sdo_hmi_continuum',  # Sunspots visible
            'sdo_aia_193',        # Corona
            'sdo_aia_171',        # Corona loops
            'soho_lasco_c2',      # CMEs
            'sdo_aia_304',        # Prominences
            'sdo_hmi_magnetogram', # Magnetic field
            'soho_eit_195',       # Active regions
            'sdo_aia_211'         # Hot corona
        ]
        
        successful_downloads = 0
        
        for source_key in priority_sources:
            if successful_downloads >= max_images:
                break
                
            try:
                result = self.download_live_image(source_key)
                if result is not None:
                    images.append(result)
                    successful_downloads += 1
                    time.sleep(1)  # Be respectful to servers
                    
            except Exception as e:
                logger.warning(f"Failed to get image from {source_key}: {e}")
                continue
        
        # Add fallback images if needed and requested
        if include_fallback and successful_downloads < max_images and self.fallback_images:
            needed = max_images - successful_downloads
            
            import random
            fallback_selection = random.sample(
                self.fallback_images, 
                min(needed, len(self.fallback_images))
            )
            
            for fallback_path in fallback_selection:
                try:
                    # Load fallback image
                    pil_image = Image.open(fallback_path)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    image_array = np.array(pil_image)
                    
                    metadata = {
                        'source': 'Local Dataset (Fallback)',
                        'description': 'Historical solar data',
                        'download_time': datetime.now(),
                        'url': fallback_path,
                        'format': 'jpg',
                        'shape': image_array.shape,
                        'live': False
                    }
                    
                    images.append((image_array, metadata))
                    successful_downloads += 1
                    
                    if successful_downloads >= max_images:
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to load fallback image {fallback_path}: {e}")
                    continue
        
        logger.info(f"Retrieved {len(images)} solar images ({successful_downloads} total)")
        return images
    
    def get_latest_solar_activity_summary(self) -> Dict:
        """Get a summary of current solar activity from live images"""
        try:
            # Download key monitoring images
            key_sources = ['sdo_hmi_continuum', 'sdo_aia_193', 'soho_lasco_c2']
            
            summary = {
                'timestamp': datetime.now(),
                'sources_available': [],
                'sources_failed': [],
                'activity_indicators': {}
            }
            
            for source_key in key_sources:
                result = self.download_live_image(source_key)
                if result is not None:
                    image_array, metadata = result
                    summary['sources_available'].append(metadata['source'])
                    
                    # Basic activity analysis
                    if 'continuum' in source_key.lower():
                        # Check for sunspot activity (dark regions)
                        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                        mean_brightness = np.mean(gray)
                        dark_threshold = mean_brightness * 0.7
                        dark_pixels = np.sum(gray < dark_threshold)
                        dark_percentage = (dark_pixels / gray.size) * 100
                        
                        summary['activity_indicators']['sunspot_activity'] = {
                            'dark_region_percentage': dark_percentage,
                            'assessment': 'HIGH' if dark_percentage > 5 else 'MEDIUM' if dark_percentage > 2 else 'LOW'
                        }
                    
                    elif 'aia_193' in source_key.lower():
                        # Check for coronal activity (bright regions)
                        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                        bright_threshold = np.percentile(gray, 95)
                        bright_pixels = np.sum(gray > bright_threshold)
                        bright_percentage = (bright_pixels / gray.size) * 100
                        
                        summary['activity_indicators']['coronal_activity'] = {
                            'bright_region_percentage': bright_percentage,
                            'assessment': 'HIGH' if bright_percentage > 8 else 'MEDIUM' if bright_percentage > 4 else 'LOW'
                        }
                    
                    elif 'lasco' in source_key.lower():
                        # Check for CME activity (outward moving structures)
                        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                        edges = cv2.Canny(gray, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size * 100
                        
                        summary['activity_indicators']['cme_activity'] = {
                            'edge_density_percentage': edge_density,
                            'assessment': 'HIGH' if edge_density > 15 else 'MEDIUM' if edge_density > 8 else 'LOW'
                        }
                
                else:
                    summary['sources_failed'].append(self.sources[source_key]['name'])
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate solar activity summary: {e}")
            return {
                'timestamp': datetime.now(),
                'sources_available': [],
                'sources_failed': list(self.sources.keys()),
                'activity_indicators': {},
                'error': str(e)
            }
    
    def cleanup_old_cache(self, max_age_hours: int = 24):
        """Clean up old cached images"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            deleted_count = 0
            for cache_file in self.cache_dir.glob("*.jpg"):
                try:
                    # Get file modification time
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    
                    if file_time < cutoff_time:
                        cache_file.unlink()
                        deleted_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old cache files")
                
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")

# Global instance
live_solar_feed = LiveSolarImageFeed()

def get_live_solar_images(max_images: int = 3) -> List[Tuple[np.ndarray, Dict]]:
    """Convenience function to get live solar images"""
    return live_solar_feed.get_live_solar_images(max_images)

def get_solar_activity_summary() -> Dict:
    """Convenience function to get solar activity summary"""
    return live_solar_feed.get_latest_solar_activity_summary()

if __name__ == "__main__":
    # Test the live solar image feed
    print("🌞 Testing Live Solar Image Feed...")
    
    feed = LiveSolarImageFeed()
    
    # Test downloading images
    images = feed.get_live_solar_images(max_images=3)
    print(f"Retrieved {len(images)} live solar images")
    
    for i, (image_array, metadata) in enumerate(images):
        print(f"Image {i+1}: {metadata['source']} - {image_array.shape} - Live: {metadata['live']}")
    
    # Test activity summary
    summary = feed.get_latest_solar_activity_summary()
    print(f"Solar Activity Summary: {len(summary['sources_available'])} sources available")
    
    for indicator, data in summary['activity_indicators'].items():
        print(f"  {indicator}: {data['assessment']}")
    
    print("✅ Live solar feed test completed!")