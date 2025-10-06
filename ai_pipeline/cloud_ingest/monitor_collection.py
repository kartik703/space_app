#!/usr/bin/env python3
"""
Massive Solar Data Collection Monitor
Monitors the progress of 100GB+ solar data collection.
"""

import os
import sqlite3
import time
from pathlib import Path
from datetime import datetime
import argparse


def format_bytes(bytes_value):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def monitor_collection(data_dir="data/massive_solar_dataset", refresh_seconds=30):
    """Monitor the massive data collection progress."""
    
    data_path = Path(data_dir)
    db_path = data_path / "massive_dataset_metadata.db"
    
    print("🌞 MASSIVE SOLAR DATA COLLECTION MONITOR")
    print("=" * 60)
    
    while True:
        try:
            # Check if database exists
            if not db_path.exists():
                print(f"⏳ Waiting for collection to start... ({datetime.now().strftime('%H:%M:%S')})")
                time.sleep(refresh_seconds)
                continue
            
            # Get statistics from database
            with sqlite3.connect(db_path) as conn:
                # Get total counts
                result = conn.execute("""
                    SELECT 
                        COUNT(*) as total_images,
                        SUM(file_size_bytes) as total_bytes,
                        COUNT(CASE WHEN data_type='storm' THEN 1 END) as storm_images,
                        COUNT(CASE WHEN data_type='pre_storm' THEN 1 END) as pre_storm_images,
                        COUNT(CASE WHEN data_type='post_storm' THEN 1 END) as post_storm_images,
                        COUNT(CASE WHEN data_type='normal' THEN 1 END) as normal_images
                    FROM solar_images
                """).fetchone()
                
                if result and result[0] > 0:
                    total_images, total_bytes, storm_images, pre_storm_images, post_storm_images, normal_images = result
                    
                    # Calculate progress
                    target_gb = 100
                    current_gb = (total_bytes or 0) / (1024**3)
                    progress_percent = (current_gb / target_gb) * 100
                    
                    # Get latest image info
                    latest = conn.execute("""
                        SELECT observation_time, wavelength, data_type, storm_event_id
                        FROM solar_images 
                        ORDER BY download_time DESC 
                        LIMIT 1
                    """).fetchone()
                    
                    # Calculate collection rate
                    first_time = conn.execute("""
                        SELECT MIN(download_time) FROM solar_images
                    """).fetchone()[0]
                    
                    if first_time:
                        start_time = datetime.fromisoformat(first_time)
                        elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
                        images_per_hour = total_images / elapsed_hours if elapsed_hours > 0 else 0
                        gb_per_hour = current_gb / elapsed_hours if elapsed_hours > 0 else 0
                        
                        # Estimate completion time
                        remaining_gb = target_gb - current_gb
                        hours_remaining = remaining_gb / gb_per_hour if gb_per_hour > 0 else 0
                    else:
                        images_per_hour = 0
                        gb_per_hour = 0
                        hours_remaining = 0
                        elapsed_hours = 0
                    
                    # Clear screen and show status
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    print("🌞 MASSIVE SOLAR DATA COLLECTION MONITOR")
                    print("=" * 60)
                    print(f"🎯 Target: {target_gb} GB")
                    print(f"📊 Progress: {current_gb:.2f} GB ({progress_percent:.1f}%)")
                    print(f"📈 Progress Bar: {'█' * int(progress_percent/2):<50} {progress_percent:.1f}%")
                    print()
                    
                    print(f"📸 Images Collected: {total_images:,}")
                    print(f"  🌪️  Storm periods: {storm_images:,}")
                    print(f"  ⚡ Pre-storm: {pre_storm_images:,}")
                    print(f"  🌅 Post-storm: {post_storm_images:,}")
                    print(f"  🌞 Normal periods: {normal_images:,}")
                    print()
                    
                    if latest:
                        obs_time, wavelength, data_type, event_id = latest
                        print(f"🔍 Latest Image: {data_type} | {wavelength}Å | {obs_time}")
                        if event_id:
                            print(f"🌪️  Event: {event_id}")
                    print()
                    
                    print(f"⏱️  Collection Time: {elapsed_hours:.1f} hours")
                    print(f"🚀 Speed: {images_per_hour:.0f} images/hour | {gb_per_hour:.2f} GB/hour")
                    if hours_remaining > 0:
                        print(f"⏳ ETA: {hours_remaining:.1f} hours remaining")
                    print()
                    
                    # File system statistics
                    for subdir, label in [
                        ("storm_periods", "Storm Periods"),
                        ("pre_storm_periods", "Pre-Storm"),
                        ("post_storm_periods", "Post-Storm"),
                        ("normal_periods", "Normal Periods")
                    ]:
                        subdir_path = data_path / subdir
                        if subdir_path.exists():
                            jpg_files = list(subdir_path.glob("*.jpg"))
                            total_size = sum(f.stat().st_size for f in jpg_files)
                            print(f"📁 {label}: {len(jpg_files):,} files ({format_bytes(total_size)})")
                    
                    print()
                    print(f"🔄 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("=" * 60)
                    
                    # Check if target reached
                    if current_gb >= target_gb:
                        print("🎉 TARGET REACHED! 100GB+ of real solar data collected!")
                        print("✅ Ready for YOLO training and storm prediction!")
                        break
                        
                else:
                    print(f"⏳ Collection starting... ({datetime.now().strftime('%H:%M:%S')})")
            
            time.sleep(refresh_seconds)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(refresh_seconds)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitor massive solar data collection")
    parser.add_argument("--data-dir", default="data/massive_solar_dataset", help="Data directory to monitor")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    try:
        monitor_collection(args.data_dir, args.refresh)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()