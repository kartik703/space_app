"""
🔍 REAL DATA SOURCE VERIFICATION SCRIPT
This script validates all data sources are live and returning real data
"""

import requests
from datetime import datetime
import json
import pandas as pd
from real_data_sources import *

def verify_all_data_sources():
    """Comprehensive verification of all live data sources"""
    
    print("🚀 SPACE INTELLIGENCE PLATFORM - DATA SOURCE VERIFICATION")
    print("="*60)
    print(f"🕒 Verification Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    # 1. Verify NASA Solar Images
    print("\n🌞 NASA SOLAR DYNAMICS OBSERVATORY (SDO)")
    print("-" * 40)
    try:
        solar_image = get_cached_solar_image()
        if solar_image:
            print("✅ NASA SDO Solar Images: LIVE & OPERATIONAL")
            print(f"   🔗 Source: https://sdo.gsfc.nasa.gov/assets/img/latest/")
            print(f"   📊 Image Size: Available")
            results['nasa_sdo'] = 'LIVE'
        else:
            print("❌ NASA SDO Solar Images: UNAVAILABLE")
            results['nasa_sdo'] = 'OFFLINE'
    except Exception as e:
        print(f"❌ NASA SDO Error: {e}")
        results['nasa_sdo'] = 'ERROR'
    
    # 2. Verify NOAA Space Weather
    print("\n🌪️ NOAA SPACE WEATHER PREDICTION CENTER")
    print("-" * 40)
    try:
        space_weather = get_cached_space_weather()
        if space_weather is not None and not space_weather.empty:
            print("✅ NOAA SWPC Space Weather: LIVE & OPERATIONAL")
            print(f"   🔗 Source: https://services.swpc.noaa.gov/")
            print(f"   📊 Data Points: {len(space_weather)} records")
            print(f"   🕒 Latest Record: {space_weather.iloc[-1]['timestamp'] if 'timestamp' in space_weather.columns else 'Available'}")
            results['noaa_swpc'] = 'LIVE'
        else:
            print("❌ NOAA SWPC Space Weather: NO DATA")
            results['noaa_swpc'] = 'NO_DATA'
    except Exception as e:
        print(f"❌ NOAA SWPC Error: {e}")
        results['noaa_swpc'] = 'ERROR'
    
    # 3. Verify ISS Location
    print("\n🏠 INTERNATIONAL SPACE STATION (ISS)")
    print("-" * 40)
    try:
        iss_location = get_cached_iss_location()
        if iss_location:
            print("✅ ISS Real-time Location: LIVE & OPERATIONAL")
            print(f"   🔗 Source: http://api.open-notify.org/iss-now.json")
            print(f"   📍 Current Position: Lat {iss_location['latitude']:.2f}, Lon {iss_location['longitude']:.2f}")
            print(f"   🕒 Timestamp: {iss_location['timestamp']}")
            results['iss_tracker'] = 'LIVE'
        else:
            print("❌ ISS Location: UNAVAILABLE")
            results['iss_tracker'] = 'OFFLINE'
    except Exception as e:
        print(f"❌ ISS Tracker Error: {e}")
        results['iss_tracker'] = 'ERROR'
    
    # 4. Verify Commodity Prices
    print("\n💰 REAL COMMODITY MARKET PRICES")
    print("-" * 40)
    try:
        commodity_prices = get_cached_commodity_prices()
        if commodity_prices:
            print("✅ Live Commodity Prices: LIVE & OPERATIONAL")
            print(f"   🔗 Source: Financial markets API")
            print(f"   💎 Available Commodities: {len(commodity_prices)} items")
            for commodity, price in list(commodity_prices.items())[:3]:
                print(f"   💰 {commodity}: ${price:.2f}")
            results['commodity_markets'] = 'LIVE'
        else:
            print("❌ Commodity Prices: UNAVAILABLE") 
            results['commodity_markets'] = 'OFFLINE'
    except Exception as e:
        print(f"❌ Commodity Markets Error: {e}")
        results['commodity_markets'] = 'ERROR'
    
    # 5. Verify Solar Flares
    print("\n☀️ REAL SOLAR FLARE DATA (NOAA GOES)")
    print("-" * 40)
    try:
        solar_flares = get_cached_solar_flares()
        if solar_flares is not None and not solar_flares.empty:
            print("✅ NOAA GOES Solar Flares: LIVE & OPERATIONAL")
            print(f"   🔗 Source: https://services.swpc.noaa.gov/json/goes/")
            print(f"   📊 X-Ray Data Points: {len(solar_flares)} records")
            print(f"   🔥 Latest Flux: {solar_flares.iloc[-1]['flux']:.2e} if available")
            results['solar_flares'] = 'LIVE'
        else:
            print("❌ Solar Flares Data: NO DATA")
            results['solar_flares'] = 'NO_DATA'
    except Exception as e:
        print(f"❌ Solar Flares Error: {e}")
        results['solar_flares'] = 'ERROR'
    
    # 6. Verify Space Alerts
    print("\n⚠️ SPACE WEATHER ALERTS")
    print("-" * 40)
    try:
        space_alerts = get_cached_space_alerts()
        if space_alerts:
            print("✅ NOAA Space Weather Alerts: LIVE & OPERATIONAL")
            print(f"   🔗 Source: https://services.swpc.noaa.gov/products/alerts.json")
            print(f"   ⚠️ Active Alerts: {len(space_alerts)} items")
            results['space_alerts'] = 'LIVE'
        else:
            print("✅ Space Weather Alerts: OPERATIONAL (No current alerts)")
            results['space_alerts'] = 'LIVE'
    except Exception as e:
        print(f"❌ Space Alerts Error: {e}")
        results['space_alerts'] = 'ERROR'
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    live_count = sum(1 for status in results.values() if status == 'LIVE')
    total_count = len(results)
    
    for source, status in results.items():
        emoji = "✅" if status == 'LIVE' else "❌"
        print(f"{emoji} {source.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 OVERALL STATUS: {live_count}/{total_count} Data Sources LIVE")
    
    if live_count == total_count:
        print("🚀 ALL SYSTEMS OPERATIONAL - READY FOR INVESTOR DEMO!")
    elif live_count >= total_count * 0.8:
        print("⚡ MOSTLY OPERATIONAL - Good for demo with minor fallbacks")
    else:
        print("⚠️ MULTIPLE SOURCES DOWN - Check network connectivity")
    
    print(f"\n🕒 Verification completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    verify_all_data_sources()