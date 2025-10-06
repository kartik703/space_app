#!/usr/bin/env python3
"""
JSOC Data Access Diagnostic Tool
Tests different query formats and date ranges to resolve 'Invalid KeyLink' issues
"""

import drms
import logging
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_jsoc_access():
    """Test JSOC data access with different approaches"""
    
    try:
        # Initialize DRMS client
        logger.info("🔌 Connecting to JSOC/DRMS...")
        client = drms.Client(email="goswamikartik429@gmail.com")
        logger.info("✅ DRMS client connected successfully")
        
        # Test 1: Very recent data (yesterday)
        logger.info("\n📅 Test 1: Recent data (yesterday)")
        test_recent_data(client)
        
        # Test 2: Known good period (2011-2015 when SDO was fully operational)
        logger.info("\n📅 Test 2: Known good period (2014)")
        test_known_good_period(client)
        
        # Test 3: Different series
        logger.info("\n📅 Test 3: Different AIA series")
        test_different_series(client)
        
        # Test 4: Simple query with minimal parameters
        logger.info("\n📅 Test 4: Minimal query")
        test_minimal_query(client)
        
    except Exception as e:
        logger.error(f"DRMS connection failed: {e}")
        return False
    
    return True

def test_recent_data(client):
    """Test with very recent data"""
    try:
        # Yesterday
        yesterday = datetime.utcnow() - timedelta(days=1)
        start_str = yesterday.strftime("%Y.%m.%d_%H:%M:%S_TAI")
        end_str = (yesterday + timedelta(hours=1)).strftime("%Y.%m.%d_%H:%M:%S_TAI")
        
        query = f"aia.lev1_euv_12s[{start_str}-{end_str}@600s][193]"
        logger.info(f"Query: {query}")
        
        result = client.query(query, key='T_REC,WAVELNTH,QUALITY')
        logger.info(f"✅ Found {len(result)} records")
        if not result.empty:
            logger.info(f"Sample record: {result.iloc[0].to_dict()}")
        
    except Exception as e:
        logger.error(f"❌ Recent data test failed: {e}")

def test_known_good_period(client):
    """Test with known good period when SDO was fully operational"""
    try:
        # 2014-01-01 - known good period
        start_str = "2014.01.01_12:00:00_TAI"
        end_str = "2014.01.01_13:00:00_TAI"
        
        query = f"aia.lev1_euv_12s[{start_str}-{end_str}@600s][193]"
        logger.info(f"Query: {query}")
        
        result = client.query(query, key='T_REC,WAVELNTH,QUALITY')
        logger.info(f"✅ Found {len(result)} records")
        if not result.empty:
            logger.info(f"Sample record: {result.iloc[0].to_dict()}")
        
    except Exception as e:
        logger.error(f"❌ Known good period test failed: {e}")

def test_different_series(client):
    """Test different AIA data series"""
    try:
        # Try AIA lev1 (less processed)
        start_str = "2014.01.01_12:00:00_TAI"
        end_str = "2014.01.01_12:30:00_TAI"
        
        # Test different series
        series_list = [
            "aia.lev1_euv_12s",  # Current
            "aia.lev1",          # Alternative
            "aia.lev1_uv_24s"    # UV version
        ]
        
        for series in series_list:
            try:
                query = f"{series}[{start_str}-{end_str}@600s][193]"
                logger.info(f"Testing series: {query}")
                
                result = client.query(query, key='T_REC,WAVELNTH')
                logger.info(f"✅ {series}: Found {len(result)} records")
                
            except Exception as e:
                logger.error(f"❌ {series} failed: {e}")
        
    except Exception as e:
        logger.error(f"❌ Different series test failed: {e}")

def test_minimal_query(client):
    """Test with absolute minimal query"""
    try:
        # Very simple query - just get latest few records
        query = "aia.lev1_euv_12s[][193]{2014.01.01_12:00:00_TAI/1h@10m}"
        logger.info(f"Minimal query: {query}")
        
        result = client.query(query, key='T_REC')
        logger.info(f"✅ Minimal query: Found {len(result)} records")
        
    except Exception as e:
        logger.error(f"❌ Minimal query failed: {e}")

def test_available_series(client):
    """Check what series are available"""
    try:
        logger.info("\n📊 Checking available series...")
        
        # Look for AIA series
        series_info = client.series(r'aia\.lev1.*')
        logger.info(f"Available AIA series: {len(series_info)} found")
        
        for series in series_info.index[:5]:  # Show first 5
            logger.info(f"  - {series}")
            
    except Exception as e:
        logger.error(f"❌ Series check failed: {e}")

if __name__ == "__main__":
    logger.info("🔍 JSOC Data Access Diagnostic Starting...")
    success = test_jsoc_access()
    
    if success:
        logger.info("✅ Diagnostic completed - check results above")
    else:
        logger.error("❌ Diagnostic failed - DRMS connection issues")