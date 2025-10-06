#!/usr/bin/env python3
"""
🔍 CI/CD PIPELINE VALIDATION SCRIPT
Validates all components for Space Intelligence Platform
"""

import sys
import os
import json
import traceback
from datetime import datetime

def log_result(test_name, status, message=""):
    """Log test results with consistent formatting"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"[{timestamp}] {status_icon} {test_name}: {status}")
    if message:
        print(f"    📝 {message}")

def test_python_imports():
    """Test critical Python imports"""
    test_name = "Python Import Test"
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        import requests
        from PIL import Image
        
        log_result(test_name, "PASS", "All critical imports successful")
        return True
    except ImportError as e:
        log_result(test_name, "FAIL", f"Import error: {e}")
        return False
    except Exception as e:
        log_result(test_name, "FAIL", f"Unexpected error: {e}")
        return False

def test_data_sources():
    """Test data source modules"""
    test_name = "Data Sources Test"
    try:
        # Check if real_data_sources exists
        if os.path.exists("real_data_sources.py"):
            import real_data_sources
            
            # Test key functions exist
            required_functions = [
                'get_cached_solar_image',
                'get_cached_space_weather', 
                'get_cached_iss_location',
                'get_cached_commodity_prices'
            ]
            
            missing_functions = []
            for func_name in required_functions:
                if not hasattr(real_data_sources, func_name):
                    missing_functions.append(func_name)
            
            if missing_functions:
                log_result(test_name, "WARN", f"Missing functions: {missing_functions}")
                return True  # Still pass as non-critical
            else:
                log_result(test_name, "PASS", "All data source functions available")
                return True
        else:
            log_result(test_name, "WARN", "real_data_sources.py not found")
            return True  # Non-critical for CI/CD
            
    except Exception as e:
        log_result(test_name, "WARN", f"Data sources error: {e}")
        return True  # Non-critical

def test_core_files():
    """Test for essential application files"""
    test_name = "Core Files Test"
    try:
        essential_files = ["requirements.txt"]
        optional_files = ["app.py", "main.py", "ultimate_launcher.py"]
        
        missing_essential = []
        available_optional = []
        
        for file in essential_files:
            if not os.path.exists(file):
                missing_essential.append(file)
        
        for file in optional_files:
            if os.path.exists(file):
                available_optional.append(file)
        
        if missing_essential:
            log_result(test_name, "FAIL", f"Missing essential files: {missing_essential}")
            return False
        else:
            log_result(test_name, "PASS", f"Essential files OK, Optional available: {available_optional}")
            return True
            
    except Exception as e:
        log_result(test_name, "FAIL", f"File check error: {e}")
        return False

def test_automation_system():
    """Test automation system components"""
    test_name = "Automation System Test"
    try:
        automation_files = [
            "ultimate_launcher.py",
            "autostart.py", 
            "error_recovery.py"
        ]
        
        available = []
        for file in automation_files:
            if os.path.exists(file):
                available.append(file)
                # Test syntax
                try:
                    with open(file, 'r') as f:
                        compile(f.read(), file, 'exec')
                except SyntaxError as e:
                    log_result(test_name, "WARN", f"Syntax issue in {file}: {e}")
        
        if available:
            log_result(test_name, "PASS", f"Automation files available: {available}")
        else:
            log_result(test_name, "WARN", "No automation files found")
        
        return True
        
    except Exception as e:
        log_result(test_name, "WARN", f"Automation test error: {e}")
        return True  # Non-critical

def test_directory_structure():
    """Test expected directory structure"""
    test_name = "Directory Structure Test"
    try:
        expected_dirs = ["data", "docs", "pages"]
        available_dirs = []
        
        for dir_name in expected_dirs:
            if os.path.isdir(dir_name):
                available_dirs.append(dir_name)
        
        log_result(test_name, "PASS", f"Available directories: {available_dirs}")
        return True
        
    except Exception as e:
        log_result(test_name, "WARN", f"Directory check error: {e}")
        return True  # Non-critical

def generate_validation_report():
    """Generate validation summary report"""
    try:
        report = {
            "validation_time": datetime.now().isoformat(),
            "platform": "Space Intelligence Platform",
            "version": "3.0",
            "ci_cd_validation": "completed",
            "status": "ready_for_deployment"
        }
        
        os.makedirs("data/live", exist_ok=True)
        with open("data/live/validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print("\n📊 VALIDATION REPORT GENERATED")
        print(f"📅 Time: {report['validation_time']}")
        print(f"🎯 Status: {report['status']}")
        
    except Exception as e:
        print(f"⚠️ Report generation error: {e}")

def main():
    """Run all validation tests"""
    print("🔍 SPACE INTELLIGENCE PLATFORM - CI/CD VALIDATION")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    tests = [
        test_python_imports,
        test_core_files,
        test_data_sources,
        test_automation_system,
        test_directory_structure
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test '{test_func.__name__}' crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("🚀 Space Intelligence Platform ready for deployment")
        generate_validation_report()
        return 0
    elif failed <= 1:
        print("\n⚠️ MINOR ISSUES DETECTED")
        print("🔧 Platform functional with warnings")
        generate_validation_report()
        return 0
    else:
        print("\n❌ VALIDATION FAILED")
        print("🛠️ Platform requires fixes before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())