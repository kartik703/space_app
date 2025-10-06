#!/usr/bin/env python3
"""
🚀 SPACE AI SYSTEM - QUICK START LAUNCHER
Simple script to launch the complete space AI system
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    print("=" * 80)
    print("🚀 SPACE AI SYSTEM - Real-Time Solar Storm Detection")
    print("=" * 80)
    print("📡 Live data from NASA, NOAA, and ground observatories")
    print("🤖 AI-powered solar flare detection and space weather prediction")
    print("📊 Real-time fusion AI dashboard with risk assessment")
    print("=" * 80)

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'requests', 'numpy', 'pandas', 
        'scikit-learn', 'opencv-python', 'torch', 'ultralytics'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    dirs = ['data', 'models', 'logs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"  📂 {dir_name}/")
    
    print("✅ Directory structure ready!")

def launch_system():
    """Launch the complete system"""
    print("\n🚀 Launching Space AI System...")
    
    # Check if data collection is needed
    data_dir = Path("data")
    if not data_dir.exists() or len(list(data_dir.glob("**/*"))) < 10:
        print("\n📡 Starting data collection (background)...")
        print("💡 This will collect real space data from NASA and NOAA")
        
        # Start data collector in background
        try:
            subprocess.Popen([
                sys.executable, "continuous_space_collector.py"
            ], cwd=os.getcwd())
            print("✅ Data collector started!")
        except Exception as e:
            print(f"⚠️  Data collector warning: {e}")
    
    # Launch dashboard
    print("\n📊 Launching AI Dashboard...")
    print("🌐 Dashboard will open at: http://localhost:8501")
    print("⏳ Please wait for Streamlit to start...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "fusion_ai_live.py", "--server.port", "8501"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Space AI System stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function"""
    print_banner()
    
    if not check_dependencies():
        return 1
    
    setup_directories()
    launch_system()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())