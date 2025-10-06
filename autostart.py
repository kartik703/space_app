#!/usr/bin/env python3
"""
🚀 Space Intelligence Platform - Automated Startup System
Comprehensive automation for environment setup, data pipeline, and application launch
"""

import os
import sys
import subprocess
import platform
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import threading
import signal
import atexit

class SpaceIntelligenceAutomation:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs" 
        self.config_dir = self.base_dir / "config"
        self.is_windows = platform.system() == "Windows"
        self.python_cmd = "python" if self.is_windows else "python3"
        self.services = []
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        self.logs_dir.mkdir(exist_ok=True)
        log_file = self.logs_dir / f"automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("SpaceIntelligence")
        
    def log_info(self, message):
        """Enhanced logging with console output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"🚀 [{timestamp}] {message}"
        self.logger.info(formatted_msg)
        print(formatted_msg)
        
    def log_error(self, message):
        """Enhanced error logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"❌ [{timestamp}] {message}"
        self.logger.error(formatted_msg)
        print(formatted_msg)
        
    def run_command(self, cmd, check=True, capture_output=False):
        """Execute command with proper error handling"""
        try:
            if isinstance(cmd, str):
                cmd = cmd.split()
            
            if capture_output:
                result = subprocess.run(cmd, check=check, capture_output=True, text=True, cwd=self.base_dir)
                return result.stdout.strip()
            else:
                subprocess.run(cmd, check=check, cwd=self.base_dir)
                return True
        except subprocess.CalledProcessError as e:
            self.log_error(f"Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
            self.log_error(f"Error: {e}")
            return False
        except Exception as e:
            self.log_error(f"Unexpected error running command: {e}")
            return False
            
    def check_system_requirements(self):
        """Verify system requirements"""
        self.log_info("Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.log_error("Python 3.8+ required")
            return False
            
        # Check available disk space (at least 1GB)
        import shutil
        total, used, free = shutil.disk_usage(self.base_dir)
        if free < 1_000_000_000:  # 1GB
            self.log_error("Insufficient disk space (need at least 1GB)")
            return False
            
        self.log_info(f"✅ System requirements met - Python {python_version.major}.{python_version.minor}")
        return True
        
    def setup_environment(self):
        """Setup Python environment and install dependencies"""
        self.log_info("Setting up Python environment...")
        
        # Create virtual environment if it doesn't exist
        venv_path = self.base_dir / "venv"
        if not venv_path.exists():
            self.log_info("Creating virtual environment...")
            if not self.run_command([self.python_cmd, "-m", "venv", "venv"]):
                return False
                
        # Activate virtual environment and install dependencies
        if self.is_windows:
            pip_cmd = str(venv_path / "Scripts" / "pip.exe")
            python_cmd = str(venv_path / "Scripts" / "python.exe")
        else:
            pip_cmd = str(venv_path / "bin" / "pip")
            python_cmd = str(venv_path / "bin" / "python")
            
        # Upgrade pip
        self.log_info("Upgrading pip...")
        if not self.run_command([python_cmd, "-m", "pip", "install", "--upgrade", "pip"]):
            return False
            
        # Install requirements
        requirements_file = self.base_dir / "requirements.txt"
        if requirements_file.exists():
            self.log_info("Installing Python dependencies...")
            if not self.run_command([pip_cmd, "install", "-r", "requirements.txt"]):
                return False
        else:
            # Install essential packages
            essential_packages = [
                "streamlit>=1.28.0",
                "requests>=2.31.0", 
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "plotly>=5.15.0",
                "opencv-python>=4.8.0",
                "pillow>=10.0.0"
            ]
            self.log_info("Installing essential packages...")
            for package in essential_packages:
                if not self.run_command([pip_cmd, "install", package]):
                    self.log_error(f"Failed to install {package}")
                    
        self.python_cmd = python_cmd
        self.log_info("✅ Python environment setup complete")
        return True
        
    def setup_directories(self):
        """Create necessary directories"""
        self.log_info("Setting up directory structure...")
        
        directories = [
            self.data_dir,
            self.data_dir / "live",
            self.data_dir / "cache", 
            self.data_dir / "backup",
            self.logs_dir,
            self.config_dir
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True, parents=True)
            
        self.log_info("✅ Directory structure created")
        return True
        
    def initialize_data_pipeline(self):
        """Initialize and start automated data pipeline"""
        self.log_info("Initializing automated data pipeline...")
        
        # Create data pipeline script
        pipeline_script = self.base_dir / "automated_data_pipeline.py"
        pipeline_code = '''
import time
import threading
import requests
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

class AutomatedDataPipeline:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data" / "live"
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.running = False
        self.logger = logging.getLogger("DataPipeline")
        
    def fetch_space_weather_data(self):
        """Fetch real-time space weather data"""
        try:
            # NOAA Space Weather data
            url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                output_file = self.data_dir / f"space_weather_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                self.logger.info(f"Space weather data saved to {output_file}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to fetch space weather data: {e}")
        return False
        
    def fetch_iss_location(self):
        """Fetch real-time ISS location"""
        try:
            url = "http://api.open-notify.org/iss-now.json"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                output_file = self.data_dir / f"iss_location_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                self.logger.info(f"ISS location data saved to {output_file}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to fetch ISS location: {e}")
        return False
        
    def fetch_satellite_data(self):
        """Fetch satellite tracking data"""
        try:
            # Sample satellite data - replace with real API
            url = "https://api.wheretheiss.at/v1/satellites/25544"  # ISS
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                output_file = self.data_dir / f"satellite_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                self.logger.info(f"Satellite data saved to {output_file}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to fetch satellite data: {e}")
        return False
        
    def data_collection_loop(self):
        """Main data collection loop"""
        self.logger.info("Starting automated data collection...")
        while self.running:
            try:
                # Fetch all data sources
                self.fetch_space_weather_data()
                time.sleep(5)  # Small delay between requests
                
                self.fetch_iss_location()
                time.sleep(5)
                
                self.fetch_satellite_data()
                
                # Wait 5 minutes before next collection
                for _ in range(300):  # 5 minutes = 300 seconds
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in data collection loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
                
    def start(self):
        """Start the automated data pipeline"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.data_collection_loop, daemon=True)
            self.thread.start()
            self.logger.info("Automated data pipeline started")
            
    def stop(self):
        """Stop the automated data pipeline"""
        if self.running:
            self.running = False
            self.logger.info("Automated data pipeline stopped")

if __name__ == "__main__":
    pipeline = AutomatedDataPipeline()
    pipeline.start()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()
'''
        
        with open(pipeline_script, 'w') as f:
            f.write(pipeline_code)
            
        self.log_info("✅ Data pipeline initialized")
        return True
        
    def start_data_pipeline(self):
        """Start the automated data pipeline as a background service"""
        self.log_info("Starting automated data pipeline...")
        
        try:
            pipeline_script = self.base_dir / "automated_data_pipeline.py"
            process = subprocess.Popen([
                self.python_cmd, 
                str(pipeline_script)
            ], cwd=self.base_dir)
            
            self.services.append(("DataPipeline", process))
            self.log_info("✅ Automated data pipeline started")
            return True
        except Exception as e:
            self.log_error(f"Failed to start data pipeline: {e}")
            return False
            
    def start_health_monitor(self):
        """Start health monitoring service"""
        self.log_info("Starting health monitoring...")
        
        monitor_script = self.base_dir / "health_monitor.py"
        monitor_code = '''
import time
import psutil
import logging
from datetime import datetime
from pathlib import Path

class HealthMonitor:
    def __init__(self):
        self.logger = logging.getLogger("HealthMonitor")
        self.base_dir = Path(__file__).parent
        
    def check_system_health(self):
        """Check system resources and health"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage  
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage(str(self.base_dir))
            disk_percent = (disk.used / disk.total) * 100
            
            # Log health status
            self.logger.info(f"System Health - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent:.1f}%")
            
            # Alert on high usage
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
            if memory_percent > 90:
                self.logger.warning(f"High memory usage: {memory_percent}%")
            if disk_percent > 90:
                self.logger.warning(f"High disk usage: {disk_percent:.1f}%")
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            
    def monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                self.check_system_health()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    monitor = HealthMonitor()
    monitor.monitor_loop()
'''
        
        with open(monitor_script, 'w') as f:
            f.write(monitor_code)
            
        try:
            process = subprocess.Popen([
                self.python_cmd, 
                str(monitor_script)
            ], cwd=self.base_dir)
            
            self.services.append(("HealthMonitor", process))
            self.log_info("✅ Health monitoring started")
            return True
        except Exception as e:
            self.log_error(f"Failed to start health monitor: {e}")
            return False
            
    def start_application(self):
        """Start the main Streamlit application"""
        self.log_info("Starting Space Intelligence Platform...")
        
        # Check if main.py exists
        main_app = self.base_dir / "main.py"
        if not main_app.exists():
            # Create a basic main.py if it doesn't exist
            app_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path

st.set_page_config(
    page_title="Space Intelligence Platform",
    page_icon="🚀",
    layout="wide"
)

def main():
    st.title("🚀 Space Intelligence Platform")
    st.subheader("Real-time Space Data Analytics with AI")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Status", "🟢 Online", "Automated")
    with col2:
        st.metric("Data Pipeline", "🔄 Active", "Real-time")  
    with col3:
        st.metric("Last Update", datetime.now().strftime("%H:%M:%S"), "Live")
        
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌤️ Space Weather", "🪨 Asteroid Mining", "🛰️ Satellite Tracking", 
        "🚀 Launch Optimizer", "🌍 Live Tracker", "🤖 AI Vision"
    ])
    
    with tab1:
        st.header("🌤️ Space Weather Monitor")
        st.info("Real-time space weather data from NOAA SWPC")
        
        # Load latest space weather data
        data_dir = Path("data/live")
        if data_dir.exists():
            weather_files = list(data_dir.glob("space_weather_*.json"))
            if weather_files:
                latest_file = max(weather_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_file) as f:
                        data = json.load(f)
                    st.success(f"Live data loaded from {latest_file.name}")
                    st.json(data[:5] if isinstance(data, list) else data)
                except:
                    st.warning("Loading sample space weather data")
            else:
                st.info("Waiting for real-time data...")
        else:
            st.info("Data pipeline initializing...")
            
    with tab2:
        st.header("🪨 Asteroid Mining Analytics")
        st.info("AI-powered asteroid mining opportunity analysis")
        
    with tab3:
        st.header("🛰️ Satellite Tracking & Collision Prediction")
        st.info("Real-time satellite monitoring and collision analysis")
        
    with tab4:
        st.header("🚀 Launch Window Optimization")
        st.info("Multi-factor launch window analysis")
        
    with tab5:
        st.header("🌍 Live Space Asset Tracker")
        st.info("Real-time ISS and satellite positions")
        
        # Load ISS data
        data_dir = Path("data/live")
        if data_dir.exists():
            iss_files = list(data_dir.glob("iss_location_*.json"))
            if iss_files:
                latest_file = max(iss_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_file) as f:
                        iss_data = json.load(f)
                    st.success("🛰️ Live ISS Position")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Latitude", f"{iss_data['iss_position']['latitude']}°")
                    with col2:
                        st.metric("Longitude", f"{iss_data['iss_position']['longitude']}°")
                except:
                    st.info("Loading ISS position...")
            else:
                st.info("Waiting for ISS data...")
        
    with tab6:
        st.header("🤖 AI Vision & Anomaly Detection")
        st.info("YOLO-powered solar image analysis and anomaly detection")
        
    # Auto-refresh every 30 seconds
    time.sleep(30)
    st.rerun()

if __name__ == "__main__":
    main()
'''
            with open(main_app, 'w') as f:
                f.write(app_code)
                
        # Start Streamlit application
        try:
            streamlit_cmd = [
                self.python_cmd, "-m", "streamlit", "run", "main.py",
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--server.headless=true",
                "--server.runOnSave=true"
            ]
            
            process = subprocess.Popen(streamlit_cmd, cwd=self.base_dir)
            self.services.append(("StreamlitApp", process))
            
            self.log_info("✅ Space Intelligence Platform started")
            self.log_info("🌐 Access the application at: http://localhost:8501")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to start application: {e}")
            return False
            
    def cleanup_services(self):
        """Cleanup background services"""
        self.log_info("Shutting down services...")
        for name, process in self.services:
            try:
                process.terminate()
                process.wait(timeout=5)
                self.log_info(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                self.log_info(f"🔥 {name} force stopped")
            except Exception as e:
                self.log_error(f"Error stopping {name}: {e}")
                
    def run_full_automation(self):
        """Run complete automation sequence"""
        self.log_info("🚀 Starting Space Intelligence Platform - Full Automation")
        self.log_info("=" * 60)
        
        # Setup cleanup handlers
        atexit.register(self.cleanup_services)
        signal.signal(signal.SIGINT, lambda s, f: self.cleanup_services())
        signal.signal(signal.SIGTERM, lambda s, f: self.cleanup_services())
        
        try:
            # Step 1: System requirements
            if not self.check_system_requirements():
                return False
                
            # Step 2: Environment setup  
            if not self.setup_environment():
                return False
                
            # Step 3: Directory structure
            if not self.setup_directories():
                return False
                
            # Step 4: Initialize data pipeline
            if not self.initialize_data_pipeline():
                return False
                
            # Step 5: Start background services
            if not self.start_data_pipeline():
                return False
                
            if not self.start_health_monitor():
                return False
                
            # Step 6: Start main application
            if not self.start_application():
                return False
                
            self.log_info("=" * 60)  
            self.log_info("🎉 SPACE INTELLIGENCE PLATFORM FULLY OPERATIONAL!")
            self.log_info("🌐 Application URL: http://localhost:8501")
            self.log_info("📊 Real-time data pipeline: ACTIVE")
            self.log_info("🔍 Health monitoring: ACTIVE") 
            self.log_info("🤖 All automation systems: ONLINE")
            self.log_info("=" * 60)
            
            # Keep main process alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.log_info("Shutdown requested...")
                
            return True
            
        except Exception as e:
            self.log_error(f"Automation failed: {e}")
            return False
        finally:
            self.cleanup_services()

def main():
    """Main entry point"""
    automation = SpaceIntelligenceAutomation()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "start":
            automation.run_full_automation()
        elif command == "setup":
            automation.check_system_requirements()
            automation.setup_environment() 
            automation.setup_directories()
        elif command == "pipeline":
            automation.start_data_pipeline()
        else:
            print("Usage: python autostart.py [start|setup|pipeline]")
    else:
        automation.run_full_automation()

if __name__ == "__main__":
    main()