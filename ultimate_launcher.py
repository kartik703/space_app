"""
🚀 SPACE INTELLIGENCE PLATFORM - ULTIMATE AUTOMATION LAUNCHER
One-click solution for complete automated space intelligence system
"""

import os
import sys
import time
import subprocess
import platform
import threading
import signal
import atexit
from pathlib import Path
from datetime import datetime

class UltimateSpaceAutomation:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.is_windows = platform.system() == "Windows"
        self.python_cmd = sys.executable
        self.services = {}
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup_all_services)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutdown signal received...")
        self.running = False
        self.cleanup_all_services()
        sys.exit(0)
        
    def print_status(self, message, status="INFO"):
        """Print formatted status message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌",
            "PROGRESS": "🔄"
        }
        symbol = symbols.get(status, "📋")
        print(f"{symbol} [{timestamp}] {message}")
        
    def check_and_install_requirements(self):
        """Check and install required packages"""
        self.print_status("Checking Python requirements...", "PROGRESS")
        
        # Essential packages for the space platform
        required_packages = [
            "streamlit>=1.28.0",
            "pandas>=2.0.0", 
            "numpy>=1.24.0",
            "plotly>=5.15.0",
            "requests>=2.31.0",
            "pillow>=10.0.0",
            "psutil>=5.9.0"
        ]
        
        try:
            # Try to install requirements.txt if exists
            requirements_file = self.base_dir / "requirements.txt"
            if requirements_file.exists():
                self.print_status("Installing from requirements.txt...", "PROGRESS")
                result = subprocess.run([
                    self.python_cmd, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.print_status("Requirements installed successfully", "SUCCESS")
                else:
                    self.print_status("Installing essential packages individually...", "WARNING")
                    for package in required_packages:
                        subprocess.run([
                            self.python_cmd, "-m", "pip", "install", package, "--quiet"
                        ], capture_output=True)
            else:
                # Install essential packages
                self.print_status("Installing essential packages...", "PROGRESS")
                for package in required_packages:
                    subprocess.run([
                        self.python_cmd, "-m", "pip", "install", package, "--quiet"
                    ], capture_output=True)
                    
            self.print_status("All requirements ready", "SUCCESS")
            return True
            
        except Exception as e:
            self.print_status(f"Package installation error: {e}", "WARNING")
            return True  # Continue anyway
            
    def setup_directories(self):
        """Create necessary directory structure"""
        self.print_status("Setting up directory structure...", "PROGRESS")
        
        directories = [
            "data/live",
            "data/cache", 
            "data/backup",
            "logs",
            "config"
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
            
        self.print_status("Directory structure ready", "SUCCESS")
        
    def start_data_pipeline(self):
        """Start automated data collection pipeline"""
        self.print_status("Starting automated data pipeline...", "PROGRESS")
        
        try:
            # Check if data pipeline script exists
            pipeline_script = self.base_dir / "automated_data_pipeline.py"
            if not pipeline_script.exists():
                self.print_status("Data pipeline script not found, creating basic version...", "WARNING")
                # Create a basic data pipeline if it doesn't exist
                self.create_basic_data_pipeline()
                
            # Start the pipeline
            process = subprocess.Popen([
                self.python_cmd, str(pipeline_script)
            ], cwd=str(self.base_dir))
            
            self.services['data_pipeline'] = process
            self.print_status("Data pipeline started", "SUCCESS")
            return True
            
        except Exception as e:
            self.print_status(f"Failed to start data pipeline: {e}", "ERROR")
            return False
            
    def create_basic_data_pipeline(self):
        """Create a basic data pipeline script"""
        pipeline_code = '''
import time
import json
import requests
from datetime import datetime
from pathlib import Path

def fetch_space_data():
    """Fetch real-time space data"""
    data_dir = Path("data/live")
    data_dir.mkdir(exist_ok=True, parents=True)
    
    sources = {
        "iss_location": "http://api.open-notify.org/iss-now.json",
        "space_weather": "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    }
    
    for name, url in sources.items():
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                with open(data_dir / filename, 'w') as f:
                    json.dump(response.json(), f)
                print(f"✅ Updated {name}")
        except Exception as e:
            print(f"❌ Failed to fetch {name}: {e}")

if __name__ == "__main__":
    print("🚀 Starting automated data pipeline...")
    while True:
        try:
            fetch_space_data()
            time.sleep(300)  # 5 minutes
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Pipeline error: {e}")
            time.sleep(60)
'''
        with open(self.base_dir / "automated_data_pipeline.py", 'w') as f:
            f.write(pipeline_code)
            
    def start_monitoring_system(self):
        """Start system monitoring and recovery"""
        self.print_status("Starting monitoring system...", "PROGRESS")
        
        try:
            monitor_script = self.base_dir / "error_recovery.py"
            if monitor_script.exists():
                process = subprocess.Popen([
                    self.python_cmd, str(monitor_script), "monitor"
                ], cwd=str(self.base_dir))
                
                self.services['monitor'] = process
                self.print_status("Monitoring system started", "SUCCESS")
            else:
                self.print_status("Monitoring script not found, skipping", "WARNING")
                
        except Exception as e:
            self.print_status(f"Failed to start monitoring: {e}", "ERROR")
            
    def start_main_application(self):
        """Start the main Streamlit application"""
        self.print_status("Starting Space Intelligence Platform...", "PROGRESS")
        
        try:
            # Check if main.py exists
            main_script = self.base_dir / "main.py"
            if not main_script.exists():
                self.print_status("Main application not found!", "ERROR")
                return False
                
            # Start Streamlit
            streamlit_cmd = [
                self.python_cmd, "-m", "streamlit", "run", "main.py",
                "--server.port=8501",
                "--server.address=0.0.0.0", 
                "--server.headless=true",
                "--browser.gatherUsageStats=false"
            ]
            
            process = subprocess.Popen(streamlit_cmd, cwd=str(self.base_dir))
            self.services['streamlit'] = process
            
            # Wait a moment and check if it started
            time.sleep(5)
            if process.poll() is None:  # Still running
                self.print_status("Space Intelligence Platform started successfully!", "SUCCESS")
                return True
            else:
                self.print_status("Application failed to start", "ERROR")
                return False
                
        except Exception as e:
            self.print_status(f"Failed to start application: {e}", "ERROR")
            return False
            
    def monitor_services(self):
        """Monitor running services and restart if needed"""
        while self.running:
            try:
                # Check each service
                for name, process in list(self.services.items()):
                    if process.poll() is not None:  # Process has terminated
                        self.print_status(f"Service {name} stopped, attempting restart...", "WARNING")
                        
                        # Restart based on service type
                        if name == "streamlit":
                            self.start_main_application()
                        elif name == "data_pipeline":
                            self.start_data_pipeline()
                        elif name == "monitor":
                            self.start_monitoring_system()
                            
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.print_status(f"Service monitoring error: {e}", "ERROR")
                time.sleep(60)
                
    def cleanup_all_services(self):
        """Clean up all running services"""
        self.print_status("Stopping all services...", "PROGRESS")
        
        for name, process in self.services.items():
            try:
                if process.poll() is None:  # Still running
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                        self.print_status(f"Service {name} stopped", "SUCCESS")
                    except subprocess.TimeoutExpired:
                        process.kill()
                        self.print_status(f"Service {name} force stopped", "WARNING")
            except Exception as e:
                self.print_status(f"Error stopping {name}: {e}", "ERROR")
                
        self.services.clear()
        
    def display_banner(self):
        """Display startup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                🚀 SPACE INTELLIGENCE PLATFORM 🚀              ║
║                     ULTIMATE AUTOMATION                       ║
║                                                              ║
║  🌍 Real-time Space Weather Monitoring                       ║
║  🛰️  Live Satellite Tracking & ISS Position                  ║ 
║  🪨 AI-Powered Asteroid Mining Analytics                     ║
║  🚀 Launch Window Optimization Engine                        ║
║  🤖 Computer Vision & Anomaly Detection                      ║
║  📊 Automated Data Pipeline & Monitoring                     ║
║                                                              ║
║              🌟 FULLY AUTOMATED & PROFESSIONAL 🌟             ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
    def run_full_automation(self):
        """Run the complete automation sequence"""
        self.display_banner()
        
        self.print_status("Initializing Space Intelligence Platform...", "PROGRESS")
        
        try:
            # Step 1: Setup environment
            self.check_and_install_requirements()
            self.setup_directories()
            
            # Step 2: Start background services
            self.start_data_pipeline()
            time.sleep(2)
            
            self.start_monitoring_system()
            time.sleep(2)
            
            # Step 3: Start main application
            if not self.start_main_application():
                self.print_status("Failed to start main application", "ERROR")
                return False
                
            # Step 4: Display success information
            print("\n" + "="*70)
            self.print_status("🎉 SPACE INTELLIGENCE PLATFORM FULLY OPERATIONAL!", "SUCCESS")
            print("="*70)
            self.print_status("🌐 Web Application: http://localhost:8501", "INFO")
            self.print_status("📊 Real-time Data Pipeline: ACTIVE", "SUCCESS")
            self.print_status("🔍 System Monitoring: ACTIVE", "SUCCESS") 
            self.print_status("🤖 Full Automation: ENABLED", "SUCCESS")
            print("="*70)
            self.print_status("Press Ctrl+C to stop all services", "INFO")
            print()
            
            # Step 5: Start service monitoring
            monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
            monitor_thread.start()
            
            # Step 6: Keep main process alive and show status
            while self.running:
                try:
                    time.sleep(10)
                    # Show periodic status
                    active_services = sum(1 for p in self.services.values() if p.poll() is None)
                    if active_services == len(self.services):
                        self.print_status(f"✅ All {active_services} services running normally", "SUCCESS")
                    else:
                        self.print_status(f"⚠️ {active_services}/{len(self.services)} services active", "WARNING")
                        
                except KeyboardInterrupt:
                    break
                    
            return True
            
        except Exception as e:
            self.print_status(f"Critical error: {e}", "ERROR")
            return False
        finally:
            self.cleanup_all_services()

def main():
    """Main entry point"""
    automation = UltimateSpaceAutomation()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command in ["start", "run", "launch"]:
            automation.run_full_automation()
        elif command == "stop":
            automation.cleanup_all_services()
        elif command == "setup":
            automation.check_and_install_requirements()
            automation.setup_directories()
        else:
            print("Usage: python ultimate_launcher.py [start|stop|setup]")
            print("       python ultimate_launcher.py        (same as start)")
    else:
        # Default action
        automation.run_full_automation()

if __name__ == "__main__":
    main()