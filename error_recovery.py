"""
🔧 Space Intelligence Platform - Automated Error Recovery & Monitoring System
Comprehensive error handling, recovery, and performance monitoring
"""

import os
import sys
import time
import json
import psutil
import logging
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import requests
from typing import Dict, List, Optional

class AutomatedRecoverySystem:
    def __init__(self, config_file: str = "config/automation_config.json"):
        self.base_dir = Path(__file__).parent
        self.config_file = self.base_dir / config_file
        self.config = self.load_config()
        self.logger = self.setup_logging()
        self.monitoring_active = False
        self.recovery_stats = {
            "restarts": 0,
            "data_recoveries": 0,
            "last_restart": None
        }
        
    def load_config(self) -> dict:
        """Load automation configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                # Default configuration
                return {
                    "monitoring": {"enabled": True, "health_check_interval": 60},
                    "automation": {"auto_recovery": True},
                    "data_pipeline": {"retry_attempts": 3, "timeout_seconds": 30}
                }
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
            
    def setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        logs_dir = self.base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        log_file = logs_dir / f"recovery_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.get("monitoring", {}).get("log_level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger("AutoRecovery")
        
    def check_system_health(self) -> Dict[str, float]:
        """Comprehensive system health check"""
        try:
            health_status = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage(str(self.base_dir)).percent,
                "boot_time": psutil.boot_time(),
                "process_count": len(psutil.pids())
            }
            
            # Check network connectivity
            try:
                response = requests.get("https://httpbin.org/status/200", timeout=5)
                health_status["network_ok"] = response.status_code == 200
            except:
                health_status["network_ok"] = False
                
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"error": str(e)}
            
    def check_application_health(self) -> bool:
        """Check if Streamlit application is running and responsive"""
        try:
            port = self.config.get("server", {}).get("port", 8501)
            response = requests.get(f"http://localhost:{port}/_stcore/health", timeout=10)
            return response.status_code == 200
        except:
            return False
            
    def check_data_pipeline_health(self) -> Dict[str, bool]:
        """Check data pipeline status"""
        data_dir = self.base_dir / "data" / "live"
        pipeline_status = {}
        
        if not data_dir.exists():
            return {"data_directory": False}
            
        # Check for recent data files
        current_time = datetime.now()
        for source_name in ["space_weather", "iss_location", "satellite_data"]:
            files = list(data_dir.glob(f"{source_name}_*.json"))
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                file_age = current_time.timestamp() - latest_file.stat().st_mtime
                # Consider data fresh if less than 1 hour old
                pipeline_status[source_name] = file_age < 3600
            else:
                pipeline_status[source_name] = False
                
        return pipeline_status
        
    def restart_application(self) -> bool:
        """Restart the Streamlit application"""
        try:
            self.logger.info("Attempting to restart application...")
            
            # Kill existing Streamlit processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'streamlit' in proc.info['name'].lower() or \
                       any('streamlit' in str(cmd).lower() for cmd in proc.info['cmdline'] or []):
                        proc.terminate()
                        proc.wait(timeout=10)
                        self.logger.info(f"Terminated Streamlit process {proc.info['pid']}")
                except:
                    pass
                    
            # Start new application
            time.sleep(5)  # Wait before restart
            
            python_cmd = sys.executable
            startup_script = self.base_dir / "autostart.py" 
            
            if startup_script.exists():
                subprocess.Popen([python_cmd, str(startup_script), "start"], 
                               cwd=str(self.base_dir))
                self.recovery_stats["restarts"] += 1
                self.recovery_stats["last_restart"] = datetime.now().isoformat()
                self.logger.info("Application restart initiated")
                return True
            else:
                self.logger.error("Startup script not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restart application: {e}")
            return False
            
    def recover_data_pipeline(self) -> bool:
        """Recover data pipeline by restarting data collection"""
        try:
            self.logger.info("Recovering data pipeline...")
            
            # Kill existing pipeline processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'automated_data_pipeline.py' in cmdline:
                        proc.terminate()
                        proc.wait(timeout=10)
                        self.logger.info(f"Terminated data pipeline process {proc.info['pid']}")
                except:
                    pass
                    
            # Start new pipeline
            time.sleep(3)
            pipeline_script = self.base_dir / "automated_data_pipeline.py"
            
            if pipeline_script.exists():
                subprocess.Popen([sys.executable, str(pipeline_script)], 
                               cwd=str(self.base_dir))
                self.recovery_stats["data_recoveries"] += 1
                self.logger.info("Data pipeline recovery initiated")
                return True
            else:
                self.logger.error("Data pipeline script not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to recover data pipeline: {e}")
            return False
            
    def cleanup_old_data(self):
        """Clean up old data files to free space"""
        try:
            data_dir = self.base_dir / "data" / "live"
            if not data_dir.exists():
                return
                
            retention_days = self.config.get("automation", {}).get("data_retention_days", 30)
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            cleaned_count = 0
            for file_path in data_dir.glob("*.json"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
                    
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old data files")
                
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
            
    def create_system_backup(self):
        """Create backup of critical system files"""
        try:
            backup_dir = self.base_dir / "data" / "backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup configuration files
            config_files = ["config", "main.py", "autostart.py", "requirements.txt"]
            for item in config_files:
                source = self.base_dir / item
                if source.exists():
                    if source.is_file():
                        import shutil
                        shutil.copy2(source, backup_dir / source.name)
                    elif source.is_dir():
                        import shutil
                        shutil.copytree(source, backup_dir / source.name, dirs_exist_ok=True)
                        
            self.logger.info(f"System backup created: {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            
    def monitoring_loop(self):
        """Main monitoring and recovery loop"""
        self.logger.info("Starting automated monitoring and recovery system...")
        self.monitoring_active = True
        
        check_interval = self.config.get("monitoring", {}).get("health_check_interval", 60)
        
        while self.monitoring_active:
            try:
                # System health check
                health = self.check_system_health()
                
                # Check for critical resource usage
                alerts = self.config.get("monitoring", {}).get("alerts", {})
                if health.get("cpu_percent", 0) > alerts.get("cpu_threshold", 90):
                    self.logger.warning(f"High CPU usage: {health['cpu_percent']}%")
                    
                if health.get("memory_percent", 0) > alerts.get("memory_threshold", 90):
                    self.logger.warning(f"High memory usage: {health['memory_percent']}%")
                    
                if health.get("disk_percent", 0) > alerts.get("disk_threshold", 90):
                    self.logger.warning(f"High disk usage: {health['disk_percent']}%")
                    self.cleanup_old_data()
                    
                # Application health check
                if not self.check_application_health():
                    self.logger.warning("Application health check failed - attempting recovery")
                    if self.config.get("automation", {}).get("auto_recovery", True):
                        self.restart_application()
                        
                # Data pipeline health check
                pipeline_status = self.check_data_pipeline_health()
                failed_sources = [source for source, status in pipeline_status.items() if not status]
                
                if failed_sources:
                    self.logger.warning(f"Data pipeline issues detected: {failed_sources}")
                    if self.config.get("automation", {}).get("auto_recovery", True):
                        self.recover_data_pipeline()
                        
                # Periodic maintenance
                current_hour = datetime.now().hour
                if current_hour == 3 and datetime.now().minute < 5:  # 3 AM maintenance window
                    if self.config.get("automation", {}).get("backup_enabled", True):
                        self.create_system_backup()
                    self.cleanup_old_data()
                    
                # Log status
                if all(pipeline_status.values()) and health.get("cpu_percent", 0) < 80:
                    self.logger.info("✅ All systems operational")
                else:
                    self.logger.info(f"🔍 Status - App: {'✅' if self.check_application_health() else '❌'}, "
                                   f"CPU: {health.get('cpu_percent', 0):.1f}%, "
                                   f"Memory: {health.get('memory_percent', 0):.1f}%")
                    
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                
            # Wait before next check
            time.sleep(check_interval)
            
    def start_monitoring(self):
        """Start the monitoring system in background"""
        if not self.monitoring_active:
            self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Automated monitoring system started")
            
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        self.logger.info("Automated monitoring system stopped")
        
    def get_status_report(self) -> dict:
        """Generate comprehensive status report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": self.check_system_health(),
            "application_health": self.check_application_health(),
            "data_pipeline_health": self.check_data_pipeline_health(),
            "recovery_stats": self.recovery_stats,
            "monitoring_active": self.monitoring_active
        }

def main():
    """Main entry point for recovery system"""
    recovery_system = AutomatedRecoverySystem()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "monitor":
            recovery_system.start_monitoring()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                recovery_system.stop_monitoring()
        elif command == "status":
            status = recovery_system.get_status_report()
            print(json.dumps(status, indent=2))
        elif command == "restart":
            recovery_system.restart_application()
        elif command == "recover":
            recovery_system.recover_data_pipeline()
        else:
            print("Usage: python error_recovery.py [monitor|status|restart|recover]")
    else:
        recovery_system.start_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            recovery_system.stop_monitoring()

if __name__ == "__main__":
    main()