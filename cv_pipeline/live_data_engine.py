#!/usr/bin/env python3
"""
Live Data Engine for Real-Time Space Weather Monitoring
Continuously fetches and processes space weather data with alert generation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from threading import Thread
import time
import queue
import sqlite3
from cv_pipeline.data_source_manager import DataSourceManager
from cv_pipeline.time_series_forecaster import SpaceWeatherForecaster
from cv_pipeline.orbital_intelligence import OrbitalIntelligence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpaceWeatherAlert:
    """Space weather alert structure"""
    timestamp: datetime
    alert_id: str
    severity: str  # LOW, MEDIUM, HIGH, EXTREME
    category: str  # GEOMAGNETIC, SOLAR_WIND, PROTON, ORBITAL
    parameter: str
    current_value: float
    threshold_value: float
    description: str
    recommendation: str
    expires_at: Optional[datetime] = None
    acknowledged: bool = False

@dataclass
class LiveDataPoint:
    """Live data point structure"""
    timestamp: datetime
    parameter: str
    value: float
    unit: str
    quality: str  # GOOD, FAIR, POOR
    source: str

class AlertManager:
    """Manages alert generation, escalation, and notifications"""
    
    def __init__(self):
        self.alert_thresholds = {
            'kp_index': {
                'LOW': 3.0,
                'MEDIUM': 5.0, 
                'HIGH': 7.0,
                'EXTREME': 8.0
            },
            'solar_wind_speed': {
                'LOW': 500.0,
                'MEDIUM': 600.0,
                'HIGH': 700.0,
                'EXTREME': 800.0
            },
            'proton_flux': {
                'LOW': 1.0,
                'MEDIUM': 10.0,
                'HIGH': 100.0,
                'EXTREME': 1000.0
            },
            'dst_index': {
                'LOW': -30.0,
                'MEDIUM': -50.0,
                'HIGH': -100.0,
                'EXTREME': -200.0
            }
        }
        
        self.active_alerts = {}
        self.alert_history = []
        self.notification_callbacks = []
        
    def add_notification_callback(self, callback: Callable):
        """Add notification callback function"""
        self.notification_callbacks.append(callback)
        
    def check_thresholds(self, data_point: LiveDataPoint) -> Optional[SpaceWeatherAlert]:
        """Check if data point triggers an alert"""
        parameter = data_point.parameter
        value = data_point.value
        
        if parameter not in self.alert_thresholds:
            return None
            
        thresholds = self.alert_thresholds[parameter]
        
        # Determine severity level
        severity = None
        threshold_value = None
        
        if parameter == 'dst_index':
            # Dst is negative, so lower values are worse
            if value <= thresholds['EXTREME']:
                severity, threshold_value = 'EXTREME', thresholds['EXTREME']
            elif value <= thresholds['HIGH']:
                severity, threshold_value = 'HIGH', thresholds['HIGH']
            elif value <= thresholds['MEDIUM']:
                severity, threshold_value = 'MEDIUM', thresholds['MEDIUM']
            elif value <= thresholds['LOW']:
                severity, threshold_value = 'LOW', thresholds['LOW']
        else:
            # Higher values are worse
            if value >= thresholds['EXTREME']:
                severity, threshold_value = 'EXTREME', thresholds['EXTREME']
            elif value >= thresholds['HIGH']:
                severity, threshold_value = 'HIGH', thresholds['HIGH']
            elif value >= thresholds['MEDIUM']:
                severity, threshold_value = 'MEDIUM', thresholds['MEDIUM']
            elif value >= thresholds['LOW']:
                severity, threshold_value = 'LOW', thresholds['LOW']
        
        if severity is None:
            return None
            
        # Create alert
        alert = SpaceWeatherAlert(
            timestamp=data_point.timestamp,
            alert_id=f"{parameter}_{severity}_{int(data_point.timestamp.timestamp())}",
            severity=severity,
            category=self._get_category(parameter),
            parameter=parameter,
            current_value=value,
            threshold_value=threshold_value,
            description=self._get_description(parameter, severity, value),
            recommendation=self._get_recommendation(parameter, severity),
            expires_at=data_point.timestamp + timedelta(hours=3)
        )
        
        return alert
        
    def _get_category(self, parameter: str) -> str:
        """Get alert category for parameter"""
        categories = {
            'kp_index': 'GEOMAGNETIC',
            'dst_index': 'GEOMAGNETIC', 
            'solar_wind_speed': 'SOLAR_WIND',
            'proton_flux': 'PROTON'
        }
        return categories.get(parameter, 'GENERAL')
        
    def _get_description(self, parameter: str, severity: str, value: float) -> str:
        """Generate alert description"""
        descriptions = {
            'kp_index': f"Geomagnetic activity {severity.lower()} - Kp index at {value:.1f}",
            'dst_index': f"Geomagnetic storm {severity.lower()} - Dst index at {value:.0f} nT",
            'solar_wind_speed': f"Solar wind speed {severity.lower()} - {value:.0f} km/s",
            'proton_flux': f"Proton flux {severity.lower()} - {value:.2f} particles/cm²/s"
        }
        return descriptions.get(parameter, f"{parameter} alert - {severity}")
        
    def _get_recommendation(self, parameter: str, severity: str) -> str:
        """Generate alert recommendation"""
        if severity == 'EXTREME':
            return "CRITICAL: Implement emergency protocols, protect sensitive systems"
        elif severity == 'HIGH':
            return "HIGH RISK: Monitor closely, prepare contingency measures"
        elif severity == 'MEDIUM':
            return "MODERATE RISK: Increased monitoring recommended"
        else:
            return "LOW RISK: Continue normal operations with awareness"
            
    def process_alert(self, alert: SpaceWeatherAlert):
        """Process and potentially issue alert"""
        # Check if similar alert is already active
        existing_alert = self.active_alerts.get(f"{alert.parameter}_{alert.severity}")
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = alert.current_value
            existing_alert.timestamp = alert.timestamp
        else:
            # New alert
            self.active_alerts[f"{alert.parameter}_{alert.severity}"] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            for callback in self.notification_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Notification callback failed: {e}")
                    
            logger.warning(f"🚨 {alert.severity} ALERT: {alert.description}")
            
    def get_active_alerts(self) -> List[SpaceWeatherAlert]:
        """Get currently active alerts"""
        now = datetime.now()
        active = []
        
        for key, alert in list(self.active_alerts.items()):
            if alert.expires_at and now > alert.expires_at:
                del self.active_alerts[key]
            else:
                active.append(alert)
                
        return active
        
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        for alert in self.active_alerts.values():
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                break

class LiveDataEngine:
    """Main engine for live data processing and monitoring"""
    
    def __init__(self, update_interval: int = 60):
        self.update_interval = update_interval  # seconds
        self.data_manager = DataSourceManager()
        self.forecaster = SpaceWeatherForecaster()
        self.orbital = OrbitalIntelligence()
        self.alert_manager = AlertManager()
        
        # Data storage
        self.live_data = queue.Queue(maxsize=1000)
        self.recent_data = {}  # Parameter -> list of recent values
        self.db_path = "data/live_data.db"
        
        # Control flags
        self.running = False
        self.thread = None
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for live data storage"""
        Path("data").mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_data (
                    timestamp TEXT,
                    parameter TEXT,
                    value REAL,
                    unit TEXT,
                    quality TEXT,
                    source TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    timestamp TEXT,
                    alert_id TEXT,
                    severity TEXT,
                    category TEXT,
                    parameter TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    description TEXT,
                    recommendation TEXT,
                    acknowledged INTEGER
                )
            """)
            
    def start_monitoring(self):
        """Start live data monitoring"""
        if self.running:
            logger.warning("Live monitoring already running")
            return
            
        self.running = True
        self.thread = Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info(f"🚀 Live monitoring started (update interval: {self.update_interval}s)")
        
    def stop_monitoring(self):
        """Stop live data monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("⏹️ Live monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._fetch_live_data()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
                
    def _fetch_live_data(self):
        """Fetch current space weather data"""
        timestamp = datetime.now()
        
        # Fetch space weather parameters
        parameters = ['kp_index', 'solar_wind_speed', 'proton_flux']
        
        for param in parameters:
            try:
                df = self.data_manager.get_space_weather_data(param)
                if not df.empty:
                    # Get latest value
                    latest = df.iloc[-1]
                    
                    data_point = LiveDataPoint(
                        timestamp=timestamp,
                        parameter=param,
                        value=float(latest['value']),
                        unit=self._get_unit(param),
                        quality='GOOD',
                        source='NOAA/SWPC'
                    )
                    
                    # Store data
                    self._store_data_point(data_point)
                    
                    # Check for alerts
                    alert = self.alert_manager.check_thresholds(data_point)
                    if alert:
                        self.alert_manager.process_alert(alert)
                        
            except Exception as e:
                logger.error(f"Error fetching {param}: {e}")
                
        # Fetch orbital data periodically (every 10 minutes)
        if int(timestamp.timestamp()) % 600 == 0:
            self._fetch_orbital_data()
            
    def _fetch_orbital_data(self):
        """Fetch orbital data and check for collision risks"""
        try:
            satellites = self.orbital.fetch_tle_data()
            if satellites:
                # Assess collision risks
                risks = self.orbital.assess_collision_risk(satellites[:50])  # Check subset
                
                # Create alerts for high-risk collisions
                for risk in risks:
                    if risk.risk_level in ['HIGH', 'EXTREME']:
                        alert = SpaceWeatherAlert(
                            timestamp=datetime.now(),
                            alert_id=f"collision_{risk.sat1_id}_{risk.sat2_id}",
                            severity=risk.risk_level,
                            category='ORBITAL',
                            parameter='collision_risk',
                            current_value=risk.probability,
                            threshold_value=0.1,
                            description=f"Collision risk between {risk.sat1_id} and {risk.sat2_id}",
                            recommendation=f"Monitor closely - closest approach in {risk.time_to_event:.1f} hours"
                        )
                        self.alert_manager.process_alert(alert)
                        
        except Exception as e:
            logger.error(f"Error fetching orbital data: {e}")
            
    def _get_unit(self, parameter: str) -> str:
        """Get unit for parameter"""
        units = {
            'kp_index': 'index',
            'dst_index': 'nT',
            'solar_wind_speed': 'km/s',
            'proton_flux': 'particles/cm²/s'
        }
        return units.get(parameter, '')
        
    def _store_data_point(self, data_point: LiveDataPoint):
        """Store data point in database and memory"""
        # Add to queue
        if not self.live_data.full():
            self.live_data.put(data_point)
            
        # Update recent data
        param = data_point.parameter
        if param not in self.recent_data:
            self.recent_data[param] = []
            
        self.recent_data[param].append(data_point)
        
        # Keep only last 100 points per parameter
        if len(self.recent_data[param]) > 100:
            self.recent_data[param] = self.recent_data[param][-100:]
            
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO live_data (timestamp, parameter, value, unit, quality, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data_point.timestamp.isoformat(),
                data_point.parameter,
                data_point.value,
                data_point.unit,
                data_point.quality,
                data_point.source
            ))
            
    def get_latest_data(self) -> Dict:
        """Get latest data for all parameters"""
        latest = {}
        for param, data_list in self.recent_data.items():
            if data_list:
                latest[param] = data_list[-1]
        return latest
        
    def get_recent_history(self, parameter: str, hours: int = 24) -> List[LiveDataPoint]:
        """Get recent history for a parameter"""
        if parameter not in self.recent_data:
            return []
            
        cutoff = datetime.now() - timedelta(hours=hours)
        return [dp for dp in self.recent_data[parameter] if dp.timestamp >= cutoff]
        
    def export_status(self) -> Dict:
        """Export current system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.running,
            'update_interval': self.update_interval,
            'active_alerts': [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
            'latest_data': {k: asdict(v) for k, v in self.get_latest_data().items()},
            'total_data_points': sum(len(data) for data in self.recent_data.values())
        }

def main():
    """Demo the live data engine"""
    print("🌟 LIVE SPACE WEATHER MONITORING ENGINE")
    print("=" * 50)
    
    # Create and start engine
    engine = LiveDataEngine(update_interval=30)  # 30 second updates for demo
    
    # Add notification callback
    def print_alert(alert: SpaceWeatherAlert):
        print(f"🚨 {alert.severity} ALERT: {alert.description}")
        print(f"   Recommendation: {alert.recommendation}")
        
    engine.alert_manager.add_notification_callback(print_alert)
    
    # Start monitoring
    engine.start_monitoring()
    
    try:
        print("🔄 Monitoring live space weather data...")
        print("📊 Press Ctrl+C to stop")
        
        while True:
            time.sleep(10)
            
            # Show status every 10 seconds
            status = engine.export_status()
            print(f"\n📈 Latest Data ({datetime.now().strftime('%H:%M:%S')}):")
            
            for param, data_point in status['latest_data'].items():
                dp = data_point
                print(f"   {param}: {dp['value']:.2f} {dp['unit']}")
                
            active_alerts = status['active_alerts']
            if active_alerts:
                print(f"🚨 Active Alerts: {len(active_alerts)}")
                for alert in active_alerts:
                    print(f"   {alert['severity']}: {alert['description']}")
            else:
                print("✅ No active alerts")
                
    except KeyboardInterrupt:
        print("\n⏹️ Stopping monitoring...")
        engine.stop_monitoring()
        print("✅ Monitoring stopped")

if __name__ == "__main__":
    main()