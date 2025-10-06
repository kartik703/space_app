"""
Orbital Intelligence Engine for Space Risk Assessment
Handles TLE processing, collision risk, drag forecasting, re-entry predictions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import requests
import json
import logging
from pathlib import Path
import math
import warnings
from .data_source_manager import DataSourceManager
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SatelliteData:
    """Satellite data structure"""
    norad_id: int
    name: str
    tle_line1: str
    tle_line2: str
    epoch: datetime
    inclination: float
    eccentricity: float
    perigee: float
    apogee: float
    period: float
    altitude: float

@dataclass
class CollisionRisk:
    """Collision risk assessment"""
    primary_object: str
    secondary_object: str
    time_of_closest_approach: datetime
    miss_distance: float  # km
    collision_probability: float
    risk_level: str
    recommended_action: str

@dataclass
class ReentryPrediction:
    """Re-entry prediction"""
    object_id: str
    predicted_reentry_time: datetime
    uncertainty_window: timedelta
    reentry_location: Tuple[float, float]  # lat, lon
    risk_assessment: str
    confidence_level: float

class OrbitalIntelligence:
    """
    Advanced orbital mechanics and space situational awareness system
    """
    
    def __init__(self, data_dir: str = "data/orbital"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data source manager
        self.data_manager = DataSourceManager()
        
        # Physical constants
        self.GM = 398600.4418  # Earth's gravitational parameter (km³/s²)
        self.R_EARTH = 6371.0  # Earth radius (km)
        self.J2 = 1.08262668e-3  # Earth's oblateness coefficient
        
        # Atmospheric model parameters
        self.atmosphere_scale_heights = {
            200: 58.5, 300: 82.5, 400: 87.5, 500: 96.5, 600: 111.5,
            700: 136.0, 800: 173.5, 900: 226.5, 1000: 309.5
        }
        
        # Collision thresholds
        self.collision_thresholds = {
            'extreme': 1.0,   # km
            'high': 5.0,      # km  
            'medium': 25.0,   # km
            'low': 100.0      # km
        }
        
        self.satellites_cache = {}
        
    def fetch_tle_data(self, source: str = 'celestrak') -> List[SatelliteData]:
        """Fetch TLE data using data source manager"""
        try:
            # Use data source manager for reliable TLE fetching
            tle_categories = ['stations', 'visual', 'weather']
            all_satellites = []
            
            for category in tle_categories:
                try:
                    tle_text = self.data_manager.get_orbital_data(category)
                    satellites = self._parse_tle_data(tle_text)
                    all_satellites.extend(satellites)
                    logger.info(f"Fetched {len(satellites)} satellites from {category}")
                except Exception as e:
                    logger.warning(f"Failed to fetch {category} TLE data: {e}")
            
            # Cache the data
            self.satellites_cache = {sat.norad_id: sat for sat in all_satellites}
            logger.info(f"Total satellites cached: {len(all_satellites)}")
            
            return all_satellites
            
        except Exception as e:
            logger.error(f"Error fetching TLE data: {e}")
            return []
            self.satellites_cache = {sat.norad_id: sat for sat in all_satellites}
            
            # Save to file
            self._save_tle_data(all_satellites)
            
            logger.info(f"Total satellites fetched: {len(all_satellites)}")
            return all_satellites
            
        except Exception as e:
            logger.error(f"Error fetching TLE data: {e}")
            # Try to load from cache
            return self._load_cached_tle_data()
    
    def _parse_tle_data(self, tle_text: str) -> List[SatelliteData]:
        """Parse TLE format data"""
        satellites = []
        lines = tle_text.strip().split('\n')
        
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
                
            try:
                name = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()
                
                # Validate TLE format
                if not (line1.startswith('1 ') and line2.startswith('2 ')):
                    continue
                
                # Extract NORAD ID
                norad_id = int(line1[2:7])
                
                # Parse orbital elements
                inclination = float(line2[8:16])
                raan = float(line2[17:25])  # Right ascension of ascending node
                eccentricity = float('0.' + line2[26:33])
                arg_perigee = float(line2[34:42])
                mean_anomaly = float(line2[43:51])
                mean_motion = float(line2[52:63])  # revolutions per day
                
                # Calculate derived parameters
                period = 24.0 / mean_motion  # hours
                semi_major_axis = ((self.GM * (period * 3600)**2) / (4 * math.pi**2))**(1/3)
                perigee = semi_major_axis * (1 - eccentricity) - self.R_EARTH
                apogee = semi_major_axis * (1 + eccentricity) - self.R_EARTH
                altitude = (perigee + apogee) / 2
                
                # Parse epoch
                epoch_year = int(line1[18:20])
                if epoch_year > 56:  # Assume 1900s if > 56, else 2000s
                    epoch_year += 1900
                else:
                    epoch_year += 2000
                epoch_day = float(line1[20:32])
                epoch = datetime(epoch_year, 1, 1) + timedelta(days=epoch_day - 1)
                
                satellite = SatelliteData(
                    norad_id=norad_id,
                    name=name,
                    tle_line1=line1,
                    tle_line2=line2,
                    epoch=epoch,
                    inclination=inclination,
                    eccentricity=eccentricity,
                    perigee=perigee,
                    apogee=apogee,
                    period=period,
                    altitude=altitude
                )
                
                satellites.append(satellite)
                
            except Exception as e:
                logger.warning(f"Failed to parse TLE for {name if 'name' in locals() else 'unknown'}: {e}")
                continue
        
        return satellites
    
    def _save_tle_data(self, satellites: List[SatelliteData]):
        """Save TLE data to local file"""
        try:
            data = []
            for sat in satellites:
                data.append({
                    'norad_id': sat.norad_id,
                    'name': sat.name,
                    'tle_line1': sat.tle_line1,
                    'tle_line2': sat.tle_line2,
                    'epoch': sat.epoch.isoformat(),
                    'inclination': sat.inclination,
                    'eccentricity': sat.eccentricity,
                    'perigee': sat.perigee,
                    'apogee': sat.apogee,
                    'period': sat.period,
                    'altitude': sat.altitude
                })
            
            cache_file = self.data_dir / 'tle_cache.json'
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved TLE data to {cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving TLE data: {e}")
    
    def _load_cached_tle_data(self) -> List[SatelliteData]:
        """Load TLE data from cache"""
        try:
            cache_file = self.data_dir / 'tle_cache.json'
            if not cache_file.exists():
                return []
            
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            satellites = []
            for item in data:
                satellite = SatelliteData(
                    norad_id=item['norad_id'],
                    name=item['name'],
                    tle_line1=item['tle_line1'],
                    tle_line2=item['tle_line2'],
                    epoch=datetime.fromisoformat(item['epoch']),
                    inclination=item['inclination'],
                    eccentricity=item['eccentricity'],
                    perigee=item['perigee'],
                    apogee=item['apogee'],
                    period=item['period'],
                    altitude=item['altitude']
                )
                satellites.append(satellite)
            
            self.satellites_cache = {sat.norad_id: sat for sat in satellites}
            logger.info(f"Loaded {len(satellites)} satellites from cache")
            return satellites
            
        except Exception as e:
            logger.error(f"Error loading cached TLE data: {e}")
            return []
    
    def calculate_position(self, satellite: SatelliteData, target_time: datetime) -> Tuple[float, float, float]:
        """Calculate satellite position at target time using simplified SGP4"""
        try:
            # Time since epoch (minutes)
            dt = (target_time - satellite.epoch).total_seconds() / 60.0
            
            # Mean motion (rad/min)
            n = 2 * math.pi / (satellite.period * 60)  # Convert period from hours to minutes
            
            # Mean anomaly at target time
            M = math.radians(satellite.inclination) + n * dt  # Simplified
            
            # Solve Kepler's equation (simplified)
            E = M  # Initial guess
            for _ in range(5):  # Newton-Raphson iterations
                E = M + satellite.eccentricity * math.sin(E)
            
            # True anomaly
            nu = 2 * math.atan2(
                math.sqrt(1 + satellite.eccentricity) * math.sin(E/2),
                math.sqrt(1 - satellite.eccentricity) * math.cos(E/2)
            )
            
            # Distance from Earth center
            r = (satellite.perigee + self.R_EARTH) * (1 + satellite.eccentricity) / (1 + satellite.eccentricity * math.cos(nu))
            
            # Position in orbital plane
            x_orbital = r * math.cos(nu)
            y_orbital = r * math.sin(nu)
            
            # Convert to Earth-fixed coordinates (simplified)
            # This is a very simplified transformation - real SGP4 is much more complex
            i = math.radians(satellite.inclination)
            x = x_orbital
            y = y_orbital * math.cos(i)
            z = y_orbital * math.sin(i)
            
            return (x, y, z)
            
        except Exception as e:
            logger.error(f"Error calculating position for {satellite.name}: {e}")
            return (0.0, 0.0, 0.0)
    
    def assess_collision_risk(self, satellites: List[SatelliteData], time_window_hours: int = 72) -> List[CollisionRisk]:
        """Assess collision risks between satellites"""
        risks = []
        
        # Check all pairs of satellites
        for i, sat1 in enumerate(satellites):
            for sat2 in satellites[i+1:]:
                # Skip if satellites are in very different orbits
                if abs(sat1.altitude - sat2.altitude) > 200:  # 200 km difference
                    continue
                
                # Calculate closest approach over time window
                min_distance = float('inf')
                closest_time = None
                
                current_time = datetime.utcnow()
                for hours_ahead in range(0, time_window_hours, 1):
                    check_time = current_time + timedelta(hours=hours_ahead)
                    
                    pos1 = self.calculate_position(sat1, check_time)
                    pos2 = self.calculate_position(sat2, check_time)
                    
                    distance = math.sqrt(
                        (pos1[0] - pos2[0])**2 + 
                        (pos1[1] - pos2[1])**2 + 
                        (pos1[2] - pos2[2])**2
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_time = check_time
                
                # Assess risk level
                risk_level = self._determine_risk_level(min_distance)
                
                if risk_level != 'none' and closest_time is not None:
                    collision_prob = self._calculate_collision_probability(min_distance, sat1, sat2)
                    
                    risk = CollisionRisk(
                        primary_object=sat1.name,
                        secondary_object=sat2.name,
                        time_of_closest_approach=closest_time,
                        miss_distance=min_distance,
                        collision_probability=collision_prob,
                        risk_level=risk_level,
                        recommended_action=self._get_collision_recommendation(risk_level, collision_prob)
                    )
                    
                    risks.append(risk)
        
        # Sort by risk level and collision probability
        risk_order = {'extreme': 4, 'high': 3, 'medium': 2, 'low': 1}
        risks.sort(key=lambda x: (risk_order.get(x.risk_level, 0), x.collision_probability), reverse=True)
        
        return risks[:50]  # Return top 50 risks
    
    def _determine_risk_level(self, distance: float) -> str:
        """Determine risk level based on miss distance"""
        if distance <= self.collision_thresholds['extreme']:
            return 'extreme'
        elif distance <= self.collision_thresholds['high']:
            return 'high'
        elif distance <= self.collision_thresholds['medium']:
            return 'medium'
        elif distance <= self.collision_thresholds['low']:
            return 'low'
        else:
            return 'none'
    
    def _calculate_collision_probability(self, distance: float, sat1: SatelliteData, sat2: SatelliteData) -> float:
        """Calculate collision probability based on distance and object characteristics"""
        # Simplified collision probability calculation
        # In reality, this would consider object sizes, uncertainties, etc.
        
        # Assume typical satellite cross-sectional area
        cross_section = 10.0  # m²
        
        # Convert to probability (simplified model)
        if distance <= 0.001:  # 1 meter
            return 1.0
        elif distance <= 0.01:  # 10 meters
            return 0.8
        elif distance <= 0.1:  # 100 meters
            return 0.5
        elif distance <= 1.0:  # 1 km
            return 0.1
        elif distance <= 10.0:  # 10 km
            return 0.01
        else:
            return 0.001
    
    def _get_collision_recommendation(self, risk_level: str, probability: float) -> str:
        """Get recommendation for collision avoidance"""
        if risk_level == 'extreme' or probability > 0.5:
            return "IMMEDIATE MANEUVER REQUIRED - Execute emergency collision avoidance"
        elif risk_level == 'high' or probability > 0.1:
            return "PLAN MANEUVER - Prepare collision avoidance maneuver"
        elif risk_level == 'medium':
            return "MONITOR CLOSELY - Increase tracking frequency and prepare for possible maneuver"
        else:
            return "ROUTINE MONITORING - Continue normal tracking"
    
    def predict_atmospheric_drag(self, satellite: SatelliteData, days_ahead: int = 30) -> Dict:
        """Predict atmospheric drag effects and orbit decay"""
        try:
            # Get atmospheric density at satellite altitude
            density = self._get_atmospheric_density(satellite.altitude)
            
            # Calculate drag coefficient (simplified)
            drag_coeff = 2.2  # Typical value for satellites
            mass = 500.0  # kg (assumed)
            area = 10.0  # m² (assumed cross-sectional area)
            
            # Calculate drag acceleration
            velocity = math.sqrt(self.GM / (satellite.altitude + self.R_EARTH))  # km/s
            drag_acceleration = -0.5 * density * drag_coeff * area * velocity**2 / mass
            
            # Predict altitude decay
            current_altitude = satellite.altitude
            altitude_predictions = []
            
            for day in range(days_ahead):
                # Simplified orbit decay model
                altitude_loss_per_day = abs(drag_acceleration) * 86400 / 1000  # Convert to km/day
                current_altitude -= altitude_loss_per_day
                
                # Update density for new altitude
                if current_altitude > 100:  # Above Karman line
                    density = self._get_atmospheric_density(current_altitude)
                else:
                    # Satellite has re-entered
                    break
                
                altitude_predictions.append({
                    'day': day + 1,
                    'altitude': current_altitude,
                    'density': density,
                    'decay_rate': altitude_loss_per_day
                })
            
            # Predict re-entry if applicable
            reentry_prediction = None
            if altitude_predictions and altitude_predictions[-1]['altitude'] < 120:
                reentry_day = len(altitude_predictions)
                reentry_time = datetime.utcnow() + timedelta(days=reentry_day)
                
                reentry_prediction = ReentryPrediction(
                    object_id=satellite.name,
                    predicted_reentry_time=reentry_time,
                    uncertainty_window=timedelta(hours=12),
                    reentry_location=(0.0, 0.0),  # Simplified - would need detailed calculation
                    risk_assessment=self._assess_reentry_risk(satellite),
                    confidence_level=0.7
                )
            
            return {
                'satellite_id': satellite.norad_id,
                'satellite_name': satellite.name,
                'current_altitude': satellite.altitude,
                'predictions': altitude_predictions,
                'reentry_prediction': reentry_prediction,
                'drag_coefficient': drag_coeff,
                'atmospheric_density': density,
                'prediction_generated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting drag for {satellite.name}: {e}")
            return {'error': str(e)}
    
    def _get_atmospheric_density(self, altitude_km: float) -> float:
        """Get atmospheric density at given altitude (kg/m³)"""
        # Simplified atmospheric model
        if altitude_km < 100:
            return 1e-6  # Very dense below 100 km
        elif altitude_km < 200:
            return 5e-12
        elif altitude_km < 300:
            return 2e-13
        elif altitude_km < 400:
            return 3e-14
        elif altitude_km < 500:
            return 8e-15
        elif altitude_km < 600:
            return 3e-15
        elif altitude_km < 800:
            return 8e-16
        elif altitude_km < 1000:
            return 3e-16
        else:
            return 1e-17
    
    def _assess_reentry_risk(self, satellite: SatelliteData) -> str:
        """Assess risk from satellite re-entry"""
        # Simplified risk assessment based on satellite characteristics
        if satellite.altitude < 300:
            if 'ISS' in satellite.name.upper() or 'STATION' in satellite.name.upper():
                return 'HIGH - Large inhabited structure'
            elif satellite.perigee < 200:
                return 'MEDIUM - Potential debris survival'
            else:
                return 'LOW - Most components will burn up'
        else:
            return 'MINIMAL - Stable orbit'
    
    def generate_risk_report(self) -> Dict:
        """Generate comprehensive orbital risk report"""
        try:
            # Fetch latest satellite data
            satellites = self.fetch_tle_data()
            
            if not satellites:
                return {'error': 'No satellite data available'}
            
            # Assess collision risks
            collision_risks = self.assess_collision_risk(satellites[:500])  # Limit for performance
            
            # Predict drag for critical satellites
            drag_predictions = []
            critical_satellites = [sat for sat in satellites if sat.altitude < 500][:50]  # Low orbit satellites
            
            for sat in critical_satellites:
                drag_pred = self.predict_atmospheric_drag(sat)
                if 'error' not in drag_pred:
                    drag_predictions.append(drag_pred)
            
            # Calculate overall risk metrics
            high_collision_risks = [r for r in collision_risks if r.risk_level in ['extreme', 'high']]
            reentry_risks = [p for p in drag_predictions if p.get('reentry_prediction')]
            
            # Generate summary
            risk_summary = {
                'total_tracked_objects': len(satellites),
                'collision_risks_identified': len(collision_risks),
                'high_priority_collisions': len(high_collision_risks),
                'objects_at_reentry_risk': len(reentry_risks),
                'overall_risk_level': self._calculate_overall_risk(collision_risks, reentry_risks)
            }
            
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'risk_summary': risk_summary,
                'collision_risks': [
                    {
                        'primary': r.primary_object,
                        'secondary': r.secondary_object,
                        'closest_approach': r.time_of_closest_approach.isoformat(),
                        'miss_distance_km': round(r.miss_distance, 3),
                        'probability': round(r.collision_probability, 4),
                        'risk_level': r.risk_level,
                        'recommendation': r.recommended_action
                    }
                    for r in collision_risks[:20]  # Top 20 risks
                ],
                'drag_predictions': drag_predictions[:10],  # Top 10 predictions
                'recommendations': self._generate_operational_recommendations(collision_risks, reentry_risks)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_risk(self, collision_risks: List[CollisionRisk], reentry_risks: List) -> str:
        """Calculate overall orbital risk level"""
        extreme_risks = len([r for r in collision_risks if r.risk_level == 'extreme'])
        high_risks = len([r for r in collision_risks if r.risk_level == 'high'])
        
        if extreme_risks > 0 or len(reentry_risks) > 3:
            return 'EXTREME'
        elif high_risks > 5 or len(reentry_risks) > 1:
            return 'HIGH'
        elif high_risks > 0 or len(reentry_risks) > 0:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_operational_recommendations(self, collision_risks: List[CollisionRisk], reentry_risks: List) -> List[str]:
        """Generate operational recommendations"""
        recommendations = []
        
        # Collision-based recommendations
        extreme_collisions = [r for r in collision_risks if r.risk_level == 'extreme']
        if extreme_collisions:
            recommendations.append("CRITICAL: Execute immediate collision avoidance maneuvers for high-risk conjunctions")
        
        high_collisions = [r for r in collision_risks if r.risk_level == 'high']
        if high_collisions:
            recommendations.append("WARNING: Prepare collision avoidance maneuvers and increase tracking frequency")
        
        # Re-entry based recommendations
        if len(reentry_risks) > 0:
            recommendations.append("ALERT: Monitor re-entering objects and coordinate with air traffic control")
        
        # General recommendations
        if len(collision_risks) > 20:
            recommendations.append("NOTICE: High conjunction activity - increase situational awareness")
        
        if not recommendations:
            recommendations.append("NORMAL: Continue routine space situational awareness operations")
        
        return recommendations

def main():
    """Test the orbital intelligence system"""
    orbital_intel = OrbitalIntelligence()
    
    # Generate risk report
    print("Generating orbital risk report...")
    report = orbital_intel.generate_risk_report()
    
    if 'error' in report:
        print(f"Error: {report['error']}")
    else:
        print(f"\n=== Orbital Risk Report ===")
        print(f"Timestamp: {report['timestamp']}")
        print(f"\nRisk Summary:")
        for key, value in report['risk_summary'].items():
            print(f"  {key}: {value}")
        
        print(f"\nTop Collision Risks:")
        for i, risk in enumerate(report['collision_risks'][:5], 1):
            print(f"  {i}. {risk['primary']} vs {risk['secondary']}")
            print(f"     Miss distance: {risk['miss_distance_km']} km")
            print(f"     Risk level: {risk['risk_level']}")
        
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

if __name__ == "__main__":
    main()