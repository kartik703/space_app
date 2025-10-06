"""
Fusion AI System - Combines CV, Time-Series, and Orbital Intelligence
Central brain for space risk assessment and prediction
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import our custom modules
from .advanced_space_cv import AdvancedSpaceCV, SpacePhenomena
from .time_series_forecaster import SpaceWeatherForecaster, ForecastResult
from .orbital_intelligence import OrbitalIntelligence, CollisionRisk, ReentryPrediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FusionPrediction:
    """Comprehensive fusion prediction"""
    timestamp: datetime
    overall_risk_score: float  # 0-100
    risk_category: str  # low, medium, high, extreme
    cv_analysis: Dict
    time_series_forecast: Dict
    orbital_assessment: Dict
    confidence_level: float
    recommendations: List[str]
    financial_impact: Dict
    alerts: List[Dict]

class RiskFusionNetwork(nn.Module):
    """Neural network for fusing different risk assessment modalities"""
    
    def __init__(self, cv_features: int = 64, ts_features: int = 32, orbital_features: int = 48):
        super(RiskFusionNetwork, self).__init__()
        
        # Feature extractors for each modality
        self.cv_encoder = nn.Sequential(
            nn.Linear(cv_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        self.ts_encoder = nn.Sequential(
            nn.Linear(ts_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        self.orbital_encoder = nn.Sequential(
            nn.Linear(orbital_features, 96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 24)
        )
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=72,  # 32 + 16 + 24
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_network = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # Risk score output
        )
        
    def forward(self, cv_features, ts_features, orbital_features):
        # Encode each modality
        cv_encoded = self.cv_encoder(cv_features)
        ts_encoded = self.ts_encoder(ts_features)
        orbital_encoded = self.orbital_encoder(orbital_features)
        
        # Concatenate encoded features
        fused_features = torch.cat([cv_encoded, ts_encoded, orbital_encoded], dim=-1)
        
        # Apply cross-attention (self-attention on fused features)
        fused_features = fused_features.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.cross_attention(fused_features, fused_features, fused_features)
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Final risk prediction
        risk_score = self.fusion_network(attended_features)
        
        return torch.sigmoid(risk_score) * 100  # Scale to 0-100

class SpaceRiskFusionAI:
    """
    Master AI system that fuses Computer Vision, Time-Series, and Orbital Intelligence
    for comprehensive space risk assessment
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component systems
        self.cv_system = AdvancedSpaceCV(str(self.models_dir))
        self.ts_forecaster = SpaceWeatherForecaster(str(self.models_dir / "forecasting"))
        self.orbital_intel = OrbitalIntelligence(str(self.models_dir / "orbital"))
        
        # Load fusion model
        self.fusion_model = RiskFusionNetwork()
        self.load_fusion_model()
        
        # Risk thresholds and weights
        self.risk_weights = {
            'cv_weight': 0.35,
            'ts_weight': 0.30,
            'orbital_weight': 0.35
        }
        
        # Financial impact models (simplified)
        self.financial_models = {
            'satellite_value': 100e6,  # $100M average satellite value
            'insurance_premium_rate': 0.05,  # 5% of satellite value
            'operational_cost_per_day': 50000,  # $50k per day
            'launch_cost': 60e6  # $60M average launch cost
        }
        
    def load_fusion_model(self):
        """Load pre-trained fusion model"""
        model_path = self.models_dir / "fusion_risk_model.pt"
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
                self.fusion_model.eval()
                logger.info("Loaded fusion model")
            except Exception as e:
                logger.warning(f"Failed to load fusion model: {e}")
        else:
            logger.info("No pre-trained fusion model found, using default initialization")
    
    def save_fusion_model(self):
        """Save fusion model"""
        model_path = self.models_dir / "fusion_risk_model.pt"
        torch.save({
            'model_state_dict': self.fusion_model.state_dict(),
            'weights': self.risk_weights,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        logger.info(f"Saved fusion model to {model_path}")
    
    def extract_cv_features(self, cv_analysis: Dict) -> np.ndarray:
        """Extract numerical features from CV analysis"""
        features = []
        
        # Basic metrics
        features.append(cv_analysis.get('total_detections', 0))
        features.append(cv_analysis.get('risk_score', 0))
        
        # Phenomenon-specific features
        phenomena_counts = {}
        for detection in cv_analysis.get('detections', []):
            phenomenon = detection.get('phenomenon', 'unknown')
            phenomena_counts[phenomenon] = phenomena_counts.get(phenomenon, 0) + 1
        
        # One-hot encoding for phenomena
        for phenomenon in SpacePhenomena:
            features.append(phenomena_counts.get(phenomenon.value, 0))
        
        # Severity features
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'extreme': 0}
        for detection in cv_analysis.get('detections', []):
            severity = detection.get('severity', 'low')
            severity_counts[severity] += 1
        
        features.extend(severity_counts.values())
        
        # Confidence features
        confidences = [d.get('confidence', 0) for d in cv_analysis.get('detections', [])]
        features.append(np.mean(confidences) if confidences else 0)
        features.append(np.max(confidences) if confidences else 0)
        features.append(np.std(confidences) if confidences else 0)
        
        # Pad or truncate to fixed size
        target_size = 64
        if len(features) < target_size:
            features.extend([0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def extract_ts_features(self, ts_forecast: Dict) -> np.ndarray:
        """Extract numerical features from time-series forecast"""
        features = []
        
        # Overall risk level
        risk_mapping = {'low': 1, 'medium': 2, 'high': 3, 'extreme': 4}
        overall_risk = ts_forecast.get('overall_risk', 'low')
        features.append(risk_mapping.get(overall_risk, 1))
        
        # Individual parameter forecasts
        forecasts = ts_forecast.get('individual_forecasts', {})
        for param in ['kp_index', 'dst_index', 'solar_wind_speed', 'proton_flux']:
            param_data = forecasts.get(param, {})
            if 'error' not in param_data:
                features.append(risk_mapping.get(param_data.get('risk_level', 'low'), 1))
                features.append(param_data.get('model_accuracy', 0.5))
                
                # Forecast values statistics
                forecast_values = param_data.get('forecast_values', [])
                if forecast_values:
                    features.extend([
                        np.mean(forecast_values),
                        np.max(forecast_values),
                        np.min(forecast_values),
                        np.std(forecast_values)
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([1, 0.5, 0, 0, 0, 0])  # Default values for missing data
        
        # Pad or truncate to fixed size
        target_size = 32
        if len(features) < target_size:
            features.extend([0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def extract_orbital_features(self, orbital_assessment: Dict) -> np.ndarray:
        """Extract numerical features from orbital assessment"""
        features = []
        
        # Risk summary features
        risk_summary = orbital_assessment.get('risk_summary', {})
        features.append(risk_summary.get('total_tracked_objects', 0))
        features.append(risk_summary.get('collision_risks_identified', 0))
        features.append(risk_summary.get('high_priority_collisions', 0))
        features.append(risk_summary.get('objects_at_reentry_risk', 0))
        
        # Overall risk level
        risk_mapping = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'EXTREME': 4}
        overall_risk = risk_summary.get('overall_risk_level', 'LOW')
        features.append(risk_mapping.get(overall_risk, 1))
        
        # Collision risk features
        collision_risks = orbital_assessment.get('collision_risks', [])
        if collision_risks:
            # Statistics from top collision risks
            miss_distances = [r.get('miss_distance_km', 1000) for r in collision_risks[:10]]
            probabilities = [r.get('probability', 0) for r in collision_risks[:10]]
            
            features.extend([
                np.mean(miss_distances),
                np.min(miss_distances),
                np.max(probabilities),
                np.mean(probabilities)
            ])
            
            # Risk level distribution
            risk_levels = [r.get('risk_level', 'low') for r in collision_risks]
            risk_dist = {'low': 0, 'medium': 0, 'high': 0, 'extreme': 0}
            for level in risk_levels:
                risk_dist[level] += 1
            features.extend(risk_dist.values())
        else:
            features.extend([1000, 1000, 0, 0, 0, 0, 0, 0])  # Default values
        
        # Drag prediction features
        drag_predictions = orbital_assessment.get('drag_predictions', [])
        if drag_predictions:
            # Average current altitude and decay rates
            altitudes = [p.get('current_altitude', 500) for p in drag_predictions]
            features.append(np.mean(altitudes))
            features.append(np.min(altitudes))
            
            # Count objects with reentry predictions
            reentry_count = sum(1 for p in drag_predictions if p.get('reentry_prediction'))
            features.append(reentry_count)
        else:
            features.extend([500, 500, 0])  # Default values
        
        # Pad or truncate to fixed size
        target_size = 48
        if len(features) < target_size:
            features.extend([0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def calculate_financial_impact(self, risk_score: float, cv_analysis: Dict, 
                                 orbital_assessment: Dict) -> Dict:
        """Calculate financial impact based on risk assessment"""
        
        # Base financial metrics
        satellite_value = self.financial_models['satellite_value']
        
        # Risk-based impact calculation
        risk_multiplier = risk_score / 100.0
        
        # CV-based impact (solar storms affect electronics)
        cv_impact = 0
        storm_detections = len([d for d in cv_analysis.get('detections', []) 
                              if d.get('phenomenon') in ['solar_flare', 'coronal_mass_ejection']])
        if storm_detections > 0:
            cv_impact = storm_detections * satellite_value * 0.1 * risk_multiplier
        
        # Orbital collision impact
        orbital_impact = 0
        collision_risks = orbital_assessment.get('collision_risks', [])
        high_risk_collisions = [r for r in collision_risks if r.get('risk_level') in ['high', 'extreme']]
        if high_risk_collisions:
            max_collision_prob = max(r.get('probability', 0) for r in high_risk_collisions)
            orbital_impact = max_collision_prob * satellite_value
        
        # Operational disruption costs
        operational_impact = risk_score * self.financial_models['operational_cost_per_day'] / 10
        
        # Insurance implications
        insurance_impact = risk_score * satellite_value * self.financial_models['insurance_premium_rate'] / 20
        
        total_impact = cv_impact + orbital_impact + operational_impact + insurance_impact
        
        return {
            'total_financial_impact': total_impact,
            'cv_related_impact': cv_impact,
            'orbital_collision_impact': orbital_impact,
            'operational_disruption': operational_impact,
            'insurance_implications': insurance_impact,
            'risk_multiplier': risk_multiplier,
            'affected_satellite_value': satellite_value,
            'impact_breakdown': {
                'immediate_risk': cv_impact + orbital_impact,
                'operational_costs': operational_impact,
                'insurance_costs': insurance_impact
            }
        }
    
    def generate_alerts(self, risk_score: float, cv_analysis: Dict, 
                       ts_forecast: Dict, orbital_assessment: Dict) -> List[Dict]:
        """Generate prioritized alerts based on fusion analysis"""
        alerts = []
        
        # High-level risk alert
        if risk_score >= 80:
            alerts.append({
                'priority': 'CRITICAL',
                'type': 'overall_risk',
                'message': f"EXTREME RISK DETECTED - Overall risk score: {risk_score:.1f}/100",
                'action_required': 'Implement emergency protocols immediately',
                'timestamp': datetime.now().isoformat()
            })
        elif risk_score >= 60:
            alerts.append({
                'priority': 'HIGH',
                'type': 'overall_risk',
                'message': f"HIGH RISK - Overall risk score: {risk_score:.1f}/100",
                'action_required': 'Activate elevated monitoring and prepare contingencies',
                'timestamp': datetime.now().isoformat()
            })
        
        # CV-specific alerts
        cv_risk = cv_analysis.get('risk_score', 0)
        if cv_risk >= 70:
            storm_events = [d for d in cv_analysis.get('detections', []) 
                          if d.get('phenomenon') in ['solar_flare', 'coronal_mass_ejection']]
            if storm_events:
                alerts.append({
                    'priority': 'HIGH',
                    'type': 'space_weather',
                    'message': f"Major solar storm activity detected: {len(storm_events)} events",
                    'action_required': 'Consider satellite safe mode and radiation protection',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Time-series alerts
        ts_risk = ts_forecast.get('overall_risk', 'low')
        if ts_risk in ['high', 'extreme']:
            alerts.append({
                'priority': 'HIGH' if ts_risk == 'high' else 'CRITICAL',
                'type': 'space_weather_forecast',
                'message': f"Space weather forecast indicates {ts_risk} risk conditions",
                'action_required': 'Monitor geomagnetic indices and prepare for enhanced activity',
                'timestamp': datetime.now().isoformat()
            })
        
        # Orbital alerts
        orbital_risk = orbital_assessment.get('risk_summary', {}).get('overall_risk_level', 'LOW')
        if orbital_risk in ['HIGH', 'EXTREME']:
            high_collision_risks = orbital_assessment.get('risk_summary', {}).get('high_priority_collisions', 0)
            if high_collision_risks > 0:
                alerts.append({
                    'priority': 'CRITICAL',
                    'type': 'collision_risk',
                    'message': f"High collision risk: {high_collision_risks} critical conjunctions detected",
                    'action_required': 'Execute collision avoidance maneuvers if necessary',
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def comprehensive_analysis(self, image_path: Optional[str] = None, 
                             include_forecasting: bool = True,
                             include_orbital: bool = True) -> FusionPrediction:
        """Perform comprehensive multi-modal space risk analysis"""
        
        try:
            logger.info("Starting comprehensive fusion analysis...")
            
            # Computer Vision Analysis
            cv_analysis = {}
            if image_path and Path(image_path).exists():
                cv_analysis = self.cv_system.analyze_image(image_path)
                logger.info("CV analysis completed")
            else:
                # Use default/dummy analysis if no image provided
                cv_analysis = {
                    'total_detections': 0,
                    'risk_score': 0,
                    'detections': [],
                    'summary': {'status': 'No image provided for analysis'}
                }
            
            # Time-Series Forecasting
            ts_forecast = {}
            if include_forecasting:
                try:
                    ts_forecast = self.ts_forecaster.multi_parameter_forecast()
                    logger.info("Time-series forecasting completed")
                except Exception as e:
                    logger.warning(f"Time-series forecasting failed: {e}")
                    ts_forecast = {
                        'overall_risk': 'low',
                        'individual_forecasts': {},
                        'error': str(e)
                    }
            
            # Orbital Intelligence
            orbital_assessment = {}
            if include_orbital:
                try:
                    orbital_assessment = self.orbital_intel.generate_risk_report()
                    logger.info("Orbital intelligence assessment completed")
                except Exception as e:
                    logger.warning(f"Orbital assessment failed: {e}")
                    orbital_assessment = {
                        'risk_summary': {'overall_risk_level': 'LOW'},
                        'collision_risks': [],
                        'error': str(e)
                    }
            
            # Extract features for fusion model
            cv_features = self.extract_cv_features(cv_analysis)
            ts_features = self.extract_ts_features(ts_forecast)
            orbital_features = self.extract_orbital_features(orbital_assessment)
            
            # Run fusion model
            self.fusion_model.eval()
            with torch.no_grad():
                cv_tensor = torch.FloatTensor(cv_features).unsqueeze(0)
                ts_tensor = torch.FloatTensor(ts_features).unsqueeze(0)
                orbital_tensor = torch.FloatTensor(orbital_features).unsqueeze(0)
                
                fusion_risk_score = self.fusion_model(cv_tensor, ts_tensor, orbital_tensor).item()
            
            # Determine risk category
            if fusion_risk_score >= 80:
                risk_category = 'extreme'
            elif fusion_risk_score >= 60:
                risk_category = 'high'
            elif fusion_risk_score >= 40:
                risk_category = 'medium'
            else:
                risk_category = 'low'
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence(cv_analysis, ts_forecast, orbital_assessment)
            
            # Generate recommendations
            recommendations = self._generate_fusion_recommendations(
                fusion_risk_score, cv_analysis, ts_forecast, orbital_assessment
            )
            
            # Calculate financial impact
            financial_impact = self.calculate_financial_impact(
                fusion_risk_score, cv_analysis, orbital_assessment
            )
            
            # Generate alerts
            alerts = self.generate_alerts(
                fusion_risk_score, cv_analysis, ts_forecast, orbital_assessment
            )
            
            # Create comprehensive prediction
            prediction = FusionPrediction(
                timestamp=datetime.now(),
                overall_risk_score=fusion_risk_score,
                risk_category=risk_category,
                cv_analysis=cv_analysis,
                time_series_forecast=ts_forecast,
                orbital_assessment=orbital_assessment,
                confidence_level=confidence_level,
                recommendations=recommendations,
                financial_impact=financial_impact,
                alerts=alerts
            )
            
            logger.info(f"Fusion analysis completed - Risk Score: {fusion_risk_score:.1f}, Category: {risk_category}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            # Return minimal prediction with error
            return FusionPrediction(
                timestamp=datetime.now(),
                overall_risk_score=0,
                risk_category='unknown',
                cv_analysis={'error': str(e)},
                time_series_forecast={'error': str(e)},
                orbital_assessment={'error': str(e)},
                confidence_level=0,
                recommendations=[f"Analysis failed: {str(e)}"],
                financial_impact={'error': str(e)},
                alerts=[{
                    'priority': 'ERROR',
                    'type': 'system_error',
                    'message': f"Fusion analysis failed: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                }]
            )
    
    def _calculate_confidence(self, cv_analysis: Dict, ts_forecast: Dict, orbital_assessment: Dict) -> float:
        """Calculate overall confidence in the analysis"""
        confidences = []
        
        # CV confidence
        if 'error' not in cv_analysis:
            detections = cv_analysis.get('detections', [])
            if detections:
                cv_conf = np.mean([d.get('confidence', 0) for d in detections])
                confidences.append(cv_conf)
            else:
                confidences.append(0.8)  # High confidence in "no detection"
        
        # TS confidence
        if 'error' not in ts_forecast:
            forecast_conf = ts_forecast.get('summary', {}).get('forecast_confidence', 'medium')
            conf_mapping = {'low': 0.3, 'medium': 0.7, 'high': 0.9}
            confidences.append(conf_mapping.get(forecast_conf, 0.5))
        
        # Orbital confidence (based on data availability)
        if 'error' not in orbital_assessment:
            objects_tracked = orbital_assessment.get('risk_summary', {}).get('total_tracked_objects', 0)
            if objects_tracked > 1000:
                confidences.append(0.9)
            elif objects_tracked > 100:
                confidences.append(0.7)
            else:
                confidences.append(0.5)
        
        return float(np.mean(confidences)) if confidences else 0.5
    
    def _generate_fusion_recommendations(self, risk_score: float, cv_analysis: Dict, 
                                       ts_forecast: Dict, orbital_assessment: Dict) -> List[str]:
        """Generate comprehensive recommendations based on fusion analysis"""
        recommendations = []
        
        # Overall risk recommendations
        if risk_score >= 80:
            recommendations.append("CRITICAL: Implement emergency space weather and collision avoidance protocols")
            recommendations.append("Consider placing all satellites in safe mode")
            recommendations.append("Activate 24/7 monitoring with reduced decision latency")
        elif risk_score >= 60:
            recommendations.append("HIGH RISK: Increase monitoring frequency and prepare contingency plans")
            recommendations.append("Review and update collision avoidance procedures")
        elif risk_score >= 40:
            recommendations.append("MODERATE RISK: Maintain enhanced situational awareness")
        else:
            recommendations.append("NORMAL: Continue routine operations with standard monitoring")
        
        # CV-specific recommendations
        cv_risk = cv_analysis.get('risk_score', 0)
        if cv_risk >= 50:
            storm_events = [d for d in cv_analysis.get('detections', []) 
                          if d.get('phenomenon') in ['solar_flare', 'coronal_mass_ejection']]
            if storm_events:
                recommendations.append("Space weather alert: Monitor for enhanced radiation and geomagnetic effects")
        
        # Orbital-specific recommendations
        orbital_risks = orbital_assessment.get('collision_risks', [])
        high_risk_collisions = [r for r in orbital_risks if r.get('risk_level') in ['high', 'extreme']]
        if high_risk_collisions:
            recommendations.append(f"Collision alert: {len(high_risk_collisions)} high-risk conjunctions require attention")
        
        # Financial recommendations
        recommendations.append("Review insurance coverage and operational contingency funds")
        
        return recommendations
    
    def export_analysis_report(self, prediction: FusionPrediction, output_path: str):
        """Export comprehensive analysis report"""
        try:
            report = {
                'meta': {
                    'timestamp': prediction.timestamp.isoformat(),
                    'analysis_type': 'fusion_ai_comprehensive',
                    'version': '1.0'
                },
                'executive_summary': {
                    'overall_risk_score': prediction.overall_risk_score,
                    'risk_category': prediction.risk_category,
                    'confidence_level': prediction.confidence_level,
                    'total_financial_impact': prediction.financial_impact.get('total_financial_impact', 0),
                    'critical_alerts': len([a for a in prediction.alerts if a.get('priority') == 'CRITICAL'])
                },
                'detailed_analysis': {
                    'computer_vision': prediction.cv_analysis,
                    'time_series_forecast': prediction.time_series_forecast,
                    'orbital_assessment': prediction.orbital_assessment
                },
                'risk_assessment': {
                    'overall_score': prediction.overall_risk_score,
                    'category': prediction.risk_category,
                    'confidence': prediction.confidence_level,
                    'financial_impact': prediction.financial_impact
                },
                'alerts_and_recommendations': {
                    'alerts': prediction.alerts,
                    'recommendations': prediction.recommendations
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Analysis report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")

def main():
    """Test the fusion AI system"""
    fusion_ai = SpaceRiskFusionAI()
    
    # Run comprehensive analysis
    print("Running comprehensive fusion AI analysis...")
    
    # Test with sample image if available
    test_image = "data/enhanced_distinct_dataset/images/quiet_sun_001.jpg"
    prediction = fusion_ai.comprehensive_analysis(
        image_path=test_image if Path(test_image).exists() else None,
        include_forecasting=True,
        include_orbital=True
    )
    
    print(f"\n=== Fusion AI Analysis Results ===")
    print(f"Timestamp: {prediction.timestamp}")
    print(f"Overall Risk Score: {prediction.overall_risk_score:.1f}/100")
    print(f"Risk Category: {prediction.risk_category.upper()}")
    print(f"Confidence Level: {prediction.confidence_level:.2f}")
    
    print(f"\nFinancial Impact:")
    print(f"  Total: ${prediction.financial_impact.get('total_financial_impact', 0):,.0f}")
    
    print(f"\nAlerts ({len(prediction.alerts)}):")
    for alert in prediction.alerts[:3]:
        print(f"  {alert.get('priority')}: {alert.get('message')}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(prediction.recommendations[:5], 1):
        print(f"  {i}. {rec}")
    
    # Export report
    output_file = "fusion_ai_analysis_report.json"
    fusion_ai.export_analysis_report(prediction, output_file)
    print(f"\nDetailed report exported to {output_file}")

if __name__ == "__main__":
    main()