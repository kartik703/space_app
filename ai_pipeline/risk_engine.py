"""
Advanced Risk Engine - Intelligent risk scoring and real-time decision support
Provides dynamic risk assessment, scenario modeling, and automated responses
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import threading
import time
from queue import Queue, PriorityQueue
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"

class RiskCategory(Enum):
    """Risk category types"""
    SPACE_WEATHER = "space_weather"
    ORBITAL_COLLISION = "orbital_collision"
    SATELLITE_MALFUNCTION = "satellite_malfunction"
    COMMUNICATION_DISRUPTION = "communication_disruption"
    FINANCIAL_LOSS = "financial_loss"
    MISSION_FAILURE = "mission_failure"
    HUMAN_SAFETY = "human_safety"

class ActionPriority(Enum):
    """Action priority levels"""
    IMMEDIATE = 1
    URGENT = 2
    HIGH = 3
    MEDIUM = 4
    LOW = 5

@dataclass
class RiskFactor:
    """Individual risk factor"""
    name: str
    category: RiskCategory
    current_value: float  # 0-100
    threshold_values: Dict[RiskLevel, float]
    weight: float  # Importance weight 0-1
    trend: str  # increasing, decreasing, stable
    confidence: float  # 0-1
    last_updated: datetime
    source: str  # Data source

@dataclass
class RiskScenario:
    """Risk scenario definition"""
    scenario_id: str
    name: str
    description: str
    probability: float  # 0-1
    impact_score: float  # 0-100
    risk_factors: List[str]  # Factor names
    triggers: Dict[str, Any]  # Trigger conditions
    consequences: List[str]
    mitigation_actions: List[str]
    estimated_cost: float
    response_time_required: timedelta

@dataclass
class RiskAlert:
    """Risk alert structure"""
    alert_id: str
    timestamp: datetime
    risk_level: RiskLevel
    category: RiskCategory
    priority: ActionPriority
    title: str
    description: str
    affected_systems: List[str]
    recommended_actions: List[str]
    estimated_impact: Dict[str, float]
    escalation_path: List[str]
    auto_resolve: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class DecisionRecommendation:
    """Automated decision recommendation"""
    recommendation_id: str
    timestamp: datetime
    risk_context: Dict[str, Any]
    recommended_action: str
    confidence_level: float
    expected_outcome: str
    cost_benefit_analysis: Dict[str, float]
    implementation_time: timedelta
    success_probability: float
    fallback_options: List[str]

class RealTimeRiskMonitor:
    """Real-time risk monitoring and alerting"""
    
    def __init__(self, alert_queue: PriorityQueue):
        self.alert_queue = alert_queue
        self.monitoring = False
        self.monitor_thread = None
        self.risk_thresholds = {
            RiskLevel.CRITICAL: 80,
            RiskLevel.HIGH: 60,
            RiskLevel.MODERATE: 40,
            RiskLevel.LOW: 20,
            RiskLevel.MINIMAL: 0
        }
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Real-time risk monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Real-time risk monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Simulate real-time risk assessment
                self._check_risk_conditions()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def _check_risk_conditions(self):
        """Check current risk conditions and generate alerts"""
        # This would integrate with real data sources
        # For now, simulate some risk checks
        current_time = datetime.now()
        
        # Example: Check for sudden risk escalation
        risk_score = np.random.uniform(0, 100)  # In real system, this comes from fusion AI
        
        if risk_score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            alert = RiskAlert(
                alert_id=f"RISK_{current_time.strftime('%Y%m%d_%H%M%S')}",
                timestamp=current_time,
                risk_level=RiskLevel.CRITICAL,
                category=RiskCategory.SPACE_WEATHER,
                priority=ActionPriority.IMMEDIATE,
                title="Critical Risk Level Detected",
                description=f"Overall risk score has reached {risk_score:.1f}",
                affected_systems=["satellites", "communication", "navigation"],
                recommended_actions=["Activate emergency protocols", "Notify mission control"],
                estimated_impact={"financial": 1e6, "operational": 90},
                escalation_path=["Operations Manager", "Mission Director", "Crisis Team"]
            )
            
            # Add to priority queue (lower number = higher priority)
            self.alert_queue.put((alert.priority.value, alert))

class AdvancedRiskEngine:
    """
    Advanced risk assessment and decision support engine
    Provides intelligent risk scoring, scenario modeling, and automated responses
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.risk_factors: Dict[str, RiskFactor] = {}
        self.risk_scenarios: Dict[str, RiskScenario] = {}
        self.alert_queue = PriorityQueue()
        self.risk_monitor = RealTimeRiskMonitor(self.alert_queue)
        self.decision_history: List[DecisionRecommendation] = []
        
        # Risk scoring models
        self.risk_weights = {
            RiskCategory.SPACE_WEATHER: 0.25,
            RiskCategory.ORBITAL_COLLISION: 0.20,
            RiskCategory.SATELLITE_MALFUNCTION: 0.15,
            RiskCategory.COMMUNICATION_DISRUPTION: 0.15,
            RiskCategory.FINANCIAL_LOSS: 0.10,
            RiskCategory.MISSION_FAILURE: 0.10,
            RiskCategory.HUMAN_SAFETY: 0.05
        }
        
        # Load configuration and initialize
        self._initialize_risk_factors()
        self._initialize_risk_scenarios()
        
    def _initialize_risk_factors(self):
        """Initialize standard risk factors"""
        
        # Space Weather Risk Factors
        self.risk_factors["kp_index"] = RiskFactor(
            name="Kp Index",
            category=RiskCategory.SPACE_WEATHER,
            current_value=30.0,
            threshold_values={
                RiskLevel.MINIMAL: 0, RiskLevel.LOW: 20, RiskLevel.MODERATE: 40,
                RiskLevel.HIGH: 60, RiskLevel.CRITICAL: 80, RiskLevel.EXTREME: 90
            },
            weight=0.8,
            trend="stable",
            confidence=0.85,
            last_updated=datetime.now(),
            source="NOAA_SWPC"
        )
        
        self.risk_factors["solar_flare_activity"] = RiskFactor(
            name="Solar Flare Activity",
            category=RiskCategory.SPACE_WEATHER,
            current_value=25.0,
            threshold_values={
                RiskLevel.MINIMAL: 0, RiskLevel.LOW: 15, RiskLevel.MODERATE: 35,
                RiskLevel.HIGH: 55, RiskLevel.CRITICAL: 75, RiskLevel.EXTREME: 90
            },
            weight=0.9,
            trend="increasing",
            confidence=0.90,
            last_updated=datetime.now(),
            source="JSOC_SDO"
        )
        
        # Orbital Risk Factors
        self.risk_factors["collision_probability"] = RiskFactor(
            name="Collision Probability",
            category=RiskCategory.ORBITAL_COLLISION,
            current_value=15.0,
            threshold_values={
                RiskLevel.MINIMAL: 0, RiskLevel.LOW: 10, RiskLevel.MODERATE: 25,
                RiskLevel.HIGH: 50, RiskLevel.CRITICAL: 75, RiskLevel.EXTREME: 90
            },
            weight=0.95,
            trend="stable",
            confidence=0.80,
            last_updated=datetime.now(),
            source="Space_Track"
        )
        
        # Communication Risk Factors
        self.risk_factors["signal_degradation"] = RiskFactor(
            name="Signal Degradation",
            category=RiskCategory.COMMUNICATION_DISRUPTION,
            current_value=20.0,
            threshold_values={
                RiskLevel.MINIMAL: 0, RiskLevel.LOW: 15, RiskLevel.MODERATE: 30,
                RiskLevel.HIGH: 50, RiskLevel.CRITICAL: 70, RiskLevel.EXTREME: 85
            },
            weight=0.7,
            trend="stable",
            confidence=0.75,
            last_updated=datetime.now(),
            source="Ground_Stations"
        )
        
        logger.info(f"Initialized {len(self.risk_factors)} risk factors")
    
    def _initialize_risk_scenarios(self):
        """Initialize predefined risk scenarios"""
        
        # Major Solar Storm Scenario
        self.risk_scenarios["major_solar_storm"] = RiskScenario(
            scenario_id="major_solar_storm",
            name="Major Solar Storm Event",
            description="Large-scale solar storm with X-class flares and fast CME",
            probability=0.15,
            impact_score=85.0,
            risk_factors=["kp_index", "solar_flare_activity", "signal_degradation"],
            triggers={
                "kp_index": {"operator": ">=", "value": 70},
                "solar_flare_activity": {"operator": ">=", "value": 60}
            },
            consequences=[
                "Satellite electronics damage",
                "GPS accuracy degradation",
                "Radio blackouts",
                "Power grid instabilities",
                "Radiation exposure risks"
            ],
            mitigation_actions=[
                "Switch satellites to safe mode",
                "Increase shielding protocols",
                "Activate backup communication systems",
                "Issue radiation warnings",
                "Prepare emergency power systems"
            ],
            estimated_cost=50e6,
            response_time_required=timedelta(hours=2)
        )
        
        # Orbital Collision Scenario
        self.risk_scenarios["high_speed_collision"] = RiskScenario(
            scenario_id="high_speed_collision",
            name="High-Speed Orbital Collision",
            description="Potential collision between active satellite and space debris",
            probability=0.08,
            impact_score=90.0,
            risk_factors=["collision_probability"],
            triggers={
                "collision_probability": {"operator": ">=", "value": 60}
            },
            consequences=[
                "Complete satellite loss",
                "Additional debris generation",
                "Mission failure",
                "Insurance claims",
                "Cascade effect risks"
            ],
            mitigation_actions=[
                "Execute collision avoidance maneuver",
                "Coordinate with other operators",
                "Update orbital tracking",
                "Prepare replacement mission",
                "Activate insurance protocols"
            ],
            estimated_cost=100e6,
            response_time_required=timedelta(hours=6)
        )
        
        logger.info(f"Initialized {len(self.risk_scenarios)} risk scenarios")
    
    def update_risk_factor(self, factor_name: str, new_value: float, 
                          confidence: float = 1.0, source: str = "manual"):
        """Update a risk factor with new data"""
        if factor_name in self.risk_factors:
            factor = self.risk_factors[factor_name]
            
            # Determine trend
            if new_value > factor.current_value * 1.1:
                trend = "increasing"
            elif new_value < factor.current_value * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Update factor
            factor.current_value = new_value
            factor.trend = trend
            factor.confidence = confidence
            factor.last_updated = datetime.now()
            factor.source = source
            
            logger.info(f"Updated risk factor '{factor_name}': {new_value:.1f} ({trend})")
            
            # Check for alerts
            self._check_factor_alerts(factor)
        else:
            logger.warning(f"Risk factor '{factor_name}' not found")
    
    def _check_factor_alerts(self, factor: RiskFactor):
        """Check if factor triggers any alerts"""
        current_level = self._get_risk_level(factor.current_value, factor.threshold_values)
        
        if current_level in [RiskLevel.CRITICAL, RiskLevel.EXTREME]:
            alert = RiskAlert(
                alert_id=f"FACTOR_{factor.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                risk_level=current_level,
                category=factor.category,
                priority=ActionPriority.URGENT if current_level == RiskLevel.CRITICAL else ActionPriority.HIGH,
                title=f"Critical Risk Factor: {factor.name}",
                description=f"{factor.name} has reached {current_level.value} level with value {factor.current_value:.1f}",
                affected_systems=[factor.category.value],
                recommended_actions=[f"Investigate {factor.name} escalation", "Implement containment measures"],
                estimated_impact={"risk_score": factor.current_value * factor.weight},
                escalation_path=["Risk Manager", "Operations Director"]
            )
            
            self.alert_queue.put((alert.priority.value, alert))
    
    def _get_risk_level(self, value: float, thresholds: Dict[RiskLevel, float]) -> RiskLevel:
        """Determine risk level based on value and thresholds"""
        for level in [RiskLevel.EXTREME, RiskLevel.CRITICAL, RiskLevel.HIGH, 
                     RiskLevel.MODERATE, RiskLevel.LOW, RiskLevel.MINIMAL]:
            if value >= thresholds[level]:
                return level
        return RiskLevel.MINIMAL
    
    def calculate_composite_risk_score(self) -> Dict[str, Any]:
        """Calculate overall composite risk score"""
        category_scores = {}
        
        # Calculate risk score for each category
        for category in RiskCategory:
            category_factors = [f for f in self.risk_factors.values() if f.category == category]
            
            if category_factors:
                # Weighted average of factors in this category
                weighted_sum = sum(f.current_value * f.weight * f.confidence for f in category_factors)
                total_weight = sum(f.weight * f.confidence for f in category_factors)
                category_score = weighted_sum / total_weight if total_weight > 0 else 0
                category_scores[category.value] = category_score
            else:
                category_scores[category.value] = 0
        
        # Calculate overall risk score
        overall_score = sum(
            category_scores[category.value] * self.risk_weights[category] 
            for category in RiskCategory
        )
        
        # Determine overall risk level
        overall_thresholds = {
            RiskLevel.MINIMAL: 0.0, RiskLevel.LOW: 20.0, RiskLevel.MODERATE: 40.0,
            RiskLevel.HIGH: 60.0, RiskLevel.CRITICAL: 80.0, RiskLevel.EXTREME: 90.0
        }
        overall_level = self._get_risk_level(overall_score, overall_thresholds)
        
        return {
            'overall_score': overall_score,
            'overall_level': overall_level.value,
            'category_scores': category_scores,
            'last_updated': datetime.now().isoformat(),
            'factor_count': len(self.risk_factors),
            'confidence': np.mean([f.confidence for f in self.risk_factors.values()])
        }
    
    def evaluate_scenarios(self) -> Dict[str, Any]:
        """Evaluate all risk scenarios and their current likelihood"""
        scenario_evaluations = {}
        
        for scenario_id, scenario in self.risk_scenarios.items():
            # Check if scenario triggers are met
            triggers_met = []
            for factor_name, trigger in scenario.triggers.items():
                if factor_name in self.risk_factors:
                    factor = self.risk_factors[factor_name]
                    operator = trigger['operator']
                    threshold = trigger['value']
                    
                    if operator == '>=' and factor.current_value >= threshold:
                        triggers_met.append(True)
                    elif operator == '>' and factor.current_value > threshold:
                        triggers_met.append(True)
                    elif operator == '<=' and factor.current_value <= threshold:
                        triggers_met.append(True)
                    elif operator == '<' and factor.current_value < threshold:
                        triggers_met.append(True)
                    elif operator == '==' and abs(factor.current_value - threshold) < 1:
                        triggers_met.append(True)
                    else:
                        triggers_met.append(False)
                else:
                    triggers_met.append(False)
            
            # Calculate scenario probability based on triggers
            trigger_probability = sum(triggers_met) / len(scenario.triggers) if scenario.triggers else 0
            adjusted_probability = scenario.probability * trigger_probability
            
            # Calculate expected impact
            expected_impact = adjusted_probability * scenario.impact_score
            
            scenario_evaluations[scenario_id] = {
                'scenario_name': scenario.name,
                'base_probability': scenario.probability,
                'current_probability': adjusted_probability,
                'impact_score': scenario.impact_score,
                'expected_impact': expected_impact,
                'triggers_met': sum(triggers_met),
                'total_triggers': len(scenario.triggers),
                'trigger_status': dict(zip(scenario.triggers.keys(), triggers_met)),
                'estimated_cost': scenario.estimated_cost,
                'response_time_hours': scenario.response_time_required.total_seconds() / 3600
            }
        
        # Sort by expected impact
        sorted_scenarios = dict(sorted(
            scenario_evaluations.items(), 
            key=lambda x: x[1]['expected_impact'], 
            reverse=True
        ))
        
        return {
            'scenarios': sorted_scenarios,
            'highest_risk_scenario': list(sorted_scenarios.keys())[0] if sorted_scenarios else None,
            'total_expected_impact': sum(s['expected_impact'] for s in sorted_scenarios.values()),
            'evaluation_time': datetime.now().isoformat()
        }
    
    def generate_decision_recommendation(self, context: Dict[str, Any]) -> DecisionRecommendation:
        """Generate intelligent decision recommendation based on current risk state"""
        
        composite_risk = self.calculate_composite_risk_score()
        scenario_eval = self.evaluate_scenarios()
        
        # Determine recommended action based on risk level
        overall_score = composite_risk['overall_score']
        overall_level = composite_risk['overall_level']
        
        if overall_score >= 80:
            recommended_action = "Activate Crisis Management Protocol"
            expected_outcome = "Minimize damage and ensure operational continuity"
            success_probability = 0.75
            implementation_time = timedelta(minutes=30)
            fallback_options = ["Emergency shutdown", "Switch to backup systems"]
        elif overall_score >= 60:
            recommended_action = "Implement Enhanced Monitoring and Preparedness"
            expected_outcome = "Early detection and prevention of critical incidents"
            success_probability = 0.85
            implementation_time = timedelta(hours=1)
            fallback_options = ["Escalate to crisis protocol", "Reduce operational tempo"]
        elif overall_score >= 40:
            recommended_action = "Increase Situational Awareness"
            expected_outcome = "Maintain operational readiness and risk visibility"
            success_probability = 0.90
            implementation_time = timedelta(hours=2)
            fallback_options = ["Enhanced monitoring", "Prepare contingency plans"]
        else:
            recommended_action = "Continue Normal Operations"
            expected_outcome = "Maintain current operational status"
            success_probability = 0.95
            implementation_time = timedelta(minutes=0)
            fallback_options = ["Increase monitoring if conditions change"]
        
        # Cost-benefit analysis
        current_operational_cost = 10000  # Daily operational cost
        risk_mitigation_cost = overall_score * 100  # Cost scales with risk
        potential_loss = overall_score * 50000  # Potential loss scales with risk
        
        cost_benefit = {
            'mitigation_cost': risk_mitigation_cost,
            'potential_loss_prevented': potential_loss,
            'net_benefit': potential_loss - risk_mitigation_cost,
            'roi': (potential_loss - risk_mitigation_cost) / risk_mitigation_cost if risk_mitigation_cost > 0 else 0
        }
        
        recommendation = DecisionRecommendation(
            recommendation_id=f"REC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            risk_context={
                'composite_risk': composite_risk,
                'scenario_evaluation': scenario_eval,
                'context': context
            },
            recommended_action=recommended_action,
            confidence_level=composite_risk['confidence'],
            expected_outcome=expected_outcome,
            cost_benefit_analysis=cost_benefit,
            implementation_time=implementation_time,
            success_probability=success_probability,
            fallback_options=fallback_options
        )
        
        self.decision_history.append(recommendation)
        logger.info(f"Generated decision recommendation: {recommended_action}")
        
        return recommendation
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """Get all active alerts from the queue"""
        alerts = []
        temp_queue = PriorityQueue()
        
        # Extract all alerts
        while not self.alert_queue.empty():
            priority, alert = self.alert_queue.get()
            if not alert.resolved:
                alerts.append(alert)
            temp_queue.put((priority, alert))
        
        # Put alerts back in queue
        while not temp_queue.empty():
            self.alert_queue.put(temp_queue.get())
        
        return sorted(alerts, key=lambda x: x.priority.value)
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Resolve an active alert"""
        temp_queue = PriorityQueue()
        resolved = False
        
        # Find and resolve the alert
        while not self.alert_queue.empty():
            priority, alert = self.alert_queue.get()
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                resolved = True
                logger.info(f"Resolved alert {alert_id}: {resolution_notes}")
            temp_queue.put((priority, alert))
        
        # Put alerts back in queue
        while not temp_queue.empty():
            self.alert_queue.put(temp_queue.get())
        
        if not resolved:
            logger.warning(f"Alert {alert_id} not found for resolution")
    
    def start_real_time_monitoring(self):
        """Start real-time risk monitoring"""
        self.risk_monitor.start_monitoring()
    
    def stop_real_time_monitoring(self):
        """Stop real-time risk monitoring"""
        self.risk_monitor.stop_monitoring()
    
    def export_risk_dashboard_data(self) -> Dict[str, Any]:
        """Export comprehensive risk data for dashboard display"""
        
        composite_risk = self.calculate_composite_risk_score()
        scenario_eval = self.evaluate_scenarios()
        active_alerts = self.get_active_alerts()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_risk': composite_risk,
            'risk_factors': {
                name: {
                    'current_value': factor.current_value,
                    'risk_level': self._get_risk_level(factor.current_value, factor.threshold_values).value,
                    'trend': factor.trend,
                    'confidence': factor.confidence,
                    'category': factor.category.value,
                    'weight': factor.weight,
                    'last_updated': factor.last_updated.isoformat()
                }
                for name, factor in self.risk_factors.items()
            },
            'scenarios': scenario_eval,
            'active_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'timestamp': alert.timestamp.isoformat(),
                    'risk_level': alert.risk_level.value,
                    'category': alert.category.value,
                    'priority': alert.priority.value,
                    'title': alert.title,
                    'description': alert.description,
                    'affected_systems': alert.affected_systems,
                    'recommended_actions': alert.recommended_actions
                }
                for alert in active_alerts
            ],
            'recent_decisions': [
                {
                    'recommendation_id': rec.recommendation_id,
                    'timestamp': rec.timestamp.isoformat(),
                    'recommended_action': rec.recommended_action,
                    'confidence_level': rec.confidence_level,
                    'expected_outcome': rec.expected_outcome,
                    'success_probability': rec.success_probability
                }
                for rec in self.decision_history[-5:]  # Last 5 decisions
            ]
        }

def main():
    """Test the advanced risk engine"""
    risk_engine = AdvancedRiskEngine()
    
    print("=== Advanced Risk Engine Test ===")
    
    # Start monitoring
    risk_engine.start_real_time_monitoring()
    
    # Simulate some risk factor updates
    print("\nUpdating risk factors...")
    risk_engine.update_risk_factor("kp_index", 75.0, confidence=0.9, source="NOAA_SWPC")
    risk_engine.update_risk_factor("solar_flare_activity", 85.0, confidence=0.95, source="JSOC_SDO")
    risk_engine.update_risk_factor("collision_probability", 45.0, confidence=0.8, source="Space_Track")
    
    # Calculate composite risk
    composite_risk = risk_engine.calculate_composite_risk_score()
    print(f"\nComposite Risk Score: {composite_risk['overall_score']:.1f}/100")
    print(f"Risk Level: {composite_risk['overall_level'].upper()}")
    
    # Evaluate scenarios
    scenarios = risk_engine.evaluate_scenarios()
    print(f"\nTop Risk Scenario: {scenarios['highest_risk_scenario']}")
    
    # Generate decision recommendation
    context = {"mission_phase": "operational", "crew_aboard": False}
    recommendation = risk_engine.generate_decision_recommendation(context)
    print(f"\nRecommended Action: {recommendation.recommended_action}")
    print(f"Confidence: {recommendation.confidence_level:.2f}")
    print(f"Success Probability: {recommendation.success_probability:.2f}")
    
    # Check alerts
    time.sleep(2)  # Allow monitoring to generate some alerts
    alerts = risk_engine.get_active_alerts()
    print(f"\nActive Alerts: {len(alerts)}")
    for alert in alerts[:3]:
        print(f"  {alert.priority.name}: {alert.title}")
    
    # Export dashboard data
    dashboard_data = risk_engine.export_risk_dashboard_data()
    print(f"\nDashboard Export: {len(dashboard_data)} sections")
    
    # Stop monitoring
    risk_engine.stop_real_time_monitoring()
    
    print("\nRisk engine test completed successfully!")

if __name__ == "__main__":
    main()