"""
Advanced Computer Vision Pipeline for Space Risk Intelligence
Detects: Solar flares, CMEs, sunspots, orbital debris with confidence scoring
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import json
import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpacePhenomena(Enum):
    """Space phenomena classes"""
    SOLAR_FLARE = "solar_flare"
    CME = "coronal_mass_ejection"
    SUNSPOT = "sunspot"
    DEBRIS = "orbital_debris"
    QUIET_SUN = "quiet_sun"
    AURORA = "aurora"
    SATELLITE = "satellite"

@dataclass
class Detection:
    """Detection result structure"""
    phenomenon: SpacePhenomena
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    severity: str  # low, medium, high, extreme
    timestamp: datetime.datetime
    image_path: str
    additional_data: Dict

class AdvancedSpaceCV:
    """
    Advanced Computer Vision system for space risk intelligence
    Combines multiple YOLO models + ensemble methods + confidence analysis
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.confidence_thresholds = {
            SpacePhenomena.SOLAR_FLARE: 0.7,
            SpacePhenomena.CME: 0.65,
            SpacePhenomena.SUNSPOT: 0.8,
            SpacePhenomena.DEBRIS: 0.75,
            SpacePhenomena.QUIET_SUN: 0.9,
            SpacePhenomena.AURORA: 0.6,
            SpacePhenomena.SATELLITE: 0.8
        }
        self.load_models()
        
    def load_models(self):
        """Load multiple specialized YOLO models"""
        try:
            # Primary detection model
            primary_model_path = self.models_dir / "enhanced_distinct_yolo" / "best.pt"
            if primary_model_path.exists():
                self.models['primary'] = YOLO(str(primary_model_path))
                logger.info(f"Loaded primary model: {primary_model_path}")
            
            # Backup models
            for model_name in ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"]:
                model_path = self.models_dir / model_name
                if model_path.exists():
                    self.models[model_name] = YOLO(str(model_path))
                    logger.info(f"Loaded backup model: {model_name}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to default YOLO
            self.models['fallback'] = YOLO('yolov8n.pt')
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Advanced image preprocessing for space imagery"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Space-specific preprocessing
        # 1. Contrast enhancement for low-light space images
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # 2. Noise reduction
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 3. Edge enhancement for better detection
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def ensemble_detection(self, image_path: str) -> List[Detection]:
        """Ensemble detection using multiple models"""
        detections = []
        processed_image = self.preprocess_image(image_path)
        
        # Run detection with all available models
        all_results = []
        for model_name, model in self.models.items():
            try:
                results = model(processed_image, conf=0.1)  # Low threshold for ensemble
                all_results.append((model_name, results))
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
        
        # Combine and filter results
        combined_detections = self._combine_detections(all_results, image_path)
        
        return combined_detections
    
    def _combine_detections(self, all_results: List, image_path: str) -> List[Detection]:
        """Combine detections from multiple models using ensemble voting"""
        combined = []
        
        for model_name, results in all_results:
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                for i in range(len(boxes.xyxy)):
                    # Extract detection data
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Map class ID to phenomenon
                    phenomenon = self._map_class_to_phenomenon(cls_id, model_name)
                    
                    # Calculate severity based on confidence and phenomenon type
                    severity = self._calculate_severity(phenomenon, conf, bbox)
                    
                    # Create detection object
                    detection = Detection(
                        phenomenon=phenomenon,
                        confidence=conf,
                        bbox=tuple(bbox),
                        severity=severity,
                        timestamp=datetime.datetime.now(),
                        image_path=image_path,
                        additional_data={
                            'model': model_name,
                            'class_id': cls_id,
                            'bbox_area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        }
                    )
                    
                    combined.append(detection)
        
        # Filter by confidence thresholds
        filtered = [d for d in combined if d.confidence >= self.confidence_thresholds.get(d.phenomenon, 0.5)]
        
        # Remove duplicate detections using NMS-like approach
        final_detections = self._non_max_suppression(filtered)
        
        return final_detections
    
    def _map_class_to_phenomenon(self, cls_id: int, model_name: str) -> SpacePhenomena:
        """Map class ID to space phenomenon"""
        # This mapping depends on your trained model classes
        class_mapping = {
            0: SpacePhenomena.QUIET_SUN,
            1: SpacePhenomena.SOLAR_FLARE,
            2: SpacePhenomena.SUNSPOT,
            3: SpacePhenomena.CME,
            4: SpacePhenomena.DEBRIS,
            5: SpacePhenomena.AURORA,
            6: SpacePhenomena.SATELLITE
        }
        
        return class_mapping.get(cls_id, SpacePhenomena.QUIET_SUN)
    
    def _calculate_severity(self, phenomenon: SpacePhenomena, confidence: float, bbox: np.ndarray) -> str:
        """Calculate severity based on phenomenon type, confidence, and size"""
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        if phenomenon == SpacePhenomena.SOLAR_FLARE:
            if confidence > 0.9 and bbox_area > 50000:
                return "extreme"
            elif confidence > 0.8 and bbox_area > 20000:
                return "high"
            elif confidence > 0.7:
                return "medium"
            else:
                return "low"
        
        elif phenomenon == SpacePhenomena.CME:
            if confidence > 0.85 and bbox_area > 80000:
                return "extreme"
            elif confidence > 0.75 and bbox_area > 40000:
                return "high"
            elif confidence > 0.65:
                return "medium"
            else:
                return "low"
        
        elif phenomenon == SpacePhenomena.DEBRIS:
            if confidence > 0.9:
                return "high"
            elif confidence > 0.8:
                return "medium"
            else:
                return "low"
        
        else:
            # Default severity calculation
            if confidence > 0.9:
                return "high"
            elif confidence > 0.7:
                return "medium"
            else:
                return "low"
    
    def _non_max_suppression(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        final_detections = []
        
        while detections:
            best = detections.pop(0)
            final_detections.append(best)
            
            # Remove overlapping detections
            detections = [d for d in detections if self._calculate_iou(best.bbox, d.bbox) < iou_threshold]
        
        return final_detections
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_image(self, image_path: str) -> Dict:
        """Complete image analysis with risk assessment"""
        try:
            detections = self.ensemble_detection(image_path)
            
            # Risk analysis
            risk_score = self._calculate_risk_score(detections)
            
            # Generate report
            report = {
                'timestamp': datetime.datetime.now().isoformat(),
                'image_path': image_path,
                'total_detections': len(detections),
                'risk_score': risk_score,
                'detections': [
                    {
                        'phenomenon': d.phenomenon.value,
                        'confidence': round(d.confidence, 3),
                        'severity': d.severity,
                        'bbox': d.bbox,
                        'additional_data': d.additional_data
                    }
                    for d in detections
                ],
                'summary': self._generate_summary(detections)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat(),
                'image_path': image_path
            }
    
    def _calculate_risk_score(self, detections: List[Detection]) -> float:
        """Calculate overall risk score (0-100)"""
        if not detections:
            return 0.0
        
        risk_weights = {
            SpacePhenomena.SOLAR_FLARE: 25,
            SpacePhenomena.CME: 30,
            SpacePhenomena.DEBRIS: 20,
            SpacePhenomena.SUNSPOT: 10,
            SpacePhenomena.AURORA: 5,
            SpacePhenomena.SATELLITE: 15,
            SpacePhenomena.QUIET_SUN: 0
        }
        
        severity_multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5,
            'extreme': 2.0
        }
        
        total_risk = 0
        for detection in detections:
            base_risk = risk_weights.get(detection.phenomenon, 5)
            severity_mult = severity_multipliers.get(detection.severity, 1.0)
            confidence_mult = detection.confidence
            
            detection_risk = base_risk * severity_mult * confidence_mult
            total_risk += detection_risk
        
        # Normalize to 0-100 scale
        risk_score = min(100, total_risk)
        
        return round(risk_score, 2)
    
    def _generate_summary(self, detections: List[Detection]) -> Dict:
        """Generate human-readable summary"""
        if not detections:
            return {'status': 'All clear - no significant space weather detected'}
        
        phenomena_counts = {}
        high_risk_events = []
        
        for detection in detections:
            phenomenon = detection.phenomenon.value
            phenomena_counts[phenomenon] = phenomena_counts.get(phenomenon, 0) + 1
            
            if detection.severity in ['high', 'extreme']:
                high_risk_events.append({
                    'type': phenomenon,
                    'severity': detection.severity,
                    'confidence': detection.confidence
                })
        
        summary = {
            'phenomena_detected': phenomena_counts,
            'high_risk_events': len(high_risk_events),
            'critical_alerts': high_risk_events[:3],  # Top 3 most critical
            'recommendation': self._get_recommendation(detections)
        }
        
        return summary
    
    def _get_recommendation(self, detections: List[Detection]) -> str:
        """Generate operational recommendations"""
        high_risk = [d for d in detections if d.severity in ['high', 'extreme']]
        
        if not high_risk:
            return "Normal operations - continue monitoring"
        
        solar_events = [d for d in high_risk if d.phenomenon in [SpacePhenomena.SOLAR_FLARE, SpacePhenomena.CME]]
        debris_events = [d for d in high_risk if d.phenomenon == SpacePhenomena.DEBRIS]
        
        if solar_events and debris_events:
            return "CRITICAL: Multiple high-risk events detected. Implement emergency protocols for both space weather and collision avoidance."
        elif solar_events:
            return "WARNING: Solar activity detected. Consider satellite safe mode and monitor for radiation effects."
        elif debris_events:
            return "ALERT: Orbital debris detected. Activate collision avoidance procedures if in proximity."
        else:
            return "CAUTION: Elevated space weather activity. Increase monitoring frequency."

def main():
    """Test the advanced CV system"""
    cv_system = AdvancedSpaceCV()
    
    # Test with sample images
    test_images = [
        "data/enhanced_distinct_dataset/images/quiet_sun_001.jpg",
        "data/enhanced_distinct_dataset/images/solar_storm_001.jpg"
    ]
    
    for image_path in test_images:
        if Path(image_path).exists():
            print(f"\n=== Analyzing {image_path} ===")
            result = cv_system.analyze_image(image_path)
            print(json.dumps(result, indent=2))
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    main()