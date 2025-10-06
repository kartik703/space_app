#!/usr/bin/env python3
"""
Endpoint Validation Script for Solar CV Models
Tests deployed Vertex AI endpoints to ensure they're working correctly
"""

import os
import sys
import json
import base64
import logging
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from google.cloud import aiplatform
from google.auth import default
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EndpointValidator:
    """Validates deployed Vertex AI endpoints"""
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        
        # Initialize AI Platform
        aiplatform.init(project=project_id, location=region)
        
        # Get default credentials
        self.credentials, _ = default()
        
    def find_endpoints(self, model_type: str) -> list:
        """Find endpoints for a specific model type"""
        endpoints = aiplatform.Endpoint.list()
        
        filtered_endpoints = []
        for endpoint in endpoints:
            if model_type.lower() in endpoint.display_name.lower():
                filtered_endpoints.append(endpoint)
                
        return filtered_endpoints
    
    def create_test_image(self, size: tuple = (640, 640)) -> Image.Image:
        """Create a test solar image for validation"""
        # Create a synthetic solar disk image
        width, height = size
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        
        # Create base image
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add solar disk
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        
        # Solar disk (yellow-orange gradient)
        img_array[mask] = [255, 200, 50]  # Orange
        
        # Add some noise for realism
        noise = np.random.randint(-20, 20, (height, width, 3))
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add some "solar features"
        for _ in range(5):
            spot_x = np.random.randint(center_x - radius//2, center_x + radius//2)
            spot_y = np.random.randint(center_y - radius//2, center_y + radius//2)
            spot_radius = np.random.randint(5, 15)
            
            y_spot, x_spot = np.ogrid[:height, :width]
            spot_mask = (x_spot - spot_x) ** 2 + (y_spot - spot_y) ** 2 <= spot_radius ** 2
            img_array[spot_mask] = [200, 100, 0]  # Darker spots
        
        return Image.fromarray(img_array)
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def test_yolo_endpoint(self, endpoint) -> dict:
        """Test YOLO endpoint with a test image"""
        logger.info(f"Testing YOLO endpoint: {endpoint.display_name}")
        
        try:
            # Create test image
            test_image = self.create_test_image((640, 640))
            
            # Prepare payload
            image_b64 = self.image_to_base64(test_image)
            
            # YOLO expects different input format
            instances = [
                {
                    "image": image_b64,
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45
                }
            ]
            
            # Make prediction
            start_time = time.time()
            response = endpoint.predict(instances=instances)
            prediction_time = time.time() - start_time
            
            # Parse response
            predictions = response.predictions[0] if response.predictions else {}
            
            result = {
                "endpoint_name": endpoint.display_name,
                "endpoint_id": endpoint.name,
                "status": "success",
                "prediction_time": round(prediction_time, 3),
                "predictions": predictions,
                "test_image_size": test_image.size
            }
            
            logger.info(f"YOLO endpoint test successful: {prediction_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"YOLO endpoint test failed: {str(e)}")
            return {
                "endpoint_name": endpoint.display_name,
                "endpoint_id": endpoint.name,
                "status": "error",
                "error": str(e)
            }
    
    def test_vit_endpoint(self, endpoint) -> dict:
        """Test ViT endpoint with a test image"""
        logger.info(f"Testing ViT endpoint: {endpoint.display_name}")
        
        try:
            # Create test image
            test_image = self.create_test_image((224, 224))  # ViT typically uses 224x224
            
            # Prepare payload
            image_b64 = self.image_to_base64(test_image)
            
            # ViT expects classification format
            instances = [
                {
                    "image": image_b64
                }
            ]
            
            # Make prediction
            start_time = time.time()
            response = endpoint.predict(instances=instances)
            prediction_time = time.time() - start_time
            
            # Parse response
            predictions = response.predictions[0] if response.predictions else {}
            
            result = {
                "endpoint_name": endpoint.display_name,
                "endpoint_id": endpoint.name,
                "status": "success",
                "prediction_time": round(prediction_time, 3),
                "predictions": predictions,
                "test_image_size": test_image.size
            }
            
            logger.info(f"ViT endpoint test successful: {prediction_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"ViT endpoint test failed: {str(e)}")
            return {
                "endpoint_name": endpoint.display_name,
                "endpoint_id": endpoint.name,
                "status": "error",
                "error": str(e)
            }
    
    def validate_all_endpoints(self) -> dict:
        """Validate all solar CV endpoints"""
        logger.info("Starting endpoint validation...")
        
        results = {
            "validation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_id": self.project_id,
            "region": self.region,
            "yolo_results": [],
            "vit_results": [],
            "summary": {}
        }
        
        # Test YOLO endpoints
        yolo_endpoints = self.find_endpoints("yolo")
        logger.info(f"Found {len(yolo_endpoints)} YOLO endpoints")
        
        for endpoint in yolo_endpoints:
            result = self.test_yolo_endpoint(endpoint)
            results["yolo_results"].append(result)
        
        # Test ViT endpoints
        vit_endpoints = self.find_endpoints("vit")
        logger.info(f"Found {len(vit_endpoints)} ViT endpoints")
        
        for endpoint in vit_endpoints:
            result = self.test_vit_endpoint(endpoint)
            results["vit_results"].append(result)
        
        # Generate summary
        total_endpoints = len(yolo_endpoints) + len(vit_endpoints)
        successful_tests = sum(1 for r in results["yolo_results"] + results["vit_results"] 
                              if r["status"] == "success")
        
        results["summary"] = {
            "total_endpoints": total_endpoints,
            "successful_tests": successful_tests,
            "failed_tests": total_endpoints - successful_tests,
            "success_rate": round(successful_tests / total_endpoints * 100, 2) if total_endpoints > 0 else 0
        }
        
        logger.info(f"Validation complete: {successful_tests}/{total_endpoints} endpoints successful")
        return results
    
    def save_results(self, results: dict, output_file: str = None):
        """Save validation results to file"""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"endpoint_validation_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")

def main():
    """Main validation function"""
    # Get project ID from environment or argument
    project_id = os.getenv('PROJECT_ID')
    if not project_id and len(sys.argv) > 1:
        project_id = sys.argv[1]
    
    if not project_id:
        logger.error("Project ID not provided. Set PROJECT_ID env var or pass as argument.")
        sys.exit(1)
    
    # Create validator
    validator = EndpointValidator(project_id)
    
    try:
        # Run validation
        results = validator.validate_all_endpoints()
        
        # Save results
        validator.save_results(results)
        
        # Print summary
        summary = results["summary"]
        print("\n" + "="*50)
        print("ENDPOINT VALIDATION SUMMARY")
        print("="*50)
        print(f"Total Endpoints: {summary['total_endpoints']}")
        print(f"Successful Tests: {summary['successful_tests']}")
        print(f"Failed Tests: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']}%")
        print("="*50)
        
        # Exit with appropriate code
        if summary['failed_tests'] > 0:
            print("Some endpoints failed validation!")
            sys.exit(1)
        else:
            print("All endpoints validated successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()