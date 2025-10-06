#!/usr/bin/env python3
"""
Model Deployment Script for Solar CV Models on Vertex AI
Deploy YOLO and ViT models as Vertex AI endpoints for real-time inference
"""

import os
import logging
from pathlib import Path
from google.cloud import aiplatform
from google.cloud import storage
import argparse
import json
from datetime import datetime
import tempfile
import zipfile
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VertexAIModelDeployer:
    """Deploy Solar CV models to Vertex AI endpoints"""
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Initialize Storage client
        self.storage_client = storage.Client(project=project_id)
        
        logger.info(f"Initialized model deployer for project: {project_id}, region: {region}")
    
    def create_model_serving_container(self, model_type: str = "yolo") -> str:
        """Create model serving container for YOLO or ViT"""
        
        if model_type == "yolo":
            dockerfile_content = """
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    ultralytics==8.2.37 \\
    torch==2.2.0 \\
    torchvision==0.17.0 \\
    fastapi==0.115.0 \\
    uvicorn==0.30.6 \\
    google-cloud-storage \\
    pillow \\
    numpy

# Copy application code
COPY yolo_predictor.py /app/
COPY model/ /app/model/
WORKDIR /app

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "yolo_predictor:app", "--host", "0.0.0.0", "--port", "8080"]
"""
            
            predictor_code = """
import os
import io
import base64
import json
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np

app = FastAPI(title="Solar YOLO Predictor")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = "/app/model/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Loaded YOLO model from {model_path}")
    else:
        print("Model file not found, using default YOLOv8n")
        model = YOLO("yolov8n.pt")

class PredictionRequest(BaseModel):
    image: Dict[str, str]  # {"b64": "base64_encoded_image"}
    confidence_threshold: float = 0.5

class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x, y, w, h]

class PredictionResponse(BaseModel):
    detections: List[Detection]
    image_shape: List[int]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Decode base64 image
        image_b64 = request.image.get("b64", "")
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run inference
        results = model(image, conf=request.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # Convert to YOLO format (x, y, w, h)
                    x = float(x1)
                    y = float(y1) 
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    
                    class_names = {0: "flare", 1: "cme", 2: "sunspot", 3: "debris"}
                    class_name = class_names.get(cls_id, f"class_{cls_id}")
                    
                    detections.append(Detection(
                        class_name=class_name,
                        class_id=cls_id,
                        confidence=conf,
                        bbox=[x, y, w, h]
                    ))
        
        return PredictionResponse(
            detections=detections,
            image_shape=list(image.size)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
"""
            
        elif model_type == "vit":
            dockerfile_content = """
FROM python:3.9-slim

# Install Python dependencies
RUN pip install --no-cache-dir \\
    torch==2.2.0 \\
    torchvision==0.17.0 \\
    timm==1.0.9 \\
    fastapi==0.115.0 \\
    uvicorn==0.30.6 \\
    google-cloud-storage \\
    pillow \\
    numpy

# Copy application code
COPY vit_predictor.py /app/
COPY model/ /app/model/
WORKDIR /app

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "vit_predictor:app", "--host", "0.0.0.0", "--port", "8080"]
"""
            
            predictor_code = """
import os
import io
import base64
import json
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import torch.nn.functional as F

app = FastAPI(title="Solar ViT Predictor")

# Load model at startup
model = None
transform = None

@app.on_event("startup")
async def load_model():
    global model, transform
    
    # Load model
    model_path = "/app/model/solar_vit_best.pth"
    if os.path.exists(model_path):
        # Load saved model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded ViT model from {model_path}")
    else:
        print("Model file not found, using default ViT")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
    
    model.eval()
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class PredictionRequest(BaseModel):
    image: Dict[str, str]  # {"b64": "base64_encoded_image"}

class PredictionResponse(BaseModel):
    predicted_class: str
    storm_probability: float
    quiet_probability: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Decode base64 image
        image_b64 = request.image.get("b64", "")
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            quiet_prob = float(probabilities[0][0])
            storm_prob = float(probabilities[0][1])
            
            predicted_class = "storm" if storm_prob > quiet_prob else "quiet"
            confidence = max(storm_prob, quiet_prob)
        
        return PredictionResponse(
            predicted_class=predicted_class,
            storm_probability=storm_prob,
            quiet_probability=quiet_prob,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
"""
        
        # Create build directory
        build_dir = f"vertex_deploy_{model_type}"
        os.makedirs(build_dir, exist_ok=True)
        
        # Write Dockerfile
        with open(f"{build_dir}/Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        # Write predictor code
        predictor_filename = f"{model_type}_predictor.py"
        with open(f"{build_dir}/{predictor_filename}", "w") as f:
            f.write(predictor_code)
        
        # Create cloudbuild config
        image_uri = f"gcr.io/{self.project_id}/solar-{model_type}-predictor:latest"
        cloudbuild_config = {
            "steps": [
                {
                    "name": "gcr.io/cloud-builders/docker",
                    "args": ["build", "-t", image_uri, "."]
                },
                {
                    "name": "gcr.io/cloud-builders/docker",
                    "args": ["push", image_uri]
                }
            ],
            "images": [image_uri]
        }
        
        with open(f"{build_dir}/cloudbuild.yaml", "w") as f:
            json.dump(cloudbuild_config, f, indent=2)
        
        logger.info(f"Created {model_type} serving container configuration in {build_dir}/")
        logger.info(f"Build with: cd {build_dir} && gcloud builds submit . --config cloudbuild.yaml")
        
        return image_uri
    
    def upload_model_to_gcs(self, local_model_path: str, gcs_model_path: str):
        """Upload model artifacts to GCS"""
        logger.info(f"Uploading model from {local_model_path} to {gcs_model_path}")
        
        # Parse GCS path
        if not gcs_model_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_model_path}")
        
        path_parts = gcs_model_path[5:].split('/', 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(local_model_path)
        logger.info(f"Model uploaded to {gcs_model_path}")
    
    def register_model(self, display_name: str, artifact_uri: str, 
                      serving_container_image_uri: str, description: str = "") -> str:
        """Register model in Vertex AI Model Registry"""
        
        logger.info(f"Registering model: {display_name}")
        
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            serving_container_ports=[8080],
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            description=description
        )
        
        logger.info(f"Model registered: {model.resource_name}")
        return model.resource_name
    
    def deploy_endpoint(self, model_resource_name: str, endpoint_display_name: str,
                       machine_type: str = "n1-standard-2", min_replica_count: int = 1,
                       max_replica_count: int = 3) -> str:
        """Deploy model to Vertex AI endpoint"""
        
        logger.info(f"Deploying endpoint: {endpoint_display_name}")
        
        # Get the model
        model = aiplatform.Model(model_resource_name)
        
        # Create endpoint
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
        
        # Deploy model to endpoint
        endpoint.deploy(
            model=model,
            deployed_model_display_name=f"{endpoint_display_name}_deployed",
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=100,
            accelerator_type=None,  # Use GPU if needed
            accelerator_count=0
        )
        
        logger.info(f"Model deployed to endpoint: {endpoint.resource_name}")
        return endpoint.resource_name
    
    def test_endpoint(self, endpoint_resource_name: str, test_image_path: str):
        """Test deployed endpoint with sample image"""
        logger.info(f"Testing endpoint: {endpoint_resource_name}")
        
        # Load test image
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()
        
        image_b64 = base64.b64encode(image_bytes).decode()
        
        # Create prediction request
        instances = [{"image": {"b64": image_b64}}]
        
        # Get endpoint
        endpoint = aiplatform.Endpoint(endpoint_resource_name)
        
        # Make prediction
        response = endpoint.predict(instances=instances)
        
        logger.info(f"Prediction response: {response}")
        return response
    
    def list_models(self):
        """List registered models"""
        models = aiplatform.Model.list()
        
        logger.info(f"Found {len(models)} registered models:")
        for model in models:
            logger.info(f"  {model.display_name}: {model.resource_name}")
        
        return models
    
    def list_endpoints(self):
        """List deployed endpoints"""
        endpoints = aiplatform.Endpoint.list()
        
        logger.info(f"Found {len(endpoints)} endpoints:")
        for endpoint in endpoints:
            logger.info(f"  {endpoint.display_name}: {endpoint.resource_name}")
        
        return endpoints


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Solar CV models to Vertex AI")
    parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", help="Vertex AI region")
    parser.add_argument("--model-type", choices=["yolo", "vit"], required=True, help="Model type to deploy")
    parser.add_argument("--model-path", help="Local path to model file")
    parser.add_argument("--gcs-model-path", help="GCS path for model artifacts")
    parser.add_argument("--container-uri", help="Container URI for serving")
    parser.add_argument("--endpoint-name", help="Endpoint display name")
    parser.add_argument("--machine-type", default="n1-standard-2", help="Machine type for serving")
    parser.add_argument("--test-image", help="Test image path")
    parser.add_argument("--list-models", action="store_true", help="List registered models")
    parser.add_argument("--list-endpoints", action="store_true", help="List endpoints")
    parser.add_argument("--create-container", action="store_true", help="Create serving container")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = VertexAIModelDeployer(args.project_id, args.region)
    
    # List models/endpoints if requested
    if args.list_models:
        deployer.list_models()
        return 0
    
    if args.list_endpoints:
        deployer.list_endpoints()
        return 0
    
    # Create container if requested
    if args.create_container:
        container_uri = deployer.create_model_serving_container(args.model_type)
        logger.info(f"Serving container configuration created for {args.model_type}")
        return 0
    
    # Full deployment pipeline
    try:
        # Set default paths
        if not args.gcs_model_path:
            args.gcs_model_path = f"gs://solar-models/{args.model_type}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not args.endpoint_name:
            args.endpoint_name = f"solar-{args.model_type}-endpoint"
        
        if not args.container_uri:
            args.container_uri = f"gcr.io/{args.project_id}/solar-{args.model_type}-predictor:latest"
        
        # Upload model if provided
        if args.model_path:
            deployer.upload_model_to_gcs(args.model_path, args.gcs_model_path + "/model")
        
        # Register model
        model_name = f"solar-{args.model_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_resource = deployer.register_model(
            display_name=model_name,
            artifact_uri=args.gcs_model_path,
            serving_container_image_uri=args.container_uri,
            description=f"Solar {args.model_type.upper()} model for event detection/classification"
        )
        
        # Deploy endpoint
        endpoint_resource = deployer.deploy_endpoint(
            model_resource_name=model_resource,
            endpoint_display_name=args.endpoint_name,
            machine_type=args.machine_type
        )
        
        # Test endpoint if test image provided
        if args.test_image:
            deployer.test_endpoint(endpoint_resource, args.test_image)
        
        logger.info(f"Deployment completed successfully!")
        logger.info(f"Model: {model_resource}")
        logger.info(f"Endpoint: {endpoint_resource}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())