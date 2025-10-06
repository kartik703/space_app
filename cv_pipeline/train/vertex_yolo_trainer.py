#!/usr/bin/env python3
"""
Vertex AI Training Script for YOLOv8 Solar Event Detection
Submit YOLO training jobs to Google Cloud Vertex AI with GPU support
"""

import os
import logging
from pathlib import Path
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip
import argparse
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VertexAIYOLOTrainer:
    """Submit YOLO training to Vertex AI"""
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Training configuration
        self.machine_types = {
            "cpu": "n1-standard-4",
            "gpu_k80": "n1-standard-8",
            "gpu_v100": "n1-standard-8", 
            "gpu_t4": "n1-standard-4",
            "gpu_a100": "a2-highgpu-1g"
        }
        
        self.accelerator_types = {
            "gpu_k80": "NVIDIA_TESLA_K80",
            "gpu_v100": "NVIDIA_TESLA_V100",
            "gpu_t4": "NVIDIA_TESLA_T4",
            "gpu_a100": "NVIDIA_TESLA_A100"
        }
        
        logger.info(f"Initialized Vertex AI trainer for project: {project_id}, region: {region}")
    
    def create_training_image(self, base_image: str = "gcr.io/deeplearning-platform-release/pytorch-gpu.1-13") -> str:
        """Create custom training container image"""
        
        dockerfile_content = f"""
FROM {base_image}

# Install ultralytics and dependencies
RUN pip install ultralytics==8.2.37 google-cloud-storage google-cloud-bigquery

# Copy training code
COPY cv_pipeline/ /app/cv_pipeline/
COPY requirements.txt /app/
WORKDIR /app

# Install additional requirements
RUN pip install -r requirements.txt

# Set entrypoint
ENTRYPOINT ["python", "cv_pipeline/train/vertex_yolo_trainer.py"]
"""
        
        # Create Cloud Build configuration
        cloudbuild_config = {
            "steps": [
                {
                    "name": "gcr.io/cloud-builders/docker",
                    "args": [
                        "build",
                        "-t", f"gcr.io/{self.project_id}/solar-yolo-trainer:latest",
                        "."
                    ]
                },
                {
                    "name": "gcr.io/cloud-builders/docker", 
                    "args": [
                        "push",
                        f"gcr.io/{self.project_id}/solar-yolo-trainer:latest"
                    ]
                }
            ],
            "images": [f"gcr.io/{self.project_id}/solar-yolo-trainer:latest"]
        }
        
        # Save files for building
        os.makedirs("vertex_build", exist_ok=True)
        
        with open("vertex_build/Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        with open("vertex_build/cloudbuild.yaml", "w") as f:
            json.dump(cloudbuild_config, f, indent=2)
        
        logger.info("Created training container configuration")
        logger.info("Build with: gcloud builds submit vertex_build/ --config=vertex_build/cloudbuild.yaml")
        
        return f"gcr.io/{self.project_id}/solar-yolo-trainer:latest"
    
    def submit_training_job(self, 
                           display_name: str,
                           container_uri: str,
                           machine_type: str = "gpu_t4",
                           replica_count: int = 1,
                           args: list = None,
                           environment_variables: dict = None,
                           max_runtime: str = "7200s") -> str:
        """Submit YOLO training job to Vertex AI"""
        
        if args is None:
            args = [
                "--data", "gs://solar-yolo/dataset.yaml",
                "--epochs", "50",
                "--imgsz", "640", 
                "--batch", "16",
                "--device", "0"
            ]
        
        if environment_variables is None:
            environment_variables = {
                "GOOGLE_CLOUD_PROJECT": self.project_id,
                "PYTHONPATH": "/app"
            }
        
        # Configure machine spec
        machine_spec = {
            "machine_type": self.machine_types[machine_type],
        }
        
        # Add accelerator if GPU machine
        if machine_type.startswith("gpu_"):
            machine_spec["accelerator_type"] = self.accelerator_types[machine_type]
            machine_spec["accelerator_count"] = 1
        
        # Create container spec
        container_spec = {
            "image_uri": container_uri,
            "args": args,
            "env": [{"name": k, "value": v} for k, v in environment_variables.items()]
        }
        
        # Create worker pool spec
        worker_pool_spec = {
            "replica_count": replica_count,
            "machine_spec": machine_spec,
            "container_spec": container_spec
        }
        
        # Create training job
        job = aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=[worker_pool_spec]
        )
        
        logger.info(f"Submitting training job: {display_name}")
        logger.info(f"Machine type: {machine_type}")
        logger.info(f"Container: {container_uri}")
        logger.info(f"Args: {args}")
        
        # Submit job
        job.run(
            timeout=max_runtime,
            restart_job_on_worker_restart=False
        )
        
        logger.info(f"Training job submitted successfully: {job.resource_name}")
        return job.resource_name
    
    def create_hyperparameter_tuning_job(self, 
                                        display_name: str,
                                        container_uri: str,
                                        machine_type: str = "gpu_t4",
                                        max_trials: int = 20,
                                        parallel_trials: int = 4):
        """Create hyperparameter tuning job for YOLO"""
        
        # Define hyperparameter search space
        hyperparameter_spec = {
            "parameters": [
                {
                    "parameter_id": "learning_rate",
                    "double_value_spec": {
                        "min_value": 0.001,
                        "max_value": 0.1
                    },
                    "scale_type": "UNIT_LOG_SCALE"
                },
                {
                    "parameter_id": "batch_size", 
                    "discrete_value_spec": {
                        "values": [8, 16, 32]
                    }
                },
                {
                    "parameter_id": "epochs",
                    "integer_value_spec": {
                        "min_value": 20,
                        "max_value": 100
                    }
                }
            ],
            "objective": {
                "type_": "MAXIMIZE",
                "metric_id": "map50"
            },
            "algorithm": "RANDOM_SEARCH"
        }
        
        # Base training arguments
        base_args = [
            "--data", "gs://solar-yolo/dataset.yaml",
            "--imgsz", "640",
            "--device", "0",
            "--project", f"gs://solar-models/yolo_runs/{display_name}"
        ]
        
        # Container spec with hyperparameter placeholders
        container_spec = {
            "image_uri": container_uri,
            "args": base_args + [
                "--lr0", "${learning_rate}",
                "--batch", "${batch_size}",
                "--epochs", "${epochs}"
            ]
        }
        
        # Machine spec
        machine_spec = {
            "machine_type": self.machine_types[machine_type],
        }
        
        if machine_type.startswith("gpu_"):
            machine_spec["accelerator_type"] = self.accelerator_types[machine_type]
            machine_spec["accelerator_count"] = 1
        
        # Create hyperparameter tuning job
        job = aiplatform.HyperparameterTuningJob(
            display_name=display_name,
            custom_job={
                "worker_pool_specs": [{
                    "replica_count": 1,
                    "machine_spec": machine_spec,
                    "container_spec": container_spec
                }]
            },
            parameter_spec=hyperparameter_spec,
            max_trial_count=max_trials,
            parallel_trial_count=parallel_trials
        )
        
        logger.info(f"Creating hyperparameter tuning job: {display_name}")
        logger.info(f"Max trials: {max_trials}, Parallel trials: {parallel_trials}")
        
        job.run()
        
        logger.info(f"Hyperparameter tuning job created: {job.resource_name}")
        return job.resource_name
    
    def list_training_jobs(self, filter_str: str = None) -> list:
        """List training jobs"""
        jobs = aiplatform.CustomJob.list(filter=filter_str)
        
        logger.info(f"Found {len(jobs)} training jobs")
        for job in jobs:
            logger.info(f"  {job.display_name}: {job.state}")
        
        return jobs
    
    def get_job_status(self, job_name: str) -> dict:
        """Get status of a training job"""
        job = aiplatform.CustomJob(job_name)
        
        return {
            "display_name": job.display_name,
            "state": job.state,
            "create_time": job.create_time,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "error": job.error
        }
    
    def cancel_job(self, job_name: str):
        """Cancel a running training job"""
        job = aiplatform.CustomJob(job_name)
        job.cancel()
        logger.info(f"Cancelled job: {job_name}")


def main():
    """Main function for Vertex AI YOLO training"""
    parser = argparse.ArgumentParser(description="Submit YOLO training to Vertex AI")
    parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", help="Vertex AI region")
    parser.add_argument("--job-name", help="Training job name")
    parser.add_argument("--container-uri", help="Container URI (if not provided, will use default)")
    parser.add_argument("--machine-type", default="gpu_t4", 
                       choices=["cpu", "gpu_k80", "gpu_v100", "gpu_t4", "gpu_a100"],
                       help="Machine type for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--dataset-path", default="gs://solar-yolo/dataset.yaml", help="Dataset path")
    parser.add_argument("--output-path", help="Model output path")
    parser.add_argument("--hyperparameter-tuning", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--list-jobs", action="store_true", help="List training jobs")
    parser.add_argument("--build-container", action="store_true", help="Create container build files")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = VertexAIYOLOTrainer(args.project_id, args.region)
    
    # List jobs if requested
    if args.list_jobs:
        trainer.list_training_jobs()
        return 0
    
    # Build container if requested
    if args.build_container:
        container_uri = trainer.create_training_image()
        logger.info(f"Container configuration created. Build with:")
        logger.info(f"cd vertex_build && gcloud builds submit . --config cloudbuild.yaml")
        return 0
    
    # Set default container URI
    if not args.container_uri:
        args.container_uri = f"gcr.io/{args.project_id}/solar-yolo-trainer:latest"
    
    # Generate job name if not provided
    if not args.job_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.job_name = f"solar-yolo-{timestamp}"
    
    # Set output path if not provided
    if not args.output_path:
        args.output_path = f"gs://solar-models/yolo/{args.job_name}"
    
    # Training arguments
    training_args = [
        "--data", args.dataset_path,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch_size),
        "--lr0", str(args.learning_rate),
        "--imgsz", "640",
        "--device", "0",
        "--project", args.output_path,
        "--name", "train"
    ]
    
    try:
        if args.hyperparameter_tuning:
            # Submit hyperparameter tuning job
            job_name = trainer.create_hyperparameter_tuning_job(
                display_name=f"{args.job_name}_hpt",
                container_uri=args.container_uri,
                machine_type=args.machine_type
            )
        else:
            # Submit regular training job
            job_name = trainer.submit_training_job(
                display_name=args.job_name,
                container_uri=args.container_uri,
                machine_type=args.machine_type,
                args=training_args
            )
        
        logger.info(f"Job submitted successfully: {job_name}")
        logger.info(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={args.project_id}")
        
    except Exception as e:
        logger.error(f"Failed to submit training job: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())