#!/usr/bin/env python3
"""
Cloud Composer DAG for Solar CV Pipeline Automation
Automates the complete pipeline: ingestion → preprocessing → training → deployment
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.providers.google.cloud.operators.vertex_ai import (
    CreateCustomJobOperator,
    CreateModelOperator,
    CreateEndpointOperator,
    DeployModelOperator
)
from airflow.providers.google.cloud.operators.bigquery import BigQueryCheckOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor
import logging

# Configuration
PROJECT_ID = "{{ var.value.project_id }}"
REGION = "us-central1"
BUCKET_RAW = "solar-raw"
BUCKET_MODELS = "solar-models"

# Default arguments
default_args = {
    'owner': 'solar-cv-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': ['admin@example.com']
}

# Create DAG
dag = DAG(
    'solar_cv_pipeline',
    default_args=default_args,
    description='Automated Solar CV Pipeline',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    tags=['solar', 'cv', 'ml', 'space-weather'],
    max_active_runs=1
)

def calculate_date_range(**context):
    """Calculate date range for data ingestion"""
    execution_date = context['execution_date']
    start_date = execution_date - timedelta(days=1)
    end_date = execution_date
    
    context['task_instance'].xcom_push(
        key='start_date', 
        value=start_date.strftime('%Y-%m-%dT%H:%M:%S')
    )
    context['task_instance'].xcom_push(
        key='end_date', 
        value=end_date.strftime('%Y-%m-%dT%H:%M:%S')
    )
    
    logging.info(f"Processing data from {start_date} to {end_date}")

def check_data_quality(**context):
    """Check data quality after ingestion"""
    ti = context['task_instance']
    start_date = ti.xcom_pull(key='start_date', task_ids='calculate_dates')
    end_date = ti.xcom_pull(key='end_date', task_ids='calculate_dates')
    
    # Add data quality checks here
    logging.info(f"Checking data quality for {start_date} to {end_date}")
    
    # Example checks:
    # - Minimum number of images ingested
    # - NOAA events count
    # - Image file integrity
    
    return True

def trigger_retraining_decision(**context):
    """Decide whether to trigger model retraining"""
    # Check conditions for retraining:
    # - Enough new data accumulated
    # - Model performance degradation
    # - Schedule-based retraining
    
    # For demo, retrain weekly
    execution_date = context['execution_date']
    should_retrain = execution_date.weekday() == 0  # Monday
    
    logging.info(f"Retraining decision: {should_retrain}")
    return should_retrain

# Task 1: Calculate date range
calculate_dates = PythonOperator(
    task_id='calculate_dates',
    python_callable=calculate_date_range,
    dag=dag
)

# Task 2: Ingest solar data
ingest_data = BashOperator(
    task_id='ingest_solar_data',
    bash_command=f"""
    python /home/airflow/gcs/dags/cv_pipeline/cloud_ingest/enhanced_ingestor.py \\
        --start "{{{{ task_instance.xcom_pull(key='start_date', task_ids='calculate_dates') }}}}" \\
        --end "{{{{ task_instance.xcom_pull(key='end_date', task_ids='calculate_dates') }}}}" \\
        --project-id {PROJECT_ID}
    """,
    dag=dag
)

# Task 3: Check data quality
check_quality = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag
)

# Task 4: Check BigQuery data
check_bq_data = BigQueryCheckOperator(
    task_id='check_bigquery_data',
    sql=f"""
    SELECT COUNT(*) as event_count
    FROM `{PROJECT_ID}.space_weather.solar_events`
    WHERE DATE(timestamp) = DATE('{{{{ ds }}}}')
    """,
    use_legacy_sql=False,
    dag=dag
)

# Task 5: List new GCS objects
list_gcs_objects = GCSListObjectsOperator(
    task_id='list_new_gcs_objects',
    bucket=BUCKET_RAW,
    prefix='aia/193/',
    dag=dag
)

# Task 6: Prepare YOLO dataset
prepare_yolo_dataset = BashOperator(
    task_id='prepare_yolo_dataset',
    bash_command=f"""
    python /home/airflow/gcs/dags/cv_pipeline/preprocess/gcs_yolo_preparer.py \\
        --project-id {PROJECT_ID} \\
        --start-date "{{{{ ds }}}}" \\
        --end-date "{{{{ ds }}}}" \\
        --max-images 100
    """,
    dag=dag
)

# Task 7: Retraining decision
retraining_decision = PythonOperator(
    task_id='retraining_decision',
    python_callable=trigger_retraining_decision,
    dag=dag
)

# Task 8: Train YOLO model (conditional)
train_yolo = CreateCustomJobOperator(
    task_id='train_yolo_model',
    staging_bucket=f"gs://{BUCKET_MODELS}",
    display_name="solar-yolo-training-{{ ds_nodash }}",
    custom_job={
        "worker_pool_specs": [
            {
                "replica_count": 1,
                "machine_spec": {
                    "machine_type": "n1-standard-8",
                    "accelerator_type": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1
                },
                "container_spec": {
                    "image_uri": f"gcr.io/{PROJECT_ID}/solar-yolo-trainer:latest",
                    "args": [
                        "--data", f"gs://{BUCKET_RAW}/yolo_dataset/dataset.yaml",
                        "--epochs", "50",
                        "--batch", "16",
                        "--imgsz", "640",
                        "--project", f"gs://{BUCKET_MODELS}/yolo/{{{{ ds_nodash }}}}",
                        "--name", "train"
                    ]
                }
            }
        ]
    },
    region=REGION,
    project_id=PROJECT_ID,
    dag=dag
)

# Task 9: Train ViT model (conditional)
train_vit = CreateCustomJobOperator(
    task_id='train_vit_model',
    staging_bucket=f"gs://{BUCKET_MODELS}",
    display_name="solar-vit-training-{{ ds_nodash }}",
    custom_job={
        "worker_pool_specs": [
            {
                "replica_count": 1,
                "machine_spec": {
                    "machine_type": "n1-standard-8",
                    "accelerator_type": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1
                },
                "container_spec": {
                    "image_uri": f"gcr.io/{PROJECT_ID}/solar-vit-trainer:latest",
                    "args": [
                        "--epochs", "25",
                        "--batch", "32",
                        "--lr", "0.0001",
                        "--output-path", f"gs://{BUCKET_MODELS}/vit/{{{{ ds_nodash }}}}"
                    ]
                }
            }
        ]
    },
    region=REGION,
    project_id=PROJECT_ID,
    dag=dag
)

# Task 10: Wait for YOLO training completion
wait_yolo_model = GCSObjectExistenceSensor(
    task_id='wait_for_yolo_model',
    bucket=BUCKET_MODELS,
    object=f"yolo/{{{{ ds_nodash }}}}/train/weights/best.pt",
    timeout=3600,  # 1 hour timeout
    poke_interval=60,  # Check every minute
    dag=dag
)

# Task 11: Wait for ViT training completion
wait_vit_model = GCSObjectExistenceSensor(
    task_id='wait_for_vit_model',
    bucket=BUCKET_MODELS,
    object=f"vit/{{{{ ds_nodash }}}}/solar_vit_best.pth",
    timeout=3600,
    poke_interval=60,
    dag=dag
)

# Task 12: Register YOLO model
register_yolo_model = CreateModelOperator(
    task_id='register_yolo_model',
    project_id=PROJECT_ID,
    region=REGION,
    model={
        "display_name": "solar-yolo-{{ ds_nodash }}",
        "artifact_uri": f"gs://{BUCKET_MODELS}/yolo/{{{{ ds_nodash }}}}",
        "container_spec": {
            "image_uri": f"gcr.io/{PROJECT_ID}/solar-yolo-predictor:latest",
            "ports": [{"container_port": 8080}],
            "predict_route": "/predict",
            "health_route": "/health"
        }
    },
    dag=dag
)

# Task 13: Register ViT model
register_vit_model = CreateModelOperator(
    task_id='register_vit_model',
    project_id=PROJECT_ID,
    region=REGION,
    model={
        "display_name": "solar-vit-{{ ds_nodash }}",
        "artifact_uri": f"gs://{BUCKET_MODELS}/vit/{{{{ ds_nodash }}}}",
        "container_spec": {
            "image_uri": f"gcr.io/{PROJECT_ID}/solar-vit-predictor:latest",
            "ports": [{"container_port": 8080}],
            "predict_route": "/predict",
            "health_route": "/health"
        }
    },
    dag=dag
)

# Task 14: Deploy YOLO endpoint
deploy_yolo_endpoint = DeployModelOperator(
    task_id='deploy_yolo_endpoint',
    project_id=PROJECT_ID,
    region=REGION,
    endpoint={
        "display_name": "solar-yolo-endpoint"
    },
    deployed_model={
        "display_name": "solar-yolo-deployed-{{ ds_nodash }}",
        "model": "{{ task_instance.xcom_pull(task_ids='register_yolo_model') }}",
        "machine_spec": {
            "machine_type": "n1-standard-2"
        },
        "min_replica_count": 1,
        "max_replica_count": 3
    },
    dag=dag
)

# Task 15: Deploy ViT endpoint
deploy_vit_endpoint = DeployModelOperator(
    task_id='deploy_vit_endpoint',
    project_id=PROJECT_ID,
    region=REGION,
    endpoint={
        "display_name": "solar-vit-endpoint"
    },
    deployed_model={
        "display_name": "solar-vit-deployed-{{ ds_nodash }}",
        "model": "{{ task_instance.xcom_pull(task_ids='register_vit_model') }}",
        "machine_spec": {
            "machine_type": "n1-standard-2"
        },
        "min_replica_count": 1,
        "max_replica_count": 3
    },
    dag=dag
)

# Task 16: Validate deployments
validate_deployments = BashOperator(
    task_id='validate_deployments',
    bash_command="""
    echo "Running endpoint validation tests..."
    # Add validation scripts here
    python /home/airflow/gcs/dags/cv_pipeline/deploy/validate_endpoints.py
    """,
    dag=dag
)

# Task 17: Send notification
send_notification = BashOperator(
    task_id='send_notification',
    bash_command="""
    echo "Solar CV Pipeline completed successfully for {{ ds }}"
    # Add notification logic (email, Slack, etc.)
    """,
    dag=dag
)

# Define dependencies
calculate_dates >> ingest_data >> [check_quality, check_bq_data, list_gcs_objects]

[check_quality, check_bq_data, list_gcs_objects] >> prepare_yolo_dataset >> retraining_decision

# Conditional training based on retraining decision
retraining_decision >> [train_yolo, train_vit]

# Wait for training completion
train_yolo >> wait_yolo_model >> register_yolo_model >> deploy_yolo_endpoint
train_vit >> wait_vit_model >> register_vit_model >> deploy_vit_endpoint

# Final validation and notification
[deploy_yolo_endpoint, deploy_vit_endpoint] >> validate_deployments >> send_notification

# Documentation
dag.doc_md = """
# Solar CV Pipeline DAG

This DAG automates the complete Solar Computer Vision pipeline:

## Pipeline Stages

1. **Data Ingestion**: Fetch solar images and NOAA events
2. **Data Quality**: Validate ingested data
3. **Preprocessing**: Prepare YOLO datasets with proper annotations
4. **Training Decision**: Determine if retraining is needed
5. **Model Training**: Train YOLO and ViT models on Vertex AI
6. **Model Registration**: Register trained models in Vertex AI
7. **Deployment**: Deploy models as Vertex AI endpoints
8. **Validation**: Test deployed endpoints
9. **Notification**: Send completion notifications

## Schedule

- **Frequency**: Daily
- **Start Date**: 2024-01-01
- **Timezone**: UTC

## Configuration

Set the following Airflow variables:
- `project_id`: Google Cloud Project ID
- `notification_email`: Email for alerts

## Monitoring

- Check Airflow UI for task status
- Monitor Vertex AI training jobs in GCP Console
- Review BigQuery tables for data quality
"""