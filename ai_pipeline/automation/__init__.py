"""
Automation module for Solar CV Pipeline
Contains Cloud Composer DAGs and orchestration scripts
"""

__version__ = "1.0.0"
__author__ = "Solar CV Team"

# Import main components
from .solar_cv_dag import dag as solar_cv_pipeline_dag

__all__ = [
    "solar_cv_pipeline_dag"
]