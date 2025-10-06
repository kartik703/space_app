#!/usr/bin/env python3
"""
🚀 SPACE INTELLIGENCE AI PLATFORM - PRODUCTION SETUP
Automated setup script for enterprise-grade architecture transformation
"""

import os
import sys
from pathlib import Path
import json
import shutil

class ProductionSetup:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.app_dir = self.base_dir / "app"
        
        print("SPACE INTELLIGENCE AI PLATFORM - PRODUCTION SETUP")
        print("=" * 60)
        
    def create_production_structure(self):
        """Create enterprise-grade directory structure"""
        
        structure = {
            # Main application
            "app": {
                "frontend": {
                    "components": {},
                    "pages": {},
                    "assets": {"css": {}, "js": {}, "images": {}},
                    "config": {}
                },
                "backend": {
                    "api": {"v1": {"endpoints": {}}},
                    "core": {"models": {}, "services": {}, "utils": {}},
                    "auth": {},
                    "database": {"models": {}, "migrations": {}}
                },
                "shared": {"schemas": {}, "constants": {}, "exceptions": {}},
                "config": {"environments": {}}
            },
            
            # AI & ML Pipeline
            "ai_pipeline": {
                "models": {"yolo": {}, "ml": {}, "fusion": {}},
                "inference": {},
                "training": {},
                "preprocessing": {}
            },
            
            # Data Pipeline
            "data_pipeline": {
                "collectors": {"nasa": {}, "noaa": {}, "ground": {}},
                "processors": {},
                "storage": {},
                "validators": {}
            },
            
            # Infrastructure
            "deployment": {
                "docker": {},
                "kubernetes": {},
                "terraform": {},
                "scripts": {}
            },
            
            # Quality & Testing
            "tests": {
                "unit": {},
                "integration": {},
                "e2e": {},
                "performance": {},
                "fixtures": {}
            },
            
            # Documentation
            "docs": {
                "api": {},
                "user_guide": {},
                "deployment": {},
                "architecture": {}
            },
            
            # Operations
            "monitoring": {"dashboards": {}, "alerts": {}},
            "logs": {"app": {}, "ai": {}, "data": {}},
            "storage": {"models": {}, "data": {}, "exports": {}}
        }
        
        print("[INFO] Creating production directory structure...")
        
        def create_dirs(base_path, structure_dict):
            for name, subdirs in structure_dict.items():
                dir_path = base_path / name
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Create __init__.py for Python packages
                if name in ["app", "ai_pipeline", "data_pipeline"]:
                    (dir_path / "__init__.py").touch()
                
                if isinstance(subdirs, dict) and subdirs:
                    create_dirs(dir_path, subdirs)
                    
        create_dirs(self.base_dir, structure)
        
        # Create .gitkeep files for empty directories
        for root, dirs, files in os.walk(self.base_dir):
            root_path = Path(root)
            if not files and not dirs:
                (root_path / ".gitkeep").touch()
                
        print("[SUCCESS] Production structure created!")
        
    def migrate_existing_code(self):
        """Migrate existing components to new structure"""
        
        print("[INFO] Migrating existing components...")
        
        migrations = [
            # AI Models & Pipeline
            ("fusion_ai_live.py", "app/frontend/main_dashboard.py"),
            ("continuous_space_collector.py", "data_pipeline/collectors/unified_collector.py"),
            ("train_all_ai_models.py", "ai_pipeline/training/model_trainer.py"),
            ("integrated_space_ai_platform.py", "app/backend/core/services/ai_service.py"),
            
            # Configuration
            ("utils.py", "app/shared/utils/helpers.py"),
            ("launch.py", "deployment/scripts/launcher.py"),
            
            # Move CV pipeline
            ("cv_pipeline/", "ai_pipeline/"),
        ]
        
        for source, target in migrations:
            source_path = self.base_dir / source
            target_path = self.base_dir / target
            
            if source_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path.is_dir():
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(source_path, target_path)
                else:
                    shutil.copy2(source_path, target_path)
                print(f"  [SUCCESS] Migrated {source} -> {target}")
        
        print("[SUCCESS] Code migration completed!")
        
    def create_config_files(self):
        """Create configuration files for production"""
        
        print("[INFO] Creating configuration files...")
        
        # FastAPI Backend Config
        backend_config = {
            "app": {
                "title": "Space Intelligence AI Platform",
                "version": "1.0.0",
                "description": "Enterprise-grade space weather monitoring and risk assessment",
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False
            },
            "ai_models": {
                "yolo_path": "storage/models/solar_yolo.pt",
                "ml_models_dir": "storage/models/",
                "inference_timeout": 30
            },
            "data_sources": {
                "nasa_sdo": "https://sdo.gsfc.nasa.gov/assets/img/latest/",
                "noaa_swpc": "https://services.swpc.noaa.gov/json/",
                "kyoto_wdc": "http://wdc.kugi.kyoto-u.ac.jp/",
                "collection_interval": 30
            },
            "security": {
                "jwt_secret": "your-secret-key-change-in-production",
                "jwt_expire_hours": 24,
                "rate_limit": "100/hour"
            }
        }
        
        config_file = self.app_dir / "config" / "settings.json"
        with open(config_file, 'w') as f:
            json.dump(backend_config, f, indent=2)
        
        # Docker Configuration
        dockerfile_content = '''# Space Intelligence AI Platform
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Run application
CMD ["python", "deployment/scripts/production_launcher.py"]
'''
        
        with open(self.base_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        # Docker Compose
        docker_compose = '''version: '3.8'

services:
  space-ai-backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    volumes:
      - ./storage:/app/storage
      - ./logs:/app/logs
    restart: unless-stopped
    
  space-ai-frontend:
    build: .
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://space-ai-backend:8000
    depends_on:
      - space-ai-backend
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
'''
        
        with open(self.base_dir / "docker-compose.yml", 'w') as f:
            f.write(docker_compose)
        
        print("[SUCCESS] Configuration files created!")
        
    def create_production_requirements(self):
        """Create production requirements with additional enterprise packages"""
        
        production_deps = """# Production Dependencies
# Core Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
streamlit>=1.28.0

# AI & ML
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
pillow>=10.0.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
aiohttp>=3.8.0

# Database & Storage
sqlalchemy>=2.0.0
alembic>=1.12.0
redis>=5.0.0
asyncpg>=0.28.0

# Authentication & Security
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.0
python-multipart>=0.0.6

# API & Documentation
pydantic>=2.4.0
pydantic-settings>=2.0.0

# Visualization
plotly>=5.15.0
matplotlib>=3.7.0

# Monitoring & Logging
prometheus-client>=0.17.0
structlog>=23.1.0

# Deployment
gunicorn>=21.2.0
supervisor>=4.2.0

# Development & Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.7.0
flake8>=6.0.0
mypy>=1.5.0

# Space Science
astropy>=5.3.0
"""
        
        with open(self.base_dir / "requirements-production.txt", 'w') as f:
            f.write(production_deps)
        
        print("[SUCCESS] Production requirements created!")

def main():
    """Main setup function"""
    setup = ProductionSetup()
    
    print("\n[INFO] Starting Production Transformation...")
    print("This will create enterprise-grade architecture")
    
    setup.create_production_structure()
    setup.migrate_existing_code()
    setup.create_config_files()
    setup.create_production_requirements()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] PRODUCTION SETUP COMPLETE!")
    print("=" * 60)
    print("""
Next Steps:
1. Review the new structure in app/ directory
2. Install production dependencies: pip install -r requirements-production.txt
3. Configure settings in app/config/settings.json
4. Run: python deployment/scripts/production_launcher.py
5. Access: http://localhost:8000 (API) & http://localhost:8501 (Dashboard)
    """)

if __name__ == "__main__":
    main()