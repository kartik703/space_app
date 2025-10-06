#!/usr/bin/env python3
"""
🚀 SPACE INTELLIGENCE AI PLATFORM - DEPLOYMENT RUNNER
Execute the complete production setup transformation
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command: str, description: str = ""):
    """Run command with error handling"""
    print(f"\n[RUNNING] {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"[SUCCESS] {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {e.stderr.strip()}")
        return False

def main():
    """Execute production setup"""
    
    print("""
SPACE INTELLIGENCE AI PLATFORM - PRODUCTION DEPLOYMENT
========================================================
Enterprise-grade deployment setup starting...
    """)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print(f"[INFO] Working directory: {project_dir.absolute()}")
    
    # Step 1: Run production setup
    print("\n" + "="*50)
    print("STEP 1: Running Production Setup Script")
    print("="*50)
    
    if run_command("python production_setup.py", "Executing production setup transformation"):
        print("[SUCCESS] Production setup completed successfully!")
    else:
        print("[ERROR] Production setup failed!")
        return False
    
    # Step 2: Build Docker images
    print("\n" + "="*50)
    print("STEP 2: Building Docker Images")
    print("="*50)
    
    # Check if Docker is available
    if not run_command("docker --version", "Checking Docker availability"):
        print("[WARNING] Docker not available, skipping image build")
    else:
        # Build backend image
        run_command("docker build -f Dockerfile.backend -t space-ai-backend .", "Building backend image")
        
        # Build frontend image
        run_command("docker build -f Dockerfile.frontend -t space-ai-frontend .", "Building frontend image")
    
    # Step 3: Setup environment
    print("\n" + "="*50)
    print("STEP 3: Environment Configuration")
    print("="*50)
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# SPACE INTELLIGENCE AI PLATFORM - ENVIRONMENT CONFIG
POSTGRES_PASSWORD=secure_space_password_2024
JWT_SECRET_KEY=your_jwt_secret_key_change_in_production
NASA_API_KEY=DEMO_KEY
NOAA_API_KEY=your_noaa_api_key
GRAFANA_PASSWORD=admin
ENVIRONMENT=production
LOG_LEVEL=INFO"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"[SUCCESS] Created .env file: {env_file.absolute()}")
    else:
        print(f"[INFO] Environment file already exists: {env_file.absolute()}")
    
    # Step 4: Deployment instructions
    print("\n" + "="*50)
    print("STEP 4: Deployment Instructions")
    print("="*50)
    
    print("""
DEPLOYMENT READY!

To start the Space Intelligence AI Platform:

1. Production Docker Deployment:
   docker-compose up -d

2. Development Mode:
   # Backend
   uvicorn backend_api:app --reload --port 8000
   
   # Frontend (in another terminal)
   streamlit run production_dashboard.py --server.port 8501

3. Access Points:
   * Frontend Dashboard: http://localhost:8501
   * Backend API: http://localhost:8000
   * API Documentation: http://localhost:8000/docs
   * Grafana Monitoring: http://localhost:3000
   * Prometheus Metrics: http://localhost:9090

4. Quick Test:
   curl http://localhost:8000/api/health

Next Steps:
   * Update API keys in .env file
   * Configure SSL certificates for production
   * Setup monitoring alerts
   * Run initial data collection
   * Train AI models

Security Notes:
   * Change default passwords in .env
   * Setup proper authentication
   * Configure firewall rules
   * Enable SSL/TLS for production
    """)
    
    print("\n[SUCCESS] Space Intelligence AI Platform is ready for deployment!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)