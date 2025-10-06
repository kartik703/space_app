#!/usr/bin/env python3
"""
🚀 Space Intelligence Platform - Model Creator
Creates placeholder ML models to get the app running
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create all required model directories"""
    dirs = [
        'models',
        'cv_pipeline/models', 
        'ai_pipeline/models',
        'data/sdo_solar_images',
        'data/live_solar_images'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def create_basic_models():
    """Create basic ML models that the app expects"""
    
    # Generate some dummy training data
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.rand(100)      # Target values
    
    # 1. Random Forest Model for space weather prediction
    print("🤖 Creating Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_dummy, y_dummy)
    
    rf_scaler = StandardScaler()
    rf_scaler.fit(X_dummy)
    
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(rf_scaler, 'models/rf_scaler.pkl')
    print("✅ Created rf_model.pkl and rf_scaler.pkl")
    
    # 2. Storm Prediction Model
    print("🌪️ Creating Storm Prediction model...")
    storm_model = RandomForestRegressor(n_estimators=30, random_state=42)
    storm_model.fit(X_dummy, y_dummy)
    
    joblib.dump(storm_model, 'models/storm_prediction_model.pkl')
    print("✅ Created storm_prediction_model.pkl")
    
    # 3. Anomaly Detection Model
    print("🔍 Creating Anomaly Detection model...")
    anomaly_model = IsolationForest(contamination=0.1, random_state=42)
    anomaly_model.fit(X_dummy)
    
    anomaly_scaler = StandardScaler()
    anomaly_scaler.fit(X_dummy)
    
    joblib.dump(anomaly_model, 'models/anomaly_model.pkl')
    joblib.dump(anomaly_scaler, 'models/anomaly_scaler.pkl')
    print("✅ Created anomaly_model.pkl and anomaly_scaler.pkl")

def create_yolo_model():
    """Create a basic YOLO model for object detection"""
    try:
        print("🎯 Creating YOLO model...")
        # Use pre-trained YOLOv8n model
        model = YOLO('yolov8n.pt')  # This will download if not exists
        
        # Save to expected location
        model.save('models/yolo_model.pt')
        print("✅ Created yolo_model.pt")
        
    except Exception as e:
        print(f"⚠️ YOLO model creation skipped: {e}")
        print("   (This is optional - app will use default)")

def create_sample_solar_images():
    """Create some dummy solar image files for testing"""
    try:
        import cv2
        
        # Create a simple dummy solar image (orange circle on black background)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(img, (256, 256), 200, (0, 165, 255), -1)  # Orange circle (BGR format)
        
        # Save sample images
        sample_dir = 'data/sdo_solar_images'
        for i in range(3):
            filename = f'{sample_dir}/sample_solar_{i+1}.jpg'
            cv2.imwrite(filename, img)
            print(f"✅ Created sample solar image: {filename}")
            
    except Exception as e:
        print(f"⚠️ Sample images creation skipped: {e}")

def main():
    """Main function to create all required models and files"""
    print("🚀 Space Intelligence Platform - Model Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    print()
    
    # Create ML models
    create_basic_models()
    print()
    
    # Create YOLO model (optional)
    create_yolo_model()
    print()
    
    # Create sample images
    create_sample_solar_images()
    print()
    
    print("🎉 Model setup complete!")
    print("📱 Your app should now run without missing model errors")
    print("🌐 Navigate to http://localhost:8501 to test")

if __name__ == "__main__":
    main()