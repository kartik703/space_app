#!/usr/bin/env python3
"""
COMPLETE AI MODEL TRAINER - Using ALL Collected Real Data
Trains all missing AI models using our collected space data:
- YOLO for solar flare detection on SDO images
- Random Forest for weather prediction on NOAA data  
- Anomaly detection for unusual patterns
- Uses ALL data from continuous_space_data folder
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveAITrainer:
    def __init__(self, data_dir="data/continuous_space_data"):
        self.data_dir = data_dir
        self.models_dir = "models"
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Data paths
        self.sdo_dir = os.path.join(data_dir, "sdo_solar_images")
        self.noaa_dir = os.path.join(data_dir, "noaa_space_weather")
        self.ground_dir = os.path.join(data_dir, "ground_observations")
        
        logger.info("🚀 COMPREHENSIVE AI TRAINER INITIALIZED")
        logger.info(f"📁 Data Directory: {data_dir}")
        logger.info(f"🧠 Models Directory: {self.models_dir}")

    def prepare_yolo_dataset(self):
        """Prepare YOLO dataset from collected solar images"""
        logger.info("🌞 PREPARING YOLO SOLAR FLARE DATASET...")
        
        if not os.path.exists(self.sdo_dir):
            logger.error(f"❌ SDO directory not found: {self.sdo_dir}")
            return False
        
        # Get all solar images
        image_files = glob.glob(os.path.join(self.sdo_dir, "*.jpg"))
        logger.info(f"📸 Found {len(image_files)} solar images")
        
        if len(image_files) < 10:
            logger.warning("⚠️ Not enough images for YOLO training")
            return False
        
        # Create YOLO dataset structure
        yolo_dataset_dir = "data/yolo_solar_dataset"
        os.makedirs(f"{yolo_dataset_dir}/images/train", exist_ok=True)
        os.makedirs(f"{yolo_dataset_dir}/images/val", exist_ok=True)
        os.makedirs(f"{yolo_dataset_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{yolo_dataset_dir}/labels/val", exist_ok=True)
        
        # Process images and create synthetic labels based on image analysis
        train_count = 0
        val_count = 0
        
        for i, img_path in enumerate(image_files):
            try:
                # Load and analyze image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                height, width = img.shape[:2]
                brightness = float(np.mean(img.astype(np.float32)))
                
                # Create filename
                filename = f"solar_{i:06d}"
                
                # Determine if train or validation (80/20 split)
                is_train = i % 5 != 0
                subset = "train" if is_train else "val"
                
                # Copy image to dataset
                img_dest = f"{yolo_dataset_dir}/images/{subset}/{filename}.jpg"
                cv2.imwrite(img_dest, img)
                
                # Generate label based on image characteristics
                label_path = f"{yolo_dataset_dir}/labels/{subset}/{filename}.txt"
                
                # Create synthetic labels for bright regions (potential flares)
                labels = []
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find bright regions
                _, thresh = cv2.threshold(gray, int(brightness * 1.2), 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Minimum area for flare
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Convert to YOLO format (normalized)
                        center_x = (x + w/2) / width
                        center_y = (y + h/2) / height
                        norm_w = w / width
                        norm_h = h / height
                        
                        # Class 0 for solar flare
                        labels.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
                
                # Write labels
                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels))
                
                if is_train:
                    train_count += 1
                else:
                    val_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Error processing {img_path}: {e}")
        
        # Create YOLO config file
        config_content = f"""
path: {os.path.abspath(yolo_dataset_dir)}
train: images/train
val: images/val

nc: 1
names: ['solar_flare']
"""
        
        with open(f"{yolo_dataset_dir}/config.yaml", 'w') as f:
            f.write(config_content)
        
        logger.info(f"✅ YOLO dataset prepared: {train_count} train, {val_count} val images")
        return yolo_dataset_dir

    def train_yolo_model(self, dataset_dir):
        """Train YOLO model for solar flare detection"""
        logger.info("🔥 TRAINING YOLO SOLAR FLARE MODEL...")
        
        try:
            # Load YOLOv8 model
            model = YOLO('yolov8n.pt')  # Start with nano model
            
            # Train the model
            results = model.train(
                data=f"{dataset_dir}/config.yaml",
                epochs=50,
                imgsz=640,
                batch=8,
                name='solar_flare_detection',
                project=self.models_dir,
                save=True,
                patience=10
            )
            
            # Save the trained model
            model_path = os.path.join(self.models_dir, "solar_yolo.pt")
            model.save(model_path)
            
            logger.info(f"✅ YOLO model saved: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ YOLO training failed: {e}")
            return False

    def prepare_weather_data(self):
        """Prepare weather data for ML training"""
        logger.info("🌪️ PREPARING WEATHER DATA FOR ML TRAINING...")
        
        if not os.path.exists(self.noaa_dir):
            logger.error(f"❌ NOAA directory not found: {self.noaa_dir}")
            return None
        
        # Get all weather JSON files
        weather_files = glob.glob(os.path.join(self.noaa_dir, "*.json"))
        logger.info(f"📊 Found {len(weather_files)} weather files")
        
        if len(weather_files) < 5:
            logger.warning("⚠️ Not enough weather data files")
            return None
        
        all_data = []
        
        for file_path in weather_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list) and len(data) > 0:
                    all_data.extend(data)
                    
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
        
        if not all_data:
            logger.error("❌ No valid weather data found")
            return None
        
        logger.info(f"📈 Loaded {len(all_data)} weather records")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Extract numerical features
        numerical_cols = []
        for col in df.columns:
            if col not in ['time_tag', 'satellite']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numerical_cols.append(col)
                except:
                    pass
        
        # Clean data
        df = df[numerical_cols].dropna()
        
        if len(df) < 100:
            logger.warning("⚠️ Insufficient clean weather data")
            return None
        
        logger.info(f"✅ Prepared {len(df)} clean weather records with {len(numerical_cols)} features")
        return df

    def train_rf_model(self, weather_df):
        """Train Random Forest model for weather prediction"""
        logger.info("🌲 TRAINING RANDOM FOREST WEATHER MODEL...")
        
        try:
            # Create target variable based on flux intensity
            if 'flux' in weather_df.columns:
                # High flux indicates potential storm
                flux_threshold = weather_df['flux'].quantile(0.8)
                y = (weather_df['flux'] > flux_threshold).astype(int)
            else:
                # Use first numerical column as proxy
                col = weather_df.columns[0]
                threshold = weather_df[col].quantile(0.8)
                y = (weather_df[col] > threshold).astype(int)
            
            # Features (exclude target column)
            feature_cols = [col for col in weather_df.columns if col != 'flux']
            X = weather_df[feature_cols]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            
            rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = rf_model.score(X_train_scaled, y_train)
            test_score = rf_model.score(X_test_scaled, y_test)
            
            logger.info(f"📊 RF Model Performance:")
            logger.info(f"   Train Accuracy: {train_score:.3f}")
            logger.info(f"   Test Accuracy: {test_score:.3f}")
            
            # Save model and scaler
            rf_path = os.path.join(self.models_dir, "rf_model.pkl")
            scaler_path = os.path.join(self.models_dir, "rf_scaler.pkl")
            
            joblib.dump(rf_model, rf_path)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"✅ RF model saved: {rf_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ RF training failed: {e}")
            return False

    def train_anomaly_model(self, weather_df):
        """Train anomaly detection model"""
        logger.info("🔍 TRAINING ANOMALY DETECTION MODEL...")
        
        try:
            # Use all numerical features
            X = weather_df.select_dtypes(include=[np.number])
            
            if len(X.columns) < 2:
                logger.warning("⚠️ Not enough features for anomaly detection")
                return False
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Isolation Forest
            anomaly_model = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_estimators=100
            )
            
            anomaly_model.fit(X_scaled)
            
            # Test anomaly detection
            anomaly_scores = anomaly_model.decision_function(X_scaled)
            anomaly_pred = anomaly_model.predict(X_scaled)
            
            anomaly_count = np.sum(anomaly_pred == -1)
            normal_count = np.sum(anomaly_pred == 1)
            
            logger.info(f"📊 Anomaly Detection Results:")
            logger.info(f"   Normal samples: {normal_count}")
            logger.info(f"   Anomaly samples: {anomaly_count}")
            logger.info(f"   Anomaly rate: {anomaly_count/(anomaly_count+normal_count):.1%}")
            
            # Save model and scaler
            anomaly_path = os.path.join(self.models_dir, "anomaly_model.pkl")
            anomaly_scaler_path = os.path.join(self.models_dir, "anomaly_scaler.pkl")
            
            joblib.dump(anomaly_model, anomaly_path)
            joblib.dump(scaler, anomaly_scaler_path)
            
            logger.info(f"✅ Anomaly model saved: {anomaly_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Anomaly training failed: {e}")
            return False

    def train_all_models(self):
        """Train all missing AI models"""
        logger.info("🎯 STARTING COMPLETE AI MODEL TRAINING...")
        
        results = {
            'yolo': False,
            'rf_model': False,
            'anomaly_model': False
        }
        
        # 1. Train YOLO model
        logger.info("\n" + "="*60)
        logger.info("🔥 PHASE 1: YOLO SOLAR FLARE DETECTION MODEL")
        logger.info("="*60)
        
        yolo_dataset = self.prepare_yolo_dataset()
        if yolo_dataset:
            results['yolo'] = self.train_yolo_model(yolo_dataset)
        
        # 2. Prepare weather data for ML models
        logger.info("\n" + "="*60)
        logger.info("🌪️ PHASE 2: WEATHER DATA PREPARATION")
        logger.info("="*60)
        
        weather_df = self.prepare_weather_data()
        
        if weather_df is not None:
            # 3. Train Random Forest model
            logger.info("\n" + "="*60)
            logger.info("🌲 PHASE 3: RANDOM FOREST WEATHER PREDICTION")
            logger.info("="*60)
            
            results['rf_model'] = self.train_rf_model(weather_df)
            
            # 4. Train Anomaly Detection model
            logger.info("\n" + "="*60)
            logger.info("🔍 PHASE 4: ANOMALY DETECTION MODEL")
            logger.info("="*60)
            
            results['anomaly_model'] = self.train_anomaly_model(weather_df)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 TRAINING COMPLETE - SUMMARY")
        logger.info("="*60)
        
        success_count = sum(results.values())
        total_models = len(results)
        
        for model, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"   {model.upper()}: {status}")
        
        logger.info(f"\n🎯 OVERALL: {success_count}/{total_models} models trained successfully")
        
        if success_count == total_models:
            logger.info("🚀 ALL AI MODELS READY FOR FUSION AI SYSTEM!")
        elif success_count > 0:
            logger.info("⚠️ PARTIAL SUCCESS - Some models trained")
        else:
            logger.error("❌ TRAINING FAILED - No models trained")
        
        return results

def main():
    """Main training execution"""
    print("🤖 COMPREHENSIVE AI MODEL TRAINER")
    print("🌞 Training on ALL collected real space data")
    print("="*60)
    
    # Check data availability
    data_dir = "data/continuous_space_data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Initialize trainer
    trainer = ComprehensiveAITrainer(data_dir)
    
    # Train all models
    results = trainer.train_all_models()
    
    # Final status
    print("\n🎉 TRAINING SESSION COMPLETE!")
    print("Check the dashboard to see all AI models active!")

if __name__ == "__main__":
    main()