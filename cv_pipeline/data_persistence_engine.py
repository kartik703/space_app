#!/usr/bin/env python3
"""
Data Persistence Engine for Space Weather Analytics
Automatically saves live data, builds datasets, and trains predictive models
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pickle
import json
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import asyncio
from threading import Thread
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpaceWeatherDatabase:
    """Database manager for space weather data"""
    
    def __init__(self, db_path: str = "data/space_weather_live.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Live data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS live_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        parameter VARCHAR(50),
                        value REAL,
                        unit VARCHAR(20),
                        quality VARCHAR(10),
                        source VARCHAR(50),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Predictions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prediction_time DATETIME,
                        target_time DATETIME,
                        parameter VARCHAR(50),
                        predicted_value REAL,
                        confidence REAL,
                        model_name VARCHAR(100),
                        actual_value REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Model performance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name VARCHAR(100),
                        parameter VARCHAR(50),
                        training_date DATETIME,
                        rmse REAL,
                        r2_score REAL,
                        data_points INTEGER,
                        model_path VARCHAR(200),
                        hyperparameters TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS space_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_time DATETIME,
                        event_type VARCHAR(50),
                        severity VARCHAR(20),
                        parameter VARCHAR(50),
                        peak_value REAL,
                        duration_hours REAL,
                        description TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_live_data(self, data_points: List[Dict]):
        """Save live data points to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for point in data_points:
                    # Convert timestamp to string if it's a pandas Timestamp
                    timestamp = point['timestamp']
                    if hasattr(timestamp, 'strftime'):
                        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    
                    cursor.execute("""
                        INSERT INTO live_data (timestamp, parameter, value, unit, quality, source)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        point['parameter'],
                        point['value'],
                        point['unit'],
                        point.get('quality', 'GOOD'),
                        point.get('source', 'NOAA_SWPC')
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(data_points)} live data points")
                
        except Exception as e:
            logger.error(f"Failed to save live data: {e}")
    
    def get_historical_data(self, parameter: str, hours_back: int = 168) -> pd.DataFrame:
        """Get historical data for a parameter"""
        try:
            query = """
                SELECT timestamp, value, quality
                FROM live_data
                WHERE parameter = ? AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp
            """.format(hours_back)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[parameter])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            logger.info(f"Retrieved {len(df)} historical records for {parameter}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {parameter}: {e}")
            return pd.DataFrame()
    
    def save_prediction(self, prediction_data: Dict):
        """Save model prediction to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO predictions (prediction_time, target_time, parameter, 
                                           predicted_value, confidence, model_name)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    prediction_data['prediction_time'],
                    prediction_data['target_time'],
                    prediction_data['parameter'],
                    prediction_data['predicted_value'],
                    prediction_data['confidence'],
                    prediction_data['model_name']
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
    
    def get_data_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Data points by parameter
                cursor.execute("""
                    SELECT parameter, COUNT(*) as count, 
                           MIN(timestamp) as first_record,
                           MAX(timestamp) as last_record
                    FROM live_data
                    GROUP BY parameter
                """)
                
                stats['data_points'] = {}
                for row in cursor.fetchall():
                    stats['data_points'][row[0]] = {
                        'count': row[1],
                        'first_record': row[2],
                        'last_record': row[3]
                    }
                
                # Total records
                cursor.execute("SELECT COUNT(*) FROM live_data")
                stats['total_records'] = cursor.fetchone()[0]
                
                # Recent predictions
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions 
                    WHERE prediction_time > datetime('now', '-24 hours')
                """)
                stats['recent_predictions'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

class PredictiveModelEngine:
    """Engine for training and managing predictive models"""
    
    def __init__(self, db: SpaceWeatherDatabase):
        """Initialize model engine"""
        self.db = db
        self.models = {}
        self.scalers = {}
        self.model_dir = Path("models/predictive")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_training_data(self, parameter: str, look_back: int = 24, 
                            forecast_hours: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for training"""
        try:
            # Get historical data
            df = self.db.get_historical_data(parameter, hours_back=168)
            
            if len(df) < look_back + forecast_hours:
                logger.warning(f"Insufficient data for {parameter}: {len(df)} records")
                return None, None
            
            # Create sequences
            values = df['value'].values
            X, y = [], []
            
            for i in range(len(values) - look_back - forecast_hours + 1):
                X.append(values[i:(i + look_back)])
                y.append(values[i + look_back + forecast_hours - 1])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Failed to prepare training data for {parameter}: {e}")
            return None, None
    
    def train_random_forest_model(self, parameter: str) -> Dict:
        """Train Random Forest model for parameter prediction"""
        try:
            logger.info(f"Training Random Forest model for {parameter}")
            
            X, y = self.prepare_training_data(parameter)
            if X is None or len(X) < 50:
                logger.warning(f"Insufficient data for training {parameter} model")
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Save model
            model_name = f"rf_{parameter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.model_dir / f"{model_name}.pkl"
            scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Store in memory
            self.models[parameter] = {
                'model': model,
                'scaler': scaler,
                'type': 'random_forest',
                'trained_at': datetime.now(),
                'rmse': rmse,
                'r2': r2
            }
            
            # Save performance to database
            self.save_model_performance(model_name, parameter, rmse, r2, len(X), str(model_path))
            
            logger.info(f"Trained {parameter} model - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            return {
                'model_name': model_name,
                'rmse': rmse,
                'r2': r2,
                'data_points': len(X)
            }
            
        except Exception as e:
            logger.error(f"Failed to train Random Forest model for {parameter}: {e}")
            return None
    
    def train_gradient_boosting_model(self, parameter: str) -> Dict:
        """Train Gradient Boosting model for parameter prediction"""
        try:
            logger.info(f"Training Gradient Boosting model for {parameter}")
            
            X, y = self.prepare_training_data(parameter)
            if X is None or len(X) < 50:
                logger.warning(f"Insufficient data for training {parameter} model")
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Save model
            model_name = f"gb_{parameter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.model_dir / f"{model_name}.pkl"
            scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Store in memory
            self.models[parameter] = {
                'model': model,
                'scaler': scaler,
                'type': 'gradient_boosting',
                'trained_at': datetime.now(),
                'rmse': rmse,
                'r2': r2
            }
            
            # Save performance to database
            self.save_model_performance(model_name, parameter, rmse, r2, len(X), str(model_path))
            
            logger.info(f"Trained {parameter} model - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            return {
                'model_name': model_name,
                'rmse': rmse,
                'r2': r2,
                'data_points': len(X)
            }
            
        except Exception as e:
            logger.error(f"Failed to train Gradient Boosting model for {parameter}: {e}")
            return None
    
    def make_prediction(self, parameter: str, forecast_hours: int = 6) -> Optional[Dict]:
        """Make prediction for a parameter"""
        try:
            if parameter not in self.models:
                logger.warning(f"No model available for {parameter}")
                return None
            
            # Get recent data
            df = self.db.get_historical_data(parameter, hours_back=24)
            if len(df) < 24:
                logger.warning(f"Insufficient recent data for {parameter} prediction")
                return None
            
            # Prepare input
            recent_values = df['value'].values[-24:]
            X = recent_values.reshape(1, -1)
            
            # Scale and predict
            model_info = self.models[parameter]
            X_scaled = model_info['scaler'].transform(X)
            prediction = model_info['model'].predict(X_scaled)[0]
            
            # Calculate confidence (simplified)
            confidence = min(0.95, max(0.5, model_info['r2']))
            
            prediction_data = {
                'prediction_time': datetime.now(),
                'target_time': datetime.now() + timedelta(hours=forecast_hours),
                'parameter': parameter,
                'predicted_value': float(prediction),
                'confidence': confidence,
                'model_name': f"{model_info['type']}_{parameter}"
            }
            
            # Save prediction
            self.db.save_prediction(prediction_data)
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Failed to make prediction for {parameter}: {e}")
            return None
    
    def save_model_performance(self, model_name: str, parameter: str, rmse: float, 
                             r2: float, data_points: int, model_path: str):
        """Save model performance to database"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO model_performance (model_name, parameter, training_date,
                                                 rmse, r2_score, data_points, model_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (model_name, parameter, datetime.now(), rmse, r2, data_points, model_path))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save model performance: {e}")

class AutoMLPipeline:
    """Automated Machine Learning Pipeline for Space Weather"""
    
    def __init__(self):
        """Initialize AutoML pipeline"""
        self.db = SpaceWeatherDatabase()
        self.model_engine = PredictiveModelEngine(self.db)
        self.is_running = False
        self.training_thread = None
    
    def start_auto_training(self, retrain_interval_hours: int = 24):
        """Start automatic model retraining"""
        self.is_running = True
        self.training_thread = Thread(target=self._auto_training_loop, 
                                    args=(retrain_interval_hours,))
        self.training_thread.daemon = True
        self.training_thread.start()
        logger.info("Started automatic model training pipeline")
    
    def stop_auto_training(self):
        """Stop automatic model training"""
        self.is_running = False
        if self.training_thread:
            self.training_thread.join()
        logger.info("Stopped automatic model training pipeline")
    
    def _auto_training_loop(self, retrain_interval_hours: int):
        """Main auto-training loop"""
        parameters = ['kp_index', 'solar_wind_speed', 'proton_flux']
        
        while self.is_running:
            try:
                logger.info("Starting automatic model training cycle")
                
                for parameter in parameters:
                    # Train both Random Forest and Gradient Boosting
                    rf_result = self.model_engine.train_random_forest_model(parameter)
                    gb_result = self.model_engine.train_gradient_boosting_model(parameter)
                    
                    if rf_result:
                        logger.info(f"RF {parameter}: RMSE={rf_result['rmse']:.4f}")
                    if gb_result:
                        logger.info(f"GB {parameter}: RMSE={gb_result['rmse']:.4f}")
                
                logger.info("Completed automatic model training cycle")
                
                # Wait for next training cycle
                time.sleep(retrain_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in auto-training loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def save_live_data_batch(self, live_data: Dict):
        """Save batch of live data points"""
        try:
            data_points = []
            
            for parameter, data in live_data.items():
                if isinstance(data, dict) and 'value' in data:
                    data_points.append({
                        'timestamp': data.get('timestamp', datetime.now()),
                        'parameter': parameter,
                        'value': data['value'],
                        'unit': data.get('unit', ''),
                        'quality': 'GOOD',
                        'source': 'LIVE_DASHBOARD'
                    })
            
            if data_points:
                self.db.save_live_data(data_points)
                logger.info(f"Saved {len(data_points)} live data points to database")
            
        except Exception as e:
            logger.error(f"Failed to save live data batch: {e}")
    
    def get_predictions_for_dashboard(self) -> Dict:
        """Get current predictions for dashboard display"""
        try:
            predictions = {}
            parameters = ['kp_index', 'solar_wind_speed', 'proton_flux']
            
            for parameter in parameters:
                prediction = self.model_engine.make_prediction(parameter)
                if prediction:
                    predictions[parameter] = prediction
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get predictions for dashboard: {e}")
            return {}
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        try:
            stats = self.db.get_data_stats()
            
            # Model status
            model_status = {}
            for param in ['kp_index', 'solar_wind_speed', 'proton_flux']:
                if param in self.model_engine.models:
                    model_info = self.model_engine.models[param]
                    model_status[param] = {
                        'available': True,
                        'type': model_info['type'],
                        'trained_at': model_info['trained_at'].isoformat(),
                        'rmse': model_info['rmse'],
                        'r2': model_info['r2']
                    }
                else:
                    model_status[param] = {'available': False}
            
            return {
                'database_stats': stats,
                'model_status': model_status,
                'auto_training_active': self.is_running,
                'system_healthy': len(stats.get('data_points', {})) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {}

# Global instance
automl_pipeline = AutoMLPipeline()

def initialize_automl_system():
    """Initialize the AutoML system"""
    try:
        logger.info("Initializing AutoML pipeline for space weather prediction")
        automl_pipeline.start_auto_training(retrain_interval_hours=6)  # Retrain every 6 hours
        return automl_pipeline
    except Exception as e:
        logger.error(f"Failed to initialize AutoML system: {e}")
        return None