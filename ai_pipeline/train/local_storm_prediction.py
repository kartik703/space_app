#!/usr/bin/env python3
"""
Local Solar Storm Prediction Model using SQLite and synthetic/mock data
This script creates a local development version of the storm prediction model
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import argparse
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalStormPredictionModel:
    """Local storm prediction model using SQLite and synthetic data"""
    
    def __init__(self, db_path: str = "solar_storm_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.setup_database()
        
    def setup_database(self):
        """Create SQLite database and tables"""
        logger.info("🗄️ Setting up local SQLite database...")
        
        cursor = self.conn.cursor()
        
        # Create storm events table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS storm_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time TIMESTAMP,
            event_type TEXT,
            event_class TEXT,
            intensity REAL,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            duration_hours REAL,
            source TEXT,
            severity_score REAL
        )
        """)
        
        # Create solar metrics table for time series features
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS solar_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            solar_wind_speed REAL,
            magnetic_field_bt REAL,
            magnetic_field_bz REAL,
            proton_density REAL,
            kp_index REAL,
            dst_index REAL,
            solar_flux_f107 REAL,
            sunspot_number INTEGER
        )
        """)
        
        self.conn.commit()
        logger.info("✅ Database setup complete")
        
    def generate_synthetic_data(self, days_back: int = 365):
        """Generate synthetic solar storm and metrics data"""
        logger.info(f"🎲 Generating {days_back} days of synthetic solar data...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Generate storm events (approximately 1 storm every 3-7 days)
        storm_times = []
        current_time = start_time
        while current_time < end_time:
            # Random interval between storms
            days_until_next = np.random.exponential(4.5)  # Average 4.5 days
            current_time += timedelta(days=days_until_next)
            if current_time < end_time:
                storm_times.append(current_time)
        
        storm_events = []
        for i, storm_time in enumerate(storm_times):
            event_type = np.random.choice(['flare', 'cme', 'hss'], p=[0.4, 0.35, 0.25])
            event_class = np.random.choice(['C', 'M', 'X'], p=[0.7, 0.25, 0.05])
            
            # Intensity based on class
            if event_class == 'C':
                intensity = np.random.uniform(1.0, 9.9)
            elif event_class == 'M':
                intensity = np.random.uniform(1.0, 9.9) * 10
            else:  # X
                intensity = np.random.uniform(1.0, 28.0) * 100
            
            duration = np.random.exponential(6.0)  # Average 6 hours
            start_offset = timedelta(hours=np.random.uniform(-2, 0))
            end_offset = timedelta(hours=np.random.uniform(duration, duration + 4))
            
            severity_score = min(intensity / 10.0, 10.0)  # Scale to 0-10
            
            storm_events.append({
                'event_time': storm_time,
                'event_type': event_type,
                'event_class': event_class,
                'intensity': intensity,
                'start_time': storm_time + start_offset,
                'end_time': storm_time + end_offset,
                'duration_hours': duration,
                'source': 'synthetic',
                'severity_score': severity_score
            })
        
        # Insert storm events
        cursor = self.conn.cursor()
        for event in storm_events:
            cursor.execute("""
            INSERT INTO storm_events 
            (event_time, event_type, event_class, intensity, start_time, end_time, duration_hours, source, severity_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event['event_time'], event['event_type'], event['event_class'],
                event['intensity'], event['start_time'], event['end_time'],
                event['duration_hours'], event['source'], event['severity_score']
            ))
        
        # Generate hourly solar metrics
        logger.info("📊 Generating hourly solar wind and magnetic field data...")
        
        solar_metrics = []
        current_time = start_time
        while current_time < end_time:
            # Base values with daily variations
            base_kp = 2.0 + np.random.normal(0, 1.0)
            base_kp = max(0, min(9, base_kp))  # Clamp to valid Kp range
            
            # Solar wind speed (normal: 300-800 km/s, storms: 400-1200)
            is_storm_time = any(abs((current_time - storm['event_time']).total_seconds()) < 3600*12 
                              for storm in storm_events)
            
            if is_storm_time:
                sw_speed = np.random.normal(600, 200)
                bt_field = np.random.normal(15, 8)
                bz_field = np.random.normal(-5, 10)  # More negative during storms
                kp_boost = np.random.uniform(1, 4)
            else:
                sw_speed = np.random.normal(400, 100)
                bt_field = np.random.normal(8, 4)
                bz_field = np.random.normal(0, 5)
                kp_boost = 0
            
            sw_speed = max(200, sw_speed)  # Minimum realistic speed
            bt_field = max(0, bt_field)    # Magnetic field magnitude always positive
            
            solar_metrics.append({
                'timestamp': current_time,
                'solar_wind_speed': sw_speed,
                'magnetic_field_bt': bt_field,
                'magnetic_field_bz': bz_field,
                'proton_density': np.random.lognormal(1.5, 0.5),  # Typical 2-20 p/cm³
                'kp_index': min(9, base_kp + kp_boost),
                'dst_index': np.random.normal(-20, 30),  # Typical range
                'solar_flux_f107': np.random.normal(120, 40),  # Solar radio flux
                'sunspot_number': max(0, np.random.poisson(50))  # Sunspot activity
            })
            
            current_time += timedelta(hours=1)
        
        # Insert solar metrics in batches
        cursor = self.conn.cursor()
        for i in range(0, len(solar_metrics), 1000):
            batch = solar_metrics[i:i+1000]
            cursor.executemany("""
            INSERT INTO solar_metrics 
            (timestamp, solar_wind_speed, magnetic_field_bt, magnetic_field_bz, 
             proton_density, kp_index, dst_index, solar_flux_f107, sunspot_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(m['timestamp'], m['solar_wind_speed'], m['magnetic_field_bt'],
                   m['magnetic_field_bz'], m['proton_density'], m['kp_index'],
                   m['dst_index'], m['solar_flux_f107'], m['sunspot_number']) for m in batch])
        
        self.conn.commit()
        logger.info(f"✅ Generated {len(storm_events)} storm events and {len(solar_metrics)} hourly metrics")
        
    def prepare_time_series_features(self, forecast_hours: int = 24):
        """Prepare features for time series prediction"""
        logger.info(f"🔧 Preparing time series features for {forecast_hours}h forecasting...")
        
        # Load solar metrics
        metrics_df = pd.read_sql_query("""
        SELECT * FROM solar_metrics 
        ORDER BY timestamp
        """, self.conn)
        
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        metrics_df.set_index('timestamp', inplace=True)
        
        # Load storm events  
        storms_df = pd.read_sql_query("""
        SELECT event_time, severity_score, event_type, event_class, intensity
        FROM storm_events
        ORDER BY event_time
        """, self.conn)
        
        storms_df['event_time'] = pd.to_datetime(storms_df['event_time'])
        
        # Create target variables for next 6, 12, 24 hours
        features_list = []
        
        for i in range(len(metrics_df) - forecast_hours):
            current_time = metrics_df.index[i]
            future_time = current_time + timedelta(hours=forecast_hours)
            
            # Current and historical features (look back 12 hours)
            lookback_start = max(0, i - 12)
            current_features = {
                'timestamp': current_time,
                # Current values
                'sw_speed': metrics_df.iloc[i]['solar_wind_speed'],
                'bt_field': metrics_df.iloc[i]['magnetic_field_bt'], 
                'bz_field': metrics_df.iloc[i]['magnetic_field_bz'],
                'kp_index': metrics_df.iloc[i]['kp_index'],
                'dst_index': metrics_df.iloc[i]['dst_index'],
                'proton_density': metrics_df.iloc[i]['proton_density'],
                'solar_flux': metrics_df.iloc[i]['solar_flux_f107'],
                'sunspot_number': metrics_df.iloc[i]['sunspot_number'],
                
                # Statistical features over last 12 hours
                'sw_speed_mean_12h': metrics_df.iloc[lookback_start:i+1]['solar_wind_speed'].mean(),
                'sw_speed_max_12h': metrics_df.iloc[lookback_start:i+1]['solar_wind_speed'].max(),
                'bz_min_12h': metrics_df.iloc[lookback_start:i+1]['magnetic_field_bz'].min(),
                'kp_max_12h': metrics_df.iloc[lookback_start:i+1]['kp_index'].max(),
                'dst_min_12h': metrics_df.iloc[lookback_start:i+1]['dst_index'].min(),
                
                # Time features
                'hour_of_day': current_time.hour,
                'day_of_year': current_time.dayofyear,
                'month': current_time.month,
            }
            
            # Target: Will there be a storm in the next forecast_hours?
            future_storms = storms_df[
                (storms_df['event_time'] > current_time) & 
                (storms_df['event_time'] <= future_time)
            ]
            
            current_features['storm_in_future'] = len(future_storms) > 0
            current_features['max_future_severity'] = future_storms['severity_score'].max() if len(future_storms) > 0 else 0
            current_features['future_storm_type'] = future_storms['event_type'].iloc[0] if len(future_storms) > 0 else 'none'
            
            features_list.append(current_features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Created {len(features_df)} feature vectors")
        
        return features_df
        
    def train_models(self, features_df: pd.DataFrame, output_dir: str = "models"):
        """Train prediction models"""
        logger.info("🤖 Training storm prediction models...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare features
        feature_cols = [col for col in features_df.columns 
                       if col not in ['timestamp', 'storm_in_future', 'max_future_severity', 'future_storm_type']]
        
        X = features_df[feature_cols].copy()
        
        # Handle categorical features
        storm_type_encoder = LabelEncoder()
        if 'future_storm_type' in features_df.columns:
            y_type = storm_type_encoder.fit_transform(features_df['future_storm_type'])
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {}
        
        # 1. Binary storm classification
        y_binary = features_df['storm_in_future'].astype(int)
        X_train, X_test, y_train_bin, y_test_bin = train_test_split(
            X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Train binary classifier
        logger.info("🎯 Training binary storm classifier...")
        binary_classifier = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
        binary_classifier.fit(X_train, y_train_bin)
        
        # Evaluate binary model
        y_pred_bin = binary_classifier.predict(X_test)
        bin_accuracy = accuracy_score(y_test_bin, y_pred_bin)
        
        logger.info(f"📊 Binary classifier accuracy: {bin_accuracy:.3f}")
        logger.info("Binary Classification Report:")
        print(classification_report(y_test_bin, y_pred_bin, target_names=['No Storm', 'Storm']))
        
        # 2. Storm severity regression (for positive cases only)
        storm_mask = features_df['storm_in_future'] == True
        if storm_mask.sum() > 10:  # Only if we have enough storm samples
            X_severity = X_scaled[storm_mask]
            y_severity = features_df.loc[storm_mask, 'max_future_severity']
            
            if len(X_severity) > 5:
                X_sev_train, X_sev_test, y_sev_train, y_sev_test = train_test_split(
                    X_severity, y_severity, test_size=0.3, random_state=42
                )
                
                logger.info("📈 Training storm severity regressor...")
                severity_regressor = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42
                )
                severity_regressor.fit(X_sev_train, y_sev_train)
                
                # Evaluate severity model
                y_sev_pred = severity_regressor.predict(X_sev_test)
                sev_rmse = np.sqrt(mean_squared_error(y_sev_test, y_sev_pred))
                
                logger.info(f"📊 Severity regressor RMSE: {sev_rmse:.3f}")
                
                results['severity_regressor'] = severity_regressor
                results['severity_rmse'] = sev_rmse
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': binary_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("🔍 Top 10 most important features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Save models
        models_to_save = {
            'binary_classifier': binary_classifier,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'storm_type_encoder': storm_type_encoder
        }
        
        if 'severity_regressor' in results:
            models_to_save['severity_regressor'] = results['severity_regressor']
        
        model_path = os.path.join(output_dir, 'storm_prediction_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(models_to_save, f)
        
        # Save feature importance
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        
        results.update({
            'binary_classifier': binary_classifier,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'binary_accuracy': bin_accuracy,
            'feature_importance': feature_importance,
            'model_path': model_path
        })
        
        logger.info(f"✅ Models saved to {model_path}")
        return results
        
    def predict_storms(self, model_path: str, hours_ahead: int = 24):
        """Make storm predictions using trained model"""
        logger.info(f"🔮 Making storm predictions for next {hours_ahead} hours...")
        
        # Load model
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
        
        # Get latest data point for prediction
        latest_df = pd.read_sql_query("""
        SELECT * FROM solar_metrics 
        ORDER BY timestamp DESC 
        LIMIT 1
        """, self.conn)
        
        if len(latest_df) == 0:
            logger.warning("No data available for prediction")
            return None
        
        # Prepare features (simplified for demo)
        latest_time = pd.to_datetime(latest_df.iloc[0]['timestamp'])
        
        # Get recent data for statistical features
        recent_df = pd.read_sql_query(f"""
        SELECT * FROM solar_metrics 
        WHERE timestamp >= datetime('{latest_time - timedelta(hours=12)}')
        ORDER BY timestamp
        """, self.conn)
        
        if len(recent_df) < 5:
            logger.warning("Insufficient recent data for prediction")
            return None
        
        # Create feature vector
        features = {
            'sw_speed': latest_df.iloc[0]['solar_wind_speed'],
            'bt_field': latest_df.iloc[0]['magnetic_field_bt'],
            'bz_field': latest_df.iloc[0]['magnetic_field_bz'],
            'kp_index': latest_df.iloc[0]['kp_index'],
            'dst_index': latest_df.iloc[0]['dst_index'],
            'proton_density': latest_df.iloc[0]['proton_density'],
            'solar_flux': latest_df.iloc[0]['solar_flux_f107'],
            'sunspot_number': latest_df.iloc[0]['sunspot_number'],
            
            'sw_speed_mean_12h': recent_df['solar_wind_speed'].mean(),
            'sw_speed_max_12h': recent_df['solar_wind_speed'].max(),
            'bz_min_12h': recent_df['magnetic_field_bz'].min(),
            'kp_max_12h': recent_df['kp_index'].max(),
            'dst_min_12h': recent_df['dst_index'].min(),
            
            'hour_of_day': latest_time.hour,
            'day_of_year': latest_time.dayofyear,
            'month': latest_time.month,
        }
        
        # Create feature vector
        X_pred = pd.DataFrame([features])[models['feature_cols']]
        X_pred_scaled = models['scaler'].transform(X_pred)
        
        # Make predictions
        storm_prob = models['binary_classifier'].predict_proba(X_pred_scaled)[0, 1]
        storm_prediction = models['binary_classifier'].predict(X_pred_scaled)[0]
        
        result = {
            'timestamp': latest_time,
            'storm_probability': float(storm_prob),
            'storm_predicted': bool(storm_prediction),
            'forecast_horizon_hours': hours_ahead,
            'current_conditions': features
        }
        
        # If storm is predicted and we have severity model
        if storm_prediction and 'severity_regressor' in models:
            predicted_severity = models['severity_regressor'].predict(X_pred_scaled)[0]
            result['predicted_severity'] = float(predicted_severity)
        
        logger.info(f"🎯 Storm prediction: {storm_prediction} (probability: {storm_prob:.3f})")
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Local Solar Storm Prediction Model')
    parser.add_argument('--db-path', default='solar_storm_data.db', help='SQLite database path')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--train', action='store_true', help='Train prediction models')
    parser.add_argument('--predict', action='store_true', help='Make storm predictions')
    parser.add_argument('--forecast-hours', type=int, default=24, help='Forecast horizon in hours')
    parser.add_argument('--output-dir', default='models', help='Output directory for models')
    
    args = parser.parse_args()
    
    # Initialize model
    predictor = LocalStormPredictionModel(args.db_path)
    
    if args.generate_data:
        predictor.generate_synthetic_data(days_back=365)
    
    if args.train:
        features_df = predictor.prepare_time_series_features(args.forecast_hours)
        results = predictor.train_models(features_df, args.output_dir)
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📈 Binary classifier accuracy: {results['binary_accuracy']:.3f}")
        
    if args.predict:
        model_path = os.path.join(args.output_dir, 'storm_prediction_model.pkl')
        if os.path.exists(model_path):
            prediction = predictor.predict_storms(model_path, args.forecast_hours)
            if prediction:
                logger.info("🔮 Latest Prediction:")
                for key, value in prediction.items():
                    if key != 'current_conditions':
                        logger.info(f"  {key}: {value}")
        else:
            logger.error(f"Model not found at {model_path}. Please train first.")

if __name__ == "__main__":
    main()