#!/usr/bin/env python3
"""
Solar Storm Time-Series Prediction Model
Uses real storm events to predict when solar storms will occur
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from typing import Dict, List, Tuple
from pathlib import Path

# Machine learning libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Google Cloud
from google.cloud import bigquery
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StormPredictionModel:
    """Time-series model to predict solar storm occurrences"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        logger.info(f"Initialized StormPredictionModel for project {project_id}")
    
    def load_storm_data(self) -> pd.DataFrame:
        """Load storm events from BigQuery"""
        logger.info("📊 Loading storm events from BigQuery...")
        
        query = """
        SELECT 
            COALESCE(peak_time, start_time) as event_time,
            event_type,
            event_class,
            intensity as magnitude,
            start_time,
            end_time,
            TIMESTAMP_DIFF(end_time, start_time, HOUR) as duration_hours,
            source,
            severity_score,
            EXTRACT(HOUR FROM COALESCE(peak_time, start_time)) as hour_of_day,
            EXTRACT(DAYOFWEEK FROM COALESCE(peak_time, start_time)) as day_of_week,
            EXTRACT(MONTH FROM COALESCE(peak_time, start_time)) as month,
            EXTRACT(YEAR FROM COALESCE(peak_time, start_time)) as year
        FROM `{}.space_weather.solar_storm_events_v2`
        WHERE COALESCE(peak_time, start_time) IS NOT NULL
        ORDER BY COALESCE(peak_time, start_time)
        """.format(self.project_id)
        
        df = self.client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} storm events")
        
        return df
    
    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-series features for prediction"""
        logger.info("🔧 Creating time-series features...")
        
        # Sort by time
        df = df.sort_values('event_time').reset_index(drop=True)
        
        # Create hourly time series with storm activity
        start_time = df['event_time'].min().replace(minute=0, second=0, microsecond=0)
        end_time = df['event_time'].max().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        # Create hourly time grid
        time_range = pd.date_range(start=start_time, end=end_time, freq='H')
        ts_df = pd.DataFrame({'time': time_range})
        
        # For each hour, check if there was storm activity
        ts_df['has_storm'] = 0
        ts_df['storm_count'] = 0
        ts_df['max_magnitude'] = 0.0
        ts_df['storm_types'] = ''
        
        for idx, row in df.iterrows():
            # Find the hour this event belongs to
            event_hour = row['event_time'].replace(minute=0, second=0, microsecond=0)
            
            # Update the time series data
            mask = ts_df['time'] == event_hour
            if mask.any():
                ts_df.loc[mask, 'has_storm'] = 1
                ts_df.loc[mask, 'storm_count'] += 1
                ts_df.loc[mask, 'max_magnitude'] = max(ts_df.loc[mask, 'max_magnitude'].iloc[0], 
                                                     row['magnitude'] if pd.notna(row['magnitude']) else 0)
                current_types = ts_df.loc[mask, 'storm_types'].iloc[0]
                if current_types:
                    ts_df.loc[mask, 'storm_types'] = current_types + ',' + str(row['event_type'])
                else:
                    ts_df.loc[mask, 'storm_types'] = str(row['event_type'])
        
        # Add temporal features
        ts_df['hour'] = ts_df['time'].dt.hour
        ts_df['day_of_week'] = ts_df['time'].dt.dayofweek
        ts_df['day_of_year'] = ts_df['time'].dt.dayofyear
        ts_df['month'] = ts_df['time'].dt.month
        ts_df['year'] = ts_df['time'].dt.year
        
        # Add lag features (storm activity in previous hours)
        for lag in [1, 6, 12, 24, 48]:
            ts_df[f'storm_lag_{lag}h'] = ts_df['has_storm'].shift(lag).fillna(0)
            ts_df[f'count_lag_{lag}h'] = ts_df['storm_count'].shift(lag).fillna(0)
        
        # Add rolling window features
        for window in [6, 12, 24]:
            ts_df[f'storm_rolling_{window}h'] = ts_df['has_storm'].rolling(window).sum().fillna(0)
            ts_df[f'count_rolling_{window}h'] = ts_df['storm_count'].rolling(window).sum().fillna(0)
        
        # Add time since last storm
        storm_times = ts_df[ts_df['has_storm'] == 1]['time']
        ts_df['hours_since_last_storm'] = 0
        
        for i, current_time in enumerate(ts_df['time']):
            prev_storms = storm_times[storm_times < current_time]
            if len(prev_storms) > 0:
                last_storm = prev_storms.max()
                ts_df.loc[i, 'hours_since_last_storm'] = (current_time - last_storm).total_seconds() / 3600
            else:
                ts_df.loc[i, 'hours_since_last_storm'] = 999  # No previous storms
        
        logger.info(f"Created time series with {len(ts_df)} hourly records")
        logger.info(f"Storm activity: {ts_df['has_storm'].sum()} hours with storms out of {len(ts_df)} total hours")
        
        return ts_df
    
    def prepare_features(self, ts_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for model training"""
        logger.info("🎯 Preparing features and target...")
        
        # Feature columns (exclude target and time)
        feature_cols = [col for col in ts_df.columns if col not in ['time', 'has_storm', 'storm_types']]
        self.feature_columns = feature_cols
        
        X = ts_df[feature_cols].values
        y = ts_df['has_storm'].values
        
        # Handle any infinite or NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Positive class ratio: {y.mean():.3f} ({y.sum()} storms / {len(y)} hours)")
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train multiple models and compare performance"""
        logger.info("🚀 Training prediction models...")
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Define models to try
        models_to_try = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=10,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        }
        
        results = {}
        
        for model_name, model in models_to_try.items():
            logger.info(f"Training {model_name}...")
            
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                # Scale features if using logistic regression
                if model_name == 'logistic_regression':
                    scaler = StandardScaler()
                    X_train_cv = scaler.fit_transform(X_train_cv)
                    X_val_cv = scaler.transform(X_val_cv)
                    self.scalers[model_name] = scaler
                
                # Train and evaluate
                model.fit(X_train_cv, y_train_cv)
                score = model.score(X_val_cv, y_val_cv)
                cv_scores.append(score)
            
            avg_score = np.mean(cv_scores)
            logger.info(f"{model_name} CV accuracy: {avg_score:.3f} ± {np.std(cv_scores):.3f}")
            
            # Train final model on all data
            if model_name == 'logistic_regression':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)
                self.scalers[model_name] = scaler
            else:
                model.fit(X, y)
            
            self.models[model_name] = model
            results[model_name] = {
                'model': model,
                'cv_score': avg_score,
                'cv_std': np.std(cv_scores)
            }
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_score'])
        self.best_model_name = best_model_name
        
        logger.info(f"🏆 Best model: {best_model_name} (CV accuracy: {results[best_model_name]['cv_score']:.3f})")
        
        return results
    
    def analyze_feature_importance(self):
        """Analyze feature importance for interpretation"""
        logger.info("🔍 Analyzing feature importance...")
        
        best_model = self.models[self.best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 most important features:")
            for _, row in feature_importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.3f}")
            
            return feature_importance_df
        else:
            logger.info("Feature importance not available for this model type")
            return None
    
    def predict_storm_probability(self, current_features: Dict) -> float:
        """Predict probability of storm in the next hour"""
        if not self.models or not self.feature_columns:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        # Prepare features
        feature_array = np.array([current_features.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
        
        # Scale if needed
        if self.best_model_name in self.scalers:
            feature_array = self.scalers[self.best_model_name].transform(feature_array)
        
        # Predict
        best_model = self.models[self.best_model_name]
        if hasattr(best_model, 'predict_proba'):
            prob = best_model.predict_proba(feature_array)[0, 1]
        else:
            prob = best_model.decision_function(feature_array)[0]
        
        return float(prob)
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'best_model_name': self.best_model_name,
            'project_id': self.project_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_columns = model_data['feature_columns']
        self.best_model_name = model_data['best_model_name']
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Solar Storm Prediction Model")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--output-dir", default="models", help="Output directory for model")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    predictor = StormPredictionModel(args.project_id)
    
    # Load and process data
    storm_df = predictor.load_storm_data()
    ts_df = predictor.create_time_series_features(storm_df)
    X, y = predictor.prepare_features(ts_df)
    
    # Train models
    results = predictor.train_models(X, y)
    
    # Analyze feature importance
    feature_importance = predictor.analyze_feature_importance()
    
    # Save model
    model_path = Path(args.output_dir) / "storm_prediction_model.pkl"
    predictor.save_model(str(model_path))
    
    # Save feature importance if available
    if feature_importance is not None:
        importance_path = Path(args.output_dir) / "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
    
    # Test prediction
    logger.info("🧪 Testing prediction capability...")
    test_features = {col: 0 for col in predictor.feature_columns}
    test_features['hour'] = 12  # Noon
    test_features['storm_lag_24h'] = 1  # Storm 24 hours ago
    
    prob = predictor.predict_storm_probability(test_features)
    logger.info(f"Test prediction (storm 24h ago, noon): {prob:.3f} probability")
    
    logger.info("✅ Storm prediction model training complete!")


if __name__ == "__main__":
    main()