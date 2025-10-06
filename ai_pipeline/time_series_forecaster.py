"""
Time-Series AI for Space Weather Forecasting
Predicts: Kp index, Dst, AE, solar wind parameters, satellite drag
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import warnings
from .data_source_manager import DataSourceManager
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForecastResult:
    """Forecast result structure"""
    parameter: str
    forecast_values: List[float]
    forecast_times: List[datetime.datetime]
    confidence_intervals: List[Tuple[float, float]]
    model_accuracy: float
    risk_level: str
    metadata: Dict

class LSTMForecaster(nn.Module):
    """LSTM neural network for time-series forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last time step for prediction
        output = self.fc_layers(attn_out[:, -1, :])
        
        return output

class TransformerForecaster(nn.Module):
    """Transformer model for time-series forecasting"""
    
    def __init__(self, input_size: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, output_size: int = 1, dropout: float = 0.1):
        super(TransformerForecaster, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        transformer_out = self.transformer(x)
        
        # Global average pooling and output projection
        output = self.output_projection(transformer_out.mean(dim=1))
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))
        
    def forward(self, x):
        # Get positional encoding with proper indexing
        seq_len = x.size(1)  # sequence length dimension
        pos_enc = self.pe[:seq_len].transpose(0, 1)  # [seq_len, 1, d_model] -> [1, seq_len, d_model]
        x = x + pos_enc
        return self.dropout(x)

class SpaceWeatherForecaster:
    """
    Advanced time-series forecasting system for space weather parameters
    """
    
    def __init__(self, models_dir: str = "models/forecasting"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.sequence_length = 168  # 7 days of hourly data
        self.forecast_horizon = 72  # 3 days ahead
        
        # Initialize data source manager
        self.data_manager = DataSourceManager()
        
        # Parameter configurations
        self.parameters = {
            'kp_index': {
                'url': 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json',
                'range': (0, 9),
                'risk_thresholds': {'low': 3, 'medium': 5, 'high': 7, 'extreme': 8}
            },
            'dst_index': {
                'url': 'https://services.swpc.noaa.gov/products/kyoto-dst.json',
                'range': (-500, 50),
                'risk_thresholds': {'low': -30, 'medium': -50, 'high': -100, 'extreme': -200}
            },
            'solar_wind_speed': {
                'url': 'https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json',
                'range': (200, 1000),
                'risk_thresholds': {'low': 500, 'medium': 600, 'high': 700, 'extreme': 800}
            },
            'proton_flux': {
                'url': 'https://services.swpc.noaa.gov/products/goes-proton-flux.json',
                'range': (0.01, 10000),
                'risk_thresholds': {'low': 1, 'medium': 10, 'high': 100, 'extreme': 1000}
            }
        }
        
        self.load_models()
    
    def fetch_historical_data(self, parameter: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical space weather data"""
        try:
            config = self.parameters.get(parameter)
            if not config:
                raise ValueError(f"Unknown parameter: {parameter}")
            
            # Fetch from NOAA SWPC
            response = requests.get(config['url'], timeout=10)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data[1:], columns=data[0])  # Skip header row
            
            # Convert time column to datetime
            time_col = df.columns[0]
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
            
            # Convert value columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill missing values
            df = df.ffill().bfill()
            
            # Filter to requested time period
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            df = df[start_date:end_date]
            
            logger.info(f"Fetched {len(df)} records for {parameter}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {parameter}: {e}")
            # Return empty DataFrame instead of synthetic data - REAL DATA ONLY
            return pd.DataFrame()
    
    def _generate_synthetic_data(self, parameter: str, days: int) -> pd.DataFrame:
        """Generate synthetic data for testing"""
        config = self.parameters[parameter]
        min_val, max_val = config['range']
        
        # Generate time series
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=days)
        times = pd.date_range(start_time, end_time, freq='H')
        
        # Generate values with realistic patterns
        np.random.seed(42)
        trend = np.linspace(min_val, max_val, len(times)) * 0.1
        seasonal = np.sin(2 * np.pi * np.arange(len(times)) / 24) * (max_val - min_val) * 0.2
        noise = np.random.normal(0, (max_val - min_val) * 0.1, len(times))
        values = (min_val + max_val) / 2 + trend + seasonal + noise
        
        # Clip to valid range
        values = np.clip(values, min_val, max_val)
        
        df = pd.DataFrame({'value': values}, index=times)
        logger.info(f"Generated synthetic data for {parameter}: {len(df)} records")
        return df
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str = 'value') -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare sequences for training"""
        values = np.array(data[target_col].values).reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_values) - self.forecast_horizon + 1):
            X.append(scaled_values[i-self.sequence_length:i])
            y.append(scaled_values[i:i+self.forecast_horizon])
        
        return np.array(X), np.array(y), scaler
    
    def train_model(self, parameter: str, model_type: str = 'lstm', epochs: int = 100) -> Dict:
        """Train forecasting model for a specific parameter"""
        logger.info(f"Training {model_type} model for {parameter}")
        
        # Fetch and prepare data
        data = self.fetch_historical_data(parameter, days=90)
        
        # Determine the correct column name
        if 'value' in data.columns:
            target_col = 'value'
        else:
            # Find first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError(f"No numeric columns found in data for {parameter}")
            target_col = numeric_cols[0]
        
        X, y, scaler = self.prepare_sequences(data, target_col=target_col)
        
        if len(X) == 0:
            raise ValueError(f"Insufficient data for training {parameter}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Split train/validation
        split_idx = int(0.8 * len(X_tensor))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        # Initialize model
        if model_type == 'lstm':
            model = LSTMForecaster(
                input_size=X.shape[2],
                hidden_size=128,
                num_layers=3,
                output_size=self.forecast_horizon
            )
        elif model_type == 'transformer':
            model = TransformerForecaster(
                input_size=X.shape[2],
                d_model=256,
                nhead=8,
                num_layers=6,
                output_size=self.forecast_horizon
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for i in range(0, len(X_train), 32):  # Batch size 32
                batch_X = X_train[i:i+32]
                batch_y = y_train[i:i+32]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val.squeeze()).item()
            
            train_losses.append(train_loss / len(X_train))
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_path = self.models_dir / f"{parameter}_{model_type}_best.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': scaler,
                    'model_type': model_type,
                    'parameter': parameter,
                    'sequence_length': self.sequence_length,
                    'forecast_horizon': self.forecast_horizon
                }, model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.6f}, Val Loss = {val_losses[-1]:.6f}")
        
        # Store model and scaler
        self.models[parameter] = model
        self.scalers[parameter] = scaler
        
        # Calculate accuracy metrics
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).numpy()
            val_true = y_val.squeeze().numpy()
            
            # Inverse scale for metrics
            val_pred_orig = scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
            val_true_orig = scaler.inverse_transform(val_true.reshape(-1, 1)).flatten()
            
            mae = mean_absolute_error(val_true_orig, val_pred_orig)
            rmse = np.sqrt(mean_squared_error(val_true_orig, val_pred_orig))
            mape = np.mean(np.abs((val_true_orig - val_pred_orig) / val_true_orig)) * 100
        
        training_result = {
            'parameter': parameter,
            'model_type': model_type,
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        logger.info(f"Training completed for {parameter}: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.1f}%")
        return training_result
    
    def load_models(self):
        """Load pre-trained models"""
        for parameter in self.parameters.keys():
            for model_type in ['lstm', 'transformer']:
                model_path = self.models_dir / f"{parameter}_{model_type}_best.pt"
                if model_path.exists():
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        
                        if model_type == 'lstm':
                            model = LSTMForecaster(
                                input_size=1,
                                output_size=checkpoint['forecast_horizon']
                            )
                        else:
                            model = TransformerForecaster(
                                input_size=1,
                                output_size=checkpoint['forecast_horizon']
                            )
                        
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()
                        
                        self.models[f"{parameter}_{model_type}"] = model
                        self.scalers[parameter] = checkpoint['scaler']
                        
                        logger.info(f"Loaded model: {parameter}_{model_type}")
                    except Exception as e:
                        logger.warning(f"Failed to load {parameter}_{model_type}: {e}")
    
    def forecast(self, parameter: str, model_type: str = 'lstm') -> ForecastResult:
        """Generate forecast for specified parameter"""
        model_key = f"{parameter}_{model_type}"
        
        if model_key not in self.models:
            logger.warning(f"Model {model_key} not found, training new model")
            self.train_model(parameter, model_type)
        
        model = self.models[model_key]
        scaler = self.scalers[parameter]
        
        # Fetch recent data for input sequence
        recent_data = self.fetch_historical_data(parameter, days=14)
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Insufficient recent data for {parameter}")
        
        # Prepare input sequence
        # Try to find the value column - use 'value' if exists, otherwise use the first numeric column
        if 'value' in recent_data.columns:
            value_col = 'value'
        else:
            # Find first numeric column
            numeric_cols = recent_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError(f"No numeric columns found in data for {parameter}")
            value_col = numeric_cols[0]
        
        values = np.array(recent_data.iloc[-self.sequence_length:][value_col].values).reshape(-1, 1)
        scaled_values = scaler.transform(values)
        input_tensor = torch.FloatTensor(scaled_values).unsqueeze(0)
        
        # Generate forecast
        model.eval()
        with torch.no_grad():
            forecast_scaled = model(input_tensor).squeeze().numpy()
        
        # Inverse scale
        forecast_values = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        
        # Generate forecast times
        last_time = recent_data.index[-1]
        forecast_times = [last_time + datetime.timedelta(hours=i+1) for i in range(self.forecast_horizon)]
        
        # Calculate confidence intervals (simplified)
        std_dev = np.std(np.array(recent_data[value_col].values))
        confidence_intervals = [
            (val - 1.96 * std_dev, val + 1.96 * std_dev)
            for val in forecast_values
        ]
        
        # Assess risk level
        risk_level = self._assess_risk_level(parameter, forecast_values)
        
        # Calculate model accuracy (placeholder)
        model_accuracy = 0.85  # Would be calculated from validation data
        
        result = ForecastResult(
            parameter=parameter,
            forecast_values=forecast_values.tolist(),
            forecast_times=forecast_times,
            confidence_intervals=confidence_intervals,
            model_accuracy=model_accuracy,
            risk_level=risk_level,
            metadata={
                'model_type': model_type,
                'forecast_horizon_hours': self.forecast_horizon,
                'sequence_length_hours': self.sequence_length,
                'generated_at': datetime.datetime.now().isoformat()
            }
        )
        
        return result
    
    def _assess_risk_level(self, parameter: str, forecast_values: np.ndarray) -> str:
        """Assess risk level based on forecast values"""
        config = self.parameters[parameter]
        thresholds = config['risk_thresholds']
        
        max_value = np.max(forecast_values)
        
        if parameter == 'dst_index':
            # For Dst, more negative is higher risk
            min_value = np.min(forecast_values)
            if min_value <= thresholds['extreme']:
                return 'extreme'
            elif min_value <= thresholds['high']:
                return 'high'
            elif min_value <= thresholds['medium']:
                return 'medium'
            else:
                return 'low'
        else:
            # For other parameters, higher values mean higher risk
            if max_value >= thresholds['extreme']:
                return 'extreme'
            elif max_value >= thresholds['high']:
                return 'high'
            elif max_value >= thresholds['medium']:
                return 'medium'
            else:
                return 'low'
    
    def multi_parameter_forecast(self) -> Dict:
        """Generate forecasts for all parameters"""
        forecasts = {}
        overall_risk = 'low'
        
        for parameter in self.parameters.keys():
            try:
                forecast = self.forecast(parameter)
                forecasts[parameter] = {
                    'forecast_values': forecast.forecast_values,
                    'risk_level': forecast.risk_level,
                    'model_accuracy': forecast.model_accuracy,
                    'metadata': forecast.metadata
                }
                
                # Update overall risk
                if forecast.risk_level == 'extreme':
                    overall_risk = 'extreme'
                elif forecast.risk_level == 'high' and overall_risk != 'extreme':
                    overall_risk = 'high'
                elif forecast.risk_level == 'medium' and overall_risk not in ['extreme', 'high']:
                    overall_risk = 'medium'
                    
            except Exception as e:
                logger.error(f"Failed to forecast {parameter}: {e}")
                forecasts[parameter] = {'error': str(e)}
        
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'overall_risk': overall_risk,
            'individual_forecasts': forecasts,
            'summary': self._generate_forecast_summary(forecasts, overall_risk)
        }
    
    def _generate_forecast_summary(self, forecasts: Dict, overall_risk: str) -> Dict:
        """Generate forecast summary and recommendations"""
        high_risk_params = [
            param for param, data in forecasts.items()
            if isinstance(data, dict) and data.get('risk_level') in ['high', 'extreme']
        ]
        
        recommendations = []
        if overall_risk == 'extreme':
            recommendations.extend([
                "CRITICAL: Implement emergency space weather protocols",
                "Consider satellite safe mode for sensitive operations",
                "Monitor radiation levels and crew exposure"
            ])
        elif overall_risk == 'high':
            recommendations.extend([
                "WARNING: Elevated space weather activity expected",
                "Increase monitoring frequency",
                "Prepare contingency plans"
            ])
        elif overall_risk == 'medium':
            recommendations.append("CAUTION: Moderate space weather activity forecasted")
        else:
            recommendations.append("NORMAL: Standard operations continue")
        
        return {
            'overall_risk_level': overall_risk,
            'high_risk_parameters': high_risk_params,
            'recommendations': recommendations,
            'forecast_confidence': 'high' if len([f for f in forecasts.values() if not f.get('error')]) > 2 else 'medium'
        }

def main():
    """Test the forecasting system"""
    forecaster = SpaceWeatherForecaster()
    
    # Train models for key parameters
    for param in ['kp_index', 'dst_index']:
        try:
            result = forecaster.train_model(param, model_type='lstm', epochs=50)
            print(f"Training result for {param}: {result}")
        except Exception as e:
            print(f"Training failed for {param}: {e}")
    
    # Generate multi-parameter forecast
    try:
        forecast_result = forecaster.multi_parameter_forecast()
        print("\n=== Multi-Parameter Forecast ===")
        print(json.dumps(forecast_result, indent=2, default=str))
    except Exception as e:
        print(f"Forecasting failed: {e}")

if __name__ == "__main__":
    main()