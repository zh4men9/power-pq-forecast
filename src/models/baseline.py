"""
Baseline forecasting methods
Implements naive and seasonal naive forecasting as benchmarks
Reference: https://otexts.com/fpp3/simple-methods.html
"""
import numpy as np
import pandas as pd
from typing import Optional


def naive_forecast(
    y_train: np.ndarray,
    horizon: int = 1,
    n_forecasts: int = 1
) -> np.ndarray:
    """
    Naive forecasting: forecast = last observation
    
    Formula: ŷ_{t+h} = y_t
    
    This is also known as the "persistence" or "random walk" forecast.
    It simply uses the most recent observation as the forecast for all
    future time points.
    
    Args:
        y_train: Training data (1D array)
        horizon: Forecast horizon (not used for naive, kept for API consistency)
        n_forecasts: Number of forecasts to generate
    
    Returns:
        Array of forecasts, shape (n_forecasts,)
    """
    last_value = y_train[-1]
    return np.full(n_forecasts, last_value)


def seasonal_naive_forecast(
    y_train: np.ndarray,
    season_length: int,
    horizon: int = 1,
    n_forecasts: int = 1
) -> np.ndarray:
    """
    Seasonal naive forecasting: forecast = observation from last season
    
    Formula: ŷ_{T+h} = y_{T+h-m(k+1)}
    where m is the seasonal period and k is the integer part of (h-1)/m
    
    For example, with hourly data and daily seasonality (m=24):
    - To forecast hour 25 (next day, hour 1), use hour 1 from last day
    - To forecast hour 26 (next day, hour 2), use hour 2 from last day
    
    Reference: https://otexts.com/fpp3/simple-methods.html
    
    Args:
        y_train: Training data (1D array)
        season_length: Length of seasonal period (e.g., 24 for hourly with daily pattern)
        horizon: Forecast horizon
        n_forecasts: Number of forecasts to generate
    
    Returns:
        Array of forecasts, shape (n_forecasts,)
    """
    forecasts = []
    
    for i in range(n_forecasts):
        # Calculate which historical point to use
        # We look back by season_length from the current forecast position
        lookback_idx = len(y_train) - season_length + (horizon - 1) + i
        
        # Wrap around if necessary
        while lookback_idx >= len(y_train):
            lookback_idx -= season_length
        
        if lookback_idx < 0:
            # If we don't have enough history, use the earliest available
            lookback_idx = i % len(y_train)
        
        forecasts.append(y_train[lookback_idx])
    
    return np.array(forecasts)


class NaiveForecaster:
    """Naive forecasting model wrapper"""
    
    def __init__(self):
        self.last_value_ = None
    
    def fit(self, X, y):
        """
        Fit the naive model (just store last value)
        
        Args:
            X: Not used, kept for API consistency
            y: Training target values
        """
        if isinstance(y, pd.DataFrame):
            self.last_value_ = y.values[-1]
        else:
            self.last_value_ = y[-1] if len(y.shape) == 1 else y[-1, :]
        return self
    
    def predict(self, X):
        """
        Generate predictions
        
        Args:
            X: Input features (only used to determine number of predictions)
        
        Returns:
            Array of predictions
        """
        n_samples = len(X) if hasattr(X, '__len__') else 1
        if isinstance(self.last_value_, np.ndarray):
            return np.tile(self.last_value_, (n_samples, 1))
        else:
            return np.full(n_samples, self.last_value_)


class SeasonalNaiveForecaster:
    """Seasonal naive forecasting model wrapper"""
    
    def __init__(self, season_length: int = 24):
        """
        Initialize seasonal naive forecaster
        
        Args:
            season_length: Length of seasonal period
        """
        self.season_length = season_length
        self.train_data_ = None
    
    def fit(self, X, y):
        """
        Fit the seasonal naive model (store training data)
        
        Args:
            X: Not used, kept for API consistency
            y: Training target values
        """
        if isinstance(y, pd.DataFrame):
            self.train_data_ = y.values
        else:
            self.train_data_ = y
        return self
    
    def predict(self, X):
        """
        Generate predictions using seasonal pattern
        
        Args:
            X: Input features (only used to determine number of predictions)
        
        Returns:
            Array of predictions
        """
        n_samples = len(X) if hasattr(X, '__len__') else 1
        
        if len(self.train_data_.shape) == 1:
            # Single target
            predictions = []
            for i in range(n_samples):
                idx = (len(self.train_data_) - self.season_length + i) % len(self.train_data_)
                predictions.append(self.train_data_[idx])
            return np.array(predictions)
        else:
            # Multiple targets
            predictions = []
            for i in range(n_samples):
                idx = (len(self.train_data_) - self.season_length + i) % len(self.train_data_)
                predictions.append(self.train_data_[idx, :])
            return np.array(predictions)
