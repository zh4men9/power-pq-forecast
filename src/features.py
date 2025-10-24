"""
Feature engineering module for time series forecasting
Creates lag features, rolling statistics, and time-based features
IMPORTANT: All features use only past information to prevent data leakage
"""
import pandas as pd
import numpy as np
from typing import Tuple, List


def create_lag_features(df: pd.DataFrame, max_lag: int = 24) -> pd.DataFrame:
    """
    Create lag features for P and Q
    IMPORTANT: Uses shift() to ensure only past values are used
    
    Args:
        df: DataFrame with P and Q columns
        max_lag: Maximum lag to create
    
    Returns:
        DataFrame with lag features
    """
    features = df.copy()
    
    for col in ['P', 'Q']:
        for lag in range(1, max_lag + 1):
            features[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return features


def create_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [6, 12, 24]
) -> pd.DataFrame:
    """
    Create rolling statistics features
    IMPORTANT: Uses rolling() with only past values
    
    Args:
        df: DataFrame with P and Q columns
        windows: List of window sizes
    
    Returns:
        DataFrame with rolling features
    """
    features = df.copy()
    
    for col in ['P', 'Q']:
        for window in windows:
            # Rolling mean
            features[f'{col}_roll_mean_{window}'] = df[col].shift(1).rolling(window).mean()
            # Rolling std
            features[f'{col}_roll_std_{window}'] = df[col].shift(1).rolling(window).std()
            # Rolling min
            features[f'{col}_roll_min_{window}'] = df[col].shift(1).rolling(window).min()
            # Rolling max
            features[f'{col}_roll_max_{window}'] = df[col].shift(1).rolling(window).max()
    
    return features


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical time features using sin/cos encoding
    
    Args:
        df: DataFrame with DatetimeIndex
    
    Returns:
        DataFrame with time features
    """
    features = df.copy()
    
    # Hour of day (0-23)
    if hasattr(df.index, 'hour'):
        features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Day of week (0-6)
    if hasattr(df.index, 'dayofweek'):
        features['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # Month (1-12)
    if hasattr(df.index, 'month'):
        features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    return features


def create_features(
    df: pd.DataFrame,
    max_lag: int = 24,
    roll_windows: List[int] = [6, 12, 24],
    use_time_features: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create all features for time series forecasting
    CRITICAL: Ensures no future information leakage
    
    Args:
        df: DataFrame with P and Q columns and DatetimeIndex
        max_lag: Maximum lag for lag features
        roll_windows: Window sizes for rolling statistics
        use_time_features: Whether to include time features
    
    Returns:
        Tuple of (X features DataFrame, Y targets DataFrame)
    """
    # Start with original data
    features = df[['P', 'Q']].copy()
    
    # Add lag features (shifts ensure no future info)
    features = create_lag_features(features, max_lag)
    
    # Add rolling features (shift+rolling ensures no future info)
    features = create_rolling_features(features, roll_windows)
    
    # Add time features (inherently no future info)
    if use_time_features:
        features = create_time_features(features)
    
    # Separate features (X) and targets (Y)
    # Remove original P and Q from features as they would cause leakage
    X = features.drop(['P', 'Q'], axis=1)
    Y = features[['P', 'Q']]
    
    # Drop rows with NaN (from lagging/rolling)
    valid_idx = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X = X[valid_idx]
    Y = Y[valid_idx]
    
    print(f"Feature engineering complete:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Features: lag (up to {max_lag}), rolling (windows {roll_windows}), time features: {use_time_features}")
    
    return X, Y


def prepare_sequences(
    df: pd.DataFrame,
    sequence_length: int = 24,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for deep learning models (LSTM, Transformer)
    CRITICAL: Sequences only use past values for prediction
    
    Args:
        df: DataFrame with P and Q columns
        sequence_length: Length of input sequence
        horizon: Prediction horizon
    
    Returns:
        Tuple of (X sequences, Y targets)
        X shape: (n_samples, sequence_length, 2)
        Y shape: (n_samples, 2)
    """
    data = df[['P', 'Q']].values
    X_list = []
    Y_list = []
    
    for i in range(len(data) - sequence_length - horizon + 1):
        # Input sequence uses [i : i+sequence_length]
        X_list.append(data[i:i + sequence_length])
        # Target is at [i + sequence_length + horizon - 1]
        Y_list.append(data[i + sequence_length + horizon - 1])
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    print(f"Prepared {len(X)} sequences:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    
    return X, Y
