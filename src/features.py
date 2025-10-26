"""
Feature engineering module for time series forecasting
Creates lag features, rolling statistics, and time-based features
IMPORTANT: All features use only past information to prevent data leakage
"""
import pandas as pd
import numpy as np
from typing import Tuple, List


def create_lag_features(df: pd.DataFrame, max_lag: int = 24, target_cols: List[str] = None) -> pd.DataFrame:
    """
    Create lag features for specified columns
    IMPORTANT: Uses shift() to ensure only past values are used
    
    Args:
        df: DataFrame with columns to create lags for
        max_lag: Maximum lag to create
        target_cols: List of columns to create lags for (default: ['P', 'Q'])
    
    Returns:
        DataFrame with lag features
    """
    if target_cols is None:
        target_cols = ['P', 'Q']
    
    features = df.copy()
    
    for col in target_cols:
        if col in df.columns:
            for lag in range(1, max_lag + 1):
                features[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return features


def create_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [6, 12, 24],
    target_cols: List[str] = None
) -> pd.DataFrame:
    """
    Create rolling statistics features
    IMPORTANT: Uses rolling() with only past values
    
    Args:
        df: DataFrame with columns to create rolling features for
        windows: List of window sizes
        target_cols: List of columns to create rolling features for (default: ['P', 'Q'])
    
    Returns:
        DataFrame with rolling features
    """
    if target_cols is None:
        target_cols = ['P', 'Q']
    
    features = df.copy()
    
    for col in target_cols:
        if col in df.columns:
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
    use_time_features: bool = True,
    exog_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create all features for time series forecasting
    CRITICAL: Ensures no future information leakage
    
    Args:
        df: DataFrame with P and Q columns (and optional exogenous columns) and DatetimeIndex
        max_lag: Maximum lag for lag features
        roll_windows: Window sizes for rolling statistics
        use_time_features: Whether to include time features
        exog_cols: List of exogenous variable column names to include in features
    
    Returns:
        Tuple of (X features DataFrame, Y targets DataFrame)
    """
    # Start with P and Q (always present)
    base_cols = ['P', 'Q']
    
    # Add exogenous columns if specified
    feature_cols = base_cols.copy()
    if exog_cols:
        for col in exog_cols:
            if col in df.columns:
                feature_cols.append(col)
            else:
                print(f"Warning: Exogenous column '{col}' not found in dataframe, skipping")
    
    # Start with the relevant columns
    features = df[feature_cols].copy()
    
    # Add lag features (shifts ensure no future info)
    features = create_lag_features(features, max_lag, target_cols=feature_cols)
    
    # Add rolling features (shift+rolling ensures no future info)
    features = create_rolling_features(features, roll_windows, target_cols=feature_cols)
    
    # Add time features (inherently no future info)
    if use_time_features:
        features = create_time_features(features)
    
    # Separate features (X) and targets (Y)
    # Remove original columns from features as they would cause leakage
    X = features.drop(feature_cols, axis=1)
    Y = features[base_cols]  # Only P and Q are targets
    
    # Drop rows with NaN (from lagging/rolling)
    valid_idx = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X = X[valid_idx]
    Y = Y[valid_idx]
    
    print(f"Feature engineering complete:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Features: lag (up to {max_lag}), rolling (windows {roll_windows}), time features: {use_time_features}")
    if exog_cols:
        print(f"  Exogenous variables used: {[c for c in exog_cols if c in df.columns]}")
    
    return X, Y


def prepare_sequences(
    df: pd.DataFrame,
    sequence_length: int = 24,
    horizon: int = 1,
    exog_cols: List[str] = None,
    target_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for deep learning models (LSTM, Transformer)
    CRITICAL: Sequences only use past values for prediction
    
    Args:
        df: DataFrame with P, Q and optionally exogenous columns
        sequence_length: Length of input sequence
        horizon: Prediction horizon
        exog_cols: List of exogenous column names to include (default: None)
        target_cols: List of target columns to predict (default: ['P', 'Q'])
    
    Returns:
        Tuple of (X sequences, Y targets)
        X shape: (n_samples, sequence_length, n_features)
        Y shape: (n_samples, n_targets)
    
    Example:
        Without exog: X shape (1000, 24, 2) - only P, Q
        With exog: X shape (1000, 24, 7) - P, Q, + 5 exog vars
    """
    # Default targets
    if target_cols is None:
        target_cols = ['P', 'Q']
    
    # Determine which columns to use for input sequences
    input_cols = ['P', 'Q']
    if exog_cols:
        # Validate exog columns exist
        available_exog = [col for col in exog_cols if col in df.columns]
        if available_exog:
            input_cols.extend(available_exog)
            print(f"Using exogenous variables in sequences: {available_exog}")
        else:
            print(f"Warning: No exogenous columns found in df. Requested: {exog_cols}")
    
    # CRITICAL: Drop rows with NaN values before creating sequences
    all_cols = list(set(input_cols + target_cols))
    df_clean = df[all_cols].dropna()
    if len(df_clean) < len(df):
        print(f"Warning: Dropped {len(df) - len(df_clean)} rows with NaN values")
    
    data = df_clean[input_cols].values
    X_list = []
    Y_list = []
    
    for i in range(len(data) - sequence_length - horizon + 1):
        # Input sequence uses [i : i+sequence_length]
        X_list.append(data[i:i + sequence_length])
        # Target at [i + sequence_length + horizon - 1]
        Y_list.append(df_clean[target_cols].values[i + sequence_length + horizon - 1])
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    print(f"Prepared {len(X)} sequences:")
    print(f"  X shape: {X.shape} (samples, sequence_length, features)")
    print(f"  Y shape: {Y.shape} (samples, targets)")
    print(f"  Input features: {input_cols}")
    print(f"  Output targets: {target_cols}")
    
    return X, Y
