"""
Sample data generator for testing the power quality forecasting system
Generates synthetic time series data with daily and weekly patterns
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(
    n_days: int = 30,
    freq: str = 'H',
    output_path: str = 'data/raw/sample_data.xlsx'
):
    """
    Generate sample power quality data
    
    Args:
        n_days: Number of days to generate
        freq: Frequency ('H' for hourly, 'T' for minutes)
        output_path: Path to save the data
    """
    # Generate time index
    start_date = datetime(2023, 1, 1)
    if freq == 'H':
        periods = n_days * 24
    elif freq == '15T':
        periods = n_days * 24 * 4
    else:
        periods = n_days * 24  # Default to hourly
    
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate base patterns
    hours = date_range.hour
    days_of_week = date_range.dayofweek
    
    # Create synthetic P (active power) with patterns
    # Base load
    P_base = 100
    
    # Daily pattern (higher during day, lower at night)
    daily_pattern = 30 * np.sin(2 * np.pi * hours / 24)
    
    # Weekly pattern (lower on weekends)
    weekly_pattern = -10 * (days_of_week >= 5).astype(float)
    
    # Random noise
    noise_p = np.random.normal(0, 5, len(date_range))
    
    # Combine patterns
    P = P_base + daily_pattern + weekly_pattern + noise_p
    P = np.maximum(P, 0)  # Ensure non-negative
    
    # Create synthetic Q (reactive power) correlated with P
    Q_base = 50
    Q_pattern = 0.4 * (P - P_base)  # Correlated with P
    noise_q = np.random.normal(0, 3, len(date_range))
    
    Q = Q_base + Q_pattern + noise_q
    
    # Create DataFrame
    df = pd.DataFrame({
        '时间': date_range,
        '有功功率': P,
        '无功功率': Q
    })
    
    # Add some missing values to test interpolation (about 1%)
    n_missing = int(len(df) * 0.01)
    missing_idx = np.random.choice(len(df), n_missing, replace=False)
    df.loc[missing_idx, '有功功率'] = np.nan
    df.loc[missing_idx[::2], '无功功率'] = np.nan  # Some missing in Q too
    
    # Save to Excel
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    
    print(f"Sample data generated successfully!")
    print(f"Path: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['时间'].min()} to {df['时间'].max()}")
    print(f"Missing values: P={df['有功功率'].isna().sum()}, Q={df['无功功率'].isna().sum()}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nStatistics:")
    print(df[['有功功率', '无功功率']].describe())


if __name__ == '__main__':
    # Generate sample data
    generate_sample_data(n_days=30, freq='H', output_path='data/raw/sample_data.xlsx')
