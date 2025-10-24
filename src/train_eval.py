"""
Training and evaluation pipeline
Implements the main workflow for training and evaluating all models
CRITICAL: Uses rolling origin cross-validation to prevent data leakage
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings

from src.config import Config
from src.cv import TimeSeriesSplit
from src.metrics import eval_metrics
from src.features import create_features, prepare_sequences
from src.models.baseline import NaiveForecaster, SeasonalNaiveForecaster
from src.models.tree import RandomForestForecaster, XGBoostForecaster
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster

warnings.filterwarnings('ignore')


def train_evaluate_tree_models(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    horizon: int,
    cv_splitter: TimeSeriesSplit,
    config: Config
) -> List[Dict]:
    """
    Train and evaluate tree-based models
    
    Args:
        X: Feature DataFrame
        Y: Target DataFrame with P and Q
        horizon: Forecast horizon
        cv_splitter: Cross-validation splitter
        config: Configuration object
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Create target shifted by horizon
    Y_h = Y.shift(-horizon).dropna()
    X_h = X.loc[Y_h.index]
    
    # Initialize models
    models = {}
    
    if config.get('models', 'rf', 'enabled', default=False):
        rf_params = config.get('models', 'rf', default={})
        models['RandomForest'] = RandomForestForecaster(
            n_estimators=rf_params.get('n_estimators', 100),
            max_depth=rf_params.get('max_depth'),
            n_jobs=rf_params.get('n_jobs', -1),
            random_state=42
        )
    
    if config.get('models', 'xgb', 'enabled', default=False):
        xgb_params = config.get('models', 'xgb', default={})
        models['XGBoost'] = XGBoostForecaster(
            n_estimators=xgb_params.get('n_estimators', 100),
            max_depth=xgb_params.get('max_depth', 6),
            learning_rate=xgb_params.get('learning_rate', 0.1),
            n_jobs=xgb_params.get('n_jobs', -1),
            random_state=42
        )
    
    # Cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X_h)):
        print(f"  Fold {fold_idx + 1}/{cv_splitter.n_splits} - Tree models")
        
        X_train = X_h.iloc[train_idx]
        Y_train = Y_h.iloc[train_idx]
        X_test = X_h.iloc[test_idx]
        Y_test = Y_h.iloc[test_idx]
        
        # Train each model
        for model_name, model in models.items():
            # Fit model
            model.fit(X_train, Y_train)
            
            # Predict
            Y_pred = model.predict(X_test)
            
            # Evaluate for each target
            for target_idx, target in enumerate(['P', 'Q']):
                y_true = Y_test[target].values
                y_pred = Y_pred[:, target_idx] if len(Y_pred.shape) > 1 else Y_pred
                
                metrics = eval_metrics(y_true, y_pred)
                
                results.append({
                    'model': model_name,
                    'horizon': horizon,
                    'fold': fold_idx,
                    'target': target,
                    **metrics
                })
    
    return results


def train_evaluate_baseline_models(
    Y: pd.DataFrame,
    horizon: int,
    cv_splitter: TimeSeriesSplit,
    config: Config
) -> List[Dict]:
    """
    Train and evaluate baseline models
    
    Args:
        Y: Target DataFrame with P and Q
        horizon: Forecast horizon
        cv_splitter: Cross-validation splitter
        config: Configuration object
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Create target shifted by horizon
    Y_h = Y.shift(-horizon).dropna()
    Y_orig = Y.loc[Y_h.index]
    
    # Determine season length from config
    season_length = config.get('features', 'season_length', default=24)
    
    # Cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(Y_orig)):
        print(f"  Fold {fold_idx + 1}/{cv_splitter.n_splits} - Baseline models")
        
        Y_train = Y_orig.iloc[train_idx]
        Y_test = Y_h.iloc[test_idx]
        
        # Naive model
        if config.get('models', 'naive', 'enabled', default=True):
            naive_model = NaiveForecaster()
            naive_model.fit(None, Y_train)
            Y_pred_naive = naive_model.predict(range(len(test_idx)))
            
            for target_idx, target in enumerate(['P', 'Q']):
                y_true = Y_test[target].values
                y_pred = Y_pred_naive[:, target_idx]
                
                metrics = eval_metrics(y_true, y_pred)
                
                results.append({
                    'model': 'Naive',
                    'horizon': horizon,
                    'fold': fold_idx,
                    'target': target,
                    **metrics
                })
        
        # Seasonal naive model
        if config.get('models', 'seasonal_naive', 'enabled', default=True):
            seasonal_model = SeasonalNaiveForecaster(season_length=season_length)
            seasonal_model.fit(None, Y_train)
            Y_pred_seasonal = seasonal_model.predict(range(len(test_idx)))
            
            for target_idx, target in enumerate(['P', 'Q']):
                y_true = Y_test[target].values
                y_pred = Y_pred_seasonal[:, target_idx]
                
                metrics = eval_metrics(y_true, y_pred)
                
                results.append({
                    'model': 'SeasonalNaive',
                    'horizon': horizon,
                    'fold': fold_idx,
                    'target': target,
                    **metrics
                })
    
    return results


def train_evaluate_deep_models(
    df: pd.DataFrame,
    horizon: int,
    cv_splitter: TimeSeriesSplit,
    config: Config
) -> List[Dict]:
    """
    Train and evaluate deep learning models (LSTM, Transformer)
    
    Args:
        df: DataFrame with P and Q
        horizon: Forecast horizon
        cv_splitter: Cross-validation splitter
        config: Configuration object
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Prepare sequences
    sequence_length = config.get('features', 'sequence_length', default=24)
    X_seq, Y_seq = prepare_sequences(df, sequence_length=sequence_length, horizon=horizon)
    
    # Initialize models
    models = {}
    
    if config.get('models', 'lstm', 'enabled', default=False):
        lstm_params = config.get('models', 'lstm', default={})
        models['LSTM'] = LSTMForecaster(
            hidden_size=lstm_params.get('hidden_size', 64),
            num_layers=lstm_params.get('num_layers', 2),
            dropout=lstm_params.get('dropout', 0.2),
            epochs=lstm_params.get('epochs', 50),
            batch_size=lstm_params.get('batch_size', 32),
            learning_rate=lstm_params.get('learning_rate', 0.001)
        )
    
    if config.get('models', 'transformer', 'enabled', default=False):
        trans_params = config.get('models', 'transformer', default={})
        models['Transformer'] = TransformerForecaster(
            d_model=trans_params.get('d_model', 64),
            nhead=trans_params.get('nhead', 4),
            num_encoder_layers=trans_params.get('num_encoder_layers', 2),
            num_decoder_layers=trans_params.get('num_decoder_layers', 2),
            dim_feedforward=trans_params.get('dim_feedforward', 256),
            dropout=trans_params.get('dropout', 0.1),
            epochs=trans_params.get('epochs', 50),
            batch_size=trans_params.get('batch_size', 32),
            learning_rate=trans_params.get('learning_rate', 0.001)
        )
    
    if not models:
        return results
    
    # Cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X_seq)):
        print(f"  Fold {fold_idx + 1}/{cv_splitter.n_splits} - Deep learning models")
        
        X_train = X_seq[train_idx]
        Y_train = Y_seq[train_idx]
        X_test = X_seq[test_idx]
        Y_test = Y_seq[test_idx]
        
        # Train each model
        for model_name, model in models.items():
            print(f"    Training {model_name}...")
            
            # Fit model
            model.fit(X_train, Y_train)
            
            # Predict
            Y_pred = model.predict(X_test)
            
            # Evaluate for each target
            for target_idx, target in enumerate(['P', 'Q']):
                y_true = Y_test[:, target_idx]
                y_pred = Y_pred[:, target_idx]
                
                metrics = eval_metrics(y_true, y_pred)
                
                results.append({
                    'model': model_name,
                    'horizon': horizon,
                    'fold': fold_idx,
                    'target': target,
                    **metrics
                })
    
    return results


def run_evaluation(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full evaluation pipeline
    
    Args:
        config: Configuration object
        df: DataFrame with P and Q columns
    
    Returns:
        DataFrame with all evaluation results
    """
    horizons = config.get('evaluation', 'horizons', default=[1, 12, 24])
    test_window = config.get('evaluation', 'test_window', default=100)
    n_splits = config.get('evaluation', 'n_splits', default=3)
    
    all_results = []
    
    # Create features for tree models
    max_lag = config.get('features', 'max_lag', default=24)
    roll_windows = config.get('features', 'roll_windows', default=[6, 12, 24])
    use_time_features = config.get('features', 'use_time_features', default=True)
    exog_cols = config.get('features', 'exog_cols', default=[])
    
    X, Y = create_features(df, max_lag=max_lag, roll_windows=roll_windows, 
                          use_time_features=use_time_features, exog_cols=exog_cols)
    
    # For each horizon
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"Evaluating horizon = {horizon}")
        print(f"{'='*60}")
        
        # Create CV splitter
        cv_splitter = TimeSeriesSplit(test_window=test_window, n_splits=n_splits)
        
        # Baseline models
        baseline_results = train_evaluate_baseline_models(Y, horizon, cv_splitter, config)
        all_results.extend(baseline_results)
        
        # Tree models
        tree_results = train_evaluate_tree_models(X, Y, horizon, cv_splitter, config)
        all_results.extend(tree_results)
        
        # Deep learning models
        deep_results = train_evaluate_deep_models(df, horizon, cv_splitter, config)
        all_results.extend(deep_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "cv_metrics.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete. Results saved to {output_dir / 'cv_metrics.csv'}")
    print(f"{'='*60}")
    
    return results_df
