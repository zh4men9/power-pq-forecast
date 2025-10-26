"""
Training and evaluation pipeline
Implements the main workflow for training and evaluating all models
CRITICAL: Uses rolling origin cross-validation to prevent data leakage
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import logging

from src.config import Config
from src.cv import TimeSeriesSplit
from src.metrics import eval_metrics
from src.features import create_features, prepare_sequences
from src.models.baseline import NaiveForecaster, SeasonalNaiveForecaster
from src.models.tree import RandomForestForecaster, XGBoostForecaster
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster

warnings.filterwarnings('ignore')


def get_target_columns(config: Config) -> List[str]:
    """
    获取要预测的目标列列表
    
    Args:
        config: 配置对象
    
    Returns:
        目标列名称列表
    """
    targets = []
    if config.get('target', 'predict_p', default=True):
        targets.append('P')
    if config.get('target', 'predict_q', default=True):
        targets.append('Q')
    
    if not targets:
        raise ValueError("至少需要预测P或Q中的一个目标")
    
    return targets


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
    
    # Get target columns to predict
    target_cols = get_target_columns(config)
    
    # Create target shifted by horizon (only keep columns to predict)
    Y_h = Y[target_cols].shift(-horizon).dropna()
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
        logging.info(f"  Fold {fold_idx + 1}/{cv_splitter.n_splits} - Tree models")
        logging.info(f"    训练集大小: {len(train_idx)} 样本, 测试集大小: {len(test_idx)} 样本")
        
        X_train = X_h.iloc[train_idx]
        Y_train = Y_h.iloc[train_idx]
        X_test = X_h.iloc[test_idx]
        Y_test = Y_h.iloc[test_idx]
        
        # Train each model
        for model_name, model in models.items():
            logging.info(f"    训练 {model_name} 模型 (特征维度: {X_train.shape[1]})...")
            
            # Fit model
            model.fit(X_train, Y_train)
            
            logging.info(f"      {model_name} 训练完成，开始预测...")
            
            # Predict
            Y_pred = model.predict(X_test)
            
            # Evaluate for each target
            for target_idx, target in enumerate(target_cols):
                y_true = Y_test[target].values
                y_pred = Y_pred[:, target_idx] if len(Y_pred.shape) > 1 else Y_pred
                
                metrics = eval_metrics(y_true, y_pred, 
                                      metric_names=config.get('evaluation', 'metrics'))
                
                logging.info(f"      {model_name} ({target}): RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
                
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
    
    # Get target columns to predict
    target_cols = get_target_columns(config)
    
    # Create target shifted by horizon (only keep columns to predict)
    Y_h = Y[target_cols].shift(-horizon).dropna()
    Y_orig = Y[target_cols].loc[Y_h.index]
    
    # Determine season length from config
    season_length = config.get('features', 'season_length', default=24)
    
    # Cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(Y_orig)):
        logging.info(f"  Fold {fold_idx + 1}/{cv_splitter.n_splits} - Baseline models")
        logging.info(f"    训练集大小: {len(train_idx)} 样本, 测试集大小: {len(test_idx)} 样本")
        
        Y_train = Y_orig.iloc[train_idx]
        Y_test = Y_h.iloc[test_idx]
        
        # Naive model
        if config.get('models', 'naive', 'enabled', default=True):
            logging.info(f"    训练 Naive 基线模型...")
            naive_model = NaiveForecaster()
            naive_model.fit(None, Y_train)
            # For baseline models, we predict one step at a time for each test sample
            # The horizon is already accounted for in Y_h (shifted by horizon)
            Y_pred_naive = naive_model.predict([0])  # Predict 1 step (using last training value)
            # Repeat the prediction for all test samples
            Y_pred_naive = np.repeat([Y_pred_naive], len(test_idx), axis=0)
            
            for target_idx, target in enumerate(target_cols):
                y_true = Y_test[target].values
                y_pred = Y_pred_naive[:, target_idx] if len(target_cols) > 1 else Y_pred_naive.flatten()
                
                metrics = eval_metrics(y_true, y_pred,
                                      metric_names=config.get('evaluation', 'metrics'))
                
                logging.info(f"      Naive ({target}): RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, ACC={metrics.get('ACC', 0):.2f}%")
                
                results.append({
                    'model': 'Naive',
                    'horizon': horizon,
                    'fold': fold_idx,
                    'target': target,
                    **metrics
                })
        
        # Seasonal naive model
        if config.get('models', 'seasonal_naive', 'enabled', default=True):
            logging.info(f"    训练 SeasonalNaive 基线模型 (周期长度: {season_length})...")
            seasonal_model = SeasonalNaiveForecaster(season_length=season_length)
            seasonal_model.fit(None, Y_train)
            # For seasonal naive, use seasonal pattern from training data
            # Predict one value for each test sample
            Y_pred_seasonal = []
            for i in range(len(test_idx)):
                # Get the seasonal value from training data
                seasonal_idx = (len(Y_train) - season_length + horizon - 1 + i) % len(Y_train)
                Y_pred_seasonal.append(Y_train.iloc[seasonal_idx].values)
            Y_pred_seasonal = np.array(Y_pred_seasonal)
            
            for target_idx, target in enumerate(target_cols):
                y_true = Y_test[target].values
                y_pred = Y_pred_seasonal[:, target_idx] if len(target_cols) > 1 else Y_pred_seasonal.flatten()
                
                metrics = eval_metrics(y_true, y_pred,
                                      metric_names=config.get('evaluation', 'metrics'))
                
                logging.info(f"      SeasonalNaive ({target}): RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, ACC={metrics.get('ACC', 0):.2f}%")
                
                results.append({
                    'model': 'SeasonalNaive',
                    'horizon': horizon,
                    'fold': fold_idx,
                    'target': target,
                    **metrics
                })
    
    return results


def train_evaluate_deep_models_once(
    df: pd.DataFrame,
    horizons: List[int],
    cv_splitter: TimeSeriesSplit,
    config: Config
) -> Tuple[List[Dict], Dict]:
    """
    Train deep learning models using specified strategy
    
    Strategies:
    - multiple_output: Train ONE model to predict all horizons (fast, default)
    - direct: Train SEPARATE model for each horizon (slow, may be more accurate)
    
    Args:
        df: DataFrame with P, Q and optionally exogenous columns
        horizons: List of forecast horizons to evaluate
        cv_splitter: Cross-validation splitter
        config: Configuration object
    
    Returns:
        Tuple of (results list, trained models dict)
    """
    # Get strategy from config
    strategy = config.get('evaluation', 'deep_learning_strategy', default='multiple_output')
    
    if strategy == 'multiple_output':
        return _train_evaluate_multiple_output(df, horizons, cv_splitter, config)
    else:  # direct
        return _train_evaluate_direct(df, horizons, cv_splitter, config)


def _train_evaluate_multiple_output(
    df: pd.DataFrame,
    horizons: List[int],
    cv_splitter: TimeSeriesSplit,
    config: Config
) -> Tuple[List[Dict], Dict]:
    """
    Multiple Output Strategy: Train ONE model for ALL horizons
    
    This is much faster as it only trains 2 models total (LSTM + Transformer)
    instead of 2*N models where N is the number of horizons.
    """
    results = []
    trained_models = {}
    
    # Get target columns to predict
    target_cols = get_target_columns(config)
    sequence_length = config.get('features', 'sequence_length', default=24)
    exog_cols = config.get('features', 'exog_cols', default=[])
    n_targets = len(target_cols)
    n_horizons = len(horizons)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"多输出策略 (Multiple Output Strategy)")
    logging.info(f"{'='*60}")
    logging.info(f"  将训练 1 个模型预测所有 {n_horizons} 个预测步长")
    logging.info(f"  预测目标: {target_cols} ({n_targets} 个)")
    logging.info(f"  预测步长: {horizons}")
    logging.info(f"  训练效率: {n_horizons}x 加速 🚀")
    logging.info(f"{'='*60}\n")
    
    # Prepare data for ALL horizons at once
    # X will be the same, but Y will be stacked [Y_h1, Y_h2, ..., Y_hn]
    logging.info(f"准备多输出数据...")
    
    # Get X from first horizon (same for all)
    X_seq, _ = prepare_sequences(df, sequence_length=sequence_length, 
                                 horizon=horizons[0], exog_cols=exog_cols,
                                 target_cols=target_cols)
    
    # Stack Y for all horizons
    Y_all_horizons = []
    for horizon in horizons:
        _, Y_h = prepare_sequences(df, sequence_length=sequence_length, 
                                   horizon=horizon, exog_cols=exog_cols,
                                   target_cols=target_cols)
        Y_all_horizons.append(Y_h)
    
    # Y shape: (n_samples, n_targets * n_horizons)
    Y_seq = np.hstack(Y_all_horizons)
    
    logging.info(f"  X shape: {X_seq.shape}")
    logging.info(f"  Y shape: {Y_seq.shape} ({n_targets} targets × {n_horizons} horizons)")
    
    # Initialize models with n_horizons parameter
    models = {}
    
    if config.get('models', 'lstm', 'enabled', default=False):
        lstm_params = config.get('models', 'lstm', default={})
        device_type = config.get('device', 'type', default='cpu')
        
        logging.info(f"\n初始化 LSTM 模型 (多输出):")
        logging.info(f"  hidden_size={lstm_params.get('hidden_size', 64)}")
        logging.info(f"  num_layers={lstm_params.get('num_layers', 2)}")
        logging.info(f"  dropout={lstm_params.get('dropout', 0.2)}")
        logging.info(f"  epochs={lstm_params.get('epochs', 50)}")
        logging.info(f"  batch_size={lstm_params.get('batch_size', 32)}")
        logging.info(f"  learning_rate={lstm_params.get('learning_rate', 0.001)}")
        logging.info(f"  n_horizons={n_horizons}")
        
        models['LSTM'] = LSTMForecaster(
            hidden_size=lstm_params.get('hidden_size', 64),
            num_layers=lstm_params.get('num_layers', 2),
            dropout=lstm_params.get('dropout', 0.2),
            epochs=lstm_params.get('epochs', 50),
            batch_size=lstm_params.get('batch_size', 32),
            learning_rate=lstm_params.get('learning_rate', 0.001),
            device=device_type,
            n_horizons=n_horizons
        )
    
    if config.get('models', 'transformer', 'enabled', default=False):
        trans_params = config.get('models', 'transformer', default={})
        device_type = config.get('device', 'type', default='cpu')
        
        logging.info(f"\n初始化 Transformer 模型 (多输出):")
        logging.info(f"  d_model={trans_params.get('d_model', 64)}")
        logging.info(f"  nhead={trans_params.get('nhead', 4)}")
        logging.info(f"  num_encoder_layers={trans_params.get('num_encoder_layers', 2)}")
        logging.info(f"  num_decoder_layers={trans_params.get('num_decoder_layers', 2)}")
        logging.info(f"  dim_feedforward={trans_params.get('dim_feedforward', 256)}")
        logging.info(f"  dropout={trans_params.get('dropout', 0.1)}")
        logging.info(f"  epochs={trans_params.get('epochs', 50)}")
        logging.info(f"  batch_size={trans_params.get('batch_size', 32)}")
        logging.info(f"  learning_rate={trans_params.get('learning_rate', 0.001)}")
        logging.info(f"  n_horizons={n_horizons}")
        
        models['Transformer'] = TransformerForecaster(
            d_model=trans_params.get('d_model', 64),
            nhead=trans_params.get('nhead', 4),
            num_encoder_layers=trans_params.get('num_encoder_layers', 2),
            num_decoder_layers=trans_params.get('num_decoder_layers', 2),
            dim_feedforward=trans_params.get('dim_feedforward', 256),
            dropout=trans_params.get('dropout', 0.1),
            epochs=trans_params.get('epochs', 50),
            batch_size=trans_params.get('batch_size', 32),
            learning_rate=trans_params.get('learning_rate', 0.001),
            device=device_type,
            n_horizons=n_horizons
        )
    
    if not models:
        return results, trained_models
    
    # Cross-validation: Train ONCE per fold
    from tqdm import tqdm
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X_seq)):
        logging.info(f"\n📊 Fold {fold_idx + 1}/{cv_splitter.n_splits}")
        
        X_train = X_seq[train_idx]
        Y_train = Y_seq[train_idx]
        X_test = X_seq[test_idx]
        Y_test = Y_seq[test_idx]
        
        logging.info(f"  训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
        
        # Train each model ONCE
        for model_name, model in models.items():
            logging.info(f"  训练 {model_name} (一次性训练所有{n_horizons}个步长)...")
            
            # Fit model
            model.fit(X_train, Y_train)
            
            # Predict: Y_pred shape = (n_samples, n_targets * n_horizons)
            Y_pred = model.predict(X_test)
            
            # Extract predictions for each horizon and evaluate
            for horizon_idx, horizon in enumerate(horizons):
                # Extract predictions for this horizon
                start_col = horizon_idx * n_targets
                end_col = start_col + n_targets
                
                Y_pred_h = Y_pred[:, start_col:end_col]
                Y_test_h = Y_test[:, start_col:end_col]
                
                # Evaluate for each target
                for target_idx, target in enumerate(target_cols):
                    y_true = Y_test_h[:, target_idx] if n_targets > 1 else Y_test_h
                    y_pred = Y_pred_h[:, target_idx] if n_targets > 1 else Y_pred_h
                    
                    metrics = eval_metrics(y_true, y_pred,
                                          metric_names=config.get('evaluation', 'metrics'))
                    
                    logging.info(f"    {model_name} (horizon={horizon}, {target}): RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
                    
                    results.append({
                        'model': model_name,
                        'horizon': horizon,
                        'fold': fold_idx,
                        'target': target,
                        **metrics
                    })
        
        # Save trained models (last fold)
        if fold_idx == cv_splitter.n_splits - 1:
            for model_name, model in models.items():
                trained_models[f"{model_name}_all_horizons"] = model
    
    return results, trained_models


def _train_evaluate_direct(
    df: pd.DataFrame,
    horizons: List[int],
    cv_splitter: TimeSeriesSplit,
    config: Config
) -> Tuple[List[Dict], Dict]:
    """
    Direct Strategy: Train SEPARATE model for EACH horizon
    
    This trains N models per algorithm, which is slower but may be more accurate.
    """
    results = []
    trained_models = {}
    
    # Get target columns to predict
    target_cols = get_target_columns(config)
    sequence_length = config.get('features', 'sequence_length', default=24)
    exog_cols = config.get('features', 'exog_cols', default=[])
    n_horizons = len(horizons)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"直接策略 (Direct Strategy)")
    logging.info(f"{'='*60}")
    logging.info(f"  将为每个预测步长训练独立模型")
    logging.info(f"  预测步长: {horizons} ({n_horizons} 个)")
    logging.info(f"  每个算法将训练 {n_horizons} 次")
    logging.info(f"{'='*60}\n")
    
    # Prepare sequences for EACH horizon
    horizon_data = {}
    for horizon in horizons:
        logging.info(f"准备 horizon={horizon} 的序列数据...")
        X_seq, Y_seq = prepare_sequences(df, sequence_length=sequence_length, 
                                         horizon=horizon, exog_cols=exog_cols,
                                         target_cols=target_cols)
        horizon_data[horizon] = (X_seq, Y_seq)
    
    # Initialize models (only once)
    models = {}
    
    if config.get('models', 'lstm', 'enabled', default=False):
        lstm_params = config.get('models', 'lstm', default={})
        device_type = config.get('device', 'type', default='cpu')
        
        logging.info(f"\n初始化 LSTM 模型参数:")
        logging.info(f"  hidden_size={lstm_params.get('hidden_size', 64)}")
        logging.info(f"  num_layers={lstm_params.get('num_layers', 2)}")
        logging.info(f"  dropout={lstm_params.get('dropout', 0.2)}")
        logging.info(f"  epochs={lstm_params.get('epochs', 50)}")
        logging.info(f"  batch_size={lstm_params.get('batch_size', 32)}")
        logging.info(f"  learning_rate={lstm_params.get('learning_rate', 0.001)}")
        
        models['LSTM'] = {
            'params': lstm_params,
            'device': device_type
        }
    
    if config.get('models', 'transformer', 'enabled', default=False):
        trans_params = config.get('models', 'transformer', default={})
        device_type = config.get('device', 'type', default='cpu')
        
        logging.info(f"\n初始化 Transformer 模型参数:")
        logging.info(f"  d_model={trans_params.get('d_model', 64)}")
        logging.info(f"  nhead={trans_params.get('nhead', 4)}")
        logging.info(f"  num_encoder_layers={trans_params.get('num_encoder_layers', 2)}")
        logging.info(f"  num_decoder_layers={trans_params.get('num_decoder_layers', 2)}")
        logging.info(f"  dim_feedforward={trans_params.get('dim_feedforward', 256)}")
        logging.info(f"  dropout={trans_params.get('dropout', 0.1)}")
        logging.info(f"  epochs={trans_params.get('epochs', 50)}")
        logging.info(f"  batch_size={trans_params.get('batch_size', 32)}")
        logging.info(f"  learning_rate={trans_params.get('learning_rate', 0.001)}")
        
        models['Transformer'] = {
            'params': trans_params,
            'device': device_type
        }
    
    if not models:
        return results, trained_models
    
    # For each horizon, train models ONCE per fold
    from tqdm import tqdm
    for horizon in horizons:
        X_seq, Y_seq = horizon_data[horizon]
        
        logging.info(f"\n📊 评估 horizon={horizon}:")
        logging.info(f"  序列长度: {sequence_length}")
        logging.info(f"  预测步长: {horizon}")
        logging.info(f"  预测目标: {target_cols}")
        if exog_cols:
            logging.info(f"  外生变量: {exog_cols}")
        
        # Cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X_seq)):
            logging.info(f"  Fold {fold_idx + 1}/{cv_splitter.n_splits} - Horizon {horizon}")
            
            X_train = X_seq[train_idx]
            Y_train = Y_seq[train_idx]
            X_test = X_seq[test_idx]
            Y_test = Y_seq[test_idx]
            
            logging.info(f"    训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
            
            # Train each model type
            for model_name, model_config in models.items():
                logging.info(f"    训练 {model_name} (horizon={horizon}, fold={fold_idx+1})...")
                
                # Create fresh model instance for this horizon
                if model_name == 'LSTM':
                    params = model_config['params']
                    model = LSTMForecaster(
                        hidden_size=params.get('hidden_size', 64),
                        num_layers=params.get('num_layers', 2),
                        dropout=params.get('dropout', 0.2),
                        epochs=params.get('epochs', 50),
                        batch_size=params.get('batch_size', 32),
                        learning_rate=params.get('learning_rate', 0.001),
                        device=model_config['device'],
                        n_horizons=1  # Direct strategy: one horizon at a time
                    )
                else:  # Transformer
                    params = model_config['params']
                    model = TransformerForecaster(
                        d_model=params.get('d_model', 64),
                        nhead=params.get('nhead', 4),
                        num_encoder_layers=params.get('num_encoder_layers', 2),
                        num_decoder_layers=params.get('num_decoder_layers', 2),
                        dim_feedforward=params.get('dim_feedforward', 256),
                        dropout=params.get('dropout', 0.1),
                        epochs=params.get('epochs', 50),
                        batch_size=params.get('batch_size', 32),
                        learning_rate=params.get('learning_rate', 0.001),
                        device=model_config['device'],
                        n_horizons=1  # Direct strategy: one horizon at a time
                    )
                
                # Fit model
                model.fit(X_train, Y_train)
                
                # Predict
                Y_pred = model.predict(X_test)
                
                # Evaluate for each target
                for target_idx, target in enumerate(target_cols):
                    y_true = Y_test[:, target_idx] if len(target_cols) > 1 else Y_test
                    y_pred = Y_pred[:, target_idx] if len(target_cols) > 1 else Y_pred
                    
                    metrics = eval_metrics(y_true, y_pred,
                                          metric_names=config.get('evaluation', 'metrics'))
                    
                    logging.info(f"      {model_name} ({target}): RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
                    
                    results.append({
                        'model': model_name,
                        'horizon': horizon,
                        'fold': fold_idx,
                        'target': target,
                        **metrics
                    })
                
                # Save trained model (last fold)
                if fold_idx == cv_splitter.n_splits - 1:
                    trained_models[f"{model_name}_h{horizon}"] = model
    
    return results, trained_models


def train_evaluate_deep_models(
    df: pd.DataFrame,
    horizon: int,
    cv_splitter: TimeSeriesSplit,
    config: Config
) -> List[Dict]:
    """
    Train and evaluate deep learning models (LSTM, Transformer)
    
    Args:
        df: DataFrame with P, Q and optionally exogenous columns
        horizon: Forecast horizon
        cv_splitter: Cross-validation splitter
        config: Configuration object
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Get target columns to predict
    target_cols = get_target_columns(config)
    
    # Prepare sequences
    sequence_length = config.get('features', 'sequence_length', default=24)
    exog_cols = config.get('features', 'exog_cols', default=[])
    
    logging.info(f"准备深度学习序列数据...")
    logging.info(f"  序列长度: {sequence_length}")
    logging.info(f"  预测步长: {horizon}")
    logging.info(f"  预测目标: {target_cols}")
    if exog_cols:
        logging.info(f"  外生变量: {exog_cols}")
    
    X_seq, Y_seq = prepare_sequences(df, sequence_length=sequence_length, 
                                     horizon=horizon, exog_cols=exog_cols,
                                     target_cols=target_cols)
    
    # Initialize models
    models = {}
    
    if config.get('models', 'lstm', 'enabled', default=False):
        lstm_params = config.get('models', 'lstm', default={})
        logging.info(f"\n初始化 LSTM 模型:")
        logging.info(f"  hidden_size={lstm_params.get('hidden_size', 64)}")
        logging.info(f"  num_layers={lstm_params.get('num_layers', 2)}")
        logging.info(f"  dropout={lstm_params.get('dropout', 0.2)}")
        logging.info(f"  epochs={lstm_params.get('epochs', 50)}")
        logging.info(f"  batch_size={lstm_params.get('batch_size', 32)}")
        logging.info(f"  learning_rate={lstm_params.get('learning_rate', 0.001)}")
        
        # Get device config
        device_type = config.get('device', 'type', default='cpu')
        
        models['LSTM'] = LSTMForecaster(
            hidden_size=lstm_params.get('hidden_size', 64),
            num_layers=lstm_params.get('num_layers', 2),
            dropout=lstm_params.get('dropout', 0.2),
            epochs=lstm_params.get('epochs', 50),
            batch_size=lstm_params.get('batch_size', 32),
            learning_rate=lstm_params.get('learning_rate', 0.001),
            device=device_type
        )
    
    if config.get('models', 'transformer', 'enabled', default=False):
        trans_params = config.get('models', 'transformer', default={})
        logging.info(f"\n初始化 Transformer 模型:")
        logging.info(f"  d_model={trans_params.get('d_model', 64)}")
        logging.info(f"  nhead={trans_params.get('nhead', 4)}")
        logging.info(f"  num_encoder_layers={trans_params.get('num_encoder_layers', 2)}")
        logging.info(f"  num_decoder_layers={trans_params.get('num_decoder_layers', 2)}")
        logging.info(f"  dim_feedforward={trans_params.get('dim_feedforward', 256)}")
        logging.info(f"  dropout={trans_params.get('dropout', 0.1)}")
        logging.info(f"  epochs={trans_params.get('epochs', 50)}")
        logging.info(f"  batch_size={trans_params.get('batch_size', 32)}")
        logging.info(f"  learning_rate={trans_params.get('learning_rate', 0.001)}")
        
        # Get device config
        device_type = config.get('device', 'type', default='cpu')
        
        models['Transformer'] = TransformerForecaster(
            d_model=trans_params.get('d_model', 64),
            nhead=trans_params.get('nhead', 4),
            num_encoder_layers=trans_params.get('num_encoder_layers', 2),
            num_decoder_layers=trans_params.get('num_decoder_layers', 2),
            dim_feedforward=trans_params.get('dim_feedforward', 256),
            dropout=trans_params.get('dropout', 0.1),
            epochs=trans_params.get('epochs', 50),
            batch_size=trans_params.get('batch_size', 32),
            learning_rate=trans_params.get('learning_rate', 0.001),
            device=device_type
        )
    
    if not models:
        return results
    
    # Cross-validation
    from tqdm import tqdm
    for fold_idx, (train_idx, test_idx) in tqdm(enumerate(cv_splitter.split(X_seq))):
        logging.info(f"  Fold {fold_idx + 1}/{cv_splitter.n_splits} - Deep learning models")
        
        X_train = X_seq[train_idx]
        Y_train = Y_seq[train_idx]
        X_test = X_seq[test_idx]
        Y_test = Y_seq[test_idx]
        
        logging.info(f"    训练集大小: {len(X_train)} 样本, 测试集大小: {len(X_test)} 样本")
        
        # Train each model
        for model_name, model in models.items():
            logging.info(f"    训练 {model_name} 模型...")
            logging.info(f"      配置: 输入序列长度={X_train.shape[1]}, 特征维度={X_train.shape[2]}, 输出维度={Y_train.shape[1]}")
            
            # Fit model
            model.fit(X_train, Y_train)
            
            logging.info(f"      {model_name} 训练完成，开始预测...")
            
            # Predict
            Y_pred = model.predict(X_test)
            
            # Evaluate for each target
            for target_idx, target in enumerate(target_cols):
                y_true = Y_test[:, target_idx] if len(target_cols) > 1 else Y_test
                y_pred = Y_pred[:, target_idx] if len(target_cols) > 1 else Y_pred
                
                metrics = eval_metrics(y_true, y_pred,
                                      metric_names=config.get('evaluation', 'metrics'))
                
                logging.info(f"      {model_name} ({target}): RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
                
                results.append({
                    'model': model_name,
                    'horizon': horizon,
                    'fold': fold_idx,
                    'target': target,
                    **metrics
                })
    
    return results


def run_evaluation(config: Config, df: pd.DataFrame, metrics_dir: str = "outputs/metrics") -> tuple:
    """
    Run full evaluation pipeline
    
    Args:
        config: Configuration object
        df: DataFrame with features
        metrics_dir: Directory to save evaluation metrics
    
    Returns:
        Tuple of (results_df, trained_models)
    """
    all_results = []
    trained_models = {}
    
    # Get horizons
    horizons = config.get('evaluation', 'horizons', default=[1, 12, 24])
    test_window = config.get('evaluation', 'test_window', default=100)
    n_splits = config.get('evaluation', 'n_splits', default=3)
    
    # Prepare features once
    max_lag = config.get('features', 'max_lag', default=24)
    roll_windows = config.get('features', 'roll_windows', default=[3, 6, 12])
    use_time_features = config.get('features', 'use_time_features', default=True)
    exog_cols = config.get('features', 'exog_cols', default=[])
    
    X, Y = create_features(df, max_lag=max_lag, roll_windows=roll_windows, 
                          use_time_features=use_time_features, exog_cols=exog_cols)
    
    # Train deep learning models ONCE for all horizons
    # Then evaluate on each horizon separately
    cv_splitter = TimeSeriesSplit(test_window=test_window, n_splits=n_splits)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"步骤 3.1: 训练深度学习模型 (一次性训练，所有步长共享)")
    logging.info(f"{'='*60}")
    logging.info(f"🔧 训练深度学习模型 (将在所有{len(horizons)}个预测步长上评估)...")
    deep_results, deep_models_trained = train_evaluate_deep_models_once(
        df, horizons, cv_splitter, config
    )
    logging.info(f"✓ 深度学习模型训练完成")
    logging.info("")
    
    # For each horizon: evaluate baseline and tree models
    for horizon_idx, horizon in enumerate(horizons):
        logging.info(f"\n{'='*60}")
        logging.info(f"步骤 3.{horizon_idx+2}: 评估预测步长 = {horizon}")
        logging.info(f"{'='*60}")
        
        logging.info(f"交叉验证配置: {n_splits} 折, 测试窗口大小={test_window}")
        
        # Baseline models
        logging.info(f"\n[1/3] 评估基线模型...")
        baseline_results = train_evaluate_baseline_models(Y, horizon, cv_splitter, config)
        all_results.extend(baseline_results)
        logging.info(f"✓ 基线模型评估完成 ({len(baseline_results)} 个结果)")
        
        # Tree models
        logging.info(f"\n[2/3] 评估树模型...")
        tree_results = train_evaluate_tree_models(X, Y, horizon, cv_splitter, config)
        all_results.extend(tree_results)
        logging.info(f"✓ 树模型评估完成 ({len(tree_results)} 个结果)")
        
        # Deep learning models - reuse results from batch training
        logging.info(f"\n[3/3] 添加深度学习模型结果...")
        deep_results_current = [r for r in deep_results if r['horizon'] == horizon]
        all_results.extend(deep_results_current)
        logging.info(f"✓ 深度学习模型结果已添加 ({len(deep_results_current)} 个结果)")
    
    # Combine results and return
    results_df = pd.DataFrame(all_results)
    trained_models.update(deep_models_trained)
    
    return results_df, trained_models
