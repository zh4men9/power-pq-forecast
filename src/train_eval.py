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
        df: DataFrame with P and Q columns
        metrics_dir: Directory to save metrics
    
    Returns:
        Tuple of (results_df, trained_models_dict)
        - results_df: DataFrame with all evaluation results
        - trained_models_dict: Dictionary of trained models {model_name: model_object}
    """
    horizons = config.get('evaluation', 'horizons', default=[1, 12, 24])
    test_window = config.get('evaluation', 'test_window', default=100)
    n_splits = config.get('evaluation', 'n_splits', default=3)
    
    all_results = []
    trained_models = {}  # 保存训练好的模型
    
    # Create features for tree models
    max_lag = config.get('features', 'max_lag', default=24)
    roll_windows = config.get('features', 'roll_windows', default=[6, 12, 24])
    use_time_features = config.get('features', 'use_time_features', default=True)
    exog_cols = config.get('features', 'exog_cols', default=[])
    
    X, Y = create_features(df, max_lag=max_lag, roll_windows=roll_windows, 
                          use_time_features=use_time_features, exog_cols=exog_cols)
    
    # For each horizon
    for horizon in horizons:
        logging.info(f"\n{'='*60}")
        logging.info(f"评估预测步长 = {horizon}")
        logging.info(f"{'='*60}")
        
        # Create CV splitter
        cv_splitter = TimeSeriesSplit(test_window=test_window, n_splits=n_splits)
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
        
        # Deep learning models
        logging.info(f"\n[3/3] 评估深度学习模型...")
        deep_results = train_evaluate_deep_models(df, horizon, cv_splitter, config)
        all_results.extend(deep_results)
        logging.info(f"✓ 深度学习模型评估完成 ({len(deep_results)} 个结果)")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_dir = Path(metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "cv_metrics.csv", index=False)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"所有评估完成！共 {len(all_results)} 个评估结果")
    logging.info(f"结果已保存至: {output_dir / 'cv_metrics.csv'}")
    logging.info(f"{'='*60}")
    
    # Train final models on full data for prediction
    logging.info(f"\n{'='*60}")
    logging.info("训练最终模型（使用全部数据）用于预测...")
    logging.info(f"{'='*60}")
    
    target_cols = get_target_columns(config)
    Y_full = df[target_cols]
    
    # Train baseline models
    if config.get('models', 'naive', 'enabled', default=True):
        model = NaiveForecaster()
        model.fit(None, Y_full)
        trained_models['Naive'] = model
        logging.info("✓ Naive 模型训练完成")
    
    if config.get('models', 'seasonal_naive', 'enabled', default=True):
        season_length = config.get('features', 'season_length', default=24)
        model = SeasonalNaiveForecaster(season_length=season_length)
        model.fit(None, Y_full)
        trained_models['SeasonalNaive'] = model
        logging.info("✓ SeasonalNaive 模型训练完成")
    
    # Prepare features for tree models (use horizon=1 for final model)
    # X and Y already have aligned indices after create_features
    # Y has only the rows that have sufficient history for lag features
    Y_h = Y[target_cols].shift(-1).dropna()
    X_h = X.loc[Y_h.index]
    
    # Train tree models
    if config.get('models', 'rf', 'enabled', default=False):
        rf_params = config.get('models', 'rf', default={})
        model = RandomForestForecaster(
            n_estimators=rf_params.get('n_estimators', 100),
            max_depth=rf_params.get('max_depth'),
            n_jobs=rf_params.get('n_jobs', -1),
            random_state=42
        )
        model.fit(X_h, Y_h)
        trained_models['RandomForest'] = model
        logging.info("✓ RandomForest 模型训练完成")
    
    if config.get('models', 'xgb', 'enabled', default=False):
        xgb_params = config.get('models', 'xgb', default={})
        model = XGBoostForecaster(
            n_estimators=xgb_params.get('n_estimators', 100),
            max_depth=xgb_params.get('max_depth', 6),
            learning_rate=xgb_params.get('learning_rate', 0.1),
            n_jobs=xgb_params.get('n_jobs', -1),
            random_state=42
        )
        model.fit(X_h, Y_h)
        trained_models['XGBoost'] = model
        logging.info("✓ XGBoost 模型训练完成")
    
    # Train deep learning models
    sequence_length = config.get('features', 'sequence_length', default=24)
    
    if config.get('models', 'lstm', 'enabled', default=False):
        lstm_params = config.get('models', 'lstm', default={})
        model = LSTMForecaster(
            hidden_size=lstm_params.get('hidden_size', 64),
            num_layers=lstm_params.get('num_layers', 2),
            dropout=lstm_params.get('dropout', 0.2),
            epochs=lstm_params.get('epochs', 50),
            batch_size=lstm_params.get('batch_size', 32),
            learning_rate=lstm_params.get('learning_rate', 0.001)
        )
        X_seq, Y_seq = prepare_sequences(df, sequence_length=sequence_length, 
                                         horizon=1, exog_cols=exog_cols,
                                         target_cols=target_cols)
        model.fit(X_seq, Y_seq)
        trained_models['LSTM'] = model
        logging.info("✓ LSTM 模型训练完成")
    
    if config.get('models', 'transformer', 'enabled', default=False):
        trans_params = config.get('models', 'transformer', default={})
        model = TransformerForecaster(
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
        X_seq, Y_seq = prepare_sequences(df, sequence_length=sequence_length, 
                                         horizon=1, exog_cols=exog_cols,
                                         target_cols=target_cols)
        model.fit(X_seq, Y_seq)
        trained_models['Transformer'] = model
        logging.info("✓ Transformer 模型训练完成")
    
    logging.info(f"✓ 共训练 {len(trained_models)} 个最终模型")
    logging.info(f"{'='*60}")
    
    return results_df, trained_models
