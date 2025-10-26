"""
Model management utilities for saving and loading trained models
"""
import pickle
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging


def save_model(model, model_name: str, output_dir: str, metadata: Dict[str, Any] = None):
    """
    保存训练好的模型
    
    Args:
        model: 训练好的模型对象
        model_name: 模型名称
        output_dir: 输出目录
        metadata: 模型元数据（如性能指标、配置等）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_file = output_path / f"{model_name}.pkl"
    
    # 对于深度学习模型，需要特殊处理
    if hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
        # PyTorch模型
        save_dict = {
            'model_class': type(model).__name__,
            'model_state_dict': model.model.state_dict(),
            'model_params': {
                'hidden_size': getattr(model, 'hidden_size', None),
                'num_layers': getattr(model, 'num_layers', None),
                'dropout': getattr(model, 'dropout', None),
                'd_model': getattr(model, 'd_model', None),
                'nhead': getattr(model, 'nhead', None),
                'num_encoder_layers': getattr(model, 'num_encoder_layers', None),
                'num_decoder_layers': getattr(model, 'num_decoder_layers', None),
                'dim_feedforward': getattr(model, 'dim_feedforward', None),
            },
            'device': str(model.device),
            'metadata': metadata
        }
        torch.save(save_dict, model_file)
    else:
        # Sklearn模型或基线模型
        save_dict = {
            'model': model,
            'model_class': type(model).__name__,
            'metadata': metadata
        }
        with open(model_file, 'wb') as f:
            pickle.dump(save_dict, f)
    
    logging.info(f"✓ 模型已保存: {model_file}")
    return str(model_file)


def load_model(model_file: str):
    """
    加载保存的模型
    
    Args:
        model_file: 模型文件路径
    
    Returns:
        加载的模型对象
    """
    model_path = Path(model_file)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    if model_path.suffix == '.pkl':
        with open(model_file, 'rb') as f:
            save_dict = pickle.load(f)
        
        if 'model_state_dict' in save_dict:
            # PyTorch模型需要重建
            logging.warning("PyTorch模型需要手动重建，请使用训练时的配置")
            return save_dict
        else:
            # Sklearn模型或基线模型
            return save_dict['model']
    else:
        raise ValueError(f"不支持的模型文件格式: {model_path.suffix}")


def get_best_model_info(metrics_df: pd.DataFrame, target: str = 'P', 
                        metric: str = 'RMSE') -> Dict[str, Any]:
    """
    从评估结果中获取最优模型信息
    
    Args:
        metrics_df: 评估指标DataFrame
        target: 目标变量 ('P' 或 'Q')
        metric: 评估指标 ('RMSE', 'MAE', etc.)
    
    Returns:
        包含最优模型信息的字典
    """
    # 筛选目标变量的数据
    target_data = metrics_df[metrics_df['target'] == target]
    
    # 按模型和步长分组，计算平均指标
    grouped = target_data.groupby(['model', 'horizon'])[metric].mean().reset_index()
    
    # 找到各个步长的最优模型
    best_by_horizon = {}
    for horizon in grouped['horizon'].unique():
        horizon_data = grouped[grouped['horizon'] == horizon]
        best_idx = horizon_data[metric].idxmin()
        best_model = horizon_data.loc[best_idx, 'model']
        best_value = horizon_data.loc[best_idx, metric]
        best_by_horizon[horizon] = {
            'model': best_model,
            metric: best_value
        }
    
    # 找到总体最优模型（平均表现最好）
    overall_best = grouped.groupby('model')[metric].mean().idxmin()
    overall_best_value = grouped.groupby('model')[metric].mean().min()
    
    return {
        'overall_best': overall_best,
        'overall_best_value': overall_best_value,
        'best_by_horizon': best_by_horizon,
        'target': target,
        'metric': metric
    }


def make_future_forecast(
    df: pd.DataFrame,
    model,
    model_name: str,
    start_date: str,
    end_date: str,
    config: Any
) -> pd.DataFrame:
    """
    使用训练好的模型进行未来预测
    
    Args:
        df: 历史数据
        model: 训练好的模型
        model_name: 模型名称
        start_date: 预测开始日期
        end_date: 预测结束日期
        config: 配置对象
    
    Returns:
        预测结果DataFrame
    """
    from datetime import datetime, timedelta
    from src.features import create_features, prepare_sequences
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    logging.info(f"\n使用 {model_name} 模型进行预测...")
    logging.info(f"预测时间范围: {start_dt.date()} 至 {end_dt.date()}")
    
    # 生成预测时间点
    forecast_dates = pd.date_range(start=start_dt, end=end_dt + timedelta(days=1), 
                                   freq='H', inclusive='left')
    
    logging.info(f"预测时间点数: {len(forecast_dates)}")
    
    # 检查是否需要深度学习序列
    is_dl_model = model_name in ['LSTM', 'Transformer']
    
    results = []
    
    if is_dl_model:
        # 深度学习模型：使用滑动窗口预测
        sequence_length = config.get('features', 'sequence_length', default=24)
        exog_cols = config.get('features', 'exog_cols', default=[])
        
        logging.info(f"使用序列长度: {sequence_length}")
        
        # 使用最后的sequence_length个数据点作为初始输入
        last_sequence = df.iloc[-sequence_length:].copy()
        
        for date in forecast_dates:
            # 准备输入序列
            if exog_cols:
                input_cols = ['P', 'Q'] + [col for col in exog_cols if col in last_sequence.columns]
            else:
                input_cols = ['P', 'Q']
            
            X_input = last_sequence[input_cols].values
            X_input = X_input.reshape(1, sequence_length, len(input_cols))
            
            # 预测
            pred = model.predict(X_input)
            pred_value = pred[0, 0] if pred.ndim > 1 else pred[0]
            
            results.append({
                '时间': date,
                '预测值': round(float(pred_value), 2)
            })
            
            # 更新序列：添加预测值，移除最早的值
            new_row = last_sequence.iloc[-1].copy()
            new_row.name = date
            new_row['P'] = pred_value
            last_sequence = pd.concat([last_sequence.iloc[1:], pd.DataFrame([new_row])])
    
    else:
        # 树模型或基线模型：使用特征工程
        max_lag = config.get('features', 'max_lag', default=24)
        
        # 使用历史数据的最后部分进行预测
        # 对于每个预测点，使用前面的数据创建特征
        extended_df = df.copy()
        
        for i, date in enumerate(forecast_dates):
            if len(extended_df) < max_lag:
                logging.warning(f"历史数据不足 {max_lag} 行，无法创建特征")
                break
            
            # 创建特征
            X, Y = create_features(
                extended_df,
                max_lag=max_lag,
                roll_windows=config.get('features', 'roll_windows', default=[6, 12, 24]),
                use_time_features=config.get('features', 'use_time_features', default=True),
                exog_cols=config.get('features', 'exog_cols', default=[])
            )
            
            if len(X) == 0:
                break
            
            # 使用最后一行特征进行预测
            X_last = X.iloc[-1:].values
            
            try:
                pred = model.predict(X_last)
                pred_value = pred[0, 0] if pred.ndim > 1 else pred[0]
                
                results.append({
                    '时间': date,
                    '预测值': round(float(pred_value), 2)
                })
                
                # 将预测值添加到历史数据中，用于下一步预测
                new_row = pd.Series({
                    'P': pred_value,
                    'Q': extended_df['Q'].iloc[-1]  # Q使用最后的值
                }, name=date)
                
                # 添加外生变量（如果有）
                for col in extended_df.columns:
                    if col not in ['P', 'Q']:
                        new_row[col] = extended_df[col].iloc[-1]
                
                extended_df = pd.concat([extended_df, pd.DataFrame([new_row])])
                
            except Exception as e:
                logging.error(f"预测失败: {e}")
                break
    
    forecast_df = pd.DataFrame(results)
    
    if len(forecast_df) > 0:
        logging.info(f"✓ 预测完成，共 {len(forecast_df)} 个时间点")
        logging.info(f"  预测均值: {forecast_df['预测值'].mean():.2f}")
        logging.info(f"  预测范围: [{forecast_df['预测值'].min():.2f}, {forecast_df['预测值'].max():.2f}]")
    else:
        logging.warning("⚠️  未生成任何预测结果")
    
    return forecast_df
