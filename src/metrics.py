"""
Evaluation metrics module for time series forecasting
Implements RMSE, MAE, SMAPE, WAPE with proper handling of edge cases
"""
import numpy as np
from typing import Dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error
    
    Definition: 100% * mean(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
    
    This metric is symmetric and bounded, but can still have issues when
    both y_true and y_pred are close to zero. We add eps to the denominator
    to prevent division by zero.
    
    Reference: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        eps: Small constant to prevent division by zero
    
    Returns:
        SMAPE value as percentage (0-200)
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return 100.0 * np.mean(np.abs(y_true - y_pred) / denominator)


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Weighted Absolute Percentage Error (also known as MAD/Mean ratio)
    
    Definition: 100% * sum(|y_true - y_pred|) / sum(|y_true|)
    
    Unlike MAPE, WAPE does not divide by individual values, making it more
    robust when y_true contains values close to zero. It represents the
    overall percentage error weighted by the magnitude of true values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        eps: Small constant to prevent division by zero
    
    Returns:
        WAPE value as percentage
    """
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true)) + eps
    return 100.0 * numerator / denominator


def acc(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.05) -> float:
    """
    近似准确率 (Approximate Accuracy)
    
    Definition: 100% * count(|y_true - y_pred| / |y_true| <= threshold) / len(y_true)
    
    计算预测值与真实值相对误差在阈值内的样本占比
    默认阈值为5%，即预测误差在±5%以内视为准确
    
    Args:
        y_true: True values
        y_pred: Predicted values
        threshold: Relative error threshold (default 0.05 for 5%)
    
    Returns:
        ACC value as percentage (0-100)
    """
    # 避免除以零
    mask = np.abs(y_true) > 1e-8
    if not mask.any():
        return 0.0
    
    relative_errors = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
    accurate_count = np.sum(relative_errors <= threshold)
    total_count = len(relative_errors)
    
    return 100.0 * accurate_count / total_count


def eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: list = None
) -> Dict[str, float]:
    """
    Evaluate multiple metrics at once
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metric_names: List of metric names to compute
                      Default: ['RMSE', 'MAE', 'SMAPE', 'WAPE', 'ACC']
    
    Returns:
        Dictionary mapping metric names to values
    """
    if metric_names is None:
        metric_names = ['RMSE', 'MAE', 'SMAPE', 'WAPE', 'ACC']
    
    metric_funcs = {
        'RMSE': rmse,
        'MAE': mae,
        'SMAPE': smape,
        'WAPE': wape,
        'ACC': acc
    }
    
    results = {}
    for name in metric_names:
        name_upper = name.upper()
        if name_upper in metric_funcs:
            results[name_upper] = metric_funcs[name_upper](y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {name}")
    
    return results
