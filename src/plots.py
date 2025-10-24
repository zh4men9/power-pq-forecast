"""
Plotting utilities with Chinese language support
All plots use Chinese labels and proper font configuration
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict


def configure_chinese_fonts(font_priority: Optional[List[str]] = None):
    """
    Configure matplotlib for Chinese character display
    
    Reference: https://jdhao.github.io/2017/05/13/guide-on-how-to-use-chinese-with-matplotlib/
    
    Args:
        font_priority: List of font names in priority order
    """
    if font_priority is None:
        font_priority = [
            'SimHei',              # Windows
            'STHeiti',             # macOS
            'WenQuanYi Micro Hei', # Linux
            'Noto Sans CJK SC',    # Linux/Cross-platform
            'Microsoft YaHei',     # Windows
            'PingFang SC',         # macOS
            'Heiti SC',            # macOS
            'DejaVu Sans',         # Fallback
            'Arial Unicode MS'     # Fallback
        ]
    
    matplotlib.rcParams['font.sans-serif'] = font_priority
    matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    matplotlib.rcParams['font.size'] = 10


def plot_overview(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/data_overview.png",
    dpi: int = 150
):
    """
    Plot data overview with P and Q
    
    Args:
        df: DataFrame with P and Q columns
        output_path: Path to save figure
        dpi: Resolution
    """
    configure_chinese_fonts()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(df.index, df['P'], linewidth=0.5, alpha=0.8, label='有功功率 P')
    axes[0].set_ylabel('有功功率 P')
    axes[0].set_title('电力数据总览', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(df.index, df['Q'], linewidth=0.5, alpha=0.8, color='orange', label='无功功率 Q')
    axes[1].set_ylabel('无功功率 Q')
    axes[1].set_xlabel('时间')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Overview plot saved to {output_path}")


def plot_missing_data(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/missing_data.png",
    dpi: int = 150
):
    """
    Plot missing data heatmap
    
    Args:
        df: DataFrame to check for missing values
        output_path: Path to save figure
        dpi: Resolution
    """
    configure_chinese_fonts()
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    missing = df.isna().astype(int)
    
    if missing.sum().sum() > 0:
        im = ax.imshow(missing.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_yticks(range(len(df.columns)))
        ax.set_yticklabels(df.columns)
        ax.set_xlabel('时间索引')
        ax.set_title('缺失值分布图', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='缺失(1) / 存在(0)')
    else:
        ax.text(0.5, 0.5, '数据完整，无缺失值', ha='center', va='center', fontsize=16)
        ax.set_title('缺失值分布图', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Missing data plot saved to {output_path}")


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_index: Optional[pd.DatetimeIndex] = None,
    target_name: str = 'P',
    model_name: str = 'Model',
    output_path: str = "outputs/figures/prediction_comparison.png",
    dpi: int = 150
):
    """
    Plot prediction vs actual comparison
    
    Args:
        y_true: True values
        y_pred: Predicted values
        time_index: Time index for x-axis
        target_name: Name of target (P or Q)
        model_name: Name of model
        output_path: Path to save figure
        dpi: Resolution
    """
    configure_chinese_fonts()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if time_index is not None:
        x = time_index
    else:
        x = np.arange(len(y_true))
    
    ax.plot(x, y_true, label='实际值', alpha=0.7, linewidth=1.5)
    ax.plot(x, y_pred, label='预测值', alpha=0.7, linewidth=1.5, linestyle='--')
    
    ax.set_xlabel('时间')
    ax.set_ylabel(f'{target_name} 值')
    ax.set_title(f'{model_name} - {target_name} 预测对比', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction comparison plot saved to {output_path}")


def plot_error_by_horizon(
    metrics_df: pd.DataFrame,
    metric_name: str = 'RMSE',
    output_path: str = "outputs/figures/error_by_horizon.png",
    dpi: int = 150
):
    """
    Plot error metrics by forecast horizon for different models
    
    Args:
        metrics_df: DataFrame with columns [model, horizon, target, metric_name]
        metric_name: Name of metric to plot
        output_path: Path to save figure
        dpi: Resolution
    """
    configure_chinese_fonts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    targets = ['P', 'Q']
    target_names = {'P': '有功功率', 'Q': '无功功率'}
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        # Filter data for this target
        target_data = metrics_df[metrics_df['target'] == target]
        
        # Group by model and horizon
        for model in target_data['model'].unique():
            model_data = target_data[target_data['model'] == model]
            grouped = model_data.groupby('horizon')[metric_name].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=model, linewidth=2)
        
        ax.set_xlabel('预测步长')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{target_names[target]} - {metric_name} vs 步长', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Error by horizon plot saved to {output_path}")


def plot_feature_importance(
    importance_dict: Dict[str, float],
    top_n: int = 20,
    output_path: str = "outputs/figures/feature_importance.png",
    dpi: int = 150
):
    """
    Plot feature importance for tree-based models
    
    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        top_n: Number of top features to display
        output_path: Path to save figure
        dpi: Resolution
    """
    configure_chinese_fonts()
    
    # Get top N features
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('重要性得分')
    ax.set_title(f'特征重要性 (Top {top_n})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")


def save_all_figures(output_dir: str = "outputs/figures"):
    """
    Ensure output directory exists
    
    Args:
        output_dir: Directory to save figures
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
