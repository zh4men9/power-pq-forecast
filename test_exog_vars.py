#!/usr/bin/env python
"""
Test script for exogenous variables functionality
Demonstrates usage of the system with and without external variables
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.data_io import load_data
from src.features import create_features
from src.plots import plot_feature_importance, configure_chinese_fonts
from src.models.tree import RandomForestForecaster
import pandas as pd
import numpy as np

def test_without_exog():
    """Test with P and Q only"""
    print("=" * 60)
    print("测试 1: 仅使用 P 和 Q 进行预测")
    print("=" * 60)
    
    # Load data without exog
    df = load_data(
        'data/raw/电气多特征.csv',
        time_col=None,
        p_col=None,
        q_col=None,
        exog_cols=None,
        freq='H',
        tz=None,
        interp_limit=3
    )
    
    print(f"\n数据形状: {df.shape}")
    print(f"列: {list(df.columns)}")
    
    # Create features
    X, Y = create_features(
        df,
        max_lag=12,
        roll_windows=[6, 12],
        use_time_features=True,
        exog_cols=None
    )
    
    print(f"\n特征矩阵 X: {X.shape}")
    print(f"目标矩阵 Y: {Y.shape}")
    print(f"总特征数: {X.shape[1]}")
    
    # Train a simple RF model
    print("\n训练随机森林模型...")
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    model = RandomForestForecaster(n_estimators=50, max_depth=10, n_jobs=-1)
    model.fit(X_train, Y_train)
    
    # Get predictions
    Y_pred = model.predict(X_test)
    
    # Calculate simple RMSE
    rmse_p = np.sqrt(np.mean((Y_test['P'].values - Y_pred[:, 0]) ** 2))
    rmse_q = np.sqrt(np.mean((Y_test['Q'].values - Y_pred[:, 1]) ** 2))
    
    print(f"\n测试集 RMSE:")
    print(f"  P: {rmse_p:.2f}")
    print(f"  Q: {rmse_q:.2f}")
    
    return model, X

def test_with_exog():
    """Test with exogenous variables"""
    print("\n" + "=" * 60)
    print("测试 2: 使用外生变量进行预测")
    print("=" * 60)
    
    # Load data with exog
    exog_cols = ['定子电流', '定子电压', '转子电压', '转子电流']
    df = load_data(
        'data/raw/电气多特征.csv',
        time_col=None,
        p_col=None,
        q_col=None,
        exog_cols=exog_cols,
        freq='H',
        tz=None,
        interp_limit=3
    )
    
    print(f"\n数据形状: {df.shape}")
    print(f"列: {list(df.columns)}")
    
    # Create features
    X, Y = create_features(
        df,
        max_lag=12,
        roll_windows=[6, 12],
        use_time_features=True,
        exog_cols=exog_cols
    )
    
    print(f"\n特征矩阵 X: {X.shape}")
    print(f"目标矩阵 Y: {Y.shape}")
    print(f"总特征数: {X.shape[1]}")
    
    # Train a simple RF model
    print("\n训练随机森林模型...")
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    model = RandomForestForecaster(n_estimators=50, max_depth=10, n_jobs=-1)
    model.fit(X_train, Y_train)
    
    # Get predictions
    Y_pred = model.predict(X_test)
    
    # Calculate simple RMSE
    rmse_p = np.sqrt(np.mean((Y_test['P'].values - Y_pred[:, 0]) ** 2))
    rmse_q = np.sqrt(np.mean((Y_test['Q'].values - Y_pred[:, 1]) ** 2))
    
    print(f"\n测试集 RMSE:")
    print(f"  P: {rmse_p:.2f}")
    print(f"  Q: {rmse_q:.2f}")
    
    # Get feature importance
    print("\n获取特征重要性...")
    importance = model.get_feature_importance()
    
    print("\nTop 10 最重要特征:")
    for i, (feat, imp) in enumerate(list(importance.items())[:10], 1):
        print(f"  {i}. {feat}: {imp:.4f}")
    
    # Plot feature importance
    print("\n生成特征重要性图表...")
    configure_chinese_fonts(['SimHei', 'Microsoft YaHei', 'STHeiti', 'PingFang SC'])
    plot_feature_importance(
        importance,
        top_n=20,
        output_path='outputs/figures/feature_importance_test.png',
        dpi=150
    )
    
    return model, X, importance

def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("外生变量功能测试")
    print("=" * 60)
    
    # Test without exog
    model1, X1 = test_without_exog()
    
    # Test with exog
    model2, X2, importance = test_with_exog()
    
    # Compare
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"不使用外生变量: {X1.shape[1]} 个特征")
    print(f"使用外生变量:   {X2.shape[1]} 个特征")
    print(f"增加特征数:     {X2.shape[1] - X1.shape[1]} 个")
    print(f"\n特征重要性图表已保存到: outputs/figures/feature_importance_test.png")
    print("\n测试成功! ✓")
    print("=" * 60)

if __name__ == '__main__':
    main()
