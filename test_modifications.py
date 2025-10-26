"""
快速测试脚本 - 验证所有修改
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("测试1: 检查配置文件")
print("="*60)

from src.config import load_config

config = load_config('config_p_only.yaml')
print("✓ 配置文件加载成功")
print(f"  预测P: {config.get('target', 'predict_p')}")
print(f"  预测Q: {config.get('target', 'predict_q')}")
print(f"  评估指标: {config.get('evaluation', 'metrics')}")
print(f"  外生变量数: {len(config.get('features', 'exog_cols', default=[]))}")
print(f"  缺失值填补方法: {config.get('data', 'imputation', 'method')}")
print()

print("="*60)
print("测试2: 数据加载和缺失值填补")
print("="*60)

from src.data_io import load_data

data_file = Path('data/raw/电气多特征.csv')
if data_file.exists():
    imputation_config = config.get('data', 'imputation', default={})
    df = load_data(
        file_path=str(data_file),
        time_col=config.get('data', 'time_col'),
        p_col=config.get('data', 'p_col'),
        q_col=config.get('data', 'q_col'),
        exog_cols=config.get('features', 'exog_cols', default=[]),
        freq=config.get('data', 'freq'),
        tz=config.get('data', 'tz'),
        interp_limit=config.get('data', 'interp_limit', default=3),
        imputation_method=imputation_config.get('method'),
        target_p_value=imputation_config.get('target_p_value', 280.0)
    )
    print(f"✓ 数据加载成功: {df.shape}")
    print(f"  列: {list(df.columns)}")
    print(f"  缺失值: P={df['P'].isna().sum()}, Q={df['Q'].isna().sum()}")
else:
    print("✗ 数据文件不存在")
    df = None
print()

print("="*60)
print("测试3: ACC指标计算")
print("="*60)

from src.metrics import acc, eval_metrics

y_true = np.array([100, 200, 300, 400, 500])
y_pred = np.array([105, 195, 310, 390, 520])

acc_value = acc(y_true, y_pred, threshold=0.1)
print(f"✓ ACC指标计算成功: {acc_value:.2f}%")

all_metrics = eval_metrics(y_true, y_pred, metric_names=['RMSE', 'MAE', 'SMAPE', 'WAPE', 'ACC'])
print(f"  所有指标: {list(all_metrics.keys())}")
for k, v in all_metrics.items():
    print(f"    {k}: {v:.4f}")
print()

print("="*60)
print("测试4: 目标列动态获取")
print("="*60)

from src.train_eval import get_target_columns

target_cols = get_target_columns(config)
print(f"✓ 目标列获取成功: {target_cols}")
print()

print("="*60)
print("测试5: 特征准备（只预测P）")
print("="*60)

if df is not None:
    from src.features import prepare_sequences
    
    X_seq, Y_seq = prepare_sequences(
        df,
        sequence_length=24,
        horizon=1,
        exog_cols=config.get('features', 'exog_cols', default=[]),
        target_cols=target_cols
    )
    print(f"✓ 序列准备成功")
    print(f"  X shape: {X_seq.shape}")
    print(f"  Y shape: {Y_seq.shape}")
    print(f"  Y列数 = 预测目标数: {Y_seq.shape[1]} = {len(target_cols)}")
print()

print("="*60)
print("测试6: 绘图函数")
print("="*60)

from src.plots import configure_chinese_fonts, plot_all_metrics_by_horizon

configure_chinese_fonts()
print("✓ 中文字体配置成功")

# 创建模拟数据
mock_data = []
models = ['Naive', 'SeasonalNaive', 'RandomForest', 'XGBoost', 'LSTM', 'Transformer']
horizons = [1, 12, 24]
for model in models:
    for horizon in horizons:
        for fold in range(3):
            mock_data.append({
                'model': model,
                'horizon': horizon,
                'fold': fold,
                'target': 'P',
                'RMSE': np.random.uniform(50, 200),
                'MAE': np.random.uniform(30, 150),
                'SMAPE': np.random.uniform(5, 25),
                'WAPE': np.random.uniform(5, 25),
                'ACC': np.random.uniform(70, 95)
            })

mock_df = pd.DataFrame(mock_data)
output_dir = Path('outputs/test')
output_dir.mkdir(parents=True, exist_ok=True)

try:
    plot_all_metrics_by_horizon(
        mock_df,
        metrics=['RMSE', 'MAE', 'SMAPE', 'WAPE', 'ACC'],
        output_path=str(output_dir / 'test_all_metrics.png'),
        dpi=100
    )
    print("✓ 多指标对比图生成成功")
    print(f"  保存位置: {output_dir / 'test_all_metrics.png'}")
except Exception as e:
    print(f"✗ 绘图失败: {e}")
print()

print("="*60)
print("所有测试完成！")
print("="*60)
print("\n下一步:")
print("1. 运行完整训练: python run_all.py --config config_p_only.yaml")
print("2. 查看结果: outputs/latest/")
print("3. 生成预测: python forecast_future.py --config config_p_only.yaml")
