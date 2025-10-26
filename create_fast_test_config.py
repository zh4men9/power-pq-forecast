"""
快速端到端测试 - 使用小规模配置
"""
import yaml
from pathlib import Path

# 创建一个快速测试配置
fast_config = {
    'data': {
        'data_path': 'data/raw',
        'file_pattern': '*.csv',
        'time_col': '时间',
        'p_col': '有功',
        'q_col': '无功',
        'freq': 'H',
        'tz': None,
        'interp_limit': 3,
        'imputation': {
            'method': 'nearest_p',
            'target_p_value': 280.0
        }
    },
    'target': {
        'predict_p': True,
        'predict_q': False
    },
    'features': {
        'max_lag': 24,
        'roll_windows': [6, 12, 24],
        'use_time_features': True,
        'sequence_length': 24,
        'season_length': 24,
        'exog_cols': ['定子电流', '定子电压', '转子电压', '转子电流', '励磁电流']
    },
    'evaluation': {
        'horizons': [1],  # 只测试一个步长
        'test_window': 50,  # 减小测试窗口
        'n_splits': 2,  # 减少折数
        'metrics': ['RMSE', 'MAE', 'SMAPE', 'WAPE', 'ACC']
    },
    'models': {
        'naive': {'enabled': True},
        'seasonal_naive': {'enabled': True},
        'rf': {
            'enabled': True,
            'n_estimators': 50,  # 减少树的数量
            'max_depth': None,
            'n_jobs': -1
        },
        'xgb': {
            'enabled': True,
            'n_estimators': 50,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_jobs': -1
        },
        'lstm': {
            'enabled': True,
            'hidden_size': 32,  # 减小模型
            'num_layers': 1,  # 减少层数
            'dropout': 0.2,
            'epochs': 10,  # 减少训练轮数
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'transformer': {
            'enabled': True,
            'd_model': 32,  # 减小模型
            'nhead': 2,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    },
    'forecast': {
        'enabled': True,
        'start_date': '2025-10-20',
        'end_date': '2025-10-20',  # 只预测一天
        'best_model': 'Transformer'
    },
    'plotting': {
        'fig_dpi': 100,  # 降低分辨率
        'font_priority': [
            'SimHei', 'Microsoft YaHei', 'STHeiti', 'PingFang SC',
            'Heiti SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
            'DejaVu Sans', 'Arial Unicode MS'
        ]
    },
    'report': {
        'generate_word': True,
        'generate_markdown': True,
        'include_forecast_table': True
    }
}

# 保存快速配置
config_path = Path('config_fast_test.yaml')
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(fast_config, f, allow_unicode=True, default_flow_style=False)

print(f"✓ 快速测试配置已创建: {config_path}")
print("\n运行测试:")
print(f"python run_all.py --config {config_path}")
print("\n预计运行时间: 5-10分钟")
print("完整运行时间: 30-60分钟")
