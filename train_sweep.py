#!/usr/bin/env python3
"""
W&B Sweep训练脚本 - Transformer超参数优化
目标: ACC_10 > 80% (horizon=1)
"""

import wandb
import yaml
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.data_io import load_data
from src.features import prepare_sequences
from src.models.transformer import TransformerForecaster
from src.cv import rolling_origin_split
from src.metrics import eval_metrics
import numpy as np
import logging


def train():
    """训练单次sweep实验"""
    
    # 1. 初始化wandb
    run = wandb.init()
    config = wandb.config
    
    # 2. 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logging.info(f"🔧 Sweep Run: {run.name}")
    logging.info(f"📊 超参数配置:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    
    try:
        # 3. 加载基础配置（使用sweep专用配置）
        base_config = Config("config_sweep.yaml")
        
        # 4. 覆盖超参数（使用config.config字典访问）
        base_config.config['models']['transformer']['d_model'] = config.d_model
        base_config.config['models']['transformer']['nhead'] = config.nhead
        base_config.config['models']['transformer']['num_encoder_layers'] = config.num_encoder_layers
        base_config.config['models']['transformer']['num_decoder_layers'] = config.num_decoder_layers
        base_config.config['models']['transformer']['dim_feedforward'] = config.dim_feedforward
        base_config.config['models']['transformer']['dropout'] = config.dropout
        base_config.config['models']['transformer']['learning_rate'] = config.learning_rate
        base_config.config['models']['transformer']['batch_size'] = config.batch_size
        base_config.config['models']['transformer']['epochs'] = config.epochs
        
        base_config.config['features']['sequence_length'] = config.sequence_length
        base_config.config['features']['max_lag'] = config.max_lag
        base_config.config['evaluation']['test_window'] = config.test_window
        
        # 5. 加载数据
        logging.info(f"📂 加载数据 (填充策略: {config.strategy})...")
        
        # 找到数据文件
        from pathlib import Path
        import glob
        data_path = Path(base_config.config['data']['data_path'])
        file_pattern = base_config.config['data']['file_pattern']
        data_files = list(data_path.glob(file_pattern))
        
        if not data_files:
            raise FileNotFoundError(f"未找到数据文件: {data_path}/{file_pattern}")
        
        data_file = data_files[0]
        logging.info(f"   数据文件: {data_file}")
        
        # 获取填充配置
        imputation_config = base_config.config.get('data', {}).get('imputation', {})
        
        # 调用load_data
        df_clean, df_before = load_data(
            file_path=str(data_file),
            time_col=base_config.config['data'].get('time_col'),
            p_col=base_config.config['data'].get('p_col'),
            q_col=base_config.config['data'].get('q_col'),
            exog_cols=base_config.config.get('features', {}).get('exog_cols', []),
            freq=base_config.config['data'].get('freq', 'H'),
            tz=base_config.config['data'].get('tz'),
            interp_limit=base_config.config['data'].get('interp_limit', 3),
            imputation_method=config.strategy,
            target_p_value=imputation_config.get('target_p_value', 349),
            day_copy_days_back=imputation_config.get('day_copy_days_back', 7),
            seasonal_period=imputation_config.get('seasonal_period', 24)
        )
        
        logging.info(f"   数据形状: {df_clean.shape}")
        
        # 6. 确定目标列
        target_cols = []
        predict_p = base_config.config.get('target', {}).get('predict_p', True)
        predict_q = base_config.config.get('target', {}).get('predict_q', False)
        if predict_p:
            target_cols.append('P')
        if predict_q:
            target_cols.append('Q')
        
        logging.info(f"   目标列: {target_cols}")
        
        # 7. 准备序列数据
        logging.info(f"🔄 准备序列数据 (horizon={config.horizon}, seq_len={config.sequence_length})...")
        X, Y = prepare_sequences(
            df=df_clean,
            horizon=config.horizon,
            target_cols=target_cols,
            exog_cols=base_config.config.get('features', {}).get('exog_cols', []),
            sequence_length=config.sequence_length
        )
        
        # 8. 时间序列分割
        splits = list(rolling_origin_split(
            n_samples=len(X),
            test_window=config.test_window,
            n_splits=1,
            gap=0
        ))
        train_idx, test_idx = splits[0]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        
        logging.info(f"📊 数据集划分:")
        logging.info(f"  训练集: {len(X_train)} 样本")
        logging.info(f"  测试集: {len(X_test)} 样本")
        
        # 9. 训练模型
        logging.info(f"🚀 开始训练Transformer...")
        model = TransformerForecaster(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            device=base_config.config.get('evaluation', {}).get('device', 'auto'),
            n_horizons=1
        )
        model.fit(X_train, y_train)
        
        # 10. 预测
        logging.info(f"🔮 进行预测...")
        y_pred = model.predict(X_test)
        
        # 11. 计算指标
        logging.info(f"📈 计算评估指标...")
        metrics = {}
        for i, target_col in enumerate(target_cols):
            y_true_col = y_test[:, i]
            y_pred_col = y_pred[:, i]
            
            col_metrics = eval_metrics(
                y_true=y_true_col,
                y_pred=y_pred_col,
                metric_names=['RMSE', 'MAE', 'SMAPE', 'WAPE', 'ACC_5', 'ACC_10']
            )
            
            # 添加前缀
            for metric_name, metric_value in col_metrics.items():
                metrics[f"{target_col}_{metric_name}"] = metric_value
        
        # 12. 记录指标到wandb
        rmse = metrics.get('P_RMSE', 999)
        mae = metrics.get('P_MAE', 999)
        acc_10 = metrics.get('P_ACC_10', 0)
        acc_5 = metrics.get('P_ACC_5', 0)
        
        wandb.log({
            'rmse': rmse,
            'mae': mae,
            'acc_10': acc_10,
            'acc_5': acc_5,
            'smape': metrics.get('P_SMAPE', 999),
            'wape': metrics.get('P_WAPE', 999),
        })
        
        logging.info(f"✅ 训练完成!")
        logging.info(f"📊 最终指标:")
        logging.info(f"  RMSE: {rmse:.2f}")
        logging.info(f"  MAE: {mae:.2f}")
        logging.info(f"  ACC_10: {acc_10:.2f}%")
        logging.info(f"  ACC_5: {acc_5:.2f}%")
        
        # 判断是否达标
        if acc_10 >= 80:
            logging.info(f"🎉 达标! ACC_10 = {acc_10:.2f}% >= 80%")
        else:
            logging.info(f"⚠️  未达标: ACC_10 = {acc_10:.2f}% < 80%")
        
    except Exception as e:
        logging.error(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 记录失败
        wandb.log({
            'rmse': 999,
            'mae': 999,
            'acc_10': 0,
            'acc_5': 0,
        })
        raise
    
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
