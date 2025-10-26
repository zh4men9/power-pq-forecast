#!/usr/bin/env python3
"""
测试train_sweep.py的所有依赖和功能
"""

import sys
from pathlib import Path

print("="*60)
print("🔍 检查 train_sweep.py 依赖")
print("="*60)

# 1. 测试导入
print("\n1️⃣ 测试导入...")
try:
    from src.config import Config
    print("   ✅ Config")
except Exception as e:
    print(f"   ❌ Config: {e}")
    sys.exit(1)

try:
    from src.data_io import load_data
    print("   ✅ load_data")
except Exception as e:
    print(f"   ❌ load_data: {e}")
    sys.exit(1)

try:
    from src.features import prepare_sequences
    print("   ✅ prepare_sequences")
except Exception as e:
    print(f"   ❌ prepare_sequences: {e}")
    sys.exit(1)

try:
    from src.models.transformer import TransformerForecaster
    print("   ✅ TransformerModel")
except Exception as e:
    print(f"   ❌ TransformerModel: {e}")
    sys.exit(1)

try:
    from src.cv import rolling_origin_split
    print("   ✅ rolling_origin_split")
except Exception as e:
    print(f"   ❌ rolling_origin_split: {e}")
    sys.exit(1)

try:
    from src.metrics import eval_metrics
    print("   ✅ eval_metrics")
except Exception as e:
    print(f"   ❌ eval_metrics: {e}")
    sys.exit(1)

# 2. 测试配置加载
print("\n2️⃣ 测试配置加载...")
try:
    config = Config("config_sweep.yaml")
    print("   ✅ config_sweep.yaml 加载成功")
    
    # 测试访问各种配置
    models = config.config.get('models', {})
    features = config.config.get('features', {})
    data_cfg = config.config.get('data', {})
    target_cfg = config.config.get('target', {})
    
    print(f"   ✅ models配置存在: {bool(models)}")
    print(f"   ✅ features配置存在: {bool(features)}")
    print(f"   ✅ data配置存在: {bool(data_cfg)}")
    print(f"   ✅ target配置存在: {bool(target_cfg)}")
    
except Exception as e:
    print(f"   ❌ 配置加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 测试数据加载
print("\n3️⃣ 测试数据加载...")
try:
    data_path = Path(config.config['data']['data_path'])
    file_pattern = config.config['data']['file_pattern']
    data_files = list(data_path.glob(file_pattern))
    
    if not data_files:
        print(f"   ❌ 未找到数据文件: {data_path}/{file_pattern}")
        sys.exit(1)
    
    data_file = data_files[0]
    print(f"   ✅ 找到数据文件: {data_file}")
    
    # 快速加载测试（只加载不填充）
    df, _ = load_data(
        file_path=str(data_file),
        time_col='时间',
        p_col='有功',
        q_col='无功',
        exog_cols=['定子电流', '定子电压', '转子电压', '转子电流', '励磁电流'],
        freq='H',
        imputation_method='nearest_p',
        target_p_value=349
    )
    print(f"   ✅ 数据加载成功: {df.shape}")
    
except Exception as e:
    print(f"   ❌ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试目标列生成
print("\n4️⃣ 测试目标列生成...")
try:
    target_cols = []
    if config.get('target', 'predict_p', default=True):
        target_cols.append('P')
    if config.get('target', 'predict_q', default=False):
        target_cols.append('Q')
    
    print(f"   ✅ 目标列: {target_cols}")
    
except Exception as e:
    print(f"   ❌ 目标列生成失败: {e}")
    sys.exit(1)

# 5. 测试序列准备（小规模测试）
print("\n5️⃣ 测试序列准备...")
try:
    X, Y = prepare_sequences(
        df=df,
        horizon=1,
        target_cols=target_cols,
        exog_cols=['定子电流', '定子电压', '转子电压', '转子电流', '励磁电流'],
        sequence_length=24
    )
    print(f"   ✅ 序列准备成功: X.shape={X.shape}, Y.shape={Y.shape}")
    
except Exception as e:
    print(f"   ❌ 序列准备失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 测试时间序列分割
print("\n6️⃣ 测试时间序列分割...")
try:
    splits = list(rolling_origin_split(
        n_samples=len(X),
        test_window=300,
        n_splits=1,
        gap=0
    ))
    train_idx, test_idx = splits[0]
    print(f"   ✅ 分割成功: 训练集={len(train_idx)}, 测试集={len(test_idx)}")
    
except Exception as e:
    print(f"   ❌ 分割失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. 测试模型初始化和训练
print("\n7️⃣ 测试模型初始化和训练...")
try:
    # 使用小模型和小数据集测试
    X_small = X[:100]
    Y_small = Y[:100]
    
    model = TransformerForecaster(
        d_model=64,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        epochs=1,  # 只训练1个epoch用于测试
        batch_size=32,
        learning_rate=0.001,
        device='auto',
        n_horizons=1
    )
    print(f"   ✅ 模型初始化成功")
    
    # 测试fit()方法
    print(f"   🔄 测试训练 (1 epoch, 100 samples)...")
    model.fit(X_small, Y_small)
    print(f"   ✅ 训练成功")
    
    # 测试predict()方法
    y_pred = model.predict(X_small[:10])
    print(f"   ✅ 预测成功: y_pred.shape={y_pred.shape}")
    
except Exception as e:
    print(f"   ❌ 模型初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 8. 测试指标计算
print("\n8️⃣ 测试指标计算...")
try:
    import numpy as np
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([105, 195, 310, 395, 505])
    
    metrics = eval_metrics(
        y_true=y_true,
        y_pred=y_pred,
        metric_names=['RMSE', 'MAE', 'ACC_10']
    )
    print(f"   ✅ 指标计算成功: {metrics}")
    
except Exception as e:
    print(f"   ❌ 指标计算失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ 所有检查通过!")
print("="*60)
print("\n💡 train_sweep.py 应该可以正常运行")
print("   可以安全启动 sweep")
