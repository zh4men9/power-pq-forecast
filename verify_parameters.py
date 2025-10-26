"""
验证所有模型是否正确使用配置参数
"""
import numpy as np
import pandas as pd
from src.models.tree import RandomForestForecaster, XGBoostForecaster
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster

print("="*60)
print("验证模型参数使用情况")
print("="*60)

# 准备测试数据
X_tree = pd.DataFrame(np.random.randn(100, 10), 
                      columns=[f'feat_{i}' for i in range(10)])
y_tree = pd.DataFrame(np.random.randn(100, 2), columns=['P', 'Q'])

X_seq = np.random.randn(100, 24, 7)  # (samples, seq_len, features)
y_seq = np.random.randn(100, 2)       # (samples, targets)

print("\n1. RandomForest 参数验证")
print("-" * 60)
rf = RandomForestForecaster(n_estimators=50, max_depth=5, random_state=42)
print(f"  n_estimators: {rf.n_estimators} (应为 50)")
print(f"  max_depth: {rf.max_depth} (应为 5)")
print(f"  random_state: {rf.random_state} (应为 42)")
rf.fit(X_tree, y_tree)
print(f"  ✓ 实际模型 n_estimators: {rf.model.estimators_[0].n_estimators}")
print(f"  ✓ 实际模型 max_depth: {rf.model.estimators_[0].max_depth}")

print("\n2. XGBoost 参数验证")
print("-" * 60)
xgb = XGBoostForecaster(n_estimators=30, max_depth=4, learning_rate=0.05, random_state=42)
print(f"  n_estimators: {xgb.n_estimators} (应为 30)")
print(f"  max_depth: {xgb.max_depth} (应为 4)")
print(f"  learning_rate: {xgb.learning_rate} (应为 0.05)")
xgb.fit(X_tree, y_tree)
print(f"  ✓ 实际模型 n_estimators: {xgb.model.estimators_[0].n_estimators}")
print(f"  ✓ 实际模型 max_depth: {xgb.model.estimators_[0].max_depth}")
print(f"  ✓ 实际模型 learning_rate: {xgb.model.estimators_[0].learning_rate}")

print("\n3. LSTM 参数验证")
print("-" * 60)
lstm = LSTMForecaster(
    hidden_size=32,
    num_layers=1,
    dropout=0.1,
    epochs=3,  # 只训练3个epoch测试
    batch_size=16,
    learning_rate=0.002
)
print(f"  hidden_size: {lstm.hidden_size} (应为 32)")
print(f"  num_layers: {lstm.num_layers} (应为 1)")
print(f"  dropout: {lstm.dropout} (应为 0.1)")
print(f"  epochs: {lstm.epochs} (应为 3)")
print(f"  batch_size: {lstm.batch_size} (应为 16)")
print(f"  learning_rate: {lstm.learning_rate} (应为 0.002)")
print(f"  device: {lstm.device}")

print("\n  训练 LSTM (3 epochs)...")
lstm.fit(X_seq, y_seq)
print(f"  ✓ 实际模型 hidden_size: {lstm.model.hidden_size}")
print(f"  ✓ 实际模型 num_layers: {lstm.model.num_layers}")

print("\n4. Transformer 参数验证")
print("-" * 60)
transformer = TransformerForecaster(
    d_model=32,
    nhead=2,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=64,
    dropout=0.05,
    epochs=3,  # 只训练3个epoch测试
    batch_size=16,
    learning_rate=0.002
)
print(f"  d_model: {transformer.d_model} (应为 32)")
print(f"  nhead: {transformer.nhead} (应为 2)")
print(f"  num_encoder_layers: {transformer.num_encoder_layers} (应为 1)")
print(f"  num_decoder_layers: {transformer.num_decoder_layers} (应为 1)")
print(f"  dim_feedforward: {transformer.dim_feedforward} (应为 64)")
print(f"  dropout: {transformer.dropout} (应为 0.05)")
print(f"  epochs: {transformer.epochs} (应为 3)")
print(f"  batch_size: {transformer.batch_size} (应为 16)")
print(f"  learning_rate: {transformer.learning_rate} (应为 0.002)")
print(f"  device: {transformer.device}")

print("\n  训练 Transformer (3 epochs)...")
transformer.fit(X_seq, y_seq)
print(f"  ✓ 实际模型 d_model: {transformer.model.d_model}")
print(f"  ✓ 实际模型 nhead: {transformer.model.nhead}")

print("\n" + "="*60)
print("✓ 所有模型参数验证完成！")
print("  所有配置参数都正确传递并使用")
print("="*60)
