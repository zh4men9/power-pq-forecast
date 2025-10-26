# Transformer 模型优化指南

## 📊 配置对比

### 原始配置 vs 优化配置

| 配置项 | 原始值 | 优化值 | 原因 |
|--------|--------|--------|------|
| **训练配置** |
| epochs | 1 (fast_test) | 200 | ❌ 训练严重不足 |
| test_window | 100 | 600 | ⚠️ 测试集太小(3%→18%) |
| n_splits | 1 | 3 | ⚠️ 单次划分不够robust |
| device | cpu | auto | 🚀 自动使用最快设备 |
| **特征工程** |
| sequence_length | 24 | 72 | 📈 从1天增加到3天 |
| max_lag | 24 | 72 | 📈 匹配序列长度 |
| roll_windows | [6,12,24] | [6,12,24,48,168] | 📈 添加周模式 |
| **Transformer架构** |
| d_model | 64 | 128 | 💪 增加模型容量 |
| nhead | 4 | 8 | 💪 更多注意力头 |
| encoder_layers | 2 | 3 | 💪 更深的网络 |
| decoder_layers | 2 | 3 | 💪 更深的网络 |
| dim_feedforward | 256 | 512 | 💪 更大的前馈网络 |
| dropout | 0.2 | 0.1 | 📉 减少过拟合风险 |
| batch_size | 32 | 64 | ⚡ 加快训练 |
| **LSTM架构** |
| hidden_size | 64 | 128 | 💪 增加模型容量 |
| dropout | 0.3 | 0.2 | 📉 减少过拟合风险 |

---

## 🎯 预期效果提升

### 当前表现 (epochs=1, 错误配置)

```
RandomForest: RMSE = 69.9  ⭐⭐⭐⭐⭐ (最好)
XGBoost:      RMSE = 75.8  ⭐⭐⭐⭐
Transformer:  RMSE = 431.1 ⭐ (差6倍！)
LSTM:         RMSE = 433.6 ⭐ (差6倍！)
```

### 预期表现 (epochs=200, 优化配置)

根据类似时序预测任务的经验：

```
预期排名:
1. Transformer: RMSE = 50-70  ⭐⭐⭐⭐⭐ (可能最好)
2. LSTM:        RMSE = 55-75  ⭐⭐⭐⭐⭐
3. XGBoost:     RMSE = 70-80  ⭐⭐⭐⭐
4. RandomForest: RMSE = 70-85  ⭐⭐⭐⭐
5. SeasonalNaive: RMSE = 180+  ⭐
```

**提升幅度**: 深度学习模型预期提升 **6-8倍** 📈

---

## 🚀 运行优化后的配置

```bash
# 使用优化后的配置运行
python run_all.py --config config_p_only.yaml
```

**预计训练时间**:
- CPU: 约 1-2 小时（200 epochs × 8 strategies）
- MPS (Apple Silicon): 约 20-30 分钟
- CUDA (NVIDIA GPU): 约 10-15 分钟

---

## 📈 优化说明

### 1. 测试集大小 (test_window)

**原理**: 
- 时序数据需要足够长的测试期来评估泛化能力
- 太小的测试集会导致结果不稳定（运气成分大）

**推荐值**:
- 小型数据集: 15-20%
- 中型数据集: 10-15%
- 大型数据集: 5-10%

当前数据: 3358 样本
- 100 样本 = 3% ❌
- 600 样本 = 18% ✅

### 2. 交叉验证折数 (n_splits)

**原理**:
- 单次划分可能偶然性强
- 多折验证可以评估模型稳定性

**推荐值**:
- 快速实验: 1
- 正常训练: 3
- 严格评估: 5

**trade-off**: n_splits=3 会使训练时间增加 3 倍

### 3. 序列长度 (sequence_length)

**原理**:
- 更长的历史可以捕获更复杂的模式
- 但也会增加计算量和可能的过拟合

**推荐值**:
- 日模式为主: 24 (1天)
- 周模式为主: 168 (1周)
- 混合模式: 72 (3天) ✅ 当前选择

**权衡**: 72 是较好的折中方案

### 4. 模型容量 (d_model, hidden_size)

**原理**:
- 更大的模型可以学习更复杂的模式
- 但需要更多数据和训练时间

**Transformer d_model**:
- 简单任务: 64
- 中等任务: 128 ✅ 当前选择
- 复杂任务: 256

**当前数据量**: 3358 样本，128 是合适的

### 5. Dropout

**原理**:
- Dropout 防止过拟合
- 但太大会导致欠拟合

**当前情况**:
- epochs=1 时，模型严重欠拟合
- 降低 dropout 可以让模型学得更充分

**调整**:
- Transformer: 0.2 -> 0.1
- LSTM: 0.3 -> 0.2

### 6. 批次大小 (batch_size)

**原理**:
- 更大的批次: 训练更稳定，速度更快
- 更小的批次: 泛化能力可能更好

**推荐值**:
- CPU: 32-64
- GPU/MPS: 64-128 (取决于显存)

**调整**: Transformer 从 32 -> 64

---

## 🔍 训练监控建议

### 1. 添加学习率调度器

如果效果还不够好，可以添加：

```yaml
transformer:
  learning_rate: 0.001  # 初始学习率
  lr_scheduler: "cosine"  # 余弦退火
  warmup_steps: 100  # 预热步数
```

### 2. Early Stopping

避免过拟合：

```yaml
transformer:
  early_stopping: true
  patience: 20  # 20个epoch没提升就停止
```

### 3. 保存最佳模型

```yaml
transformer:
  save_best_only: true  # 只保存验证集上最好的模型
```

---

## 📊 预期训练过程

### Epoch 1-20: 快速学习阶段
```
Loss: 177608 -> 50000
RMSE: 430 -> 150
```

### Epoch 21-50: 稳定提升阶段
```
Loss: 50000 -> 10000
RMSE: 150 -> 80
```

### Epoch 51-100: 细节优化阶段
```
Loss: 10000 -> 5000
RMSE: 80 -> 60
```

### Epoch 101-200: 收敛阶段
```
Loss: 5000 -> 3000
RMSE: 60 -> 50-55
```

---

## ⚠️ 注意事项

### 1. 如果训练过程中出现 NaN

**原因**: 学习率太大或梯度爆炸

**解决方案**:
```yaml
transformer:
  learning_rate: 0.0001  # 降低学习率
  gradient_clip: 1.0     # 添加梯度裁剪
```

### 2. 如果训练很慢

**检查**:
- 是否使用了 GPU/MPS？
- batch_size 是否太小？

**优化**:
```yaml
device:
  type: "auto"  # 确保使用最快的设备

transformer:
  batch_size: 128  # 如果显存够，增加批次
```

### 3. 如果过拟合（训练集好，测试集差）

**解决方案**:
- 增加 dropout (0.1 -> 0.2)
- 增加正则化
- 减少模型大小
- 增加训练数据

### 4. 如果欠拟合（训练集和测试集都差）

**解决方案**:
- 增加 epochs
- 增大模型 (d_model: 128 -> 256)
- 降低 dropout (0.1 -> 0.05)
- 增加特征

---

## 🎓 进阶优化技巧

### 1. 位置编码优化

Transformer 使用正弦位置编码，可能不适合你的数据周期。

考虑：
- 使用学习的位置编码
- 添加显式的时间特征（小时、星期、月份）

### 2. 注意力机制改进

- 使用局部注意力（Local Attention）
- 使用相对位置编码
- 添加时间注意力层

### 3. 多任务学习

同时预测 P 和 Q，可能相互促进：

```yaml
target:
  predict_p: true
  predict_q: true  # 改为 true
```

### 4. 集成学习

将多个模型的预测结果集成：
- Transformer + LSTM + XGBoost
- 加权平均或 Stacking

---

## 📞 下一步

1. **立即运行优化配置**:
   ```bash
   python run_all.py --config config_p_only.yaml
   ```

2. **监控训练日志**:
   ```bash
   tail -f outputs/latest/logs/training.log
   ```

3. **查看结果**:
   - 检查生成的Word报告
   - 对比不同 horizon 的表现
   - 分析误差随 horizon 的变化趋势

4. **根据结果调整**:
   - 如果效果好 → 可以尝试更大的模型
   - 如果效果差 → 检查数据质量和特征工程
   - 如果过拟合 → 增加正则化
   - 如果欠拟合 → 增加模型容量

---

**预祝训练成功！🎉**

如有问题，请查看日志文件或联系支持。
