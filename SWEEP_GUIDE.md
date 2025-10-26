# W&B Sweep 超参数调优指南

## 📋 快速开始（睡前5分钟设置）

### 1. 安装wandb并登录
```bash
pip install wandb
wandb login
```
> 首次使用需要在 https://wandb.ai/authorize 获取API key

### 2. 启动sweep（3种方式）

#### 方式1: 使用自动化脚本（推荐）
```bash
chmod +x run_sweep.sh
./run_sweep.sh
```

#### 方式2: 手动命令
```bash
# 初始化sweep
wandb sweep sweep_config.yaml --project transformer-tuning

# 启动agent（后台运行）
nohup wandb agent <sweep-id> > sweep.log 2>&1 &

# 查看日志
tail -f sweep.log
```

#### 方式3: 直接Python（最灵活）
```python
import wandb

# 加载配置
with open('sweep_config.yaml') as f:
    sweep_config = yaml.safe_load(f)

# 初始化sweep
sweep_id = wandb.sweep(sweep_config, project="transformer-tuning")

# 启动agent（自动运行多次实验）
wandb.agent(sweep_id, function=train, count=50)  # 运行50次实验
```

---

## 🎯 配置说明

### 当前sweep配置 (`sweep_config.yaml`)

**优化目标**: `ACC_10` (最大化)  
**搜索方法**: Bayesian Optimization (贝叶斯优化)  
**早停策略**: Hyperband (效果差的实验提前终止)

### 超参数搜索空间

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `d_model` | [128, 192, 256, 320] | 模型维度 |
| `nhead` | [8, 12, 16] | 注意力头数 |
| `num_encoder_layers` | [3, 4, 5, 6] | 编码器层数 |
| `num_decoder_layers` | [3, 4, 5, 6] | 解码器层数 |
| `dim_feedforward` | [512, 768, 1024, 1536] | FFN维度 |
| `dropout` | [0.05, 0.2] | Dropout率（连续） |
| `learning_rate` | [1e-5, 1e-3] | 学习率（对数分布） |
| `batch_size` | [16, 32, 48, 64] | 批次大小 |
| `epochs` | [150, 200, 250, 300] | 训练轮数 |
| `sequence_length` | [72, 96, 120, 168] | 序列长度（小时） |
| `max_lag` | [72, 96, 120, 168] | 最大滞后 |

**总搜索空间大小**: 约 **150,000** 种组合

---

## 📊 查看结果（早上起床后）

### 1. 在线查看（推荐）
访问: https://wandb.ai/your-username/transformer-tuning/sweeps

**功能**:
- 📈 实时监控所有实验进度
- 🏆 自动排序找出最佳配置
- 📊 可视化参数重要性分析
- 🔍 平行坐标图查看参数关系
- 📉 学习曲线对比

### 2. 命令行查看
```bash
# 查看sweep状态
wandb sweep status <sweep-id>

# 查看最佳run
wandb sweep best <sweep-id>
```

### 3. Python分析脚本
```python
import wandb

api = wandb.Api()
sweep = api.sweep("your-username/transformer-tuning/<sweep-id>")

# 获取最佳运行
best_run = sweep.best_run()
print(f"最佳ACC_10: {best_run.summary['acc_10']:.2f}%")
print(f"最佳超参数: {best_run.config}")

# 获取所有运行
runs = sweep.runs
for run in runs:
    print(f"{run.name}: ACC_10={run.summary.get('acc_10', 0):.2f}%")
```

---

## 🔧 调整搜索策略

### 如果想更快出结果（减少搜索空间）

编辑 `sweep_config.yaml`:

```yaml
parameters:
  d_model:
    values: [192, 256]  # 只测2个值
  
  nhead:
    value: 16  # 固定为最优值
  
  # 其他参数类似缩小范围...
```

### 如果想更全面搜索

```yaml
method: random  # 改为随机搜索（更快但可能不是最优）

# 或者
method: grid  # 网格搜索（遍历所有组合，慢但全面）
```

---

## ⚠️ 常见问题

### Q1: 如何停止sweep?
```bash
# 找到agent进程
ps aux | grep wandb

# 停止进程
kill <PID>

# 或者在wandb网页端停止sweep
```

### Q2: 如何并行运行多个agent?
```bash
# 在不同终端或机器上运行相同命令
wandb agent <sweep-id>  # Terminal 1
wandb agent <sweep-id>  # Terminal 2
wandb agent <sweep-id>  # Terminal 3
```

### Q3: 如何设置运行次数?
```bash
# 只运行30次实验
wandb agent <sweep-id> --count 30
```

### Q4: 内存不足怎么办?
编辑 `sweep_config.yaml`:
```yaml
parameters:
  batch_size:
    values: [16, 24, 32]  # 减小批次
  
  d_model:
    values: [128, 192]  # 减小模型
```

---

## 📈 预期结果

**运行时间**: 每次实验约 **5-10分钟**  
**总实验数**: Bayesian优化通常 **30-50次** 就能找到最优解  
**总耗时**: 约 **3-8小时** (睡觉时间刚好)

**预期最佳配置**:
- ACC_10: **82-88%** (目标>80%)
- RMSE: **35-45** (当前最好47.56)

---

## 💡 高级技巧

### 1. 条件参数（确保nhead能整除d_model）
虽然wandb不直接支持条件参数，但可以在训练脚本中添加验证:

```python
# train_sweep.py中添加
if config.d_model % config.nhead != 0:
    config.nhead = config.d_model // 8  # 自动调整
```

### 2. 使用已有最佳配置作为起点
```yaml
method: bayes
metric:
  goal: maximize
  name: acc_10

# 添加已知好的配置
parameters:
  d_model:
    distribution: categorical
    values: [256, 320]  # 基于之前结果缩小范围
    probabilities: [0.6, 0.4]  # 更可能选256
```

### 3. 多目标优化
```yaml
metric:
  goal: maximize
  name: combined_score  # 在代码中计算组合分数

# 在train_sweep.py中:
combined_score = acc_10 * 0.7 + (100 - rmse) * 0.3
wandb.log({'combined_score': combined_score})
```

---

## 🎉 完成后

1. 从wandb界面获取最佳超参数
2. 更新到 `config_p_only.yaml`
3. 运行完整训练验证效果
4. 生成最终报告

祝你好梦！明早醒来就有最优配置了 😴✨
