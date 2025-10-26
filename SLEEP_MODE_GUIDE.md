# 🌙 睡前快速启动指南

## 方式1: 一键启动（最简单）⭐

```bash
python start_sweep.py
```

按提示选择：
- **选项2** (标准搜索30次) - 推荐，约4-6小时
- **选项4** (彻夜运行) - 早上起来停止

## 方式2: 命令行启动

```bash
# 1. 安装wandb
pip install wandb

# 2. 登录（首次使用）
wandb login
# 访问 https://wandb.ai/authorize 获取key

# 3. 初始化sweep
wandb sweep sweep_config.yaml --project transformer-tuning

# 4. 启动agent（会显示sweep ID）
wandb agent <sweep-id>

# 如果要后台运行：
nohup wandb agent <sweep-id> > sweep.log 2>&1 &
```

## ⏰ 早上起床后

### 1. 查看结果（网页端）

访问: https://wandb.ai （登录后进入项目）

- 点击 **Sweeps** 标签
- 查看 **transformer-tuning** 项目
- 按 **acc_10** 排序，找到最佳配置

### 2. 停止sweep（如果还在运行）

```bash
# 查找进程
ps aux | grep wandb

# 停止
kill <PID>
```

### 3. 应用最佳配置

从wandb网页获取最佳超参数，更新到 `config_p_only.yaml`:

```yaml
models:
  transformer:
    d_model: 256        # 从sweep结果复制
    nhead: 16           # 从sweep结果复制
    num_encoder_layers: 4  # 从sweep结果复制
    # ... 其他参数
```

## 📊 预期结果

- **运行时间**: 每次实验 5-10分钟
- **总实验数**: 30-50次
- **总耗时**: 3-8小时
- **目标ACC_10**: >80%

## 💤 晚安！

明早醒来就有最优超参数配置了～
