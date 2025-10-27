# 电力质量预测系统

**专业的时间序列预测解决方案**

---

## 📋 目录

- [系统概述](#系统概述)
- [主要功能](#主要功能)
- [项目文件结构](#项目文件结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [输出说明](#输出说明)
- [技术参考](#技术参考)

---

## 🎯 系统概述

本系统是一个完整的电力时序预测解决方案，用于预测电力系统的**有功功率（P）**和**无功功率（Q）**。系统集成了多种预测模型，采用标准的时间序列交叉验证方法，提供准确可靠的预测结果和专业的评估报告。

### 系统特点

- **多模型集成**：包含基线模型、机器学习模型和深度学习模型，可自动对比选择最优方案
- **严格验证**：采用滚动起点交叉验证，确保评估结果的可靠性
- **多维评估**：提供RMSE、MAE、SMAPE、WAPE、准确率等多种评估指标
- **灵活配置**：支持多种数据填充策略、预测步长和模型参数配置
- **自动化流程**：一键完成从数据加载到报告生成的全部流程
- **可视化报告**：自动生成包含图表和详细分析的专业报告

### 适用场景

- 电力系统短期和中期功率预测
- 多步预测（1步、12步、24步等任意步长）
- 支持引入外部变量（电压、电流、温度等）提升预测精度
- 模型对比和性能评估

---

## 🎉 主要功能

### 1. 多模型预测

系统集成了6种预测模型，自动训练和对比：

| 模型类别 | 模型名称 | 特点 |
|---------|---------|------|
| **基线模型** | Naive（朴素预测） | 使用最后观测值作为预测 |
| | SeasonalNaive（季节朴素） | 使用上一季节同时刻的值 |
| **树模型** | RandomForest（随机森林） | 集成学习，可解释性强 |
| | XGBoost | 梯度提升，高精度 |
| **深度学习** | LSTM | 长短期记忆网络，适合长序列 |
| | Transformer | 注意力机制，捕捉复杂模式 |

### 2. 数据填充策略

支持8种数据填充方法，可自动对比不同策略的效果：

- **nearest_p**：基于P值接近性填充
- **forward**：前向填充
- **backward**：后向填充
- **interpolate**：线性插值
- **mean**：均值填充
- **median**：中位数填充
- **day_copy**：复制前N天同时刻的值
- **seasonal**：季节性分解填充

### 3. 超参数优化（W&B Sweep）

- 自动搜索最优模型参数（支持Transformer模型）
- 贝叶斯优化算法，智能搜索参数空间
- Hyperband早停机制，节省计算资源
- Web界面实时监控训练进度
- 已验证最佳配置：准确率可达89%

### 4. 模型加载与重用

- 支持加载已训练的LSTM和Transformer模型
- 快速模型：仅重新训练快速的基线和树模型
- 避免重复训练耗时的深度学习模型
- 适用于增量更新和快速验证场景

### 5. 完整的评估体系

- **6项评估指标**：RMSE、MAE、SMAPE、WAPE、ACC_5、ACC_10
- **滚动起点交叉验证**：严格防止数据泄漏
- **多步长评估**：评估不同预测步长的性能
- **可视化对比**：自动生成模型对比图表

### 6. 专业报告生成

- **Word格式报告**：专业排版，包含完整的图表和分析
- **评估指标表格**：详细的性能对比数据
- **可视化图表**：数据总览、误差对比、特征重要性
- **配置记录**：自动备份每次运行的配置参数

### 7. 随机种子管理

- 统一管理Python、NumPy、PyTorch的随机种子
- 确保实验结果100%可复现
- 支持配置启用/禁用随机种子

### 8. 进程管理工具

- 安全终止W&B相关进程
- 保护主训练程序不被误杀
- 支持后台运行和长时间训练任务

---

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- 建议使用虚拟环境（venv/conda）

### 安装步骤

**1. 克隆或下载项目**

```bash
cd power-pq-forecast
```

**2. 安装依赖**

```bash
pip install -r requirements.txt
```

**3. 准备数据**

将电力数据文件（Excel或CSV格式）放置在 `data/raw/` 目录下。

数据要求：
- 必须包含时间列
- 必须包含有功功率列（P）
- 可选：无功功率列（Q）、其他外部变量

**4. 运行预测**

```bash
python run_all.py --config config_p_only_newdata.yaml
```

**5. 查看结果**

运行完成后，在 `outputs/latest/` 目录查看：
- `report/` - Word格式的评估报告
- `figures/` - 可视化图表
- `metrics/` - 详细的评估指标CSV文件
- `models/` - 训练好的模型文件

### 可选：超参数优化

如需优化Transformer模型参数：

```bash
# 安装额外依赖
pip install -r requirements_sweep.txt

# 启动超参数优化
python start_sweep.py
```

系统会自动搜索最优参数组合并保存结果。

---

---

## ⚙️ 配置说明

### 主要配置项

系统通过YAML配置文件进行参数设置。以下是主要配置项说明：

#### 数据配置

```yaml
data:
  data_path: "data/raw"           # 数据目录
  file_pattern: "*.csv"           # 文件匹配模式
  freq: "H"                       # 时间频率（H=小时）
  
  # 数据填充策略
  imputation:
    method: "interpolate"         # 单一策略
    # 或使用多策略对比
    strategies:                   # 多策略列表
      - nearest_p
      - forward
      - interpolate
```

#### 特征工程配置

```yaml
features:
  max_lag: 24                     # 最大滞后步数
  roll_windows: [6, 12, 24]       # 滚动窗口大小
  sequence_length: 24             # 深度学习序列长度
  use_time_features: true         # 是否使用时间特征
  exog_cols: []                   # 外生变量列表
```

#### 评估配置

```yaml
evaluation:
  horizons: "1,12,24"             # 预测步长
  test_window: 200                # 测试窗口大小
  n_splits: 1                     # 交叉验证折数
```

#### 模型配置

```yaml
models:
  lstm:
    enabled: true
    hidden_size: 64
    num_layers: 2
    epochs: 100
    batch_size: 64
    
  transformer:
    enabled: true
    d_model: 256
    nhead: 16
    epochs: 200
```

#### 随机种子配置

```yaml
random_seed:
  enabled: true                   # 启用随机种子
  seed: 42                        # 种子值
```

#### 设备配置

```yaml
device:
  type: "cpu"                     # 计算设备：cpu/cuda/mps/auto
```

### 配置文件说明

| 配置文件 | 用途 |
|---------|------|
| `config_p_only.yaml` | 标准配置，仅使用P/Q数据 |
| `config_p_only_newdata.yaml` | 新数据集配置 |
| `config_p_only_newdata-transformer89.yaml` | 最佳参数配置（89%准确率） |
| `config_fast_test.yaml` | 快速测试配置 |

---

## 📖 使用方法

### 基本使用

**运行完整流程**

```bash
python run_all.py --config config_p_only_newdata.yaml
```

**加载已训练模型**

如果已有训练好的LSTM和Transformer模型，可以跳过这两个模型的训练：

```bash
python run_all.py \
  --config config_p_only_newdata.yaml \
  --load-models outputs/output-2025-10-27-0952/models_nearest_p
```

这样可以节省大量训练时间，只重新训练快速的基线和树模型。

### 超参数优化

**启动Sweep优化**

```bash
# 安装依赖
pip install -r requirements_sweep.txt

# 启动优化
python start_sweep.py
```

**查看优化结果**

访问 W&B 控制台查看实时训练进度和结果对比。

### 进程管理

**终止wandb进程**

```bash
# 使用Bash脚本
./kill_wandb.sh

# 或使用Python脚本
python kill_wandb.py --force
```

### 报告重生成

如果需要修改报告但不想重新训练：

```bash
python regenerate_report.py \
  --output outputs/output-2025-10-27-0952 \
  --force
```

---

## 📊 输出说明

### 输出目录结构

每次运行会生成带时间戳的输出目录：

```
outputs/
├── output-YYYY-MM-DD-HHMM/          # 时间戳目录
│   ├── config_used.yaml              # 本次运行的配置备份
│   ├── logs/
│   │   └── training.log              # 完整训练日志
│   ├── figures_[strategy]/           # 可视化图表
│   │   ├── data_overview.png
│   │   ├── error_by_horizon_rmse.png
│   │   └── all_metrics_by_horizon.png
│   ├── metrics_[strategy]/           # 评估指标
│   │   └── cv_metrics.csv
│   ├── models_[strategy]/            # 训练好的模型
│   │   ├── LSTM_h1.pkl
│   │   ├── Transformer_h1.pkl
│   │   └── ...
│   └── report/                       # 评估报告
│       └── 项目评估报告_[strategy].docx
└── latest/                           # 指向最新结果的快捷方式
```

### 评估指标说明

系统提供6项评估指标：

| 指标 | 说明 | 取值范围 |
|------|------|---------|
| **RMSE** | 均方根误差，对大误差敏感 | ≥0，越小越好 |
| **MAE** | 平均绝对误差，线性惩罚 | ≥0，越小越好 |
| **SMAPE** | 对称平均绝对百分比误差 | 0-200%，越小越好 |
| **WAPE** | 加权绝对百分比误差 | 0-100%，越小越好 |
| **ACC_5** | 5%阈值内的预测准确率 | 0-100%，越大越好 |
| **ACC_10** | 10%阈值内的预测准确率 | 0-100%，越大越好 |

### 报告内容

Word报告包含以下内容：

1. **项目概述**：预测目标和方法说明
2. **数据描述**：数据基本信息和处理过程
3. **模型说明**：各模型的原理和参数配置
4. **评估结果**：详细的性能对比表格
5. **可视化图表**：误差对比、特征重要性等
6. **结论建议**：最优模型推荐和应用建议

---

## 🔧 技术参考

### 滚动起点交叉验证

系统采用时间序列专用的滚动起点交叉验证方法：

- 严格按时间顺序划分训练集和测试集
- 训练集逐步扩展（Expanding Window）
- 测试集大小固定，向前滚动
- 完全防止使用未来信息

### 数据泄漏防范

多层次防护机制：

1. **特征工程层**：只使用历史数据生成特征
2. **交叉验证层**：训练集时间严格早于测试集
3. **标准化层**：只在训练集上拟合，测试集仅变换

### 模型说明

#### 基线模型
- **Naive**：使用最后观测值
- **SeasonalNaive**：使用上季节同时刻值

#### 树模型
- **RandomForest**：集成学习，可解释性强
- **XGBoost**：梯度提升，高精度

#### 深度学习
- **LSTM**：长短期记忆网络，捕捉长期依赖
- **Transformer**：注意力机制，处理复杂模式

---

## 📚 文档

详细文档请参考：

- `USAGE.md` - 详细使用指南
- `SWEEP_GUIDE.md` - 超参数优化指南
- `KILL_WANDB_GUIDE.md` - 进程管理指南
- `PROJECT_SUMMARY.md` - 项目实施总结

---

## 📄 许可证

MIT License

---

**新增8种填充方法**：

| 策略 | 说明 | 适用场景 | 参数 |
|------|------|----------|------|
| **nearest_p** | 基于P值接近性填充 | 原有方法，有目标P值参考 | `target_p_value` |
| **forward** | 前向填充(ffill) | 数据缓慢变化，连续性好 | - |
| **backward** | 后向填充(bfill) | 需要向前传播已知值 | - |
| **interpolate** | 线性插值 | 连续平滑数据，缺失较少 | - |
| **mean** | 均值填充 | 数据稳定，无明显趋势 | - |
| **median** | 中位数填充 | 存在异常值，需要鲁棒性 | - |
| **day_copy** | 日期复制 | 强日周期性(电力负荷) | `days_back=7` |
| **seasonal** | 季节分解 | 复杂季节模式 | `period=24` |

**配置示例**：
```yaml
data:
  imputation:
    # 多策略模式：循环运行，每个策略生成独立报告
    strategies:
      - nearest_p
      - forward
      - interpolate
      - day_copy
    
    # 单一策略模式（如果strategies为空）
    method: "nearest_p"
    
    # 参数配置
    target_p_value: 400       # nearest_p: 目标有功功率值
    day_copy_days_back: 7     # day_copy: 回溯天数(默认7天)
    seasonal_period: 24       # seasonal: 季节周期(小时数据)
```

**统一接口**：
```python
# 调用示例
df_filled = impute_data(
    df, 
    method='day_copy',      # 填充方法
    days_back=7             # 方法参数
)
```

---

#### 🔄 4. 多策略批量对比模式

**核心功能**：一次运行自动对比所有填充策略的效果

**工作流程**：
```
1. 读取config中的strategies列表
2. 对每个策略:
   ├─ 独立加载和填充数据
   ├─ 训练所有模型(Naive/RF/XGB/LSTM/Transformer)
   ├─ 生成独立的metrics/figures/models目录
   └─ 生成独立的Word报告: 项目评估报告_<策略>.docx
3. 所有结果保存在同一个output目录
```

**目录结构**：
```
outputs/output-2025-10-26-2045/
├── config_used.yaml              # ✅ 备份的配置文件
├── logs/training.log              # ✅ 完整日志
├── report/                        # ✅ 报告目录
│   ├── 项目评估报告_nearest_p.docx
│   ├── 项目评估报告_forward.docx
│   ├── 项目评估报告_backward.docx
│   ├── 项目评估报告_interpolate.docx
│   ├── 项目评估报告_mean.docx
│   ├── 项目评估报告_median.docx
│   ├── 项目评估报告_day_copy.docx
│   └── 项目评估报告_seasonal.docx
├── figures_nearest_p/            # 各策略独立图表
├── figures_forward/
├── metrics_nearest_p/            # 各策略独立指标
├── models_nearest_p/             # 各策略独立模型
└── ...
```

**运行命令**：
```bash
# 运行所有策略(默认8个)
python run_all.py

# 查看报告
ls outputs/latest/report/*.docx
```

**日志输出示例**：
```
🔄 多策略模式: 将依次运行 8 个填充策略
策略列表: nearest_p, forward, backward, interpolate, mean, median, day_copy, seasonal

████████████████████████████████████████████████████████████
运行策略 [1/8]: nearest_p
████████████████████████████████████████████████████████████

🔧 Applying nearest_p imputation...
✓ Forward fill imputed 1234 missing values
✓ Word报告已生成: report/项目评估报告_nearest_p.docx

████████████████████████████████████████████████████████████
运行策略 [2/8]: forward
████████████████████████████████████████████████████████████
...
```

**单策略模式**：
```yaml
# 如果不想批量对比，注释掉strategies
data:
  imputation:
    # strategies: [...]  # 注释掉
    method: "interpolate"  # 只用一种方法
```

---

#### 📝 5. 配置文件版本管理

**新增功能**：每次运行自动备份配置到output目录

**实现机制**：
```python
# run_all.py 自动执行
config_backup_path = output_dir / 'config_used.yaml'
shutil.copy2(args.config, config_backup_path)

# 报告生成时读取备份的配置
word_report_path = generate_word_report(
    results_df,
    config_path=str(config_backup_path),  # ✅ 读取备份
    ...
)
```

**优势**：
- ✅ **可追溯**：每个实验的配置永久保存
- ✅ **参数一致**：报告参数与实际训练100%一致
- ✅ **便于复现**：可以基于历史配置重新运行
- ✅ **版本对比**：对比不同配置的效果

**使用示例**：
```bash
# 查看某次运行的配置
cat outputs/output-2025-10-26-1430/config_used.yaml

# 基于历史配置重新运行
python run_all.py --config outputs/output-2025-10-26-1430/config_used.yaml

# 对比两次配置
diff outputs/output-2025-10-26-1430/config_used.yaml \
     outputs/output-2025-10-26-1645/config_used.yaml
```

---

### 2025年10月更新（之前版本）

#### 🎯 1. 双准确率指标系统 (ACC_5 & ACC_10)

**新增功能**：将原有的单一ACC指标拆分为两个独立指标，提供更细粒度的准确率评估

| 指标 | 阈值 | 含义 | 适用场景 |
|------|------|------|----------|
| **ACC_5** | 5% | 严格准确率 | 高精度要求场景 |
| **ACC_10** | 10% | 宽松准确率 | 实用性评估 |

**示例解读**：
- ACC_5 = 75%：表示75%的预测误差在5%以内（高精度）
- ACC_10 = 92%：表示92%的预测误差在10%以内（实用可接受）

**实现细节**：
```python
# src/metrics.py 新增函数
def acc_5(y_true, y_pred):
    """5%阈值准确率"""
    return acc(y_true, y_pred, threshold=0.05)

def acc_10(y_true, y_pred):
    """10%阈值准确率"""
    return acc(y_true, y_pred, threshold=0.10)
```

**配置更新**：
```yaml
evaluation:
  metrics:
    - RMSE
    - MAE
    - SMAPE
    - WAPE
    - ACC_5   # 新增：5%阈值
    - ACC_10  # 新增：10%阈值
```

**报告更新**：
- ✅ 所有评估表格包含ACC_5和ACC_10
- ✅ 图表说明更新为6个指标
- ✅ 最优模型判断同时考虑两个ACC指标

---

#### ⚡ 2. 可配置交叉验证折数 (n_splits)

**新增功能**：用户可自由选择单次划分或多折交叉验证

**配置选项**：
```yaml
evaluation:
  n_splits: 1  # 可选：1, 2, 3, 5...
```

**性能对比**：

| n_splits | 训练时间 | 结果稳定性 | 适用场景 |
|----------|----------|------------|----------|
| **1** | 1x (基准) | ⭐⭐⭐ | 快速开发、模型选型 |
| **3** | 3x | ⭐⭐⭐⭐⭐ | 正式报告、论文发表 |
| **5** | 5x | ⭐⭐⭐⭐⭐ | 小样本数据集 |

**使用建议**：
- 📊 **日常开发**：`n_splits: 1`（默认配置）
  - 训练速度快3倍
  - 3384样本数据量充足，单次划分已够稳定
  - 适合快速迭代和参数调优
  
- 📑 **正式报告**：`n_splits: 3`
  - 更可靠的性能评估
  - 减少随机性影响
  - 符合学术标准

**实现细节**：
- 使用滚动起点交叉验证（Rolling Origin CV）
- n_splits=1时退化为单次时间序列划分
- 保持时间顺序，严格防止数据泄漏

---

#### 🖥️ 3. 可配置硬件加速设备

**新增功能**：通过配置文件控制深度学习模型的计算设备

**配置语法**：
```yaml
device:
  type: "cpu"  # 可选: "auto", "cpu", "mps", "cuda"
```

**设备选项说明**：

| 选项 | 行为 | 适用场景 |
|------|------|----------|
| **auto** | 自动检测（cuda > mps > cpu） | 默认推荐 |
| **cpu** | 强制使用CPU | 小模型、快速稳定 |
| **mps** | 强制使用Apple Metal | M芯片Mac + 大batch |
| **cuda** | 强制使用NVIDIA GPU | NVIDIA显卡 |

**性能基准测试结果**（基于实际数据3000样本）：

**LSTM模型**：
```
batch=32: CPU 0.53s vs MPS 0.59s → CPU快11%
batch=64: CPU 0.40s vs MPS 0.64s → CPU快62%
```

**Transformer模型**：
```
batch=32: CPU 1.17s vs MPS 1.64s → CPU快40%
batch=64: CPU 1.03s vs MPS 0.90s → MPS快15%
```

**结论与建议**：
- ✅ **推荐配置**：`device: cpu`（默认）
- 原因：数据量3000+，模型小，CPU已经很快
- 只有Transformer + batch≥64时MPS才略快
- CPU更稳定，避免MPS兼容性问题

**代码实现**：
```python
# 从配置读取设备类型
device_type = config.get('device', 'type', default='cpu')

# 传递给模型
model = LSTMForecaster(..., device=device_type)
model = TransformerForecaster(..., device=device_type)
```

---

#### 📊 4. 报告从配置文件读取参数

**新增功能**：Word报告动态显示实际使用的模型参数

**更新内容**：
- ✅ 所有模型参数从配置文件读取（不再硬编码）
- ✅ 自动显示实际epoch、batch_size、dropout等
- ✅ 硬件加速设备类型实时显示

**示例报告输出**：
```
2.5 LSTM（长短期记忆网络）
参数配置：
• hidden_size: 64（隐藏层维度）
• num_layers: 2（LSTM堆叠层数）
• dropout: 0.3（防止过拟合的丢弃率）
• epochs: 100（训练轮数）
• batch_size: 64（批次大小）
• learning_rate: 0.001（Adam优化器学习率）
• 硬件加速: CPU
```

**优势**：
- 报告参数与实际训练配置100%一致
- 修改config后报告自动更新
- 便于实验记录和复现

---

### 2025年10月更新（之前版本）

#### ✨ 1. 硬件加速与进度可视化

**GPU/MPS自动检测**：深度学习模型（LSTM、Transformer）现支持：
- ✅ **CUDA GPU加速**（NVIDIA显卡）- 优先级1
- ✅ **Apple MPS加速**（M1/M2/M3芯片）- 优先级2
- ✅ **CPU后备**（通用兼容）- 优先级3

**训练进度条**：使用tqdm实时显示：
```
训练 LSTM (P): 100%|██████████| 50/50 [00:02<00:00, 19.5 epoch/s, loss=0.0234, device=mps]
```
- 显示当前epoch进度
- 实时损失值监控
- 使用设备类型（cuda/mps/cpu）
- 预估剩余时间

**性能提升**（基于M系列Mac测试）：
- LSTM训练：2-5分钟（50 epochs，MPS加速）
- Transformer训练：3-8分钟（50 epochs，MPS加速）
- CPU相比可提速3-5倍

#### 📊 2. 增强的报告生成

**图表详细解释**：每个图表现在都包含：
- **图表说明**：横纵轴含义、数据解读
- **关键观察点**：如何判断模型好坏
- **应用价值**：如何利用图表做决策

示例（误差对比图）：
> - 左右两图分别展示P和Q的预测误差
> - 横轴为预测步长（越大越难），纵轴为RMSE误差（越小越好）
> - 通常树模型和深度学习模型显著优于基线，曲线越平缓说明长期预测能力越强

**完整模型参数**：6个模型的训练配置完全透明：
- **基线模型**：参数说明（如seasonal_period=96）
- **树模型**：n_estimators, max_depth, learning_rate等
- **深度学习**：hidden_size, num_layers, dropout, batch_size等
- **硬件配置**：自动检测的加速设备类型

**双格式支持**：
- ✅ Markdown报告：适合在线查看和版本控制
- ✅ Word报告：专业排版，可直接打印汇报

#### 🗂️ 3. 时间戳输出管理

**自动时间戳文件夹**：每次运行生成独立输出：
```
outputs/
├── output-2025-10-26-1430/    # 带时间戳的输出
│   ├── figures/
│   ├── metrics/
│   └── report/
└── latest/                     # 指向最新运行结果的符号链接
```

**优势**：
- ✅ 历史结果永久保存，便于对比
- ✅ `latest/`快捷访问最新结果
- ✅ 支持实验版本管理
- ✅ 避免意外覆盖重要结果

#### 🎨 4. 完善的中文支持

**跨平台字体配置**：
- **Windows**：SimHei（黑体）、Microsoft YaHei（微软雅黑）
- **macOS**：PingFang SC（苹方）、STHeiti（华文黑体）
- **Linux**：WenQuanYi Micro Hei（文泉驿）、Noto Sans CJK SC（思源黑体）

**智能字体回退**：自动检测系统可用字体，确保图表中文正常显示

#### 📈 5. 完整的图表集成

所有4个关键图表现已包含在报告中：
1. **数据总览**：P和Q的时间序列可视化
2. **缺失值分布**：数据质量热图
3. **误差对比**：6个模型在不同步长的性能对比
4. **特征重要性**：树模型的特征排序（含外生变量）

每个图表都配有详细的解释和应用指导！

---

---

## ⭐ 核心功能

### 多模型 + 多策略 + 多步长

- **多模型集成**：经典模型(Naive/SeasonalNaive)、机器学习(随机森林/XGBoost)、深度学习(LSTM/Transformer)
- **多种填充策略**：8种数据填充方法(nearest_p/forward/backward/interpolate/mean/median/day_copy/seasonal)
- **严格交叉验证**：时间序列专用Split，保证未来数据不泄露
- **完整性能指标**：RMSE/MAE/SMAPE/WAPE/ACC_5/ACC_10 全方位评估
- **多步预测**：支持1-n步预测，horizons可灵活配置
- **自动报告生成**：生成Word和Markdown格式报告，包含图表和指标
- **端到端流程**：数据加载→特征工程→模型训练→评估→报告生成

---

## 📖 配置参数详解

### 数据配置 (data)

#### `file_path` - 数据文件路径
- **含义**：原始数据CSV文件的路径
- **示例**：`data/raw/电气多特征.csv`
- **要求**：文件必须包含时间列和目标变量(P)

#### `target_column` - 目标变量
- **含义**：需要预测的目标列名
- **示例**：`P` (有功功率)
- **说明**：可以是`P`、`Q`或其他数值列

#### `interp_limit` - 最大连续插值数量
- **含义**：线性插值时允许填充的最大连续缺失值数量
- **示例**：`interp_limit: 3` 表示最多插值3个连续缺失点
- **建议值**：
  - 3-5: 保守策略，避免过度插值
  - 10-20: 激进策略，适用于缺失较多的场景
- **影响**：过大会引入不真实的数据，过小会导致删除过多样本

#### `imputation` - 填充策略配置
- **strategies** (列表): 多策略对比模式
  ```yaml
  strategies:
    - nearest_p    # 基于P值接近性
    - forward      # 前向填充
    - interpolate  # 线性插值
    - day_copy     # 日期复制
  ```
- **method** (字符串): 单一策略模式(strategies为空时生效)
  ```yaml
  method: "interpolate"
  ```

#### 各填充方法的专用参数

| 参数 | 适用方法 | 含义 | 默认值 | 建议 |
|------|----------|------|--------|------|
| `target_p_value` | nearest_p | 目标有功功率参考值 | 400 | 选择数据集中常见的P值 |
| `day_copy_days_back` | day_copy | 回溯天数 | 7 | 7(周周期), 1(日周期) |
| `seasonal_period` | seasonal | 季节周期长度 | 24 | 24(小时), 96(15分钟) |

---

### 特征工程配置 (features)

#### `max_lag` - 最大滞后步数
- **含义**：使用多少个历史时刻的值作为输入特征
- **示例**：`max_lag: 24` → 使用 `P_lag_1`, `P_lag_2`, ..., `P_lag_24`
- **建议值**：
  - 至少等于最大预测步长 (≥ max(horizons))
  - 对于小时数据：24 (1天), 48 (2天), 168 (1周)
- **影响**：
  - 过小：模型学不到长期依赖关系
  - 过大：特征维度爆炸，训练变慢

#### `roll_windows` - 滚动窗口大小列表
- **含义**：计算滚动统计(均值/标准差)的窗口大小
- **示例**：`roll_windows: [6, 12, 24]`
  - `P_roll_mean_6`: 过去6小时的均值
  - `P_roll_std_12`: 过去12小时的标准差
  - `P_roll_mean_24`: 过去24小时的均值
- **建议值**：选择有业务意义的时间段
  - 小时数据：`[6, 12, 24]` (1/4天, 1/2天, 1天)
  - 15分钟数据：`[4, 24, 96]` (1小时, 6小时, 1天)
- **影响**：增加窗口捕获更长期趋势，但可能引入滞后

#### `use_time_features` - 是否使用时间特征
- **含义**：添加时间的周期性编码(sin/cos变换)
- **示例**：`true` → 生成 `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`
- **建议**：`true`（对于有周期性的时间序列非常有效）
- **原理**：使用sin/cos保持周期连续性(23点和0点相邻，而不是23→0跳变)

#### `sequence_length` - 序列长度(深度学习)
- **含义**：LSTM/Transformer的输入序列长度
- **示例**：`sequence_length: 24` → 输入形状为 `[batch, 24, features]`
- **建议值**：
  - 小时数据：24 (捕获日周期)
  - 15分钟数据：96 (捕获日周期)
  - 需要周周期：168 (7天×24小时)
- **影响**：
  - 过小：学不到足够的时序模式
  - 过大：训练慢，容易过拟合

#### `season_length` - 季节周期长度
- **含义**：SeasonalNaive模型使用的周期长度
- **示例**：`season_length: 24` → "今天10点的值 = 昨天10点的值"
- **建议值**：
  - 小时数据：24 (日周期)
  - 15分钟数据：96 (日周期)
  - 周周期：168 (小时数据), 672 (15分钟数据)
- **说明**：SeasonalNaive就是"复制N个周期前的值"

---

### 模型配置 (models)

#### LSTM配置
```yaml
lstm:
  batch_size: 64      # 批次大小，影响训练稳定性
  epochs: 100         # 训练轮数
  learning_rate: 0.001
  hidden_size: 64     # 隐藏层神经元数量
  num_layers: 2       # LSTM层数
```

**参数说明**：
- `hidden_size`: 隐藏层维度，影响模型容量
- `num_layers`: LSTM堆叠层数，增加可学习更复杂模式
- `batch_size`: 建议64 (CPU环境下最优)

#### Transformer配置
```yaml
transformer:
  batch_size: 32      # Transformer通常用更小的batch
  epochs: 200         # 需要更多轮数收敛
  learning_rate: 0.0001
  d_model: 64         # 模型维度
  nhead: 4            # 多头注意力头数(必须能整除d_model)
  num_layers: 2       # Transformer层数
```

**参数说明**：
- `d_model`: 模型维度，影响特征表达能力
- `nhead`: 多头注意力头数，必须能整除`d_model`
- `batch_size`: 建议32 (Transformer内存需求更大)

---

### 评估配置 (evaluation)

#### `horizons` - 预测步长列表
- **含义**：需要预测未来多少步(多步预测)
- **格式支持**：
  ```yaml
  # 方式1: 逗号分隔字符串(推荐)
  horizons: "1,2,3,4,5,6,7,8,9,10,11,12"
  
  # 方式2: YAML列表
  horizons: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  
  # 方式3: 非连续步长
  horizons: "1,6,12,24"
  ```
- **建议**：
  - 短期预测(1-12步): 适合日内调度
  - 中期预测(1-24步): 适合日前规划
  - 长期预测(1-168步): 适合周计划

#### `metrics` - 评估指标列表
- **含义**：用于评估模型性能的指标
- **可选值**：`RMSE`, `MAE`, `SMAPE`, `WAPE`, `ACC_5`, `ACC_10`
- **建议**：全部启用，多角度评估

---

### 交叉验证配置 (cv)

#### `n_splits` - 折数
- **含义**：将数据分成多少折进行交叉验证
- **建议值**：
  - `n_splits: 1` → 简单训练集/测试集划分(快速实验)
  - `n_splits: 3` → 3折交叉验证(平衡速度和准确性)
  - `n_splits: 5` → 5折交叉验证(更可靠，但慢5倍)
- **影响**：
  - 更多折数 → 更可靠的性能估计，但训练时间线性增加
  - 1折 → 最快，但可能受数据划分影响

#### `test_size` - 测试集比例
- **含义**：测试集占总数据的比例
- **示例**：`test_size: 0.2` → 80%训练，20%测试
- **建议**：0.2-0.3

---

### 输出配置 (output)

#### `base_dir` - 输出根目录
- **含义**：所有结果保存的根目录
- **示例**：`outputs/`

#### `device` - 计算设备
- **含义**：深度学习模型使用的设备
- **可选值**：
  - `"cpu"`: CPU计算(稳定，通用)
  - `"cuda"`: NVIDIA GPU(需要CUDA)
  - `"mps"`: Apple Silicon GPU(macOS专用)
  - `"auto"`: 自动检测最佳设备
- **建议**：
  - 小数据集(<10000样本): `cpu` 更快
  - 大数据集: `cuda` 或 `mps`

---

### 配置示例：典型场景

#### 场景1: 快速实验(单策略，少horizon)
```yaml
data:
  imputation:
    method: "interpolate"  # 只用线性插值
evaluation:
  horizons: "1,6,12"      # 只测3个步长
cv:
  n_splits: 1             # 不做交叉验证
```

#### 场景2: 全面对比(多策略，多horizon)
```yaml
data:
  imputation:
    strategies:           # 对比所有方法
      - nearest_p
      - forward
      - backward
      - interpolate
      - mean
      - median
      - day_copy
      - seasonal
evaluation:
  horizons: "1,2,3,4,5,6,7,8,9,10,11,12"  # 完整步长
cv:
  n_splits: 3            # 3折交叉验证
```

#### 场景3: 生产环境(高鲁棒性)
```yaml
data:
  interp_limit: 3        # 保守插值
  imputation:
    method: "day_copy"   # 使用稳定的日期复制
    day_copy_days_back: 7
features:
  max_lag: 48            # 2天历史
  roll_windows: [6, 12, 24, 48]  # 多尺度特征
evaluation:
  horizons: "1,6,12,24"  # 关键时间点
cv:
  n_splits: 5            # 高可靠性验证
output:
  device: "auto"         # 自动选择最佳设备
```

---

## 📊 详细功能说明

### 1. 多模型预测框架

| 模型类别 | 模型名称 | 特点 | 适用场景 |
|---------|---------|------|---------|
| **基线模型** | 朴素预测 (Naive) | 使用最后观测值 | 基准对比 |
| | 季节朴素 (Seasonal Naive) | 使用上季节值 | 季节性数据 |
| **树模型** | 随机森林 (RF) | 集成学习，特征重要性 | 中等规模数据 |
| | XGBoost | 梯度提升，高性能 | 大规模数据 |
| **深度学习** | LSTM | 长短期记忆网络 | 复杂时序模式 |
| | Transformer | 注意力机制 | 长序列依赖 |

### 2. 严格的验证机制

```
滚动起点交叉验证 (Rolling Origin Cross-Validation)
───────────────────────────────────────────────────────────
训练集        测试集
├──────────┤ ├──────┤
│ Fold 1   │ │      │
├──────────────────┤ ├──────┤
│ Fold 2           │ │      │
├──────────────────────────┤ ├──────┤
│ Fold 3                   │ │      │
───────────────────────────────────────────────────────────
时间 →
```

**关键特性**：
- ✅ 训练集始终在测试集之前（严格时间顺序）
- ✅ 训练集随折数递增（Expanding Window）
- ✅ 测试集大小固定，向前滚动
- ✅ 完全防止使用未来信息

**参考文献**：[Hyndman & Athanasopoulos - Time series cross-validation](https://otexts.com/fpp3/tscv.html)

### 3. 全面的评估体系

| 指标 | 公式 | 特点 | 应用场景 |
|------|------|------|----------|
| **RMSE** | √(Σ(yᵢ-ŷᵢ)²/n) | 对大误差敏感 | 主要指标 |
| **MAE** | Σ\|yᵢ-ŷᵢ\|/n | 线性惩罚 | 稳健评估 |
| **SMAPE** | 100·Σ(\|yᵢ-ŷᵢ\|/((\|yᵢ\|+\|ŷᵢ\|)/2))/n | 对称，0-200% | 相对误差 |
| **WAPE** | 100·Σ\|yᵢ-ŷᵢ\|/Σ\|yᵢ\| | 加权平均 | 避免零值问题 |

**注意**：本项目**不使用MAPE**，因其在真实值接近零时会产生极大值或未定义值。

### 4. 数据泄漏防范机制

**多层次防护**：

```python
# 特征工程层面
X[t] = {
    P[t-1], P[t-2], ..., P[t-lag],           # 滞后特征
    rolling_mean(P[t-lag:t]),                # 滚动统计
    hour_sin, hour_cos, ...                  # 时间特征
}
Y[t+h] = P[t+h]  # 预测目标

# 交叉验证层面
train: [0, t_train]
test:  [t_train+1, t_train+window]

# 标准化层面
scaler.fit(X_train)      # 只在训练集拟合
X_test_scaled = scaler.transform(X_test)  # 测试集变换
```

### 5. 外生变量支持（协变量）

**新增功能**：支持引入外部变量进行预测，例如：
- 电压（U）、电流（I）
- 转子电压（Vf）、转子电流（If）
- 蒸汽压力、温度、流量等

**关键原则**：
- ✅ 只使用**当前和过去**的外生变量值
- ✅ 为外生变量生成**滞后特征**和**滚动统计**
- ✅ 不使用未来外生变量值（防止信息泄漏）
- ✅ 符合动态回归（Dynamic Regression）标准

**参考文献**：[Hyndman & Athanasopoulos - Dynamic regression models](https://otexts.com/fpp2/dynamic.html)

### 6. 专业报告生成

- **Markdown报告**：公文式正式语言，包含方法论、结果表格、图表引用
- **Word报告**：使用python-docx生成，内联图片，专业排版
- **中文图表**：跨平台中文字体支持（Windows/macOS/Linux）
- **特征重要性**：树模型自动提取并可视化特征重要性

---

## 🏗️ 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        电力质量预测系统                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      1. 数据输入层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ CSV/Excel    │  │ 时间列检测    │  │ P/Q列检测    │          │
│  │ 数据文件      │→ │ 自动识别      │→ │ 自动识别      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                   │                 │                  │
│         └───────────────────┴─────────────────┘                  │
│                              │                                    │
│                              ▼                                    │
│  ┌───────────────────────────────────────────────────┐          │
│  │ 外生变量支持（可选）：电压、电流、温度等            │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      2. 特征工程层                                │
│  ┌──────────────────────────────────────────────────┐           │
│  │ 滞后特征 (Lag Features)                           │           │
│  │ • P[t-1], P[t-2], ..., P[t-lag]                  │           │
│  │ • Q[t-1], Q[t-2], ..., Q[t-lag]                  │           │
│  │ • 外生变量滞后特征（如启用）                       │           │
│  └──────────────────────────────────────────────────┘           │
│  ┌──────────────────────────────────────────────────┐           │
│  │ 滚动统计 (Rolling Statistics)                     │           │
│  │ • rolling_mean, rolling_std                      │           │
│  │ • rolling_min, rolling_max                       │           │
│  └──────────────────────────────────────────────────┘           │
│  ┌──────────────────────────────────────────────────┐           │
│  │ 时间特征 (Time Features)                          │           │
│  │ • hour_sin, hour_cos                             │           │
│  │ • dow_sin, dow_cos (星期)                        │           │
│  │ • month_sin, month_cos                           │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   3. 交叉验证与训练层                             │
│  ┌───────────────────────────────────────────────┐              │
│  │ 滚动起点交叉验证 (Rolling Origin CV)            │              │
│  │                                                │              │
│  │ Fold 1: ├────────train────────┤ ├─test─┤     │              │
│  │ Fold 2: ├────────train──────────────┤ ├─test─┤              │
│  │ Fold 3: ├────────train────────────────────┤ ├─test─┤        │
│  └───────────────────────────────────────────────┘              │
│                                                                   │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐              │
│  │ 基线模型    │  │ 树模型      │  │ 深度学习模型 │              │
│  ├────────────┤  ├────────────┤  ├──────────────┤              │
│  │ • Naive    │  │ • RF       │  │ • LSTM       │              │
│  │ • Seasonal │  │ • XGBoost  │  │ • Transformer│              │
│  └────────────┘  └────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      4. 评估与报告层                              │
│  ┌──────────────────────────────────────────────┐               │
│  │ 评估指标                                       │               │
│  │ RMSE │ MAE │ SMAPE │ WAPE                    │               │
│  └──────────────────────────────────────────────┘               │
│                              │                                    │
│              ┌───────────────┴───────────────┐                   │
│              ▼                               ▼                   │
│  ┌────────────────────┐         ┌────────────────────┐          │
│  │ 可视化图表          │         │ 报告生成            │          │
│  ├────────────────────┤         ├────────────────────┤          │
│  │ • 数据总览          │         │ • Markdown报告      │          │
│  │ • 误差对比          │         │ • Word报告          │          │
│  │ • 特征重要性        │         │ • 评估指标CSV       │          │
│  └────────────────────┘         └────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 项目文件结构

```
power-pq-forecast/
│
├── 📁 核心运行脚本
│   ├── run_all.py                              # 主程序：一键运行完整预测流程
│   ├── forecast_future.py                      # 未来预测：使用训练好的模型进行未来时段预测
│   ├── regenerate_report.py                    # 报告重生成：基于已有结果重新生成报告
│   └── test_model_loading.py                   # 测试工具：验证模型加载功能
│
├── 📁 超参数优化相关（W&B Sweep）
│   ├── start_sweep.py                          # Sweep启动器：创建并启动超参数优化
│   ├── train_sweep.py                          # Sweep训练器：单次超参数实验的训练逻辑
│   ├── sweep_config.yaml                       # Sweep配置：定义搜索空间和优化策略
│   ├── run_sweep.sh                            # Sweep脚本：后台运行sweep（nohup模式）
│   ├── requirements_sweep.txt                  # Sweep依赖：W&B相关包
│   └── SWEEP_GUIDE.md                          # Sweep指南：超参数优化使用文档
│
├── 📁 配置文件
│   ├── config_p_only.yaml                      # 标准配置：仅使用P/Q数据
│   ├── config_p_only_newdata.yaml              # 新数据配置：适用于新数据集
│   ├── config_p_only_newdata-transformer89.yaml # 最佳配置：89%准确率的Transformer参数
│   ├── config_fast_test.yaml                   # 快速测试：简化配置，用于调试
│   └── config_sweep.yaml                       # Sweep配置：超参数优化专用
│
├── 📁 工具脚本
│   ├── kill_wandb.sh                           # 进程管理：安全终止wandb进程（Bash版）
│   ├── kill_wandb.py                           # 进程管理：安全终止wandb进程（Python版）
│   ├── stop.sh                                 # 停止脚本：终止后台运行的训练
│   ├── regenerate_report.sh                    # 报告脚本：批量重新生成报告
│   ├── generate_architecture_diagram.py        # 架构图生成：绘制系统架构图
│   └── generate_sample_data.py                 # 数据生成：创建示例数据集
│
├── 📁 文档
│   ├── README.md                               # 项目说明：完整的功能介绍和使用指南
│   ├── USAGE.md                                # 使用指南：详细的操作步骤
│   ├── PROJECT_SUMMARY.md                      # 项目总结：实施总结和经验
│   ├── IMPLEMENTATION_NOTES.md                 # 实现笔记：技术细节和设计决策
│   ├── SWEEP_GUIDE.md                          # Sweep指南：超参数优化文档
│   ├── KILL_WANDB_GUIDE.md                     # 进程管理指南：wandb进程管理文档
│   └── SLEEP_MODE_GUIDE.md                     # 休眠模式指南：后台运行和进程管理
│
├── 📁 依赖配置
│   ├── requirements.txt                        # Python依赖：主程序所需包
│   └── requirements_sweep.txt                  # Sweep依赖：超参数优化额外包
│
├── 📁 源代码 (src/)
│   ├── __init__.py                             # 包初始化
│   ├── config.py                               # 配置管理：加载和验证配置文件
│   ├── data_io.py                              # 数据IO：读取、清洗、填充数据
│   ├── features.py                             # 特征工程：滞后、滚动统计、时间特征
│   ├── cv.py                                   # 交叉验证：时间序列专用的滚动起点验证
│   ├── metrics.py                              # 评估指标：RMSE/MAE/SMAPE/WAPE/ACC
│   ├── train_eval.py                           # 训练评估：模型训练和交叉验证主流程
│   ├── plots.py                                # 可视化：生成各类图表（中文支持）
│   ├── report_docx.py                          # Word报告：生成专业格式的Word文档
│   ├── model_manager.py                        # 模型管理：保存、加载、预测管理
│   ├── seed.py                                 # 随机种子：统一管理随机性
│   │
│   └── 📁 models/                              # 模型实现
│       ├── __init__.py                         # 模型包初始化
│       ├── baseline.py                         # 基线模型：Naive, SeasonalNaive
│       ├── tree.py                             # 树模型：RandomForest, XGBoost, LightGBM
│       ├── lstm.py                             # LSTM：长短期记忆网络
│       └── transformer.py                      # Transformer：注意力机制模型
│
├── 📁 数据目录 (data/raw/)
│   └── [用户数据文件.csv/.xlsx]                # 原始数据：时间序列CSV或Excel文件
│
└── 📁 输出目录 (outputs/)                       # 自动生成，包含所有运行结果
    ├── output-YYYY-MM-DD-HHMM/                 # 时间戳输出文件夹
    │   ├── config_used.yaml                    # 备份配置：记录本次运行的配置
    │   ├── logs/training.log                   # 训练日志：完整的运行日志
    │   ├── figures_[strategy]/                 # 图表：数据可视化和模型对比
    │   ├── metrics_[strategy]/                 # 指标：评估结果CSV文件
    │   ├── models_[strategy]/                  # 模型：训练好的模型文件(.pkl)
    │   └── report/                             # 报告：Word格式的评估报告
    └── latest/                                  # 快捷方式：指向最新运行结果
```

### 文件功能说明

#### 🎯 核心运行脚本

| 文件 | 功能说明 |
|------|---------|
| `run_all.py` | **主程序**。执行完整的预测流程：数据加载→特征工程→模型训练→交叉验证→生成报告。支持多种数据填充策略、模型加载等功能。 |
| `forecast_future.py` | **未来预测工具**。使用已训练的模型对未来时间段进行预测，生成预测结果CSV文件。 |
| `regenerate_report.py` | **报告重生成工具**。当训练已完成但报告需要修改时，基于已保存的结果重新生成Word报告，避免重新训练。 |
| `test_model_loading.py` | **模型加载测试**。验证已保存的LSTM和Transformer模型能否正确加载，用于调试模型持久化功能。 |

#### 🔧 超参数优化系统

| 文件 | 功能说明 |
|------|---------|
| `start_sweep.py` | **Sweep启动器**。创建W&B Sweep并启动超参数优化，支持快速测试、标准、深度等多种模式。 |
| `train_sweep.py` | **Sweep训练器**。单次超参数实验的训练逻辑，由W&B调用执行不同参数组合的训练。 |
| `sweep_config.yaml` | **Sweep配置文件**。定义超参数搜索空间（d_model、nhead、lr等）和优化策略（贝叶斯优化+Hyperband早停）。 |
| `run_sweep.sh` | **后台运行脚本**。使用nohup在后台运行sweep，适合长时间优化实验。 |
| `SWEEP_GUIDE.md` | **Sweep使用指南**。详细的超参数优化教程，包括配置说明、最佳实践和结果分析。 |

#### ⚙️ 配置文件

| 文件 | 功能说明 |
|------|---------|
| `config_p_only.yaml` | **标准配置**。仅使用有功功率(P)和无功功率(Q)进行预测，不使用外生变量。 |
| `config_p_only_newdata.yaml` | **新数据配置**。针对新数据集的配置，包含适配的参数设置。 |
| `config_p_only_newdata-transformer89.yaml` | **最佳参数配置**。通过W&B Sweep找到的最优Transformer参数（ACC_10=89%），可直接使用。 |
| `config_fast_test.yaml` | **快速测试配置**。简化的配置，用于快速验证流程，适合开发调试。 |

#### 🛠️ 工具脚本

| 文件 | 功能说明 |
|------|---------|
| `kill_wandb.sh` | **进程管理工具(Bash)**。安全终止所有wandb相关进程，但保护run_all.py主程序，避免误杀。 |
| `kill_wandb.py` | **进程管理工具(Python)**。功能同上，提供更强的进程识别能力，支持--force强制终止。 |
| `stop.sh` | **停止脚本**。快速终止后台运行的训练进程。 |
| `regenerate_report.sh` | **批量报告生成**。批量重新生成多个输出目录的报告。 |

#### 📚 文档

| 文件 | 功能说明 |
|------|---------|
| `README.md` | **项目主文档**。完整的功能介绍、使用方法、技术架构说明，是了解项目的入口文档。 |
| `USAGE.md` | **使用指南**。详细的操作步骤和配置说明，适合实际使用时查阅。 |
| `PROJECT_SUMMARY.md` | **项目总结**。项目实施的经验总结和技术要点。 |
| `IMPLEMENTATION_NOTES.md` | **实现笔记**。技术细节、设计决策和代码实现说明。 |
| `SWEEP_GUIDE.md` | **超参数优化指南**。W&B Sweep的完整使用教程。 |
| `KILL_WANDB_GUIDE.md` | **进程管理指南**。wandb进程管理工具的使用说明。 |
| `SLEEP_MODE_GUIDE.md` | **后台运行指南**。如何在后台运行训练和管理长时间任务。 |

#### 🔍 源代码模块

| 文件 | 功能说明 |
|------|---------|
| `src/config.py` | **配置管理**。加载YAML配置文件，提供统一的配置访问接口，支持嵌套配置和默认值。 |
| `src/data_io.py` | **数据处理**。读取CSV/Excel数据，处理时间序列，支持8种数据填充策略（nearest_p、forward、interpolate等）。 |
| `src/features.py` | **特征工程**。生成滞后特征、滚动统计特征、时间周期特征，支持外生变量的特征提取。 |
| `src/cv.py` | **交叉验证**。实现时间序列专用的滚动起点交叉验证，严格防止数据泄漏。 |
| `src/metrics.py` | **评估指标**。计算RMSE、MAE、SMAPE、WAPE、ACC_5、ACC_10等多种评估指标。 |
| `src/train_eval.py` | **训练评估**。模型训练和交叉验证的主流程，支持模型加载、多策略对比等功能。 |
| `src/plots.py` | **可视化**。生成数据总览、误差对比、特征重要性等图表，支持跨平台中文字体。 |
| `src/report_docx.py` | **报告生成**。生成专业格式的Word评估报告，包含表格、图表和详细说明。 |
| `src/model_manager.py` | **模型管理**。模型的保存、加载、未来预测等管理功能。 |
| `src/seed.py` | **随机种子**。统一管理Python、NumPy、PyTorch的随机种子，确保实验可复现。 |

#### 🤖 模型实现

| 文件 | 功能说明 |
|------|---------|
| `src/models/baseline.py` | **基线模型**。实现Naive（朴素预测）和SeasonalNaive（季节朴素）基线模型，用于对比评估。 |
| `src/models/tree.py` | **树模型**。实现RandomForest、XGBoost、LightGBM等基于决策树的集成学习模型。 |
| `src/models/lstm.py` | **LSTM模型**。长短期记忆网络实现，支持多步预测和GPU/MPS加速。 |
| `src/models/transformer.py` | **Transformer模型**。基于注意力机制的Transformer实现，支持encoder-decoder架构和多步预测。 |

#### 📦 依赖管理

| 文件 | 功能说明 |
|------|---------|
| `requirements.txt` | **主程序依赖**。numpy、pandas、scikit-learn、xgboost、torch、matplotlib等核心包。 |
| `requirements_sweep.txt` | **Sweep额外依赖**。wandb等超参数优化相关的额外包。 |

---

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- 推荐使用虚拟环境（venv/conda）

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/zh4men9/power-pq-forecast.git
cd power-pq-forecast
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包列表**：
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
torch>=1.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=6.0
python-docx>=0.8.11
openpyxl>=3.0.0
tqdm>=4.62.0              # 新增：进度条显示
```

#### 3. 准备数据

将您的电力数据文件（Excel或CSV格式）放置在 `data/raw/` 目录下。

**数据格式要求**：
- ✅ 必须包含**时间列**（支持中英文：时间、时间戳、日期、time、timestamp等）
- ✅ 必须包含**有功功率列**（支持：有功、P、active、power等）
- ✅ 必须包含**无功功率列**（支持：无功、Q、reactive等）
- ✅ 可选包含**外生变量列**（电压、电流、温度等）
- ✅ 时间列应为标准日期时间格式

**数据示例**：

| 时间 | 有功 | 无功 | 定子电流 | 定子电压 |
|------|------|------|----------|----------|
| 2025-06-01 00:00:00 | 300.07 | -53.48 | 9179.52 | 19.07 |
| 2025-06-01 01:00:00 | 309.18 | -70.31 | 9579.14 | 18.99 |
| ... | ... | ... | ... | ... |

#### 4. 配置参数（可选）

编辑 `config.yaml` 文件以调整预测参数：

```yaml
# 数据配置
data:
  data_path: "data/raw"
  file_pattern: "*.csv"          # 匹配数据文件
  freq: "H"                      # 时间频率（H=小时）

# 特征工程配置
features:
  max_lag: 24                    # 最大滞后步数
  roll_windows: [6, 12, 24]      # 滚动窗口大小
  exog_cols: []                  # 外生变量列表（留空则仅使用P/Q）
  # 示例：exog_cols: ['定子电流', '定子电压', '转子电压', '转子电流']

# 评估配置
evaluation:
  horizons: [1, 12, 24]          # 预测步长
  test_window: 100               # 测试窗口大小
  n_splits: 3                    # 交叉验证折数
  
  # 深度学习多步预测策略（新功能！）
  # multiple_output: 训练1个模型预测所有horizons（快速，默认）
  # direct: 为每个horizon训练独立模型（慢，可能更准确）
  deep_learning_strategy: "multiple_output"

# 模型配置（可启用/禁用特定模型）
models:
  naive: {enabled: true}
  seasonal_naive: {enabled: true}
  rf: {enabled: true}
  xgb: {enabled: true}
  lstm: {enabled: true}
  transformer: {enabled: true}
```

#### 5. 运行预测

```bash
python run_all.py --config config.yaml
```

**运行流程**：
```
步骤 1/6: 加载配置文件
步骤 2/6: 加载数据
步骤 3/6: 模型训练与评估
步骤 4/6: 生成图表
步骤 5/6: 生成Markdown报告
步骤 6/6: 生成Word报告
```

#### 6. 查看结果

运行完成后，查看以下文件：

- **评估指标**：`outputs/metrics/cv_metrics.csv`
- **图表**：`outputs/figures/`
- **Markdown报告**：`outputs/report/项目评估报告.md`
- **Word报告**：`outputs/report/项目评估报告.docx`

---

## 📖 详细使用

### 使用场景 A：仅使用 P/Q 进行预测（最简单）

**适用情况**：
- 只有有功功率（P）和无功功率（Q）数据
- 希望快速建立基线模型
- 数据量较小，无需复杂特征

**配置**：
```yaml
features:
  max_lag: 24
  roll_windows: [6, 12, 24]
  exog_cols: []                  # 留空或注释掉此行
```

**运行**：
```bash
python run_all.py --config config.yaml
```

**预期结果**：
- 特征数量：约 14 个（2个目标 × (3个滞后 + 4个滚动统计) + 时间特征）
- 适合快速验证和基线对比

### 使用场景 B：引入外生变量（更强预测能力）

**适用情况**：
- 有额外的测量数据（电压、电流、温度等）
- 希望利用更多信息提升预测精度
- 数据量充足，可以支撑更多特征

**配置**：
```yaml
features:
  max_lag: 24
  roll_windows: [6, 12, 24]
  exog_cols: ['定子电流', '定子电压', '转子电压', '转子电流']  # 指定外生变量
```

**运行**：
```bash
python run_all.py --config config.yaml
```

**预期结果**：
- 特征数量：约 42 个（6个变量 × 7个衍生特征）
- 树模型会生成特征重要性图，显示哪些变量最有用
- 深度学习模型自动适配更高维度输入

**关键原则**：
- ✅ 外生变量只使用**当前和历史值**
- ✅ 不使用未来外生变量值（防止信息泄漏）
- ✅ 符合动态回归（Dynamic Regression）标准

### 调整数据频率

如果数据是**1分钟采样**而非1小时：

```yaml
data:
  freq: "1min"                   # 或 "T"

evaluation:
  test_window: 1440              # 1天 = 1440分钟
  horizons: [1, 12, 60]          # 1分钟、12分钟、1小时

features:
  season_length: 1440            # 1天的季节周期
```

### 快速测试（使用测试配置）

使用 `config_test.yaml` 快速验证流程：

```bash
python run_all.py --config config_test.yaml
```

**测试配置特点**：
- 更小的测试窗口
- 更少的交叉验证折数
- 禁用部分耗时模型
- 适合开发调试

---

## 📚 方法说明

### 滚动起点交叉验证 (Rolling Origin Cross-Validation)

**原理**：时间序列预测的标准验证方法，确保评估结果可靠。

**关键特性**：
1. **时间顺序**：训练集始终在测试集之前
2. **扩展窗口**：训练集随每折递增（Expanding Window）
3. **固定测试集**：测试集大小固定，向前滚动
4. **防止泄漏**：严格不使用未来信息

**图示**：
```
数据：[═══════════════════════════════════════════════]

Fold 1: [════════训练════════] [测试]
Fold 2: [═══════════训练═══════════] [测试]
Fold 3: [══════════════训练══════════════] [测试]

时间 →
```

**参考文献**：
- [Hyndman & Athanasopoulos - Time series cross-validation](https://otexts.com/fpp3/tscv.html)
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

### 评估指标说明

#### RMSE (Root Mean Squared Error)
- **公式**：√(Σ(yᵢ-ŷᵢ)²/n)
- **特点**：对大误差敏感，惩罚离群点
- **应用**：主要评估指标

#### MAE (Mean Absolute Error)
- **公式**：Σ|yᵢ-ŷᵢ|/n
- **特点**：线性惩罚，稳健性好
- **应用**：辅助评估

#### SMAPE (Symmetric Mean Absolute Percentage Error)
- **公式**：100·Σ(|yᵢ-ŷᵢ|/((|yᵢ|+|ŷᵢ|)/2))/n
- **特点**：对称，范围0-200%
- **应用**：相对误差评估

#### WAPE (Weighted Absolute Percentage Error)
- **公式**：100·Σ|yᵢ-ŷᵢ|/Σ|yᵢ|
- **特点**：加权平均，避免零值问题
- **应用**：替代MAPE

**为什么不使用MAPE**：
- ❌ 真实值接近零时会产生极大值
- ❌ 对正负误差不对称
- ❌ 不适合功率预测（可能出现零值或接近零值）

**参考文献**：
- [Rob J Hyndman - Forecast errors](https://robjhyndman.com/hyndsight/forecastmse/)
- [statworx - WAPE vs MAPE](https://www.statworx.com/en/content-hub/blog/what-the-mape-is-falsely-blamed-for-its-true-weaknesses-and-better-alternatives)

### 数据泄漏防范

**三层防护机制**：

#### 1. 特征工程层面
```python
# ✅ 正确：使用shift()确保只用过去
P_lag_1 = df['P'].shift(1)          # t-1时刻的值
rolling_mean = df['P'].shift(1).rolling(6).mean()  # t-6到t-1的均值

# ❌ 错误：直接使用当前值
P_current = df['P']                  # 泄漏！
rolling_mean = df['P'].rolling(6).mean()  # 泄漏！
```

#### 2. 交叉验证层面
```python
# ✅ 正确：时间顺序划分
train: [0, 1000]
test:  [1001, 1100]

# ❌ 错误：随机划分
train, test = train_test_split(X, y, shuffle=True)  # 泄漏！
```

#### 3. 标准化层面
```python
# ✅ 正确：在训练集拟合，测试集变换
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)

# ❌ 错误：在全量数据拟合
scaler.fit(X)  # 泄漏！
```

### 外生变量使用规则

**动态回归（Dynamic Regression）原则**：

```python
# 预测 t+h 时刻的 P 和 Q
Y[t+h] = f(P[t-1:t-lag], Q[t-1:t-lag], 
           U[t-1:t-lag], I[t-1:t-lag],  # 外生变量的历史值
           time_features[t])

# ✅ 只使用 t 及之前的外生变量
# ❌ 不使用 t+1 到 t+h 的外生变量（未来不可得）
```

**参考文献**：
- [Hyndman & Athanasopoulos - Dynamic regression](https://otexts.com/fpp2/dynamic.html)

---

## 📊 输出说明

### 1. 评估指标表 (`outputs/metrics/cv_metrics.csv`)

**格式**：

| 列名 | 说明 |
|------|------|
| model | 模型名称（Naive, RF, XGBoost, LSTM等） |
| horizon | 预测步长（1, 12, 24等） |
| fold | 交叉验证折编号（0, 1, 2） |
| target | 目标变量（P或Q） |
| RMSE | 均方根误差 |
| MAE | 平均绝对误差 |
| SMAPE | 对称平均绝对百分比误差 |
| WAPE | 加权绝对百分比误差 |

**用途**：
- 对比不同模型的性能
- 分析不同步长的误差变化
- 评估交叉验证的稳定性

### 2. 可视化图表 (`outputs/figures/`)

#### `data_overview.png` - 数据总览
- 显示P和Q的时间序列曲线
- 帮助识别趋势和季节性

#### `missing_data.png` - 缺失值分布
- 热图显示缺失值位置
- 评估数据质量

#### `error_by_horizon.png` - 误差随步长变化
- 对比不同模型在各步长的表现
- 帮助选择最佳模型

#### `feature_importance.png` - 特征重要性（树模型）
- 显示哪些特征对预测最有用
- 仅在使用树模型时生成

### 3. 报告文件 (`outputs/report/`)

#### Markdown报告 (`项目评估报告.md`)
- **方法论说明**：介绍使用的模型和验证方法
- **结果表格**：汇总评估指标
- **图表引用**：内嵌图表链接
- **结论建议**：专业分析和建议
- **格式**：适合在线查看、版本控制

#### Word报告 (`项目评估报告.docx`)
- **专业排版**：标题、表格、图片格式化
- **内联图片**：图表直接嵌入文档
- **可编辑**：可进一步修改和打印
- **格式**：适合正式汇报和存档

---

## 💡 最佳实践

### 1. 数据准备

✅ **推荐做法**：
- 检查数据完整性（缺失值、异常值）
- 确保时间列格式正确
- 查看数据总览图，了解数据特征

❌ **避免做法**：
- 使用未清洗的原始数据
- 忽略大量缺失值
- 不检查数据分布

### 2. 模型选择

✅ **推荐做法**：
- 先运行基线模型作为参照
- 从简单模型（RF）到复杂模型（LSTM）逐步尝试
- 对比不同模型的RMSE和稳定性

❌ **避免做法**：
- 直接使用最复杂的模型
- 忽略基线模型
- 只看单一指标

### 3. 超参数调整

✅ **推荐做法**：
- 使用`config_test.yaml`快速测试
- 逐步调整`max_lag`、`roll_windows`
- 记录不同配置的结果

❌ **避免做法**：
- 随意调整参数
- 不记录实验结果
- 过度拟合训练集

### 4. 外生变量使用

✅ **推荐做法**：
- 先用P/Q建立基线
- 再逐步添加外生变量
- 检查特征重要性，移除无用变量

❌ **避免做法**：
- 一次性添加所有变量
- 使用未来不可得的变量
- 不检查特征重要性

### 5. 结果解读

✅ **推荐做法**：
- 综合考虑RMSE、MAE、SMAPE、WAPE
- 评估不同折之间的方差（稳定性）
- 分析误差随步长的变化趋势

❌ **避免做法**：
- 只看单一指标
- 忽略交叉验证方差
- 不分析失败案例

### 6. 字体配置（跨平台）

**Windows系统**：
```yaml
plotting:
  font_priority:
    - SimHei              # 黑体
    - Microsoft YaHei     # 微软雅黑
```

**macOS系统**：
```yaml
plotting:
  font_priority:
    - PingFang SC         # 苹方-简
    - STHeiti             # 华文黑体
```

**Linux系统**：
```yaml
plotting:
  font_priority:
    - WenQuanYi Micro Hei # 文泉驿微米黑
    - Noto Sans CJK SC    # 思源黑体
```

### 7. 硬件加速（深度学习模型）

✅ **自动检测**：系统自动选择最优计算设备
```python
# 优先级顺序
1. CUDA GPU (NVIDIA显卡)
2. Apple MPS (M1/M2/M3芯片)
3. CPU (通用兼容)
```

✅ **推荐配置**：
- **NVIDIA GPU**：确保安装CUDA版本的PyTorch
- **Apple Silicon**：PyTorch 2.0+自动支持MPS
- **性能监控**：观察进度条显示的device类型

❌ **避免做法**：
- 强制指定不支持的设备
- 在CPU上训练超大模型
- 忽略GPU驱动更新

### 8. 输出管理

✅ **推荐做法**：
- 使用时间戳文件夹管理实验结果
- 通过`latest/`快速访问最新结果
- 定期清理旧的输出文件夹

**示例**：
```bash
# 查看最新结果
ls outputs/latest/

# 对比两次运行
diff outputs/output-2025-10-26-1430/metrics/cv_metrics.csv \
     outputs/output-2025-10-26-1645/metrics/cv_metrics.csv

# 清理旧结果（保留最近5次）
ls -t outputs/output-* | tail -n +6 | xargs rm -rf
```

---

## 🔧 常见问题

### Q1: 如何处理缺失值？

**A**: 系统自动进行短缺口插值（默认最多3个连续缺失值）。长缺口不自动填充，避免引入虚假模式。

可调整：
```yaml
data:
  interp_limit: 3  # 修改插值限制
```

### Q2: 如何选择最佳模型？

**A**: 综合考虑：
1. **精度**：在所有步长的平均RMSE
2. **稳定性**：交叉验证的方差
3. **计算成本**：训练和预测时间
4. **可解释性**：树模型提供特征重要性

### Q3: 为什么某些模型比基线还差？

**A**: 可能原因：
1. 数据量不足以训练复杂模型
2. 模型超参数需要调整
3. 数据模式过于简单，复杂模型过拟合

**解决方案**：
- 增加数据量
- 调整超参数（减小`max_depth`、增加`dropout`）
- 使用更简单的模型

### Q4: 如何加快运行速度？

**A**: 优化方法：
1. 减少交叉验证折数：`n_splits: 2`
2. 减少预测步长数量
3. 禁用耗时模型（LSTM、Transformer）
4. 减少树的数量：`n_estimators: 50`
5. 使用GPU（深度学习自动检测）

### Q5: 图表中文显示为方块？

**A**: 字体问题：
1. 确保系统安装了中文字体
2. 修改`config.yaml`中的`font_priority`列表
3. 将常用字体放在列表前面

### Q6: 如何添加更多外生变量？

**A**: 编辑配置：
```yaml
features:
  exog_cols: 
    - '定子电流'
    - '定子电压'
    - '转子电压'
    - '转子电流'
    - '励磁电流'
    # 添加更多...
```

确保列名与数据文件中的列名**完全一致**。

### Q7: 如何查看训练进度？

**A**: 深度学习模型（LSTM、Transformer）自动显示tqdm进度条：

```
训练 LSTM (P): 100%|██████████| 50/50 [00:02<00:00, 19.5 epoch/s, loss=0.0234, device=mps]
```

**信息说明**：
- **进度百分比**：当前训练进度
- **Epoch数**：完成轮次/总轮次
- **耗时**：已用时间 < 预估剩余时间
- **速度**：每秒训练的epoch数
- **Loss**：当前损失值
- **Device**：使用的计算设备（cuda/mps/cpu）

### Q8: GPU加速不工作怎么办？

**A**: 检查步骤：

1. **确认硬件支持**：
   - NVIDIA GPU：运行 `nvidia-smi` 检查显卡
   - Apple Silicon：运行 `system_profiler SPHardwareDataType` 查看芯片

2. **验证PyTorch安装**：
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

3. **重新安装PyTorch**：
   - CUDA版本：访问 [pytorch.org](https://pytorch.org/get-started/locally/)
   - MPS版本：`pip install torch>=2.0.0`

4. **查看训练日志**：
   ```bash
   # 运行测试脚本
   python test_gpu_acceleration.py
   ```

---

## 📚 参考文献

### 教材与书籍

1. **Hyndman, R.J., & Athanasopoulos, G.** (2021). *Forecasting: Principles and Practice*, 3rd edition, OTexts.
   - 时间序列预测的权威教材
   - [在线阅读](https://otexts.com/fpp3/)

2. **Hyndman, R.J., & Athanasopoulos, G.** (2018). *Forecasting: Principles and Practice*, 2nd edition, OTexts.
   - 动态回归章节
   - [在线阅读](https://otexts.com/fpp2/dynamic.html)

### 学术论文

3. **Vaswani et al.** (2017). *Attention is All You Need*. NeurIPS.
   - Transformer架构原始论文

4. **Mei, H., & Eisner, J.M.** (2020). *The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process*. arXiv:1612.09328.
   - 事件序列模型（**注意**：不适用于规则采样的功率曲线）

### 技术文档

5. **scikit-learn Documentation**
   - [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
   - [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

6. **XGBoost Documentation**
   - [官方文档](https://xgboost.readthedocs.io/)

7. **PyTorch Documentation**
   - [官方文档](https://pytorch.org/docs/)

8. **python-docx Documentation**
   - [官方文档](https://python-docx.readthedocs.io/)

### 博客与教程

9. **Rob J Hyndman's blog**
   - [Forecast errors](https://robjhyndman.com/hyndsight/forecastmse/)
   - 为什么MAPE有问题

10. **statworx.com**
    - [WAPE vs MAPE](https://www.statworx.com/en/content-hub/blog/what-the-mape-is-falsely-blamed-for-its-true-weaknesses-and-better-alternatives)

11. **Cross Validated (StackExchange)**
    - [ARIMAX Forecasting](https://stats.stackexchange.com/questions/180217/arimax-forecasting-in-spss-vs-r)

12. **jdhao's blog**
    - [Guide on how to use Chinese with Matplotlib](https://jdhao.github.io/2017/05/13/guide-on-how-to-use-chinese-with-matplotlib/)

---

## 📄 许可证

MIT License

---

## 👥 贡献与支持

### 贡献指南

欢迎提交Issue和Pull Request！

**贡献方式**：
1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 联系方式

- **Issues**: [GitHub Issues](https://github.com/zh4men9/power-pq-forecast/issues)
- **Email**: 联系项目维护者

---

## 🌟 致谢

本项目遵循时间序列预测的学术标准和工业最佳实践，感谢以下资源的启发：

- Rob J Hyndman教授的《预测：原理与实践》教材
- scikit-learn、XGBoost、PyTorch等开源社区
- 所有贡献者和用户的反馈

---

<div align="center">

**如果这个项目对您有帮助，请给个 ⭐️ Star！**

[返回顶部](#电力质量预测系统-power-quality-forecasting-system)

</div>
