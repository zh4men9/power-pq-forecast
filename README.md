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

## � 项目文件结构

```
power-pq-forecast/
│
├── run_all.py                          # 主程序：完整预测流程
├── forecast_future.py                  # 未来预测工具
├── regenerate_report.py                # 报告重生成工具
├── test_model_loading.py               # 模型加载测试
│
├── start_sweep.py                      # Sweep启动器
├── train_sweep.py                      # Sweep训练器
├── sweep_config.yaml                   # Sweep配置
├── run_sweep.sh                        # Sweep后台运行脚本
│
├── config_p_only.yaml                  # 标准配置
├── config_p_only_newdata.yaml          # 新数据配置
├── config_p_only_newdata-transformer89.yaml  # 最佳配置（89%准确率）
├── config_fast_test.yaml               # 快速测试配置
│
├── kill_wandb.sh                       # 进程管理（Bash版）
├── kill_wandb.py                       # 进程管理（Python版）
├── stop.sh                             # 停止脚本
│
├── README.md                           # 项目说明
├── USAGE.md                            # 使用指南
├── SWEEP_GUIDE.md                      # Sweep指南
├── KILL_WANDB_GUIDE.md                 # 进程管理指南
├── PROJECT_SUMMARY.md                  # 项目总结
│
├── requirements.txt                    # Python依赖
├── requirements_sweep.txt              # Sweep依赖
│
├── src/                                # 源代码目录
│   ├── config.py                       # 配置管理
│   ├── data_io.py                      # 数据IO
│   ├── features.py                     # 特征工程
│   ├── cv.py                           # 交叉验证
│   ├── metrics.py                      # 评估指标
│   ├── train_eval.py                   # 训练评估
│   ├── plots.py                        # 可视化
│   ├── report_docx.py                  # 报告生成
│   ├── model_manager.py                # 模型管理
│   ├── seed.py                         # 随机种子
│   └── models/                         # 模型实现
│       ├── baseline.py                 # 基线模型
│       ├── tree.py                     # 树模型
│       ├── lstm.py                     # LSTM
│       └── transformer.py              # Transformer
│
├── data/raw/                           # 原始数据目录
│   └── [数据文件.csv/.xlsx]
│
└── outputs/                            # 输出目录（自动生成）
    ├── output-YYYY-MM-DD-HHMM/         # 时间戳输出
    │   ├── config_used.yaml
    │   ├── logs/
    │   ├── figures_[strategy]/
    │   ├── metrics_[strategy]/
    │   ├── models_[strategy]/
    │   └── report/
    └── latest/                         # 最新结果快捷方式
```

---

## �🚀 快速开始

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

**电力质量预测系统 © 2025**
