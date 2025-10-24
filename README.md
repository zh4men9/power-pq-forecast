# 电力质量预测项目 (Power Quality Forecasting)

## 项目简介

本项目是一个完整的电力时序预测系统，用于预测电力系统的有功功率（P）和无功功率（Q）。项目采用多种先进的预测模型，包括基线方法、传统机器学习和深度学习方法，并使用滚动起点交叉验证确保评估的可靠性。

## 主要特性

- **多种预测模型**：朴素基线、季节朴素、随机森林、XGBoost、LSTM、Transformer
- **严格的验证方法**：滚动起点交叉验证（Rolling Origin），防止数据泄漏
- **全面的评估指标**：RMSE、MAE、SMAPE、WAPE
- **中文报告生成**：自动生成Markdown和Word格式的专业报告
- **可视化图表**：全中文图表，支持多种字体

## 项目结构

```
pq_forecast/
├── data/
│   └── raw/                 # 原始数据目录（放置Excel/CSV文件）
├── outputs/
│   ├── figures/             # 生成的图表
│   ├── metrics/             # 评估指标CSV
│   └── report/              # Markdown和Word报告
├── src/
│   ├── config.py            # 配置加载与验证
│   ├── data_io.py           # 数据读取与预处理
│   ├── features.py          # 特征工程
│   ├── metrics.py           # 评估指标
│   ├── cv.py                # 交叉验证
│   ├── models/
│   │   ├── baseline.py      # 基线模型
│   │   ├── tree.py          # 树模型（RF/XGB）
│   │   ├── lstm.py          # LSTM模型
│   │   └── transformer.py   # Transformer模型
│   ├── train_eval.py        # 训练评估主流程
│   ├── plots.py             # 绘图工具
│   ├── report_md.py         # Markdown报告生成
│   └── report_docx.py       # Word报告生成
├── config.yaml              # 配置文件
├── run_all.py               # 一键运行脚本
├── requirements.txt         # Python依赖
└── README.md                # 本文件
```

## 安装说明

### 1. 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备数据

将您的电力数据文件（Excel或CSV格式）放置在 `data/raw/` 目录下。

数据要求：
- 必须包含时间列（支持中英文列名：时间、时间戳、日期、time、timestamp等）
- 必须包含有功功率列（支持：有功、P、active、power等）
- 必须包含无功功率列（支持：无功、Q、reactive等）
- 时间列应为标准日期时间格式

数据示例：

| 时间 | 有功功率 | 无功功率 |
|------|---------|---------|
| 2023-01-01 00:00:00 | 100.5 | 50.2 |
| 2023-01-01 01:00:00 | 105.3 | 52.1 |
| ... | ... | ... |

## 使用说明

### 基本使用

```bash
python run_all.py --config config.yaml
```

### 配置文件说明

编辑 `config.yaml` 来调整项目参数：

#### 数据配置
- `data_path`: 数据文件目录
- `freq`: 时间频率（H=小时，D=日，W=周，M=月，T=分钟）
- `interp_limit`: 短缺口插值最大步数

#### 特征配置
- `max_lag`: 最大滞后步数（用于特征工程）
- `roll_windows`: 滚动窗口大小列表
- `sequence_length`: 深度学习序列长度
- `season_length`: 季节周期（用于季节基线）

#### 评估配置
- `horizons`: 预测步长列表（如 [1, 12, 24]）
- `test_window`: 测试窗口大小
- `n_splits`: 交叉验证折数

#### 模型配置
每个模型都可以单独启用/禁用和配置超参数。

## 方法说明

### 交叉验证方法

本项目使用**滚动起点交叉验证**（Rolling Origin Cross-Validation），这是时间序列预测的标准方法：

- 训练集始终在测试集之前（时间顺序）
- 训练集随每折递增（expanding window）
- 测试集大小固定，向前滚动
- 严格防止使用未来信息

参考：[Forecasting: Principles and Practice - Time series cross-validation](https://otexts.com/fpp3/tscv.html)

### 评估指标

- **RMSE** (Root Mean Squared Error): 均方根误差，对大误差敏感
- **MAE** (Mean Absolute Error): 平均绝对误差
- **SMAPE** (Symmetric Mean Absolute Percentage Error): 对称平均绝对百分比误差
- **WAPE** (Weighted Absolute Percentage Error): 加权绝对百分比误差

注：本项目不使用MAPE，因其在真实值接近零时会产生极大值。

### 基线模型

- **朴素预测** (Naive): 使用最后一个观测值作为预测
- **季节朴素预测** (Seasonal Naive): 使用上一季节的对应值作为预测

所有模型的性能都应与基线进行比较。

## 输出说明

运行完成后，将生成以下输出：

1. **指标表** (`outputs/metrics/cv_metrics.csv`)
   - 包含所有模型在每折、每步长、每目标上的详细指标

2. **图表** (`outputs/figures/`)
   - `data_overview.png`: 数据总览
   - `missing_data.png`: 缺失值分布
   - `error_by_horizon.png`: 误差随步长变化

3. **报告**
   - `outputs/report/项目评估报告.md`: Markdown格式报告
   - `outputs/report/项目评估报告.docx`: Word格式报告

## 注意事项

### 数据泄漏防范

本项目在多个层面防止数据泄漏：

1. **特征工程**：所有滞后特征和滚动统计只使用历史数据
2. **交叉验证**：严格的时间顺序划分
3. **标准化**：在训练集上拟合，在测试集上变换

### 性能考虑

- 深度学习模型（LSTM、Transformer）训练时间较长
- 可以在 `config.yaml` 中禁用某些模型以加快运行
- 树模型支持并行计算（`n_jobs=-1`）

### 字体问题

如果图表中文显示为方块：

1. 确保系统安装了中文字体
2. 修改 `config.yaml` 中的 `font_priority` 列表
3. 常见字体：
   - Windows: SimHei, Microsoft YaHei
   - macOS: STHeiti, PingFang SC
   - Linux: WenQuanYi Micro Hei, Noto Sans CJK SC

## 参考文献

- [Forecasting: Principles and Practice (Hyndman & Athanasopoulos)](https://otexts.com/fpp3/)
- [TimeSeriesSplit - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Why MAPE is problematic](https://www.statworx.com/en/content-hub/blog/what-the-mape-is-falsely-blamed-for-its-true-weaknesses-and-better-alternatives)
- [python-docx Documentation](https://python-docx.readthedocs.io/)

## 常见问题

**Q: 如何添加新的数据文件？**

A: 将文件放入 `data/raw/` 目录，确保包含时间、P、Q列。

**Q: 如何只运行部分模型？**

A: 在 `config.yaml` 中设置不需要的模型的 `enabled: false`。

**Q: 如何修改预测步长？**

A: 在 `config.yaml` 的 `evaluation.horizons` 中修改列表。

**Q: GPU加速支持？**

A: 深度学习模型（LSTM、Transformer）会自动检测并使用可用的CUDA GPU。

## 许可证

MIT License

## 联系方式

如有问题或建议，请创建Issue。
