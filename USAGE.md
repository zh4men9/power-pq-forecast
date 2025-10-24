# 使用示例 (Usage Example)

本文档提供了电力质量预测系统的详细使用示例。

## 快速开始

### 1. 生成示例数据

如果你没有自己的数据，可以使用提供的脚本生成示例数据：

```bash
python generate_sample_data.py
```

这将在 `data/raw/` 目录下生成一个名为 `sample_data.xlsx` 的文件，包含30天的小时级电力数据。

### 2. 运行完整流程

使用默认配置运行：

```bash
python run_all.py --config config.yaml
```

使用测试配置（更快的运行速度）：

```bash
python run_all.py --config config_test.yaml
```

### 3. 查看结果

运行完成后，查看以下文件：

- **评估指标**: `outputs/metrics/cv_metrics.csv`
- **图表**: `outputs/figures/`
- **Markdown报告**: `outputs/report/项目评估报告.md`
- **Word报告**: `outputs/report/项目评估报告.docx`

## 配置说明

### 基本配置项

编辑 `config.yaml` 文件来调整项目参数。

#### 数据配置

```yaml
data:
  data_path: "data/raw"        # 数据文件目录
  file_pattern: "*.xlsx"       # 文件匹配模式
  freq: "h"                    # 时间频率（h=小时，D=日）
  interp_limit: 3              # 短缺口插值最大步数
```

#### 评估配置

```yaml
evaluation:
  horizons:                    # 预测步长
    - 1
    - 12
    - 24
  test_window: 100            # 测试窗口大小
  n_splits: 3                 # 交叉验证折数
```

#### 模型选择

启用或禁用特定模型：

```yaml
models:
  naive:
    enabled: true             # 启用朴素基线
  
  rf:
    enabled: true             # 启用随机森林
    n_estimators: 100         # 树的数量
    max_depth: null           # 最大深度（null=不限制）
  
  lstm:
    enabled: false            # 禁用LSTM（加快速度）
```

## 高级用法

### 自定义数据格式

如果你的数据列名不是标准格式，可以在配置中指定：

```yaml
data:
  time_col: "timestamp"       # 指定时间列名
  p_col: "active_power"       # 指定有功功率列名
  q_col: "reactive_power"     # 指定无功功率列名
```

### 调整特征工程

```yaml
features:
  max_lag: 24                 # 最大滞后步数
  roll_windows:               # 滚动窗口大小
    - 6
    - 12
    - 24
  sequence_length: 24         # 深度学习序列长度
```

### 中文字体配置

如果图表中文显示为方块，调整字体优先列表：

```yaml
plotting:
  font_priority:
    - SimHei              # Windows
    - Microsoft YaHei     # Windows
    - STHeiti             # macOS
    - PingFang SC         # macOS
    - WenQuanYi Micro Hei # Linux
    - Noto Sans CJK SC    # Linux/跨平台
```

## 性能优化

### 加快运行速度

1. **减少交叉验证折数**:
```yaml
evaluation:
  n_splits: 2  # 从3改为2
```

2. **减少预测步长**:
```yaml
evaluation:
  horizons:
    - 1
    - 12  # 只测试2个步长
```

3. **禁用深度学习模型**:
```yaml
models:
  lstm:
    enabled: false
  transformer:
    enabled: false
```

4. **减少树的数量**:
```yaml
models:
  rf:
    n_estimators: 50  # 从100改为50
  xgb:
    n_estimators: 50
```

### GPU加速

深度学习模型（LSTM、Transformer）会自动检测并使用可用的CUDA GPU。

检查GPU是否可用：

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 常见问题解答

### Q1: 数据文件格式要求？

A: 支持 Excel (.xlsx, .xls) 和 CSV (.csv) 格式。必须包含：
- 时间列（日期时间格式）
- 有功功率列（数值）
- 无功功率列（数值）

### Q2: 如何处理缺失值？

A: 系统会自动进行短缺口插值（默认最多3个连续缺失值）。长缺口不会自动填充，以避免引入虚假模式。

### Q3: 为什么某些模型比基线还差？

A: 这可能表明：
1. 数据量不足以训练复杂模型
2. 模型超参数需要调整
3. 数据中的模式过于简单，复杂模型反而过拟合

### Q4: 如何选择最佳模型？

A: 综合考虑：
1. **精度**: 在所有步长和指标上的平均性能
2. **稳定性**: 不同折之间的方差
3. **计算成本**: 训练和预测时间
4. **可解释性**: 树模型提供特征重要性

### Q5: 滚动起点验证是什么？

A: 这是时间序列的标准验证方法：
- 训练集总是在测试集之前
- 训练集随每折递增
- 测试集固定大小，向前滚动
- 严格防止使用未来信息

参考：[Forecasting: Principles and Practice](https://otexts.com/fpp3/tscv.html)

## 输出文件说明

### cv_metrics.csv

包含所有模型的详细评估结果：

| 列名 | 说明 |
|------|------|
| model | 模型名称 |
| horizon | 预测步长 |
| fold | 交叉验证折编号 |
| target | 目标变量（P或Q） |
| RMSE | 均方根误差 |
| MAE | 平均绝对误差 |
| SMAPE | 对称平均绝对百分比误差 |
| WAPE | 加权绝对百分比误差 |

### 图表文件

- `data_overview.png`: 原始数据的P和Q曲线
- `missing_data.png`: 缺失值分布热图
- `error_by_horizon.png`: 不同模型在各步长的误差对比

### 报告文件

- `项目评估报告.md`: Markdown格式，便于在线查看
- `项目评估报告.docx`: Word格式，可直接打印或进一步编辑

## 最佳实践

1. **先用小数据集测试**: 使用 `config_test.yaml` 快速验证流程
2. **检查数据质量**: 查看 `data_overview.png` 和 `missing_data.png`
3. **从基线开始**: 确保复杂模型能超越基线
4. **逐步增加复杂度**: 先树模型，再深度学习
5. **记录实验结果**: 保存不同配置的 `cv_metrics.csv`
6. **定期重新训练**: 使用新数据更新模型

## 扩展开发

### 添加新模型

1. 在 `src/models/` 下创建新文件
2. 实现 `fit()` 和 `predict()` 方法
3. 在 `src/train_eval.py` 中集成
4. 在 `config.yaml` 中添加配置

### 添加新指标

1. 在 `src/metrics.py` 中添加函数
2. 在 `config.yaml` 的 `metrics` 列表中添加名称
3. 更新报告模板以显示新指标

### 自定义图表

1. 在 `src/plots.py` 中添加绘图函数
2. 在 `run_all.py` 中调用
3. 在报告中引用新图表

## 技术支持

遇到问题？检查：

1. Python版本是否 >= 3.8
2. 所有依赖是否正确安装
3. 数据文件是否在正确位置
4. 配置文件语法是否正确（YAML格式）

查看详细日志：

```bash
python run_all.py --config config.yaml 2>&1 | tee run.log
```

## 参考资源

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [python-docx Documentation](https://python-docx.readthedocs.io/)
