# 更新说明 (Update Notes)

## 版本更新 - 外生变量支持 (Exogenous Variables Support)

### 更新日期：2025-10-24

---

## 🎯 主要更新内容

### 1. 新增外生变量（协变量）支持

**功能描述**：现在可以引入额外的测量数据来提升预测精度，例如：
- 电压（U）、电流（I）
- 转子电压（Vf）、转子电流（If）
- 蒸汽压力、温度、给水流量等

**使用方法**：在配置文件中指定外生变量列名

```yaml
features:
  exog_cols:
    - '定子电流'
    - '定子电压'
    - '转子电压'
    - '转子电流'
```

**关键原则**：
- ✅ 只使用**当前和过去**的外生变量值
- ✅ 自动生成滞后特征和滚动统计
- ✅ 防止信息泄漏
- ✅ 符合动态回归（Dynamic Regression）标准

### 2. 跨平台中文字体支持优化

**优化内容**：
- 更新字体优先列表，支持 Windows、macOS、Linux
- 自动抑制无关字体警告
- 确保图表在各平台正确显示中文

**字体配置**（已在 config.yaml 中更新）：
```yaml
plotting:
  font_priority:
    - SimHei            # Windows 黑体
    - Microsoft YaHei   # Windows 微软雅黑
    - STHeiti           # macOS 华文黑体
    - PingFang SC       # macOS 苹方-简
    - WenQuanYi Micro Hei  # Linux 文泉驿微米黑
    - Noto Sans CJK SC  # Linux/跨平台 思源黑体
```

### 3. 专业版 README 文档

**增强内容**：
- 扩充从 211 行到 907 行（4.3倍）
- 新增系统架构图（ASCII 图示）
- 新增滚动起点交叉验证可视化
- 详细的使用场景说明（A: 仅P/Q，B: 含外生变量）
- 最佳实践指南
- 常见问题解答（FAQ）
- 12 篇参考文献引用

---

## 📝 使用指南

### 场景 A：仅使用 P 和 Q（原有功能）

**配置**：
```yaml
features:
  exog_cols: []  # 留空或省略此行
```

**运行**：
```bash
python run_all.py --config config.yaml
```

**特点**：
- 简单快速
- 适合快速建立基线
- 特征数量较少（约 46 个）

### 场景 B：使用外生变量（新功能）

**配置**：
```yaml
features:
  exog_cols:
    - '定子电流'
    - '定子电压'
    - '转子电压'
    - '转子电流'
```

**运行**：
```bash
python run_all.py --config config_exog.yaml
```

**特点**：
- 利用更多信息
- 可能提升预测精度
- 特征数量增加（约 126 个）
- 自动生成特征重要性图表

---

## 🧪 测试验证

### 测试结果

**测试 1：仅 P/Q**
- 数据：3384 行 × 2 列
- 特征：2746 × 46
- RMSE: P=44.12, Q=28.93

**测试 2：含外生变量**
- 数据：3384 行 × 6 列
- 特征：2746 × 126（增加 80 个）
- RMSE: P=44.97, Q=29.44
- Top 3 重要特征：
  1. P_lag_1 (43.3%)
  2. Q_lag_1 (39.1%)
  3. 定子电压_lag_1 (4.4%)

### 特征重要性

通过随机森林模型可以查看哪些特征最重要：
- 滞后 1 步的 P 和 Q 是最重要的特征
- 外生变量的滞后特征也有一定贡献
- 系统会自动生成特征重要性图表

---

## 🔄 向后兼容性

**100% 兼容**：
- ✅ 所有原有功能保持不变
- ✅ 不使用 exog_cols 时行为完全相同
- ✅ 配置文件向后兼容
- ✅ 无需修改现有代码

---

## 📦 新增文件

1. **config_exog.yaml** - 使用外生变量的示例配置
2. **test_exog_vars.py** - 功能测试脚本
3. **UPDATE_NOTES.md** - 本文件

---

## 🚀 快速开始

### 1. 更新配置文件

**选项 1：不使用外生变量**（原有方式）
```yaml
# config.yaml
features:
  exog_cols: []
```

**选项 2：使用外生变量**（新功能）
```yaml
# config.yaml 或 config_exog.yaml
features:
  exog_cols: ['定子电流', '定子电压', '转子电压', '转子电流']
```

### 2. 运行预测

```bash
# 使用默认配置（P/Q only）
python run_all.py --config config.yaml

# 使用外生变量配置
python run_all.py --config config_exog.yaml

# 运行测试
python test_exog_vars.py
```

### 3. 查看结果

- **评估指标**：`outputs/metrics/cv_metrics.csv`
- **图表**：`outputs/figures/`
  - `data_overview.png` - 数据总览
  - `error_by_horizon.png` - 误差对比
  - `feature_importance.png` - 特征重要性（树模型）
- **报告**：`outputs/report/`
  - `项目评估报告.md` - Markdown 格式
  - `项目评估报告.docx` - Word 格式

---

## 🔧 技术细节

### 特征工程

对于每个外生变量，系统会自动生成：
1. **滞后特征**：变量在 t-1, t-2, ..., t-lag 时刻的值
2. **滚动统计**：
   - rolling_mean（滚动均值）
   - rolling_std（滚动标准差）
   - rolling_min（滚动最小值）
   - rolling_max（滚动最大值）

### 数据泄漏防范

系统确保不会使用未来信息：
```python
# ✅ 正确：只使用过去值
X[t] = {
    P[t-1], P[t-2], ...,           # P 的滞后
    U[t-1], U[t-2], ...,           # 外生变量的滞后
    rolling_mean(P[t-lag:t]),      # 滚动统计
    ...
}
Y[t+h] = P[t+h]  # 预测目标

# ❌ 错误：使用当前或未来值
X[t] = { P[t], U[t+1], ... }      # 信息泄漏！
```

### 模型适配

所有模型自动适配外生变量：
- **树模型（RF/XGBoost）**：特征数量自动增加
- **深度学习（LSTM/Transformer）**：输入维度自动调整
- **基线模型**：不受影响（只使用 P/Q）

---

## ❓ 常见问题

### Q1: 外生变量必须是未来已知的吗？

**A**: 不需要。我们只使用外生变量的**历史值**（t-1, t-2, ...），不需要知道未来的外生变量值。这符合动态回归（Dynamic Regression）的标准做法。

参考：[Hyndman & Athanasopoulos - Dynamic regression](https://otexts.com/fpp2/dynamic.html)

### Q2: 添加外生变量一定能提升精度吗？

**A**: 不一定。效果取决于：
- 外生变量与目标变量的相关性
- 数据质量
- 数据量是否充足

建议：
1. 先用 P/Q 建立基线
2. 逐步添加外生变量
3. 通过特征重要性图判断哪些变量有用

### Q3: 如何判断哪些外生变量有用？

**A**: 使用特征重要性分析：
1. 运行带外生变量的预测
2. 查看 `outputs/figures/feature_importance.png`
3. 移除重要性低的变量

### Q4: 数据频率改变时如何调整？

**A**: 修改配置：
```yaml
data:
  freq: "1min"  # 1分钟采样

evaluation:
  test_window: 1440  # 1天 = 1440分钟
  horizons: [1, 12, 60]

features:
  season_length: 1440
```

### Q5: 图表中文显示为方块怎么办？

**A**: 
1. 确保系统安装了中文字体
2. 修改 `config.yaml` 中的 `font_priority`
3. 将系统已安装的字体放在列表前面

---

## 📚 参考文献

1. **Hyndman & Athanasopoulos** - Forecasting: Principles and Practice
   - [滚动起点验证](https://otexts.com/fpp3/tscv.html)
   - [动态回归](https://otexts.com/fpp2/dynamic.html)

2. **scikit-learn** - [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

3. **Rob J Hyndman** - [Why MAPE is problematic](https://robjhyndman.com/hyndsight/forecastmse/)

---

## 📞 支持与反馈

如有问题或建议：
- 查看 [README.md](README.md) 获取详细文档
- 查看 [USAGE.md](USAGE.md) 获取使用指南
- 提交 [GitHub Issue](https://github.com/zh4men9/power-pq-forecast/issues)

---

## ✅ 更新总结

| 项目 | 原有 | 更新后 |
|------|------|--------|
| 外生变量支持 | ❌ | ✅ |
| 特征重要性可视化 | ✅ | ✅ |
| 跨平台字体支持 | 部分 | ✅ 完整 |
| README 文档 | 211 行 | 907 行 |
| 示例配置 | 2 个 | 3 个 |
| 测试脚本 | 0 个 | 1 个 |
| 向后兼容性 | - | ✅ 100% |

---

**祝使用愉快！如有任何问题，欢迎反馈。**
