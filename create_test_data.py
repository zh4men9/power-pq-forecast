#!/usr/bin/env python
"""
创建带缺失值的测试数据，验证处理前后对比功能
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 读取原始数据
df = pd.read_csv('data/raw/电气多特征.csv', parse_dates=['日期'])
df = df.set_index('日期')

# 创建一个带缺失值的副本用于测试
df_test = df.copy()

# 在P和Q中人为创建一些缺失值
# 选择P接近280的索引
p_near_280 = df_test[np.abs(df_test['P'] - 280) < 5].index

# 随机选择10个时间点设为NaN
np.random.seed(42)
missing_indices = np.random.choice(df_test.index, size=10, replace=False)
df_test.loc[missing_indices, 'P'] = np.nan
df_test.loc[missing_indices, 'Q'] = np.nan

# 保存测试数据
output_dir = Path('data/raw')
output_dir.mkdir(parents=True, exist_ok=True)
df_test.reset_index().to_csv(output_dir / '测试数据_带缺失值.csv', index=False)

print(f"✓ 测试数据已创建: data/raw/测试数据_带缺失值.csv")
print(f"  - 总数据点: {len(df_test)}")
print(f"  - P列缺失值: {df_test['P'].isna().sum()}")
print(f"  - Q列缺失值: {df_test['Q'].isna().sum()}")
print(f"\n缺失值位置:")
for idx in missing_indices[:5]:
    print(f"  - {idx}")
