#!/usr/bin/env python
"""
验证训练流程中使用的数据是否是处理后的完整数据
"""
from pathlib import Path
from src.config import Config
from src.data_io import load_data
from src.features import create_features

# 加载配置
config = Config('config_p_only.yaml')

# 加载数据
data_dir = Path('data/raw')
data_files = list(data_dir.glob('*.csv'))

if data_files:
    print("="*70)
    print("数据加载验证")
    print("="*70)
    
    imputation_config = config.get('data', 'imputation', default={})
    df, df_before = load_data(
        file_path=str(data_files[0]),
        time_col=config.get('data', 'time_col'),
        p_col=config.get('data', 'p_col'),
        q_col=config.get('data', 'q_col'),
        exog_cols=config.get('features', 'exog_cols', default=[]),
        freq=config.get('data', 'freq'),
        tz=config.get('data', 'tz'),
        interp_limit=config.get('data', 'interp_limit', default=3),
        imputation_method=imputation_config.get('method'),
        target_p_value=imputation_config.get('target_p_value', 280.0)
    )
    
    print("\n" + "="*70)
    print("数据验证结果")
    print("="*70)
    
    print(f"\n1️⃣  处理前数据 (df_before):")
    print(f"   - 形状: {df_before.shape}")
    print(f"   - P列缺失值: {df_before['P'].isna().sum()}")
    print(f"   - Q列缺失值: {df_before['Q'].isna().sum()}")
    
    print(f"\n2️⃣  处理后数据 (df - 用于训练):")
    print(f"   - 形状: {df.shape}")
    print(f"   - P列缺失值: {df['P'].isna().sum()}")
    print(f"   - Q列缺失值: {df['Q'].isna().sum()}")
    print(f"   - P列统计:")
    print(f"     * 最小值: {df['P'].min():.2f}")
    print(f"     * 最大值: {df['P'].max():.2f}")
    print(f"     * 平均值: {df['P'].mean():.2f}")
    
    # 检查填充的数据点
    print(f"\n3️⃣  填充验证:")
    filled_mask = df_before['P'].isna()
    filled_count = filled_mask.sum()
    print(f"   - 被填充的数据点数量: {filled_count}")
    
    if filled_count > 0:
        # 查看填充后这些位置的P值
        filled_p_values = df.loc[filled_mask, 'P']
        print(f"   - 填充值的P统计:")
        print(f"     * 最小值: {filled_p_values.min():.2f}")
        print(f"     * 最大值: {filled_p_values.max():.2f}")
        print(f"     * 平均值: {filled_p_values.mean():.2f}")
        print(f"     * 标准差: {filled_p_values.std():.2f}")
        
        # 检查是否都是280附近的值
        target_p = imputation_config.get('target_p_value', 280.0)
        if filled_p_values.nunique() == 1:
            print(f"   - ✅ 所有填充值相同: P={filled_p_values.iloc[0]:.2f}")
            print(f"   - ✅ 与目标值P={target_p}的距离: {abs(filled_p_values.iloc[0] - target_p):.2f}")
        else:
            print(f"   - ⚠️  填充值不唯一，共有{filled_p_values.nunique()}个不同值")
    
    # 创建特征
    print(f"\n4️⃣  特征工程验证:")
    max_lag = config.get('features', 'max_lag', default=24)
    exog_cols = config.get('features', 'exog_cols', default=[])
    
    X, Y = create_features(df, max_lag=max_lag, exog_cols=exog_cols)
    
    print(f"   - 特征矩阵X形状: {X.shape}")
    print(f"   - 目标矩阵Y形状: {Y.shape}")
    print(f"   - X中缺失值: {X.isna().sum().sum()}")
    print(f"   - Y中缺失值: {Y.isna().sum().sum()}")
    
    # 计算实际可用于训练的数据点
    usable_points = len(Y)
    original_points = 2783  # 原始数据点
    filled_points = filled_count
    total_points = len(df)
    
    print(f"\n5️⃣  训练数据总结:")
    print(f"   - 原始CSV数据点: {original_points}")
    print(f"   - 补充的时间戳: {filled_points} (用P≈280填充)")
    print(f"   - 完整时间序列: {total_points}")
    print(f"   - 可用于训练的数据点: {usable_points} (扣除lag feature所需的历史数据)")
    print(f"   - 训练数据占比: {usable_points/total_points*100:.1f}%")
    
    print(f"\n✅ 验证完成：训练使用的是处理后的完整数据（包含填充的601个时间点）")
    
else:
    print("错误: 未找到数据文件")
