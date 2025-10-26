#!/usr/bin/env python
"""
测试图表生成和中文显示
"""
import sys
from pathlib import Path
from src.config import Config
from src.data_io import load_data, generate_diagnostic_plots

# 加载配置
config = Config('config_fast_test.yaml')

# 加载数据
data_dir = Path('data/raw')
data_files = list(data_dir.glob('*.csv'))

if data_files:
    print(f"使用数据文件: {data_files[0]}")
    
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
    
    print(f"\n处理前数据形状: {df_before.shape}")
    print(f"处理前缺失值 - P: {df_before['P'].isna().sum()}, Q: {df_before['Q'].isna().sum()}")
    print(f"\n处理后数据形状: {df.shape}")
    print(f"处理后缺失值 - P: {df['P'].isna().sum()}, Q: {df['Q'].isna().sum()}")
    
    # 生成图表
    output_dir = 'outputs/test_plots'
    print(f"\n生成诊断图表到: {output_dir}")
    generate_diagnostic_plots(df, df_before=df_before, output_dir=output_dir)
    
    print("\n✓ 测试完成！请查看生成的图表:")
    print(f"  - {output_dir}/data_overview.png")
    print(f"  - {output_dir}/missing_data.png")
else:
    print("错误: 未找到数据文件")
    sys.exit(1)
