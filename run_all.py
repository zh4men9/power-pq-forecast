"""
Main execution script for power quality forecasting project
Runs the complete pipeline: data loading -> training -> evaluation -> reporting
"""
import argparse
from pathlib import Path
import sys

from src.config import load_config
from src.data_io import load_data, generate_diagnostic_plots
from src.train_eval import run_evaluation
from src.plots import plot_error_by_horizon, configure_chinese_fonts
from src.report_md import generate_markdown_report
from src.report_docx import generate_word_report
import pandas as pd


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='电力质量预测项目 - 一键运行脚本')
    parser.add_argument('--config', type=str, default='config_exog.yaml',
                       help='配置文件路径 (默认: config_exog.yaml)')
    args = parser.parse_args()
    
    # Load configuration
    print("="*60)
    print("步骤 1/6: 加载配置文件")
    print("="*60)
    
    config = load_config(args.config)
    print(f"配置文件加载成功: {args.config}")
    print()
    
    # Load data
    print("="*60)
    print("步骤 2/6: 加载数据")
    print("="*60)
    
    data_path = config.get('data', 'data_path', default='data/raw')
    file_pattern = config.get('data', 'file_pattern', default='*.xlsx')
    
    # Find data file
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_path}")
        print("请将数据文件放置在 data/raw/ 目录中")
        sys.exit(1)
    
    data_files = list(data_dir.glob(file_pattern))
    if not data_files:
        print(f"错误: 未找到匹配的数据文件: {data_path}/{file_pattern}")
        print("请将数据文件放置在 data/raw/ 目录中")
        sys.exit(1)
    
    data_file = data_files[0]
    print(f"使用数据文件: {data_file}")
    
    # Load data with config parameters
    df = load_data(
        file_path=str(data_file),
        time_col=config.get('data', 'time_col'),
        p_col=config.get('data', 'p_col'),
        q_col=config.get('data', 'q_col'),
        exog_cols=config.get('features', 'exog_cols', default=[]),
        freq=config.get('data', 'freq'),
        tz=config.get('data', 'tz'),
        interp_limit=config.get('data', 'interp_limit', default=3)
    )
    
    # Generate diagnostic plots
    generate_diagnostic_plots(df, output_dir='outputs/figures')
    print()
    
    # Run evaluation
    print("="*60)
    print("步骤 3/6: 模型训练与评估")
    print("="*60)
    
    results_df = run_evaluation(config, df)
    print()
    
    # Generate plots
    print("="*60)
    print("步骤 4/6: 生成图表")
    print("="*60)
    
    configure_chinese_fonts(config.get('plotting', 'font_priority'))
    
    # Plot error by horizon
    plot_error_by_horizon(
        results_df,
        metric_name='RMSE',
        output_path='outputs/figures/error_by_horizon.png',
        dpi=config.get('plotting', 'fig_dpi', default=150)
    )
    
    print()
    
    # Generate Markdown report
    print("="*60)
    print("步骤 5/6: 生成Markdown报告")
    print("="*60)
    
    md_report_path = generate_markdown_report(
        results_df,
        config_path=args.config,
        output_path='outputs/report/项目评估报告.md'
    )
    print()
    
    # Generate Word report
    print("="*60)
    print("步骤 6/6: 生成Word报告")
    print("="*60)
    
    word_report_path = generate_word_report(
        results_df,
        config_path=args.config,
        figures_dir='outputs/figures',
        output_path='outputs/report/项目评估报告.docx'
    )
    print()
    
    # Summary
    print("="*60)
    print("运行完成!")
    print("="*60)
    print(f"指标表路径: outputs/metrics/cv_metrics.csv")
    print(f"图表目录: outputs/figures/")
    print(f"Markdown报告: {md_report_path}")
    print(f"Word报告: {word_report_path}")
    print("="*60)


if __name__ == '__main__':
    main()
