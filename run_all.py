"""
Main execution script for power quality forecasting project
Runs the complete pipeline: data loading -> training -> evaluation -> reporting
"""
import argparse
from pathlib import Path
import sys
from datetime import datetime
import shutil
import logging

from src.config import load_config
from src.data_io import load_data, generate_diagnostic_plots
from src.train_eval import run_evaluation
from src.plots import plot_error_by_horizon, configure_chinese_fonts
from src.report_md import generate_markdown_report
from src.report_docx import generate_word_report
import pandas as pd


def setup_logging(output_dir: Path):
    """
    设置日志系统：同时输出到控制台和文件
    
    Args:
        output_dir: 输出目录路径
    """
    # 创建日志目录
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志文件路径
    log_file = log_dir / 'training.log'
    
    # 配置日志格式
    log_format = '%(asctime)s | %(levelname)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logging.info(f"日志系统已初始化")
    logging.info(f"日志文件: {log_file}")
    
    return str(log_file)


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='电力质量预测项目 - 一键运行脚本')
    parser.add_argument('--config', type=str, default='config_exog.yaml',
                       help='配置文件路径 (默认: config_exog.yaml)')
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
    output_base = Path('outputs')
    output_dir = output_base / f'output-{timestamp}'
    
    # Setup logging (before any other output)
    log_file = setup_logging(output_dir)
    
    # Load configuration
    logging.info("="*60)
    logging.info("步骤 1/6: 加载配置文件")
    logging.info("="*60)
    
    config = load_config(args.config)
    logging.info(f"配置文件加载成功: {args.config}")
    logging.info(f"输出目录: {output_dir}")
    logging.info("")
    
    # Load data
    logging.info("="*60)
    logging.info("步骤 2/6: 加载数据")
    logging.info("="*60)
    
    data_path = config.get('data', 'data_path', default='data/raw')
    file_pattern = config.get('data', 'file_pattern', default='*.xlsx')
    
    # Find data file
    data_dir = Path(data_path)
    if not data_dir.exists():
        logging.error(f"错误: 数据目录不存在: {data_path}")
        logging.error("请将数据文件放置在 data/raw/ 目录中")
        sys.exit(1)
    
    data_files = list(data_dir.glob(file_pattern))
    if not data_files:
        logging.error(f"错误: 未找到匹配的数据文件: {data_path}/{file_pattern}")
        logging.error("请将数据文件放置在 data/raw/ 目录中")
        sys.exit(1)
    
    data_file = data_files[0]
    logging.info(f"使用数据文件: {data_file}")
    
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
    figures_dir = output_dir / 'figures'
    generate_diagnostic_plots(df, output_dir=str(figures_dir))
    print()
    
    # Run evaluation
    print("="*60)
    print("步骤 3/6: 模型训练与评估")
    print("="*60)
    
    metrics_dir = output_dir / 'metrics'
    results_df = run_evaluation(config, df, metrics_dir=str(metrics_dir))
    logging.info("")
    
    # Generate plots
    logging.info("="*60)
    logging.info("步骤 4/6: 生成图表")
    logging.info("="*60)
    
    configure_chinese_fonts(config.get('plotting', 'font_priority'))
    
    # Plot error by horizon
    plot_error_by_horizon(
        results_df,
        metric_name='RMSE',
        output_path=str(figures_dir / 'error_by_horizon.png'),
        dpi=config.get('plotting', 'fig_dpi', default=150)
    )
    
    logging.info("")
    
    # Generate Markdown report
    logging.info("="*60)
    logging.info("步骤 5/6: 生成Markdown报告")
    logging.info("="*60)
    
    report_dir = output_dir / 'report'
    md_report_path = generate_markdown_report(
        results_df,
        config_path=args.config,
        figures_dir=str(figures_dir),
        output_path=str(report_dir / '项目评估报告.md')
    )
    logging.info("")
    
    # Generate Word report
    logging.info("="*60)
    logging.info("步骤 6/6: 生成Word报告")
    logging.info("="*60)
    
    word_report_path = generate_word_report(
        results_df,
        config_path=args.config,
        figures_dir=str(figures_dir),
        output_path=str(report_dir / '项目评估报告.docx')
    )
    logging.info("")
    
    # Copy to latest outputs folder for convenience
    latest_dir = output_base / 'latest'
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(output_dir, latest_dir)
    
    # Summary
    logging.info("="*60)
    logging.info("运行完成!")
    logging.info("="*60)
    logging.info(f"本次运行输出目录: {output_dir}")
    logging.info(f"最新结果链接: {latest_dir}")
    logging.info(f"  - 日志文件: {log_file}")
    logging.info(f"  - 指标表: {metrics_dir / 'cv_metrics.csv'}")
    logging.info(f"  - 图表目录: {figures_dir}")
    logging.info(f"  - Markdown报告: {md_report_path}")
    logging.info(f"  - Word报告: {word_report_path}")
    logging.info("="*60)


if __name__ == '__main__':
    main()
