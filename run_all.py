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
from src.plots import plot_error_by_horizon, plot_all_metrics_by_horizon, configure_chinese_fonts

from src.report_docx import generate_word_report
from src.model_manager import save_model, get_best_model_info, make_future_forecast
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


def run_single_strategy(config, data_file, output_dir, config_backup_path, imputation_strategy=None):
    """
    运行单个填充策略的完整流程
    
    Args:
        config: 配置对象
        data_file: 数据文件路径
        output_dir: 输出目录
        config_backup_path: 备份的配置文件路径
        imputation_strategy: 填充策略名称 (None表示使用配置中的method)
    
    Returns:
        Tuple of (results_df, forecast_df, figures_dir)
    """
    # Load data with specified imputation strategy
    logging.info("="*60)
    logging.info(f"步骤 2/7: 加载数据 (填充策略: {imputation_strategy or 'default'})")
    logging.info("="*60)
    
    imputation_config = config.get('data', 'imputation', default={})
    
    # Use specified strategy or fall back to config
    method = imputation_strategy if imputation_strategy else imputation_config.get('method')
    
    df, df_before = load_data(
        file_path=str(data_file),
        time_col=config.get('data', 'time_col'),
        p_col=config.get('data', 'p_col'),
        q_col=config.get('data', 'q_col'),
        exog_cols=config.get('features', 'exog_cols', default=[]),
        freq=config.get('data', 'freq'),
        tz=config.get('data', 'tz'),
        interp_limit=config.get('data', 'interp_limit', default=3),
        imputation_method=method,
        target_p_value=imputation_config.get('target_p_value', 280.0),
        day_copy_days_back=imputation_config.get('day_copy_days_back', 7),
        seasonal_period=imputation_config.get('seasonal_period', 24)
    )
    
    # Generate diagnostic plots with before/after comparison
    strategy_suffix = f"_{imputation_strategy}" if imputation_strategy else ""
    figures_dir = output_dir / f'figures{strategy_suffix}'
    generate_diagnostic_plots(df, df_before=df_before, output_dir=str(figures_dir))
    logging.info("")
    
    # Run evaluation
    logging.info("="*60)
    logging.info("步骤 3/7: 模型训练与评估")
    logging.info("="*60)
    
    metrics_dir = output_dir / f'metrics{strategy_suffix}'
    results_df, trained_models = run_evaluation(config, df, metrics_dir=str(metrics_dir))
    logging.info("")
    
    # Save trained models
    logging.info("="*60)
    logging.info("步骤 4/7: 保存训练好的模型")
    logging.info("="*60)
    
    models_dir = output_dir / f'models{strategy_suffix}'
    for model_name, model in trained_models.items():
        # Get model performance from results
        model_results = results_df[results_df['model'] == model_name]
        if len(model_results) > 0:
            metadata = {
                'avg_rmse': model_results['RMSE'].mean(),
                'avg_mae': model_results['MAE'].mean(),
                'model_name': model_name
            }
        else:
            metadata = {'model_name': model_name}
        
        save_model(model, model_name, str(models_dir), metadata=metadata)
    
    logging.info(f"✓ 所有模型已保存至: {models_dir}")
    logging.info("")
    
    # Generate future forecast
    forecast_df = None
    forecast_config = config.get('forecast', default={})
    if forecast_config.get('enabled', False):
        logging.info("="*60)
        logging.info("步骤 5/7: 生成未来预测")
        logging.info("="*60)
        
        # Get best model
        target_cols = []
        if config.get('target', 'predict_p', default=True):
            target_cols.append('P')
        if config.get('target', 'predict_q', default=True):
            target_cols.append('Q')
        
        target = target_cols[0]  # 使用第一个目标变量
        best_model_info = get_best_model_info(results_df, target=target, metric='RMSE')
        best_model_name = forecast_config.get('best_model', best_model_info['overall_best'])
        
        logging.info(f"最优模型（基于RMSE）: {best_model_info['overall_best']}")
        logging.info(f"使用模型进行预测: {best_model_name}")
        
        if best_model_name in trained_models:
            forecast_df = make_future_forecast(
                df=df,
                model=trained_models[best_model_name],
                model_name=best_model_name,
                start_date=forecast_config.get('start_date', '2025-10-20'),
                end_date=forecast_config.get('end_date', '2025-10-22'),
                config=config
            )
            
            # Save forecast results
            if forecast_df is not None and len(forecast_df) > 0:
                forecast_path = output_dir / f'forecast_results{strategy_suffix}.csv'
                forecast_df.to_csv(forecast_path, index=False, encoding='utf-8-sig')
                logging.info(f"✓ 预测结果已保存: {forecast_path}")
        else:
            logging.warning(f"⚠️  模型 {best_model_name} 不可用，跳过预测")
        
        logging.info("")
    
    # Generate plots
    logging.info("="*60)
    logging.info("步骤 6/7: 生成图表")
    logging.info("="*60)
    
    configure_chinese_fonts(config.get('plotting', 'font_priority'))
    
    # Plot error by horizon for RMSE
    plot_error_by_horizon(
        results_df,
        metric_name='RMSE',
        output_path=str(figures_dir / 'error_by_horizon_rmse.png'),
        dpi=config.get('plotting', 'fig_dpi', default=150)
    )
    
    # Plot all metrics by horizon
    available_metrics = [m for m in config.get('evaluation', 'metrics', default=[]) 
                        if m in results_df.columns]
    if available_metrics:
        plot_all_metrics_by_horizon(
            results_df,
            metrics=available_metrics,
            output_path=str(figures_dir / 'all_metrics_by_horizon.png'),
            dpi=config.get('plotting', 'fig_dpi', default=150)
        )
    
    logging.info("")
    
    return results_df, forecast_df, figures_dir


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='电力质量预测项目 - 一键运行脚本')
    parser.add_argument('--config', type=str, default='config_p_only.yaml',
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
    logging.info("步骤 1/7: 加载配置文件")
    logging.info("="*60)
    
    config = load_config(args.config)
    logging.info(f"配置文件加载成功: {args.config}")
    logging.info(f"输出目录: {output_dir}")
    
    # Copy configuration file to output directory
    config_backup_path = output_dir / 'config_used.yaml'
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, config_backup_path)
    logging.info(f"✓ 配置文件已备份至: {config_backup_path}")
    logging.info("")
    
    # Find data file
    data_path = config.get('data', 'data_path', default='data/raw')
    file_pattern = config.get('data', 'file_pattern', default='*.xlsx')
    
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
    logging.info("")
    
    # Check if using multiple imputation strategies
    imputation_config = config.get('data', 'imputation', default={})
    strategies = imputation_config.get('strategies', [])
    
    if strategies and len(strategies) > 0:
        # Multi-strategy mode
        logging.info("="*60)
        logging.info(f"🔄 多策略模式: 将依次运行 {len(strategies)} 个填充策略")
        logging.info(f"策略列表: {', '.join(strategies)}")
        logging.info("="*60)
        logging.info("")
        
        report_dir = output_dir / 'report'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, strategy in enumerate(strategies, 1):
            logging.info("")
            logging.info("█"*60)
            logging.info(f"运行策略 [{idx}/{len(strategies)}]: {strategy}")
            logging.info("█"*60)
            logging.info("")
            
            try:
                results_df, forecast_df, figures_dir = run_single_strategy(
                    config, data_file, output_dir, config_backup_path, 
                    imputation_strategy=strategy
                )
                
                # Generate Word report for this strategy
                logging.info("="*60)
                logging.info(f"步骤 7/7: 生成报告 (策略: {strategy})")
                logging.info("="*60)
                
                word_report_path = generate_word_report(
                    results_df,
                    config_path=str(config_backup_path),
                    figures_dir=str(figures_dir),
                    output_path=str(report_dir / f'项目评估报告_{strategy}.docx'),
                    forecast_df=forecast_df
                )
                logging.info(f"✓ Word报告已生成: {word_report_path}")
                logging.info("")
                
            except Exception as e:
                logging.error(f"❌ 策略 {strategy} 运行失败: {e}")
                logging.error(f"跳过该策略,继续下一个...")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        logging.info("="*60)
        logging.info(f"✓ 所有策略运行完成! 共生成 {len(strategies)} 个报告")
        logging.info("="*60)
        
    else:
        # Single strategy mode
        single_method = imputation_config.get('method', 'nearest_p')
        logging.info(f"单一策略模式: {single_method}")
        logging.info("")
        
        results_df, forecast_df, figures_dir = run_single_strategy(
            config, data_file, output_dir, config_backup_path, 
            imputation_strategy=None
        )
        
        # Generate report
        logging.info("="*60)
        logging.info("步骤 7/7: 生成报告")
        logging.info("="*60)
        
        report_dir = output_dir / 'report'
        
        word_report_path = generate_word_report(
            results_df,
            config_path=str(config_backup_path),
            figures_dir=str(figures_dir),
            output_path=str(report_dir / '项目评估报告.docx'),
            forecast_df=forecast_df
        )
        logging.info(f"Word报告已生成: {word_report_path}")
    
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
    if strategies and len(strategies) > 0:
        logging.info(f"  - 报告目录: {report_dir} (共 {len(strategies)} 个报告)")
    else:
        logging.info(f"  - 报告目录: {report_dir}")
    logging.info("="*60)


if __name__ == '__main__':
    main()
