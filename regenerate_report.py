#!/usr/bin/env python3
"""
重新生成报告脚本
基于已有的训练结果重新生成Word和图表
使用方法: python regenerate_report.py --output outputs/output-2025-10-27-0952
"""

import argparse
import sys
from pathlib import Path
import logging
import pandas as pd
import shutil

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.plots import plot_error_by_horizon, plot_all_metrics_by_horizon, configure_chinese_fonts
from src.report_docx import generate_word_report

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def find_strategy_suffix(output_dir):
    """检测策略后缀"""
    output_path = Path(output_dir)
    
    # 查找 metrics 文件
    metrics_files = list(output_path.glob("metrics*/cv_metrics.csv"))
    
    if not metrics_files:
        logging.error("❌ 未找到任何 metrics 文件")
        return None
    
    if len(metrics_files) > 1:
        logging.info(f"找到 {len(metrics_files)} 个 metrics 文件:")
        for f in metrics_files:
            logging.info(f"  - {f.parent.name}")
        # 使用第一个
        logging.info(f"使用: {metrics_files[0].parent.name}")
    
    # 提取后缀
    metrics_dir = metrics_files[0].parent.name
    if metrics_dir == "metrics":
        return ""
    elif metrics_dir.startswith("metrics_"):
        return "_" + metrics_dir.replace("metrics_", "")
    
    return ""

def regenerate_report(output_dir, config_file=None, force=False):
    """重新生成报告"""
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        logging.error(f"❌ 输出目录不存在: {output_dir}")
        return False
    
    logging.info("="*60)
    logging.info("🔄 重新生成报告")
    logging.info("="*60)
    logging.info(f"输出目录: {output_dir}")
    
    # 1. 查找配置文件
    if config_file is None:
        config_backup = output_path / "config_used.yaml"
        if config_backup.exists():
            config_file = str(config_backup)
            logging.info(f"✓ 使用备份的配置: {config_file}")
        else:
            logging.error("❌ 未找到配置文件，请使用 --config 指定")
            return False
    else:
        if not Path(config_file).exists():
            logging.error(f"❌ 配置文件不存在: {config_file}")
            return False
        logging.info(f"✓ 使用指定的配置: {config_file}")
    
    # 2. 加载配置
    config = load_config(config_file)
    
    # 3. 检测策略后缀
    strategy_suffix = find_strategy_suffix(output_dir)
    if strategy_suffix is None:
        return False
    
    if strategy_suffix:
        logging.info(f"✓ 检测到策略后缀: {strategy_suffix}")
        strategy_name = strategy_suffix.lstrip("_")
    else:
        strategy_name = ""
    
    # 4. 查找metrics文件
    metrics_dir = output_path / f"metrics{strategy_suffix}"
    metrics_file = metrics_dir / "cv_metrics.csv"
    
    if not metrics_file.exists():
        logging.error(f"❌ 未找到metrics文件: {metrics_file}")
        # 尝试重新搜索
        metrics_files = list(output_path.glob("metrics*/cv_metrics.csv"))
        if metrics_files:
            metrics_file = metrics_files[0]
            logging.info(f"✓ 使用找到的metrics文件: {metrics_file}")
        else:
            return False
    
    logging.info(f"✓ 找到metrics文件: {metrics_file}")
    
    # 5. 加载结果
    results_df = pd.read_csv(metrics_file)
    logging.info(f"✓ 加载了 {len(results_df)} 条评估记录")
    
    # 6. 设置输出目录
    figures_dir = output_path / f"figures{strategy_suffix}"
    figures_dir.mkdir(exist_ok=True)
    
    report_dir = output_path / "report"
    report_dir.mkdir(exist_ok=True)
    
    # 7. 重新生成图表
    logging.info("")
    logging.info("="*60)
    logging.info("步骤 1/2: 重新生成图表")
    logging.info("="*60)
    
    configure_chinese_fonts(config.get('plotting', 'font_priority'))
    
    # Plot error by horizon for RMSE
    rmse_plot_path = figures_dir / 'error_by_horizon_rmse.png'
    plot_error_by_horizon(
        results_df,
        metric_name='RMSE',
        output_path=str(rmse_plot_path)
    )
    logging.info(f"✓ RMSE误差图已生成: {rmse_plot_path}")
    
    # Plot all metrics by horizon
    all_metrics_plot_path = figures_dir / 'all_metrics_by_horizon.png'
    plot_all_metrics_by_horizon(
        results_df,
        output_path=str(all_metrics_plot_path)
    )
    logging.info(f"✓ 所有指标图已生成: {all_metrics_plot_path}")
    
    # 8. 重新生成Word报告
    logging.info("")
    logging.info("="*60)
    logging.info("步骤 2/2: 重新生成Word报告")
    logging.info("="*60)
    
    if strategy_name:
        report_filename = f"项目评估报告_{strategy_name}.docx"
    else:
        report_filename = "项目评估报告.docx"
    
    word_report_path = report_dir / report_filename
    
    # 如果报告已存在且不是强制模式，询问
    if word_report_path.exists() and not force:
        response = input(f"⚠️  报告已存在: {word_report_path}\n是否覆盖? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            logging.info("❌ 已取消操作")
            return False
    
    # 生成报告
    word_report = generate_word_report(
        results_df,
        config_path=config_file,
        output_dir=str(report_dir),
        figures_dir=str(figures_dir),
        strategy_name=strategy_name if strategy_name else None
    )
    
    if word_report:
        logging.info(f"✓ Word报告已生成: {word_report}")
    else:
        logging.error("❌ Word报告生成失败")
        return False
    
    # 9. 总结
    logging.info("")
    logging.info("="*60)
    logging.info("✅ 报告重新生成完成!")
    logging.info("="*60)
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"  - 图表目录: {figures_dir}")
    logging.info(f"  - 报告文件: {word_report}")
    logging.info("="*60)
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='基于已有训练结果重新生成报告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（自动查找备份的配置）
  python regenerate_report.py --output outputs/output-2025-10-27-0952
  
  # 指定配置文件
  python regenerate_report.py --output outputs/output-2025-10-27-0952 --config config.yaml
  
  # 强制覆盖已有报告
  python regenerate_report.py --output outputs/output-2025-10-27-0952 --force
  
  # 使用latest链接
  python regenerate_report.py --output outputs/latest
        """
    )
    
    parser.add_argument('--output', '-o', required=True,
                       help='输出目录路径 (例如: outputs/output-2025-10-27-0952)')
    parser.add_argument('--config', '-c', default=None,
                       help='配置文件路径 (默认: 使用output目录中的config_used.yaml)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='强制覆盖已有报告，不询问')
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        success = regenerate_report(
            output_dir=args.output,
            config_file=args.config,
            force=args.force
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logging.info("\n\n❌ 操作已取消")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
