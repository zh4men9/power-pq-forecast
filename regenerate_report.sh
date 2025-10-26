#!/bin/bash
# 重新生成评估报告
# 用法: ./regenerate_report.sh [输出目录]
#
# 示例:
#   ./regenerate_report.sh                    # 使用默认输出目录 outputs/latest
#   ./regenerate_report.sh outputs/latest     # 指定输出目录
#   ./regenerate_report.sh outputs_backup     # 使用备份目录

set -e  # 遇到错误立即退出

# 默认输出目录
OUTPUT_DIR="${1:-outputs/output-2025-10-26-1855}"

echo "=================================================="
echo "重新生成评估报告"
echo "=================================================="
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查输出目录是否存在
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "错误: 输出目录不存在: $OUTPUT_DIR"
    echo "请先运行 python run_all.py 生成评估结果"
    exit 1
fi

# 检查必需的文件
METRICS_FILE="$OUTPUT_DIR/metrics/cv_metrics.csv"
FIGURES_DIR="$OUTPUT_DIR/figures"
REPORT_DIR="$OUTPUT_DIR/report"

if [ ! -f "$METRICS_FILE" ]; then
    echo "错误: 未找到评估指标文件: $METRICS_FILE"
    echo "请先运行 python run_all.py 生成评估结果"
    exit 1
fi

if [ ! -d "$FIGURES_DIR" ]; then
    echo "错误: 未找到图表目录: $FIGURES_DIR"
    echo "请先运行 python run_all.py 生成评估结果"
    exit 1
fi

# 创建报告目录（如果不存在）
mkdir -p "$REPORT_DIR"

echo "✓ 检查完成，开始生成报告..."
echo ""

# 生成Word报告
python -c "
import pandas as pd
from pathlib import Path
from src.report_docx import generate_word_report

# 读取评估指标
metrics_df = pd.read_csv('$METRICS_FILE')

# 检查是否有预测结果
forecast_file = Path('$OUTPUT_DIR/forecast/future_forecast.csv')
forecast_df = None
if forecast_file.exists():
    forecast_df = pd.read_csv(forecast_file)
    print('✓ 找到预测结果文件')

# 生成Word报告
output_path = '$REPORT_DIR/项目评估报告.docx'
print(f'正在生成Word报告: {output_path}')

generate_word_report(
    metrics_df=metrics_df,
    figures_dir='$FIGURES_DIR',
    output_path=output_path,
    forecast_df=forecast_df
)

print(f'✓ Word报告生成完成: {output_path}')
"

echo ""
echo "=================================================="
echo "报告生成完成！"
echo "=================================================="
echo "Word报告: $REPORT_DIR/项目评估报告.docx"
echo ""
echo "提示: 可以直接打开Word文档查看完整报告"
