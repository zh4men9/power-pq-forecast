"""
Markdown report generation module
Generates comprehensive evaluation report in Markdown format
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional


def generate_markdown_report(
    metrics_df: pd.DataFrame,
    config_path: str = "config.yaml",
    output_path: str = "outputs/report/项目评估报告.md"
) -> str:
    """
    Generate Markdown report from evaluation results
    
    Args:
        metrics_df: DataFrame with evaluation metrics
        config_path: Path to configuration file
        output_path: Path to save report
    
    Returns:
        Path to generated report
    """
    # Aggregate results by model and horizon
    agg_results = metrics_df.groupby(['model', 'horizon', 'target']).agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'SMAPE': 'mean',
        'WAPE': 'mean'
    }).reset_index()
    
    # Generate report content
    report_lines = []
    
    # Title and metadata
    report_lines.append("# 电力质量预测项目评估报告")
    report_lines.append("")
    report_lines.append(f"**报告生成日期**: {datetime.now().strftime('%Y年%m月%d日')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Executive summary
    report_lines.append("## 一、项目概况")
    report_lines.append("")
    report_lines.append("本项目针对电力系统的有功功率（P）和无功功率（Q）进行时间序列预测分析。")
    report_lines.append("采用多种预测模型进行对比评估，包括朴素基线、季节性基线、随机森林、XGBoost、LSTM和Transformer等方法。")
    report_lines.append("")
    
    # Model descriptions
    report_lines.append("## 二、模型介绍")
    report_lines.append("")
    report_lines.append("本项目采用六种不同类型的预测模型，涵盖基线方法、传统机器学习和深度学习方法：")
    report_lines.append("")
    
    report_lines.append("### 2.1 朴素预测（Naive）")
    report_lines.append("")
    report_lines.append("朴素预测是最简单的基线模型，使用最后一个观测值作为未来所有时间步的预测值。")
    report_lines.append("虽然方法简单，但在许多实际应用中表现出色，常作为其他模型的基准对比。")
    report_lines.append("")
    
    report_lines.append("### 2.2 季节朴素预测（Seasonal Naive）")
    report_lines.append("")
    report_lines.append("季节朴素预测考虑了数据的季节性特征，使用上一个季节周期对应时刻的观测值进行预测。")
    report_lines.append("适用于具有明显周期性模式的时间序列数据，如电力负荷的日周期或周周期特性。")
    report_lines.append("")
    
    report_lines.append("### 2.3 随机森林（Random Forest）")
    report_lines.append("")
    report_lines.append("随机森林是一种集成学习方法，通过构建多个决策树并综合其预测结果来提高模型的准确性和稳定性。")
    report_lines.append("该模型能够自动捕获特征之间的非线性关系，并提供特征重要性分析，具有较强的泛化能力和抗过拟合能力。")
    report_lines.append("")
    
    report_lines.append("### 2.4 XGBoost")
    report_lines.append("")
    report_lines.append("XGBoost是一种高效的梯度提升决策树算法，通过迭代方式构建多个弱学习器并加权组合。")
    report_lines.append("相比随机森林，XGBoost在处理大规模数据时性能更优，且能够更好地处理缺失值和异常值。")
    report_lines.append("该算法在各类机器学习竞赛中表现优异，广泛应用于时间序列预测任务。")
    report_lines.append("")
    
    report_lines.append("### 2.5 LSTM（长短期记忆网络）")
    report_lines.append("")
    report_lines.append("LSTM是一种特殊的循环神经网络（RNN），专门设计用于处理序列数据和长期依赖问题。")
    report_lines.append("通过引入门控机制（输入门、遗忘门、输出门），LSTM能够有效捕获时间序列中的长期依赖关系，")
    report_lines.append("避免了传统RNN的梯度消失问题。适用于复杂的时序模式识别和多步预测任务。")
    report_lines.append("")
    
    report_lines.append("### 2.6 Transformer")
    report_lines.append("")
    report_lines.append("Transformer基于自注意力机制（Self-Attention），摒弃了传统的循环结构，能够并行处理序列数据。")
    report_lines.append("通过多头注意力机制，模型可以同时关注序列中不同位置的信息，捕获长距离依赖关系。")
    report_lines.append("位置编码（Positional Encoding）保留了序列的时序信息。相比LSTM，Transformer在处理长序列时更加高效，")
    report_lines.append("并在自然语言处理和时间序列预测等领域取得了显著成果。")
    report_lines.append("")
    
    # Methodology
    report_lines.append("## 三、评估方法")
    report_lines.append("")
    report_lines.append("### 3.1 验证策略")
    report_lines.append("")
    report_lines.append("本项目采用**滚动起点交叉验证**（Rolling Origin Cross-Validation）方法，")
    report_lines.append("确保训练集始终位于测试集之前，严格避免使用未来信息进行训练。")
    report_lines.append("此方法是时间序列预测的标准验证方式，符合实际应用场景。")
    report_lines.append("")
    report_lines.append("### 3.2 评估指标")
    report_lines.append("")
    report_lines.append("本项目采用以下四项指标综合评估模型性能：")
    report_lines.append("")
    report_lines.append("- **RMSE** (Root Mean Squared Error): 均方根误差，反映预测误差的绝对大小，对大误差更敏感")
    report_lines.append("- **MAE** (Mean Absolute Error): 平均绝对误差，反映预测误差的平均水平")
    report_lines.append("- **SMAPE** (Symmetric Mean Absolute Percentage Error): 对称平均绝对百分比误差，百分比形式的相对误差，取值范围0-200%")
    report_lines.append("- **WAPE** (Weighted Absolute Percentage Error): 加权绝对百分比误差，相比MAPE更稳健，不受零值影响")
    report_lines.append("")
    report_lines.append("说明：由于MAPE在真实值接近零时会产生极大值，本项目采用SMAPE和WAPE作为相对误差指标，")
    report_lines.append("配合RMSE和MAE作为绝对误差指标，形成完整的评估体系。")
    report_lines.append("")
    
    # Results
    report_lines.append("## 四、评估结果")
    report_lines.append("")
    
    # Get unique horizons and targets
    horizons = sorted(agg_results['horizon'].unique())
    targets = ['P', 'Q']
    target_names = {'P': '有功功率', 'Q': '无功功率'}
    
    for target in targets:
        report_lines.append(f"### 4.{targets.index(target) + 1} {target_names[target]}预测结果")
        report_lines.append("")
        
        target_data = agg_results[agg_results['target'] == target]
        
        for metric in ['RMSE', 'MAE', 'SMAPE', 'WAPE']:
            report_lines.append(f"#### {metric}指标")
            report_lines.append("")
            
            # Create table
            report_lines.append("| 模型 | " + " | ".join([f"步长{h}" for h in horizons]) + " |")
            report_lines.append("|" + "---|" * (len(horizons) + 1))
            
            for model in target_data['model'].unique():
                model_data = target_data[target_data['model'] == model]
                row = [model]
                for h in horizons:
                    h_data = model_data[model_data['horizon'] == h]
                    if len(h_data) > 0:
                        value = h_data[metric].values[0]
                        row.append(f"{value:.4f}")
                    else:
                        row.append("N/A")
                report_lines.append("| " + " | ".join(row) + " |")
            
            report_lines.append("")
        
        # Add figure reference
        report_lines.append(f"**图表**: 不同预测步长下的误差变化")
        report_lines.append("")
        report_lines.append(f"![{target}误差曲线](../figures/error_by_horizon.png)")
        report_lines.append("")
    
    # Conclusions
    report_lines.append("## 五、结论与建议")
    report_lines.append("")
    report_lines.append("### 5.1 主要发现")
    report_lines.append("")
    
    # Find best model for each target
    for target in targets:
        target_data = agg_results[agg_results['target'] == target]
        best_models = {}
        for metric in ['RMSE', 'MAE', 'SMAPE', 'WAPE']:
            if metric in ['RMSE', 'MAE', 'SMAPE', 'WAPE']:
                # Lower is better
                best = target_data.groupby('model')[metric].mean().idxmin()
                best_models[metric] = best
        
        report_lines.append(f"- **{target_names[target]}**: ")
        for metric, best_model in best_models.items():
            report_lines.append(f"  - {metric}最优模型: {best_model}")
    
    report_lines.append("")
    report_lines.append("### 5.2 基线对比")
    report_lines.append("")
    report_lines.append("所有模型均与朴素基线（Naive）和季节性朴素基线（SeasonalNaive）进行对比。")
    report_lines.append("只有在各项指标上显著优于基线的模型才具有实际应用价值。")
    report_lines.append("")
    
    report_lines.append("### 5.3 建议")
    report_lines.append("")
    report_lines.append("1. 模型选择应综合考虑预测精度、计算成本和可解释性")
    report_lines.append("2. 建议在实际部署前进行更多折数的交叉验证以确保稳定性")
    report_lines.append("3. 可根据实际需求调整预测步长和模型超参数")
    report_lines.append("4. 定期使用新数据重新训练和评估模型")
    report_lines.append("")
    
    # Appendix
    report_lines.append("## 六、附录")
    report_lines.append("")
    report_lines.append("### 6.1 数据说明")
    report_lines.append("")
    report_lines.append("- 数据来源: 电力系统监测数据")
    report_lines.append("- 包含变量: 有功功率（P）、无功功率（Q）")
    report_lines.append("- 时间粒度: 根据配置文件设定")
    report_lines.append("")
    
    report_lines.append("### 6.2 可复现性")
    report_lines.append("")
    report_lines.append(f"- 配置文件: `{config_path}`")
    report_lines.append("- 详细指标: `outputs/metrics/cv_metrics.csv`")
    report_lines.append("- 图表目录: `outputs/figures/`")
    report_lines.append("- 执行命令: `python run_all.py --config config.yaml`")
    report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    report_lines.append(f"*本报告由电力质量预测系统自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Markdown report generated: {output_file}")
    
    return str(output_file)
