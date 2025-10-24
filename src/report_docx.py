"""
Word (DOCX) report generation module
Generates comprehensive evaluation report in Word format using python-docx
Reference: https://python-docx.readthedocs.io/en/latest/
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from typing import Optional


def add_heading_with_style(doc: Document, text: str, level: int = 1):
    """Add a heading with custom style"""
    heading = doc.add_heading(text, level=level)
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return heading


def add_table_from_dataframe(doc: Document, df: pd.DataFrame, title: Optional[str] = None):
    """
    Add a table to document from DataFrame
    
    Args:
        doc: Document object
        df: DataFrame to convert to table
        title: Optional table title
    """
    if title:
        doc.add_paragraph(title, style='Heading 3')
    
    # Create table
    table = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    table.style = 'Light Grid Accent 1'
    
    # Header row
    header_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        header_cells[i].text = str(col)
        # Bold header
        for paragraph in header_cells[i].paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data rows
    for i, row in enumerate(df.itertuples(index=False)):
        cells = table.rows[i + 1].cells
        for j, value in enumerate(row):
            cells[j].text = str(value)
    
    doc.add_paragraph()  # Add spacing


def generate_word_report(
    metrics_df: pd.DataFrame,
    config_path: str = "config.yaml",
    figures_dir: str = "outputs/figures",
    output_path: str = "outputs/report/项目评估报告.docx"
) -> str:
    """
    Generate Word (DOCX) report from evaluation results
    
    Uses python-docx for inline image insertion. Images are saved first,
    then inserted using add_picture() method.
    
    Reference: https://python-docx.readthedocs.io/en/latest/user/shapes.html
    
    Args:
        metrics_df: DataFrame with evaluation metrics
        config_path: Path to configuration file
        figures_dir: Directory containing figures
        output_path: Path to save report
    
    Returns:
        Path to generated report
    """
    # Create document
    doc = Document()
    
    # Title page
    title = doc.add_heading('电力质量预测项目评估报告', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Date
    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_run = date_para.add_run(f'报告生成日期: {datetime.now().strftime("%Y年%m月%d日")}')
    date_run.font.size = Pt(12)
    
    doc.add_page_break()
    
    # Aggregate results
    agg_results = metrics_df.groupby(['model', 'horizon', 'target']).agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'SMAPE': 'mean',
        'WAPE': 'mean'
    }).reset_index()
    
    # Section 1: Overview
    add_heading_with_style(doc, '一、项目概况', level=1)
    
    doc.add_paragraph(
        '本项目针对电力系统的有功功率（P）和无功功率（Q）进行时间序列预测分析。'
        '采用多种预测模型进行对比评估，包括朴素基线、季节性基线、随机森林、'
        'XGBoost、LSTM和Transformer等方法。'
    )
    
    doc.add_paragraph()
    
    # Section 2: Methodology
    add_heading_with_style(doc, '二、评估方法', level=1)
    
    add_heading_with_style(doc, '2.1 验证策略', level=2)
    doc.add_paragraph(
        '本项目采用滚动起点交叉验证（Rolling Origin Cross-Validation）方法，'
        '确保训练集始终位于测试集之前，严格避免使用未来信息进行训练。'
        '此方法是时间序列预测的标准验证方式，符合实际应用场景。'
    )
    
    add_heading_with_style(doc, '2.2 评估指标', level=2)
    doc.add_paragraph('本项目采用以下四项指标综合评估模型性能：')
    
    metrics_list = [
        'RMSE (Root Mean Squared Error): 均方根误差，反映预测误差的绝对大小，对大误差更敏感',
        'MAE (Mean Absolute Error): 平均绝对误差，反映预测误差的平均水平',
        'SMAPE (Symmetric Mean Absolute Percentage Error): 对称平均绝对百分比误差，百分比形式的相对误差',
        'WAPE (Weighted Absolute Percentage Error): 加权绝对百分比误差，相比MAPE更稳健，不受零值影响'
    ]
    
    for metric_desc in metrics_list:
        doc.add_paragraph(metric_desc, style='List Bullet')
    
    doc.add_paragraph(
        '说明：由于MAPE在真实值接近零时会产生极大值，本项目采用SMAPE和WAPE作为相对误差指标，'
        '配合RMSE和MAE作为绝对误差指标，形成完整的评估体系。'
    )
    
    # Section 3: Results
    add_heading_with_style(doc, '三、评估结果', level=1)
    
    horizons = sorted(agg_results['horizon'].unique())
    targets = ['P', 'Q']
    target_names = {'P': '有功功率', 'Q': '无功功率'}
    
    for target in targets:
        add_heading_with_style(doc, f'3.{targets.index(target) + 1} {target_names[target]}预测结果', level=2)
        
        target_data = agg_results[agg_results['target'] == target]
        
        for metric in ['RMSE', 'MAE', 'SMAPE', 'WAPE']:
            add_heading_with_style(doc, f'{metric}指标', level=3)
            
            # Create pivot table
            pivot_data = target_data.pivot(index='model', columns='horizon', values=metric)
            pivot_data.columns = [f'步长{h}' for h in pivot_data.columns]
            pivot_data = pivot_data.reset_index()
            pivot_data.columns.name = None
            
            # Format values
            for col in pivot_data.columns:
                if col != 'model':
                    pivot_data[col] = pivot_data[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            
            pivot_data.rename(columns={'model': '模型'}, inplace=True)
            
            add_table_from_dataframe(doc, pivot_data)
    
    # Add error by horizon figure
    add_heading_with_style(doc, '3.3 预测步长误差变化图', level=2)
    
    error_fig_path = Path(figures_dir) / 'error_by_horizon.png'
    if error_fig_path.exists():
        doc.add_paragraph('图1: 不同模型在各预测步长的误差对比')
        doc.add_picture(str(error_fig_path), width=Inches(6))
        doc.add_paragraph()
    else:
        doc.add_paragraph('注: 误差变化图未生成')
    
    # Section 4: Conclusions
    add_heading_with_style(doc, '四、结论与建议', level=1)
    
    add_heading_with_style(doc, '4.1 主要发现', level=2)
    
    # Find best models
    for target in targets:
        doc.add_paragraph(f'{target_names[target]}预测:', style='List Bullet')
        target_data = agg_results[agg_results['target'] == target]
        
        for metric in ['RMSE', 'MAE', 'SMAPE', 'WAPE']:
            best_model = target_data.groupby('model')[metric].mean().idxmin()
            best_value = target_data.groupby('model')[metric].mean().min()
            doc.add_paragraph(
                f'{metric}最优模型: {best_model} (平均值: {best_value:.4f})',
                style='List Number'
            )
    
    add_heading_with_style(doc, '4.2 基线对比', level=2)
    doc.add_paragraph(
        '所有模型均与朴素基线（Naive）和季节性朴素基线（SeasonalNaive）进行对比。'
        '只有在各项指标上显著优于基线的模型才具有实际应用价值。'
    )
    
    add_heading_with_style(doc, '4.3 建议', level=2)
    
    recommendations = [
        '模型选择应综合考虑预测精度、计算成本和可解释性',
        '建议在实际部署前进行更多折数的交叉验证以确保稳定性',
        '可根据实际需求调整预测步长和模型超参数',
        '定期使用新数据重新训练和评估模型'
    ]
    
    for rec in recommendations:
        doc.add_paragraph(rec, style='List Number')
    
    # Section 5: Appendix
    add_heading_with_style(doc, '五、附录', level=1)
    
    add_heading_with_style(doc, '5.1 数据说明', level=2)
    doc.add_paragraph('数据来源: 电力系统监测数据', style='List Bullet')
    doc.add_paragraph('包含变量: 有功功率（P）、无功功率（Q）', style='List Bullet')
    doc.add_paragraph('时间粒度: 根据配置文件设定', style='List Bullet')
    
    add_heading_with_style(doc, '5.2 可复现性', level=2)
    doc.add_paragraph(f'配置文件: {config_path}', style='List Bullet')
    doc.add_paragraph('详细指标: outputs/metrics/cv_metrics.csv', style='List Bullet')
    doc.add_paragraph('图表目录: outputs/figures/', style='List Bullet')
    doc.add_paragraph('执行命令: python run_all.py --config config.yaml', style='List Bullet')
    
    # Footer
    doc.add_page_break()
    footer_para = doc.add_paragraph()
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer_para.add_run(
        f'本报告由电力质量预测系统自动生成于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    )
    footer_run.font.size = Pt(10)
    footer_run.font.italic = True
    
    # Save document
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_file))
    
    print(f"Word report generated: {output_file}")
    
    return str(output_file)
