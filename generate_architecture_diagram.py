"""
System architecture visualization
Creates a diagram showing the project structure and data flow
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.use('Agg')

# Configure Chinese fonts
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, '电力质量预测系统架构', fontsize=18, fontweight='bold', ha='center')

# Color scheme
color_data = '#E8F4F8'
color_process = '#FFF4E6'
color_model = '#E8F5E9'
color_output = '#FCE4EC'

def add_box(ax, x, y, width, height, text, color, fontsize=9):
    """Add a rounded box with text"""
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color, 
                         edgecolor='black', 
                         linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
           ha='center', va='center', fontsize=fontsize, fontweight='bold')

def add_arrow(ax, x1, y1, x2, y2, label=''):
    """Add an arrow between boxes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='#666666')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, style='italic')

# Layer 1: Data Input
add_box(ax, 0.5, 9.5, 2, 0.8, '数据输入\n(Excel/CSV)', color_data)
add_box(ax, 3, 9.5, 2, 0.8, 'config.yaml\n配置文件', color_data)

# Layer 2: Data Processing
add_box(ax, 0.5, 8, 1.5, 0.8, 'data_io.py\n数据加载', color_process)
add_box(ax, 2.3, 8, 1.5, 0.8, 'features.py\n特征工程', color_process)
add_box(ax, 4.1, 8, 1.5, 0.8, 'cv.py\n交叉验证', color_process)

# Layer 3: Models
add_box(ax, 0.2, 6, 1.3, 0.7, 'Naive\nBaseline', color_model, fontsize=8)
add_box(ax, 1.6, 6, 1.3, 0.7, 'Seasonal\nNaive', color_model, fontsize=8)
add_box(ax, 3.0, 6, 1.3, 0.7, 'Random\nForest', color_model, fontsize=8)
add_box(ax, 4.4, 6, 1.3, 0.7, 'XGBoost', color_model, fontsize=8)
add_box(ax, 0.2, 5, 1.3, 0.7, 'LSTM', color_model, fontsize=8)
add_box(ax, 1.6, 5, 1.3, 0.7, 'Transformer', color_model, fontsize=8)

# Layer 4: Evaluation
add_box(ax, 1, 3.5, 4, 0.8, 'metrics.py - 评估指标\n(RMSE, MAE, SMAPE, WAPE)', color_process)

# Layer 5: Visualization & Reporting
add_box(ax, 0.5, 2, 1.8, 0.7, 'plots.py\n中文图表', color_output, fontsize=9)
add_box(ax, 2.5, 2, 1.5, 0.7, 'report_md.py\nMarkdown', color_output, fontsize=9)
add_box(ax, 4.2, 2, 1.5, 0.7, 'report_docx.py\nWord报告', color_output, fontsize=9)

# Layer 6: Output
add_box(ax, 0.3, 0.5, 2.5, 0.8, 'outputs/metrics/\ncv_metrics.csv', color_data, fontsize=8)
add_box(ax, 3, 0.5, 1.5, 0.8, 'outputs/figures/\n*.png', color_data, fontsize=8)
add_box(ax, 4.7, 0.5, 2, 0.8, 'outputs/report/\n*.md, *.docx', color_data, fontsize=8)

# Add arrows showing data flow
add_arrow(ax, 1.5, 9.5, 1.25, 8.8)
add_arrow(ax, 4, 9.5, 3.5, 8.8)
add_arrow(ax, 1.25, 8, 1.5, 6.7)
add_arrow(ax, 3.05, 8, 3, 6.7)
add_arrow(ax, 4.85, 8, 4.4, 6.7)

# Models to evaluation
for x in [0.85, 2.25, 3.65, 5.05]:
    add_arrow(ax, x, 6, 3, 4.3)
for x in [0.85, 2.25]:
    add_arrow(ax, x, 5, 3, 4.3)

# Evaluation to visualization
add_arrow(ax, 2.5, 3.5, 1.4, 2.7)
add_arrow(ax, 3, 3.5, 3.25, 2.7)
add_arrow(ax, 3.5, 3.5, 4.95, 2.7)

# Visualization to output
add_arrow(ax, 1.4, 2, 1.55, 1.3)
add_arrow(ax, 3.25, 2, 3.75, 1.3)
add_arrow(ax, 4.95, 2, 5.7, 1.3)

# Add key features boxes on the right
feature_y = 10
features = [
    ('滚动起点验证', '防止数据泄漏'),
    ('多模型对比', '6种预测方法'),
    ('4项评估指标', 'RMSE/MAE/SMAPE/WAPE'),
    ('中文支持', '图表和报告'),
    ('自动化流程', '一键运行')
]

ax.text(7.5, 10.8, '核心特性', fontsize=12, fontweight='bold', ha='center')
for i, (feature, desc) in enumerate(features):
    y = feature_y - i * 0.9
    add_box(ax, 6.3, y - 0.4, 2.4, 0.7, f'{feature}\n{desc}', '#E3F2FD', fontsize=8)

# Add workflow indicator
ax.text(7.5, 4.5, '工作流程', fontsize=12, fontweight='bold', ha='center')
workflow = [
    '1. 加载数据和配置',
    '2. 特征工程',
    '3. 模型训练',
    '4. 交叉验证',
    '5. 生成报告'
]
for i, step in enumerate(workflow):
    ax.text(6.5, 3.8 - i * 0.5, step, fontsize=9, ha='left')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=color_data, edgecolor='black', label='数据/配置'),
    mpatches.Patch(facecolor=color_process, edgecolor='black', label='处理模块'),
    mpatches.Patch(facecolor=color_model, edgecolor='black', label='预测模型'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='输出模块')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, 
         bbox_to_anchor=(0.98, 0.02))

plt.tight_layout()
plt.savefig('outputs/system_architecture.png', dpi=150, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print("System architecture diagram saved to outputs/system_architecture.png")
