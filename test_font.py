#!/usr/bin/env python
"""
清除matplotlib缓存并测试中文字体
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shutil
from pathlib import Path

# 清除matplotlib缓存
cache_dir = Path(matplotlib.get_cachedir())
print(f"Matplotlib缓存目录: {cache_dir}")

if cache_dir.exists():
    for item in cache_dir.glob("*.cache"):
        print(f"删除缓存文件: {item}")
        item.unlink()

# 重建字体列表
fm._load_fontmanager(try_read_cache=False)

# 查找可用的中文字体
print("\n可用的中文字体:")
available_fonts = set()
for f in fm.fontManager.ttflist:
    if any(cn in f.name for cn in ['Hei', 'Song', 'Kai', 'PingFang', 'ST', 'Arial Unicode']):
        available_fonts.add(f.name)
        
for font in sorted(available_fonts):
    print(f"  - {font}")

# 测试绘图
plt.rcParams['font.sans-serif'] = ['STHeiti', 'Heiti TC', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.monospace'] = ['STHeiti', 'Courier New']  # 也设置等宽字体

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot([1, 2, 3], [1, 4, 2])
ax.set_title('测试中文标题 - 处理前后对比', fontsize=14)
ax.set_xlabel('时间索引', fontsize=12)
ax.set_ylabel('有功功率 P (kW)', fontsize=12)
ax.text(2, 2, '数据字段完整，但时间序列有间隙', fontsize=10, ha='center')

output_dir = Path('outputs/test_plots')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'font_test.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ 测试图表已保存到: {output_dir / 'font_test.png'}")
print("请查看图表，确认中文是否正确显示")
