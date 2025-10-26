#!/usr/bin/env python3
"""
快速检查sweep配置是否正确
"""

import sys
from pathlib import Path

print("="*60)
print("🔍 Sweep配置检查")
print("="*60)

# 1. 检查文件存在
required_files = [
    'sweep_config.yaml',
    'config_sweep.yaml', 
    'train_sweep.py',
    'data/raw2/复制数据.xlsx'
]

print("\n1️⃣ 检查必需文件...")
all_exist = True
for file in required_files:
    exists = Path(file).exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {file}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ 缺少必需文件，请检查!")
    sys.exit(1)

# 2. 测试配置加载
print("\n2️⃣ 测试配置加载...")
try:
    from src.config import Config
    config = Config('config_sweep.yaml')
    print("   ✅ config_sweep.yaml 加载成功")
    print(f"      数据路径: {config.config['data']['data_path']}")
    print(f"      文件模式: {config.config['data']['file_pattern']}")
except Exception as e:
    print(f"   ❌ 配置加载失败: {e}")
    sys.exit(1)

# 3. 测试数据加载
print("\n3️⃣ 测试数据加载...")
try:
    from src.data_io import load_data
    data_path = Path(config.config['data']['data_path'])
    file_pattern = config.config['data']['file_pattern']
    data_files = list(data_path.glob(file_pattern))
    
    if not data_files:
        print(f"   ❌ 未找到数据文件: {data_path}/{file_pattern}")
        sys.exit(1)
    
    data_file = data_files[0]
    print(f"   ✅ 找到数据文件: {data_file}")
    
    # 快速加载测试
    df, _ = load_data(
        file_path=str(data_file),
        time_col='时间',
        p_col='有功',
        q_col='无功',
        exog_cols=['定子电流', '定子电压', '转子电压', '转子电流', '励磁电流'],
        freq='H',
        imputation_method='nearest_p',
        target_p_value=349
    )
    print(f"   ✅ 数据加载成功: {df.shape[0]} 样本, {df.shape[1]} 列")
    
except Exception as e:
    print(f"   ❌ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 检查wandb
print("\n4️⃣ 检查wandb...")
try:
    import wandb
    print("   ✅ wandb已安装")
    
    # 检查登录状态
    try:
        wandb.ensure_configured()
        if wandb.api.api_key:
            print("   ✅ wandb已登录")
        else:
            print("   ⚠️  wandb未登录，运行前请先: wandb login")
    except:
        print("   ⚠️  wandb未登录，运行前请先: wandb login")
        
except ImportError:
    print("   ❌ wandb未安装")
    print("      请运行: pip install wandb")
    sys.exit(1)

# 5. 估算搜索空间
print("\n5️⃣ 搜索空间估算...")
import yaml
with open('sweep_config.yaml') as f:
    sweep_config = yaml.safe_load(f)

params = sweep_config['parameters']
d_model_count = len(params['d_model']['values'])
nhead_count = len(params['nhead']['values'])
encoder_count = len(params['num_encoder_layers']['values'])
decoder_count = len(params['num_decoder_layers']['values'])
dim_ff_count = len(params['dim_feedforward']['values'])
batch_count = len(params['batch_size']['values'])
epochs_count = len(params['epochs']['values'])
seq_count = len(params['sequence_length']['values'])
lag_count = len(params['max_lag']['values'])
test_count = len(params['test_window']['values'])

total_combinations = (d_model_count * nhead_count * encoder_count * decoder_count * 
                     dim_ff_count * batch_count * epochs_count * seq_count * 
                     lag_count * test_count)

print(f"   总搜索空间: {total_combinations:,} 种组合")
print(f"   贝叶斯优化建议: 30-50次实验")
print(f"   预计时间: 2.5-5小时 (每次5-10分钟)")

# 6. 显示启动命令
print("\n" + "="*60)
print("✅ 所有检查通过!")
print("="*60)
print("\n🚀 启动Sweep的三种方式:")
print("\n方式1 (推荐): 使用start_sweep.py")
print("   python start_sweep.py")
print("   选择运行模式即可")

print("\n方式2: 手动命令")
print("   # 初始化sweep")
print("   wandb sweep sweep_config.yaml --project transformer-tuning")
print("   # 启动agent (会输出sweep ID)")
print("   wandb agent <sweep-id>")

print("\n方式3: 后台运行")
print("   wandb sweep sweep_config.yaml --project transformer-tuning")
print("   nohup wandb agent <sweep-id> > sweep.log 2>&1 &")
print("   tail -f sweep.log")

print("\n💡 提示:")
print("   - 可以启动多个agent并行搜索")
print("   - Ctrl+C 可以随时停止")
print("   - 已完成的实验结果会保存")
print("   - 在wandb网页端查看实时结果")

print("\n" + "="*60)
