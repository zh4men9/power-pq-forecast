#!/usr/bin/env python3
"""
模拟 wandb sweep 完整运行流程
测试所有可能的参数组合是否有效
"""

import sys
import logging
from unittest.mock import MagicMock, patch
import random

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

print("=" * 60)
print("🧪 模拟 W&B Sweep 运行测试")
print("=" * 60)

# 1. 模拟 wandb
print("\n1️⃣ 创建 wandb mock...")
mock_wandb = MagicMock()
mock_run = MagicMock()
mock_wandb.run = mock_run

# 测试配置集 - 包含各种边界情况
test_configs = [
    {
        "name": "最小配置",
        "config": {
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 1,  # 快速测试
            "sequence_length": 24,
            "max_lag": 24,
            "horizon": 1,
            "test_window": 100,
            "strategy": "nearest_p"
        }
    },
    {
        "name": "中等配置",
        "config": {
            "d_model": 128,
            "nhead": 8,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.15,
            "learning_rate": 0.0005,
            "batch_size": 48,
            "epochs": 1,
            "sequence_length": 48,
            "max_lag": 48,
            "horizon": 1,
            "test_window": 300,
            "strategy": "nearest_p"
        }
    },
    {
        "name": "大模型配置",
        "config": {
            "d_model": 256,
            "nhead": 16,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "learning_rate": 0.0001,
            "batch_size": 64,
            "epochs": 1,
            "sequence_length": 96,
            "max_lag": 96,
            "horizon": 1,
            "test_window": 300,
            "strategy": "nearest_p"
        }
    },
    {
        "name": "特殊配置 d_model=192 nhead=12",
        "config": {
            "d_model": 192,
            "nhead": 12,  # 192能被12整除
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "dim_feedforward": 768,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "epochs": 1,
            "sequence_length": 48,
            "max_lag": 48,
            "horizon": 1,
            "test_window": 300,
            "strategy": "nearest_p"
        }
    },
    {
        "name": "边界配置 - 小batch",
        "config": {
            "d_model": 64,
            "nhead": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.05,
            "learning_rate": 0.00001,
            "batch_size": 8,  # 最小batch
            "epochs": 1,
            "sequence_length": 120,
            "max_lag": 120,
            "horizon": 1,
            "test_window": 450,
            "strategy": "nearest_p"
        }
    }
]

# 2. 验证 d_model 和 nhead 的组合
print("\n2️⃣ 验证参数组合...")
all_valid = True
for test in test_configs:
    d_model = test["config"]["d_model"]
    nhead = test["config"]["nhead"]
    if d_model % nhead != 0:
        print(f"   ❌ {test['name']}: d_model={d_model} 不能被 nhead={nhead} 整除")
        all_valid = False
    else:
        print(f"   ✅ {test['name']}: d_model={d_model} % nhead={nhead} = 0")

if not all_valid:
    print("\n❌ 配置验证失败!")
    sys.exit(1)

# 3. 导入 train_sweep (只导入一次)
print("\n3️⃣ 导入 train_sweep...")
with patch.dict('sys.modules', {'wandb': mock_wandb}):
    import train_sweep

# 4. 运行模拟测试
print("\n4️⃣ 运行模拟训练测试...")
print("   (每个配置训练1个epoch,只使用部分数据)")

passed = 0
failed = 0

for i, test in enumerate(test_configs, 1):
    print(f"\n{'='*60}")
    print(f"测试 {i}/{len(test_configs)}: {test['name']}")
    print(f"{'='*60}")
    
    # 设置 mock config
    mock_config = MagicMock()
    for key, value in test["config"].items():
        setattr(mock_config, key, value)
    mock_wandb.config = mock_config
    
    try:
        # 直接调用 train 函数
        train_sweep.train()
        print(f"✅ {test['name']} - 训练成功!")
        passed += 1
        
    except Exception as e:
        print(f"❌ {test['name']} - 训练失败:")
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

# 5. 总结
print("\n" + "="*60)
print("📊 测试总结")
print("="*60)
print(f"✅ 通过: {passed}/{len(test_configs)}")
print(f"❌ 失败: {failed}/{len(test_configs)}")

if failed == 0:
    print("\n🎉 所有配置测试通过!")
    print("💡 train_sweep.py 可以安全启动 sweep")
    sys.exit(0)
else:
    print(f"\n⚠️  有 {failed} 个配置失败")
    print("🔧 请修复错误后再启动 sweep")
    sys.exit(1)
