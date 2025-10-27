#!/usr/bin/env python
"""
测试模型加载功能
"""
import sys
from pathlib import Path

def test_model_loading():
    """测试模型加载功能的基本流程"""
    
    print("="*60)
    print("测试模型加载功能")
    print("="*60)
    
    # 1. 检查模型文件是否存在
    models_dir = Path("outputs/output-2025-10-27-0952/models_nearest_p")
    
    print(f"\n1. 检查模型目录: {models_dir}")
    if not models_dir.exists():
        print(f"   ❌ 模型目录不存在")
        return False
    print(f"   ✓ 模型目录存在")
    
    # 2. 列出模型文件
    print(f"\n2. 列出模型文件:")
    model_files = list(models_dir.glob("*.pkl"))
    if not model_files:
        print(f"   ❌ 未找到任何 .pkl 模型文件")
        return False
    
    for mf in model_files:
        print(f"   - {mf.name}")
    
    # 3. 尝试加载一个模型文件
    print(f"\n3. 测试加载 LSTM 模型:")
    lstm_file = models_dir / "LSTM_h1.pkl"
    if not lstm_file.exists():
        print(f"   ⚠️  LSTM_h1.pkl 不存在")
    else:
        import pickle
        import torch
        try:
            # Try torch.load first (for PyTorch models)
            try:
                save_dict = torch.load(lstm_file, map_location='cpu')
                print(f"   ✓ 文件加载成功 (使用 torch.load)")
            except:
                # Fallback to pickle
                with open(lstm_file, 'rb') as f:
                    save_dict = pickle.load(f)
                print(f"   ✓ 文件加载成功 (使用 pickle.load)")
            
            print(f"   字典键: {list(save_dict.keys())}")
            
            if 'model_state_dict' in save_dict:
                print(f"   ✓ 包含 model_state_dict")
                # 尝试查看第一个键
                keys = list(save_dict['model_state_dict'].keys())
                print(f"   state_dict 包含 {len(keys)} 个参数")
                if keys:
                    print(f"   第一个参数: {keys[0]}")
            else:
                print(f"   ❌ 缺少 model_state_dict")
                return False
            
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 4. 测试 Transformer 模型
    print(f"\n4. 测试加载 Transformer 模型:")
    transformer_file = models_dir / "Transformer_h1.pkl"
    if not transformer_file.exists():
        print(f"   ⚠️  Transformer_h1.pkl 不存在")
    else:
        import torch
        try:
            # Try torch.load first (for PyTorch models)
            try:
                save_dict = torch.load(transformer_file, map_location='cpu')
                print(f"   ✓ 文件加载成功 (使用 torch.load)")
            except:
                # Fallback to pickle
                with open(transformer_file, 'rb') as f:
                    save_dict = pickle.load(f)
                print(f"   ✓ 文件加载成功 (使用 pickle.load)")
            
            print(f"   字典键: {list(save_dict.keys())}")
            
            if 'model_state_dict' in save_dict:
                print(f"   ✓ 包含 model_state_dict")
                keys = list(save_dict['model_state_dict'].keys())
                print(f"   state_dict 包含 {len(keys)} 个参数")
                if keys:
                    print(f"   第一个参数: {keys[0]}")
            else:
                print(f"   ❌ 缺少 model_state_dict")
                return False
            
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 5. 检查配置文件
    print(f"\n5. 检查配置文件:")
    config_file = Path("config_p_only_newdata-transformer89.yaml")
    if not config_file.exists():
        print(f"   ❌ 配置文件不存在: {config_file}")
        return False
    print(f"   ✓ 配置文件存在: {config_file}")
    
    print("\n" + "="*60)
    print("✓ 所有基础检查通过!")
    print("="*60)
    print("\n现在可以运行:")
    print(f"  python run_all.py --config {config_file} --load-models {models_dir}")
    print()
    
    return True

if __name__ == '__main__':
    success = test_model_loading()
    sys.exit(0 if success else 1)
