#!/usr/bin/env python3
"""
一键启动W&B Sweep超参数调优
使用方法: python start_sweep.py
"""

import subprocess
import sys
import os

def check_wandb():
    """检查并安装wandb"""
    try:
        import wandb
        print("✅ wandb已安装")
        return True
    except ImportError:
        print("❌ wandb未安装")
        response = input("是否安装wandb? (y/n): ")
        if response.lower() == 'y':
            print("📦 正在安装wandb...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
            print("✅ wandb安装成功")
            return True
        return False

def login_wandb():
    """检查wandb登录状态"""
    import wandb
    try:
        wandb.ensure_configured()
        if wandb.api.api_key:
            print("✅ wandb已登录")
            return True
    except:
        pass
    
    print("⚠️  需要登录wandb")
    print("   1. 访问 https://wandb.ai/authorize 获取API key")
    print("   2. 输入API key登录")
    
    try:
        wandb.login()
        return True
    except:
        print("❌ 登录失败")
        return False

def initialize_sweep():
    """初始化sweep"""
    import wandb
    import yaml
    
    print("\n🎯 初始化Sweep...")
    
    # 加载配置
    with open('sweep_config.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # 初始化
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="transformer-tuning"
    )
    
    print(f"✅ Sweep已创建: {sweep_id}")
    return sweep_id

def start_agent(sweep_id, count=None, background=False):
    """启动sweep agent"""
    import wandb
    
    print(f"\n🤖 启动Sweep Agent...")
    print(f"   Sweep ID: {sweep_id}")
    if count:
        print(f"   运行次数: {count}")
    else:
        print(f"   运行次数: 无限制（直到手动停止）")
    
    if background:
        print(f"   模式: 后台运行")
        cmd = f"nohup wandb agent {sweep_id}"
        if count:
            cmd += f" --count {count}"
        cmd += " > sweep_output.log 2>&1 &"
        
        os.system(cmd)
        print("✅ Agent已在后台启动")
        print("   日志文件: sweep_output.log")
        print("   查看日志: tail -f sweep_output.log")
    else:
        print(f"   模式: 前台运行（Ctrl+C停止）")
        from train_sweep import train
        
        wandb.agent(
            sweep_id,
            function=train,
            count=count
        )

def main():
    print("=" * 60)
    print("🚀 W&B Sweep - Transformer超参数优化")
    print("   目标: ACC_10 > 80% (horizon=1)")
    print("=" * 60)
    
    # 1. 检查wandb
    if not check_wandb():
        print("❌ 无法继续，请先安装wandb")
        return
    
    # 2. 登录wandb
    if not login_wandb():
        print("❌ 无法继续，请先登录wandb")
        return
    
    # 3. 选择模式
    print("\n📋 选择运行模式:")
    print("   1. 快速测试 (运行10次实验)")
    print("   2. 标准搜索 (运行30次实验)")
    print("   3. 深度搜索 (运行50次实验)")
    print("   4. 彻夜运行 (无限制，直到手动停止)")
    print("   5. 后台运行 (自定义次数)")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    count_map = {
        '1': 10,
        '2': 30,
        '3': 50,
        '4': None
    }
    
    if choice in count_map:
        # 4. 初始化sweep
        sweep_id = initialize_sweep()
        
        # 5. 启动agent
        start_agent(sweep_id, count=count_map[choice], background=False)
        
    elif choice == '5':
        count = input("运行次数 (留空表示无限制): ").strip()
        count = int(count) if count else None
        
        # 初始化并后台运行
        sweep_id = initialize_sweep()
        start_agent(sweep_id, count=count, background=True)
        
    else:
        print("❌ 无效选择")
        return
    
    print("\n" + "=" * 60)
    print("🌐 查看实时结果:")
    print("   https://wandb.ai/sweeps (在Runs页面)")
    print("\n💡 提示:")
    print("   - wandb会自动保存所有实验结果")
    print("   - 可以在网页端实时查看和对比")
    print("   - 贝叶斯优化会自动找最优超参数")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
