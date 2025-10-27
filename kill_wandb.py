#!/usr/bin/env python3
"""
安全终止所有 wandb 相关进程，但保留 run_all.py
使用方法: python kill_wandb.py [--force]
"""

import psutil
import sys
import time
import argparse

def get_wandb_processes():
    """获取所有 wandb 相关进程，但排除 run_all.py 和本脚本"""
    wandb_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
        try:
            # 获取完整命令行
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            cmdline_lower = cmdline.lower()
            
            # 检查是否是 wandb 相关进程
            if 'wandb' in cmdline_lower or 'wandb' in proc.info['name'].lower():
                # 排除条件
                if 'run_all' in cmdline_lower:
                    continue
                if 'kill_wandb' in cmdline_lower:
                    continue
                
                wandb_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cmdline': cmdline,
                    'username': proc.info['username'],
                    'proc': proc
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return wandb_processes

def check_run_all_running():
    """检查 run_all.py 是否在运行"""
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'run_all.py' in cmdline:
                return proc.info['pid'], cmdline
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None, None

def kill_processes(processes, force=False):
    """终止进程"""
    killed = []
    failed = []
    
    for proc_info in processes:
        try:
            proc = proc_info['proc']
            pid = proc_info['pid']
            
            # 尝试优雅地终止
            proc.terminate()
            
            # 等待进程结束
            try:
                proc.wait(timeout=3)
                killed.append(proc_info)
                print(f"  ✅ 已终止 PID {pid}: {proc_info['name']}")
            except psutil.TimeoutExpired:
                if force:
                    # 强制终止
                    proc.kill()
                    proc.wait(timeout=1)
                    killed.append(proc_info)
                    print(f"  ⚠️  强制终止 PID {pid}: {proc_info['name']}")
                else:
                    failed.append(proc_info)
                    print(f"  ❌ 无法终止 PID {pid}: {proc_info['name']} (使用 --force 强制终止)")
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            failed.append(proc_info)
            print(f"  ❌ 无法终止 PID {pid}: {e}")
    
    return killed, failed

def main():
    parser = argparse.ArgumentParser(description='安全终止所有 wandb 相关进程')
    parser.add_argument('--force', '-f', action='store_true', 
                       help='强制终止无响应的进程')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='跳过确认提示')
    args = parser.parse_args()
    
    print("🔍 查找 wandb 相关进程...")
    print("=" * 60)
    
    # 获取 wandb 进程
    wandb_procs = get_wandb_processes()
    
    if not wandb_procs:
        print("✅ 没有找到需要终止的 wandb 进程")
        
        # 检查 run_all.py
        run_all_pid, run_all_cmd = check_run_all_running()
        if run_all_pid:
            print(f"\n✅ run_all.py 仍在运行 (已保护):")
            print(f"  PID: {run_all_pid}")
            print(f"  CMD: {run_all_cmd}")
        
        return 0
    
    # 显示将要终止的进程
    print(f"\n找到 {len(wandb_procs)} 个 wandb 相关进程:\n")
    for proc_info in wandb_procs:
        print(f"  PID: {proc_info['pid']:<8} USER: {proc_info['username']:<12}")
        print(f"  CMD: {proc_info['cmdline'][:100]}...")
        print()
    
    # 检查 run_all.py
    run_all_pid, run_all_cmd = check_run_all_running()
    if run_all_pid:
        print(f"🛡️  检测到 run_all.py 正在运行 (将被保护):")
        print(f"  PID: {run_all_pid}")
        print(f"  CMD: {run_all_cmd}\n")
    
    # 确认
    if not args.yes:
        response = input("⚠️  确定要终止这些进程吗? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ 已取消操作")
            return 1
    
    # 终止进程
    print("\n🔨 正在终止进程...")
    killed, failed = kill_processes(wandb_procs, force=args.force)
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 执行结果:")
    print(f"  ✅ 成功终止: {len(killed)} 个进程")
    if failed:
        print(f"  ❌ 终止失败: {len(failed)} 个进程")
        if not args.force:
            print("  💡 提示: 使用 --force 参数强制终止")
    
    # 验证
    print("\n🔍 验证结果...")
    remaining = get_wandb_processes()
    if remaining:
        print(f"⚠️  还有 {len(remaining)} 个 wandb 进程仍在运行:")
        for proc_info in remaining:
            print(f"  PID: {proc_info['pid']} - {proc_info['name']}")
    else:
        print("✅ 所有 wandb 进程已终止")
    
    # 再次检查 run_all.py
    run_all_pid, run_all_cmd = check_run_all_running()
    if run_all_pid:
        print(f"\n✅ run_all.py 仍在运行 (已保护):")
        print(f"  PID: {run_all_pid}")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ 操作已取消")
        sys.exit(1)
