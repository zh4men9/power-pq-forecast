#!/bin/bash
# 安全终止所有 wandb 相关进程，但保留 run_all.py
# 使用方法: bash kill_wandb.sh

echo "🔍 查找 wandb 相关进程..."
echo "=========================================="

# 查找所有包含 wandb 的进程，但排除:
# 1. run_all.py (用户的主程序)
# 2. 本脚本自身 (kill_wandb.sh)
# 3. grep 命令自身

PIDS=$(ps aux | grep -i wandb | grep -v grep | grep -v run_all | grep -v kill_wandb | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "✅ 没有找到需要终止的 wandb 进程"
    exit 0
fi

# 显示将要终止的进程
echo "找到以下 wandb 相关进程:"
echo ""
ps aux | grep -i wandb | grep -v grep | grep -v run_all | grep -v kill_wandb | awk '{printf "  PID: %-8s USER: %-12s CMD: %s\n", $2, $1, substr($0, index($0,$11))}'
echo ""

# 询问确认
read -p "⚠️  确定要终止这些进程吗? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消操作"
    exit 1
fi

# 终止进程
echo ""
echo "🔨 正在终止进程..."
for PID in $PIDS; do
    # 检查进程是否仍然存在
    if ps -p $PID > /dev/null 2>&1; then
        # 获取进程命令
        CMD=$(ps -p $PID -o command=)
        echo "  终止 PID $PID: $CMD"
        kill $PID 2>/dev/null
        
        # 等待1秒，如果进程还在就强制kill
        sleep 1
        if ps -p $PID > /dev/null 2>&1; then
            echo "  强制终止 PID $PID"
            kill -9 $PID 2>/dev/null
        fi
    fi
done

echo ""
echo "✅ 完成！"
echo ""

# 验证是否还有 wandb 进程
REMAINING=$(ps aux | grep -i wandb | grep -v grep | grep -v run_all | grep -v kill_wandb)
if [ -z "$REMAINING" ]; then
    echo "✅ 所有 wandb 进程已终止"
else
    echo "⚠️  以下进程仍在运行:"
    echo "$REMAINING"
fi

# 检查 run_all.py 是否还在运行
RUN_ALL=$(ps aux | grep run_all.py | grep -v grep)
if [ ! -z "$RUN_ALL" ]; then
    echo ""
    echo "✅ run_all.py 进程仍在运行 (已保护):"
    echo "$RUN_ALL" | awk '{printf "  PID: %-8s CMD: %s\n", $2, substr($0, index($0,$11))}'
fi
