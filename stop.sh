#!/bin/bash
# 停止所有 run_all.py 相关进程

echo "正在查找 run_all.py 相关进程..."
pids=$(ps aux | grep "[r]un_all.py" | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "✓ 没有找到正在运行的 run_all.py 进程"
else
    echo "找到以下进程："
    ps aux | grep "[r]un_all.py"
    echo ""
    echo "正在停止进程..."
    for pid in $pids; do
        echo "  终止进程 $pid"
        kill -9 $pid 2>/dev/null
    done
    echo "✓ 所有进程已停止"
fi

# 同时清理可能卡住的 python 进程
echo ""
echo "检查其他可能卡住的 Python 进程..."
stuck_pids=$(ps aux | grep "[p]ython.*config" | awk '{print $2}')
if [ ! -z "$stuck_pids" ]; then
    echo "找到卡住的进程："
    ps aux | grep "[p]ython.*config"
    echo ""
    for pid in $stuck_pids; do
        echo "  终止进程 $pid"
        kill -9 $pid 2>/dev/null
    done
    echo "✓ 清理完成"
else
    echo "✓ 没有其他卡住的进程"
fi

echo ""
echo "完成！"
