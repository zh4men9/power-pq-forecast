#!/bin/bash
# W&B Sweep自动调参脚本
# 使用方法: ./run_sweep.sh

set -e

echo "🚀 W&B Sweep - Transformer超参数优化"
echo "目标: ACC_10 > 80% (horizon=1)"
echo ""

# 1. 检查wandb是否安装
echo "📦 检查依赖..."
if ! python -c "import wandb" 2>/dev/null; then
    echo "❌ wandb未安装，正在安装..."
    pip install wandb
else
    echo "✅ wandb已安装"
fi

# 2. 登录wandb（如果需要）
echo ""
echo "🔑 检查wandb登录状态..."
if ! wandb login --relogin 2>/dev/null; then
    echo "⚠️  请先登录wandb:"
    echo "   wandb login"
    echo "   或者设置环境变量: export WANDB_API_KEY=your_api_key"
    exit 1
fi

# 3. 初始化sweep
echo ""
echo "🎯 初始化sweep..."
SWEEP_ID=$(wandb sweep sweep_config.yaml --project transformer-tuning 2>&1 | grep "wandb agent" | awk '{print $3}')

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Sweep初始化失败"
    exit 1
fi

echo "✅ Sweep已创建: $SWEEP_ID"
echo ""

# 4. 启动sweep agent
echo "🤖 启动sweep agent (后台运行)..."
echo "   可以启动多个agent并行搜索"
echo "   命令: wandb agent $SWEEP_ID"
echo ""

# 选择运行模式
read -p "选择运行模式 [1=前台运行, 2=后台运行, 3=只显示命令]: " mode

case $mode in
    1)
        echo "▶️  前台运行 (Ctrl+C可停止)..."
        wandb agent $SWEEP_ID
        ;;
    2)
        echo "▶️  后台运行..."
        nohup wandb agent $SWEEP_ID > sweep_output.log 2>&1 &
        PID=$!
        echo "✅ Agent已启动 (PID: $PID)"
        echo "   日志文件: sweep_output.log"
        echo "   停止命令: kill $PID"
        echo "   查看日志: tail -f sweep_output.log"
        ;;
    3)
        echo "📋 手动运行命令:"
        echo "   wandb agent $SWEEP_ID"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "🌐 查看实时结果:"
echo "   https://wandb.ai/your-username/transformer-tuning/sweeps"
echo ""
echo "💡 提示:"
echo "   - 可以在wandb网页端实时查看所有实验结果"
echo "   - 自动排序找出最佳超参数组合"
echo "   - 可视化参数对ACC_10的影响"
echo "   - 支持提前停止效果差的实验"
