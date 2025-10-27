# Kill W&B 进程脚本使用说明

## 📝 功能说明

安全地终止所有 wandb 相关进程（如 sweep agent、训练进程等），但**保护**你的 `run_all.py` 主程序不被终止。

---

## 🚀 使用方法

### 方法1: Bash脚本（推荐，无依赖）

```bash
# 基本使用
bash kill_wandb.sh

# 或直接执行
./kill_wandb.sh
```

**特点**：
- ✅ 无需额外依赖
- ✅ 交互式确认
- ✅ 自动保护 run_all.py
- ✅ 显示详细信息

---

### 方法2: Python脚本（功能更强大）

```bash
# 基本使用（交互式）
python kill_wandb.py

# 跳过确认
python kill_wandb.py --yes

# 强制终止无响应的进程
python kill_wandb.py --force

# 组合使用
python kill_wandb.py --yes --force
```

**特点**：
- ✅ 更准确的进程识别（使用 psutil）
- ✅ 优雅终止 + 强制终止选项
- ✅ 详细的执行报告
- ✅ 自动验证结果

**依赖**：需要安装 `psutil`
```bash
pip install psutil
```

---

## 📋 使用场景

### 场景1: Sweep实验卡住了
```bash
# 终止所有 wandb sweep agent
./kill_wandb.sh
```

### 场景2: 有多个sweep在后台运行，想全部停止
```bash
python kill_wandb.py --yes
```

### 场景3: 进程无响应，需要强制终止
```bash
python kill_wandb.py --force
```

### 场景4: 误启动了多个训练任务
```bash
# 先查看
ps aux | grep wandb

# 再终止
./kill_wandb.sh
```

---

## 🛡️ 安全保护

两个脚本都会**自动保护**以下进程，不会被终止：

1. ✅ `run_all.py` - 你的主训练程序
2. ✅ `kill_wandb.sh` / `kill_wandb.py` - 脚本自身
3. ✅ 系统进程和其他非wandb进程

---

## 📊 输出示例

### Bash脚本输出:
```
🔍 查找 wandb 相关进程...
==========================================
找到以下 wandb 相关进程:

  PID: 12345    USER: zh4men9      CMD: python train_sweep.py
  PID: 12346    USER: zh4men9      CMD: wandb agent abc123

⚠️  确定要终止这些进程吗? (y/N): y

🔨 正在终止进程...
  终止 PID 12345: python train_sweep.py
  终止 PID 12346: wandb agent abc123

✅ 完成！

✅ 所有 wandb 进程已终止

✅ run_all.py 进程仍在运行 (已保护):
  PID: 12340   CMD: python run_all.py --config config.yaml
```

### Python脚本输出:
```
🔍 查找 wandb 相关进程...
============================================================

找到 2 个 wandb 相关进程:

  PID: 12345    USER: zh4men9
  CMD: python train_sweep.py...

  PID: 12346    USER: zh4men9
  CMD: wandb agent abc123...

🛡️  检测到 run_all.py 正在运行 (将被保护):
  PID: 12340
  CMD: python run_all.py --config config.yaml

⚠️  确定要终止这些进程吗? (y/N): y

🔨 正在终止进程...
  ✅ 已终止 PID 12345: train_sweep.py
  ✅ 已终止 PID 12346: wandb

============================================================
📊 执行结果:
  ✅ 成功终止: 2 个进程

🔍 验证结果...
✅ 所有 wandb 进程已终止

✅ run_all.py 仍在运行 (已保护):
  PID: 12340
```

---

## 🔧 故障排查

### 问题1: "Permission denied"

**原因**: 脚本没有执行权限

**解决**:
```bash
chmod +x kill_wandb.sh kill_wandb.py
```

### 问题2: Python脚本报错 "No module named 'psutil'"

**原因**: 缺少依赖

**解决**:
```bash
pip install psutil
```

### 问题3: 进程终止失败

**解决**:
```bash
# 使用强制模式
python kill_wandb.py --force

# 或手动强制终止
kill -9 <PID>
```

### 问题4: 误终止了 run_all.py

**不会发生**: 脚本会自动排除包含 "run_all" 的进程，双重保护。

但如果真的需要终止 run_all.py:
```bash
# 手动查找
ps aux | grep run_all

# 手动终止
kill <PID>
```

---

## 💡 提示

1. **优先使用 Bash脚本** - 简单、无依赖、交互友好
2. **Python脚本更适合自动化** - 可以集成到其他脚本中
3. **养成好习惯** - sweep运行前检查是否有旧进程:
   ```bash
   ps aux | grep wandb
   ```
4. **监控资源** - 大量sweep可能占用大量内存:
   ```bash
   htop  # 或 top
   ```

---

## 🔗 相关命令

```bash
# 查看所有 wandb 进程
ps aux | grep wandb

# 查看 run_all.py 进程
ps aux | grep run_all

# 查看系统资源占用
htop

# 查看 wandb 日志
ls -lh wandb/

# 清理 wandb 缓存
wandb artifact cache cleanup
```

---

## 📚 参考资料

- [W&B Agent 文档](https://docs.wandb.ai/guides/sweeps/agents)
- [Linux进程管理](https://man7.org/linux/man-pages/man1/kill.1.html)
- [Python psutil文档](https://psutil.readthedocs.io/)
