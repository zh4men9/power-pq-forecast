"""
设备性能基准测试：对比 CPU vs MPS
测试 LSTM 和 Transformer 在不同设备上的训练速度
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import pandas as pd

# 导入模型
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster

def create_dummy_data(n_samples=3000, seq_length=24, n_features=9, batch_size=64):
    """创建模拟数据"""
    X = torch.randn(n_samples, seq_length, n_features)
    y = torch.randn(n_samples, 1)
    
    # 创建 DataLoader
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    return loader

def benchmark_model(model_type, device, n_epochs=1, batch_size=64, **model_kwargs):
    """
    基准测试单个模型
    
    Args:
        model_type: 'lstm' 或 'transformer'
        device: 'cpu', 'mps', 或 'cuda'
        n_epochs: 训练轮数
        batch_size: 批次大小
        **model_kwargs: 模型参数
    
    Returns:
        训练时间（秒）
    """
    # 创建数据
    loader = create_dummy_data(batch_size=batch_size)
    
    # 创建模型（直接用PyTorch的底层模型）
    if model_type == 'lstm':
        from src.models.lstm import LSTMModel
        model = LSTMModel(
            input_size=9,
            hidden_size=model_kwargs.get('hidden_size', 64),
            num_layers=model_kwargs.get('num_layers', 2),
            output_size=1,
            dropout=model_kwargs.get('dropout', 0.2)
        )
    else:  # transformer
        from src.models.transformer import TransformerModel
        model = TransformerModel(
            input_size=9,
            d_model=model_kwargs.get('d_model', 64),
            nhead=model_kwargs.get('nhead', 4),
            num_encoder_layers=model_kwargs.get('num_encoder_layers', 2),
            num_decoder_layers=model_kwargs.get('num_decoder_layers', 2),
            dim_feedforward=model_kwargs.get('dim_feedforward', 256),
            dropout=model_kwargs.get('dropout', 0.1),
            output_size=1
        )
    
    model = model.to(device)
    
    # 优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 预热（避免首次运行的初始化开销）
    model.train()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        break  # 只跑一个batch预热
    
    # 正式计时
    start_time = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    elapsed_time = time.time() - start_time
    
    return elapsed_time

def main():
    print("="*70)
    print("设备性能基准测试：CPU vs MPS")
    print("="*70)
    
    # 检查可用设备
    print("\n📱 可用设备:")
    print(f"  - CPU: ✓")
    print(f"  - MPS: {'✓' if torch.backends.mps.is_available() else '✗'}")
    print(f"  - CUDA: {'✓' if torch.cuda.is_available() else '✗'}")
    
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')
    
    # 测试配置
    configs = [
        {
            'name': 'LSTM (batch=32)',
            'model_type': 'lstm',
            'batch_size': 32,
            'model_kwargs': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.3
            }
        },
        {
            'name': 'LSTM (batch=64)',
            'model_type': 'lstm',
            'batch_size': 64,
            'model_kwargs': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.3
            }
        },
        {
            'name': 'Transformer (batch=32)',
            'model_type': 'transformer',
            'batch_size': 32,
            'model_kwargs': {
                'd_model': 64,
                'nhead': 4,
                'num_encoder_layers': 2,
                'num_decoder_layers': 2,
                'dim_feedforward': 256,
                'dropout': 0.2
            }
        },
        {
            'name': 'Transformer (batch=64)',
            'model_type': 'transformer',
            'batch_size': 64,
            'model_kwargs': {
                'd_model': 64,
                'nhead': 4,
                'num_encoder_layers': 2,
                'num_decoder_layers': 2,
                'dim_feedforward': 256,
                'dropout': 0.2
            }
        }
    ]
    
    # 运行基准测试
    results = []
    
    for config in configs:
        print("\n" + "="*70)
        print(f"🧪 测试: {config['name']}")
        print("="*70)
        
        for device in devices:
            print(f"\n▶ 设备: {device.upper()}")
            try:
                elapsed = benchmark_model(
                    model_type=config['model_type'],
                    device=device,
                    n_epochs=1,
                    batch_size=config['batch_size'],
                    **config['model_kwargs']
                )
                print(f"  ✓ 完成时间: {elapsed:.2f} 秒")
                
                results.append({
                    'Model': config['name'],
                    'Device': device.upper(),
                    'Time (s)': elapsed,
                    'Batch Size': config['batch_size']
                })
            except Exception as e:
                print(f"  ✗ 失败: {e}")
    
    # 生成报告
    print("\n" + "="*70)
    print("📊 基准测试结果汇总")
    print("="*70)
    
    df = pd.DataFrame(results)
    
    # 按模型分组显示
    for model_name in df['Model'].unique():
        model_df = df[df['Model'] == model_name]
        print(f"\n{model_name}:")
        print("-" * 50)
        
        for device in devices:
            device_data = model_df[model_df['Device'] == device.upper()]
            if not device_data.empty:
                time_val = device_data['Time (s)'].values[0]
                print(f"  {device.upper():8s}: {time_val:6.2f}秒")
        
        # 计算加速比
        if len(devices) > 1:
            cpu_time = model_df[model_df['Device'] == 'CPU']['Time (s)'].values
            if 'mps' in devices:
                mps_data = model_df[model_df['Device'] == 'MPS']
                if not mps_data.empty and len(cpu_time) > 0:
                    mps_time = mps_data['Time (s)'].values[0]
                    speedup = cpu_time[0] / mps_time
                    if speedup > 1:
                        print(f"  💡 MPS 加速: {speedup:.2f}x 快于CPU")
                    else:
                        print(f"  ⚠️  MPS 反而慢: CPU快 {1/speedup:.2f}x")
    
    # 保存结果
    output_file = 'benchmark_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 结果已保存至: {output_file}")
    
    # 结论
    print("\n" + "="*70)
    print("💡 结论与建议")
    print("="*70)
    
    if 'mps' in devices:
        cpu_avg = df[df['Device'] == 'CPU']['Time (s)'].mean()
        mps_avg = df[df['Device'] == 'MPS']['Time (s)'].mean()
        
        if mps_avg < cpu_avg:
            speedup = cpu_avg / mps_avg
            print(f"✅ MPS 平均快 {speedup:.2f}x，建议使用 MPS")
        else:
            slowdown = mps_avg / cpu_avg
            print(f"⚠️  MPS 平均慢 {slowdown:.2f}x，建议使用 CPU")
            print("\n可能原因:")
            print("  1. Batch size 太小 (< 64)，GPU启动开销大于计算收益")
            print("  2. 模型太小，CPU已经很快")
            print("  3. 数据传输开销 (CPU ↔ MPS) 占比大")
            print("\n建议:")
            print("  - 增大 batch_size 到 128-256")
            print("  - 或者直接用 CPU（对小模型更高效）")

if __name__ == '__main__':
    main()
