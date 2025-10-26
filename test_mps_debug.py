"""
Debug script to find MPS segmentation fault issue
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO)

def test_step_by_step():
    """Test each step to find where the segfault occurs"""
    
    print("=" * 60)
    print("Step 1: Create sample data")
    X = np.random.randn(3060, 24, 7)
    y = np.random.randn(3060, 2)
    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    
    print("\nStep 2: Normalize data")
    scaler_mean = np.mean(X, axis=(0, 1))
    scaler_std = np.std(X, axis=(0, 1)) + 1e-8
    X_scaled = (X - scaler_mean) / scaler_std
    print(f"  X_scaled dtype: {X_scaled.dtype}")
    
    print("\nStep 3: Convert to CPU tensors")
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)
    print(f"  X_tensor: {X_tensor.shape}, {X_tensor.dtype}, device: {X_tensor.device}")
    print(f"  y_tensor: {y_tensor.shape}, {y_tensor.dtype}, device: {y_tensor.device}")
    
    print("\nStep 4: Create TensorDataset")
    dataset = TensorDataset(X_tensor, y_tensor)
    print(f"  Dataset length: {len(dataset)}")
    
    print("\nStep 5: Create DataLoader")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"  DataLoader batches: {len(dataloader)}")
    
    print("\nStep 6: Get device")
    device = torch.device('mps')
    print(f"  Device: {device}")
    
    print("\nStep 7: Create simple model")
    model = nn.LSTM(input_size=7, hidden_size=64, num_layers=2, batch_first=True)
    print(f"  Model created")
    
    print("\nStep 8: Move model to MPS")
    model = model.to(device)
    print(f"  Model moved to MPS")
    
    print("\nStep 9: Iterate through first batch")
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        print(f"  Batch {batch_idx}: X={batch_X.shape}, y={batch_y.shape}")
        print(f"    X device: {batch_X.device}, y device: {batch_y.device}")
        
        print(f"\nStep 10: Move batch to MPS")
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        print(f"    After .to(device): X device={batch_X.device}, y device={batch_y.device}")
        
        print(f"\nStep 11: Forward pass")
        with torch.no_grad():
            output, _ = model(batch_X)
            print(f"    Output shape: {output.shape}")
        
        print("\n✓ First batch completed successfully!")
        break
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_step_by_step()
