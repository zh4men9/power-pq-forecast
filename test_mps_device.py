"""测试MPS设备使用情况"""
import torch
import numpy as np
from tqdm import tqdm
import time

print('='*60)
print('Testing PyTorch with MPS...')
print('='*60)
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')
print()

# Create test data
print('Creating test data...')
x = torch.randn(1000, 24, 7).to(device)
y = torch.randn(1000, 2).to(device)
print(f'✓ Tensor on device: {x.device}')
print()

# Create LSTM model
print('Creating LSTM model...')
class TestLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(7, 64, 2, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(64, 2)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = TestLSTM().to(device)
print(f'✓ Model on device: {next(model.parameters()).device}')
print()

# Test forward pass
print('Testing forward pass...')
try:
    with torch.no_grad():
        result = model(x)
    print(f'✓ Forward pass successful')
    print(f'  Output shape: {result.shape}')
    print(f'  Output device: {result.device}')
except Exception as e:
    print(f'✗ Forward pass failed: {e}')
    import traceback
    traceback.print_exc()

print()

# Test training with tqdm
print('Testing training loop with tqdm...')
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

try:
    model.train()
    start_time = time.time()
    
    for epoch in tqdm(range(10), desc='Training', ncols=80):
        # Forward
        output = model(x[:100])
        loss = criterion(output, y[:100])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start_time
    print(f'✓ Training loop successful ({elapsed:.2f}s for 10 epochs)')
except Exception as e:
    print(f'✗ Training failed: {e}')
    import traceback
    traceback.print_exc()
