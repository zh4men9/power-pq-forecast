"""
LSTM model for time series forecasting
Implements sequence-to-one LSTM for P and Q prediction
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple
import platform
import time
from tqdm import tqdm


def get_optimal_device():
    """
    自动检测并返回最优计算设备
    
    优先级:
    1. CUDA GPU (NVIDIA)
    2. MPS (Apple Silicon Mac)
    3. CPU
    
    Returns:
        torch.device: 最优设备
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ 使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        system = platform.system()
        machine = platform.machine()
        print(f"✓ 使用 Apple Metal (MPS) 加速")
        print(f"  系统: {system} {machine}")
    else:
        device = torch.device('cpu')
        print(f"⚠ 使用 CPU (建议使用GPU以加快训练)")
    
    return device


class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting"""
    
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features (P and Q)
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            output_size: Number of outputs (2 for P and Q)
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Use the last time step
        last_out = lstm_out[:, -1, :]
        
        # Linear layer
        output = self.fc(last_out)
        
        return output


class LSTMForecaster:
    """LSTM forecaster wrapper"""
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize LSTM forecaster
        
        Args:
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Set device with auto-detection
        if device is None:
            self.device = get_optimal_device()
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler_mean_ = None
        self.scaler_std_ = None
    
    def _prepare_data(self, X, y=None):
        """Prepare data for PyTorch"""
        # Normalize X
        if self.scaler_mean_ is None:
            # Fit scaler on training data
            # X shape: (n_samples, seq_len, n_features)
            self.scaler_mean_ = np.mean(X, axis=(0, 1))
            self.scaler_std_ = np.std(X, axis=(0, 1)) + 1e-8
        
        X_scaled = (X - self.scaler_mean_) / self.scaler_std_
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            return X_tensor, y_tensor
        else:
            return X_tensor
    
    def fit(self, X, y):
        """
        Fit the LSTM model
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
            y: Target values of shape (n_samples, n_targets)
        """
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y)
        
        # Determine dimensions
        input_size = X.shape[2]
        output_size = y.shape[1] if len(y.shape) > 1 else 1
        
        # Create model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=output_size,
            dropout=self.dropout
        ).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop with progress bar
        self.model.train()
        start_time = time.time()
        
        # Use tqdm for progress visualization
        epoch_pbar = tqdm(range(self.epochs), desc='LSTM训练', unit='epoch', 
                         ncols=100, colour='green')
        
        for epoch in epoch_pbar:
            epoch_loss = 0
            batch_count = 0
            
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'device': str(self.device).upper()
            })
            
            # Estimate time
            if epoch == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed * self.epochs
                epoch_pbar.set_description(
                    f'LSTM训练 (预计{estimated_total:.0f}秒)'
                )
        
        total_time = time.time() - start_time
        print(f"✓ LSTM训练完成，用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        
        return self
    
    def predict(self, X):
        """
        Generate predictions
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
        
        Returns:
            Predictions of shape (n_samples, n_targets)
        """
        self.model.eval()
        
        X_tensor = self._prepare_data(X)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
