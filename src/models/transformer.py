"""
Transformer model for time series forecasting
Implements basic Transformer with positional encoding for P and Q prediction
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
from typing import Optional
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


class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine functions"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Dimension of model
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """Transformer model for time series forecasting"""
    
    def __init__(
        self,
        input_size: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_size: int = 2
    ):
        """
        Initialize Transformer model
        
        Args:
            input_size: Number of input features
            d_model: Dimension of model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            output_size: Number of outputs
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_size = input_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_size)
    
    def forward(self, src):
        """
        Forward pass for single-step prediction
        
        Args:
            src: Source sequence of shape (batch, seq_len, input_size)
        
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # Project input to d_model dimension
        src = self.input_proj(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # For single-step prediction, use last position as decoder input
        # Create a simple target of shape (batch, 1, d_model)
        tgt = src[:, -1:, :]
        
        # Transformer forward
        output = self.transformer(src, tgt)
        
        # Take the last output and project
        output = self.output_proj(output[:, -1, :])
        
        return output


class TransformerForecaster:
    """Transformer forecaster wrapper"""
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize Transformer forecaster
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            device: Device to use
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
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
        Fit the Transformer model
        
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
        self.model = TransformerModel(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            output_size=output_size
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
        epoch_pbar = tqdm(range(self.epochs), desc='Transformer训练', unit='epoch',
                         ncols=100, colour='blue')
        
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
                    f'Transformer训练 (预计{estimated_total:.0f}秒)'
                )
        
        total_time = time.time() - start_time
        print(f"✓ Transformer训练完成，用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        
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
