"""
LSTM model for time series forecasting
Implements sequence-to-one LSTM for P and Q prediction
"""
import os
# Fix OpenMP threading issue that causes segfault with PyTorch 2.4.0
# Must be set before importing torch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# Set MPS fallback before importing torch to avoid segfaults
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple
import platform
import time
import logging
from tqdm import tqdm


def get_optimal_device(config_device: str = 'auto'):
    """
    根据配置返回计算设备
    
    Args:
        config_device: 设备配置 ('auto', 'cpu', 'mps', 'cuda')
    
    优先级 (当 config_device='auto'):
    1. CUDA GPU (NVIDIA)
    2. MPS (Apple Silicon Mac)
    3. CPU
    
    Returns:
        torch.device: 计算设备
    """
    import os
    
    # 如果指定了具体设备，直接使用
    if config_device == 'cpu':
        device = torch.device('cpu')
        logging.info(f"✓ 使用 CPU (配置指定)")
        return device
    
    if config_device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logging.info(f"✓ 使用 Apple Metal (MPS) 加速 (配置指定)")
            return device
        else:
            logging.warning(f"⚠️  MPS不可用，回退到CPU")
            device = torch.device('cpu')
            logging.info(f"✓ 使用 CPU")
            return device
    
    if config_device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info(f"✓ 使用 CUDA GPU: {torch.cuda.get_device_name(0)} (配置指定)")
            logging.info(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return device
        else:
            logging.warning(f"⚠️  CUDA不可用，回退到CPU")
            device = torch.device('cpu')
            logging.info(f"✓ 使用 CPU")
            return device
    
    # 自动检测 (config_device='auto' 或其他)
    # 检查是否通过环境变量禁用 MPS
    disable_mps = os.environ.get('DISABLE_MPS', '0') == '1'
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"✓ 使用 CUDA GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and not disable_mps:
        device = torch.device('mps')
        system = platform.system()
        machine = platform.machine()
        logging.info(f"✓ 使用 Apple Metal (MPS) 加速")
        logging.info(f"  系统: {system} {machine}")
    else:
        device = torch.device('cpu')
        if disable_mps:
            logging.info(f"ℹ 使用 CPU (MPS已被禁用)")
        else:
            logging.info(f"✓ 使用 CPU")
    
    return device


class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting
    
    Supports two modes:
    - Single output: predict one horizon at a time (output_size = n_targets)
    - Multiple output: predict all horizons at once (output_size = n_targets * n_horizons)
    """
    
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 2,
        dropout: float = 0.2,
        n_horizons: int = 1  # Number of forecast horizons (for multiple output)
    ):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features (P and Q)
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            output_size: Number of outputs per horizon (2 for P and Q)
            dropout: Dropout rate
            n_horizons: Number of forecast horizons (for multiple output strategy)
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_horizons = n_horizons
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer: output_size * n_horizons for multiple output
        self.fc = nn.Linear(hidden_size, output_size * n_horizons)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch, output_size * n_horizons)
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Use the last time step
        last_out = lstm_out[:, -1, :]
        
        # Linear layer
        output = self.fc(last_out)
        
        return output


class LSTMForecaster:
    """LSTM forecaster wrapper
    
    Supports two strategies for multi-step forecasting:
    - Direct: Train separate model for each horizon (n_horizons=1)
    - Multiple Output: Train one model for all horizons (n_horizons=N)
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        n_horizons: int = 1  # Number of horizons for multiple output strategy
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
            n_horizons: Number of forecast horizons (for multiple output strategy)
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_horizons = n_horizons
        
        # Set device from config or auto-detection
        if device is None:
            self.device = get_optimal_device('auto')
        else:
            self.device = get_optimal_device(device)
        
        self.model = None
        self.scaler_mean_ = None
        self.scaler_std_ = None
    
    def _prepare_data(self, X, y=None):
        """Prepare data for PyTorch"""
        logging.info(f"      准备数据: X shape={X.shape}, device={self.device}")
        
        # Check for NaN/Inf in input
        if np.any(np.isnan(X)):
            logging.error(f"      ❌ 输入X包含 {np.isnan(X).sum()} 个NaN值!")
        if np.any(np.isinf(X)):
            logging.error(f"      ❌ 输入X包含 {np.isinf(X).sum()} 个Inf值!")
        
        # Normalize X
        if self.scaler_mean_ is None:
            # Fit scaler on training data
            # X shape: (n_samples, seq_len, n_features)
            logging.info(f"      计算归一化参数...")
            self.scaler_mean_ = np.mean(X, axis=(0, 1))
            self.scaler_std_ = np.std(X, axis=(0, 1)) + 1e-8
            
            # Debug: print normalization params
            logging.info(f"      归一化均值: {self.scaler_mean_}")
            logging.info(f"      归一化标准差: {self.scaler_std_}")
            
            # Check for problematic values
            if np.any(np.isnan(self.scaler_mean_)):
                logging.error("      ❌ 均值包含NaN!")
            if np.any(np.isnan(self.scaler_std_)):
                logging.error("      ❌ 标准差包含NaN!")
            if np.any(self.scaler_std_ < 1e-6):
                logging.warning(f"      ⚠️  标准差过小: {self.scaler_std_[self.scaler_std_ < 1e-6]}")
        
        logging.info(f"      应用归一化...")
        X_scaled = (X - self.scaler_mean_) / self.scaler_std_
        
        # Check scaled data
        if np.any(np.isnan(X_scaled)):
            logging.error(f"      ❌ 归一化后包含 {np.isnan(X_scaled).sum()} 个NaN值!")
        if np.any(np.isinf(X_scaled)):
            logging.error(f"      ❌ 归一化后包含 {np.isinf(X_scaled).sum()} 个Inf值!")
        logging.info(f"      归一化后数据范围: [{np.min(X_scaled):.4f}, {np.max(X_scaled):.4f}]")
        
        logging.info(f"      转换为Tensor (保持在CPU)...")
        # Use torch.from_numpy() with explicit copy and float32 conversion
        # This is more stable than FloatTensor() with MPS
        X_scaled_float32 = X_scaled.astype(np.float32)
        X_tensor = torch.from_numpy(X_scaled_float32).clone()
        
        # Check tensor
        if torch.isnan(X_tensor).any():
            logging.error(f"      ❌ X Tensor包含 {torch.isnan(X_tensor).sum()} 个NaN值!")
        if torch.isinf(X_tensor).any():
            logging.error(f"      ❌ X Tensor包含 {torch.isinf(X_tensor).sum()} 个Inf值!")
        logging.info(f"      X Tensor范围: [{X_tensor.min():.4f}, {X_tensor.max():.4f}]")
        logging.info(f"      X Tensor创建完成 (CPU)")
        
        if y is not None:
            logging.info(f"      处理y数据: shape={y.shape}")
            
            # Check y for NaN/Inf
            if np.any(np.isnan(y)):
                logging.error(f"      ❌ 输入y包含 {np.isnan(y).sum()} 个NaN值!")
            if np.any(np.isinf(y)):
                logging.error(f"      ❌ 输入y包含 {np.isinf(y).sum()} 个Inf值!")
            logging.info(f"      y数据范围: [{np.min(y):.4f}, {np.max(y):.4f}]")
            
            y_float32 = y.astype(np.float32)
            y_tensor = torch.from_numpy(y_float32).clone()
            
            if torch.isnan(y_tensor).any():
                logging.error(f"      ❌ y Tensor包含 {torch.isnan(y_tensor).sum()} 个NaN值!")
            if torch.isinf(y_tensor).any():
                logging.error(f"      ❌ y Tensor包含 {torch.isinf(y_tensor).sum()} 个Inf值!")
            
            logging.info(f"      y Tensor创建完成 (CPU)")
            return X_tensor, y_tensor
        else:
            return X_tensor
    
    def fit(self, X, y):
        """
        Fit the LSTM model
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
            y: Target values of shape:
               - (n_samples, n_targets) for single horizon
               - (n_samples, n_targets * n_horizons) for multiple horizons
        """
        
        logging.info(f"      [1/6] 准备数据 (X shape: {X.shape}, y shape: {y.shape})...")
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y)
        
        logging.info(f"      [2/6] 确定模型维度...")
        # Determine dimensions
        input_size = X.shape[2]
        output_size_total = y.shape[1] if len(y.shape) > 1 else 1
        
        # Calculate output_size per horizon
        output_size = output_size_total // self.n_horizons
        
        logging.info(f"      模型配置: n_horizons={self.n_horizons}, output_size={output_size} (total={output_size_total})")
        
        logging.info(f"      [3/6] 创建LSTM模型 (input={input_size}, hidden={self.hidden_size}, output={output_size}, horizons={self.n_horizons})...")
        # Create model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=output_size,
            dropout=self.dropout,
            n_horizons=self.n_horizons
        ).to(self.device)
        
        logging.info(f"      [4/6] 模型已创建并移至设备: {self.device}")
        
        # Create dataset and dataloader
        logging.info(f"      [5/6] 创建数据加载器 (batch_size={self.batch_size})...")
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        logging.info(f"      数据加载器创建完成: {len(dataloader)} 个批次")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop with progress bar
        self.model.train()
        start_time = time.time()
        
        logging.info(f"      [6/6] 开始训练 ({self.epochs} epochs, lr={self.learning_rate})...")
        
        # Use tqdm for progress visualization
        # file=sys.stdout ensures progress bar works with logging
        import sys
        epoch_pbar = tqdm(range(self.epochs), desc='LSTM训练', unit='epoch', 
                         ncols=100, colour='green', leave=True, position=0, 
                         file=sys.stdout, dynamic_ncols=True)
        
        for epoch in epoch_pbar:
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                # Move batch to device (this is more efficient than moving all data at once)
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Debug first batch
                if epoch == 0 and batch_idx == 0:
                    logging.info(f"      🔍 第一个批次调试:")
                    logging.info(f"         batch_X shape: {batch_X.shape}")
                    logging.info(f"         batch_X 范围: [{batch_X.min():.4f}, {batch_X.max():.4f}]")
                    logging.info(f"         batch_X 是否有NaN: {torch.isnan(batch_X).any()}")
                    logging.info(f"         batch_y shape: {batch_y.shape}")
                    logging.info(f"         batch_y 范围: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
                    logging.info(f"         batch_y 是否有NaN: {torch.isnan(batch_y).any()}")
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Debug first batch output
                if epoch == 0 and batch_idx == 0:
                    logging.info(f"         模型输出 shape: {outputs.shape}")
                    logging.info(f"         模型输出 范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
                    logging.info(f"         模型输出 是否有NaN: {torch.isnan(outputs).any()}")
                
                loss = criterion(outputs, batch_y)
                
                # Debug first batch loss
                if epoch == 0 and batch_idx == 0:
                    logging.info(f"         Loss值: {loss.item()}")
                    logging.info(f"         Loss是否为NaN: {torch.isnan(loss)}")
                
                # Check for NaN loss
                if torch.isnan(loss):
                    if batch_idx < 3:  # Only log first 3 NaN warnings
                        logging.warning(f"      NaN loss detected at epoch {epoch}, batch {batch_idx}")
                    continue
                
                # Backward pass with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # Check if any batches completed successfully
            if batch_count == 0:
                logging.error(f"Epoch {epoch}: All batches failed with NaN. Stopping training.")
                break
            
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
        logging.info(f"      ✓ LSTM训练完成，用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        
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
        
        # Move to device for prediction
        X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
