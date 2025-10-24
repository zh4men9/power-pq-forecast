"""
Cross-validation module for time series
Implements rolling origin (expanding window) cross-validation
CRITICAL: Ensures training data is always in the past, test data in the future
"""
import numpy as np
from typing import Iterator, Tuple


def rolling_origin_split(
    n_samples: int,
    test_window: int,
    n_splits: int,
    gap: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate rolling origin cross-validation splits
    
    This implements the time series cross-validation method where:
    - Training set grows with each fold (expanding window)
    - Test set has fixed size and moves forward
    - Training data is always before test data (no future information leakage)
    
    This follows the methodology described in:
    "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos
    https://otexts.com/fpp3/tscv.html
    
    Example with n_samples=100, test_window=10, n_splits=3:
    Fold 0: train=[0:60],  test=[60:70]
    Fold 1: train=[0:70],  test=[70:80]
    Fold 2: train=[0:80],  test=[80:90]
    
    Args:
        n_samples: Total number of samples
        test_window: Size of each test window
        n_splits: Number of splits to generate
        gap: Number of samples to skip between train and test (default 0)
    
    Yields:
        Tuple of (train_indices, test_indices)
    
    Raises:
        ValueError: If parameters are invalid
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if test_window <= 0:
        raise ValueError("test_window must be positive")
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    
    # Calculate minimum samples needed
    min_train = test_window  # At least one test window for training
    min_samples = min_train + gap + test_window * n_splits
    
    if n_samples < min_samples:
        raise ValueError(
            f"Not enough samples. Need at least {min_samples} samples "
            f"for test_window={test_window}, n_splits={n_splits}, gap={gap}, "
            f"but got {n_samples} samples"
        )
    
    # Calculate the size of the first training set
    first_train_size = n_samples - test_window * n_splits - gap
    
    for i in range(n_splits):
        # Training set grows with each fold
        train_end = first_train_size + i * test_window
        train_indices = np.arange(0, train_end)
        
        # Test set starts after gap
        test_start = train_end + gap
        test_end = test_start + test_window
        test_indices = np.arange(test_start, test_end)
        
        yield train_indices, test_indices


class TimeSeriesSplit:
    """
    Time series cross-validation using rolling origin method
    
    Compatible with scikit-learn's cross-validation interface while
    implementing proper time series validation:
    - Training always precedes test data chronologically
    - No random shuffling
    - Expanding window (training set grows)
    - Fixed size test windows
    
    Reference:
    sklearn.model_selection.TimeSeriesSplit uses a similar approach but with
    more flexibility. Our implementation focuses on the rolling origin method
    with fixed test window sizes.
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
    """
    
    def __init__(self, test_window: int, n_splits: int, gap: int = 0):
        """
        Initialize TimeSeriesSplit
        
        Args:
            test_window: Size of each test window
            n_splits: Number of splits
            gap: Gap between train and test (default 0)
        """
        self.test_window = test_window
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices
        
        Args:
            X: Array-like with shape (n_samples, n_features) or just length
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
        
        return rolling_origin_split(
            n_samples=n_samples,
            test_window=self.test_window,
            n_splits=self.n_splits,
            gap=self.gap
        )
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits"""
        return self.n_splits
