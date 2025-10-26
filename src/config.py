"""
Configuration module for loading and validating config.yaml
Loads project configuration including data paths, model hyperparameters, 
validation settings, and plotting options.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List


class Config:
    """Configuration class for the power PQ forecast project"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Load and validate configuration from YAML file
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self._validate()
    
    def _validate(self):
        """Validate configuration consistency"""
        # Parse and validate horizons
        horizons_raw = self.config.get('evaluation', {}).get('horizons', [])
        
        # Support both string (comma-separated) and list formats
        if isinstance(horizons_raw, str):
            # Parse comma-separated string: "1,2,3,4,5,6,7,8,9,10,11,12"
            try:
                horizons = [int(h.strip()) for h in horizons_raw.split(',')]
                self.config['evaluation']['horizons'] = horizons
            except ValueError:
                raise ValueError("horizons string must contain comma-separated integers")
        elif isinstance(horizons_raw, list):
            horizons = horizons_raw
        else:
            raise ValueError("horizons must be a string (comma-separated) or list of integers")
        
        # Validate horizons are positive integers
        if not horizons or not all(isinstance(h, int) and h > 0 for h in horizons):
            raise ValueError("horizons must contain positive integers")
        
        # Validate test_window and n_splits
        test_window = self.config.get('evaluation', {}).get('test_window', 0)
        n_splits = self.config.get('evaluation', {}).get('n_splits', 0)
        
        if test_window <= 0:
            raise ValueError("test_window must be positive")
        if n_splits <= 0:
            raise ValueError("n_splits must be positive")
        
        # Validate frequency if provided
        freq = self.config.get('data', {}).get('freq')
        if freq:
            valid_freqs = ['H', 'D', 'W', 'M', 'T', '15T', '30T']
            if freq not in valid_freqs:
                print(f"Warning: frequency '{freq}' may not be standard. Common: {valid_freqs}")
    
    def get(self, *keys, default=None):
        """
        Get nested configuration value
        
        Args:
            *keys: Nested keys to access
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def __getitem__(self, key):
        """Dictionary-style access"""
        return self.config[key]
    
    def __contains__(self, key):
        """Check if key exists"""
        return key in self.config


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Config object
    """
    return Config(config_path)
