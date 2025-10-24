"""
Tree-based models for time series forecasting
Implements RandomForest and XGBoost regressors with feature importance extraction
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from typing import Dict, Optional


class TreeForecaster:
    """Wrapper for tree-based forecasting models"""
    
    def __init__(
        self,
        model_type: str = 'rf',
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        learning_rate: float = 0.1,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize tree-based forecaster
        
        Args:
            model_type: 'rf' for RandomForest or 'xgb' for XGBoost
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate (XGBoost only)
            n_jobs: Number of parallel jobs
            random_state: Random seed
            **kwargs: Additional model parameters
        """
        self.model_type = model_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = None
        self.feature_names_ = None
    
    def _create_model(self, n_targets: int = 1):
        """Create the underlying model"""
        if self.model_type == 'rf':
            base_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **self.kwargs
            )
        elif self.model_type == 'xgb':
            base_model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth if self.max_depth else 6,
                learning_rate=self.learning_rate,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Use 'rf' or 'xgb'")
        
        # Use MultiOutputRegressor for multiple targets
        if n_targets > 1:
            return MultiOutputRegressor(base_model, n_jobs=1)  # Parallel at base level
        else:
            return base_model
    
    def fit(self, X, y):
        """
        Fit the model
        
        Args:
            X: Feature matrix
            y: Target values (can be 1D or 2D for multiple targets)
        """
        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        
        # Determine number of targets
        if len(y.shape) == 1:
            n_targets = 1
        else:
            n_targets = y.shape[1]
        
        # Create and fit model
        self.model = self._create_model(n_targets)
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X):
        """
        Generate predictions
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Get importance from base model or average across multiple outputs
        if isinstance(self.model, MultiOutputRegressor):
            # Average importance across all output models
            importances = []
            for estimator in self.model.estimators_:
                importances.append(estimator.feature_importances_)
            importance_scores = np.mean(importances, axis=0)
        else:
            importance_scores = self.model.feature_importances_
        
        # Create dictionary with feature names
        if self.feature_names_:
            importance_dict = dict(zip(self.feature_names_, importance_scores))
        else:
            importance_dict = {f'feature_{i}': score 
                             for i, score in enumerate(importance_scores)}
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True))
        
        return importance_dict


class RandomForestForecaster(TreeForecaster):
    """Random Forest forecaster"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 n_jobs: int = -1, random_state: int = 42, **kwargs):
        super().__init__(
            model_type='rf',
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )


class XGBoostForecaster(TreeForecaster):
    """XGBoost forecaster"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, n_jobs: int = -1, 
                 random_state: int = 42, **kwargs):
        super().__init__(
            model_type='xgb',
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
