"""
Base class for all forecasting models.

This module provides an abstract base class that defines the interface
for all forecasting models in the framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """
    Abstract base class for time series forecasting models.
    
    All forecasting models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str = "BaseForecaster"):
        """
        Initialize the base forecaster.
        
        Parameters
        ----------
        name : str
            Name of the forecaster model
        """
        self.name = name
        self.is_fitted = False
        self.model = None
        self.training_history = {}
        
    @abstractmethod
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None, **kwargs) -> 'BaseForecaster':
        """
        Fit the forecasting model to the training data.
        
        Parameters
        ----------
        y : pd.Series
            Target time series data
        X : pd.DataFrame, optional
            Exogenous variables
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        self : BaseForecaster
            Fitted forecaster instance
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """
        Generate forecasts for future time steps.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        X : pd.DataFrame, optional
            Future exogenous variables
        **kwargs : dict
            Additional parameters for prediction
            
        Returns
        -------
        predictions : pd.Series
            Forecasted values
        """
        pass
    
    def predict_with_intervals(
        self, 
        steps: int, 
        confidence: float = 0.95,
        X: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate forecasts with prediction intervals.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        confidence : float
            Confidence level for prediction intervals (default: 0.95)
        X : pd.DataFrame, optional
            Future exogenous variables
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        predictions : pd.Series
            Point forecasts
        lower_bound : pd.Series
            Lower bound of prediction interval
        upper_bound : pd.Series
            Upper bound of prediction interval
        """
        predictions = self.predict(steps, X, **kwargs)
        
        # Default implementation: use standard deviation from residuals
        if hasattr(self, 'residuals_') and self.residuals_ is not None:
            std = np.std(self.residuals_)
            z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
            margin = z_score * std
            
            lower_bound = predictions - margin
            upper_bound = predictions + margin
        else:
            # If no residuals available, return predictions without intervals
            lower_bound = predictions.copy()
            upper_bound = predictions.copy()
            
        return predictions, lower_bound, upper_bound
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters of the forecaster.
        
        Returns
        -------
        params : dict
            Dictionary of model parameters
        """
        return {
            'name': self.name,
            'is_fitted': self.is_fitted
        }
    
    def set_params(self, **params) -> 'BaseForecaster':
        """
        Set parameters of the forecaster.
        
        Parameters
        ----------
        **params : dict
            Parameters to set
            
        Returns
        -------
        self : BaseForecaster
            Forecaster instance with updated parameters
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def __repr__(self) -> str:
        """String representation of the forecaster."""
        return f"{self.name}(fitted={self.is_fitted})"

