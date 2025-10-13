"""
Ensemble forecasting model.

This module implements ensemble methods that combine multiple forecasting
models to improve prediction accuracy and robustness.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from .base_forecaster import BaseForecaster


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble forecasting model combining multiple base forecasters.
    
    Supports different aggregation methods including simple averaging,
    weighted averaging, and median.
    """
    
    def __init__(
        self,
        forecasters: List[BaseForecaster],
        method: str = 'average',
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble forecaster.
        
        Parameters
        ----------
        forecasters : list of BaseForecaster
            List of base forecasting models
        method : str
            Aggregation method: 'average', 'weighted', or 'median'
        weights : list of float, optional
            Weights for weighted average (must sum to 1)
        """
        super().__init__(name="EnsembleForecaster")
        self.forecasters = forecasters
        self.method = method
        self.weights = weights
        
        if method == 'weighted' and weights is None:
            raise ValueError("Weights must be provided for weighted ensemble")
        
        if weights is not None:
            if len(weights) != len(forecasters):
                raise ValueError("Number of weights must match number of forecasters")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1")
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None, **kwargs) -> 'EnsembleForecaster':
        """
        Fit all base forecasters.
        
        Parameters
        ----------
        y : pd.Series
            Target time series
        X : pd.DataFrame, optional
            Exogenous features
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        self : EnsembleForecaster
            Fitted ensemble instance
        """
        for forecaster in self.forecasters:
            forecaster.fit(y, X, **kwargs)
        
        self.is_fitted = True
        
        # Calculate ensemble residuals
        all_predictions = []
        for forecaster in self.forecasters:
            if hasattr(forecaster, 'residuals_') and forecaster.residuals_ is not None:
                # Use in-sample predictions if available
                pred = y.values[-len(forecaster.residuals_):] - forecaster.residuals_
                all_predictions.append(pred)
        
        if all_predictions:
            ensemble_pred = self._aggregate_predictions(all_predictions)
            self.residuals_ = y.values[-len(ensemble_pred):] - ensemble_pred
        
        return self
    
    def _aggregate_predictions(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate predictions from multiple models.
        
        Parameters
        ----------
        predictions : list of np.ndarray
            Predictions from each base forecaster
            
        Returns
        -------
        aggregated : np.ndarray
            Aggregated predictions
        """
        predictions_array = np.array(predictions)
        
        if self.method == 'average':
            return np.mean(predictions_array, axis=0)
        elif self.method == 'weighted':
            return np.average(predictions_array, axis=0, weights=self.weights)
        elif self.method == 'median':
            return np.median(predictions_array, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
    
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """
        Generate ensemble forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        X : pd.DataFrame, optional
            Future exogenous features
        **kwargs : dict
            Additional prediction parameters
            
        Returns
        -------
        predictions : pd.Series
            Ensemble forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from all forecasters
        all_predictions = []
        for forecaster in self.forecasters:
            pred = forecaster.predict(steps, X, **kwargs)
            all_predictions.append(pred.values)
        
        # Aggregate predictions
        ensemble_pred = self._aggregate_predictions(all_predictions)
        
        return pd.Series(ensemble_pred, name='forecast')
    
    def predict_with_intervals(
        self,
        steps: int,
        confidence: float = 0.95,
        X: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate ensemble forecasts with prediction intervals.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        confidence : float
            Confidence level
        X : pd.DataFrame, optional
            Future exogenous features
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions and intervals from all forecasters
        all_predictions = []
        all_lower = []
        all_upper = []
        
        for forecaster in self.forecasters:
            pred, lower, upper = forecaster.predict_with_intervals(
                steps, confidence, X, **kwargs
            )
            all_predictions.append(pred.values)
            all_lower.append(lower.values)
            all_upper.append(upper.values)
        
        # Aggregate predictions
        ensemble_pred = self._aggregate_predictions(all_predictions)
        
        # For intervals, use the range of predictions
        predictions_array = np.array(all_predictions)
        lower_bound = np.min(predictions_array, axis=0)
        upper_bound = np.max(predictions_array, axis=0)
        
        return (
            pd.Series(ensemble_pred, name='forecast'),
            pd.Series(lower_bound, name='lower'),
            pd.Series(upper_bound, name='upper')
        )
    
    def get_individual_predictions(
        self,
        steps: int,
        X: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, pd.Series]:
        """
        Get predictions from each individual forecaster.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        X : pd.DataFrame, optional
            Future exogenous features
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        predictions : dict
            Dictionary mapping forecaster names to predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        for forecaster in self.forecasters:
            pred = forecaster.predict(steps, X, **kwargs)
            predictions[forecaster.name] = pred
        
        return predictions
    
    def get_params(self) -> dict:
        """Get ensemble parameters."""
        params = super().get_params()
        params.update({
            'method': self.method,
            'weights': self.weights,
            'num_forecasters': len(self.forecasters),
            'forecaster_names': [f.name for f in self.forecasters]
        })
        return params

