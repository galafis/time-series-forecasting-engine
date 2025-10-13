"""
ARIMA-based forecasting model.

This module implements ARIMA (AutoRegressive Integrated Moving Average) models
with automatic parameter selection using auto_arima.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False
import warnings

from .base_forecaster import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA forecasting model with automatic parameter selection.
    
    This class implements ARIMA and SARIMA models with support for
    automatic order selection, seasonal components, and exogenous variables.
    """
    
    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto_select: bool = True,
        **kwargs
    ):
        """
        Initialize ARIMA forecaster.
        
        Parameters
        ----------
        order : tuple, optional
            ARIMA order (p, d, q). If None and auto_select=True, will be determined automatically
        seasonal_order : tuple, optional
            Seasonal ARIMA order (P, D, Q, s)
        auto_select : bool
            Whether to automatically select optimal parameters
        **kwargs : dict
            Additional parameters for auto_arima or SARIMAX
        """
        super().__init__(name="ARIMAForecaster")
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_select = auto_select
        self.kwargs = kwargs
        self.residuals_ = None
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None, **kwargs) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to time series data.
        
        Parameters
        ----------
        y : pd.Series
            Target time series
        X : pd.DataFrame, optional
            Exogenous variables
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        self : ARIMAForecaster
            Fitted model instance
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            if self.auto_select and self.order is None:
                # Automatic parameter selection
                if HAS_PMDARIMA:
                    auto_model = auto_arima(
                        y,
                        X=X,
                        seasonal=self.seasonal_order is not None,
                        m=self.seasonal_order[3] if self.seasonal_order else 1,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        trace=False,
                        **self.kwargs
                    )
                    self.order = auto_model.order
                    if self.seasonal_order is None and hasattr(auto_model, 'seasonal_order'):
                        self.seasonal_order = auto_model.seasonal_order
                else:
                    # Default order if pmdarima not available
                    self.order = (1, 1, 1)
                    
            # Fit SARIMAX model
            self.model = SARIMAX(
                y,
                exog=X,
                order=self.order,
                seasonal_order=self.seasonal_order if self.seasonal_order else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted_model_ = self.model.fit(disp=False)
            self.residuals_ = self.fitted_model_.resid
            self.is_fitted = True
            
            # Store training metrics
            self.training_history = {
                'aic': self.fitted_model_.aic,
                'bic': self.fitted_model_.bic,
                'order': self.order,
                'seasonal_order': self.seasonal_order
            }
            
        return self
    
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """
        Generate point forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        X : pd.DataFrame, optional
            Future exogenous variables
        **kwargs : dict
            Additional prediction parameters
            
        Returns
        -------
        predictions : pd.Series
            Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        forecast = self.fitted_model_.forecast(steps=steps, exog=X)
        return pd.Series(forecast, name='forecast')
    
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
            Confidence level (default: 0.95)
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        forecast_result = self.fitted_model_.get_forecast(steps=steps, exog=X)
        predictions = pd.Series(forecast_result.predicted_mean, name='forecast')
        
        conf_int = forecast_result.conf_int(alpha=1-confidence)
        lower_bound = pd.Series(conf_int.iloc[:, 0].values, name='lower')
        upper_bound = pd.Series(conf_int.iloc[:, 1].values, name='upper')
        
        return predictions, lower_bound, upper_bound
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'auto_select': self.auto_select
        })
        return params

