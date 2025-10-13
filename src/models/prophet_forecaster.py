"""
Prophet-based forecasting model.

This module implements Facebook Prophet for time series forecasting
with support for holidays, seasonality, and trend changepoints.
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from prophet import Prophet
import warnings

from .base_forecaster import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """
    Prophet forecasting model.
    
    Implements Facebook Prophet algorithm with automatic detection of
    trends, seasonality, and holiday effects.
    """
    
    def __init__(
        self,
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = 'additive',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        **kwargs
    ):
        """
        Initialize Prophet forecaster.
        
        Parameters
        ----------
        growth : str
            'linear' or 'logistic' growth
        changepoint_prior_scale : float
            Flexibility of trend changes
        seasonality_prior_scale : float
            Strength of seasonality
        seasonality_mode : str
            'additive' or 'multiplicative'
        yearly_seasonality : bool
            Include yearly seasonality
        weekly_seasonality : bool
            Include weekly seasonality
        daily_seasonality : bool
            Include daily seasonality
        **kwargs : dict
            Additional Prophet parameters
        """
        super().__init__(name="ProphetForecaster")
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.kwargs = kwargs
        self.regressors_ = []
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None, **kwargs) -> 'ProphetForecaster':
        """
        Fit Prophet model to time series data.
        
        Parameters
        ----------
        y : pd.Series
            Target time series with datetime index
        X : pd.DataFrame, optional
            Exogenous regressors
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        self : ProphetForecaster
            Fitted model instance
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Initialize Prophet model
            self.model = Prophet(
                growth=self.growth,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                **self.kwargs
            )
            
            # Prepare data in Prophet format
            df = pd.DataFrame({
                'ds': y.index,
                'y': y.values
            })
            
            # Add regressors if provided
            if X is not None:
                for col in X.columns:
                    self.model.add_regressor(col)
                    self.regressors_.append(col)
                    df[col] = X[col].values
            
            # Fit model
            self.model.fit(df)
            self.is_fitted = True
            
            # Calculate residuals
            predictions = self.model.predict(df)
            self.residuals_ = y.values - predictions['yhat'].values
            
            # Store training metrics
            self.training_history = {
                'regressors': self.regressors_,
                'changepoints': len(self.model.changepoints),
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
            Future exogenous regressors
        **kwargs : dict
            Additional prediction parameters
            
        Returns
        -------
        predictions : pd.Series
            Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps, include_history=False)
        
        # Add regressors if provided
        if X is not None and len(self.regressors_) > 0:
            for col in self.regressors_:
                if col in X.columns:
                    future[col] = X[col].values[:steps]
        
        # Generate forecast
        forecast = self.model.predict(future)
        predictions = pd.Series(
            forecast['yhat'].values,
            index=future['ds'],
            name='forecast'
        )
        
        return predictions
    
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
            Future exogenous regressors
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
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps, include_history=False)
        
        # Add regressors if provided
        if X is not None and len(self.regressors_) > 0:
            for col in self.regressors_:
                if col in X.columns:
                    future[col] = X[col].values[:steps]
        
        # Generate forecast with intervals
        forecast = self.model.predict(future)
        
        predictions = pd.Series(
            forecast['yhat'].values,
            index=future['ds'],
            name='forecast'
        )
        lower_bound = pd.Series(
            forecast['yhat_lower'].values,
            index=future['ds'],
            name='lower'
        )
        upper_bound = pd.Series(
            forecast['yhat_upper'].values,
            index=future['ds'],
            name='upper'
        )
        
        return predictions, lower_bound, upper_bound
    
    def plot_components(self):
        """
        Plot forecast components (trend, seasonality, etc.).
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure with component plots
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        future = self.model.make_future_dataframe(periods=30)
        forecast = self.model.predict(future)
        return self.model.plot_components(forecast)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'growth': self.growth,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'seasonality_mode': self.seasonality_mode,
            'regressors': self.regressors_
        })
        return params

