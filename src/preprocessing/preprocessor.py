"""
Time series preprocessing utilities.

This module provides tools for preprocessing time series data including
handling missing values, outlier detection, detrending, and differencing.
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TimeSeriesPreprocessor:
    """
    Comprehensive time series preprocessing toolkit.
    
    Provides methods for cleaning, transforming, and preparing time series
    data for forecasting models.
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scaler = None
        self.trend_ = None
        self.seasonal_ = None
        
    def handle_missing_values(
        self,
        data: pd.Series,
        method: str = 'interpolate',
        **kwargs
    ) -> pd.Series:
        """
        Handle missing values in time series.
        
        Parameters
        ----------
        data : pd.Series
            Time series with potential missing values
        method : str
            Method to use: 'interpolate', 'forward_fill', 'backward_fill', 'mean'
        **kwargs : dict
            Additional parameters for the chosen method
            
        Returns
        -------
        filled_data : pd.Series
            Time series with missing values handled
        """
        if method == 'interpolate':
            return data.interpolate(method='time', **kwargs)
        elif method == 'forward_fill':
            return data.ffill(**kwargs)
        elif method == 'backward_fill':
            return data.bfill(**kwargs)
        elif method == 'mean':
            return data.fillna(data.mean())
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect_outliers(
        self,
        data: pd.Series,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers in time series.
        
        Parameters
        ----------
        data : pd.Series
            Time series data
        method : str
            Detection method: 'iqr' or 'zscore'
        threshold : float
            Threshold for outlier detection
            
        Returns
        -------
        outliers : pd.Series
            Boolean series indicating outliers
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            return z_scores > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def remove_outliers(
        self,
        data: pd.Series,
        method: str = 'iqr',
        threshold: float = 3.0,
        replacement: str = 'interpolate'
    ) -> pd.Series:
        """
        Remove or replace outliers in time series.
        
        Parameters
        ----------
        data : pd.Series
            Time series data
        method : str
            Detection method: 'iqr' or 'zscore'
        threshold : float
            Threshold for outlier detection
        replacement : str
            How to replace outliers: 'interpolate', 'mean', 'median', or 'nan'
            
        Returns
        -------
        cleaned_data : pd.Series
            Time series with outliers handled
        """
        outliers = self.detect_outliers(data, method, threshold)
        cleaned = data.copy()
        
        if replacement == 'nan':
            cleaned[outliers] = np.nan
        elif replacement == 'mean':
            cleaned[outliers] = data[~outliers].mean()
        elif replacement == 'median':
            cleaned[outliers] = data[~outliers].median()
        elif replacement == 'interpolate':
            cleaned[outliers] = np.nan
            cleaned = cleaned.interpolate(method='time')
        else:
            raise ValueError(f"Unknown replacement method: {replacement}")
        
        return cleaned
    
    def scale_data(
        self,
        data: pd.Series,
        method: str = 'standard',
        feature_range: Tuple[float, float] = (0, 1)
    ) -> pd.Series:
        """
        Scale time series data.
        
        Parameters
        ----------
        data : pd.Series
            Time series to scale
        method : str
            Scaling method: 'standard' or 'minmax'
        feature_range : tuple
            Range for MinMax scaling
            
        Returns
        -------
        scaled_data : pd.Series
            Scaled time series
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=feature_range)
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaled_values = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled_values, index=data.index, name=data.name)
    
    def inverse_scale(self, data: pd.Series) -> pd.Series:
        """
        Inverse transform scaled data.
        
        Parameters
        ----------
        data : pd.Series
            Scaled time series
            
        Returns
        -------
        original_scale : pd.Series
            Data in original scale
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted. Call scale_data first.")
        
        original_values = self.scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()
        return pd.Series(original_values, index=data.index, name=data.name)
    
    def difference(
        self,
        data: pd.Series,
        periods: int = 1,
        order: int = 1
    ) -> pd.Series:
        """
        Apply differencing to make series stationary.
        
        Parameters
        ----------
        data : pd.Series
            Time series to difference
        periods : int
            Number of periods to shift
        order : int
            Order of differencing (how many times to difference)
            
        Returns
        -------
        differenced : pd.Series
            Differenced time series
        """
        result = data.copy()
        for _ in range(order):
            result = result.diff(periods=periods)
        return result.dropna()
    
    def inverse_difference(
        self,
        differenced: pd.Series,
        original: pd.Series,
        periods: int = 1,
        order: int = 1
    ) -> pd.Series:
        """
        Reverse differencing operation.
        
        Parameters
        ----------
        differenced : pd.Series
            Differenced time series
        original : pd.Series
            Original time series (needed for reconstruction)
        periods : int
            Number of periods that were shifted
        order : int
            Order of differencing that was applied
            
        Returns
        -------
        reconstructed : pd.Series
            Reconstructed time series
        """
        result = differenced.copy()
        
        for _ in range(order):
            # Get the last 'periods' values from original to start reconstruction
            cumsum = result.cumsum()
            result = cumsum + original.iloc[-periods]
        
        return result
    
    def decompose(
        self,
        data: pd.Series,
        model: str = 'additive',
        period: Optional[int] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Parameters
        ----------
        data : pd.Series
            Time series to decompose
        model : str
            'additive' or 'multiplicative'
        period : int, optional
            Period for seasonal decomposition
            
        Returns
        -------
        trend : pd.Series
            Trend component
        seasonal : pd.Series
            Seasonal component
        residual : pd.Series
            Residual component
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        result = seasonal_decompose(
            data,
            model=model,
            period=period,
            extrapolate_trend='freq'
        )
        
        self.trend_ = result.trend
        self.seasonal_ = result.seasonal
        
        return result.trend, result.seasonal, result.resid
    
    def create_lag_features(
        self,
        data: pd.Series,
        lags: Union[int, list]
    ) -> pd.DataFrame:
        """
        Create lagged features for machine learning models.
        
        Parameters
        ----------
        data : pd.Series
            Time series data
        lags : int or list
            Number of lags or list of specific lag values
            
        Returns
        -------
        lagged_df : pd.DataFrame
            DataFrame with lagged features
        """
        if isinstance(lags, int):
            lags = list(range(1, lags + 1))
        
        lagged_data = {}
        for lag in lags:
            lagged_data[f'lag_{lag}'] = data.shift(lag)
        
        return pd.DataFrame(lagged_data, index=data.index).dropna()
    
    def create_rolling_features(
        self,
        data: pd.Series,
        windows: Union[int, list],
        functions: list = ['mean', 'std']
    ) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Parameters
        ----------
        data : pd.Series
            Time series data
        windows : int or list
            Window sizes
        functions : list
            Aggregation functions to apply
            
        Returns
        -------
        rolling_df : pd.DataFrame
            DataFrame with rolling features
        """
        if isinstance(windows, int):
            windows = [windows]
        
        rolling_data = {}
        for window in windows:
            for func in functions:
                col_name = f'rolling_{func}_{window}'
                rolling_data[col_name] = data.rolling(window=window).agg(func)
        
        return pd.DataFrame(rolling_data, index=data.index).dropna()

