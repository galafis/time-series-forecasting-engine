"""
Model evaluation utilities for time series forecasting.

This module provides comprehensive metrics and evaluation methods
for assessing forecasting model performance.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    """
    Comprehensive model evaluation toolkit for time series forecasting.
    
    Provides various metrics and cross-validation methods specific to
    time series data.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.results_ = {}
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive forecasting metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns
        -------
        metrics : dict
            Dictionary of metric names and values
        """
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Percentage metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Symmetric MAPE (handles zero values better)
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        # R-squared
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Scaled Error (MASE)
        # Uses naive forecast (previous value) as baseline
        naive_forecast = np.roll(y_true, 1)[1:]
        mae_naive = mean_absolute_error(y_true[1:], naive_forecast)
        mase = mae / mae_naive if mae_naive != 0 else np.inf
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'sMAPE': smape,
            'R2': r2,
            'MASE': mase
        }
    
    def evaluate_model(
        self,
        model,
        y_train: pd.Series,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Evaluate a forecasting model on test data.
        
        Parameters
        ----------
        model : BaseForecaster
            Fitted forecasting model
        y_train : pd.Series
            Training data (for fitting if needed)
        y_test : pd.Series
            Test data
        X_train : pd.DataFrame, optional
            Training exogenous features
        X_test : pd.DataFrame, optional
            Test exogenous features
            
        Returns
        -------
        metrics : dict
            Evaluation metrics
        """
        # Fit model if not already fitted
        if not model.is_fitted:
            model.fit(y_train, X_train)
        
        # Generate predictions
        predictions = model.predict(len(y_test), X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test.values, predictions.values)
        
        # Store results
        self.results_[model.name] = {
            'metrics': metrics,
            'predictions': predictions,
            'actuals': y_test
        }
        
        return metrics
    
    def time_series_cv(
        self,
        model,
        data: pd.Series,
        n_splits: int = 5,
        test_size: int = 10,
        X: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Uses expanding window approach where training set grows
        and test set slides forward.
        
        Parameters
        ----------
        model : BaseForecaster
            Forecasting model to evaluate
        data : pd.Series
            Complete time series
        n_splits : int
            Number of CV splits
        test_size : int
            Size of test set in each split
        X : pd.DataFrame, optional
            Exogenous features
            
        Returns
        -------
        cv_metrics : dict
            Dictionary of metric lists across folds
        """
        metrics_per_fold = []
        
        # Calculate split points
        total_size = len(data)
        min_train_size = total_size - (n_splits * test_size)
        
        for i in range(n_splits):
            # Define train and test indices
            train_end = min_train_size + (i * test_size)
            test_start = train_end
            test_end = test_start + test_size
            
            # Split data
            y_train = data.iloc[:train_end]
            y_test = data.iloc[test_start:test_end]
            
            X_train = X.iloc[:train_end] if X is not None else None
            X_test = X.iloc[test_start:test_end] if X is not None else None
            
            # Fit and predict
            model.fit(y_train, X_train)
            predictions = model.predict(len(y_test), X_test)
            
            # Calculate metrics
            fold_metrics = self.calculate_metrics(y_test.values, predictions.values)
            metrics_per_fold.append(fold_metrics)
        
        # Aggregate metrics across folds
        cv_metrics = {}
        for metric_name in metrics_per_fold[0].keys():
            values = [fold[metric_name] for fold in metrics_per_fold]
            cv_metrics[f'{metric_name}_mean'] = np.mean(values)
            cv_metrics[f'{metric_name}_std'] = np.std(values)
            cv_metrics[f'{metric_name}_folds'] = values
        
        return cv_metrics
    
    def compare_models(
        self,
        models: List,
        y_train: pd.Series,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.
        
        Parameters
        ----------
        models : list
            List of forecasting models
        y_train : pd.Series
            Training data
        y_test : pd.Series
            Test data
        X_train : pd.DataFrame, optional
            Training exogenous features
        X_test : pd.DataFrame, optional
            Test exogenous features
            
        Returns
        -------
        comparison : pd.DataFrame
            DataFrame comparing model performance
        """
        results = []
        
        for model in models:
            metrics = self.evaluate_model(model, y_train, y_test, X_train, X_test)
            metrics['Model'] = model.name
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index('Model')
        
        return comparison_df.round(4)
    
    def residual_analysis(
        self,
        model,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, any]:
        """
        Perform residual analysis.
        
        Parameters
        ----------
        model : BaseForecaster
            Fitted model
        y_true : pd.Series
            True values
        y_pred : pd.Series
            Predicted values
            
        Returns
        -------
        analysis : dict
            Residual analysis results
        """
        residuals = y_true.values - y_pred.values
        
        # Basic statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Normality test (Shapiro-Wilk)
        from scipy.stats import shapiro
        _, p_value_normality = shapiro(residuals)
        
        # Autocorrelation (Ljung-Box test)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        
        return {
            'mean': mean_residual,
            'std': std_residual,
            'normality_p_value': p_value_normality,
            'is_normal': p_value_normality > 0.05,
            'ljung_box_p_value': lb_test['lb_pvalue'].values[0],
            'is_white_noise': lb_test['lb_pvalue'].values[0] > 0.05,
            'residuals': residuals
        }
    
    def forecast_accuracy_by_horizon(
        self,
        model,
        y_train: pd.Series,
        y_test: pd.Series,
        horizons: List[int],
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Evaluate forecast accuracy at different horizons.
        
        Parameters
        ----------
        model : BaseForecaster
            Forecasting model
        y_train : pd.Series
            Training data
        y_test : pd.Series
            Test data
        horizons : list of int
            Forecast horizons to evaluate
        X_train : pd.DataFrame, optional
            Training exogenous features
        X_test : pd.DataFrame, optional
            Test exogenous features
            
        Returns
        -------
        horizon_metrics : pd.DataFrame
            Metrics at each horizon
        """
        model.fit(y_train, X_train)
        
        results = []
        for h in horizons:
            if h > len(y_test):
                continue
            
            y_test_h = y_test.iloc[:h]
            X_test_h = X_test.iloc[:h] if X_test is not None else None
            
            predictions = model.predict(h, X_test_h)
            metrics = self.calculate_metrics(y_test_h.values, predictions.values)
            metrics['Horizon'] = h
            results.append(metrics)
        
        return pd.DataFrame(results).set_index('Horizon')

