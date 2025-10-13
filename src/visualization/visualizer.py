"""
Visualization utilities for time series forecasting.

This module provides comprehensive visualization tools for time series
data, forecasts, and model diagnostics.
"""

from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TimeSeriesVisualizer:
    """
    Comprehensive visualization toolkit for time series forecasting.
    
    Provides static (matplotlib) and interactive (plotly) visualizations
    for data exploration, forecast analysis, and model diagnostics.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        style : str
            Matplotlib style to use
        """
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
    
    def plot_time_series(
        self,
        data: pd.Series,
        title: str = "Time Series",
        figsize: Tuple[int, int] = (14, 6),
        show_trend: bool = False
    ) -> plt.Figure:
        """
        Plot time series data.
        
        Parameters
        ----------
        data : pd.Series
            Time series to plot
        title : str
            Plot title
        figsize : tuple
            Figure size
        show_trend : bool
            Whether to show trend line
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(data.index, data.values, linewidth=2, label='Actual')
        
        if show_trend:
            z = np.polyfit(range(len(data)), data.values, 1)
            p = np.poly1d(z)
            ax.plot(data.index, p(range(len(data))), 
                   "--", linewidth=2, alpha=0.7, label='Trend')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_forecast(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        predictions: pd.Series,
        lower_bound: Optional[pd.Series] = None,
        upper_bound: Optional[pd.Series] = None,
        title: str = "Forecast vs Actual",
        figsize: Tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Plot forecast against actual values.
        
        Parameters
        ----------
        y_train : pd.Series
            Training data
        y_test : pd.Series
            Test data
        predictions : pd.Series
            Forecasted values
        lower_bound : pd.Series, optional
            Lower prediction interval
        upper_bound : pd.Series, optional
            Upper prediction interval
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training data
        ax.plot(y_train.index, y_train.values, 
               linewidth=2, label='Training Data', color=self.colors[0])
        
        # Plot test data
        ax.plot(y_test.index, y_test.values, 
               linewidth=2, label='Actual', color=self.colors[1])
        
        # Plot predictions
        ax.plot(y_test.index, predictions.values, 
               linewidth=2, label='Forecast', color=self.colors[2], linestyle='--')
        
        # Plot prediction intervals if provided
        if lower_bound is not None and upper_bound is not None:
            ax.fill_between(
                y_test.index,
                lower_bound.values,
                upper_bound.values,
                alpha=0.2,
                color=self.colors[2],
                label='Prediction Interval'
            )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(
        self,
        residuals: np.ndarray,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Plot residual diagnostics.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Residuals over time
        axes[0, 0].plot(residuals, linewidth=1)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribution of Residuals', fontweight='bold')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ACF plot
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, lags=40, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title('Autocorrelation Function', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'RMSE',
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot comparison of multiple models.
        
        Parameters
        ----------
        comparison_df : pd.DataFrame
            DataFrame with model comparison results
        metric : str
            Metric to compare
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        models = comparison_df.index
        values = comparison_df[metric].values
        
        bars = ax.bar(models, values, color=self.colors[:len(models)], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f'Model Comparison - {metric}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def plot_interactive_forecast(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        predictions: pd.Series,
        lower_bound: Optional[pd.Series] = None,
        upper_bound: Optional[pd.Series] = None,
        title: str = "Interactive Forecast"
    ) -> go.Figure:
        """
        Create interactive forecast plot using Plotly.
        
        Parameters
        ----------
        y_train : pd.Series
            Training data
        y_test : pd.Series
            Test data
        predictions : pd.Series
            Forecasted values
        lower_bound : pd.Series, optional
            Lower prediction interval
        upper_bound : pd.Series, optional
            Upper prediction interval
        title : str
            Plot title
            
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive figure
        """
        fig = go.Figure()
        
        # Training data
        fig.add_trace(go.Scatter(
            x=y_train.index,
            y=y_train.values,
            mode='lines',
            name='Training Data',
            line=dict(color='blue', width=2)
        ))
        
        # Actual test data
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=y_test.values,
            mode='lines',
            name='Actual',
            line=dict(color='green', width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=predictions.values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Prediction intervals
        if lower_bound is not None and upper_bound is not None:
            fig.add_trace(go.Scatter(
                x=y_test.index,
                y=upper_bound.values,
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=y_test.index,
                y=lower_bound.values,
                mode='lines',
                name='Prediction Interval',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.2)',
                fill='tonexty'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def plot_decomposition(
        self,
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Plot time series decomposition.
        
        Parameters
        ----------
        trend : pd.Series
            Trend component
        seasonal : pd.Series
            Seasonal component
        residual : pd.Series
            Residual component
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Trend
        axes[0].plot(trend.index, trend.values, linewidth=2, color=self.colors[0])
        axes[0].set_title('Trend Component', fontweight='bold')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Seasonal
        axes[1].plot(seasonal.index, seasonal.values, linewidth=2, color=self.colors[1])
        axes[1].set_title('Seasonal Component', fontweight='bold')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)
        
        # Residual
        axes[2].plot(residual.index, residual.values, linewidth=1, color=self.colors[2])
        axes[2].set_title('Residual Component', fontweight='bold')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Value')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

