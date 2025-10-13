"""
Complete example demonstrating the Time Series Forecasting Engine.

This example shows how to:
1. Load and preprocess time series data
2. Train multiple forecasting models
3. Evaluate and compare models
4. Generate forecasts with prediction intervals
5. Visualize results

Author: Gabriel Demetrios Lafis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.models import ARIMAForecaster, ProphetForecaster, LSTMForecaster, EnsembleForecaster
from src.preprocessing import TimeSeriesPreprocessor
from src.evaluation import ModelEvaluator
from src.visualization import TimeSeriesVisualizer


def generate_sample_data(n_points=500):
    """Generate sample time series data with trend, seasonality, and noise."""
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    
    # Trend component
    trend = np.linspace(100, 200, n_points)
    
    # Seasonal component (yearly)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 365)
    
    # Weekly seasonality
    weekly = 10 * np.sin(2 * np.pi * np.arange(n_points) / 7)
    
    # Random noise
    noise = np.random.normal(0, 5, n_points)
    
    # Combine components
    values = trend + seasonal + weekly + noise
    
    return pd.Series(values, index=dates, name='value')


def main():
    """Main execution function."""
    print("=" * 80)
    print("Time Series Forecasting Engine - Complete Example")
    print("=" * 80)
    print()
    
    # 1. Generate sample data
    print("1. Generating sample time series data...")
    data = generate_sample_data(n_points=500)
    print(f"   Generated {len(data)} data points")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print()
    
    # 2. Preprocess data
    print("2. Preprocessing data...")
    preprocessor = TimeSeriesPreprocessor()
    
    # Handle outliers
    data_clean = preprocessor.remove_outliers(data, method='iqr', threshold=3.0)
    print(f"   Removed outliers: {(data != data_clean).sum()} points")
    
    # Check for missing values
    if data_clean.isna().sum() > 0:
        data_clean = preprocessor.handle_missing_values(data_clean, method='interpolate')
        print(f"   Filled missing values: {data_clean.isna().sum()} points")
    print()
    
    # 3. Split data
    print("3. Splitting data into train and test sets...")
    train_size = int(len(data_clean) * 0.8)
    y_train = data_clean[:train_size]
    y_test = data_clean[train_size:]
    print(f"   Training set: {len(y_train)} points")
    print(f"   Test set: {len(y_test)} points")
    print()
    
    # 4. Initialize models
    print("4. Initializing forecasting models...")
    models = [
        ARIMAForecaster(auto_select=True),
        ProphetForecaster(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
    ]
    
    # Try to add LSTM if TensorFlow is available
    try:
        lstm_model = LSTMForecaster(
            lookback=30,
            lstm_units=64,
            num_layers=2,
            epochs=50,
            batch_size=32
        )
        models.append(lstm_model)
    except ImportError:
        print("   Note: Skipping LSTM model (TensorFlow not installed)")
    
    print(f"   Initialized {len(models)} models:")
    for model in models:
        print(f"   - {model.name}")
    print()
    
    # 5. Train and evaluate models
    print("5. Training and evaluating models...")
    evaluator = ModelEvaluator()
    
    for i, model in enumerate(models, 1):
        print(f"   Training {model.name}...")
        metrics = evaluator.evaluate_model(model, y_train, y_test)
        print(f"   ✓ {model.name} trained successfully")
        print(f"     RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")
    print()
    
    # 6. Create ensemble model
    print("6. Creating ensemble model...")
    ensemble = EnsembleForecaster(models, method='average')
    ensemble_metrics = evaluator.evaluate_model(ensemble, y_train, y_test)
    print(f"   ✓ Ensemble model created")
    print(f"     RMSE: {ensemble_metrics['RMSE']:.4f}, MAE: {ensemble_metrics['MAE']:.4f}, MAPE: {ensemble_metrics['MAPE']:.2f}%")
    print()
    
    # 7. Compare models
    print("7. Comparing model performance...")
    all_models = models + [ensemble]
    comparison = evaluator.compare_models(all_models, y_train, y_test)
    print(comparison)
    print()
    
    # 8. Generate forecasts with intervals
    print("8. Generating forecasts with prediction intervals...")
    best_model = models[0]  # Use ARIMA for demonstration
    predictions, lower, upper = best_model.predict_with_intervals(
        steps=len(y_test),
        confidence=0.95
    )
    print(f"   ✓ Generated {len(predictions)} forecasts with 95% confidence intervals")
    print()
    
    # 9. Visualize results
    print("9. Creating visualizations...")
    visualizer = TimeSeriesVisualizer()
    
    # Plot original time series
    fig1 = visualizer.plot_time_series(
        data_clean,
        title="Original Time Series",
        show_trend=True
    )
    plt.savefig('time_series.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: time_series.png")
    
    # Plot forecast
    fig2 = visualizer.plot_forecast(
        y_train,
        y_test,
        predictions,
        lower,
        upper,
        title=f"{best_model.name} - Forecast vs Actual"
    )
    plt.savefig('forecast.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: forecast.png")
    
    # Plot residuals
    if hasattr(best_model, 'residuals_') and best_model.residuals_ is not None:
        fig3 = visualizer.plot_residuals(best_model.residuals_)
        plt.savefig('residuals.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: residuals.png")
    
    # Plot model comparison
    fig4 = visualizer.plot_model_comparison(comparison, metric='RMSE')
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: model_comparison.png")
    print()
    
    # 10. Cross-validation
    print("10. Performing time series cross-validation...")
    cv_results = evaluator.time_series_cv(
        ARIMAForecaster(auto_select=True),
        data_clean,
        n_splits=5,
        test_size=20
    )
    print(f"    RMSE: {cv_results['RMSE_mean']:.4f} ± {cv_results['RMSE_std']:.4f}")
    print(f"    MAE: {cv_results['MAE_mean']:.4f} ± {cv_results['MAE_std']:.4f}")
    print()
    
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

