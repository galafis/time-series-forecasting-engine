# 📊 Visualization Module

## Overview

The visualization module provides comprehensive plotting tools for time series analysis and forecasting. It includes both static (Matplotlib) and interactive (Plotly) visualizations for exploring data, comparing models, and communicating results.

## TimeSeriesVisualizer

The main class `TimeSeriesVisualizer` offers:
- Time series plotting with trends
- Forecast visualization with prediction intervals
- Residual diagnostics
- Model comparison charts
- Seasonal decomposition plots
- Interactive dashboards

---

## Basic Time Series Plots

### Plot Time Series Data

```python
from src.visualization import TimeSeriesVisualizer
import pandas as pd

visualizer = TimeSeriesVisualizer()

# Basic plot
fig = visualizer.plot_time_series(
    data=series,
    title="Monthly Sales Data",
    figsize=(14, 6)
)
```

### Plot with Trend Line

```python
# Show linear trend
fig = visualizer.plot_time_series(
    data=series,
    title="Sales with Trend",
    show_trend=True
)
```

**Features:**
- Automatic date formatting
- Grid for readability
- Customizable figure size
- Optional trend line

---

## Forecast Visualization

### Plot Forecast vs Actual

```python
from src.visualization import TimeSeriesVisualizer

visualizer = TimeSeriesVisualizer()

# Basic forecast plot
fig = visualizer.plot_forecast(
    y_train=train_data,
    y_test=test_data,
    predictions=forecast,
    title="ARIMA Forecast vs Actual"
)
```

### Plot with Prediction Intervals

```python
# With confidence intervals
predictions, lower, upper = model.predict_with_intervals(
    steps=len(test),
    confidence=0.95
)

fig = visualizer.plot_forecast(
    y_train=train_data,
    y_test=test_data,
    predictions=predictions,
    lower_bound=lower,
    upper_bound=upper,
    title="Forecast with 95% Confidence Intervals"
)
```

**Features:**
- Training data in blue
- Test data in green
- Predictions in red
- Shaded confidence intervals
- Vertical line separating train/test
- Legend and grid

---

## Residual Diagnostics

### Comprehensive Residual Analysis

```python
from src.visualization import TimeSeriesVisualizer

visualizer = TimeSeriesVisualizer()

# Calculate residuals
residuals = test.values - predictions.values

# Create diagnostic plots
fig = visualizer.plot_residuals(residuals, figsize=(14, 10))
```

**Creates 4 subplots:**

1. **Residuals Over Time**
   - Check for patterns or trends
   - Should look random around zero

2. **Histogram of Residuals**
   - Check distribution shape
   - Should be approximately normal

3. **Q-Q Plot**
   - Check normality quantitatively
   - Points should fall on diagonal line

4. **Autocorrelation (ACF)**
   - Check for autocorrelation
   - Most lags should be within blue bands

### Interpretation Guide

```python
# Good residuals:
# - Random scatter around zero (no pattern)
# - Normal distribution (bell curve)
# - Q-Q points on line
# - ACF within confidence bands

# Bad residuals indicating issues:
# - Systematic pattern → model missing information
# - Non-normal distribution → consider transformation
# - ACF outside bands → model inadequate
# - Increasing variance → heteroscedasticity
```

---

## Model Comparison

### Compare Multiple Models

```python
from src.evaluation import ModelEvaluator
from src.visualization import TimeSeriesVisualizer

evaluator = ModelEvaluator()
visualizer = TimeSeriesVisualizer()

# Evaluate models
models = [arima, prophet, lstm, ensemble]
comparison = evaluator.compare_models(models, train, test)

# Visualize comparison
fig = visualizer.plot_model_comparison(
    comparison_df=comparison,
    metric='RMSE',
    figsize=(12, 6)
)
```

**Features:**
- Bar chart of selected metric
- Models sorted by performance
- Color-coded bars
- Value labels on bars

### Compare Multiple Metrics

```python
# Create subplot for each metric
metrics_to_plot = ['RMSE', 'MAE', 'MAPE']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, metric in zip(axes, metrics_to_plot):
    visualizer.plot_model_comparison(
        comparison_df=comparison,
        metric=metric,
        ax=ax
    )

plt.tight_layout()
plt.savefig('model_comparison_all_metrics.png', dpi=150, bbox_inches='tight')
```

---

## Seasonal Decomposition

### Plot Decomposition Components

```python
from src.preprocessing import TimeSeriesPreprocessor
from src.visualization import TimeSeriesVisualizer

preprocessor = TimeSeriesPreprocessor()
visualizer = TimeSeriesVisualizer()

# Decompose series
decomposition = preprocessor.decompose(
    data,
    model='additive',
    period=12
)

# Visualize components
fig = visualizer.plot_decomposition(
    trend=decomposition.trend,
    seasonal=decomposition.seasonal,
    residual=decomposition.residual,
    figsize=(14, 10)
)
```

**Creates 4 subplots:**
1. **Original:** The input time series
2. **Trend:** Long-term movement
3. **Seasonal:** Repeating patterns
4. **Residual:** Remaining noise

### Interpretation

```python
# Trend:
# - Upward → growth
# - Downward → decline
# - Flat → stable

# Seasonal:
# - Regular pattern → strong seasonality
# - Consistent amplitude → additive seasonality
# - Growing amplitude → multiplicative seasonality

# Residual:
# - Random scatter → good decomposition
# - Patterns remaining → model inadequate
# - Outliers → unusual events
```

---

## Interactive Visualizations

### Interactive Forecast Plot

```python
from src.visualization import TimeSeriesVisualizer

visualizer = TimeSeriesVisualizer()

# Create interactive Plotly figure
fig = visualizer.plot_interactive_forecast(
    y_train=train_data,
    y_test=test_data,
    predictions=forecast,
    lower_bound=lower,
    upper_bound=upper,
    title="Interactive Forecast Dashboard"
)

# Display in Jupyter
fig.show()

# Save as HTML
fig.write_html('forecast_dashboard.html')
```

**Interactive Features:**
- Zoom and pan
- Hover tooltips with values
- Toggle series on/off
- Export as PNG
- Responsive layout

### Multi-Model Interactive Comparison

```python
# Plot multiple forecasts
forecasts_dict = {
    'ARIMA': arima_forecast,
    'Prophet': prophet_forecast,
    'LSTM': lstm_forecast,
    'Ensemble': ensemble_forecast
}

import plotly.graph_objects as go

fig = go.Figure()

# Add actual data
fig.add_trace(go.Scatter(
    x=test.index, y=test.values,
    name='Actual',
    mode='lines',
    line=dict(color='black', width=2)
))

# Add each forecast
colors = ['blue', 'red', 'green', 'purple']
for (name, forecast), color in zip(forecasts_dict.items(), colors):
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values,
        name=name,
        mode='lines',
        line=dict(color=color, width=2, dash='dash')
    ))

fig.update_layout(
    title='Model Comparison - Interactive',
    xaxis_title='Date',
    yaxis_title='Value',
    hovermode='x unified',
    template='plotly_white'
)

fig.show()
```

---

## Advanced Visualizations

### Forecast Error Distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate errors
errors = test.values - predictions.values

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Error distribution
axes[0, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_title('Error Distribution')
axes[0, 0].set_xlabel('Error')
axes[0, 0].set_ylabel('Frequency')

# 2. Error over time
axes[0, 1].plot(errors, linewidth=1)
axes[0, 1].axhline(0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_title('Errors Over Time')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Error')

# 3. Absolute errors
axes[1, 0].plot(np.abs(errors), linewidth=1)
axes[1, 0].set_title('Absolute Errors')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('|Error|')

# 4. Cumulative error
axes[1, 1].plot(np.cumsum(errors), linewidth=2)
axes[1, 1].axhline(0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_title('Cumulative Error')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Cumulative Error')

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
```

### Prediction Interval Coverage

```python
# Visualize interval coverage
predictions, lower, upper = model.predict_with_intervals(len(test), confidence=0.95)

within = (test.values >= lower.values) & (test.values <= upper.values)
outside = ~within

fig, ax = plt.subplots(figsize=(14, 6))

# Plot intervals
ax.fill_between(test.index, lower.values, upper.values, alpha=0.2, label='95% Interval')

# Plot predictions
ax.plot(test.index, predictions.values, 'b-', label='Forecast', linewidth=2)

# Plot actual with different colors for within/outside interval
ax.plot(test.index[within], test.values[within], 'go', label='Within Interval', markersize=4)
ax.plot(test.index[outside], test.values[outside], 'ro', label='Outside Interval', markersize=6)

coverage = within.mean()
ax.set_title(f'Prediction Interval Coverage: {coverage:.1%} (Expected: 95%)')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interval_coverage.png', dpi=150, bbox_inches='tight')
```

### Forecast Horizon Accuracy

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate at different horizons
horizons = [1, 7, 14, 30, 60, 90]
results = evaluator.forecast_accuracy_by_horizon(
    model, train, test, horizons
)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(results['Horizon'], results['RMSE'], 'o-', linewidth=2, markersize=8, label='RMSE')
ax.plot(results['Horizon'], results['MAE'], 's-', linewidth=2, markersize=8, label='MAE')

ax.set_xlabel('Forecast Horizon (days)', fontsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.set_title('Forecast Accuracy by Horizon', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add annotations
for i, horizon in enumerate(results['Horizon']):
    ax.annotate(
        f"{results['RMSE'].iloc[i]:.1f}",
        xy=(horizon, results['RMSE'].iloc[i]),
        xytext=(0, 10),
        textcoords='offset points',
        fontsize=9
    )

plt.tight_layout()
plt.savefig('horizon_accuracy.png', dpi=150, bbox_inches='tight')
```

### Feature Importance (for ML models)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Example: feature importance from preprocessor
from src.preprocessing import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()
features = preprocessor.create_features(
    data,
    lags=[1, 7, 14, 30],
    rolling_windows=[7, 14, 30],
    time_features=True
)

# Calculate correlations with target
correlations = features.corr()['value'].drop('value').abs().sort_values(ascending=False)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['green' if x > 0.5 else 'orange' if x > 0.3 else 'red' for x in correlations.values]
correlations.head(15).plot(kind='barh', ax=ax, color=colors)

ax.set_xlabel('|Correlation| with Target', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Top 15 Features by Correlation', fontsize=14, fontweight='bold')
ax.axvline(0.3, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.3)')
ax.legend()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
```

---

## Styling and Customization

### Custom Style

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Or custom colors
custom_colors = {
    'train': '#2E86AB',
    'test': '#A23B72',
    'forecast': '#F18F01',
    'interval': '#C73E1D'
}

# Use in plots
visualizer.plot_forecast(
    train, test, predictions,
    train_color=custom_colors['train'],
    test_color=custom_colors['test'],
    pred_color=custom_colors['forecast']
)
```

### Publication-Ready Plots

```python
# High-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Create plot
fig = visualizer.plot_forecast(train, test, predictions)

# Save with high quality
plt.savefig(
    'forecast_publication.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
```

---

## Dashboard Creation

### Multi-Panel Dashboard

```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create dashboard layout
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. Time series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(train.index, train.values, label='Train')
ax1.plot(test.index, test.values, label='Test')
ax1.plot(predictions.index, predictions.values, label='Forecast', linestyle='--')
ax1.set_title('Time Series Forecast', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Residuals
ax2 = fig.add_subplot(gs[1, 0])
residuals = test.values - predictions.values
ax2.plot(residuals, linewidth=1)
ax2.axhline(0, color='r', linestyle='--')
ax2.set_title('Residuals', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Residual distribution
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
ax3.axvline(0, color='r', linestyle='--')
ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')

# 4. ACF
ax4 = fig.add_subplot(gs[2, 0])
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=30, ax=ax4)
ax4.set_title('Autocorrelation', fontsize=12, fontweight='bold')

# 5. Model comparison
ax5 = fig.add_subplot(gs[2, 1])
comparison = evaluator.compare_models(models, train, test)
comparison['RMSE'].plot(kind='barh', ax=ax5)
ax5.set_title('Model RMSE Comparison', fontsize=12, fontweight='bold')

plt.suptitle('Forecasting Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
```

---

## Best Practices

### 1. Always Include Context

```python
# Bad: no context
plt.plot(predictions)

# Good: full context
visualizer.plot_forecast(
    y_train=train,
    y_test=test,
    predictions=predictions,
    title=f"{model.name} - {dataset_name} - {date_range}",
    xlabel='Date',
    ylabel='Sales (units)'
)
```

### 2. Use Appropriate Figure Sizes

```python
# Time series: wide and short
fig = visualizer.plot_time_series(data, figsize=(14, 6))

# Residual diagnostics: square
fig = visualizer.plot_residuals(residuals, figsize=(14, 10))

# Dashboard: large
fig = create_dashboard(figsize=(16, 12))
```

### 3. Save in Multiple Formats

```python
# Save plot
fig = visualizer.plot_forecast(train, test, predictions)

# PNG for presentations
plt.savefig('forecast.png', dpi=150, bbox_inches='tight')

# PDF for publications
plt.savefig('forecast.pdf', bbox_inches='tight')

# SVG for web
plt.savefig('forecast.svg', bbox_inches='tight')
```

### 4. Use Consistent Colors

```python
# Define color scheme once
COLORS = {
    'train': '#1f77b4',
    'test': '#2ca02c',
    'forecast': '#d62728',
    'interval': '#d62728',
    'residual': '#ff7f0e'
}

# Use throughout
visualizer.plot_forecast(
    train, test, predictions,
    train_color=COLORS['train'],
    test_color=COLORS['test'],
    pred_color=COLORS['forecast']
)
```

### 5. Add Grid and Labels

```python
# Always add these for clarity
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Descriptive Title', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
```

---

## Troubleshooting

### Issue: Overlapping Date Labels
```python
# Solution: rotate labels
plt.xticks(rotation=45, ha='right')

# Or use automatic date formatting
from matplotlib.dates import DateFormatter, AutoDateLocator
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(AutoDateLocator())
plt.gcf().autofmt_xdate()
```

### Issue: Legend Outside Plot Area
```python
# Solution: adjust position
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
```

### Issue: Low Quality Saved Images
```python
# Solution: increase DPI
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

### Issue: Plot Too Busy
```python
# Solution: use interactive plot
fig = visualizer.plot_interactive_forecast(...)
# Users can toggle series on/off
```

---

## Gallery Examples

See the `/examples` directory for complete working examples:
- `basic_plotting.py` - Basic visualization examples
- `advanced_dashboard.py` - Complex dashboard creation
- `interactive_plots.py` - Plotly interactive visualizations
- `publication_quality.py` - Publication-ready plots

---

## References

- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
- Plotly Technologies Inc. (2015). Collaborative data science.
- Waskom, M. (2021). seaborn: statistical data visualization.

