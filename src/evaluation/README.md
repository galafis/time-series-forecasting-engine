# 📈 Evaluation Module

## Overview

The evaluation module provides comprehensive tools for assessing time series forecasting model performance. It includes various error metrics, statistical tests, cross-validation methods, and comparison utilities.

## ModelEvaluator

The main class `ModelEvaluator` offers:
- Multiple error metrics (MAE, RMSE, MAPE, etc.)
- Residual analysis
- Cross-validation for time series
- Model comparison
- Forecast accuracy by horizon

---

## Error Metrics

### Available Metrics

The evaluator provides 7 standard metrics for assessing forecast accuracy:

#### 1. MAE (Mean Absolute Error)
Average absolute difference between actual and predicted values.

**Formula:** MAE = (1/n) × Σ|yᵢ - ŷᵢ|

**Interpretation:**
- Same scale as original data
- Robust to outliers
- Easy to interpret

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.calculate_metrics(y_true, y_pred)
print(f"MAE: {metrics['MAE']:.2f}")
```

**When to use:** General error measurement, when all errors are equally important

---

#### 2. RMSE (Root Mean Squared Error)
Square root of average squared differences.

**Formula:** RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]

**Interpretation:**
- Same scale as original data
- Penalizes large errors more heavily
- Sensitive to outliers

```python
print(f"RMSE: {metrics['RMSE']:.2f}")
```

**When to use:** When large errors are particularly undesirable

---

#### 3. MSE (Mean Squared Error)
Average of squared differences.

**Formula:** MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

**Interpretation:**
- Squared scale (harder to interpret)
- Heavily penalizes large errors
- Differentiable (good for optimization)

```python
print(f"MSE: {metrics['MSE']:.2f}")
```

**When to use:** Model training optimization, when scale doesn't matter

---

#### 4. MAPE (Mean Absolute Percentage Error)
Average absolute percentage difference.

**Formula:** MAPE = (100/n) × Σ|((yᵢ - ŷᵢ)/yᵢ)|

**Interpretation:**
- Scale-independent (percentage)
- Easy to communicate to stakeholders
- Undefined when actual value is zero
- Asymmetric (penalizes over-forecasts more)

```python
print(f"MAPE: {metrics['MAPE']:.2f}%")
```

**When to use:** Comparing accuracy across different scales, business reporting

---

#### 5. sMAPE (Symmetric Mean Absolute Percentage Error)
Symmetric version of MAPE.

**Formula:** sMAPE = (100/n) × Σ(|yᵢ - ŷᵢ|/(|yᵢ| + |ŷᵢ|))

**Interpretation:**
- Scale-independent
- Bounded between 0% and 100%
- Symmetric (treats over/under-forecasts equally)
- More robust than MAPE

```python
print(f"sMAPE: {metrics['sMAPE']:.2f}%")
```

**When to use:** When MAPE is too asymmetric, comparing models across datasets

---

#### 6. R² (Coefficient of Determination)
Proportion of variance explained by the model.

**Formula:** R² = 1 - (Σ(yᵢ - ŷᵢ)²/Σ(yᵢ - ȳ)²)

**Interpretation:**
- Range: -∞ to 1 (typically 0 to 1)
- 1 = perfect fit
- 0 = model no better than mean
- Negative = model worse than mean

```python
print(f"R²: {metrics['R2']:.4f}")
```

**When to use:** Assessing model fit quality, comparing regression models

---

#### 7. MASE (Mean Absolute Scaled Error)
Error scaled by naive forecast error.

**Formula:** MASE = MAE / (1/(n-1) × Σ|yᵢ - yᵢ₋₁|)

**Interpretation:**
- Scale-independent
- < 1: Better than naive forecast
- = 1: Same as naive forecast
- > 1: Worse than naive forecast

```python
print(f"MASE: {metrics['MASE']:.4f}")
```

**When to use:** Comparing performance to baseline, benchmarking across datasets

---

## Using Metrics

### Calculate All Metrics

```python
from src.evaluation import ModelEvaluator
import numpy as np

# True and predicted values
y_true = np.array([100, 105, 110, 115, 120])
y_pred = np.array([98, 107, 108, 116, 122])

evaluator = ModelEvaluator()
metrics = evaluator.calculate_metrics(y_true, y_pred)

print("Forecast Error Metrics:")
print(f"  MAE:   {metrics['MAE']:.2f}")
print(f"  RMSE:  {metrics['RMSE']:.2f}")
print(f"  MSE:   {metrics['MSE']:.2f}")
print(f"  MAPE:  {metrics['MAPE']:.2f}%")
print(f"  sMAPE: {metrics['sMAPE']:.2f}%")
print(f"  R²:    {metrics['R2']:.4f}")
print(f"  MASE:  {metrics['MASE']:.4f}")
```

### Metric Selection Guide

| Scenario | Primary Metric | Secondary Metric |
|----------|----------------|------------------|
| Same scale data | RMSE | MAE |
| Different scales | MAPE | sMAPE |
| Large errors critical | RMSE | MSE |
| Outliers present | MAE | sMAPE |
| Business reporting | MAPE | R² |
| Benchmarking | MASE | RMSE |
| Model comparison | R² | RMSE |

---

## Model Evaluation

### Evaluate Single Model

```python
from src.models import ARIMAForecaster
from src.evaluation import ModelEvaluator

# Split data
train = data[:int(len(data) * 0.8)]
test = data[int(len(data) * 0.8):]

# Train and evaluate
model = ARIMAForecaster(auto_select=True)
evaluator = ModelEvaluator()

metrics = evaluator.evaluate_model(model, train, test)

print(f"Model: {model.name}")
print(f"RMSE: {metrics['RMSE']:.2f}")
print(f"MAE: {metrics['MAE']:.2f}")
print(f"MAPE: {metrics['MAPE']:.2f}%")
```

### Compare Multiple Models

```python
from src.models import ARIMAForecaster, ProphetForecaster, LSTMForecaster
from src.evaluation import ModelEvaluator

# Initialize models
models = [
    ARIMAForecaster(auto_select=True),
    ProphetForecaster(),
    LSTMForecaster(lookback=30, epochs=50)
]

# Compare
evaluator = ModelEvaluator()
comparison = evaluator.compare_models(models, train, test)

print(comparison)
```

**Output:**
```
                    MAE      MSE     RMSE    MAPE  sMAPE      R2    MASE
Model                                                                    
ARIMAForecaster    5.23   42.15     6.49    2.34   2.35   0.842   0.95
ProphetForecaster  4.87   35.28     5.94    2.18   2.19   0.867   0.88
LSTMForecaster     6.12   51.34     7.17    2.74   2.75   0.807   1.11
```

### Find Best Model

```python
# Get best model by RMSE
best_model_name = comparison['RMSE'].idxmin()
print(f"Best model: {best_model_name}")

# Sort by multiple metrics
comparison_sorted = comparison.sort_values(['RMSE', 'MAE'])
print("\nModels ranked by RMSE then MAE:")
print(comparison_sorted)
```

---

## Residual Analysis

Residual analysis helps diagnose model adequacy by examining prediction errors.

### Basic Residual Analysis

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Fit model
model.fit(train)
predictions = model.predict(len(test))

# Analyze residuals
residual_stats = evaluator.residual_analysis(model, test, predictions)

print("Residual Statistics:")
print(f"  Mean: {residual_stats['mean']:.4f}")
print(f"  Std Dev: {residual_stats['std']:.4f}")
print(f"  Normality p-value: {residual_stats['normality_p_value']:.4f}")
print(f"  Is normal: {residual_stats['is_normal']}")
print(f"  Ljung-Box p-value: {residual_stats['ljung_box_p_value']:.4f}")
print(f"  Is white noise: {residual_stats['is_white_noise']}")
```

### Interpretation

#### Mean of Residuals
- **Should be:** Close to 0
- **If not:** Model is biased (systematically over/under-predicting)

#### Normality Test (Shapiro-Wilk)
- **p > 0.05:** Residuals are normally distributed ✓
- **p < 0.05:** Residuals are not normal (consider transformation or different model)

#### White Noise Test (Ljung-Box)
- **p > 0.05:** Residuals are white noise (no autocorrelation) ✓
- **p < 0.05:** Residuals show patterns (model missing information)

### Visual Residual Diagnostics

```python
from src.visualization import TimeSeriesVisualizer

visualizer = TimeSeriesVisualizer()
residuals = test.values - predictions.values

# Plot comprehensive residual diagnostics
fig = visualizer.plot_residuals(residuals)
```

This creates:
1. **Residuals over time:** Check for patterns
2. **Histogram:** Check normality
3. **Q-Q plot:** Check normal distribution
4. **ACF plot:** Check autocorrelation

---

## Time Series Cross-Validation

Standard k-fold cross-validation doesn't work for time series because it violates temporal order. Use time series cross-validation instead.

### Rolling Window Cross-Validation

```python
from src.models import ARIMAForecaster
from src.evaluation import ModelEvaluator

model = ARIMAForecaster(auto_select=True)
evaluator = ModelEvaluator()

# Perform cross-validation
cv_results = evaluator.time_series_cv(
    model=model,
    data=data,
    n_splits=5,      # Number of splits
    test_size=30     # Size of test set
)

print("Cross-Validation Results:")
print(f"  RMSE: {cv_results['RMSE_mean']:.2f} ± {cv_results['RMSE_std']:.2f}")
print(f"  MAE:  {cv_results['MAE_mean']:.2f} ± {cv_results['MAE_std']:.2f}")
print(f"  MAPE: {cv_results['MAPE_mean']:.2f}% ± {cv_results['MAPE_std']:.2f}%")
```

### How It Works

```
Data: [--------------------------------------------]

Split 1: [----------train----------][--test--]
Split 2: [-------------train-------------][--test--]
Split 3: [----------------train----------------][--test--]
Split 4: [-------------------train-------------------][--test--]
Split 5: [----------------------train----------------------][--test--]
```

Each split:
1. Uses all data up to that point for training
2. Tests on next `test_size` points
3. Moves forward in time

### Expanding vs Sliding Window

```python
# Expanding window (default): train size grows
cv_results = evaluator.time_series_cv(
    model, data, n_splits=5, test_size=30
)

# Sliding window: train size fixed
cv_results = evaluator.time_series_cv(
    model, data, n_splits=5, test_size=30, train_size=200
)
```

---

## Forecast Accuracy by Horizon

Analyze how accuracy changes with forecast horizon.

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate at different horizons
horizons = [1, 7, 14, 30]  # 1 day, 1 week, 2 weeks, 1 month
horizon_results = evaluator.forecast_accuracy_by_horizon(
    model=model,
    y_train=train,
    y_test=test,
    horizons=horizons
)

print(horizon_results)
```

**Output:**
```
   Horizon   RMSE    MAE   MAPE
0        1   3.21   2.45   1.12
1        7   5.87   4.32   1.98
2       14   8.14   6.21   2.84
3       30  12.45   9.87   4.51
```

**Interpretation:**
- Errors typically increase with forecast horizon
- If errors don't increase: Model may be too simple (just predicting mean)
- If errors increase rapidly: Consider ensemble or simpler model for long-term

### Visualize Horizon Accuracy

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(horizon_results['Horizon'], horizon_results['RMSE'], marker='o', label='RMSE')
plt.plot(horizon_results['Horizon'], horizon_results['MAE'], marker='s', label='MAE')
plt.xlabel('Forecast Horizon (days)')
plt.ylabel('Error')
plt.title('Forecast Accuracy by Horizon')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Statistical Tests

### Diebold-Mariano Test

Compare forecast accuracy of two models statistically.

```python
from scipy.stats import ttest_rel

# Get predictions from two models
pred1 = model1.predict(len(test))
pred2 = model2.predict(len(test))

# Calculate squared errors
errors1 = (test.values - pred1.values) ** 2
errors2 = (test.values - pred2.values) ** 2

# Test if errors are significantly different
statistic, p_value = ttest_rel(errors1, errors2)

if p_value < 0.05:
    if errors1.mean() < errors2.mean():
        print("Model 1 is significantly better (p < 0.05)")
    else:
        print("Model 2 is significantly better (p < 0.05)")
else:
    print("No significant difference between models (p >= 0.05)")
```

---

## Advanced Evaluation

### Prediction Intervals Coverage

Evaluate if prediction intervals have correct coverage.

```python
# Get predictions with intervals
predictions, lower, upper = model.predict_with_intervals(
    steps=len(test),
    confidence=0.95
)

# Calculate coverage
within_interval = (test.values >= lower.values) & (test.values <= upper.values)
coverage = within_interval.mean()

print(f"95% interval coverage: {coverage:.1%}")
print(f"Expected: 95%")

if coverage < 0.90:
    print("⚠ Intervals too narrow - underestimating uncertainty")
elif coverage > 0.98:
    print("⚠ Intervals too wide - overestimating uncertainty")
else:
    print("✓ Interval coverage is appropriate")
```

### Bias Analysis

```python
# Calculate bias over time
residuals = test.values - predictions.values
cumulative_bias = np.cumsum(residuals)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(residuals)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals Over Time')
plt.xlabel('Time')
plt.ylabel('Residual')

plt.subplot(1, 2, 2)
plt.plot(cumulative_bias)
plt.axhline(0, color='r', linestyle='--')
plt.title('Cumulative Bias')
plt.xlabel('Time')
plt.ylabel('Cumulative Residual')

plt.tight_layout()
plt.show()

# Statistical bias test
mean_residual = np.mean(residuals)
se = np.std(residuals) / np.sqrt(len(residuals))
t_stat = mean_residual / se

from scipy.stats import t
p_value = 2 * (1 - t.cdf(abs(t_stat), len(residuals) - 1))

if p_value < 0.05:
    print(f"⚠ Significant bias detected (p = {p_value:.4f})")
else:
    print(f"✓ No significant bias (p = {p_value:.4f})")
```

### Rolling Forecast Accuracy

```python
# Evaluate on rolling basis
window_size = 30
results = []

for i in range(window_size, len(test)):
    window_test = test.iloc[i-window_size:i]
    window_pred = predictions.iloc[i-window_size:i]
    
    metrics = evaluator.calculate_metrics(window_test.values, window_pred.values)
    results.append({
        'index': i,
        'date': test.index[i],
        'rmse': metrics['RMSE'],
        'mae': metrics['MAE']
    })

results_df = pd.DataFrame(results)

# Plot rolling accuracy
plt.figure(figsize=(12, 5))
plt.plot(results_df['date'], results_df['rmse'], label='RMSE')
plt.plot(results_df['date'], results_df['mae'], label='MAE')
plt.xlabel('Date')
plt.ylabel('Error')
plt.title(f'Rolling {window_size}-day Forecast Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Best Practices

### 1. Always Use Multiple Metrics

```python
# Don't rely on just one metric
metrics = evaluator.calculate_metrics(y_true, y_pred)

# Look at multiple metrics
print(f"RMSE: {metrics['RMSE']:.2f}")  # For large errors
print(f"MAE: {metrics['MAE']:.2f}")    # For average error
print(f"MAPE: {metrics['MAPE']:.2f}%") # For percentage error
print(f"R²: {metrics['R2']:.4f}")      # For fit quality
```

### 2. Perform Residual Analysis

```python
# Always check residuals
residual_stats = evaluator.residual_analysis(model, test, predictions)

if not residual_stats['is_white_noise']:
    print("⚠ Model may be missing important patterns")
```

### 3. Use Time Series Cross-Validation

```python
# Never use random k-fold for time series
# Always use time series CV
cv_results = evaluator.time_series_cv(model, data, n_splits=5)
```

### 4. Check Prediction Intervals

```python
# Evaluate interval coverage
predictions, lower, upper = model.predict_with_intervals(len(test))
coverage = ((test >= lower) & (test <= upper)).mean()
print(f"Interval coverage: {coverage:.1%}")
```

### 5. Compare to Baseline

```python
# Always compare to naive forecast
naive_forecast = test.shift(1).dropna()
naive_metrics = evaluator.calculate_metrics(
    test.iloc[1:].values,
    naive_forecast.values
)

print(f"Naive RMSE: {naive_metrics['RMSE']:.2f}")
print(f"Model RMSE: {metrics['RMSE']:.2f}")
print(f"Improvement: {(1 - metrics['RMSE']/naive_metrics['RMSE'])*100:.1f}%")
```

---

## Common Issues and Solutions

### Issue: MAPE is Infinite
**Cause:** Division by zero (actual value is zero)  
**Solution:** Use sMAPE or add small constant

```python
# Use sMAPE instead
metrics = evaluator.calculate_metrics(y_true, y_pred)
print(f"sMAPE: {metrics['sMAPE']:.2f}%")
```

### Issue: R² is Negative
**Cause:** Model worse than predicting mean  
**Solution:** Try different model or preprocessing

```python
if metrics['R2'] < 0:
    print("⚠ Model performing worse than mean baseline")
    print("Consider: different model, preprocessing, or hyperparameters")
```

### Issue: Residuals Show Patterns
**Cause:** Model missing information  
**Solution:** Add features, use more complex model, or increase model capacity

```python
# Check residual autocorrelation
residual_stats = evaluator.residual_analysis(model, test, predictions)
if not residual_stats['is_white_noise']:
    print("Try: adding features, using ensemble, or adjusting hyperparameters")
```

---

## References

- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
- Makridakis, S., et al. (2018). Statistical and Machine Learning forecasting methods.
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy.

