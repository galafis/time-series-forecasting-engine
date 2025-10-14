# 🔧 Preprocessing Module

## Overview

The preprocessing module provides comprehensive tools for preparing time series data for modeling. Proper preprocessing is crucial for accurate forecasting and includes handling missing values, detecting outliers, transforming data, and engineering features.

## TimeSeriesPreprocessor

The main class `TimeSeriesPreprocessor` provides a complete preprocessing pipeline with methods for:
- Missing value imputation
- Outlier detection and removal
- Data transformation
- Stationarity testing and transformation
- Feature engineering
- Seasonal decomposition

---

## Missing Value Imputation

### Methods

#### 1. Linear Interpolation
Fills missing values using linear interpolation between known values.

```python
from src.preprocessing import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()
clean_data = preprocessor.handle_missing_values(data, method='interpolate')
```

**Best for:** Smooth, continuous time series with occasional gaps

#### 2. Forward Fill
Propagates last known value forward.

```python
clean_data = preprocessor.handle_missing_values(data, method='ffill')
```

**Best for:** Step-like data, categorical time series

#### 3. Backward Fill
Propagates next known value backward.

```python
clean_data = preprocessor.handle_missing_values(data, method='bfill')
```

**Best for:** Data where future values are known

#### 4. Mean/Median Imputation
Fills missing values with mean or median.

```python
clean_data = preprocessor.handle_missing_values(data, method='mean')
clean_data = preprocessor.handle_missing_values(data, method='median')
```

**Best for:** Data with random missing values, no strong temporal pattern

#### 5. Seasonal Decomposition Imputation
Uses seasonal decomposition to intelligently fill missing values.

```python
clean_data = preprocessor.handle_missing_values(
    data, 
    method='seasonal',
    period=12  # for monthly data with yearly seasonality
)
```

**Best for:** Strongly seasonal data with gaps

### Example: Comparing Methods

```python
import pandas as pd
import numpy as np
from src.preprocessing import TimeSeriesPreprocessor

# Create data with missing values
dates = pd.date_range('2020-01-01', periods=100, freq='D')
values = np.sin(np.arange(100) * 2 * np.pi / 30) + np.random.normal(0, 0.1, 100)
data = pd.Series(values, index=dates)
data.iloc[20:25] = np.nan  # Create gap

preprocessor = TimeSeriesPreprocessor()

# Try different methods
interpolated = preprocessor.handle_missing_values(data, method='interpolate')
forward_filled = preprocessor.handle_missing_values(data, method='ffill')
mean_filled = preprocessor.handle_missing_values(data, method='mean')

print(f"Original missing: {data.isna().sum()}")
print(f"After imputation: {interpolated.isna().sum()}")
```

---

## Outlier Detection and Removal

### Methods

#### 1. IQR (Interquartile Range)
Detects outliers using the IQR method (Q1 - threshold×IQR, Q3 + threshold×IQR).

```python
from src.preprocessing import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()
clean_data = preprocessor.remove_outliers(
    data, 
    method='iqr',
    threshold=1.5  # Standard: 1.5, Strict: 3.0
)
```

**Best for:** Symmetric distributions, robust to extreme values

#### 2. Z-Score
Detects outliers based on standard deviations from mean.

```python
clean_data = preprocessor.remove_outliers(
    data,
    method='zscore',
    threshold=3.0  # Standard: 3.0, Strict: 2.5
)
```

**Best for:** Normally distributed data

#### 3. Modified Z-Score
Uses median absolute deviation (MAD), more robust than standard z-score.

```python
clean_data = preprocessor.remove_outliers(
    data,
    method='modified_zscore',
    threshold=3.5
)
```

**Best for:** Data with outliers affecting mean/std

#### 4. Isolation Forest
Machine learning-based anomaly detection.

```python
clean_data = preprocessor.remove_outliers(
    data,
    method='isolation_forest',
    contamination=0.05  # Expected proportion of outliers
)
```

**Best for:** Complex patterns, multivariate outliers

### Outlier Handling Strategies

```python
# Detect only (don't remove)
outliers_mask = preprocessor.detect_outliers(data, method='iqr')
print(f"Detected {outliers_mask.sum()} outliers")

# Remove outliers
data_no_outliers = preprocessor.remove_outliers(data, method='iqr')

# Replace outliers with interpolation
data_filled = data.copy()
outliers_mask = preprocessor.detect_outliers(data, method='iqr')
data_filled[outliers_mask] = np.nan
data_filled = preprocessor.handle_missing_values(data_filled, method='interpolate')

# Winsorization (cap at percentiles)
data_winsorized = preprocessor.winsorize(data, lower=0.05, upper=0.95)
```

---

## Data Transformation

### Methods

#### 1. Log Transformation
Stabilizes variance and handles exponential growth.

```python
from src.preprocessing import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()
transformed = preprocessor.transform(data, method='log')

# Inverse transform
original = preprocessor.inverse_transform(transformed, method='log')
```

**Best for:** Exponential trends, multiplicative seasonality

#### 2. Box-Cox Transformation
Automatically finds optimal power transformation.

```python
transformed = preprocessor.transform(data, method='boxcox')

# Inverse transform
original = preprocessor.inverse_transform(transformed, method='boxcox')
```

**Best for:** Stabilizing variance, normalizing distribution

#### 3. Min-Max Scaling
Scales data to [0, 1] range.

```python
scaled = preprocessor.scale(data, method='minmax')

# Inverse scale
original = preprocessor.inverse_scale(scaled, method='minmax')
```

**Best for:** Neural networks, preserving zero values

#### 4. Standard Scaling
Scales to zero mean and unit variance.

```python
scaled = preprocessor.scale(data, method='standard')

# Inverse scale
original = preprocessor.inverse_scale(scaled, method='standard')
```

**Best for:** Algorithms sensitive to scale, maintaining distribution shape

#### 5. Differencing
Makes series stationary by taking differences.

```python
# First difference
diff_data = preprocessor.difference(data, periods=1)

# Seasonal difference
seasonal_diff = preprocessor.difference(data, periods=12)  # monthly

# Inverse differencing
original = preprocessor.inverse_difference(diff_data, original_values=data.iloc[0])
```

**Best for:** Removing trends, achieving stationarity

---

## Stationarity Testing

### Augmented Dickey-Fuller Test

```python
from src.preprocessing import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()

# Test stationarity
is_stationary, test_result = preprocessor.test_stationarity(data)

print(f"Is stationary: {is_stationary}")
print(f"ADF Statistic: {test_result['adf_statistic']:.4f}")
print(f"p-value: {test_result['p_value']:.4f}")
print(f"Critical values: {test_result['critical_values']}")

# Make stationary if needed
if not is_stationary:
    stationary_data = preprocessor.make_stationary(data)
```

### Making Data Stationary

```python
# Automatic method selection
stationary_data = preprocessor.make_stationary(data)

# Manual methods
# 1. Differencing
diff_data = preprocessor.difference(data, periods=1)

# 2. Log + Differencing
log_data = preprocessor.transform(data, method='log')
stationary_data = preprocessor.difference(log_data, periods=1)

# 3. Detrending
detrended = preprocessor.detrend(data, method='linear')
```

---

## Feature Engineering

### Lag Features

```python
from src.preprocessing import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()

# Create lag features
features = preprocessor.create_lag_features(
    data,
    lags=[1, 2, 3, 7, 14, 30]  # 1 day, 2 days, ..., 30 days
)

print(features.head())
# Output:
#              value  lag_1  lag_2  lag_3  lag_7  lag_14  lag_30
# 2020-01-31   100.0   99.5   99.0   98.5   95.0   90.0    80.0
```

**Best for:** Autoregressive models, capturing temporal patterns

### Rolling Statistics

```python
# Create rolling features
features = preprocessor.create_rolling_features(
    data,
    windows=[7, 14, 30],
    statistics=['mean', 'std', 'min', 'max']
)

print(features.head())
# Output:
#              value  rolling_mean_7  rolling_std_7  rolling_min_7  ...
# 2020-01-08   100.0           98.5            2.1           95.0  ...
```

**Best for:** Smoothing, trend detection, volatility measures

### Time-Based Features

```python
# Extract time features
features = preprocessor.create_time_features(data)

print(features.head())
# Output:
#              value  day_of_week  day_of_month  month  quarter  year  is_weekend
# 2020-01-01   100.0            2             1      1        1  2020           0
```

**Best for:** Capturing calendar patterns, seasonality

### Seasonal Indicators

```python
# Create seasonal dummies
features = preprocessor.create_seasonal_features(
    data,
    periods={'monthly': 12, 'quarterly': 4}
)
```

**Best for:** Handling seasonal effects in linear models

### Complete Feature Pipeline

```python
from src.preprocessing import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()

# Create comprehensive feature set
features = preprocessor.create_features(
    data,
    lags=[1, 2, 3, 7, 14, 30],
    rolling_windows=[7, 14, 30],
    time_features=True,
    seasonal_features=True
)

print(f"Created {features.shape[1]} features")
```

---

## Seasonal Decomposition

### Additive Decomposition
Suitable when seasonal variations are constant.

```python
from src.preprocessing import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()

# Decompose series
decomposition = preprocessor.decompose(
    data,
    model='additive',
    period=12  # monthly data with yearly seasonality
)

# Access components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.residual

# Visualize
from src.visualization import TimeSeriesVisualizer
visualizer = TimeSeriesVisualizer()
visualizer.plot_decomposition(trend, seasonal, residual)
```

**Formula:** Y = Trend + Seasonal + Residual

### Multiplicative Decomposition
Suitable when seasonal variations increase with level.

```python
decomposition = preprocessor.decompose(
    data,
    model='multiplicative',
    period=12
)
```

**Formula:** Y = Trend × Seasonal × Residual

### Using Decomposition for Forecasting

```python
# Decompose training data
decomposition = preprocessor.decompose(train_data, model='additive', period=12)

# Forecast each component
from src.models import ARIMAForecaster

# Forecast trend
trend_model = ARIMAForecaster()
trend_model.fit(decomposition.trend.dropna())
trend_forecast = trend_model.predict(steps=12)

# Extend seasonal pattern
seasonal_forecast = decomposition.seasonal[-12:].values

# Combine forecasts
final_forecast = trend_forecast + seasonal_forecast
```

---

## Complete Preprocessing Pipeline

### Example: End-to-End Pipeline

```python
import pandas as pd
from src.preprocessing import TimeSeriesPreprocessor
from src.visualization import TimeSeriesVisualizer

# Load data
data = pd.read_csv('sales.csv', index_col='date', parse_dates=True)['sales']

# Initialize preprocessor
preprocessor = TimeSeriesPreprocessor()
visualizer = TimeSeriesVisualizer()

# 1. Visualize original data
visualizer.plot_time_series(data, title='Original Data')

# 2. Handle missing values
print(f"Missing values: {data.isna().sum()}")
if data.isna().sum() > 0:
    data = preprocessor.handle_missing_values(data, method='interpolate')
    print("✓ Filled missing values")

# 3. Detect and handle outliers
outliers_mask = preprocessor.detect_outliers(data, method='iqr')
print(f"Detected {outliers_mask.sum()} outliers")
if outliers_mask.sum() > 0:
    data[outliers_mask] = np.nan
    data = preprocessor.handle_missing_values(data, method='interpolate')
    print("✓ Handled outliers")

# 4. Check stationarity
is_stationary, test_result = preprocessor.test_stationarity(data)
print(f"Stationary: {is_stationary} (p-value: {test_result['p_value']:.4f})")

if not is_stationary:
    # Transform to stabilize variance
    data_transformed = preprocessor.transform(data, method='log')
    
    # Difference to remove trend
    data_stationary = preprocessor.difference(data_transformed, periods=1)
    print("✓ Made data stationary")
else:
    data_stationary = data

# 5. Seasonal decomposition (optional)
decomposition = preprocessor.decompose(data, model='additive', period=12)
visualizer.plot_decomposition(
    decomposition.trend,
    decomposition.seasonal,
    decomposition.residual
)

# 6. Create features for ML models
features = preprocessor.create_features(
    data,
    lags=[1, 7, 14, 30],
    rolling_windows=[7, 14, 30],
    time_features=True
)
print(f"✓ Created {features.shape[1]} features")

# 7. Scale features (for neural networks)
features_scaled = preprocessor.scale(features, method='standard')

# Now ready for modeling!
```

---

## Best Practices

### 1. Order of Operations

Recommended preprocessing order:
1. Handle missing values
2. Detect and handle outliers
3. Transform to stabilize variance (log, Box-Cox)
4. Difference to achieve stationarity
5. Scale/normalize (if needed for model)
6. Create features

### 2. Data Splitting

Always split before preprocessing to avoid data leakage:

```python
# Split first
train = data[:int(len(data) * 0.8)]
test = data[int(len(data) * 0.8):]

# Then preprocess
preprocessor = TimeSeriesPreprocessor()

# Fit on train only
train_clean = preprocessor.handle_missing_values(train, method='interpolate')
train_scaled = preprocessor.scale(train_clean, method='standard')

# Apply same transformations to test
test_clean = preprocessor.handle_missing_values(test, method='interpolate')
test_scaled = preprocessor.scale(test_clean, method='standard')
```

### 3. Preserving Transform Parameters

```python
# Fit on training data
preprocessor = TimeSeriesPreprocessor()
train_scaled = preprocessor.scale(train, method='standard')

# Get parameters
params = preprocessor.get_scale_params()

# Apply to new data with same parameters
test_scaled = preprocessor.scale(test, method='standard', params=params)
```

### 4. Validation

Always validate preprocessing results:

```python
# Check for remaining NaN values
assert not data_clean.isna().any(), "Still has missing values"

# Check value ranges
print(f"Min: {data_clean.min():.2f}, Max: {data_clean.max():.2f}")

# Verify stationarity
is_stationary, _ = preprocessor.test_stationarity(data_clean)
assert is_stationary, "Data is not stationary"

# Check feature correlations
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()
```

---

## Advanced Topics

### Handling Multiple Time Series

```python
# Process multiple series consistently
preprocessor = TimeSeriesPreprocessor()

series_list = [series1, series2, series3]
processed_list = []

for series in series_list:
    # Apply same preprocessing
    clean = preprocessor.handle_missing_values(series, method='interpolate')
    clean = preprocessor.remove_outliers(clean, method='iqr')
    processed_list.append(clean)
```

### Custom Transformations

```python
def custom_transform(x):
    """Custom transformation function."""
    return np.sqrt(x + 1)

def custom_inverse(x):
    """Inverse of custom transformation."""
    return x**2 - 1

# Apply custom transformation
transformed = data.apply(custom_transform)
original = transformed.apply(custom_inverse)
```

### Handling Seasonal Breaks

```python
# Handle structural breaks in seasonality
preprocessor = TimeSeriesPreprocessor()

# Decompose before and after break
before_break = data['2020':'2021']
after_break = data['2022':]

decomp_before = preprocessor.decompose(before_break, model='additive', period=12)
decomp_after = preprocessor.decompose(after_break, model='additive', period=12)
```

---

## Troubleshooting

### Issue: Stationarity Test Fails
**Solution:** Try combining transformations
```python
# Log + difference
log_data = preprocessor.transform(data, method='log')
stationary = preprocessor.difference(log_data, periods=1)
```

### Issue: Box-Cox Requires Positive Values
**Solution:** Shift data before transformation
```python
min_val = data.min()
if min_val <= 0:
    data_shifted = data - min_val + 1
    transformed = preprocessor.transform(data_shifted, method='boxcox')
```

### Issue: Too Many Missing Values
**Solution:** Use seasonal imputation or model-based imputation
```python
# Try seasonal imputation
filled = preprocessor.handle_missing_values(data, method='seasonal', period=12)

# Or use forward/backward fill combination
filled = data.fillna(method='ffill').fillna(method='bfill')
```

### Issue: Outliers Due to Rare Events
**Solution:** Don't remove them - they may be important!
```python
# Keep outliers but flag them
outliers_mask = preprocessor.detect_outliers(data, method='iqr')
data_with_flags = data.copy()
data_with_flags['is_outlier'] = outliers_mask
```

---

## References

- Box, G. E., & Cox, D. R. (1964). An analysis of transformations. Journal of the Royal Statistical Society.
- Tukey, J. W. (1977). Exploratory data analysis.
- Cleveland, R. B., et al. (1990). STL: A seasonal-trend decomposition procedure based on loess.
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series.

