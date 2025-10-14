# 📊 Models Module

## Overview

The models module provides a comprehensive collection of time series forecasting algorithms, from classical statistical models to modern deep learning architectures. All models inherit from a common `BaseForecaster` interface, ensuring consistent API and seamless interoperability.

## Available Models

### 1. ARIMAForecaster

**Type:** Statistical Model  
**Best For:** Linear trends, seasonal patterns, stationary or near-stationary time series

#### Description
ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical model for time series forecasting. It combines:
- **AR (AutoRegressive):** Uses past values to predict future values
- **I (Integrated):** Differences the data to achieve stationarity
- **MA (Moving Average):** Uses past forecast errors

#### Features
- Automatic parameter selection with `auto_select=True`
- Support for seasonal ARIMA (SARIMA)
- Confidence intervals for predictions
- AIC/BIC model selection criteria

#### Usage Example

```python
from src.models import ARIMAForecaster

# Simple ARIMA with auto parameter selection
model = ARIMAForecaster(auto_select=True)
model.fit(train_data)
forecast = model.predict(steps=30)

# Manual parameter specification
model = ARIMAForecaster(order=(2, 1, 2))
model.fit(train_data)
forecast, lower, upper = model.predict_with_intervals(steps=30, confidence=0.95)

# Seasonal ARIMA
model = ARIMAForecaster(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)  # 12-month seasonality
)
```

#### Parameters
- `order`: Tuple (p, d, q) for ARIMA parameters
- `seasonal_order`: Tuple (P, D, Q, s) for seasonal parameters
- `auto_select`: Boolean to enable automatic parameter selection
- `information_criterion`: 'aic' or 'bic' for model selection

---

### 2. ProphetForecaster

**Type:** Machine Learning Model  
**Best For:** Multiple seasonalities, holidays, missing data, outliers

#### Description
Facebook Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

#### Features
- Handles multiple seasonalities automatically
- Robust to missing data and outliers
- Support for holiday effects
- Flexible trend changepoints
- Interpretable parameters

#### Usage Example

```python
from src.models import ProphetForecaster

# Basic Prophet model
model = ProphetForecaster()
model.fit(train_data)
forecast = model.predict(steps=30)

# Advanced configuration
model = ProphetForecaster(
    growth='linear',  # or 'logistic'
    changepoint_prior_scale=0.05,  # Flexibility of trend changes
    seasonality_prior_scale=10.0,  # Strength of seasonality
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(train_data)

# Add custom regressors
model.fit(train_data, X=external_features)
forecast = model.predict(steps=30, X=future_features)

# Visualize components
model.plot_components()
```

#### Parameters
- `growth`: 'linear' or 'logistic' growth
- `changepoint_prior_scale`: Controls trend flexibility (0.001-0.5)
- `seasonality_prior_scale`: Controls seasonality strength (0.01-10)
- `seasonality_mode`: 'additive' or 'multiplicative'
- `yearly_seasonality`, `weekly_seasonality`, `daily_seasonality`: Boolean flags

---

### 3. LSTMForecaster

**Type:** Deep Learning Model  
**Best For:** Complex non-linear patterns, long-term dependencies

#### Description
LSTM (Long Short-Term Memory) networks are a type of recurrent neural network capable of learning long-term dependencies. They are particularly effective for sequences with complex temporal patterns.

#### Features
- Multiple LSTM layers support
- Dropout for regularization
- Early stopping to prevent overfitting
- GPU acceleration (if available)
- Customizable architecture

#### Usage Example

```python
from src.models import LSTMForecaster

# Basic LSTM model
model = LSTMForecaster(
    lookback=30,  # Use 30 past values
    lstm_units=64,
    num_layers=2,
    epochs=50,
    batch_size=32
)
model.fit(train_data)
forecast = model.predict(steps=30)

# Advanced architecture
model = LSTMForecaster(
    lookback=60,
    lstm_units=128,
    num_layers=3,
    dropout=0.2,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    early_stopping=True,
    patience=10
)
model.fit(train_data)
```

#### Parameters
- `lookback`: Number of past time steps to use for prediction
- `lstm_units`: Number of LSTM units per layer
- `num_layers`: Number of LSTM layers (1-5)
- `dropout`: Dropout rate for regularization (0.0-0.5)
- `epochs`: Number of training epochs
- `batch_size`: Training batch size
- `validation_split`: Fraction of data for validation
- `early_stopping`: Enable early stopping
- `patience`: Epochs to wait before early stopping

#### Architecture

```
Input (lookback timesteps)
    ↓
LSTM Layer 1 (units)
    ↓
Dropout
    ↓
LSTM Layer 2 (units)
    ↓
Dropout
    ↓
...
    ↓
Dense Layer
    ↓
Output (1 value)
```

---

### 4. EnsembleForecaster

**Type:** Hybrid Model  
**Best For:** Maximum accuracy, robust predictions, combining strengths of multiple models

#### Description
Ensemble methods combine predictions from multiple models to achieve better accuracy and robustness than any individual model. The ensemble can use different aggregation strategies.

#### Features
- Multiple aggregation methods (average, weighted, median)
- Automatic model training
- Individual model access
- Prediction interval computation

#### Usage Example

```python
from src.models import ARIMAForecaster, ProphetForecaster, LSTMForecaster, EnsembleForecaster

# Create individual models
arima = ARIMAForecaster(auto_select=True)
prophet = ProphetForecaster()
lstm = LSTMForecaster(lookback=30, epochs=50)

# Simple averaging ensemble
ensemble = EnsembleForecaster(
    forecasters=[arima, prophet, lstm],
    method='average'
)
ensemble.fit(train_data)
forecast = ensemble.predict(steps=30)

# Weighted ensemble (weights must sum to 1)
ensemble = EnsembleForecaster(
    forecasters=[arima, prophet, lstm],
    method='weighted',
    weights=[0.3, 0.4, 0.3]  # Give more weight to Prophet
)
ensemble.fit(train_data)
forecast = ensemble.predict(steps=30)

# Median ensemble (robust to outliers)
ensemble = EnsembleForecaster(
    forecasters=[arima, prophet, lstm],
    method='median'
)

# Get individual predictions
individual_preds = ensemble.get_individual_predictions(steps=30)
print(individual_preds['ARIMAForecaster'])
print(individual_preds['ProphetForecaster'])
```

#### Aggregation Methods
- **Average:** Simple mean of all predictions
- **Weighted:** Weighted average (useful when models have different accuracy)
- **Median:** Robust to outlier predictions

#### Choosing Weights
Determine weights based on cross-validation performance:
```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate each model
arima_metrics = evaluator.evaluate_model(arima, train, test)
prophet_metrics = evaluator.evaluate_model(prophet, train, test)
lstm_metrics = evaluator.evaluate_model(lstm, train, test)

# Calculate weights inversely proportional to RMSE
errors = [
    arima_metrics['RMSE'],
    prophet_metrics['RMSE'],
    lstm_metrics['RMSE']
]
inv_errors = [1/e for e in errors]
weights = [ie/sum(inv_errors) for ie in inv_errors]

# Create weighted ensemble
ensemble = EnsembleForecaster(
    forecasters=[arima, prophet, lstm],
    method='weighted',
    weights=weights
)
```

---

## Base Forecaster API

All models inherit from `BaseForecaster` and implement the following interface:

### Core Methods

#### `fit(y, X=None, **kwargs)`
Train the model on time series data.
- **y:** Time series data (pd.Series)
- **X:** Optional exogenous features (pd.DataFrame)
- **Returns:** self

#### `predict(steps, X=None, **kwargs)`
Generate point forecasts.
- **steps:** Number of steps to forecast (int)
- **X:** Optional future exogenous features
- **Returns:** Forecasted values (pd.Series)

#### `predict_with_intervals(steps, confidence=0.95, X=None, **kwargs)`
Generate forecasts with prediction intervals.
- **steps:** Number of steps to forecast
- **confidence:** Confidence level (0.0-1.0)
- **X:** Optional future exogenous features
- **Returns:** Tuple of (predictions, lower_bound, upper_bound)

#### `save(path)`
Save trained model to disk.
- **path:** File path (str)

#### `load(path)`
Load trained model from disk.
- **path:** File path (str)
- **Returns:** Loaded model instance

#### `get_params()`
Get model parameters and configuration.
- **Returns:** Dictionary of parameters

### Properties

- `is_fitted`: Boolean indicating if model is trained
- `name`: Model name string
- `residuals_`: Training residuals (if available)

---

## Model Selection Guide

| Scenario | Recommended Model | Alternative |
|----------|------------------|-------------|
| Short-term forecast (< 30 steps) | ARIMA | Prophet |
| Long-term forecast (> 90 steps) | Prophet | Ensemble |
| Strong seasonality | Prophet, SARIMA | LSTM |
| Multiple seasonalities | Prophet | Ensemble |
| Non-linear patterns | LSTM | Ensemble |
| Missing data | Prophet | Preprocessing + ARIMA |
| Need interpretability | ARIMA, Prophet | - |
| Maximum accuracy | Ensemble | LSTM |
| Limited training data | ARIMA, Prophet | - |
| Large dataset | LSTM | Prophet |
| Real-time predictions | ARIMA | Prophet |

---

## Performance Considerations

### Training Time
- **ARIMA:** Fast (seconds) - O(n)
- **Prophet:** Medium (seconds to minutes) - O(n log n)
- **LSTM:** Slow (minutes to hours) - O(n × epochs)
- **Ensemble:** Sum of individual models

### Memory Usage
- **ARIMA:** Low (< 100 MB)
- **Prophet:** Low-Medium (< 500 MB)
- **LSTM:** High (> 1 GB with GPU)
- **Ensemble:** Sum of individual models

### Prediction Speed
All models: O(steps) - very fast for inference

---

## Advanced Topics

### Multivariate Time Series

Use exogenous features with Prophet:
```python
# Prepare features
features = pd.DataFrame({
    'temperature': [...],
    'holiday': [...],
    'promotion': [...]
}, index=dates)

# Train with features
model = ProphetForecaster()
model.fit(target_series, X=features)

# Predict with future features
future_features = pd.DataFrame({
    'temperature': [...],  # Future temperatures
    'holiday': [...],      # Future holidays
    'promotion': [...]     # Future promotions
}, index=future_dates)

forecast = model.predict(steps=30, X=future_features)
```

### Transfer Learning

Reuse trained LSTM model:
```python
# Train on one dataset
model = LSTMForecaster(lookback=30, lstm_units=64)
model.fit(data1)
model.save('pretrained_model.h5')

# Fine-tune on another dataset
model2 = LSTMForecaster(lookback=30, lstm_units=64)
model2.load('pretrained_model.h5')
model2.fit(data2, epochs=10)  # Fewer epochs for fine-tuning
```

### Custom Models

Extend BaseForecaster to create custom models:
```python
from src.models.base_forecaster import BaseForecaster
import pandas as pd

class CustomForecaster(BaseForecaster):
    def __init__(self):
        super().__init__(name="CustomForecaster")
    
    def fit(self, y, X=None, **kwargs):
        # Your training logic
        self.is_fitted = True
        return self
    
    def predict(self, steps, X=None, **kwargs):
        # Your prediction logic
        return pd.Series([...])
    
    def predict_with_intervals(self, steps, confidence=0.95, X=None, **kwargs):
        # Your interval prediction logic
        predictions = self.predict(steps, X, **kwargs)
        lower = predictions - 1.96 * std
        upper = predictions + 1.96 * std
        return predictions, lower, upper
```

---

## Troubleshooting

### ARIMA Not Converging
- Try different order parameters
- Check if series is stationary (use differencing)
- Reduce model complexity

### Prophet Slow Training
- Reduce changepoint_prior_scale
- Disable unnecessary seasonalities
- Use smaller dataset for testing

### LSTM Overfitting
- Increase dropout rate
- Add more training data
- Reduce model complexity (fewer layers/units)
- Use early stopping

### Ensemble Predictions Unrealistic
- Check individual model predictions
- Adjust weights based on validation performance
- Use median instead of average for robustness

---

## References

- **ARIMA:** Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). Time series analysis: forecasting and control.
- **Prophet:** Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.
- **LSTM:** Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

