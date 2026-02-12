# Time Series Forecasting Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

[English](#english) | [Português](#português)

---

<a name="english"></a>
## 🇬🇧 English

### 📊 Overview

**Time Series Forecasting Engine** is a Python framework for time series forecasting that brings together statistical models (ARIMA), machine learning (Prophet), and deep learning (LSTM) behind a consistent interface. It includes preprocessing, evaluation, and visualization utilities.

Built for data scientists, ML engineers, and researchers working on problems like demand forecasting, financial predictions, and energy consumption analysis.

### ✨ Key Features

#### 🎯 Multiple Forecasting Algorithms

| Model | Type | Best For | Complexity |
|-------|------|----------|------------|
| **ARIMA/SARIMA** | Statistical | Linear trends, seasonal patterns | Low |
| **Prophet** | ML-based | Multiple seasonalities, holidays | Medium |
| **LSTM** | Deep Learning | Complex non-linear patterns | High |
| **Ensemble** | Hybrid | Maximum accuracy, robust predictions | High |

#### 🔧 Comprehensive Preprocessing

- **Missing Value Imputation**
  - Linear interpolation
  - Forward/backward fill
  - Mean/median imputation
  - Seasonal decomposition-based filling

- **Outlier Detection & Removal**
  - IQR (Interquartile Range) method
  - Z-score method
  - Modified Z-score
  - Isolation Forest

- **Data Transformation**
  - Log transformation
  - Box-Cox transformation
  - Min-Max scaling
  - Standard scaling
  - Differencing for stationarity

- **Feature Engineering**
  - Lag features (1-30 lags)
  - Rolling statistics (mean, std, min, max)
  - Time-based features (day, month, quarter, year)
  - Seasonal indicators

#### 📈 Advanced Evaluation Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **MAE** | Mean Absolute Error | General accuracy |
| **RMSE** | Root Mean Squared Error | Penalizes large errors |
| **MAPE** | Mean Absolute Percentage Error | Relative accuracy |
| **sMAPE** | Symmetric MAPE | Balanced percentage error |
| **R²** | Coefficient of Determination | Model fit quality |
| **MASE** | Mean Absolute Scaled Error | Benchmark comparison |

#### 📊 Rich Visualizations

- Forecast plots with prediction intervals
- Residual diagnostics (ACF, PACF, Q-Q plots)
- Model comparison charts
- Interactive Plotly dashboards
- Seasonal decomposition plots
- Error distribution analysis

### 🏗️ Architecture

```
time-series-forecasting-engine/
├── src/
│   ├── models/                    # Forecasting models
│   │   ├── base_forecaster.py     # Abstract base class
│   │   ├── arima_forecaster.py    # ARIMA/SARIMA implementation
│   │   ├── prophet_forecaster.py  # Facebook Prophet wrapper
│   │   ├── lstm_forecaster.py     # LSTM neural network
│   │   └── ensemble_forecaster.py # Ensemble methods
│   ├── preprocessing/             # Data preprocessing
│   │   └── preprocessor.py        # Complete preprocessing pipeline
│   ├── evaluation/                # Model evaluation
│   │   └── evaluator.py           # Metrics and diagnostics
│   └── visualization/             # Visualization tools
│       └── visualizer.py          # Plotting utilities
├── examples/                      # Usage examples
│   └── complete_example.py        # End-to-end example
├── tests/                         # Unit tests
│   └── test_models.py             # Model tests
├── notebooks/                     # Jupyter notebooks
├── data/                          # Data directory
│   ├── raw/                       # Raw data
│   └── processed/                 # Processed data
├── models/                        # Saved models
├── config/                        # Configuration files
├── requirements.txt               # Python dependencies
└── setup.py                       # Package setup
```

### 🚀 Quick Start

#### Installation

```bash
# Clone the repository
git clone https://github.com/galafis/time-series-forecasting-engine.git
cd time-series-forecasting-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

#### Basic Usage Example

```python
import pandas as pd
import numpy as np
from models import ARIMAForecaster, ProphetForecaster, LSTMForecaster, EnsembleForecaster
from preprocessing import TimeSeriesPreprocessor
from evaluation import ModelEvaluator
from visualization import TimeSeriesVisualizer

# Load your time series data
data = pd.read_csv('data/sales.csv', index_col='date', parse_dates=True)
ts = data['sales']

# Split into train/test
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# 1. Preprocessing
preprocessor = TimeSeriesPreprocessor()

# Handle missing values
train_clean = preprocessor.impute_missing(train, method='interpolation')

# Remove outliers
train_clean = preprocessor.remove_outliers(train_clean, method='iqr')

# Check stationarity and difference if needed
if not preprocessor.is_stationary(train_clean):
    train_clean = preprocessor.make_stationary(train_clean)

# 2. Model Training - ARIMA
arima = ARIMAForecaster(order=(2, 1, 2))
arima.fit(train_clean)
arima_forecast = arima.predict(steps=len(test))

# 3. Model Training - Prophet
prophet = ProphetForecaster()
prophet.fit(train_clean)
prophet_forecast = prophet.predict(steps=len(test))

# 4. Model Training - LSTM
lstm = LSTMForecaster(lookback=30, epochs=50)
lstm.fit(train_clean)
lstm_forecast = lstm.predict(steps=len(test))

# 5. Ensemble Model
ensemble = EnsembleForecaster(models=[arima, prophet, lstm], weights=[0.3, 0.4, 0.3])
ensemble_forecast = ensemble.predict(steps=len(test))

# 6. Evaluation
evaluator = ModelEvaluator()

print("ARIMA Metrics:")
arima_metrics = evaluator.calculate_metrics(test.values, arima_forecast.values)
print(f"  RMSE: {arima_metrics['RMSE']:.2f}")
print(f"  MAE: {arima_metrics['MAE']:.2f}")
print(f"  MAPE: {arima_metrics['MAPE']:.2f}%")

print("\nProphet Metrics:")
prophet_metrics = evaluator.calculate_metrics(test.values, prophet_forecast.values)
print(f"  RMSE: {prophet_metrics['RMSE']:.2f}")
print(f"  MAE: {prophet_metrics['MAE']:.2f}")
print(f"  MAPE: {prophet_metrics['MAPE']:.2f}%")

print("\nEnsemble Metrics:")
ensemble_metrics = evaluator.calculate_metrics(test.values, ensemble_forecast.values)
print(f"  RMSE: {ensemble_metrics['RMSE']:.2f}")
print(f"  MAE: {ensemble_metrics['MAE']:.2f}")
print(f"  MAPE: {ensemble_metrics['MAPE']:.2f}%")

# 7. Visualization
visualizer = TimeSeriesVisualizer()

# Plot forecasts
visualizer.plot_forecast(
    train=train,
    test=test,
    forecasts={'ARIMA': arima_forecast, 'Prophet': prophet_forecast, 'Ensemble': ensemble_forecast},
    title='Sales Forecasting Comparison'
)

# Plot residuals
visualizer.plot_residuals(test.values, ensemble_forecast.values)

# Save model
ensemble.save('models/sales_ensemble_model.pkl')
```

### 📚 Advanced Examples

#### Example 1: Seasonal Decomposition and Forecasting

```python
from preprocessing import TimeSeriesPreprocessor
from visualization import TimeSeriesVisualizer

preprocessor = TimeSeriesPreprocessor()
visualizer = TimeSeriesVisualizer()

# Decompose time series
decomposition = preprocessor.decompose(ts, model='additive', period=12)

# Visualize components
visualizer.plot_decomposition(decomposition)

# Forecast each component separately
trend_forecast = arima.fit(decomposition.trend.dropna()).predict(12)
seasonal_forecast = decomposition.seasonal[-12:]  # Repeat last season
residual_forecast = np.zeros(12)  # Assume zero residuals

# Combine forecasts
final_forecast = trend_forecast + seasonal_forecast + residual_forecast
```

#### Example 2: Cross-Validation for Time Series

```python
from evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Time series cross-validation
cv_results = evaluator.time_series_cv(
    data=ts,
    model=ARIMAForecaster(order=(2,1,2)),
    n_splits=5,
    test_size=30
)

print(f"Average RMSE: {np.mean(cv_results['rmse']):.2f}")
print(f"Average MAE: {np.mean(cv_results['mae']):.2f}")
print(f"Std RMSE: {np.std(cv_results['rmse']):.2f}")
```

#### Example 3: Hyperparameter Tuning

```python
from models import ARIMAForecaster
from evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Grid search for ARIMA parameters
best_score = float('inf')
best_params = None

for p in range(0, 3):
    for d in range(0, 2):
        for q in range(0, 3):
            try:
                model = ARIMAForecaster(order=(p, d, q))
                model.fit(train)
                forecast = model.predict(len(test))
                metrics = evaluator.calculate_metrics(test.values, forecast.values)
                
                if metrics['RMSE'] < best_score:
                    best_score = metrics['RMSE']
                    best_params = (p, d, q)
                    
            except:
                continue

print(f"Best ARIMA parameters: {best_params}")
print(f"Best RMSE: {best_score:.2f}")
```

### 🎯 Use Cases

#### 1. **Demand Forecasting**
Predict product demand for inventory optimization and supply chain management.

```python
# Retail sales forecasting
model = ProphetForecaster()
model.fit(historical_sales)
demand_forecast = model.predict(steps=30)  # Next 30 days
```

#### 2. **Financial Predictions**
Forecast stock prices, currency exchange rates, or cryptocurrency values.

```python
# Stock price forecasting
lstm = LSTMForecaster(lookback=60, layers=[50, 50], dropout=0.2)
lstm.fit(stock_prices)
price_forecast = lstm.predict(steps=10)
```

#### 3. **Energy Consumption**
Predict electricity demand for grid management and renewable energy integration.

```python
# Energy demand forecasting with seasonality
sarima = ARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,24))
sarima.fit(hourly_consumption)
energy_forecast = sarima.predict(steps=168)  # Next week
```

#### 4. **Weather Forecasting**
Predict temperature, precipitation, or other meteorological variables.

```python
# Temperature forecasting
ensemble = EnsembleForecaster(
    models=[ARIMAForecaster(), ProphetForecaster(), LSTMForecaster()],
    weights=[0.3, 0.4, 0.3]
)
ensemble.fit(temperature_data)
temp_forecast = ensemble.predict(steps=7)  # Next 7 days
```

### 📊 Supported Evaluation Metrics

All models can be evaluated using the built-in `ModelEvaluator` with metrics including MAE, RMSE, MAPE, sMAPE, R², and MASE. See the [Evaluation Documentation](src/evaluation/README.md) for details on each metric and guidance on which to use for your problem.

### 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### 📖 Documentation

Comprehensive documentation is available for all modules:

- **[Models Documentation](src/models/README.md)** - Complete guide to all forecasting models (ARIMA, Prophet, LSTM, Ensemble)
- **[Preprocessing Documentation](src/preprocessing/README.md)** - Data preparation and feature engineering guide
- **[Evaluation Documentation](src/evaluation/README.md)** - Model evaluation metrics and techniques
- **[Visualization Documentation](src/visualization/README.md)** - Plotting and visualization guide
- **[Architecture Overview](docs/architecture.md)** - System architecture and component interaction

#### 📓 Jupyter Notebooks

Interactive tutorials are available in the `notebooks/` directory:

- **[Tutorial 1: Introduction](notebooks/01_introducao_basica.ipynb)** - Getting started with basic forecasting
- **[Tutorial 2: Advanced Preprocessing](notebooks/02_preprocessamento_avancado.ipynb)** - Advanced data preparation techniques

### 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For detailed guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

### 🙏 Acknowledgments

- Facebook Prophet team for the excellent forecasting library
- Statsmodels contributors for ARIMA implementation
- TensorFlow/Keras team for deep learning framework

---

## ❓ Frequently Asked Questions (FAQ)

### General Questions

**Q: Which model should I use for my time series?**  
A: It depends on your data characteristics:
- **ARIMA**: Good for stationary or near-stationary data with linear trends
- **Prophet**: Excellent for data with multiple seasonalities and holidays
- **LSTM**: Best for complex non-linear patterns with long-term dependencies
- **Ensemble**: Combines multiple models for maximum accuracy

See the [Model Selection Guide](src/models/README.md#model-selection-guide) for detailed comparison.

**Q: How much data do I need?**  
A: Minimum recommendations:
- **ARIMA**: 50-100 observations
- **Prophet**: 100-200 observations (at least 2 seasons)
- **LSTM**: 500+ observations (more data = better performance)
- Generally, more data leads to better forecasts

**Q: Can I use external features (exogenous variables)?**  
A: Yes! Prophet and LSTM support external features. See the [documentation](src/models/README.md) for examples.

**Q: How do I handle missing values?**  
A: Multiple methods are available:
- Interpolation (recommended for most cases)
- Forward/backward fill
- Mean/median imputation
- Seasonal decomposition-based filling

See [Preprocessing Documentation](src/preprocessing/README.md#missing-value-imputation).

### Technical Questions

**Q: Why is my ARIMA model not converging?**  
A: Common solutions:
1. Check if your data is stationary (use ADF test)
2. Try differencing your data
3. Reduce model complexity (lower p, d, q values)
4. Remove outliers and extreme values

**Q: Prophet training is slow. How can I speed it up?**  
A: Try these approaches:
1. Reduce `changepoint_prior_scale`
2. Disable unnecessary seasonalities
3. Use a smaller dataset for initial testing
4. Consider using ARIMA for faster training

**Q: My LSTM model is overfitting. What should I do?**  
A: Strategies to reduce overfitting:
1. Increase dropout rate (0.2-0.5)
2. Reduce model complexity (fewer layers/units)
3. Add more training data
4. Use early stopping
5. Apply regularization

**Q: How do I save and load trained models?**  
A: All models support save/load:
```python
# Save
model.save('my_model.pkl')

# Load
from src.models import ARIMAForecaster
loaded_model = ARIMAForecaster.load('my_model.pkl')
```

### Evaluation Questions

**Q: Which metric should I use to evaluate my model?**  
A: Depends on your needs:
- **RMSE**: Penalizes large errors, good for most cases
- **MAE**: More robust to outliers
- **MAPE**: Scale-independent, good for comparing across datasets
- **R²**: Measures explained variance, good for model fit assessment

See [Evaluation Documentation](src/evaluation/README.md) for detailed explanation.

**Q: My R² is negative. Is something wrong?**  
A: A negative R² means your model performs worse than simply predicting the mean. This indicates:
1. Model is inappropriate for the data
2. Poor hyperparameter selection
3. Insufficient preprocessing
4. Data leakage in train/test split

**Q: How do I perform cross-validation for time series?**  
A: Use time series cross-validation (not regular k-fold):
```python
cv_results = evaluator.time_series_cv(
    model=model,
    data=data,
    n_splits=5,
    test_size=30
)
```

See [Evaluation Documentation](src/evaluation/README.md#time-series-cross-validation).

### Troubleshooting

**Q: I'm getting import errors. What's wrong?**  
A: Make sure you've installed the package:
```bash
pip install -r requirements.txt
pip install -e .
```

**Q: TensorFlow/LSTM is not working.**  
A: Check TensorFlow installation:
```bash
pip install tensorflow>=2.13.0
```
For GPU support, install `tensorflow-gpu`.

**Q: Prophet is giving warnings about holidays.**  
A: This is normal if you haven't specified holidays. To suppress:
```python
import logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
```

**Q: My forecasts look unrealistic (too high/low).**  
A: Check these:
1. Verify data preprocessing (especially transformations)
2. Check for data leakage
3. Validate model assumptions
4. Try simpler models first (ARIMA → Prophet → LSTM)
5. Inspect residual plots for patterns

**Q: Tests are failing. What should I do?**  
A: Run tests with verbose output:
```bash
pytest tests/ -v --tb=long
```
Check that all dependencies are installed and up-to-date.

### Performance Questions

**Q: How can I speed up LSTM training?**  
A: Several options:
1. Use GPU acceleration (install `tensorflow-gpu`)
2. Reduce batch size
3. Use fewer epochs
4. Reduce model complexity
5. Use smaller lookback window

**Q: Can I use this for real-time forecasting?**  
A: Yes, but consider:
- **ARIMA**: Very fast inference (~milliseconds)
- **Prophet**: Fast inference (~seconds)
- **LSTM**: Moderate speed (requires model loading)
- Pre-train models and cache them for best performance

**Q: How do I handle large datasets (millions of points)?**  
A: Strategies:
1. Sample data for model development
2. Use batching for LSTM
3. Consider simpler models (ARIMA/Prophet) first
4. Use incremental learning if available
5. Aggregate to lower frequency (hourly → daily)

### Best Practices

**Q: What's the recommended workflow?**  
A:
1. **Explore**: Visualize data, check statistics
2. **Preprocess**: Handle missing values, outliers
3. **Split**: Create train/test sets (80/20)
4. **Baseline**: Start with simple models (ARIMA)
5. **Iterate**: Try more complex models (Prophet, LSTM)
6. **Ensemble**: Combine best models
7. **Evaluate**: Use cross-validation
8. **Deploy**: Save best model

**Q: Should I always use ensemble models?**  
A: Not necessarily:
- **Pros**: Usually more accurate, robust
- **Cons**: Slower, more complex, harder to interpret
- **Use when**: Accuracy is critical, have computational resources
- **Skip when**: Need speed, interpretability, or simple model

**Q: How often should I retrain my model?**  
A: Depends on data characteristics:
- **Static data**: Rarely (quarterly/annually)
- **Slow drift**: Monthly
- **Fast drift**: Weekly/daily
- **Real-time**: Continuous/incremental learning
- Monitor model performance and retrain when accuracy degrades

---

<a name="português"></a>
## 🇧🇷 Português

### 📊 Visão Geral

**Time Series Forecasting Engine** é um framework Python para previsão de séries temporais que reúne modelos estatísticos (ARIMA), machine learning (Prophet) e deep learning (LSTM) em uma interface consistente. Inclui utilitários de pré-processamento, avaliação e visualização.

Construído para cientistas de dados, engenheiros de ML e pesquisadores trabalhando em problemas como previsão de demanda, predições financeiras e análise de consumo de energia.

### ✨ Principais Recursos

#### 🎯 Múltiplos Algoritmos de Previsão

| Modelo | Tipo | Melhor Para | Complexidade |
|--------|------|-------------|--------------|
| **ARIMA/SARIMA** | Estatístico | Tendências lineares, padrões sazonais | Baixa |
| **Prophet** | Baseado em ML | Múltiplas sazonalidades, feriados | Média |
| **LSTM** | Deep Learning | Padrões não-lineares complexos | Alta |
| **Ensemble** | Híbrido | Máxima precisão, predições robustas | Alta |

#### 🔧 Pré-processamento Abrangente

- **Imputação de Valores Faltantes**
  - Interpolação linear
  - Preenchimento forward/backward
  - Imputação por média/mediana
  - Preenchimento baseado em decomposição sazonal

- **Detecção e Remoção de Outliers**
  - Método IQR (Intervalo Interquartil)
  - Método Z-score
  - Z-score modificado
  - Isolation Forest

- **Transformação de Dados**
  - Transformação logarítmica
  - Transformação Box-Cox
  - Escalonamento Min-Max
  - Escalonamento padrão
  - Diferenciação para estacionariedade

- **Engenharia de Features**
  - Features de lag (1-30 lags)
  - Estatísticas móveis (média, desvio padrão, mín, máx)
  - Features baseadas em tempo (dia, mês, trimestre, ano)
  - Indicadores sazonais

#### 📈 Métricas de Avaliação Avançadas

| Métrica | Descrição | Caso de Uso |
|---------|-----------|-------------|
| **MAE** | Erro Absoluto Médio | Precisão geral |
| **RMSE** | Raiz do Erro Quadrático Médio | Penaliza erros grandes |
| **MAPE** | Erro Percentual Absoluto Médio | Precisão relativa |
| **sMAPE** | MAPE Simétrico | Erro percentual balanceado |
| **R²** | Coeficiente de Determinação | Qualidade do ajuste do modelo |
| **MASE** | Erro Absoluto Médio Escalado | Comparação com benchmark |

### 🚀 Início Rápido

#### Instalação

```bash
# Clone o repositório
git clone https://github.com/galafis/time-series-forecasting-engine.git
cd time-series-forecasting-engine

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt

# Instale o pacote em modo de desenvolvimento
pip install -e .
```

#### Exemplo de Uso Básico

```python
import pandas as pd
from models import ARIMAForecaster, ProphetForecaster, EnsembleForecaster
from preprocessing import TimeSeriesPreprocessor
from evaluation import ModelEvaluator

# Carregue seus dados de série temporal
data = pd.read_csv('data/vendas.csv', index_col='data', parse_dates=True)
ts = data['vendas']

# Divida em treino/teste
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# 1. Pré-processamento
preprocessor = TimeSeriesPreprocessor()
train_clean = preprocessor.impute_missing(train, method='interpolation')
train_clean = preprocessor.remove_outliers(train_clean, method='iqr')

# 2. Treinamento de Modelos
arima = ARIMAForecaster(order=(2, 1, 2))
arima.fit(train_clean)
arima_forecast = arima.predict(steps=len(test))

prophet = ProphetForecaster()
prophet.fit(train_clean)
prophet_forecast = prophet.predict(steps=len(test))

# 3. Modelo Ensemble
ensemble = EnsembleForecaster(models=[arima, prophet], weights=[0.5, 0.5])
ensemble_forecast = ensemble.predict(steps=len(test))

# 4. Avaliação
evaluator = ModelEvaluator()
metrics = evaluator.calculate_metrics(test.values, ensemble_forecast.values)

print(f"RMSE: {metrics['RMSE']:.2f}")
print(f"MAE: {metrics['MAE']:.2f}")
print(f"MAPE: {metrics['MAPE']:.2f}%")
```

### 📊 Métricas de Avaliação Suportadas

Todos os modelos podem ser avaliados usando o `ModelEvaluator` integrado, com métricas como MAE, RMSE, MAPE, sMAPE, R² e MASE. Consulte a [Documentação de Avaliação](src/evaluation/README.md) para detalhes sobre cada métrica.

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

