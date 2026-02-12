# 🚀 Time Series Forecasting Engine

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-1.4-F7931E.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Português](#português)

---

## English

### 🎯 Overview

**Time Series Forecasting Engine** — Scalable time series forecasting engine supporting ARIMA, Prophet, LSTM, and ensemble methods. Features automated model selection, cross-validation, and forecast evaluation.

Total source lines: **3,270** across **18** files in **1** language.

### ✨ Key Features

- **Production-Ready Architecture**: Modular, well-documented, and following best practices
- **Comprehensive Implementation**: Complete solution with all core functionality
- **Clean Code**: Type-safe, well-tested, and maintainable codebase
- **Easy Deployment**: Docker support for quick setup and deployment

### 🚀 Quick Start

#### Prerequisites
- Python 3.12+


#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/galafis/time-series-forecasting-engine.git
cd time-series-forecasting-engine
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```





### 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Project Structure

```
time-series-forecasting-engine/
├── config/
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   └── architecture.md
├── examples/
│   └── complete_example.py
├── models/
├── notebooks/
├── src/
│   ├── evaluation/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── models/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── arima_forecaster.py
│   │   ├── base_forecaster.py
│   │   ├── ensemble_forecaster.py
│   │   ├── lstm_forecaster.py
│   │   └── prophet_forecaster.py
│   ├── preprocessing/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── visualization/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── visualizer.py
│   └── __init__.py
├── tests/
│   ├── test_evaluation.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── CONTRIBUTING.md
├── README.md
├── requirements.txt
└── setup.py
```

### 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | 18 files |

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

---

## Português

### 🎯 Visão Geral

**Time Series Forecasting Engine** — Scalable time series forecasting engine supporting ARIMA, Prophet, LSTM, and ensemble methods. Features automated model selection, cross-validation, and forecast evaluation.

Total de linhas de código: **3,270** em **18** arquivos em **1** linguagem.

### ✨ Funcionalidades Principais

- **Arquitetura Pronta para Produção**: Modular, bem documentada e seguindo boas práticas
- **Implementação Completa**: Solução completa com todas as funcionalidades principais
- **Código Limpo**: Type-safe, bem testado e manutenível
- **Fácil Implantação**: Suporte Docker para configuração e implantação rápidas

### 🚀 Início Rápido

#### Pré-requisitos
- Python 3.12+


#### Instalação

1. **Clone the repository**
```bash
git clone https://github.com/galafis/time-series-forecasting-engine.git
cd time-series-forecasting-engine
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```




### 🧪 Testes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Estrutura do Projeto

```
time-series-forecasting-engine/
├── config/
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   └── architecture.md
├── examples/
│   └── complete_example.py
├── models/
├── notebooks/
├── src/
│   ├── evaluation/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── models/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── arima_forecaster.py
│   │   ├── base_forecaster.py
│   │   ├── ensemble_forecaster.py
│   │   ├── lstm_forecaster.py
│   │   └── prophet_forecaster.py
│   ├── preprocessing/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── visualization/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── visualizer.py
│   └── __init__.py
├── tests/
│   ├── test_evaluation.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── CONTRIBUTING.md
├── README.md
├── requirements.txt
└── setup.py
```

### 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| Python | 18 files |

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)
