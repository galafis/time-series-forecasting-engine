# 🤝 Contributing to Time Series Forecasting Engine

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/time-series-forecasting-engine.git
   cd time-series-forecasting-engine
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/galafis/time-series-forecasting-engine.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   # Install package in editable mode with dev dependencies
   pip install -e .
   pip install -r requirements.txt
   
   # Install development tools
   pip install pytest pytest-cov black flake8 mypy
   ```

3. **Verify installation**:
   ```bash
   python -c "import src; print('Installation successful!')"
   pytest tests/ -v
   ```

## Making Contributions

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug fixes**
- ✨ **New features**
- 📝 **Documentation improvements**
- 🧪 **Test additions**
- 🎨 **Code quality improvements**
- 🌐 **Translations**

### Finding Issues to Work On

- Check the [Issues](https://github.com/galafis/time-series-forecasting-engine/issues) page
- Look for issues tagged with `good first issue` or `help wanted`
- Feel free to create new issues for bugs or feature requests

### Creating a Branch

Create a descriptive branch name:

```bash
# For features
git checkout -b feature/add-gru-model

# For bug fixes
git checkout -b fix/arima-convergence-issue

# For documentation
git checkout -b docs/improve-readme
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Group and sort (stdlib, third-party, local)
- **Docstrings**: Use NumPy style documentation

### Code Formatting

Use **Black** for code formatting:

```bash
black src/ tests/ examples/
```

### Linting

Use **Flake8** for linting:

```bash
flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
```

### Type Hints

Use type hints for all function signatures:

```python
def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> pd.Series:
    """Generate forecasts."""
    ...
```

Check types with **mypy**:

```bash
mypy src/
```

### Docstring Format

Use NumPy style docstrings:

```python
def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate forecast error metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    metrics : dict
        Dictionary containing error metrics
        
    Examples
    --------
    >>> evaluator = ModelEvaluator()
    >>> metrics = evaluator.calculate_metrics(y_true, y_pred)
    >>> print(metrics['RMSE'])
    12.34
    """
    ...
```

## Testing

### Writing Tests

All new code should include tests:

1. **Create test file**: `tests/test_<module>.py`
2. **Use pytest fixtures** for common setup
3. **Test edge cases** and error conditions
4. **Maintain test independence**

Example test structure:

```python
import pytest
import numpy as np
import pandas as pd
from src.models import YourModel

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.random.randn(100).cumsum()
    return pd.Series(values, index=dates)

class TestYourModel:
    """Tests for YourModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = YourModel()
        assert model is not None
    
    def test_fit_predict(self, sample_data):
        """Test fitting and prediction."""
        model = YourModel()
        train = sample_data[:80]
        
        model.fit(train)
        predictions = model.predict(steps=20)
        
        assert len(predictions) == 20
        assert predictions.isna().sum() == 0
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=src tests/

# Generate coverage report
pytest --cov=src --cov-report=html tests/
```

### Test Coverage

Aim for:
- **Minimum**: 80% coverage for new code
- **Target**: 90%+ coverage
- **Critical modules**: 95%+ coverage

## Documentation

### Code Documentation

- **All public methods** must have docstrings
- **Complex algorithms** should have inline comments
- **Type hints** required for all functions

### Module Documentation

Each module should have a `README.md` with:

- Overview and purpose
- Usage examples
- API reference
- Best practices
- Troubleshooting

### Updating Main README

When adding features, update:

1. Feature list
2. Usage examples
3. API documentation link
4. Changelog (if applicable)

### Adding Notebooks

Tutorial notebooks should:

1. Have clear objectives
2. Include explanatory text
3. Show complete examples
4. Run without errors
5. Be well-commented

Place in `notebooks/` directory with naming:
- `01_topic_name.ipynb`
- `02_another_topic.ipynb`

## Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Run all checks**:
   ```bash
   # Format code
   black src/ tests/
   
   # Lint
   flake8 src/ tests/
   
   # Type check
   mypy src/
   
   # Run tests
   pytest tests/ -v
   ```

3. **Update documentation** if needed

4. **Add yourself** to `CONTRIBUTORS.md` (if it exists)

### Commit Message Format

Use conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `style`: Code style changes
- `chore`: Maintenance tasks

Examples:

```
feat(models): add GRU forecaster

Implement GRU-based neural network forecaster with:
- Multi-layer support
- Dropout regularization
- GPU acceleration

Closes #123
```

```
fix(ensemble): handle variable length predictions

Fix bug where ensemble forecaster crashed with predictions
of different lengths from component models.

Fixes #456
```

### Submitting Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR** on GitHub with:
   - Clear title
   - Detailed description
   - Link to related issue
   - Screenshots (if UI changes)
   - Checklist completion

3. **PR Template** (automatic):
   ```markdown
   ## Description
   Brief description of changes
   
   ## Related Issue
   Closes #issue_number
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Code refactoring
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Comments added for complex code
   - [ ] Documentation updated
   - [ ] Tests added/updated
   - [ ] All tests pass
   - [ ] No new warnings
   ```

### Review Process

1. **Automated checks** will run (tests, linting)
2. **Maintainer review** will be assigned
3. **Address feedback** by pushing new commits
4. **Approval** and merge by maintainer

### After Merge

1. **Delete your branch**:
   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

2. **Update your fork**:
   ```bash
   git checkout master
   git pull upstream master
   git push origin master
   ```

## Development Best Practices

### Code Quality

- **Keep functions small** (< 50 lines ideally)
- **Single responsibility** principle
- **Avoid deep nesting** (max 3 levels)
- **Use meaningful names**
- **Don't repeat yourself** (DRY)

### Performance

- **Profile before optimizing**
- **Use vectorized operations** (NumPy/Pandas)
- **Cache expensive computations**
- **Consider memory usage** for large datasets

### Error Handling

- **Use specific exceptions**
- **Provide helpful error messages**
- **Validate inputs early**
- **Document expected errors**

Example:

```python
def predict(self, steps: int) -> pd.Series:
    """Generate forecasts."""
    if not self.is_fitted:
        raise ValueError(
            "Model must be fitted before prediction. "
            "Call fit() with training data first."
        )
    
    if steps < 1:
        raise ValueError(
            f"steps must be >= 1, got {steps}"
        )
    
    if steps > 1000:
        warnings.warn(
            f"Large forecast horizon ({steps} steps) may be unreliable"
        )
    
    ...
```

### Logging

Use Python's logging module:

```python
import logging

logger = logging.getLogger(__name__)

def train_model(self, data):
    logger.info(f"Training model with {len(data)} samples")
    logger.debug(f"Model parameters: {self.get_params()}")
    
    try:
        # Training code
        ...
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    logger.info("Training completed successfully")
```

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/galafis/time-series-forecasting-engine/discussions)
- **Bugs**: Open an [Issue](https://github.com/galafis/time-series-forecasting-engine/issues)
- **Chat**: Join our community (if available)

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing! 🎉
