"""
Unit tests for forecasting models.

Author: Gabriel Demetrios Lafis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models import ARIMAForecaster, ProphetForecaster, LSTMForecaster, EnsembleForecaster


@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    trend = np.linspace(100, 150, 200)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(200) / 30)
    noise = np.random.normal(0, 2, 200)
    values = trend + seasonal + noise
    return pd.Series(values, index=dates, name='value')


@pytest.fixture
def train_test_split(sample_data):
    """Split data into train and test sets."""
    train_size = int(len(sample_data) * 0.8)
    y_train = sample_data[:train_size]
    y_test = sample_data[train_size:]
    return y_train, y_test


class TestARIMAForecaster:
    """Tests for ARIMA forecaster."""
    
    def test_initialization(self):
        """Test ARIMA model initialization."""
        model = ARIMAForecaster(order=(1, 1, 1))
        assert model.name == "ARIMAForecaster"
        assert model.order == (1, 1, 1)
        assert not model.is_fitted
    
    def test_fit_predict(self, train_test_split):
        """Test fitting and prediction."""
        y_train, y_test = train_test_split
        model = ARIMAForecaster(auto_select=True)
        
        # Fit model
        model.fit(y_train)
        assert model.is_fitted
        assert model.order is not None
        
        # Predict
        predictions = model.predict(steps=len(y_test))
        assert len(predictions) == len(y_test)
        assert isinstance(predictions, pd.Series)
    
    def test_predict_with_intervals(self, train_test_split):
        """Test prediction with intervals."""
        y_train, y_test = train_test_split
        model = ARIMAForecaster(auto_select=True)
        model.fit(y_train)
        
        predictions, lower, upper = model.predict_with_intervals(
            steps=len(y_test),
            confidence=0.95
        )
        
        assert len(predictions) == len(y_test)
        assert len(lower) == len(y_test)
        assert len(upper) == len(y_test)
        assert all(lower.values <= predictions.values)
        assert all(predictions.values <= upper.values)


class TestProphetForecaster:
    """Tests for Prophet forecaster."""
    
    def test_initialization(self):
        """Test Prophet model initialization."""
        model = ProphetForecaster(growth='linear')
        assert model.name == "ProphetForecaster"
        assert model.growth == 'linear'
        assert not model.is_fitted
    
    def test_fit_predict(self, train_test_split):
        """Test fitting and prediction."""
        y_train, y_test = train_test_split
        model = ProphetForecaster()
        
        # Fit model
        model.fit(y_train)
        assert model.is_fitted
        
        # Predict
        predictions = model.predict(steps=len(y_test))
        assert len(predictions) == len(y_test)
        assert isinstance(predictions, pd.Series)
    
    def test_predict_with_intervals(self, train_test_split):
        """Test prediction with intervals."""
        y_train, y_test = train_test_split
        model = ProphetForecaster()
        model.fit(y_train)
        
        predictions, lower, upper = model.predict_with_intervals(
            steps=len(y_test),
            confidence=0.95
        )
        
        assert len(predictions) == len(y_test)
        assert len(lower) == len(y_test)
        assert len(upper) == len(y_test)


class TestLSTMForecaster:
    """Tests for LSTM forecaster."""
    
    def test_initialization(self):
        """Test LSTM model initialization."""
        model = LSTMForecaster(lookback=30, lstm_units=64)
        assert model.name == "LSTMForecaster"
        assert model.lookback == 30
        assert model.lstm_units == 64
        assert not model.is_fitted
    
    def test_fit_predict(self, train_test_split):
        """Test fitting and prediction."""
        y_train, y_test = train_test_split
        model = LSTMForecaster(
            lookback=20,
            lstm_units=32,
            epochs=10,
            batch_size=16
        )
        
        # Fit model
        model.fit(y_train)
        assert model.is_fitted
        assert model.model is not None
        
        # Predict
        predictions = model.predict(steps=len(y_test))
        assert len(predictions) == len(y_test)
        assert isinstance(predictions, pd.Series)


class TestEnsembleForecaster:
    """Tests for Ensemble forecaster."""
    
    def test_initialization(self):
        """Test Ensemble model initialization."""
        models = [
            ARIMAForecaster(order=(1, 1, 1)),
            ProphetForecaster()
        ]
        ensemble = EnsembleForecaster(models, method='average')
        assert ensemble.name == "EnsembleForecaster"
        assert len(ensemble.forecasters) == 2
        assert ensemble.method == 'average'
    
    def test_fit_predict(self, train_test_split):
        """Test fitting and prediction."""
        y_train, y_test = train_test_split
        
        models = [
            ARIMAForecaster(auto_select=True),
            ProphetForecaster()
        ]
        ensemble = EnsembleForecaster(models, method='average')
        
        # Fit model
        ensemble.fit(y_train)
        assert ensemble.is_fitted
        
        # Predict
        predictions = ensemble.predict(steps=len(y_test))
        assert len(predictions) == len(y_test)
        assert isinstance(predictions, pd.Series)
    
    def test_weighted_ensemble(self, train_test_split):
        """Test weighted ensemble."""
        y_train, y_test = train_test_split
        
        models = [
            ARIMAForecaster(auto_select=True),
            ProphetForecaster()
        ]
        weights = [0.6, 0.4]
        ensemble = EnsembleForecaster(models, method='weighted', weights=weights)
        
        ensemble.fit(y_train)
        predictions = ensemble.predict(steps=len(y_test))
        assert len(predictions) == len(y_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

