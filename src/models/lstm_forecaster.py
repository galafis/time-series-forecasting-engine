"""
LSTM-based deep learning forecasting model.

This module implements LSTM (Long Short-Term Memory) neural networks
for time series forecasting with support for multivariate inputs.
"""

from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from tensorflow import keras as keras_typing
else:
    keras_typing = None
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None
    keras = None
    layers = None
from sklearn.preprocessing import StandardScaler
import warnings

from .base_forecaster import BaseForecaster


class LSTMForecaster(BaseForecaster):
    """
    LSTM deep learning forecasting model.
    
    Implements LSTM neural networks for univariate and multivariate
    time series forecasting with automatic sequence generation.
    """
    
    def __init__(
        self,
        lookback: int = 30,
        lstm_units: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        **kwargs
    ):
        """
        Initialize LSTM forecaster.
        
        Parameters
        ----------
        lookback : int
            Number of time steps to look back
        lstm_units : int
            Number of LSTM units per layer
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for optimizer
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data for validation
        **kwargs : dict
            Additional parameters
        """
        super().__init__(name="LSTMForecaster")
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for LSTMForecaster. Install with: pip install tensorflow")
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.kwargs = kwargs
        
        self.scaler_y = StandardScaler()
        self.scaler_X = None
        self.history_ = None
        
    def _create_sequences(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Parameters
        ----------
        y : np.ndarray
            Target values
        X : np.ndarray, optional
            Exogenous features
            
        Returns
        -------
        X_seq : np.ndarray
            Input sequences
        y_seq : np.ndarray
            Target values
        """
        X_seq, y_seq = [], []
        
        for i in range(len(y) - self.lookback):
            if X is not None:
                # Combine target history with exogenous features
                seq = np.column_stack([
                    y[i:i+self.lookback],
                    X[i:i+self.lookback]
                ])
            else:
                seq = y[i:i+self.lookback].reshape(-1, 1)
            
            X_seq.append(seq)
            y_seq.append(y[i+self.lookback])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _build_model(self, input_shape: Tuple[int, int]):
        """
        Build LSTM neural network architecture.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of input sequences (timesteps, features)
            
        Returns
        -------
        model : keras.Model
            Compiled LSTM model
        """
        model = keras.Sequential(name="LSTM_Forecaster")
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.lstm_units,
            return_sequences=self.num_layers > 1,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(self.dropout))
        
        # Additional LSTM layers
        for i in range(1, self.num_layers):
            return_seq = i < self.num_layers - 1
            model.add(layers.LSTM(self.lstm_units, return_sequences=return_seq))
            model.add(layers.Dropout(self.dropout))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None, **kwargs) -> 'LSTMForecaster':
        """
        Fit LSTM model to time series data.
        
        Parameters
        ----------
        y : pd.Series
            Target time series
        X : pd.DataFrame, optional
            Exogenous features
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        self : LSTMForecaster
            Fitted model instance
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Scale target variable
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
            
            # Scale exogenous features if provided
            X_scaled = None
            if X is not None:
                self.scaler_X = StandardScaler()
                X_scaled = self.scaler_X.fit_transform(X.values)
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(y_scaled, X_scaled)
            
            # Build model
            input_shape = (X_seq.shape[1], X_seq.shape[2])
            self.model = self._build_model(input_shape)
            
            # Train model
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            self.history_ = self.model.fit(
                X_seq, y_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.is_fitted = True
            
            # Calculate residuals
            predictions = self.model.predict(X_seq, verbose=0)
            predictions_original = self.scaler_y.inverse_transform(predictions).flatten()
            y_original = self.scaler_y.inverse_transform(y_seq.reshape(-1, 1)).flatten()
            self.residuals_ = y_original - predictions_original
            
            # Store last sequence for prediction
            self.last_sequence_ = X_seq[-1]
            self.last_y_values_ = y_scaled[-self.lookback:]
            self.last_X_values_ = X_scaled[-self.lookback:] if X_scaled is not None else None
            
            # Store training metrics
            self.training_history = {
                'final_loss': self.history_.history['loss'][-1],
                'final_val_loss': self.history_.history['val_loss'][-1],
                'epochs_trained': len(self.history_.history['loss'])
            }
            
        return self
    
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """
        Generate point forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        X : pd.DataFrame, optional
            Future exogenous features
        **kwargs : dict
            Additional prediction parameters
            
        Returns
        -------
        predictions : pd.Series
            Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        current_seq = self.last_sequence_.copy()
        
        for step in range(steps):
            # Predict next value
            pred_scaled = self.model.predict(
                current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]),
                verbose=0
            )[0, 0]
            
            # Inverse transform prediction
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            if X is not None and self.scaler_X is not None:
                # Use provided future exogenous features
                X_scaled = self.scaler_X.transform(X.iloc[[step]].values)
                new_row = np.column_stack([[pred_scaled], X_scaled[0]])
            else:
                new_row = np.array([[pred_scaled]])
            
            current_seq = np.vstack([current_seq[1:], new_row])
        
        return pd.Series(predictions, name='forecast')
    
    def get_training_history(self) -> dict:
        """
        Get training history.
        
        Returns
        -------
        history : dict
            Training and validation metrics
        """
        if self.history_ is None:
            return {}
        return self.history_.history
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'lookback': self.lookback,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        })
        return params

