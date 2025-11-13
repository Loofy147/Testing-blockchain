import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from collections import deque

class FeePredictorLSTM:
    """
    Predicts future transaction fees using a Long Short-Term Memory (LSTM) model.
    """

    def __init__(self, n_features: int = 2, n_steps: int = 10):
        self.n_features = n_features
        self.n_steps = n_steps
        self.model = self._build_model()
        self.history = deque(maxlen=200) # Store recent data for scaling
        self.scaler = MinMaxScaler()
        self.is_trained = False

    def _build_model(self):
        """Builds the LSTM model architecture."""
        model = Sequential([
            Input(shape=(self.n_steps, self.n_features)),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self, data: np.ndarray, for_training: bool = True):
        """
        Prepares time-series data for the LSTM model.

        Args:
            data (np.ndarray): A 2D array of historical data (e.g., [congestion, fee]).
            for_training (bool): If True, prepares data with labels (X, y).
                                 If False, prepares the last sequence for prediction.

        Returns:
            If for_training: (np.ndarray, np.ndarray) - X, y
            If not for_training: np.ndarray - X
        """
        # Fit scaler on the full history to avoid data leakage during training
        self.scaler.fit(data)
        scaled_data = self.scaler.transform(data)

        X, y = [], []
        for i in range(len(scaled_data) - self.n_steps):
            X.append(scaled_data[i:i+self.n_steps])
            y.append(scaled_data[i+self.n_steps, -1]) # Target is the last feature (fee)

        X, y = np.array(X), np.array(y)

        if for_training:
            return X, y
        else:
            # Return only the last sequence for prediction
            return X[-1].reshape(1, self.n_steps, self.n_features)

    def train(self, historical_data: list):
        """
        Trains the LSTM model on historical data.

        Args:
            historical_data (list): A list of tuples, e.g., [(congestion, fee), ...].
        """
        if len(historical_data) < self.n_steps + 1:
            print("--- Not enough data to train LSTM model yet.")
            return

        data = np.array(historical_data)
        self.history.extend(data) # Update history

        X, y = self.prepare_data(np.array(self.history))

        print(f"--- Training LSTM model with {len(X)} samples...")
        self.model.fit(X, y, epochs=10, batch_size=1, verbose=0)
        self.is_trained = True
        print("--- LSTM model training complete.")

    def predict(self, recent_data: list) -> float:
        """
        Predicts the next fee based on recent historical data.

        Args:
            recent_data (list): A list of recent data points, must be at least n_steps long.

        Returns:
            float: The predicted fee.
        """
        if not self.is_trained or len(recent_data) < self.n_steps:
            return 0.01 # Default fee if not ready

        data = np.array(recent_data)

        # We need to transform the prediction data using the same scale as training
        # We assume self.history is representative for fitting the scaler
        if not hasattr(self.scaler, 'scale_'):
             # Fit the scaler if it hasn't been fitted
            self.scaler.fit(np.array(self.history))

        scaled_data = self.scaler.transform(data)

        X = np.array([scaled_data[-self.n_steps:]])

        # Predict the scaled value
        predicted_scaled = self.model.predict(X, verbose=0)

        # We need to inverse_transform this prediction.
        # The scaler expects an array of shape (n_samples, n_features).
        # We create a dummy array with the prediction in the correct column.
        dummy_row = np.zeros((1, self.n_features))
        dummy_row[0, -1] = predicted_scaled

        # Inverse transform and extract the fee
        predicted_fee = self.scaler.inverse_transform(dummy_row)[0, -1]

        return max(0.001, predicted_fee) # Ensure fee is not negative
