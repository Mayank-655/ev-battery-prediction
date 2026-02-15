"""
Data preprocessing for battery time-series data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import pickle
from pathlib import Path


class BatteryPreprocessor:
    """Preprocess battery data for time-series modeling"""
    
    def __init__(self, sequence_length: int = 50, normalize: bool = True):
        """
        Initialize preprocessor
        
        Args:
            sequence_length: Number of time steps for LSTM/GRU
            normalize: Whether to normalize features
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.feature_scaler = StandardScaler() if normalize else None
        self.target_scaler = MinMaxScaler() if normalize else None
        
    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series prediction
        
        Args:
            features: Feature array (n_samples, n_features)
            targets: Target array (n_samples, n_targets)
            
        Returns:
            X: Sequences (n_sequences, sequence_length, n_features)
            y: Targets (n_sequences, n_targets)
        """
        X, y = [], []
        
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(targets[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def preprocess(self, features: np.ndarray, targets: np.ndarray, 
                   train_size: float = 0.7, val_size: float = 0.15) -> Tuple:
        """
        Full preprocessing pipeline
        
        Args:
            features: Feature array
            targets: Target array
            train_size: Training set proportion
            val_size: Validation set proportion
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Normalize features
        if self.normalize and self.feature_scaler:
            features = self.feature_scaler.fit_transform(features)
            targets = self.target_scaler.fit_transform(targets)
        
        # Create sequences
        X, y = self.create_sequences(features, targets)
        
        # Split data
        n_samples = len(X)
        n_train = int(n_samples * train_size)
        n_val = int(n_samples * val_size)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_scalers(self, save_path: str):
        """Save fitted scalers"""
        if self.feature_scaler and self.target_scaler:
            scalers = {
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(scalers, f)
    
    def load_scalers(self, load_path: str):
        """Load fitted scalers"""
        with open(load_path, 'rb') as f:
            scalers = pickle.load(f)
            self.feature_scaler = scalers['feature_scaler']
            self.target_scaler = scalers['target_scaler']

