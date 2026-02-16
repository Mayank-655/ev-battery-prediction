"""
Model generalization testing across different batteries and conditions
"""

import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.lstm_model import BatteryLSTM, BatteryGRU
from src.data.data_loader import BatteryDataLoader
from src.data.preprocess import BatteryPreprocessor
import pickle


class GeneralizationTester:
    """Test model generalization across different batteries"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize tester"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train_test_split_by_battery(self, df, train_ratio=0.7):
        """
        Split data by battery ID (not random) to test generalization
        
        Args:
            df: DataFrame with battery data
            train_ratio: Ratio of batteries for training
            
        Returns:
            train_df, test_df
        """
        unique_batteries = df['battery_id'].unique()
        n_train = int(len(unique_batteries) * train_ratio)
        
        train_batteries = unique_batteries[:n_train]
        test_batteries = unique_batteries[n_train:]
        
        train_df = df[df['battery_id'].isin(train_batteries)].copy()
        test_df = df[df['battery_id'].isin(test_batteries)].copy()
        
        return train_df, test_df, train_batteries, test_batteries
    
    def test_generalization(self, model_path: str = "models/best_model.pth"):
        """
        Test model on batteries it hasn't seen during training
        
        Args:
            model_path: Path to trained model
        """
        print("="*60)
        print("MODEL GENERALIZATION TEST")
        print("="*60)
        
        # Load data
        print("\n[1/4] Loading data...")
        loader = BatteryDataLoader(self.config['data']['raw_data_path'])
        df = loader.load_data()
        
        # Split by battery (not random)
        print("\n[2/4] Splitting by battery ID...")
        train_df, test_df, train_batteries, test_batteries = self.train_test_split_by_battery(df)
        
        print(f"  Training batteries: {len(train_batteries)}")
        print(f"    {list(train_batteries)}")
        print(f"  Test batteries: {len(test_batteries)}")
        print(f"    {list(test_batteries)}")
        print(f"  Training samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")
        
        # Preprocess training data
        print("\n[3/4] Preprocessing...")
        train_features = train_df[['voltage', 'current', 'temperature', 'cycle', 'capacity']].values
        train_targets = train_df[['soh', 'range_km', 'degradation_rate']].values
        
        test_features = test_df[['voltage', 'current', 'temperature', 'cycle', 'capacity']].values
        test_targets = test_df[['soh', 'range_km', 'degradation_rate']].values
        
        preprocessor = BatteryPreprocessor(
            sequence_length=self.config['data']['sequence_length'],
            normalize=True
        )
        
        # Fit on training data only
        X_train, _, _, y_train, _, _ = preprocessor.preprocess(
            train_features, train_targets, train_size=1.0, val_size=0.0
        )
        
        # Transform test data using training scalers
        preprocessor.feature_scaler.fit(train_features)
        preprocessor.target_scaler.fit(train_targets)
        
        test_features_scaled = preprocessor.feature_scaler.transform(test_features)
        test_targets_scaled = preprocessor.target_scaler.transform(test_targets)
        
        X_test, y_test = preprocessor.create_sequences(test_features_scaled, test_targets_scaled)
        
        # Load model
        print("\n[4/4] Loading model and evaluating...")
        model = self._load_model(model_path)
        
        # Evaluate
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
        
        # Inverse transform
        y_test_original = preprocessor.target_scaler.inverse_transform(y_test)
        y_pred_original = preprocessor.target_scaler.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test_original, y_pred_original)
        
        # Print results
        print("\n" + "="*60)
        print("GENERALIZATION TEST RESULTS")
        print("="*60)
        print(f"\nTested on {len(test_batteries)} unseen batteries:")
        for battery in test_batteries:
            print(f"  - {battery}")
        
        print("\nPerformance Metrics:")
        for target_name, target_metrics in metrics.items():
            print(f"\n{target_name}:")
            print(f"  RMSE:  {target_metrics['RMSE']:.4f}")
            print(f"  MAE:   {target_metrics['MAE']:.4f}")
            print(f"  R²:    {target_metrics['R²']:.4f}")
        
        # Save results
        results_path = Path(self.config['paths']['results_dir']) / 'generalization_results.txt'
        with open(results_path, 'w') as f:
            f.write("Model Generalization Test Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Training batteries: {len(train_batteries)}\n")
            f.write(f"Test batteries: {len(test_batteries)}\n\n")
            for target_name, target_metrics in metrics.items():
                f.write(f"{target_name}:\n")
                f.write(f"  RMSE: {target_metrics['RMSE']:.4f}\n")
                f.write(f"  MAE: {target_metrics['MAE']:.4f}\n")
                f.write(f"  R²: {target_metrics['R²']:.4f}\n\n")
        
        print(f"\nResults saved to: {results_path}")
        
        return metrics
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        model_type = self.config['model']['type'].lower()
        
        model_config = {
            'input_size': self.config['model']['input_size'],
            'hidden_size': self.config['model']['hidden_size'],
            'num_layers': self.config['model']['num_layers'],
            'dropout': self.config['model']['dropout'],
            'output_size': self.config['model']['output_size'],
            'bidirectional': self.config['model'].get('bidirectional', False)
        }
        
        if model_type == 'lstm':
            model = BatteryLSTM(**model_config)
        elif model_type == 'gru':
            model = BatteryGRU(**model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _calculate_metrics(self, y_true, y_pred, target_names=['SOH', 'Range', 'Degradation Rate']):
        """Calculate evaluation metrics"""
        metrics = {}
        
        for i, name in enumerate(target_names):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            
            rmse = np.sqrt(mean_squared_error(y_true_i, y_pred_i))
            mae = mean_absolute_error(y_true_i, y_pred_i)
            r2 = r2_score(y_true_i, y_pred_i)
            
            metrics[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
        
        return metrics


if __name__ == "__main__":
    tester = GeneralizationTester()
    metrics = tester.test_generalization()

