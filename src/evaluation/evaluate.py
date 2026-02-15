"""
Evaluation script for battery prediction models
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.lstm_model import BatteryLSTM, BatteryGRU, BatteryBiLSTM
from src.data.data_loader import BatteryDataLoader
from src.data.preprocess import BatteryPreprocessor
import pickle


class BatteryEvaluator:
    """Evaluate battery prediction models"""
    
    def __init__(self, config_path: str = "config/config.yaml", model_path: str = "models/best_model.pth"):
        """Initialize evaluator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load scalers
        scaler_path = f"{self.config['paths']['model_save_dir']}/scalers.pkl"
        if Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.target_scaler = scalers['target_scaler']
        else:
            self.target_scaler = None
        
        # Create results directory
        Path(self.config['paths']['results_dir']).mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
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
        elif model_type == 'bilstm':
            model = BatteryBiLSTM(**model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def calculate_metrics(self, y_true, y_pred, target_names=['SOH', 'Range', 'Degradation Rate']):
        """Calculate evaluation metrics"""
        metrics = {}
        
        for i, name in enumerate(target_names):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            
            rmse = np.sqrt(mean_squared_error(y_true_i, y_pred_i))
            mae = mean_absolute_error(y_true_i, y_pred_i)
            r2 = r2_score(y_true_i, y_pred_i)
            mape = np.mean(np.abs((y_true_i - y_pred_i) / (y_true_i + 1e-8))) * 100
            
            metrics[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }
        
        return metrics
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        model = self.load_model()
        
        # Make predictions
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred = model(X_test_tensor).cpu().numpy()
        
        # Inverse transform if normalized
        if self.target_scaler:
            y_test_original = self.target_scaler.inverse_transform(y_test)
            y_pred_original = self.target_scaler.inverse_transform(y_pred)
        else:
            y_test_original = y_test
            y_pred_original = y_pred
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test_original, y_pred_original)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for target_name, target_metrics in metrics.items():
            print(f"\n{target_name}:")
            print(f"  RMSE:  {target_metrics['RMSE']:.4f}")
            print(f"  MAE:   {target_metrics['MAE']:.4f}")
            print(f"  R²:    {target_metrics['R²']:.4f}")
            print(f"  MAPE:  {target_metrics['MAPE']:.2f}%")
        
        # Plot results
        if self.config['evaluation']['plot_results']:
            self.plot_predictions(y_test_original, y_pred_original)
        
        # Save predictions
        if self.config['evaluation']['save_predictions']:
            np.savez(
                f"{self.config['paths']['results_dir']}/predictions.npz",
                y_true=y_test_original,
                y_pred=y_pred_original
            )
        
        return metrics, y_test_original, y_pred_original
    
    def plot_predictions(self, y_true, y_pred, target_names=['SOH', 'Range (km)', 'Degradation Rate']):
        """Plot prediction vs actual"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, (ax, name) in enumerate(zip(axes, target_names)):
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel(f'Actual {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name} Prediction')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = Path(self.config['paths']['results_dir']) / 'predictions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPredictions plot saved to {save_path}")
        plt.close()
        
        # Plot time series for first target (SOH)
        fig, ax = plt.subplots(figsize=(12, 6))
        n_samples = min(200, len(y_true))
        indices = np.arange(n_samples)
        
        ax.plot(indices, y_true[:n_samples, 0], label='Actual SOH', alpha=0.7)
        ax.plot(indices, y_pred[:n_samples, 0], label='Predicted SOH', alpha=0.7)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('SOH')
        ax.set_title('SOH Prediction Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = Path(self.config['paths']['results_dir']) / 'soh_timeseries.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SOH time series plot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    from src.data.data_loader import BatteryDataLoader
    from src.data.preprocess import BatteryPreprocessor
    
    # Load and preprocess test data
    loader = BatteryDataLoader("data/raw")
    df = loader.load_data()
    
    features = loader.get_features()
    targets = loader.get_targets()
    
    preprocessor = BatteryPreprocessor(
        sequence_length=50,
        normalize=True
    )
    
    # Load scalers
    preprocessor.load_scalers("models/scalers.pkl")
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(
        features, targets,
        train_size=0.7,
        val_size=0.15
    )
    
    # Evaluate
    evaluator = BatteryEvaluator()
    metrics, y_true, y_pred = evaluator.evaluate(X_test, y_test)

