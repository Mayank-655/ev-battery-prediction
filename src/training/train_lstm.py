"""
Training script for LSTM/GRU battery prediction models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.lstm_model import BatteryLSTM, BatteryGRU, BatteryBiLSTM
from src.data.data_loader import BatteryDataLoader
from src.data.preprocess import BatteryPreprocessor


class BatteryTrainer:
    """Trainer class for battery prediction models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(
            self.config['training']['device'] 
            if torch.cuda.is_available() and self.config['training']['device'] == 'cuda' else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Create directories
        Path(self.config['paths']['model_save_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load and preprocess data"""
        print("Loading data...")
        loader = BatteryDataLoader(self.config['data']['raw_data_path'])
        df = loader.load_data()
        
        features = loader.get_features()
        targets = loader.get_targets()
        
        # Preprocess
        preprocessor = BatteryPreprocessor(
            sequence_length=self.config['data']['sequence_length'],
            normalize=True
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(
            features, targets,
            train_size=self.config['data']['train_split'],
            val_size=self.config['data']['val_split']
        )
        
        # Save scalers
        preprocessor.save_scalers(f"{self.config['paths']['model_save_dir']}/scalers.pkl")
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.X_val = torch.FloatTensor(X_val).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Test samples: {len(self.X_test)}")
        
        return preprocessor
    
    def create_model(self):
        """Create model based on configuration"""
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
        
        model = model.to(self.device)
        return model
    
    def train(self):
        """Main training loop"""
        # Load data
        preprocessor = self.load_data()
        
        # Create model
        model = self.create_model()
        print(f"Model: {self.config['model']['type']}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['training']['early_stopping_patience']
        
        print("\nStarting training...")
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, f"{self.config['paths']['model_save_dir']}/best_model.pth")
                print(f"  -> Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses)
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
        return model
    
    def plot_training_history(self, train_losses, val_losses):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        save_path = Path(self.config['paths']['log_dir']) / 'training_history.png'
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    trainer = BatteryTrainer()
    model = trainer.train()

