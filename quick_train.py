"""
Quick training script with reduced epochs for testing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.train_lstm import BatteryTrainer
import yaml

# Load config
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Modify for quick training
config['training']['num_epochs'] = 5  # Reduced for quick test
config['training']['batch_size'] = 16  # Smaller batch for faster iteration
config['model']['hidden_size'] = 64  # Smaller model
config['model']['num_layers'] = 1

# Save temporary config
import tempfile
import os
temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
yaml.dump(config, temp_config)
temp_config.close()

print("="*60)
print("QUICK TRAINING (5 epochs for testing)")
print("="*60)
print("\nThis will train a smaller model quickly for testing.")
print("For production, use: python main.py train\n")

trainer = BatteryTrainer(temp_config.name)
model = trainer.train()

# Clean up
os.unlink(temp_config.name)

print("\n" + "="*60)
print("✅ Quick training complete!")
print("="*60)
print("\nModel saved to: models/best_model.pth")
print("You can now use the API or Dashboard!")

