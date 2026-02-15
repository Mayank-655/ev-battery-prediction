# 🚀 Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Verify Dataset

The NASA Battery dataset should be in `data/raw/`. Verify it's there:

```bash
python main.py explore
```

You should see:
- Total records: 2,744+
- Number of batteries: 32

## Step 3: Train the Model

```bash
python main.py train
```

This will:
- Load and preprocess the NASA dataset
- Create sequences for time-series modeling
- Train an LSTM model (configurable in `config/config.yaml`)
- Save the best model to `models/best_model.pth`
- Generate training history plot

**Expected time:** 10-30 minutes depending on your hardware

## Step 4: Evaluate the Model

```bash
python main.py evaluate
```

This will:
- Load the trained model
- Evaluate on test set
- Calculate RMSE, MAE, R², MAPE metrics
- Generate prediction plots

## Step 5: View Results

Check the following directories:
- `models/` - Saved model checkpoints
- `logs/` - Training history plots
- `results/` - Evaluation plots and predictions

## Customization

### Change Model Type

Edit `config/config.yaml`:
```yaml
model:
  type: "gru"  # Options: "lstm", "gru", "bilstm"
```

### Adjust Hyperparameters

Edit `config/config.yaml`:
```yaml
model:
  hidden_size: 256  # Increase for more capacity
  num_layers: 3     # Deeper network
  dropout: 0.3      # More regularization

training:
  batch_size: 64    # Larger batches
  learning_rate: 0.0005  # Lower learning rate
  num_epochs: 200   # More epochs
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `hidden_size` or `num_layers`
- Use GRU instead of LSTM (lighter)

### Training Too Slow
- Reduce `num_epochs`
- Use smaller `hidden_size`
- Enable GPU if available (set `device: "cuda"`)

### Poor Performance
- Increase `sequence_length` (more context)
- Increase model capacity (`hidden_size`, `num_layers`)
- Train for more epochs
- Try bidirectional LSTM (`bilstm`)

## Next Steps

1. **Experiment with different models**: Try GRU, BiLSTM
2. **Feature engineering**: Add more features or transformations
3. **Hyperparameter tuning**: Use grid search or random search
4. **Visualization**: Explore the notebooks for deeper analysis
5. **Deployment**: Create an API or web interface for predictions

## Example Python Usage

```python
# Load data
from src.data.data_loader import BatteryDataLoader
loader = BatteryDataLoader("data/raw")
df = loader.load_data()

# Train model
from src.training.train_lstm import BatteryTrainer
trainer = BatteryTrainer("config/config.yaml")
model = trainer.train()

# Evaluate
from src.evaluation.evaluate import BatteryEvaluator
evaluator = BatteryEvaluator("config/config.yaml", "models/best_model.pth")
metrics, y_true, y_pred = evaluator.evaluate(X_test, y_test)
```

Happy training! 🎉

