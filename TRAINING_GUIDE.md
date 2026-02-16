# 🎓 Training Guide

## Quick Training (For Testing)

**Recommended for first-time users:**

```bash
python quick_train.py
```

- **Time**: 2-5 minutes
- **Epochs**: 5
- **Purpose**: Quick test of API/Dashboard
- **Model**: Smaller, faster to train

## Full Training (For Production)

**For best model performance:**

```bash
python main.py train
```

- **Time**: 30-60 minutes
- **Epochs**: 100 (with early stopping)
- **Purpose**: Production-ready model
- **Model**: Full size, optimized

## Training Status

Check if model exists:
```bash
# Windows PowerShell
Test-Path models/best_model.pth

# Or check file
ls models/best_model.pth
```

## After Training

Once training completes, you'll have:
- ✅ `models/best_model.pth` - Trained model
- ✅ `models/scalers.pkl` - Data scalers
- ✅ `logs/training_history.png` - Training curves

Then you can:
1. **Start Flask API**: `python api/app.py`
2. **Start Streamlit**: `streamlit run dashboard/app.py`
3. **Evaluate**: `python main.py evaluate`

## Troubleshooting

**Out of Memory?**
- Reduce `batch_size` in config.yaml
- Reduce `hidden_size` or `num_layers`
- Use `quick_train.py` instead

**Training Too Slow?**
- Use GPU: Set `device: "cuda"` in config
- Reduce epochs temporarily
- Use smaller model

**Model Not Found?**
- Make sure training completed successfully
- Check `models/` directory exists
- Verify no errors during training

