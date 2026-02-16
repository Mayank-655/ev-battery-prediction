# 🚀 Quick Start - Train Model First!

## ⚠️ Important: Train Model Before Using API/Dashboard

The Flask API and Streamlit Dashboard require a trained model. Follow these steps:

## Step 1: Train the Model

### Option A: Quick Training (Recommended for Testing)
```bash
python quick_train.py
```
- **Time**: 2-5 minutes
- **Epochs**: 5
- **Best for**: Testing deployments

### Option B: Full Training (For Production)
```bash
python main.py train
```
- **Time**: 30-60 minutes  
- **Epochs**: 100
- **Best for**: Production use

## Step 2: Check Training Status

```bash
python check_training.py
```

You should see:
```
✅ Training Complete!
✅ Model: models/best_model.pth
✅ Scalers: models/scalers.pkl
```

## Step 3: Start Deployment

### Flask API
```bash
cd api
pip install -r requirements.txt
python app.py
```

### Streamlit Dashboard
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting

**"Model not found" error?**
- Make sure training completed successfully
- Check: `python check_training.py`
- If not complete, run: `python quick_train.py`

**Training taking too long?**
- Use `quick_train.py` instead (5 epochs)
- Reduce batch_size in config.yaml
- Use smaller model (reduce hidden_size)

**Out of memory?**
- Reduce batch_size: 16 or 8
- Reduce hidden_size: 64
- Use quick_train.py (smaller model)

---

**Once training completes, you're ready to deploy!** 🎉

