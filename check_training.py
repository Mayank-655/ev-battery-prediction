"""Check if training is complete"""
from pathlib import Path
import time

print("Checking training status...")
print("="*60)

model_path = Path("models/best_model.pth")
scaler_path = Path("models/scalers.pkl")

if model_path.exists() and scaler_path.exists():
    print("✅ Training Complete!")
    print(f"✅ Model: {model_path}")
    print(f"✅ Scalers: {scaler_path}")
    print("\n🚀 You can now:")
    print("  1. Start Flask API: python api/app.py")
    print("  2. Start Streamlit: streamlit run dashboard/app.py")
    print("  3. Test API: python api/test_api.py")
else:
    print("⏳ Training in progress or not started...")
    if not model_path.exists():
        print(f"   - Model not found: {model_path}")
    if not scaler_path.exists():
        print(f"   - Scalers not found: {scaler_path}")
    print("\n💡 Start training with:")
    print("  python quick_train.py  (quick - 5 epochs)")
    print("  python main.py train   (full - 100 epochs)")

