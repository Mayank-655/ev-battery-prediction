"""Quick test script to check dependencies and test data loading"""
import sys

print("="*60)
print("QUICK TEST - Checking Dependencies")
print("="*60)

# Check dependencies
missing = []
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except ImportError:
    print("✗ PyTorch: NOT INSTALLED")
    missing.append("torch")

try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except ImportError:
    print("✗ NumPy: NOT INSTALLED")
    missing.append("numpy")

try:
    import pandas
    print(f"✓ Pandas: {pandas.__version__}")
except ImportError:
    print("✗ Pandas: NOT INSTALLED")
    missing.append("pandas")

try:
    import scipy
    print(f"✓ SciPy: {scipy.__version__}")
except ImportError:
    print("✗ SciPy: NOT INSTALLED")
    missing.append("scipy")

try:
    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")
except ImportError:
    print("✗ Scikit-learn: NOT INSTALLED")
    missing.append("scikit-learn")

try:
    import yaml
    print("✓ PyYAML: OK")
except ImportError:
    print("✗ PyYAML: NOT INSTALLED")
    missing.append("pyyaml")

try:
    import tqdm
    print("✓ tqdm: OK")
except ImportError:
    print("✗ tqdm: NOT INSTALLED")
    missing.append("tqdm")

print("\n" + "="*60)
if missing:
    print(f"Missing packages: {', '.join(missing)}")
    print("\nInstall with: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("✓ All dependencies installed!")
    print("\n" + "="*60)
    print("Testing data loading...")
    print("="*60)
    
    # Test data loading
    try:
        from src.data.data_loader import BatteryDataLoader
        
        loader = BatteryDataLoader("data/raw")
        df = loader.load_data()
        
        print(f"\n✓ Data loaded successfully!")
        print(f"  Records: {len(df)}")
        print(f"  Batteries: {df['battery_id'].nunique()}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n✓ First few rows:")
        print(df.head(3))
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nProject is ready! You can now:")
        print("  - Train: python main.py train")
        print("  - Evaluate: python main.py evaluate")
        
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

