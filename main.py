"""
Main entry point for EV Battery Degradation & Range Prediction System
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.train_lstm import BatteryTrainer
from src.evaluation.evaluate import BatteryEvaluator
from src.data.data_loader import BatteryDataLoader
from src.data.preprocess import BatteryPreprocessor


def train_model(config_path: str = "config/config.yaml"):
    """Train the battery prediction model"""
    print("="*60)
    print("EV BATTERY DEGRADATION PREDICTION - TRAINING")
    print("="*60)
    
    trainer = BatteryTrainer(config_path)
    model = trainer.train()
    
    print("\nTraining completed successfully!")
    return model


def evaluate_model(config_path: str = "config/config.yaml", 
                   model_path: str = "models/best_model.pth"):
    """Evaluate the trained model"""
    print("="*60)
    print("EV BATTERY DEGRADATION PREDICTION - EVALUATION")
    print("="*60)
    
    # Load and preprocess test data
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    loader = BatteryDataLoader(config['data']['raw_data_path'])
    df = loader.load_data()
    
    features = loader.get_features()
    targets = loader.get_targets()
    
    preprocessor = BatteryPreprocessor(
        sequence_length=config['data']['sequence_length'],
        normalize=True
    )
    
    # Load scalers
    scaler_path = f"{config['paths']['model_save_dir']}/scalers.pkl"
    if Path(scaler_path).exists():
        preprocessor.load_scalers(scaler_path)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(
        features, targets,
        train_size=config['data']['train_split'],
        val_size=config['data']['val_split']
    )
    
    # Evaluate
    evaluator = BatteryEvaluator(config_path, model_path)
    metrics, y_true, y_pred = evaluator.evaluate(X_test, y_test)
    
    return metrics


def explore_data():
    """Explore the dataset"""
    print("="*60)
    print("EV BATTERY DEGRADATION PREDICTION - DATA EXPLORATION")
    print("="*60)
    
    loader = BatteryDataLoader("data/raw")
    df = loader.load_data()
    
    print(f"\nDataset Overview:")
    print(f"  Total records: {len(df)}")
    print(f"  Number of batteries: {df['battery_id'].nunique()}")
    print(f"  Columns: {list(df.columns)}")
    
    print(f"\nData Statistics:")
    print(df.describe())
    
    print(f"\nSample data:")
    print(df.head(10))
    
    return df


def test_generalization(config_path: str = "config/config.yaml",
                        model_path: str = "models/best_model.pth"):
    """Test model generalization across different batteries"""
    print("="*60)
    print("EV BATTERY DEGRADATION PREDICTION - GENERALIZATION TEST")
    print("="*60)
    
    from src.evaluation.generalization_test import GeneralizationTester
    
    tester = GeneralizationTester(config_path)
    metrics = tester.test_generalization(model_path)
    
    return metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='EV Battery Degradation & Range Prediction System'
    )
    parser.add_argument(
        'mode',
        choices=['train', 'evaluate', 'explore', 'generalization'],
        help='Mode: train, evaluate, explore data, or test generalization'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.pth',
        help='Path to model file (for evaluation)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args.config)
    elif args.mode == 'evaluate':
        evaluate_model(args.config, args.model)
    elif args.mode == 'explore':
        explore_data()
    elif args.mode == 'generalization':
        test_generalization(args.config, args.model)


if __name__ == "__main__":
    main()

