"""
Flask REST API for EV Battery Degradation Prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import yaml
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model import BatteryLSTM, BatteryGRU, BatteryBiLSTM
from src.data.data_loader import BatteryDataLoader

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None
scaler = None
config = None
device = None


def load_model_and_scalers():
    """Load model and scalers at startup"""
    global model, scaler, config, device
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load scalers
    scaler_path = Path(__file__).parent.parent / config['paths']['model_save_dir'] / "scalers.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            scaler = {
                'feature': scalers['feature_scaler'],
                'target': scalers['target_scaler']
            }
    else:
        scaler = None
        print("Warning: Scalers not found. Model may not work correctly.")
    
    # Load model
    model_path = Path(__file__).parent.parent / config['paths']['model_save_dir'] / "best_model.pth"
    if model_path.exists():
        # Load checkpoint first to inspect architecture
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # Infer model architecture from state_dict
        # Check if it's bidirectional by looking at weight names
        is_bidirectional = any('reverse' in key for key in state_dict.keys())
        
        # Infer hidden_size from weight dimensions
        # LSTM weight_ih_l0 shape is [4*hidden_size, input_size]
        if 'lstm.weight_ih_l0' in state_dict:
            weight_shape = state_dict['lstm.weight_ih_l0'].shape
            hidden_size = weight_shape[0] // 4
            input_size = weight_shape[1]
        elif 'gru.weight_ih_l0' in state_dict:
            weight_shape = state_dict['gru.weight_ih_l0'].shape
            hidden_size = weight_shape[0] // 3
            input_size = weight_shape[1]
        else:
            # Fallback to config
            hidden_size = config['model']['hidden_size']
            input_size = config['model']['input_size']
        
        # Infer num_layers by counting layer keys
        num_layers = max([int(key.split('_l')[1].split('.')[0]) for key in state_dict.keys() 
                          if '_l' in key and key.split('_l')[1][0].isdigit()]) + 1
        
        # Infer output_size from fc3 weight
        if 'fc3.weight' in state_dict:
            output_size = state_dict['fc3.weight'].shape[0]
        else:
            output_size = config['model']['output_size']
        
        # Determine model type
        if 'lstm' in state_dict and any('lstm' in key for key in state_dict.keys()):
            model_type = 'lstm'
        elif 'gru' in state_dict and any('gru' in key for key in state_dict.keys()):
            model_type = 'gru'
        else:
            model_type = config['model']['type'].lower()
        
        # Get dropout from config (not stored in state_dict)
        dropout = config['model']['dropout']
        
        print(f"Detected model architecture:")
        print(f"  Type: {model_type}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Layers: {num_layers}")
        print(f"  Bidirectional: {is_bidirectional}")
        print(f"  Output size: {output_size}")
        
        # Create model with detected architecture
        model_config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'output_size': output_size,
            'bidirectional': is_bidirectional
        }
        
        if model_type == 'lstm':
            model = BatteryLSTM(**model_config)
        elif model_type == 'gru':
            model = BatteryGRU(**model_config)
        elif model_type == 'bilstm':
            model = BatteryBiLSTM(**model_config)
        else:
            model = BatteryLSTM(**model_config)
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully!")
    else:
        print("Warning: Model not found. Please train a model first.")


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        "message": "EV Battery Degradation Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict SOH, Range, and Degradation Rate",
            "/health": "GET - Health check",
            "/model_info": "GET - Model information"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 404
    
    # Get actual model architecture
    if hasattr(model, 'lstm'):
        model_type = 'lstm'
        hidden_size = model.hidden_size
        num_layers = model.num_layers
        bidirectional = model.bidirectional
    elif hasattr(model, 'gru'):
        model_type = 'gru'
        hidden_size = model.hidden_size
        num_layers = model.num_layers
        bidirectional = model.bidirectional
    else:
        model_type = 'unknown'
        hidden_size = config['model']['hidden_size']
        num_layers = config['model']['num_layers']
        bidirectional = False
    
    return jsonify({
        "model_type": model_type,
        "input_size": config['model']['input_size'],
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "output_size": config['model']['output_size'],
        "sequence_length": config['data']['sequence_length'],
        "parameters": sum(p.numel() for p in model.parameters())
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict battery health metrics
    
    Expected JSON:
    {
        "features": [
            [voltage, current, temperature, cycle, capacity],
            ...
        ]  # Must be at least sequence_length (50) samples
    }
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model or scalers not loaded. Please train a model first."}), 500
    
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        features = np.array(data['features'])
        
        # Check if we have enough samples for sequence
        sequence_length = config['data']['sequence_length']
        if len(features) < sequence_length:
            return jsonify({
                "error": f"Need at least {sequence_length} samples. Got {len(features)}"
            }), 400
        
        # Use last sequence_length samples
        features = features[-sequence_length:]
        
        # Normalize features
        features_scaled = scaler['feature'].transform(features.reshape(1, -1, features.shape[1]))
        features_scaled = features_scaled.reshape(1, sequence_length, -1)
        
        # Predict
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            prediction_scaled = model(features_tensor).cpu().numpy()
        
        # Inverse transform
        prediction = scaler['target'].inverse_transform(prediction_scaled)
        
        # Format response
        result = {
            "predictions": {
                "soh": float(prediction[0][0]),
                "range_km": float(prediction[0][1]),
                "degradation_rate": float(prediction[0][2])
            },
            "units": {
                "soh": "percentage (0-1)",
                "range_km": "kilometers",
                "degradation_rate": "per cycle"
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple sequences
    
    Expected JSON:
    {
        "sequences": [
            [[voltage, current, temp, cycle, capacity], ...],  # sequence 1
            [[voltage, current, temp, cycle, capacity], ...],  # sequence 2
            ...
        ]
    }
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model or scalers not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if 'sequences' not in data:
            return jsonify({"error": "Missing 'sequences' in request"}), 400
        
        sequences = data['sequences']
        sequence_length = config['data']['sequence_length']
        
        predictions = []
        
        for seq in sequences:
            seq_array = np.array(seq)
            
            if len(seq_array) < sequence_length:
                continue
            
            # Use last sequence_length samples
            seq_array = seq_array[-sequence_length:]
            
            # Normalize
            seq_scaled = scaler['feature'].transform(seq_array.reshape(1, -1, seq_array.shape[1]))
            seq_scaled = seq_scaled.reshape(1, sequence_length, -1)
            
            # Predict
            with torch.no_grad():
                seq_tensor = torch.FloatTensor(seq_scaled).to(device)
                pred_scaled = model(seq_tensor).cpu().numpy()
            
            # Inverse transform
            pred = scaler['target'].inverse_transform(pred_scaled)
            
            predictions.append({
                "soh": float(pred[0][0]),
                "range_km": float(pred[0][1]),
                "degradation_rate": float(pred[0][2])
            })
        
        return jsonify({
            "predictions": predictions,
            "count": len(predictions)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Loading model and scalers...")
    load_model_and_scalers()
    print("Starting Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)

