"""
Streamlit Dashboard for EV Battery Degradation Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import yaml
import torch
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import BatteryDataLoader
from src.models.lstm_model import BatteryLSTM, BatteryGRU, BatteryBiLSTM

# Page config
st.set_page_config(
    page_title="EV Battery Degradation Prediction",
    page_icon="🔋",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔋 EV Battery Degradation & Range Prediction</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["📊 Data Overview", "🔮 Predictions", "📈 Model Performance", "⚙️ Settings"])

# Load config
@st.cache_data
def load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Load data
@st.cache_data
def load_battery_data():
    loader = BatteryDataLoader("data/raw")
    df = loader.load_data()
    return df, loader

if page == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    
    df, loader = load_battery_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Batteries", df['battery_id'].nunique())
    with col3:
        st.metric("Avg Cycles per Battery", f"{df.groupby('battery_id')['cycle'].max().mean():.0f}")
    with col4:
        st.metric("Avg SOH", f"{df['soh'].mean():.2%}")
    
    # Battery selection
    st.subheader("Select Battery to Analyze")
    battery_ids = sorted(df['battery_id'].unique())
    selected_battery = st.selectbox("Battery ID", battery_ids)
    
    battery_data = df[df['battery_id'] == selected_battery].sort_values('cycle')
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SOH Degradation")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(battery_data['cycle'], battery_data['soh'], linewidth=2, color='#1f77b4')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('State of Health (SOH)')
        ax.set_title(f'Battery {selected_battery} - SOH Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Capacity vs Cycle")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(battery_data['cycle'], battery_data['capacity'], linewidth=2, color='#ff7f0e')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Capacity (Ah)')
        ax.set_title(f'Battery {selected_battery} - Capacity Degradation')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Statistics
    st.subheader("Battery Statistics")
    st.dataframe(battery_data[['cycle', 'voltage', 'current', 'temperature', 'capacity', 'soh', 'range_km']].describe())

elif page == "🔮 Predictions":
    st.header("🔮 Battery Health Prediction")
    
    # Check if model exists
    model_path = Path(__file__).parent.parent / config['paths']['model_save_dir'] / "best_model.pth"
    
    if not model_path.exists():
        st.warning("⚠️ Model not found. Please train a model first using: `python main.py train`")
        st.info("You can still explore the data in the Data Overview page.")
    else:
        # Load model
        @st.cache_resource
        def load_model():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load checkpoint first to inspect architecture
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['model_state_dict']
            
            # Infer model architecture from state_dict
            is_bidirectional = any('reverse' in key for key in state_dict.keys())
            
            # Infer hidden_size from weight dimensions
            if 'lstm.weight_ih_l0' in state_dict:
                weight_shape = state_dict['lstm.weight_ih_l0'].shape
                hidden_size = weight_shape[0] // 4
                input_size = weight_shape[1]
            elif 'gru.weight_ih_l0' in state_dict:
                weight_shape = state_dict['gru.weight_ih_l0'].shape
                hidden_size = weight_shape[0] // 3
                input_size = weight_shape[1]
            else:
                hidden_size = config['model']['hidden_size']
                input_size = config['model']['input_size']
            
            # Infer num_layers
            num_layers = max([int(key.split('_l')[1].split('.')[0]) for key in state_dict.keys() 
                              if '_l' in key and key.split('_l')[1][0].isdigit()]) + 1
            
            # Infer output_size
            if 'fc3.weight' in state_dict:
                output_size = state_dict['fc3.weight'].shape[0]
            else:
                output_size = config['model']['output_size']
            
            # Determine model type
            if any('lstm' in key for key in state_dict.keys()):
                model_type = 'lstm'
            elif any('gru' in key for key in state_dict.keys()):
                model_type = 'gru'
            else:
                model_type = config['model']['type'].lower()
            
            dropout = config['model']['dropout']
            
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
            else:
                model = BatteryLSTM(**model_config)
            
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            return model, device
        
        model, device = load_model()
        
        # Load scalers
        scaler_path = Path(__file__).parent.parent / config['paths']['model_save_dir'] / "scalers.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                feature_scaler = scalers['feature_scaler']
                target_scaler = scalers['target_scaler']
        else:
            st.error("Scalers not found!")
            st.stop()
        
        st.subheader("Input Battery Data")
        
        # Option 1: Upload CSV
        uploaded_file = st.file_uploader("Upload CSV file with battery data", type=['csv'])
        
        # Option 2: Use existing battery
        df, _ = load_battery_data()
        battery_ids = sorted(df['battery_id'].unique())
        selected_battery = st.selectbox("Or select from existing batteries", [None] + list(battery_ids))
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.dataframe(input_df.head())
        elif selected_battery:
            battery_data = df[df['battery_id'] == selected_battery].sort_values('cycle')
            input_df = battery_data[['voltage', 'current', 'temperature', 'cycle', 'capacity']].tail(50)
            st.dataframe(input_df)
        else:
            st.info("Please upload a CSV file or select a battery from the dataset.")
            st.stop()
        
        # Prepare features
        if st.button("🔮 Predict", type="primary"):
            features = input_df[['voltage', 'current', 'temperature', 'cycle', 'capacity']].values
            
            sequence_length = config['data']['sequence_length']
            
            if len(features) < sequence_length:
                st.error(f"Need at least {sequence_length} samples. Got {len(features)}")
            else:
                # Use last sequence_length samples
                features = features[-sequence_length:]
                
                # Normalize
                features_scaled = feature_scaler.transform(features.reshape(1, -1, features.shape[1]))
                features_scaled = features_scaled.reshape(1, sequence_length, -1)
                
                # Predict
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features_scaled).to(device)
                    prediction_scaled = model(features_tensor).cpu().numpy()
                
                # Inverse transform
                prediction = target_scaler.inverse_transform(prediction_scaled)
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("State of Health (SOH)", f"{prediction[0][0]:.2%}")
                with col2:
                    st.metric("Range (km)", f"{prediction[0][1]:.2f}")
                with col3:
                    st.metric("Degradation Rate", f"{prediction[0][2]:.6f}")
                
                # Visualization
                st.subheader("Prediction Visualization")
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                metrics = ['SOH', 'Range (km)', 'Degradation Rate']
                values = [prediction[0][0], prediction[0][1], prediction[0][2]]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                
                for ax, metric, value, color in zip(axes, metrics, values, colors):
                    ax.bar([0], [value], color=color, alpha=0.7)
                    ax.set_title(metric)
                    ax.set_ylabel('Value')
                    ax.set_xticks([])
                
                plt.tight_layout()
                st.pyplot(fig)

elif page == "📈 Model Performance":
    st.header("📈 Model Performance Metrics")
    
    results_path = Path(__file__).parent.parent / config['paths']['results_dir']
    
    if (results_path / "predictions.npz").exists():
        data = np.load(results_path / "predictions.npz")
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        st.subheader("Prediction vs Actual")
        
        metrics = ['SOH', 'Range (km)', 'Degradation Rate']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (ax, metric) in enumerate(zip(axes, metrics)):
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20)
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax.set_xlabel(f'Actual {metric}')
            ax.set_ylabel(f'Predicted {metric}')
            ax.set_title(f'{metric} Prediction')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No evaluation results found. Run `python main.py evaluate` first.")

elif page == "⚙️ Settings":
    st.header("⚙️ Model Settings")
    
    st.subheader("Current Configuration")
    st.json(config)
    
    st.subheader("Model Information")
    model_path = Path(__file__).parent.parent / config['paths']['model_save_dir'] / "best_model.pth"
    
    if model_path.exists():
        st.success(f"✅ Model found: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        st.json({
            "epoch": checkpoint.get('epoch', 'N/A'),
            "validation_loss": checkpoint.get('val_loss', 'N/A')
        })
    else:
        st.warning("⚠️ Model not found. Train a model first.")

