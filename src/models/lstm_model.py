"""
LSTM/GRU models for battery degradation prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatteryLSTM(nn.Module):
    """LSTM model for battery SOH, Range, and Degradation Rate prediction"""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 output_size: int = 3, bidirectional: bool = False):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Number of output targets (SOH, Range, Degradation)
            bidirectional: Whether to use bidirectional LSTM
        """
        super(BatteryLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out


class BatteryGRU(nn.Module):
    """GRU model for battery prediction (lighter alternative to LSTM)"""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 3, bidirectional: bool = False):
        """
        Initialize GRU model
        
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden state size
            num_layers: Number of GRU layers
            dropout: Dropout probability
            output_size: Number of output targets
            bidirectional: Whether to use bidirectional GRU
        """
        super(BatteryGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate GRU output size
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(gru_output_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor (batch_size, output_size)
        """
        # GRU forward pass
        gru_out, h_n = self.gru(x)
        
        # Use the last output
        last_output = gru_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out


class BatteryBiLSTM(nn.Module):
    """Bidirectional LSTM for enhanced temporal understanding"""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 3):
        """
        Initialize Bidirectional LSTM
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Number of output targets
        """
        super(BatteryBiLSTM, self).__init__()
        
        # Use bidirectional LSTM
        self.lstm = BatteryLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
            bidirectional=True
        )
        
    def forward(self, x):
        """Forward pass"""
        return self.lstm(x)

