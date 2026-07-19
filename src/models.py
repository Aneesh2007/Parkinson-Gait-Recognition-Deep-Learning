import torch
import torch.nn as nn
import math


class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 18, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1)

        )

    def forward(self, x):

        return self.network(x)
    

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=18,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(32, 1)

        )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.features(x)

        x = self.classifier(x)

        return x
    
class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=18,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Last timestep from both directions
        x = output[:, -1, :]

        x = self.classifier(x)

        return x
    



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]
    

class TransformerClassifier(nn.Module):

    def __init__(
        self,
        input_size=18,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.3
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        self.position_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(

            nn.Linear(d_model, 64),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(64, 1)

        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.position_encoding(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.classifier(x)

        return x
    

class GatedResidualNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        dropout=0.3
    ):

        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.elu = nn.ELU()

        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

        self.skip = nn.Linear(input_dim, output_dim) \
            if input_dim != output_dim else nn.Identity()

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):

        residual = self.skip(x)

        x = self.fc1(x)

        x = self.elu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        gate = self.gate(x)

        x = gate * x + residual

        x = self.norm(x)

        return x
    
class VariableSelectionNetwork(nn.Module):

    def __init__(
        self,
        num_features,
        hidden_dim
    ):

        super().__init__()

        self.weights = nn.Sequential(

            nn.Linear(num_features, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, num_features),

            nn.Softmax(dim=-1)

        )

    def forward(self, x):

        attention = self.weights(x)

        return x * attention
    
class TFTClassifier(nn.Module):

    def __init__(
        self,
        input_size=18,
        hidden_size=64,
        num_lstm_layers=1,
        num_heads=4,
        dropout=0.3
    ):

        super().__init__()

        # Variable Selection
        self.variable_selection = VariableSelectionNetwork(
            num_features=input_size,
            hidden_dim=hidden_size
        )

        # Feature projection
        self.input_projection = nn.Linear(
            input_size,
            hidden_size
        )

        # Gated Residual Network
        self.grn = GatedResidualNetwork(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            dropout=dropout
        )

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Multi-head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(hidden_size)

        # Classification Head
        self.classifier = nn.Sequential(

            nn.Linear(hidden_size, 64),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(64, 1)

        )

    def forward(self, x):

        # Variable Selection
        x = self.variable_selection(x)

        # Feature Projection
        x = self.input_projection(x)

        # GRN
        x = self.grn(x)

        # LSTM
        x, _ = self.lstm(x)

        # Self Attention
        attn_output, _ = self.attention(
            x,
            x,
            x
        )

        x = self.norm(x + attn_output)

        # Global Average Pooling
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)

        return x