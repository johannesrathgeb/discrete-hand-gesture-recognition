import torch
import torch.nn as nn

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class LSTM(nn.Module):
    def __init__(self, model_config):
        super(LSTM, self).__init__()
        self.num_layers = model_config["num_layers"]
        self.hidden_size = model_config["hidden_size"]
        
        # Temporal Extraction
        self.lstm = nn.LSTM(model_config["in_channels"], self.hidden_size, self.num_layers, batch_first=True, dropout=model_config["dropout"])
        
        # Classification Head
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, model_config["n_classes"])
        
        # Activation and normalization
        self.instance_norm = nn.InstanceNorm1d(self.hidden_size)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.apply(initialize_weights)

    def forward(self, gestures, lengths):
        """
        @param gestures: batch of gesture rms-windows
        @param lengths: batch of original number of windows of gesture (without padding)
        @return: final model predictions (logits)
        """
        batch_size = gestures.size(0)
        # reset LSTM hidden state and cell state before starting new sequence
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(gestures.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(gestures.device)

        # Pass the entire batch through the LSTM
        lstm_out, _ = self.lstm(gestures, (h0, c0))

        # Use instance normalization on the LSTM output
        lstm_out = self.instance_norm(lstm_out.permute(0, 2, 1)).permute(0, 2, 1)

        # use only the original num_windows (without padding) after LSTM layers
        true_end_outputs = torch.stack([lstm_out[i, lengths[i] - 1, :] for i in range(batch_size)])

        # Pass through fully connected layers
        x = self.fc1(true_end_outputs)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        logits = self.fc_out(x)
        return logits


def get_model(n_classes=6, in_channels=8, hidden_size=128, num_layers=3, dropout=0.3):
    """
    @param n_classes: number of the classes to predict
    @param in_channels: input channels of the emg windows
    @param hidden_size: hidden size of the model
    @param num_layers: number of LSTM Layers
    @param dropout: dropout applied to LSTM Layers
    @return: full neural network model based on the specified configs 
    """
    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "num_layers": num_layers,
    }

    m = LSTM(model_config)
    return m