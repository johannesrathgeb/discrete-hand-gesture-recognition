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
    def __init__(self, n_classes=6, in_channels=8, hidden_size=128, num_layers=3, dropout=0.3):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Temporal Extraction
        self.lstm = nn.LSTM(in_channels, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        # Classification Head
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, n_classes)
        # Activation and normalization
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.instance_norm_1 = nn.InstanceNorm1d(128)
        self.instance_norm_2 = nn.InstanceNorm1d(64)

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
        # use only the original num_windows (without padding) after LSTM layers
        true_end_outputs = torch.stack([lstm_out[i, lengths[i] - 1, :] for i in range(batch_size)])

        # Pass through fully connected layers
        x = self.fc1(true_end_outputs)
        x = self.instance_norm_1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.instance_norm_2(x)
        x = self.activation2(x)
        out = self.fc_out(x)
        return out
    
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=128, in_channels=8, n_classes=6, num_layers=3):
        """
        Initialize the gesture recognition model.
        
        Args:
            input_channels (int): Number of input channels (e.g., EMG channels).
            num_classes (int): Number of gesture classes to predict.
            hidden_size (int): Hidden size of the LSTM layers.
            lstm_layers (int): Number of LSTM layers.
        """
        super(CNN_LSTM, self).__init__()
        # 1D Convolutional Layer
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=hidden_size, kernel_size=3, padding=1, stride=1) 
        self.downpool = nn.MaxPool1d(kernel_size=5, stride=5)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()

        # LSTM Layers
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.layer_norm = nn.LayerNorm(hidden_size)
        # Fully Connected Layer for Classification
        self.fc = nn.Linear(hidden_size, n_classes)
    
    def forward(self, gestures, lengths):
        """
        @param gestures: batch of gesture rms-windows
        @param lengths: batch of original length of gestures (without padding)
        @return: final model predictions (logits)
        """
        # Apply 1D Convolution
        gestures = self.conv1d(gestures)  # Shape: (batch_size, hidden_size, sequence_length)
        gestures = self.downpool(gestures)  # Downsample the output of conv1d
        gestures = self.batch_norm(gestures)
        gestures = self.relu(gestures)
        # Transpose for LSTM (batch_size, sequence_length, hidden_size)
        gestures = gestures.transpose(1, 2)
        # LSTM Forward
        lstm_out, _ = self.lstm(gestures)

        # Apply Layer Normalization
        lstm_out = self.layer_norm(lstm_out)

        # Recalculate sequence lengths:
        #  after Conv1d
        k_c = self.conv1d.kernel_size[0]
        p_c = self.conv1d.padding[0]
        s_c = self.conv1d.stride[0]
        lengths = ((lengths + 2 * p_c - (k_c - 1) - 1) // s_c) + 1
        # # after Pooling
        kp = self.downpool.kernel_size
        if isinstance(kp, tuple):
            kp = kp[0]
        sp = self.downpool.stride if self.downpool.stride is not None else kp
        if isinstance(sp, tuple):
            sp = sp[0]
            
        lengths = ((lengths - (kp - 1) - 1) // sp) + 1
        out = lstm_out[torch.arange(lstm_out.size(0)), lengths - 1]
        # Fully connected layer
        logits = self.fc(out)  # Shape: (batch_size, num_classes)
        return logits

def get_model(config):
    if config.model == "cnn-lstm":
        m = CNN_LSTM(
                        config.hidden_size, 
                        config.in_channels, 
                        config.n_classes, 
                        config.num_layers
                    )
    else:
        m = LSTM(
                    config.n_classes,
                    config.in_channels, 
                    config.hidden_size, 
                    config.num_layers, 
                    config.dropout
                )
    return m