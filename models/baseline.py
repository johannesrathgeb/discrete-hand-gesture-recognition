import torch
import torch.nn as nn
import torch.fft as fft

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
    
class GLFCNN(nn.Module):
    def __init__(self, num_channels=8, window_length=77, num_classes=6, dropout=0.2):
        super(GLFCNN, self).__init__()
        # Local Feature Extraction Branch
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=128, kernel_size=10, stride=6, padding=0),
            nn.ELU(),
            nn.LayerNorm([128, (window_length - 10) // 6 + 1])  # Normalize across channels and time
        )
        
        # Global Feature Extraction Branch
        self.global_conv = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=128, kernel_size=10, stride=6, padding=0),
            nn.ELU(),
            nn.LayerNorm([128, (window_length - 10) // 6 + 1])
        )

        self.depthwise_separable = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, groups=256, padding=1),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),  # Reduce to 128 channels
            nn.ELU()
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 512),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, lengths):
        # Local Feature Extraction
        batch_size, num_windows, num_channels, window_length = x.shape
        
        # Process each window independently
        x = x.view(batch_size * num_windows, num_channels, window_length)  # Flatten batch and windows
        local_features = self.local_conv(x)
        # Global Feature Extraction
        # Apply FFT, then IFFT after convolution for global features
        global_features_freq = fft.fft(x, dim=-1)
        global_features_time = fft.ifft(self.global_conv(torch.abs(global_features_freq)), dim=-1).real
        
        # Combine local and global features
        combined_features = torch.cat((local_features, global_features_time), dim=1)  # Shape: (batch_size * num_windows, 256, time_steps)
        # Step 2: Apply Depthwise Separable Convolution
        depthwise_output = self.depthwise_separable(combined_features)
        # Step 3: Add the output of the DWConv back to the combined features (residual connection)
        fused_features = (local_features + global_features_time) + depthwise_output
        
        # Global Average Pooling
        pooled_features = self.global_avg_pool(fused_features).squeeze(-1)
        window_preds = self.fc(pooled_features)  # Shape: (batch_size * num_windows, num_classes)
        # Reshape window-level predictions back to (batch_size, num_windows, num_classes)
        window_preds = window_preds.view(batch_size, num_windows, -1)
        gesture_preds = []
        for i in range(batch_size):
            valid_window_preds = window_preds[i, :lengths[i]]  # Select only the valid windows for this batch element
            gesture_pred = valid_window_preds.mean(dim=0)  # Aggregate predictions for the valid windows
            gesture_preds.append(gesture_pred)

        gesture_preds = torch.stack(gesture_preds)  # Shape: (batch_size, num_classes)
        return gesture_preds

class DiscreteGestureRecognitionModel(nn.Module):
    def __init__(self, hidden_size=128, input_channels=8, num_classes=6, lstm_layers=3):
        """
        Initialize the gesture recognition model.
        
        Args:
            input_channels (int): Number of input channels (e.g., EMG channels).
            num_classes (int): Number of gesture classes to predict.
            hidden_size (int): Hidden size of the LSTM layers.
            lstm_layers (int): Number of LSTM layers.
        """
        super(DiscreteGestureRecognitionModel, self).__init__()
        
        # 1D Convolutional Layer
        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=hidden_size, kernel_size=3, padding=1, stride=1) # No stride because data already is in 200Hz
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # LSTM Layers
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Fully Connected Layer for Classification
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length).
            lengths (torch.Tensor): Original lengths of each sequence in the batch.
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # Apply 1D Convolution
        x = self.conv1d(x)  # Shape: (batch_size, hidden_size, sequence_length)
        x = self.relu(x)
        x = self.batch_norm(x)

        # Transpose for LSTM (batch_size, sequence_length, hidden_size)
        x = x.transpose(1, 2)
        # LSTM Forward
        lstm_out, _ = self.lstm(x)        

        # Apply Layer Normalization
        lstm_out = self.layer_norm(lstm_out)

        # Use the last time-step output for classification
        out = lstm_out[torch.arange(lstm_out.size(0)), lengths - 1]
        
        # Fully connected layer
        logits = self.fc(out)  # Shape: (batch_size, num_classes)
        return logits

def get_model(config):
    if config.model == "cnn-lstm":
        m = DiscreteGestureRecognitionModel(config.hidden_size, 
                                            config.in_channels, 
                                            config.n_classes, 
                                            config.num_layers
                                        )
    elif config.model == "cnn":
        m = GLFCNN(config.in_channels, 
                   int((config.window_length * 1000) / (1000 / config.sample_freq)), 
                   config.n_classes, 
                   config.dropout
                )
    else:
        m = LSTM(config.n_classes,
                 config.in_channels, 
                 config.hidden_size, 
                 config.num_layers, 
                 config.dropout
                )
    return m