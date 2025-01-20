import os
import torch
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
import mne
from sklearn.preprocessing import MinMaxScaler

DATASET_DIR = 'raw_data/eeg-motor-movement/'
assert DATASET_DIR is not None, "Specify 'EEG Motor Imaginary Movement dataset' location in variable " \
                                "'DATASET_DIR'. The dataset can be downloaded from this URL:" \
                                " https://zenodo.org/records/4421500"

dataset_config = {
    "meta_csv": os.path.join(DATASET_DIR, "eeg-motor-movement-metadata.csv"),
}
from scipy.signal import butter, lfilter, iirnotch
class EEGMotorMovementDataset(TorchDataset):
    """
    Basic EEG Motor Imaginary Movement Dataset: loads data from files
    """
    def __init__(self, grouped_df, fs=160, window_size=0.25, overlap=0.0, max_samples=656, min_samples=476, window_mode="rms", scaling=False):
        """
        @param grouped_df: dataset df grouped by File Path
        @param fs: frequency sample rate in Hz
        @param window_size: length of window in ms
        @param overlap: overlap of windows in ms
        @param max_samples: max sample length in dataset (599 for EMG-EPN612)
        """  
        self.fs = fs
        self.window_mode = window_mode
        self.window_samples = int(window_size * fs)
        self.step_samples = int(self.window_samples * (1 - overlap))
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.max_windows = (self.max_samples - self.window_samples) // self.step_samples + 1
        self.scaling = scaling

        print("max samples", self.max_samples)
        print("max windows", self.max_windows)
        labels = []
        locations = []
        start_indices = []
        end_indices = []

        all_data = []
        for folder_path, group in grouped_df:
            for label in group["Label"].unique():
                label_group = group[group["Label"] == label]
                labels.extend(label_group["Label"].values)
                locations.extend(label_group["File_Path"].values)
                start_indices.extend(label_group["Start_Index"].values)
                end_indices.extend(label_group["End_Index"].values)
                if scaling:
                    for _, row in label_group.iterrows():
                        file_path = row["File_Path"]
                        start_idx = row["Start_Index"]
                        end_idx = row["End_Index"]
                        eeg_data = self.get_eeg_data(file_path, start_idx, end_idx, str(file_path) + '.csv', True)
                        all_data.append(eeg_data)
        if scaling:
            all_data = np.stack(all_data)  
            reshaped_x = all_data.reshape(all_data.shape[0], all_data.shape[1] * all_data.shape[2])
            self.scaler = MinMaxScaler()
            self.scaler.fit(reshaped_x)  
            print(f"Scaler fitted: Min={self.scaler.data_min_}, Max={self.scaler.data_max_}")
            del all_data

        labels = np.array(labels)
        locations = np.array(locations)
        start_indices = np.array(start_indices)
        end_indices = np.array(end_indices)

        # Encode labels into integer values, print unique counts
        le = preprocessing.LabelEncoder()
        labels = torch.from_numpy(le.fit_transform(labels.reshape(-1)))
        self.print_unique_labels(labels)   

        # save all data into flat array for later use
        self.all_gestures = self.create_flat_array(labels, locations, start_indices, end_indices)
        print(self.all_gestures.shape)

    def print_unique_labels(self, labels):
        unique_labels, counts = torch.unique(labels, return_counts=True)
        print("Unique labels and counts:")
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            print(f"Label {label}: {count} occurrences")

    def create_flat_array(self, labels, locations, start_indices, end_indices):
        flat_array = []
        label_idx = 0
        for file_path in locations:
            flat_array.append([file_path, labels[label_idx], start_indices[label_idx], end_indices[label_idx]])
            label_idx += 1
        return np.array(flat_array, dtype=object)
    

    def get_eeg_data(self, file_path, start_idx, end_idx, output_file, padding=False):
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        

        raw_data_segment, times = raw[:, start_idx:end_idx]

        eeg_data = np.array(raw_data_segment)
        selected_channels = [10, 17, 18, 11, 16, 9, 50, 51]
        if selected_channels:
            eeg_data = eeg_data[selected_channels, :]

        def apply_notch_filter(data, freq, fs, quality_factor=30):
            notch_freq = freq / (fs / 2)  # Normalized frequency
            b, a = iirnotch(notch_freq, quality_factor)
            return lfilter(b, a, data, axis=1)

        # Define Butterworth band-pass filter
        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return lfilter(b, a, data, axis=1)

        eeg_data = apply_notch_filter(eeg_data, freq=60, fs=160)

        # Apply Butterworth band-pass filter
        eeg_data = butter_bandpass_filter(eeg_data, lowcut=2, highcut=60, fs=160)

        if padding:
            num_samples = eeg_data.shape[1]
            pad_size = self.max_samples - num_samples
            if pad_size > 0:
                padding = np.zeros((eeg_data.shape[0], pad_size))
                eeg_data = np.hstack((eeg_data, padding))
            eeg_data = eeg_data[:, :self.max_samples]
        return eeg_data 

    def get_windows_rms(self, eeg_data):
        # calculate number of windows for data length
        num_samples = eeg_data.shape[1]
        if num_samples < self.window_samples:
            num_windows = 1
        else:
            num_windows = (num_samples - self.window_samples) // self.step_samples + 1

        rms_windows = []    
        for i in range(num_windows):
            # calculate start and end point for window
            start = i * self.step_samples
            end = start + self.window_samples
            
            # zero-pad window if it's longer than remaining data
            if end > num_samples:
                segment = np.pad(eeg_data, ((0, 0), (0, end - num_samples)), mode='constant')
            else:
                segment = eeg_data[:, start:end]
            # make sure window has correct length
            segment = segment[:, :self.window_samples]
            # Compute RMS for each channel in the window
            rms_feature = np.sqrt(np.mean(np.square(segment), axis=1))  
            rms_windows.append(rms_feature)
        windows_np = np.array(rms_windows) 

        if len(windows_np) > self.max_windows:
            # Trim to max_windows if there are too many
            windows_np = windows_np[:self.max_windows]
            num_windows = self.max_windows
        # pad array of window to max number of windows in dataset to have consistent shapes
        pad_size = self.max_windows - len(windows_np)
        if pad_size > 0:
            padding = np.zeros((pad_size, windows_np.shape[1]))  # Pad with zeros
            windows_np = np.vstack((windows_np, padding))
        
        # return windows tensor of consistent shape and true number of windows (without padding) 
        return torch.tensor(windows_np, dtype=torch.float32), num_windows

    def get_high_pass_filtered_data(self, eeg_data):
        num_samples = eeg_data.shape[1]
        pad_size = self.max_samples - num_samples
        if pad_size > 0:
            padding = np.zeros((eeg_data.shape[0], pad_size))
            eeg_data = np.hstack((eeg_data, padding))
        
        if num_samples > self.max_samples:
            num_samples = self.max_samples
            eeg_data = eeg_data[:, :self.max_samples]
        # Ensure emg_data has contiguous memory layout
        eeg_data = eeg_data.copy()
        # return windows tensor of consistent shape and true number of windows (without padding) 
        return torch.tensor(eeg_data, dtype=torch.float32), num_samples

    def __getitem__(self, index):
        # Step 1: Fetch file path, gesture index, label and location
        file_path, label, start_idx, end_idx = self.all_gestures[index]
        # Step 2: Load EMG data from edf file
        emg_data = self.get_eeg_data(file_path, start_idx, end_idx, str(index) + '.csv', False)
        # if self.scaling:
        #     eeg_data_scaled_flat = self.scaler.transform(emg_data.reshape(1, -1))
        #     eeg_data_scaled = eeg_data_scaled_flat.reshape(emg_data.shape)  # Shape: (channels, samples)
        #     emg_data = eeg_data_scaled
        # Step 3: Create RMS windows and get original length
        match self.window_mode:
            case "rms":
                windows, original_length = self.get_windows_rms(emg_data)
            case "highpass":
                windows, original_length = self.get_high_pass_filtered_data(emg_data)
            case _:  # Default case
                windows, original_length = self.get_windows_rms(emg_data)

        return windows, original_length, label

    def __len__(self):
        return len(self.all_gestures)


def get_training_set_eeg(config, validation=True):
    """
    @param validation: flag wether training set should be split into train/val
    @return: TorchDataset with Training Samples, TorchDataset with Validation Samples (will be None if validation=False)
    """
    meta_csv = dataset_config['meta_csv']
    df = pd.read_csv(meta_csv)
    df['Folder_Path'] = df['File_Path'].apply(lambda x: os.path.dirname(x))
    grouped = df.groupby('Folder_Path')
    num_groups = len(grouped)
    group_keys = list(grouped.groups.keys())  # Get the group keys (folder paths)

    # Shuffle group keys randomly
    original_state = random.getstate()
    random.seed(config.running_seed)
    random.shuffle(group_keys)
    # random.setstate(original_state)



    split_index = int(num_groups * 0.8)  # Calculate 80% index
    selected_groups = group_keys[:split_index]  # Select the first 80%


    # Filter the original DataFrame to include only the selected groups
    grouped = df[df['Folder_Path'].isin(selected_groups)]
    grouped = grouped.groupby('Folder_Path')
    
    if validation:
        # Get all unique groups (users) and nsure there are enough users for both splits
        all_groups = list(grouped.groups.keys())
        # print(len(all_groups), config.n_train_users, config.n_val_users)
        assert len(all_groups) >= (config.n_train_users + config.n_val_users), \
            "Not enough unique users for the specified training and validation splits."

        # Randomly select training users
        selected_train_groups = random.sample(all_groups, config.n_train_users)
        # Remaining groups after selecting training users
        remaining_groups = [group for group in all_groups if group not in selected_train_groups]
        # Randomly select validation users from the remaining groups
        selected_val_groups = random.sample(remaining_groups, config.n_val_users)
        # Filter the DataFrame for training users and validation users
        training_df = grouped.filter(lambda x: x.name in selected_train_groups)
        validation_df = grouped.filter(lambda x: x.name in selected_val_groups)

        validation_df = validation_df.groupby("Folder_Path")
        training_df = training_df.groupby("Folder_Path")
        ds_training = EEGMotorMovementDataset(training_df, window_mode=config.window_mode, window_size=config.window_length)
        ds_validation = EEGMotorMovementDataset(validation_df, window_mode=config.window_mode, window_size=config.window_length)

        random.setstate(original_state)
        return ds_training, ds_validation, selected_groups
    else:
        ds_training = EEGMotorMovementDataset(grouped, window_mode=config.window_mode, window_size=config.window_length)
        random.setstate(original_state)
        return ds_training, None

def get_testing_set_eeg(config, training_group_keys):
    """
    Returns:
    Single TorchDataset with Testing Samples
    """
    meta_csv = dataset_config['meta_csv']
    df = pd.read_csv(meta_csv)
    df['Folder_Path'] = df['File_Path'].apply(lambda x: os.path.dirname(x))
    grouped = df.groupby('Folder_Path')
    group_keys = list(grouped.groups.keys())  # Get the group keys (folder paths)

    original_state = random.getstate()
    random.seed(config.running_seed)
    random.shuffle(group_keys)

    testing_group_keys = [key for key in group_keys if key not in training_group_keys]

    # Filter the original DataFrame to include only the selected groups
    grouped = df[df['Folder_Path'].isin(testing_group_keys)]
    grouped = grouped.groupby('Folder_Path')
    ds_testing = EEGMotorMovementDataset(grouped, window_mode=config.window_mode, window_size=config.window_length)
    random.setstate(original_state)
    return ds_testing
