import os
import h5py
import torch
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random

DATASET_DIR = 'raw_data/EMG-EPN612/'
assert DATASET_DIR is not None, "Specify 'EMG-EPN612 dataset' location in variable " \
                                "'DATASET_DIR'. The dataset can be downloaded from this URL:" \
                                " https://zenodo.org/records/4421500"

dataset_config = {
    "training_meta_csv": os.path.join(DATASET_DIR, "training-metadata.csv"),
    "testing_meta_csv": os.path.join(DATASET_DIR, "testing-metadata.csv"),
}

class EMGEPN612Dataset(TorchDataset):
    """
    EMG-EPN612 Dataset: loads data from files
    """
    def __init__(self, grouped_df, fs=200, window_size=0.025, overlap=0.0, max_samples=599, min_samples=76, num_reps=50, window_mode="rms"):
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

        labels = []
        locations = []
        gestures = []

        for file_path, group in grouped_df:
            for label in group["Label"].unique():
                label_group = group[group["Label"] == label]
                sampled = label_group.sample(num_reps, replace=False, random_state=42)
                # Extend the lists with the sampled data
                labels.extend(sampled["Label"].values)
                locations.extend(sampled["Location"].values)
                gestures.append((file_path, sampled["Gesture_Index"].values.tolist()))

        labels = np.array(labels)
        locations = np.array(locations)
        gestures = np.array(gestures, dtype=object)

        # Encode labels into integer values, print unique counts
        le = preprocessing.LabelEncoder()
        labels = torch.from_numpy(le.fit_transform(labels.reshape(-1)))
        self.print_unique_labels(labels)   

        # save all data into flat array for later use
        self.all_gestures = self.create_flat_array(labels, gestures, locations)

    def print_unique_labels(self, labels):
        unique_labels, counts = torch.unique(labels, return_counts=True)
        print("Unique labels and counts:")
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            print(f"Label {label}: {count} occurrences")

    def create_flat_array(self, labels, gestures, locations):
        flat_array = []
        label_idx = 0
        for file_path, gesture_indices in gestures:
            for gesture_index in gesture_indices:
                flat_array.append([file_path, gesture_index, labels[label_idx], locations[label_idx]])
                label_idx += 1
        return np.array(flat_array, dtype=object)
    
    def get_emg_data(self, file_path, sample, location):
        # convert file path to h5 instead of json
        file_path = os.path.splitext(file_path)[0] + '.h5'
        with h5py.File(file_path, 'r') as h5f:
            # Access the EMG group
            sample_data = h5f[location + '/' + str(sample) + '/']
            gestureName = sample_data.attrs['gestureName']
            emg_group = h5f[location + '/' + str(sample) + '/emg']

            # set start and end points (except for noGesture --> endpoint = max length in dataset)
            startingPoint = 0
            endingPoint = 0
            if gestureName != 'noGesture':
                ground_truth_index = sample_data["groundTruthIndex"][:]  
                if ground_truth_index is not None:
                    if hasattr(ground_truth_index, "__getitem__"):  # Array-like (e.g., HDF5 dataset)
                        startingPoint = int(ground_truth_index[0])
                        endingPoint = int(ground_truth_index[1])
                    else:
                        raise ValueError("groundTruthIndex must be an array-like object.")        
            else:
                endingPoint = random.randint(self.min_samples, self.max_samples)

            # load channel data for set length
            channels = []
            for channel_name in emg_group.keys():  # Iterate over 'ch1', 'ch2', ...
                channel_data = emg_group[channel_name][startingPoint:endingPoint]
                channels.append(channel_data)
            emg_data = np.array(channels)
        return emg_data  

    def get_windows_rms(self, emg_data):
        # calculate number of windows for data length
        num_samples = emg_data.shape[1]
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
                segment = np.pad(emg_data, ((0, 0), (0, end - num_samples)), mode='constant')
            else:
                segment = emg_data[:, start:end]
            # make sure window has correct length
            segment = segment[:, :self.window_samples]
            # Compute RMS for each channel in the window
            rms_feature = np.sqrt(np.mean(np.square(segment), axis=1))  
            rms_windows.append(rms_feature)
        windows_np = np.array(rms_windows) 

        # pad array of window to max number of windows in dataset to have consistent shapes
        pad_size = self.max_windows - len(windows_np)
        if pad_size > 0:
            padding = np.zeros((pad_size, windows_np.shape[1]))  # Pad with zeros
            windows_np = np.vstack((windows_np, padding))
        
        # return windows tensor of consistent shape and true number of windows (without padding) 
        return torch.tensor(windows_np, dtype=torch.float32), num_windows
    
    def get_raw_data(self, emg_data):
        # pad array of window to max number of windows in dataset to have consistent shapes
        num_samples = emg_data.shape[1]
        pad_size = self.max_samples - num_samples
        if pad_size > 0:
            padding = np.zeros((emg_data.shape[0], pad_size))
            emg_data = np.hstack((emg_data, padding))
        # Ensure emg_data has contiguous memory layout
        emg_data = emg_data.copy()
        # return windows tensor of consistent shape and true number of windows (without padding) 
        return torch.tensor(emg_data, dtype=torch.float32), num_samples

    def __getitem__(self, index):
        # Step 1: Fetch file path, gesture index, label and location
        file_path, gesture_idx, label, location = self.all_gestures[index]
        # Step 2: Load EMG data from h5 file
        emg_data = self.get_emg_data(file_path, gesture_idx, location)
        # Step 3: Create RMS windows and get original length
        match self.window_mode:
            case "rms":
                windows, original_length = self.get_windows_rms(emg_data)
            case "raw":
                windows, original_length = self.get_raw_data(emg_data)
            case _:  # Default case
                windows, original_length = self.get_windows_rms(emg_data)
        return windows, original_length, label

    def __len__(self):
        return len(self.all_gestures)
    
def get_training_set(config, validation=True):
    """
    @param validation: flag wether training set should be split into train/val
    @return: TorchDataset with Training Samples, TorchDataset with Validation Samples (will be None if validation=False)
    """
    meta_csv = dataset_config['training_meta_csv']
    df = pd.read_csv(meta_csv)
    grouped = df.groupby("File_Path")

    if config.tuning_split:
        # 80/20 split for tuning
        n_train_users = 245
        n_val_users = 61
    else:
        n_train_users = config.n_train_users
        n_val_users = config.n_val_users

    if validation:
        original_state = random.getstate()
        random.seed(config.running_seed)

        # Get all unique groups (users) and nsure there are enough users for both splits
        all_groups = list(grouped.groups.keys())
        assert len(all_groups) >= (n_train_users + n_val_users), \
            "Not enough unique users for the specified training and validation splits."

        # Randomly select training users
        selected_train_groups = random.sample(all_groups, n_train_users)
        # Remaining groups after selecting training users
        remaining_groups = [group for group in all_groups if group not in selected_train_groups]
        # Randomly select validation users from the remaining groups
        selected_val_groups = random.sample(remaining_groups, n_val_users)
        # Filter the DataFrame for training users and validation users
        training_df = grouped.filter(lambda x: x.name in selected_train_groups)
        validation_df = grouped.filter(lambda x: x.name in selected_val_groups)

        validation_df = validation_df.groupby("File_Path")
        training_df = training_df.groupby("File_Path")
        ds_training = EMGEPN612Dataset(training_df, config.sample_freq, config.window_length, config.window_overlap, config.max_samples, config.min_samples, config.n_reps, config.window_mode)
        ds_validation = EMGEPN612Dataset(validation_df, config.sample_freq, config.window_length, config.window_overlap, config.max_samples, config.min_samples, 50, config.window_mode)

        random.setstate(original_state)
        return ds_training, ds_validation
    else:
        ds_training = EMGEPN612Dataset(grouped, config.sample_freq, config.window_length, config.window_overlap, config.max_samples, config.min_samples, config.n_reps, config.window_mode)
        return ds_training, None
    
def get_testing_set(config):
    """
    Returns:
    Single TorchDataset with Testing Samples
    """
    meta_csv = dataset_config['testing_meta_csv']
    df = pd.read_csv(meta_csv)
    grouped = df.groupby("File_Path")
    ds_testing = EMGEPN612Dataset(grouped, config.sample_freq, config.window_length, config.window_overlap, config.max_samples, config.min_samples, 25, config.window_mode)
    return ds_testing