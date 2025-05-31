import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from glob import glob
import pyedflib

from helpers.utils import preprocess_data

DATASET_DIR = 'raw_data/eeg-motor-movement/'
assert DATASET_DIR is not None, "Specify 'EEG Motor Imaginary Movement dataset' location in variable " \
                                "'DATASET_DIR'. The dataset can be downloaded from this URL:" \
                                " https://zenodo.org/records/4421500"

dataset_config = {
    "meta_csv": os.path.join(DATASET_DIR, "eeg-motor-movement-metadata.csv"),
}

# This function is adapted from the EEGMotorImagery repository by rootskar:
# https://github.com/rootskar/EEGMotorImagery/tree/master (March 2025)
def load_data(nr_of_subj=109, chunk_data=True, chunks=8, base_folder=DATASET_DIR, sample_rate=160,
              samples=640, cpu_format=False, preprocessing=False, hp_freq=0.5, bp_low=2, bp_high=60, notch=False,
              hp_filter=False, bp_filter=False, artifact_removal=False):
    # Get file paths
    PATH = base_folder
    SUBS = glob(PATH + 'S[0-9]*')
    FNAMES = sorted([x[-4:] for x in SUBS])
    FNAMES = FNAMES[:nr_of_subj]

    # Remove the subjects with incorrectly annotated data that will be omitted from the final dataset
    subjects = ['S038', 'S088', 'S089', 'S092', 'S100', 'S104']
    try:
        for sub in subjects:
            FNAMES.remove(sub)
    except:
        pass

    """
    @input - label (String)
            
    Helper method that converts trial labels into integer representations

    @output - data (Numpy array); target labels (Numpy array)
    """

    def convert_label_to_int(str):
        if str == 'T1':
            return 0
        if str == 'T2':
            return 1
        raise Exception("Invalid label %s" % str)

    """
    @input - data (array); number of chunks to divide the list into (int)
            
    Helper method that divides the input list into a given number of arrays

    @output - 2D array of divided input data
    """

    def divide_chunks(data, chunks):
        for i in range(0, len(data), chunks):
            yield data[i:i + chunks]

    executed_trials = '03,07,11'.split(',')
    imagined_trials = '04,08,12'.split(',')
    both_trials = executed_trials + imagined_trials
    file_numbers = imagined_trials

    samples_per_chunk = int(samples / chunks)
    X = []
    y = []

    # Iterate over different subjects
    for subj in FNAMES:
        # Load the file names for given subject
        fnames = glob(os.path.join(PATH, subj, subj + 'R*.edf'))
        fnames = [name for name in fnames if name[-6:-4] in file_numbers]
        # Iterate over the trials for each subject
        for file_name in fnames:
            # Load the file
            loaded_file = pyedflib.EdfReader(file_name)
            annotations = loaded_file.readAnnotations()
            times = annotations[0]
            durations = annotations[1]
            tasks = annotations[2]

            # Load the data signals into a buffer
            signals = loaded_file.signals_in_file
            # signal_labels = loaded_file.getSignalLabels()
            sigbufs = np.zeros((signals, loaded_file.getNSamples()[0]))
            for i in np.arange(signals):
                sigbufs[i, :] = loaded_file.readSignal(i)

            # initialize the result arrays with preferred shapes
            if chunk_data:
                trial_data = np.zeros((15, 64, chunks, samples_per_chunk))
            else:
                trial_data = np.zeros((15, 64, samples))
            labels = []
            signal_start = 0
            k = 0
            # Iterate over tasks in the trial run
            for i in range(len(times)):
                # Collects only the 15 non-rest tasks in each run
                if k == 15:
                    break
                current_duration = durations[i]
                signal_end = signal_start + samples
                # Skipping tasks where the user was resting
                if tasks[i] == 'T0':
                    signal_start += int(sample_rate * current_duration)
                    continue

                # Iterate over each channel
                for j in range(len(sigbufs)):
                    channel_data = sigbufs[j][signal_start:signal_end]
                    if preprocessing:
                        channel_data = preprocess_data(channel_data, sample_rate=sample_rate, ac_freq=60,
                                                       hp_freq=hp_freq, bp_low=bp_low, bp_high=bp_high, notch=notch,
                                                       hp_filter=hp_filter, bp_filter=bp_filter,
                                                       artifact_removal=artifact_removal)
                    if chunk_data:
                        channel_data = list(divide_chunks(channel_data, samples_per_chunk))
                    # Add data for the current channel and task to the result
                    trial_data[k][j] = channel_data

                # add label(s) for the current task to the result
                if chunk_data:
                    # multiply the labels by the chunk size for chunked mode
                    labels.extend([convert_label_to_int(tasks[i])] * chunks)
                else:
                    labels.append(convert_label_to_int(tasks[i]))

                signal_start += int(sample_rate * current_duration)
                k += 1

            # Add labels and data for the current run into the final output numpy arrays
            y.extend(labels)
            if cpu_format:
                if chunk_data:
                    # (15, 64, 8, 80) => (15, 64, 80, 8) => (15, 8, 80, 64) => (120, 80, 64)
                    X.extend(trial_data.swapaxes(2, 3).swapaxes(1, 3).reshape((-1, samples_per_chunk, 64)))
                else:
                    # (15, 64, 640) => (15, 640, 64)
                    X.extend(trial_data.swapaxes(1, 2))
            else:
                if chunk_data:
                    # (15, 64, 8, 80) => (15, 8, 64, 80) => (120, 64, 80)
                    X.extend(trial_data.swapaxes(1, 2).reshape((-1, 64, samples_per_chunk)))
                else:
                    # (15, 64, 640)
                    X.extend(trial_data)

    # Shape the final output arrays to the correct format
    X = np.stack(X)
    y = np.array(y).reshape((-1, 1))
    return X, y

class EEGMotorMovementDataset(TorchDataset):
    def __init__(self, X, y, rms_feature=False):
        self.X = X
        self.y = y
        self.rms_feature = rms_feature
        self.window_samples = 4
        self.num_windows = 160
        self.fs = 160
        
    def get_windows_rms(self, eeg_data):   
        rms_windows = []    
        for i in range(self.num_windows):
            # calculate start and end point for window
            start = i * self.window_samples
            end = start + self.window_samples          
            segment = eeg_data[:, start:end]
            # Compute RMS for each channel in the window
            segment = segment
            rms_feature = np.sqrt(np.mean(np.square(segment.cpu().numpy()), axis=1))  
            rms_windows.append(rms_feature)
        windows_np = np.array(rms_windows) 
        # return windows tensor of consistent shape and true number of windows (without padding) 
        return torch.tensor(windows_np, dtype=torch.float32), self.num_windows
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_data = torch.tensor(self.X[idx], dtype=torch.float32)
        # Make sure y is a single integer (no extra dim)
        y_data = torch.tensor(self.y[idx], dtype=torch.long).squeeze()

        if self.rms_feature:
            selected_channels = [43, 59, 54, 20, 45, 53, 58, 52] # rms 0.025 ['T10', 'Po8', 'P8', 'Cp6', 'Tp8', 'P6', 'Po4', 'P4']
        else:
            selected_channels = [3, 10, 4, 11, 33, 34, 18, 2] # default ['Fcz', 'Cz', 'Fc2', 'C2', 'Fz', 'F2', 'Cp2', 'Fc1']
        
        if selected_channels:
                x_data = x_data[selected_channels, :]

        if self.rms_feature:
            x_data, length = self.get_windows_rms(x_data)
        else:
            length = x_data.shape[1]
            
        return x_data, length, y_data

def get_train_val_test_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42, rms_feature=False):
    """
    Split arrays (X and y) into random train, val, test subsets.

    Args:
        X: numpy array or list-like of shape (N, ...)
        y: numpy array or list-like of shape (N,)
        train_ratio: float between (0, 1)
        val_ratio: float between (0, 1)
        test_ratio: float between (0, 1)
        random_seed: int or None, for reproducible output across multiple calls
        rms_feature: bool, if True, use RMS feature extraction

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-8, \
        "train_ratio + val_ratio + test_ratio must equal 1."
    
    X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=(1.0 - train_ratio),
            random_state=random_seed,
            stratify=y
        )
    
    val_portion = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1.0 - val_portion),
            random_state=random_seed,
            stratify=y_temp
        )
        
    train_dataset = EEGMotorMovementDataset(X_train, y_train, rms_feature)
    val_dataset   = EEGMotorMovementDataset(X_val,   y_val, rms_feature)
    test_dataset  = EEGMotorMovementDataset(X_test,  y_test, rms_feature)
    return train_dataset, val_dataset, test_dataset