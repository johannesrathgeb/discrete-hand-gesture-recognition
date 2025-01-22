import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import mne

def csv_from_np(labels, locations, gestures, save_location):
    rows = []
    label_idx = 0  
    for gesture in gestures:
        file_path, gesture_indices = gesture
        for idx in gesture_indices:
            rows.append({
                "Label": labels[label_idx],
                "Location": locations[label_idx],
                "File_Path": file_path,
                "Gesture_Index": idx
            })
            label_idx += 1
    df = pd.DataFrame(rows)
    df.to_csv(save_location, index=False, encoding="utf-8")
    print(f"Data saved to {save_location}")

def emgepn612(dataset_dir, data_subfolder, save_location, use_testing):
    user_folder = os.listdir(os.path.join(dataset_dir, data_subfolder))
    files = [f"{folder}/{folder}.json" for folder in user_folder]
    labels = []
    locations = []
    gestures = []
    with tqdm(total=len(files), desc="Processing Users", unit="user") as pbar:
        for file in files:
            gesture_idx = []
            file_path = os.path.join(dataset_dir, data_subfolder, file)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                            user_data = json.load(f)
                            if "trainingSamples" in user_data:
                                for sample_key, sample_data in user_data["trainingSamples"].items():
                                    labels.append(sample_data["gestureName"])
                                    locations.append("trainingSamples")
                                    gesture_idx.append(sample_key)
                            if "testingSamples" in user_data and use_testing == True:
                                for sample_key, sample_data in user_data["testingSamples"].items():
                                    labels.append(sample_data["gestureName"])
                                    locations.append("testingSamples")
                                    gesture_idx.append(sample_key)
            gestures.append((file_path, np.array(gesture_idx, dtype=object)))
            pbar.update(1)
    locations = np.array(locations)
    labels = np.array(labels)
    gestures = np.array(gestures, dtype=object)
    csv_from_np(labels, locations, gestures, save_location)

def csv_from_np_eeg(start_idx, end_idx, labels, file_paths, save_location):
    rows = []
    label_idx = 0  
    for file_path in file_paths:
        rows.append({
            "Label": labels[label_idx],
            "File_Path": file_path,
            "Start_Index": start_idx[label_idx],
            "End_Index": end_idx[label_idx]
        })
        label_idx += 1
    df = pd.DataFrame(rows)
    df.to_csv(save_location, index=False, encoding="utf-8")
    print(f"Data saved to {save_location}")

def eegmotormovement(dataset_dir, save_location):
    user_folder = os.listdir(dataset_dir)
    excluded_folders = {'S038', 'S088', 'S089', 'S092', 'S100', 'S104'}
    user_folder = [folder for folder in user_folder if folder not in excluded_folders]
    #files = [f"{folder}/{folder}.edf" for folder in user_folder]
    files = [f"{folder}/{folder}R{str(i).zfill(2)}.edf" 
         for folder in user_folder 
         for i in range(3, 15)]
    print(user_folder)
    #print(files)

    # Runs corresponding to left fist (T1) and right fist (T2)
    all_runs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    fist_runs = [3, 4, 7, 8, 11, 12]
    both_runs = [5, 6, 9, 10, 13, 14]
    imagined_fist_runs = [4, 8, 12]
    real_fist_runs = [3, 7, 11]
    run_filter = imagined_fist_runs

    # Filter the files based on the specified runs
    files = [file for file in files if int(file.split('R')[-1][:2]) in run_filter]
    #print(fist_files)

    labels = []
    start_idx_list = []
    end_idx_list = []
    file_paths = []
    s160 = 0
    sother = 0
    with tqdm(total=len(files), desc="Processing Users", unit="user") as pbar:
        for file in files:
            b_count = 0
            l_count = 0
            r_count = 0
            lr_count = 0
            f_count = 0

            file_path = os.path.join(dataset_dir, file)
            if os.path.isfile(file_path):
                raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
                annotations = raw.annotations
                sfreq = raw.info['sfreq']
                if sfreq == 160:
                    for desc, onset, duration in zip(annotations.description, annotations.onset, annotations.duration):
                        run = file.split('R')[-1][:2]
                        if desc == "T0":
                            b_count = b_count + 1
                            label = "B"
                        elif desc == "T1" and int(run) in fist_runs:
                            l_count = l_count + 1
                            label = "L"
                        elif desc == "T1" and int(run) in both_runs:
                            lr_count = lr_count + 1
                            label = "LR"
                        elif desc == "T2" and int(run) in fist_runs:
                            r_count = r_count + 1
                            label = "R"
                        elif desc == "T2" and int(run) in both_runs:
                            f_count = f_count + 1
                            label = "F"
                        
                        if (label == "B" and b_count < 8) or \
                        (label == "L" and l_count < 8) or \
                        (label == "R" and r_count < 8):
                            start_idx = int(onset * sfreq)
                            end_idx = int((onset + duration) * sfreq)
                            labels.append(label)
                            start_idx_list.append(start_idx)
                            end_idx_list.append(end_idx)
                            file_paths.append(file_path)   
                            print("B", b_count, "L", l_count, "R", r_count, "LR", lr_count, "F", f_count)   
                                  
            pbar.update(1)
    start_idx_list = np.array(start_idx_list)
    end_idx_list = np.array(end_idx_list)
    labels = np.array(labels)
    file_paths = np.array(file_paths)
    csv_from_np_eeg(start_idx_list, end_idx_list, labels, file_paths, save_location)

def get_max_length(dataset_dir, data_subfolder):
    user_folder = os.listdir(os.path.join(dataset_dir, data_subfolder))
    files = [f"{folder}/{folder}.json" for folder in user_folder]
    max_length = 0
    min_length = 10000
    with tqdm(total=len(files), desc="Processing Users", unit="user") as pbar:
        for file in files:
            
            file_path = os.path.join(dataset_dir, data_subfolder, file)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                            user_data = json.load(f)
                            if "trainingSamples" in user_data:
                                for sample_key, sample_data in user_data["trainingSamples"].items():
                                    if "gestureName" in sample_data:
                                        startingPoint = 0
                                        endingPoint = 0
                                        if sample_data["gestureName"] != "noGesture":
                                            startingPoint = sample_data["groundTruthIndex"][0]
                                            endingPoint = sample_data["groundTruthIndex"][1]
                                            #print(endingPoint)
                                        else:
                                             endingPoint = len(sample_data["emg"]["ch1"])
                                             #print(endingPoint)
                                        emg_data = np.array([
                                            sample_data["emg"]["ch1"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch2"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch3"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch4"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch5"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch6"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch7"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch8"][startingPoint:endingPoint]
                                        ])
                                        if sample_data["gestureName"] != "noGesture":
                                            sample_length = len(emg_data[0])
                                            max_length = max(max_length, sample_length)  
                                            min_length = min(min_length, sample_length)
                            if "testingSamples" in user_data:
                                for sample_key, sample_data in user_data["testingSamples"].items():
                                    if "gestureName" in sample_data:
                                        startingPoint = 0
                                        endingPoint = 0
                                        if sample_data["gestureName"] != "noGesture":
                                            startingPoint = sample_data["groundTruthIndex"][0]
                                            endingPoint = sample_data["groundTruthIndex"][1]
                                            #print(endingPoint)
                                        else:
                                             endingPoint = len(sample_data["emg"]["ch1"])
                                             #print(endingPoint)
                                        emg_data = np.array([
                                            sample_data["emg"]["ch1"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch2"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch3"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch4"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch5"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch6"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch7"][startingPoint:endingPoint],
                                            sample_data["emg"]["ch8"][startingPoint:endingPoint]
                                        ])
                                        if sample_data["gestureName"] != "noGesture":
                                            sample_length = len(emg_data[0])
                                            max_length = max(max_length, sample_length) 
                                            min_length = min(min_length, sample_length)                                 
            pbar.update(1)
    print(max_length)
    print(min_length)

from collections import Counter
def get_max_length_eeg(dataset_dir):
    user_folder = os.listdir(dataset_dir)
    user_folder = os.listdir(dataset_dir)
    excluded_folders = {'S038', 'S088', 'S089', 'S092', 'S100', 'S104'}
    user_folder = [folder for folder in user_folder if folder not in excluded_folders]
    # files = [f"{folder}/{folder}.edf" for folder in user_folder]
    files = [f"{folder}/{folder}R{str(i).zfill(2)}.edf" 
             for folder in user_folder 
             for i in range(2, 15)]
    all_runs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    fist_runs = [3, 4, 7, 8, 11, 12]
    both_runs = [5, 6, 9, 10, 13, 14]
    imagined_fist_runs = [4, 8, 12]
    real_fist_runs = [3, 7, 11]
    run_filter = imagined_fist_runs

    # Filter the files based on the specified runs
    files = [file for file in files if int(file.split('R')[-1][:2]) in run_filter]

    max_length = 0
    min_length = float('inf')
    lengths = []

    with tqdm(total=len(files), desc="Processing Users", unit="user") as pbar:
        for file in files:
            file_path = os.path.join(dataset_dir, file)
            if os.path.isfile(file_path):
                raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
                annotations = raw.annotations
                sfreq = raw.info['sfreq']
                if sfreq == 160:
                    for desc, onset, duration in zip(annotations.description, annotations.onset, annotations.duration):
                        if desc != "T0":
                            start_idx = int(onset * sfreq)
                            end_idx = int((onset + duration) * sfreq)
                            length = end_idx - start_idx 
                            # if length < 640:
                            #     print("below 640", file, desc)
                            lengths.append(length)  # Collect lengths
                            max_length = max(max_length, length)
                            min_length = min(min_length, length)
            pbar.update(1)

    # Print the maximum and minimum length
    print("Maximum Length:", max_length)
    print("Minimum Length:", min_length)

    # Print the list of all lengths
    print("All Lengths:", lengths)

    # Print the unique counts of lengths
    length_counts = Counter(lengths)
    print("Unique Length Counts:", length_counts)


if __name__ == '__main__':
    #emgepn612('raw_data/EMG-EPN612/', 'testingJSON/', 'emg-epn612-testing-metadata.csv', False)
    #emgepn612('raw_data/EMG-EPN612/', 'trainingJSON/', 'emg-epn612-training-metadata.csv', False)
    eegmotormovement('raw_data/eeg-motor-movement/', 'eeg-motor-movement-metadata.csv')
    #get_max_length('raw_data/EMG-EPN612/', 'testingJSON/') # Training --> 599 76 (1052), Testing --> 599 (1123)
    get_max_length_eeg('raw_data/eeg-motor-movement/')