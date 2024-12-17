import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

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

if __name__ == '__main__':
    emgepn612('raw_data/EMG-EPN612/', 'testingJSON/', 'emg-epn612-testing-metadata.csv', False)
    emgepn612('raw_data/EMG-EPN612/', 'trainingJSON/', 'emg-epn612-training-metadata.csv', False)
    #get_max_length('raw_data/EMG-EPN612/', 'testingJSON/') # Training --> 599 76 (1052), Testing --> 599 (1123)