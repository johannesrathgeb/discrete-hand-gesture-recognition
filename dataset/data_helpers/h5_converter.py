import json
import h5py
import os
from tqdm import tqdm

def write_h5(group, data):
    """
    Recursively write JSON-like data into an HDF5 group.
    """
    for key, value in data.items():
        if isinstance(value, dict):  # Nested structure, create a subgroup
            subgroup = group.create_group(key)
            write_h5(subgroup, value)
        elif isinstance(value, list):  # Handle lists as datasets
            group.create_dataset(key, data=value)
        elif isinstance(value, (int, float, str)):  # Handle scalar values
            group.attrs[key] = value
        else:
            raise TypeError(f"Unsupported data type for key: {key}, value: {value}")

def convert_json(dataset_dir, data_subfolder):
    user_folder = os.listdir(os.path.join(dataset_dir, data_subfolder))
    files = [f"{folder}/{folder}.json" for folder in user_folder]
    with tqdm(total=len(files), desc="Processing Users", unit="user") as pbar:
        for file in files:
            file_path = os.path.join(dataset_dir, data_subfolder, file)
            h5_file_path = os.path.splitext(file_path)[0] + '.h5'
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            with h5py.File(h5_file_path, 'w') as h5f:
                write_h5(h5f, json_data)
            print(f"Data successfully written to {h5_file_path}")
            pbar.update(1)

if __name__ == '__main__':
    convert_json('raw_data/EMG-EPN612/', 'testingJSON/')
    convert_json('raw_data/EMG-EPN612/', 'trainingJSON/')