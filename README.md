<h1 align="center" id="title">Discrete Hand Gesture Recognition</h1>

<p id="description">EEG to EMG Transfer Learning for Discrete Hand Gesture Recognition using the <a href="https://zenodo.org/records/4421500">EMG-EPN612</a> and <a href="https://physionet.org/content/eegmmidb/1.0.0/">Physionet Motor Movement/Imagery</a> datasets.</p>

<h2>Setup</h2>

<p>Save the <a href="https://zenodo.org/records/4421500">EMG-EPN612</a> and <a href="https://physionet.org/content/eegmmidb/1.0.0/">Physionet Motor Movement/Imagery</a> datasets into the "raw_data" folder. The structure should look like this:</p>

```
raw_data\EMG-EPN612\trainingJSON\
raw_data\EMG-EPN612\testingJSON\
raw_data\eeg-motor-movement\
```

<p>Run the metadata_creator.py file located at</p>

```
dataset\helpers\metadata_creator.py
```

<p>to create a metadata.csv file for the <a href="https://zenodo.org/records/4421500">EMG-EPN612</a> dataset. Two files will be created and located at</p>

```
raw_data\EMG-EPN612\
```

<p>To speed up <a href="https://zenodo.org/records/4421500">EMG-EPN612</a> data loading, the .json files will be converted to .h5 files by running h5_converter.py located at</p>

```
dataset\helpers\h5_converter.py
```

<h2>How to run</h2>

<details>
<summary>EEG Data</summary>

<p>To start an automated train-test loop run one of the .ps1 files, depending on the model you want to use</p>

```
auto_train\auto_train_lstm_eeg.ps1
auto_train\auto_train_cnn_lstm_eeg.ps1
```

<p>The weights of the models will be saved at</p>

```
models\weights\eeg_lstm\
models\weights\eeg_cnn_lstm\
```
</details>

<details>
<summary>EMG Data</summary>

<p>To start an automated train-test loop run one of the .ps1 files, depending on the model you want to use</p>

```
auto_train\auto_train_lstm.ps1
auto_train\auto_train_cnn_lstm.ps1
```
</details>

<details>
<summary>EMG Data with Pretrained Model</summary>

<p>To start an automated train-test loop run one of the .ps1 files, depending on the model you want to use</p>
<p>Additionally you will have to edit the files and add a '--load_weights' flag. Also consider changing the used tags and experiment names</p>

```
auto_train\auto_train_lstm.ps1
auto_train\auto_train_cnn_lstm.ps1
```
</details>

<details>
<summary>Run manually</summary>

<p>To start training and testing loop manually run main.py (If no additional flags are set, the LSTM with EMG data will be used. Flags used for each model can be found in .ps1 files)</p>

```
python -u "main.py"
```
</details>

<h2>Logging</h2>

<p>All results will be logged to your <a href="https://wandb.ai/">wandb</a> account. Use --experiment_name= and --group_name flags to change the name and group (for CV) in wandb.</p>

```
python -u "main.py" --experiment_name=LSTM --group_name=LSTM
```