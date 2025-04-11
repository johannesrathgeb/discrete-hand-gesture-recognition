# Generate experiment name and group name dynamically
$EXPERIMENT_NAME = "lstm_eeg"
$GROUP_NAME = "lstm_eeg"

Write-Host "Starting script with --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME..."

# Call the Python script with the current dropout value and experiment/group name
python -u "main.py" --experiment_name=$EXPERIMENT_NAME --group_name=$GROUP_NAME --es_start_epoch=2000 --in_channels=8 --n_classes=2 --data_type=eeg --batch_size=64 --save_weights --tags EEG LSTM UNTRAINED

# Check if the Python script exited successfully
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python script failed with --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME. Exiting."
    exit 1
}

Write-Host "Completed script with --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME."

Write-Host "All scripts executed successfully."