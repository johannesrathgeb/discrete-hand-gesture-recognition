# Experiment and group name used for logging
$EXPERIMENT_NAME = "cnn_lstm_eeg"
$GROUP_NAME = "cnn_lstm_eeg"

Write-Host "Starting script with --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME..."

# Call the Python script with the current dropout value and experiment/group name
python -u "main.py" --experiment_name=$EXPERIMENT_NAME --group_name=$GROUP_NAME --window_mode=raw --model=cnn-lstm --lr=1e-3 --optimizer=adamw --scheduler=step --optimzer_step=25 --optimizer_gamma=0.5 --es_start_epoch=30 --gradient_clip=0.5 --in_channels=8 --n_classes=2 --data_type=eeg --batch_size=64 --tags EEG CNN-LSTM UNTRAINED --save_weights

# Check if the Python script exited successfully
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python script failed with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME. Exiting."
    exit 1
}
Write-Host "Completed script with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME."