$TRAIN_USERS_LIST = @(50, 10, 5, 1)
# Loop through each dropout value
foreach ($TRAIN_USERS in $TRAIN_USERS_LIST) {
        # Generate experiment name and group name dynamically
        $EXPERIMENT_NAME = "cnn_lstm_eeg_${TRAIN_USERS}"
        $GROUP_NAME = "cnn_lstm_eeg_${TRAIN_USERS}"
        
        Write-Host "Starting script with --n_train_users=$TRAIN_USERS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME..."
        
        # Call the Python script with the current dropout value and experiment/group name
        python -u "main.py" --n_train_users=$TRAIN_USERS --experiment_name=$EXPERIMENT_NAME --group_name=$GROUP_NAME --window_mode=raw --model=cnn-lstm --lr=1e-3 --optimizer=adamw --scheduler=step --optimzer_step=25 --optimizer_gamma=0.5 --es_start_epoch=30 --gradient_clip=0.5 --in_channels=8 --n_classes=2 --data_type=eeg --batch_size=64 --tags EEG CNN-LSTM UNTRAINED --cv_runs=1
        
        # Check if the Python script exited successfully
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Python script failed with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME. Exiting."
            exit 1
        }
        
        Write-Host "Completed script with --n_train_users=$TRAIN_USERS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME."
    }

Write-Host "All scripts executed successfully."

# Loop through each dropout value
foreach ($TRAIN_USERS in $TRAIN_USERS_LIST) {
    # Generate experiment name and group name dynamically
    $EXPERIMENT_NAME = "lstm_eeg_${TRAIN_USERS}"
    $GROUP_NAME = "lstm_eeg_${TRAIN_USERS}"
    
    Write-Host "Starting script with --n_train_users=$TRAIN_USERS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME..."
    
    # Call the Python script with the current dropout value and experiment/group name
    python -u "main.py" --n_train_users=$TRAIN_USERS --experiment_name=$EXPERIMENT_NAME --group_name=$GROUP_NAME --es_start_epoch=2000 --in_channels=8 --n_classes=2 --data_type=eeg --batch_size=64 --tags EEG LSTM UNTRAINED --cv_runs=1
    
    # Check if the Python script exited successfully
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python script failed with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME. Exiting."
        exit 1
    }
    
    Write-Host "Completed script with --n_train_users=$TRAIN_USERS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME."
}

Write-Host "All scripts executed successfully."