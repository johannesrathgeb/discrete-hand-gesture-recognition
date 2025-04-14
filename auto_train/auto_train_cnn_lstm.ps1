# List of users and repetitions per gesture to loop through
$TRAIN_USERS_LIST = @(300, 150, 50, 20, 5, 1)
$REPS_LIST = @(50, 30, 10, 5, 1)
foreach ($TRAIN_USERS in $TRAIN_USERS_LIST) {
    foreach ($REPS in $REPS_LIST) {
        # Generate experiment name and group name dynamically
        $EXPERIMENT_NAME = "cnn_lstm_${TRAIN_USERS}_${REPS}"
        $GROUP_NAME = "cnn_lstm_${TRAIN_USERS}_${REPS}"
        
        Write-Host "Starting script with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME..."

        # Call the Python script
        python -u "main.py" --n_train_users=$TRAIN_USERS --n_reps=$REPS --experiment_name=$EXPERIMENT_NAME --group_name=$GROUP_NAME --window_mode=raw --model=cnn-lstm --lr=1e-3 --optimizer=adamw --scheduler=step --optimzer_step=25 --optimizer_gamma=0.5 --es_start_epoch=30 --n_epochs=30 --gradient_clip=0.5 --tags EMG CNN-LSTM UNTRAINED
        
        # Check if the Python script exited successfully
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Python script failed with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME. Exiting."
            exit 1
        }
        Write-Host "Completed script with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME."
    }
}
Write-Host "All scripts executed successfully."