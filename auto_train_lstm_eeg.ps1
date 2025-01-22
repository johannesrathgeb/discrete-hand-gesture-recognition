# List of possible dropout values
$TRAIN_USERS_LIST = @(76)
$REPS_LIST = @(1)
# Loop through each dropout value
foreach ($TRAIN_USERS in $TRAIN_USERS_LIST) {
    foreach ($REPS in $REPS_LIST) {
        # Generate experiment name and group name dynamically
        $EXPERIMENT_NAME = "lstm_eeg_${TRAIN_USERS}_${REPS}"
        $GROUP_NAME = "lstm_eeg_${TRAIN_USERS}_${REPS}"
        
        Write-Host "Starting script with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME..."
        
        # Call the Python script with the current dropout value and experiment/group name
        python -u "main.py" --n_train_users=$TRAIN_USERS --n_reps=$REPS --experiment_name=$EXPERIMENT_NAME --group_name=$GROUP_NAME --es_start_epoch=30 --in_channels=8 --n_classes=3 --data_type=eeg --batch_size=64 --n_val_users=6 --window_length=0.5
        
        # Check if the Python script exited successfully
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Python script failed with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME. Exiting."
            exit 1
        }
        
        Write-Host "Completed script with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME."
    }
}

Write-Host "All scripts executed successfully."


#python -u "main.py" --n_train_users=76 --n_reps=1 --es_start_epoch=30 --in_channels=8 --n_classes=3 --data_type=eeg --batch_size=64 --n_val_users=6 --window_length=0.5 --running_seed=45 --cv_runs=1 --update_seed=0