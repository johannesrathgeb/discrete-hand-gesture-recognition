# List of possible dropout values
$TRAIN_USERS_LIST = @(300, 150, 50)
$REPS_LIST = @(50, 30, 10)
# $TRAIN_USERS_LIST = @(300, 150)
# $REPS_LIST = @(50, 30)
# min 75 epochs
# Loop through each dropout value
foreach ($TRAIN_USERS in $TRAIN_USERS_LIST) {
    foreach ($REPS in $REPS_LIST) {
        # Generate experiment name and group name dynamically
        $EXPERIMENT_NAME = "fused_cnn_${TRAIN_USERS}_${REPS}"
        $GROUP_NAME = "fused_cnn_${TRAIN_USERS}_${REPS}"
        
        Write-Host "Starting script with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME..."
        
        # Call the Python script with the current dropout value and experiment/group name
        python -u "main.py" --n_train_users=$TRAIN_USERS --n_reps=$REPS --experiment_name=$EXPERIMENT_NAME --group_name=$GROUP_NAME --window_mode=normalized --model=cnn --window_length=0.385 --window_overlap=0.36 --dropout=0.2 --lr=5e-3 --optimizer=adamw --scheduler=none --n_epochs=75 --es_patience=1000 --batch_size=1024 --num_workers=10
        
        # Check if the Python script exited successfully
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Python script failed with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME. Exiting."
            exit 1
        }
        
        Write-Host "Completed script with --n_train_users=$TRAIN_USERS, --n_reps=$REPS, --experiment_name=$EXPERIMENT_NAME, and --group_name=$GROUP_NAME."
    }
}

Write-Host "All scripts executed successfully."
