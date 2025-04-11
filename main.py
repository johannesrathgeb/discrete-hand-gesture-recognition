import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import gc
import wandb

from dataset.emg_epn612 import get_training_set, get_testing_set
from dataset.eeg_motor_movement import get_train_val_test_split, load_data as load_data_eeg
from helpers.init import worker_init_fn
from models.baseline import get_model

class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse, contains all configurations for our experiment

        # the baseline model
        self.model = get_model(config)        
        if config.data_type == "eeg":
            self.label_ids = ['left', 'right']
        else:
            self.label_ids = ['fist', 'noGesture', 'open', 'pinch', 'waveIn', 'waveOut']

        # pl 2 containers:
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x, lengths):
        """
        @param x: batch of gesture rms-windows
        @param lengths: batch of original number of windows of gesture (without padding)
        @return: final model predictions (logits)
        """
        x = self.model(x, lengths)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """
        if self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr) 
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr) 
        
        if self.config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.optimzer_step, gamma=self.config.optimizer_gamma)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, lengths, labels = train_batch
        y_hat = self.model(x, lengths)

        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss.detach().cpu())
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, val_batch, batch_idx):
        x, lengths, labels = val_batch
        y_hat = self.forward(x, lengths)

        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()
        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()
        n_pred = torch.as_tensor(len(labels), device=self.device)
        accuracy = n_correct.float() / n_pred

        results = {
            'loss': loss,
            'accuracy': accuracy
        }
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        avg_acc = outputs['accuracy'].mean()
        logs = {'acc': avg_acc, 'loss': avg_loss}
        
        # prefix with 'val' for logging
        self.log_dict({"val/" + k: logs[k] for k in logs})
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        x, lengths, labels = test_batch
        y_hat = self.model(x, lengths)

        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()
        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()
        n_pred = torch.as_tensor(len(labels), device=self.device)
        accuracy = n_correct.float() / n_pred

        results = {
            'loss': loss,
            'true_labels': labels,
            'predictions': preds,
            'accuracy': accuracy
        }
        self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            if k in ['true_labels', 'predictions']:  # These have varying sizes
                outputs[k] = torch.cat(outputs[k], dim=0)
            else:  # Scalars (like accuracy, loss) can be stacked
                outputs[k] = torch.stack(outputs[k])
                
        avg_loss = outputs['loss'].mean()
        avg_acc = outputs['accuracy'].mean()

        # Generate confusion matrix
        all_labels = torch.cat([x['true_labels'] for x in self.test_step_outputs]).cpu().numpy()
        all_preds = torch.cat([x['predictions'] for x in self.test_step_outputs]).cpu().numpy()
        cm = wandb.plot.confusion_matrix(
            y_true=all_labels, preds=all_preds, class_names=self.label_ids
        )

        self.logger.experiment.log({"confusion_matrix": cm})
        logs = {'acc': avg_acc, 'loss': avg_loss}
        # prefix with 'test' for logging
        self.log_dict({"test/" + k: logs[k] for k in logs})
        self.test_step_outputs.clear()

    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        pass
        x, lengths, files = eval_batch

        # assure fp16
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x, lengths)

        return files, y_hat

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, monitor="val/acc", min_delta=0.01, patience=5, verbose=True, mode="max", start_epoch=30):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        # Only apply early stopping after start_epoch
        if trainer.current_epoch < self.start_epoch:
            return
        # Call the original on_validation_end to handle early stopping
        super().on_train_epoch_end(trainer, pl_module)

def save_model_weights(pl_module, path):
    torch.save(pl_module.model.state_dict(), path)
    print(f"Model weights saved to {path}")

def load_model_weights(pl_module, path, skip_keys=[]):
    """
    Load model weights while skipping specific keys (e.g., output layer).
    :param pl_module: The Lightning module containing the model.
    :param path: Path to the saved weights file.
    :param skip_keys: List of keys to skip when loading weights.
    """
    state_dict = torch.load(path)
    model_state_dict = pl_module.model.state_dict()
    # Filter out the keys to skip
    filtered_state_dict = {k: v for k, v in state_dict.items() if k not in skip_keys}
    # Load the filtered state_dict
    model_state_dict.update(filtered_state_dict)
    pl_module.model.load_state_dict(model_state_dict)
    print(f"Model weights loaded from {path}, except keys: {skip_keys}")

def train(config, save_path=None, load_path=None, X_eeg=None, y_eeg=None):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        tags=config.tags,
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )

    if config.data_type == "eeg":
        ds_train, ds_val, ds_test = get_train_val_test_split(X=X_eeg, y=y_eeg, rms_feature=args.window_mode=="rms", random_seed=config.running_seed)
        test_dl = DataLoader(dataset=ds_test,
                         worker_init_fn=worker_init_fn,
                         num_workers=4,
                         batch_size=config.batch_size,
                         persistent_workers=True)
    else:
        ds_train, ds_val = get_training_set(config, validation=True)
        test_dl = DataLoader(dataset=get_testing_set(config),
                        worker_init_fn=worker_init_fn,
                        num_workers=4,
                        batch_size=config.batch_size,
                        persistent_workers=True)

    train_dl = DataLoader(dataset=ds_train,
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          persistent_workers=True,
                          shuffle=True)
    val_dl = DataLoader(dataset=ds_val,
                          worker_init_fn=worker_init_fn,
                          num_workers=2,
                          batch_size=config.batch_size,
                          persistent_workers=True)
    
    # create pytorch lightening module
    pl_module = PLModule(config)
    if load_path:
        # load model weights, except for the output layer
        if config.model == "lstm":
            output_layer_keys = ["fc_out.weight", "fc_out.bias"]
        else:
            output_layer_keys = ["fc.weight", "fc.bias"]
        load_model_weights(pl_module, load_path, output_layer_keys)

    #specify early stopping strategy
    early_stop_callback = CustomEarlyStopping(monitor="val/acc", min_delta=config.es_delta, patience=config.es_patience, verbose=True, mode="max", start_epoch=config.es_start_epoch)

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger, on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision,
                         callbacks=[early_stop_callback, pl.callbacks.ModelCheckpoint(save_last=True)],
                         gradient_clip_val=config.gradient_clip)
    
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, val_dl)
    if save_path:
        # save model weights after training
        save_model_weights(pl_module, save_path + str(config.running_seed) + ".pth")
    del train_dl, val_dl

    # final test step
    trainer.test(ckpt_path='last', dataloaders=test_dl)
    wandb.finish()
    del trainer
    del pl_module
    del test_dl
    gc.collect()
    torch.cuda.empty_cache()

def evaluate_args(parser):
    # general
    parser.add_argument('--project_name', type=str, default="DISCRETE-HAND-GESTURE-RECOGNITION")
    parser.add_argument('--experiment_name', type=str, default="Baseline")
    parser.add_argument('--group_name', type=str, default="Baseline")
    parser.add_argument('--tags', type=str, nargs='+', default=["EMG"])
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for training dataloader
    parser.add_argument('--seed', type=int, default=42) # set base seed
    parser.add_argument('--running_seed', type=int, default=42) # set running seed (incremented for each run)
    parser.add_argument('--update_seed', type=int, default=1) # set to 1 if you want to update the seed for each run
    parser.add_argument('--precision', type=str, default="32") # 16 or 32 bit precision

    # window creation
    parser.add_argument('--data_type', type=str, default="emg") # emg or eeg
    parser.add_argument('--max_samples', type=int, default=599) # maximum number of samples per item
    parser.add_argument('--min_samples', type=int, default=76) # minimum number of samples per item
    parser.add_argument('--sample_freq', type=int, default=200) # sample frequency of the data (200Hz for EMG, 160Hz for EEG)
    parser.add_argument('--window_length', type=float, default=0.025) # window length in seconds
    parser.add_argument('--window_overlap', type=float, default=0.0) # window overlap in seconds
    parser.add_argument('--window_mode', type=str, default="rms") # rms (windows of rms feature) or raw (raw data structure)

    # optimizers
    parser.add_argument('--optimizer', type=str, default="adam") # adam or adamw
    parser.add_argument('--scheduler', type=str, default="step") # step
    parser.add_argument('--lr', type=float, default=1e-4) # learning rate
    parser.add_argument('--optimzer_step', type=int, default=5) # step size for learning rate scheduler
    parser.add_argument('--optimizer_gamma', type=float, default=0.9) # gamma for learning rate scheduler
    parser.add_argument('--gradient_clip', type=float, default=0.0) # gradient clipping value

    # model
    parser.add_argument('--model', type=str, default="lstm") # lstm or cnn-lstm
    parser.add_argument('--n_classes', type=int, default=6)  # classification model with 'n_classes' output neurons
    parser.add_argument('--in_channels', type=int, default=8) # number of input channels
    parser.add_argument('--hidden_size', type=int, default=128) # hidden size of the lstm layers
    parser.add_argument('--dropout', type=float, default=0.3) # dropout value
    parser.add_argument('--num_layers', type=int, default=3) # number of lstm layers

    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default=None)  # for loading trained model, corresponds to wandb id

    # training
    parser.add_argument('--cv_runs', type=int, default=10) # number of cross-validation runs
    parser.add_argument('--n_epochs', type=int, default=300) # number of max epochs to train
    parser.add_argument('--batch_size', type=int, default=1000) # batch size for training
    parser.add_argument('--es_delta', type=float, default=0.001) # minimum change in monitored quantity to qualify as an improvement
    parser.add_argument('--es_patience', type=int, default=5) # number of epochs with no improvement after which training will be stopped
    parser.add_argument('--es_start_epoch', type=int, default=30) # epoch from which early stopping is applied
    parser.add_argument('--n_val_users', type=int, default=6) # number of users for validation
    parser.add_argument('--n_train_users', type=int, default=300) # number of users for training
    parser.add_argument('--n_reps', type=int, default=50) # number of repetitions for each user

    # pretraining
    parser.add_argument('--save_weights', action='store_true') # save model weights
    parser.add_argument('--load_weights', action='store_true') # load model weights

    return parser.parse_args()

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description='argument parser')
    args = evaluate_args(parser)

    # set general variables
    base_experiment_name = args.experiment_name
    EEG_LSTM_PATH = "models/weights/eeg_lstm/lstm_"
    EEG_CNN_LSTM_PATH = "models/weights/eeg_cnn_lstm/cnn_lstm_"
    save_path = None
    load_path = None
    if args.save_weights:
        if args.model == "lstm":
            save_path = EEG_LSTM_PATH
        else:
            save_path = EEG_CNN_LSTM_PATH
    if args.load_weights:
        if args.model == "lstm":
            load_path = EEG_LSTM_PATH + "47" + ".pth"
        else:
            load_path = EEG_CNN_LSTM_PATH + "42" + ".pth"

    # load eeg data into memory here to avoid loading it for each run
    if args.data_type == "eeg":
        X, y = load_data_eeg(nr_of_subj=109, chunk_data=not(args.window_mode=="rms"), chunks=8, cpu_format=False,
                preprocessing=not(args.window_mode=="rms"), hp_freq=0.5, bp_low=2, bp_high=60, notch=True,
                hp_filter=False, bp_filter=True, artifact_removal=True)

    # run cv for the specified number of runs and set the seed for each run        
    for run_id in range(args.cv_runs):
        seed_everything(args.seed)
        if args.update_seed >= 1:
            args.running_seed = args.seed + run_id
        args.experiment_name = f"{base_experiment_name}_{args.running_seed}"
        
        if args.data_type == "eeg":
            train(args, save_path=save_path, load_path=load_path, X_eeg=X, y_eeg=y)
        else:
            train(args, save_path=save_path, load_path=load_path, X_eeg=None, y_eeg=None)