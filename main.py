import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import transformers
import wandb

from dataset.emg_epn612 import get_training_set, get_testing_set
from helpers.init import worker_init_fn
from models.baseline import get_model
from helpers import nessi

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning import seed_everything

class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse, contains all configurations for our experiment

        # the baseline model
        self.model = get_model( n_classes=config.n_classes,
                                in_channels=config.in_channels,
                                hidden_size=config.hidden_size,
                                num_layers=config.num_layers,
                                dropout=config.dropout
                               )

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.optimzer_step, gamma=self.config.optimizer_gamma)
        return [optimizer], [scheduler]

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

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()
        n_pred = torch.as_tensor(len(labels), device=self.device)
        accuracy = n_correct.float() / n_pred

        results = {
            'loss': samples_loss.mean(),
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

        # TODO: use 16bit if necessary
        # maximum memory allowance for parameters: 128 KB
        # baseline has 61148 parameters -> we can afford 16-bit precision
        # since 61148 * 16 bit ~ 122 kB

        # assure fp16
        #self.model.half()
        #x = self.mel_forward(x)
        #x = x.half()
        y_hat = self.model(x, lengths)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()
        n_pred = torch.as_tensor(len(labels), device=self.device)
        accuracy = n_correct.float() / n_pred

        results = {
            'loss': samples_loss.mean(),
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

def train(config):
    # logging is done using wandb
    #TODO: change notes and tags
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'24 Task 1.",
        tags=["DCASE24"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )

    ds_train, ds_val = get_training_set(config, validation=True)
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
    test_dl = DataLoader(dataset=get_testing_set(config),
                         worker_init_fn=worker_init_fn,
                         num_workers=4,
                         batch_size=config.batch_size,
                         persistent_workers=True)
    
    # create pytorch lightening module
    pl_module = PLModule(config)

    #TODO: implement if necessary
    # get model complexity from nessi and log results to wandb
    # sample = next(iter(test_dl))[0][0].unsqueeze(0)
    # shape = pl_module.mel_forward(sample).size()
    # shape = sample.size()
    #macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    # log MACs and number of parameters for our model
    #wandb_logger.experiment.config['MACs'] = macs
    #wandb_logger.experiment.config['Parameters'] = params

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    early_stop_callback = EarlyStopping(monitor="val/acc", min_delta=config.es_delta, patience=config.es_patience, verbose=True, mode="max")
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision,
                         callbacks=[early_stop_callback, pl.callbacks.ModelCheckpoint(save_last=True)])
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, val_dl)
    # final test step
    trainer.test(ckpt_path='last', dataloaders=test_dl)
    wandb.finish()

    del trainer
    del pl_module
    del train_dl, val_dl, test_dl
    import gc
    gc.collect()
    torch.cuda.empty_cache()

"""
def evaluate(config):
    pass
    # import os
    # from sklearn import preprocessing
    # import pandas as pd
    # import torch.nn.functional as F
    # from dataset.dcase24 import dataset_config

    assert config.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select" \
                                      f"the desired checkpoint."

    # create folder to store predictions
    os.makedirs("predictions", exist_ok=True)
    out_dir = os.path.join("predictions", config.ckpt_id)
    os.makedirs(out_dir, exist_ok=True)

    # load lightning module from checkpoint
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    trainer = pl.Trainer(logger=False,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision)

    # evaluate lightning module on development-test split
    test_dl = DataLoader(dataset=get_test_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # get model complexity from nessi
    sample = next(iter(test_dl))[0][0].unsqueeze(0).to(pl_module.device)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)

    print(f"Model Complexity: MACs: {macs}, Params: {params}")
    assert macs <= nessi.MAX_MACS, "The model exceeds the MACs limit and must not be submitted to the challenge!"
    assert params <= nessi.MAX_PARAMS_MEMORY, \
        "The model exceeds the parameter limit and must not be submitted to the challenge!"

    allowed_precision = int(nessi.MAX_PARAMS_MEMORY / params * 8)
    print(f"ATTENTION: According to the number of model parameters and the memory limits that apply in the challenge,"
          f" you are allowed to use at max the following precision for model parameters: {allowed_precision} bit.")

    # obtain and store details on model for reporting in the technical report
    info = {}
    info['MACs'] = macs
    info['Params'] = params
    res = trainer.test(pl_module, test_dl)
    info['test'] = res

    # generate predictions on evaluation set
    eval_dl = DataLoader(dataset=get_eval_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    predictions = trainer.predict(pl_module, dataloaders=eval_dl)
    # all filenames
    all_files = [item[len("audio/"):] for files, _ in predictions for item in files]
    # all predictions
    all_predictions = torch.cat([torch.as_tensor(p) for _, p in predictions], 0)
    all_predictions = F.softmax(all_predictions, dim=1)

    # write eval set predictions to csv file
    df = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    le = preprocessing.LabelEncoder()
    le.fit_transform(df[['scene_label']].values.reshape(-1))
    class_names = le.classes_
    df = {'filename': all_files}
    scene_labels = [class_names[i] for i in torch.argmax(all_predictions, dim=1)]
    df['scene_label'] = scene_labels
    for i, label in enumerate(class_names):
        df[label] = all_predictions[:, i]
    df = pd.DataFrame(df)

    # save eval set predictions, model state_dict and info to output folder
    df.to_csv(os.path.join(out_dir, 'output.csv'), sep='\t', index=False)
    torch.save(pl_module.model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
    with open(os.path.join(out_dir, "info.json"), "w") as json_file:
        json.dump(info, json_file)
"""

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description='argument parser')

    # general
    parser.add_argument('--project_name', type=str, default="DCASE24_Task1")
    parser.add_argument('--experiment_name', type=str, default="Baseline")
    parser.add_argument('--group_name', type=str, default="Baseline")
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for training dataloader
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--running_seed', type=int, default=42)
    parser.add_argument('--precision', type=str, default="32")

    # window creation
    parser.add_argument('--max_samples', type=int, default=599)
    parser.add_argument('--min_samples', type=int, default=76)
    parser.add_argument('--sample_freq', type=int, default=200)
    parser.add_argument('--window_length', type=float, default=0.025)
    parser.add_argument('--window_overlap', type=float, default=0.0)

    # optimizers
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimzer_step', type=int, default=5)
    parser.add_argument('--optimizer_gamma', type=float, default=0.9)

    # model
    parser.add_argument('--n_classes', type=int, default=6)  # classification model with 'n_classes' output neurons
    parser.add_argument('--in_channels', type=int, default=8)
    # channels per sample
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_layers', type=int, default=3)

    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default=None)  # for loading trained model, corresponds to wandb id

    # training
    parser.add_argument('--cv_runs', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=300) 
    parser.add_argument('--batch_size', type=int, default=1000) 
    parser.add_argument('--es_delta', type=float, default=0.001) 
    parser.add_argument('--es_patience', type=int, default=5) 
    parser.add_argument('--n_val_users', type=int, default=6)
    parser.add_argument('--n_train_users', type=int, default=300)
    parser.add_argument('--n_reps', type=int, default=50)

    args = parser.parse_args()
    #if args.evaluate:
        #evaluate(args)
    #else:
    base_experiment_name = args.experiment_name
    for run_id in range(args.cv_runs):
        seed_everything(args.seed)
        args.running_seed = args.seed + run_id
        args.experiment_name = f"{base_experiment_name}_{args.running_seed}"
        train(args)