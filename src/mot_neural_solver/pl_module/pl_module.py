import os
import os.path as osp

import dataclasses

import pandas as pd

from torch_geometric.data import DataLoader

import torch

from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from torch.nn import functional as F

import pytorch_lightning as pl

from mot_neural_solver.data.mot_graph_dataset import MOTGraphDataset
from mot_neural_solver.models.mpn import MOTMPNet
from mot_neural_solver.models.resnet import resnet50_fc256, load_pretrained_weights
from mot_neural_solver.path_cfg import OUTPUT_PATH
from mot_neural_solver.utils.evaluation import compute_perform_metrics
from mot_neural_solver.tracker.mpn_tracker import MPNTracker

class MOTNeuralSolver(pl.LightningModule):
    """
    Pytorch Lightning wrapper around the MPN defined in model/mpn.py.
    (see https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html)

    It includes all data loading and train / val logic., and it is used for both training and testing models.
    """
    def __init__(self, hparams):
        super().__init__()

        self._val_step_outputs = []
        if isinstance(hparams, dict):
          hparams = hparams
        elif dataclasses.is_dataclass(hparams):
            hparams = dataclasses.asdict(hparams) 
        elif hasattr(hparams, "__dict__"):
            hparams = vars(hparams)
        else: 
            hparams = {"hparams": hparams}
        self.save_hyperparameters(hparams)
        self.model, self.cnn_model = self.load_model()
    
    def forward(self, x):
        self.model(x)

    def load_model(self):
        cnn_arch = self.hparams['graph_model_params']['cnn_params']['arch']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model =  MOTMPNet(self.hparams['graph_model_params']).to(device)
        
        cnn_model = resnet50_fc256(10, loss='xent', pretrained=True).to(device)
        load_pretrained_weights(cnn_model,
                                osp.join(OUTPUT_PATH, self.hparams['graph_model_params']['cnn_params']['model_weights_path'][cnn_arch]))
        cnn_model.return_embeddings = True

        return model, cnn_model

    def _get_data(self, mode, return_data_loader = True):
        assert mode in ('train', 'val', 'test')

        dataset = MOTGraphDataset(dataset_params=self.hparams['dataset_params'],
                                  mode=mode,
                                  cnn_model=self.cnn_model,
                                  splits= self.hparams['data_splits'][mode],
                                  logger=None)

        if return_data_loader and len(dataset) > 0:
            train_dataloader = DataLoader(dataset,
                                          batch_size = self.hparams['train_params']['batch_size'],
                                          shuffle = True if mode == 'train' else False,
                                          num_workers=self.hparams['train_params']['num_workers'])
            return train_dataloader
        
        elif return_data_loader and len(dataset) == 0:
            return []
        
        else:
            return dataset

    def train_dataloader(self):
        return self._get_data(mode = 'train')

    def val_dataloader(self):
        return self._get_data('val')

    def test_dataset(self, return_data_loader=False):
        return self._get_data('test', return_data_loader = return_data_loader)

    def configure_optimizers(self):
        optim_class = getattr(optim_module, self.hparams['train_params']['optimizer']['type'])
        optimizer = optim_class(self.model.parameters(), **self.hparams['train_params']['optimizer']['args'])

        if self.hparams['train_params']['lr_scheduler']['type'] is not None:
            lr_sched_class = getattr(lr_sched_module, self.hparams['train_params']['lr_scheduler']['type'])
            lr_scheduler = lr_sched_class(optimizer, **self.hparams['train_params']['lr_scheduler']['args'])

            return [optimizer], [lr_scheduler]

        else:
            return optimizer

    def _compute_loss(self, outputs, batch):
        # Define Balancing weight
        positive_vals = batch.edge_labels.sum()

        if positive_vals:
            pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals

        else: # If there are no positives labels, avoid dividing by zero
            pos_weight = 0

        # Compute Weighted BCE:
        loss = 0
        num_steps = len(outputs['classified_edges'])
        for step in range(num_steps):
            loss += F.binary_cross_entropy_with_logits(outputs['classified_edges'][step].view(-1),
                                                            batch.edge_labels.view(-1),
                                                            pos_weight= pos_weight)
        return loss

    def _train_val_step(self, batch, batch_idx, train_val):
        device = (next(self.model.parameters())).device
        batch.to(device)

        outputs = self.model(batch)
        loss = self._compute_loss(outputs, batch)

        metrics = compute_perform_metrics(outputs, batch)
        metrics["loss"] = loss
        metrics = {f"{key}/{train_val}": val for key, val in metrics.items()}

        on_step = (train_val == "train")
        for key, value in metrics.items():
            self.log(key, value, on_step=on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

        return {"loss": loss} if train_val == "train" else None

    def training_step(self, batch, batch_idx):
        return self._train_val_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._train_val_step(batch, batch_idx, 'val')

    def on_validation_epoch_end(self):
        if not self._val_step_outputs:
            return

        metrics = pd.DataFrame(self._val_step_outputs).mean(axis=0).to_dict()

        for key, value in metrics.items():
            val_tensor = torch.as_tensor(value)
            self.log('val_loss' if key == 'loss/val' else key,
                     val_tensor,
                     prog_bar=(key == 'loss/val'),
                     logger=True,
                     sync_dist=False)

        self._val_step_outputs.clear()  

    def track_all_seqs(self, output_files_dir, dataset, use_gt = False, verbose = False):
        tracker = MPNTracker(dataset=dataset,
                             graph_model=self.model,
                             use_gt=use_gt,
                             eval_params=self.hparams['eval_params'],
                             dataset_params=self.hparams['dataset_params'])

        constraint_sr = pd.Series(dtype=float)
        for seq_name in dataset.seq_names:
            print("Tracking", seq_name)
            if verbose:
                print("Tracking sequence ", seq_name)

            os.makedirs(output_files_dir, exist_ok=True)
            _, constraint_sr[seq_name] = tracker.track(seq_name, output_path=osp.join(output_files_dir, seq_name + '.txt'))

            if verbose:
                print("Done! \n")


        constraint_sr['OVERALL'] = constraint_sr.mean()

        return constraint_sr
