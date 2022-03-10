
import os
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path


import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from aiplatform.s3utility import S3Callback, S3Utils
from aiplatform.config import cfg as aip_cfg
from .model import Seq2Seq

from .config import cfg
from .dataset import FlightDataset


# TODO: check tensor types


class Experiment():

   # should init as arguments here
    def __init__(self, cfg, clearml_task=None):
        self.clearml_task = clearml_task
        self.cfg = cfg

    def run_experiment(self):
        # if os.path.exists(self.checkpoint_dir):
        #     shutil.rmtree(self.checkpoint_dir)

        # os.makedirs(os.path.join(self.checkpoint_dir,'logs'), exist_ok=True)

        pl.seed_everything(self.seed)

        self._set_datasets()

        self._set_dataloaders()

        self._log_metadata()

        # load from checkpoint
        if cfg['train']['resume_from_checkpoint']:
            self._load_model()
        else:
            self._initialize_model()
        self._set_callbacks()
        self._set_logger()

        self._set_trainer()

        self._start_training()

    def _start_training(self):

        if self.cfg.auto_lr:
            lr_finder = self.trainer.tuner.lr_find(
                self.model, self.train_loader, self.valid_loader)
            new_lr = lr_finder.suggestion()
            self.model.learning_rate = new_lr
            self.print('Found a starting LR of {}'.format(new_lr))
        self.trainer.fit(self.model, self.train_loader, self.valid_loader)

    def _set_trainer(self):
        self.trainer = pl.Trainer(
            gpus=self.n_gpu,
            accelerator=self.accelerator if self.n_gpu > 1 else None,
            callbacks=self.callbacks,
            logger=self.logger,
            max_epochs=self.n_epochs,
            default_root_dir=self.checkpoint_dir,
            log_every_n_steps=self.log_every_n_steps
        )

    def _log_metadata(self):

        class_weights = {
            'label_segment_count': train_dataset.get_class_weights('label_segment_count'),
            'label_point_count': train_dataset.get_class_weights('label_point_count'),
            'None': train_dataset.get_class_weights('None')
        }
        # labels_map = train_dataset.labels_map
        # n_callsign_tokens = len(train_dataset.CALLSIGN_CHAR2IDX)
        # n_mode3_tokens = len(train_dataset.MODE3_CHAR2IDX)
        # n_classes = train_dataset.n_classes
        # distributed = self.n_gpu > 1
        # if self.clearml_task:
        #     if self.weight_by != 'None':
        #         self.clearml_task.connect_configuration({str(i): val for i, val in enumerate(
        #             class_weights[self.weight_by].cpu().numpy())}, name='Class Weights')
        #     self.clearml_task.connect_configuration(
        #         labels_map, name='Labels Map')

        # metas = {'Train': train_dataset.metadata.copy(
        #       ), 'Valid': valid_dataset.metadata.copy()}
        #   for meta in metas.keys():
        #        for key in ['labels', 'length', 'track_ids']:
        #             metas[meta].pop(key)
        #         self.clearml_task.connect_configuration(
        #             metas[meta], name='{} Metadata'.format(meta))

    def _initialize_model(self):
        self.model = Seq2Seq(self.learning_rate,
                             self.lr_schedule,
                             self.n_features,
                             self.d_model,
                             self.dim_feedforward,
                             self.nhead,
                             self.num_encoder_layers,
                             self.num_decoder_layers,
                             self.enc_dropout,
                             self.dec_dropout,
                             self.input_dropout,
                             self.transformer_activation,
                             self.id_embed_dim,
                             n_mode3_tokens,
                             self.n_mode3_token_embedding,
                             self.n_mode3_token_layers,
                             n_callsign_tokens,
                             self.n_callsign_token_embedding,
                             self.n_callsign_token_layers,
                             n_classes,
                             class_weights,
                             self.weight_by,
                             labels_map,
                             distributed)

    def _load_model(self):
        s3_utils = S3Utils(aip_cfg.s3.bucket,
                           aip_cfg.s3.model_artifact_path)
        model_path = os.path.join(
            cfg['train']['checkpoint_dir'], 'latest_model.ckpt')
        s3_utils.s3_download_file(os.path.join(
            self.clearml_task.name, 'latest_model.ckpt'), model_path)
        self.model = Seq2Seq.load_from_checkpoint(checkpoint_path=model_path)

    def _set_datasets(self,):
        self.train_dataset = FlightDataset(self.datapath, self.features, self.label, self.mode3_column,
                                           self.callsign_column, "train", self.transforms, self.time_encoding_dims)
        self.valid_dataset = FlightDataset(self.datapath, self.features, self.label, self.mode3_column,
                                           self.callsign_column, "valid", self.transforms, self.time_encoding_dims)

    def _set_dataloaders(self):
        y_padding = self.train_dataset.labels_map['pad']
        callsign_padding = self.train_dataset.CALLSIGN_CHAR2IDX['_']
        mode3_padding = train_dataset.MODE3_CHAR2IDX['_']
        self.train_loader = DataLoader(self.train_dataset, collate_fn=lambda x: default_collate(x, y_padding, mode3_padding, callsign_padding),
                                       batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.valid_loader = DataLoader(self.valid_dataset, collate_fn=lambda x: default_collate(x, y_padding, mode3_padding, callsign_padding),
                                       batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def _get_logger(self):
        logger = TensorBoardLogger(self.checkpoint_dir, name='logs')
        return logger

    def _set_callbacks(self):
        self.callbacks = []

        # checkpoint_callback = CustomCheckpoint(
        #     task_name=self.clearml_task.name,
        #     dirpath=self.checkpoint_dir,
        #     filename = '-{epoch}',
        #     save_top_k= self.save_top_k,
        #     verbose=True,
        #     monitor='val_loss',
        #     mode='min',
        #     period = self.model_save_period
        #     )
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename='{k}-{epoch}',
            save_top_k=self.save_top_k,
            verbose=True,
            save_last=True,
            monitor='val_loss',
            mode='min',
            every_n_val_epochs=self.model_save_period
        )
        self.callbacks.append(checkpoint_callback)
        if self.lr_schedule['scheduler']:
            lr_logging_callback = LearningRateMonitor(logging_interval='step')
            self.callbacks.append(lr_logging_callback)

        if self.clearml_task:
            self.callbacks.append(S3Callback(self.clearml_task.name))

    @staticmethod
    def add_experiment_args(parent_parser):

        def get_unnested_dict(d, root=''):
            unnested_dict = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    unnested_dict.update(
                        get_unnested_dict(value, root+key+'_'))
                else:
                    unnested_dict[root+key] = value
            return unnested_dict
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        unnested_args = get_unnested_dict(cfg)
        for key, value in unnested_args.items():
            # only parse int,float or str
            if isinstance(value, (int, str, float)):
                # do not parse transforms and lr schedule as we want them as nested dicts
                if 'transforms' not in key and 'lr_schedule' not in key:
                    parser.add_argument('--'+key, default=value)

        return parser

    @staticmethod
    def create_torchscript_model(model_name):
        model = Seq2Seq.load_from_checkpoint(os.path.join(
            cfg['train']['checkpoint_dir'], model_name))

        model.eval()

        # remove_empty_attributes(model)
        # print(vars(model._modules['input_mapper']))
        # print('These attributes should have been removed', remove_attributes)
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(
            cfg['train']['checkpoint_dir'], "model.pt"))

    @staticmethod
    def create_torchscript_cpu_model(model_name):
        model = Seq2Seq.load_from_checkpoint(os.path.join(
            cfg['train']['checkpoint_dir'], model_name))

        model.to('cpu')
        model.eval()

        # remove_empty_attributes(model)
        # print(vars(model._modules['input_mapper']))
        # print('These attributes should have been removed', remove_attributes)
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(
            cfg['train']['checkpoint_dir'], "model_cpu.pt"))


def remove_empty_attributes(module):
    remove_attributes = []
    for key, value in vars(module).items():
        if value is None:

            if key == 'trainer' or '_' == key[0]:
                remove_attributes.append(key)
        elif key == '_modules':
            for mod in value.keys():

                remove_empty_attributes(value[mod])
    print('To be removed', remove_attributes)
    for key in remove_attributes:

        delattr(module, key)


def calc_accuracy(output, Y, mask):
    """
    Calculate the accuracy (point by point evaluation)
    :param output: output from the model (tensor)
    :param Y: ground truth given by dataset (tensor)
    :param mask: used to mask out the padding (tensor)
    :return: accuracy used for validation logs (float)
    """
    _, max_indices = torch.max(output.data, 1)
    max_indices = max_indices.view(mask.shape[1], mask.shape[0]).permute(1, 0)
    Y = Y.view(mask.shape[1], mask.shape[0]).permute(1, 0)
    max_indices = torch.masked_select(max_indices, mask)
    Y = torch.masked_select(Y, mask)
    train_acc = (max_indices == Y).sum().item()/max_indices.size()[0]
    return train_acc, max_indices, Y
