
import os
from typing import Dict


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from clearml import StorageManager


from config.config import cfg
from .textual_model import Elmo
from .dataset import VGTextDataset
from .preprocessor import PreProcessor


class EmbeddingGenerator():

   # should init as arguments here
    def __init__(self, cfg: Dict, clearml_task=None) -> None:
        self.clearml_task = clearml_task
        self.cfg = cfg

    def run(self) -> None:
        # if os.path.exists(self.checkpoint_dir):
        #     shutil.rmtree(self.checkpoint_dir)

        # os.makedirs(os.path.join(self.checkpoint_dir,'logs'), exist_ok=True)
        pl.seed_everything(self.cfg['training']['seed'])

        self._set_datasets()

        self._set_dataloaders()

        self._initialize_model()

        self._generate()

    def _initialize_model(self) -> None:
        self.model = Elmo(self.cfg)

    def _set_datasets(self) -> None:
        preprocessor = PreProcessor(cfg)
        self.train_dataset = VGTextDataset('train', preprocessor)
        self.valid_dataset = VGTextDataset('valid', preprocessor)

    def _set_dataloaders(self) -> None:
        # self.cfg['training']['batch_size']
        self.train_loader = DataLoader(self.train_dataset, collate_fn=self.train_dataset.preprocessor.collate,
                                       batch_size=1, shuffle=False, num_workers=self.cfg['training']['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset, collate_fn=self.valid_dataset.preprocessor.collate,
                                       batch_size=1, shuffle=False, num_workers=self.cfg['training']['num_workers'])

    def _generate(self, dataset_root):

        folders = ['train', 'valid']

        self.model.cuda()

        print('elmo embedder started on gpu?:',
              next(self.model.parameters()).is_cuda)

        for f in folders:
            path = os.path.join(dataset_root, f, 'text')

            if not os.path.exists(path):
                os.makedirs(path)
        for batch in self.train_loader:
            batch_text, batch_index = batch
            output = self.model(batch_text.cuda()).squeeze(0)

            torch.save(
                output, '/data/embeddings/train/text/{}'.format(batch_index[0]))
            break
        for batch in self.valid_loader:
            batch_text, batch_index = batch
            output = self.model(batch_text.cuda()).squeeze(0)

            torch.save(
                output, '/data/embeddings/valid/text/{}'.format(batch_index[0]))
            break

    @staticmethod
    def create_torchscript_model(model_name: str) -> None:
        model = Elmo.load_from_checkpoint(model_name)
        model.eval()
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(
            '/ models/trained_models /', "model.pt"))
