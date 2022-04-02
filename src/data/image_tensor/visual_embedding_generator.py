
import os
from typing import Dict


import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader


from config.config import cfg
from .visual_model import VGG
from .dataset import VGImageDataset
from .preprocessor import PreProcessor


class EmbeddingGenerator():

   # should init as arguments here
    def __init__(self, cfg: Dict, clearml_task=None) -> None:
        self.clearml_task = clearml_task
        self.cfg = cfg

    def run(self, dataset_root) -> None:
        # if os.path.exists(self.checkpoint_dir):
        #     shutil.rmtree(self.checkpoint_dir)

        # os.makedirs(os.path.join(self.checkpoint_dir,'logs'), exist_ok=True)
        pl.seed_everything(self.cfg['training']['seed'])

        self._set_datasets()

        self._set_dataloaders()

        self._initialize_model()

        self._generate(dataset_root)

    def _initialize_model(self) -> None:
        self.model = VGG(self.cfg)

    def _set_datasets(self) -> None:
        preprocessor = PreProcessor(cfg)
        self.train_dataset = VGImageDataset('train', preprocessor)
        self.valid_dataset = VGImageDataset('valid', preprocessor)

    def _set_dataloaders(self) -> None:
        # self.cfg['training']['batch_size']
        self.train_loader = DataLoader(self.train_dataset, collate_fn=self.train_dataset.preprocessor.collate,
                                       batch_size=1, shuffle=False, num_workers=self.cfg['training']['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset, collate_fn=self.valid_dataset.preprocessor.collate,
                                       batch_size=1, shuffle=False, num_workers=self.cfg['training']['num_workers'])

    def _generate(self, dataset_root):

        REPORT_INTERVAL = 100
        folders = ['train', 'valid']

        self.model.cuda()
        self.model.eval()

        print('vgg embedder started on gpu?:',
              next(self.model.parameters()).is_cuda)

        for f in folders:
            path = os.path.join(dataset_root, f, 'image')

            if not os.path.exists(path):
                os.makedirs(path)
        for batch in self.train_loader:
            batch_image, batch_index = batch
            index = batch_index[0]
            output = self.model(batch_image.cuda())

            if index % REPORT_INTERVAL == 0:
                print('processing train image of index {}'.format(index))

            torch.save(
                output, '/data/embeddings/train/image/{}'.format(index))
            break

        for batch in self.valid_loader:

            batch_image, batch_index = batch
            index = batch_index[0]

            output = self.model(batch_image.cuda())
            if index % REPORT_INTERVAL == 0:
                print('processing valid image of index {}'.format(index))

            torch.save(
                output, '/data/embeddings/valid/image/{}'.format(index))
            break

    @staticmethod
    def create_torchscript_model(model_name: str) -> None:
        model = VGG.load_from_checkpoint(model_name)
        model.eval()
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(
            '/ models/trained_models /', "model.pt"))
