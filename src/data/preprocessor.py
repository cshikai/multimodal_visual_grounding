import os
from typing import Dict, Tuple
import torch
import numpy as np

from PIL.Image import Image
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from data import transforms


class PreProcessor():
    ELMO_OPTIONS_FILE = "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    ELMO_WEIGHT_FILE = "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

    def __init__(self, cfg: Dict) -> None:

        self.cfg = cfg
        # self.image_transforms = transforms.get_transforms(
        #     self.cfg['data']['transforms']['image'])
        # self.text_transforms = transforms.get_transforms(
        #     self.cfg['data']['transforms']['text'])

        # model_root = cfg['training']['input_models']['elmo']['path']
        # self.elmo = ElmoTokenEmbedder(
        #     options_file=os.path.join(model_root, self.ELMO_OPTIONS_FILE), weight_file=os.path.join(model_root, self.ELMO_WEIGHT_FILE), dropout=cfg['model']['embeddings']['elmo']['dropout'])
        # # self.elmo.cuda()
        # print('elmo embedder initialized on gpu?:',
        #       next(self.elmo.parameters()).is_cuda)

        # for param in self.elmo._elmo.parameters():
        #     param.requires_grad = False

        # self.elmo._elmo._modules['_elmo_lstm']._elmo_lstm.stateful = False

    def __call__(self, data: Tuple[Image, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        All pre-trained models expect input images normalized in the same way,
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
        The images have to be loaded in to a range of [0, 1]
        and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        '''
        image, text = data

        image = self.image_transforms(image)

        text = self.text_transforms(text)  # .to('cuda')

        embeddings = self.elmo(text.unsqueeze(0)).squeeze(0)  # .to('cpu')

        return image, embeddings, embeddings.shape[0]

    def collate(self, batch):
        # sort by len
        batch.sort(key=lambda x: x[-1], reverse=True)
        batch_image, batch_text, batch_len = zip(*batch)
        batch_pad_text = torch.nn.utils.rnn.pad_sequence(
            batch_text, batch_first=True, padding_value=0)
        batch_image = torch.stack(batch_image, 0)
        batch_len = torch.Tensor(batch_len).type(torch.int64).unsqueeze(1)
        return batch_image, batch_pad_text, batch_len
