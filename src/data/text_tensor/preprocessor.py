import os
from typing import Dict, Tuple
import torch
import numpy as np

from PIL.Image import Image
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from data import transforms


class PreProcessor():

    def __init__(self, cfg: Dict) -> None:

        self.cfg = cfg

        self.text_transforms = transforms.get_transforms(
            self.cfg['data']['transforms']['text'])

    def __call__(self, text:  str) -> torch.Tensor:
        '''
        All pre-trained models expect input images normalized in the same way,
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
        The images have to be loaded in to a range of [0, 1]
        and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        '''

        text = self.text_transforms(text)  # .to('cuda')

        return text

    def collate(self, batch):
        # sort by len
        # batch.sort(key=lambda x: x[-1], reverse=True)
        batch_text, batch_index,  = zip(*batch)
        # batch_pad_text = torch.nn.utils.rnn.pad_sequence(
        #     batch_text, batch_first=True, padding_value=0)

        batch_text = torch.stack(batch_text, 0)

        return batch_text, batch_index
