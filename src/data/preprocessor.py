
from typing import Dict, Tuple
import torch
import numpy as np

from PIL.Image import Image
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from data import transforms


class PreProcessor():
    ELMO_OPTIONS_FILE = "/models/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    ELMO_WEIGHT_FILE = "/models/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

    def __init__(self, cfg: Dict) -> None:

        self.cfg = cfg
        self.image_transforms = transforms.get_transforms(
            self.cfg['data']['transforms']['image'])
        self.text_transforms = transforms.get_transforms(
            self.cfg['data']['transforms']['text'])

        self.elmo = ElmoTokenEmbedder(
            options_file=self.ELMO_OPTIONS_FILE, weight_file=self.ELMO_WEIGHT_FILE, dropout=cfg['model']['embeddings']['elmo']['dropout'])
        self.elmo.cuda()
        print('elmo embedder initialized on gpu?:',
              next(self.elmo.parameters()).is_cuda)

        self.elmo._elmo._modules['_elmo_lstm']._elmo_lstm.stateful = False

    def __call__(self, data: Tuple[Image, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        All pre-trained models expect input images normalized in the same way,
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
        The images have to be loaded in to a range of [0, 1]
        and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        '''
        image, text = data

        image = self.image_transforms(image)

        text = self.text_transforms(text).cuda()

        embeddings = self.elmo(text.unsqueeze(0)).squeeze(0)

        return image, embeddings, embeddings.shape[1]

    def collate(self, batch):
        # sort by len
        batch.sort(key=lambda x: x[-1], reverse=True)
        batch_image, batch_text, batch_len = zip(*batch)
        batch_pad_text = torch.nn.utils.rnn.pad_sequence(
            batch_text, batch_first=True, padding_value=0).numpy()
        batch_pad_mode3 = torch.nn.utils.rnn.pad_sequence(
            batch_mode3, batch_first=True, padding_value=self.mode3_padding_value).numpy()
        batch_pad_callsign = torch.nn.utils.rnn.pad_sequence(
            batch_callsign, batch_first=True, padding_value=self.callsign_padding_value).numpy()

        return batch_pad_x, batch_pad_mode3, batch_pad_callsign, batch_len, batch_id
