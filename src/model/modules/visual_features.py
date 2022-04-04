
from typing import Dict, Callable
from collections import OrderedDict
import torch
import pytorch_lightning as pl
import torchvision.models as models


class VisualFeatures(pl.LightningModule):
    NUM_LAYERS = 4
    # layer ids correspond to layers 4.1,4.3,5.1,5.3 in the vgg16 architecture

    def __init__(self, cfg: Dict) -> None:
        super().__init__()

        self.num_conv = cfg['model']['visual']['num_conv']
        self.D = cfg['model']['feature_hidden_dimension']
        self.alpha = cfg['model']['leaky_relu_alpha']

        self.visual_features = OrderedDict()

        # no explict need to reference these hooks ,but reference them for potential future use

        for l in range(self.NUM_LAYERS):
            for i in range(self.num_conv):
                layer_name = str(l)+'_conv_' + str(i)
                if i == 0:
                    setattr(self, layer_name, torch.nn.Conv2d(
                        512, self.D,  kernel_size=(1, 1), stride=1))
                else:
                    setattr(self, layer_name, torch.nn.Conv2d(
                        self.D, self.D,  kernel_size=(1, 1), stride=1))

        self.leaky_relu = torch.nn.LeakyReLU(self.alpha, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in range(self.NUM_LAYERS):
            layer = x[:, l, ...]
            for i in range(self.num_conv):
                layer_name = str(l)+'_conv_' + str(i)
                layer = getattr(self, layer_name)(layer)
                layer = self.leaky_relu(layer)
            self.visual_features[l] = layer

        x = torch.stack([self.visual_features[l]
                        for l in self.visual_features.keys()], 1)
        x = torch.nn.functional.normalize(
            x, p=2, dim=2).permute((0, 3, 4, 1, 2))

        return x
