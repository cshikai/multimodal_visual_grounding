
from typing import Dict, Callable
from collections import OrderedDict
import torch
import pytorch_lightning as pl
import torchvision.models as models


class VisualFeatures(pl.LightningModule):
    FEATURE_LAYERS_ID = ['18', '22', '25', '29']
    # layer ids correspond to layers 4.1,4.3,5.1,5.3 in the vgg16 architecture

    def __init__(self, cfg: Dict) -> None:
        super().__init__()

        self.num_conv = cfg['model']['visual']['num_conv']
        self.M = cfg['model']['visual']['heatmap_dim']
        self.D = cfg['model']['feature_hidden_dimension']
        self.alpha = cfg['model']['leaky_relu_alpha']

        image_model = models.vgg16(pretrained=False)
        checkpoint = torch.load(
            cfg['training']['input_models']['vgg']['path'])
        image_model.load_state_dict(checkpoint)
        self.pretrained_model = list(image_model.children())[0]

        for parameter in self.pretrained_model.parameters():
            parameter.requires_grad = False

        self.raw_visual_features = OrderedDict()
        self.visual_features = OrderedDict()

        # no explict need to reference these hooks ,but reference them for potential future use
        self.forward_hooks = []

        for l in list(self.pretrained_model._modules.keys()):

            if l in self.FEATURE_LAYERS_ID:
                self.forward_hooks.append(
                    getattr(self.pretrained_model, l).register_forward_hook(self._forward_hook(l)))

        self.resize = torch.nn.Upsample(
            size=(self.M, self.M), scale_factor=None, mode='bilinear', align_corners=False)

        for l in self.FEATURE_LAYERS_ID:
            for i in range(self.num_conv):
                layer_name = l+'_conv_' + str(i)
                if i == 0:
                    setattr(self, layer_name, torch.nn.Conv2d(
                        512, self.D,  kernel_size=(1, 1), stride=1))
                else:
                    setattr(self, layer_name, torch.nn.Conv2d(
                        self.D, self.D,  kernel_size=(1, 1), stride=1))
        self.leaky_relu = torch.nn.LeakyReLU(self.alpha, inplace=True)

    def _forward_hook(self, layer_id: str):
        def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> Callable:
            self.raw_visual_features[layer_id] = output
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _out = self.pretrained_model(x)
        for l in self.FEATURE_LAYERS_ID:
            x = self.resize(self.raw_visual_features[l])
            for i in range(self.num_conv):
                layer_name = l+'_conv_' + str(i)
                x = getattr(self, layer_name)(x)
                x = self.leaky_relu(x)
            self.visual_features[l] = x
        x = torch.stack([self.visual_features[l]
                        for l in self.visual_features.keys()], 1)
        x = torch.nn.functional.normalize(
            x, p=2, dim=2).permute((0, 3, 4, 1, 2))
        return x
