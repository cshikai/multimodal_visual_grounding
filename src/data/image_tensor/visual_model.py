from collections import OrderedDict
from typing import Dict, List,  Tuple, Callable
import os
import torch
import pytorch_lightning as pl
import torchvision.models as models


class VGG(pl.LightningModule):
    """
    """
    FEATURE_LAYERS_ID = ['18', '22', '25', '29']

    def __init__(self, cfg: Dict) -> None:
        """
        """
        super().__init__()

        self.M = cfg['model']['visual']['heatmap_dim']

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

        self.pretrained_model.eval()

    def _forward_hook(self, layer_id: str):

        def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> Callable:
            self.raw_visual_features[layer_id] = output
        return hook

    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        """

        """
        # x = self.elmo(x.unsqueeze(0)).squeeze(0)
        _out = self.pretrained_model(x)
        output = []
        for l in self.FEATURE_LAYERS_ID:
            output.append(self.resize(self.raw_visual_features[l]))

        output = torch.stack(output, 1)

        return output


class ResNet(pl.LightningModule):
    """
    """
    FEATURE_LAYERS_ID = [(7, 0),(7, 1),(7, 2)] # Conv of Layer 4 of ResNet 34
    layer_dict_key = [str(a) for a in range(len(FEATURE_LAYERS_ID))]

    def __init__(self, cfg: Dict) -> None:
        """
        """
        super().__init__()

        self.M = cfg['model']['visual']['heatmap_dim']

        image_model = models.resnet34(pretrained=True)
        # checkpoint = torch.load(
        #     cfg['training']['input_models']['vgg']['path'])
        # image_model.load_state_dict(checkpoint)
        self.pretrained_model = image_model

        for parameter in self.pretrained_model.parameters():
            parameter.requires_grad = False

        self.raw_visual_features = OrderedDict()
        self.visual_features = OrderedDict()
        self.output=[]

        # no explict need to reference these hooks ,but reference them for potential future use
        self.forward_hooks = []

        for key, layer in zip(self.layer_dict_key, self.FEATURE_LAYERS_ID):
            outer = getattr(self.pretrained_model, list(self.pretrained_model._modules.keys())[layer[0]])
            mid = getattr(outer, list(outer._modules.keys())[layer[1]])
            mid.relu.register_forward_hook(self._forward_hook(key))

        
        self.resize = torch.nn.Upsample(
            size=(self.M, self.M), scale_factor=None, mode='bilinear', align_corners=False)

        self.pretrained_model.eval()

    def _forward_hook(self, layer_id: str):

        def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> Callable:
            self.raw_visual_features[layer_id] = output
        return hook


    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        """
        """
        # x = self.elmo(x.unsqueeze(0)).squeeze(0)
        _out = self.pretrained_model(x)
        print(self.raw_visual_features)
        output = []
        for l in self.layer_dict_key:
            output.append(self.resize(self.raw_visual_features[l]))

        output = torch.stack(output,dim=1)

        return output
