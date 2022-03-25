
from typing import Dict, List, Tuple, Any

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torch.optim import Optimizer

from .modules.textual_features import TextualFeatures
from .modules.visual_features import VisualFeatures
from .modules.multimodal_attention import MultimodalAttention
from .modules.multimodal_loss import MultimodalLoss


class VisualGroundingModel(pl.LightningModule):
    """
    """
    # individual args so that they can be serialized in torchscript

    def __init__(self, cfg: Dict, distributed: bool) -> None:
        """
        :param dec_dropout: dropout ratio for decoder
        """
        super().__init__()
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = cfg['training']['learning_rate']
        self.lr_schedule = cfg['training']['lr_schedule']
        self.text_features = TextualFeatures(cfg)
        self.visual_features = VisualFeatures(cfg)
        self.attention = MultimodalAttention(cfg)
        self.loss = MultimodalLoss(
            cfg['model']['loss']['gamma_1'], cfg['model']['loss']['gamma_2'])

        self.distributed = distributed
        self.save_hyperparameters()

    def forward(self,
                image: torch.Tensor,
                text: torch.Tensor,
                seq_len: List[int]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        """
        word_features, sentence_features = self.text_features(text, seq_len)
        image_features = self.visual_features(image)
        word_image_heatmap, sentence_image_heatmap, word_image_score, sentence_image_score = self.attention(
            image_features, word_features, sentence_features)

        return word_image_heatmap, sentence_image_heatmap, word_image_score, sentence_image_score

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        '''
        Pytorch Lightning Trainer (training)
        '''
        image = batch[0]
        text = batch[1]
        seq_len = batch[2]

        # this is calling the forward implicitly
        word_image_heatmap, sentence_image_heatmap, word_image_score, sentence_image_score = self(
            image, text, seq_len)
        loss = self.loss(word_image_score, sentence_image_score, seq_len)
        self.log('train_loss', loss, sync_dist=self.distributed)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        '''
        Pytorch Lightning Trainer (validation)
        '''

        image = batch[0]
        text = batch[1]
        seq_len = batch[2]

        # this is calling the forward implicitly
        word_image_heatmap, sentence_image_heatmap, word_image_score, sentence_image_score = self(
            image, text, seq_len)
        loss = self.loss(word_image_score, sentence_image_score, seq_len)

        self.log('val_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=self.distributed)

        return {
            'val_loss': loss,
        }

    def validation_epoch_end(self, validation_step_outputs: Dict[str, Any]) -> None:
        pass
        # Log confusion matrices into tensorboard

        # preds_list = list(
        #     map(lambda x: x['predictions'], validation_step_outputs))

        # y_pred = torch.cat(preds_list).cpu().numpy()

        # self.logger.experiment.add_figure(
        #     'True vs Predicted Labels', cm_fig, global_step=self.current_epoch)

        # if self.target_classes:
        #     cm_fig = plot_confusion_matrix(y_true, y_pred, self.class_names, self.target_classes)
        #     self.logger.experiment.add_figure('True vs Predicted Labels for Targeted Classes', cm_fig, global_step=self.current_epoch)

    def configure_optimizers(self) -> Optimizer:
        '''
        '''

        if self.lr_schedule['scheduler'] == 'lr_decay':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.lr_schedule['lr_decay']['factor'],
                # will drop lr AFTER the patience + 1 epoch
                patience=self.lr_schedule['lr_decay']['patience'],
                cooldown=self.lr_schedule['lr_decay']['cooldown'],
                eps=self.lr_schedule['lr_decay']['eps'],
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': self.lr_schedule['lr_decay']['metric_to_track']
            }

        elif self.lr_schedule['scheduler'] == 'lr_cyclic':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate)

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                               base_lr=self.lr_schedule['lr_cyclic']['lower_lr'],
                                                               max_lr=self.learning_rate,
                                                               # * self.batch_per_epoch,
                                                               step_size_up=self.lr_schedule['lr_cyclic']['epoch_size_up'],
                                                               # , * self.batch_per_epoch,
                                                               step_size_down=self.lr_schedule[
                                                                   'lr_cyclic']['epoch_size_down'],
                                                               mode=self.lr_schedule['lr_cyclic']['mode'],
                                                               cycle_momentum=False,
                                                               ),
                'interval': 'epoch',
            }

            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
            }
        # use scale fn and scale mode to overwrite mode.
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate)
            return optimizer
