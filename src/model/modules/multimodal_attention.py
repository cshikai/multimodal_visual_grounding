

from typing import Dict, Tuple

import torch
import pytorch_lightning as pl
import torch.nn.functional as F


class MultimodalAttention(pl.LightningModule):

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.L = cfg['model']['visual']['num_features']
        self.M = cfg['model']['visual']['heatmap_dim']
        self.gamma_1 = cfg['model']['loss']['gamma_1']

    def forward(self,
                image_feature: torch.Tensor,
                word_feature: torch.Tensor,
                sentence_feature: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        word_feature dims (B,T,D)
        image_feature dims (B,M,M,L,D)
        '''
        sentence_feature = sentence_feature.unsqueeze(1)
        batch_size = word_feature.shape[0]

        word_image_heatmap = self.get_attention_heatmap(
            image_feature, word_feature, batch_size)
        sentence_image_heatmap = self.get_attention_heatmap(
            image_feature, sentence_feature, batch_size).squeeze(2)

        word_image_score = self.get_pertinence_scores(
            image_feature, word_feature,  word_image_heatmap, batch_size)
        sentence_image_score = self.get_pertinence_scores(
            image_feature, sentence_feature, sentence_image_heatmap, batch_size)
        # aggregated_sentence_image_score = torch.exp(
        #     word_image_max_score * self.gamma_1) * self.get_len_mask(batch_size, max_word_len, seq_lens)
        # aggregated_sentence_image_score = torch.log(
        #     torch.sum(aggregated_sentence_image_score, 2) ** (1/self.gamma_1))
    # aggregated_sentence_image_score, sentence_image_score
        return word_image_heatmap, sentence_image_heatmap, word_image_score, sentence_image_score

    def get_attention_heatmap(self, image_feature: torch.Tensor, word_feature: torch.Tensor, batch_size: int):
        max_word_len = word_feature.shape[1]
        # #reshape to (B,B'M,M,T,L,D) repeated along dim 0
        reshaped_word_feature = word_feature.unsqueeze(1).unsqueeze(1).unsqueeze(
            4).unsqueeze(0).expand(batch_size, -1, self.M, self.M, -1, self.L, -1)

        # reshape to (B,B'M,M,T,L,D) repeated along dim 1
        reshaped_image_feature = image_feature.unsqueeze(3).unsqueeze(
            1).expand(-1, batch_size, -1, -1, max_word_len, -1, -1)
        # heatmap dims (B,B',M,M,T,L)
        similarity_heatmap = F.relu(F.cosine_similarity(
            reshaped_word_feature, reshaped_image_feature, dim=6))
        return similarity_heatmap

    def get_pertinence_scores(self, image_feature: torch.Tensor,
                              text_feature: torch.Tensor,
                              similarity_heatmap: torch.Tensor, batch_size: int):
        max_word_len = text_feature.shape[1]
        # collapse image width and heigh dimensions into single dim for weighted summing via matrix mult
        # reshape so that dimension to sum across is at the end
        # (B,B' T, L, 1, MXM)
        similarity_heatmap_flat = torch.flatten(
            similarity_heatmap, start_dim=2, end_dim=3).permute(0, 1, 3, 4, 2).unsqueeze(4)
        # (B,B',T,L,MXM,D)
        # collapse width and height dims for image_feature for weighted summing via matrix mult
        image_feature_flat = torch.flatten(image_feature, start_dim=1, end_dim=2).permute(
            0, 2, 1, 3).unsqueeze(1).unsqueeze(1).expand(-1, batch_size, max_word_len, -1, -1, -1)

        visual_word_attention = torch.matmul(
            similarity_heatmap_flat, image_feature_flat).squeeze(4)

        # (B,B',T,L,D)
        visual_word_attention = torch.nn.functional.normalize(
            visual_word_attention, p=2, dim=4)

        # (B,B',T,L)
        word_image_pertinence_score = F.cosine_similarity(text_feature.unsqueeze(2).unsqueeze(
            1).expand(-1, batch_size, -1, self.L, -1), visual_word_attention, dim=4)

        # (B,B',T,)
        word_image_max_pertinence_score, _ = torch.max(
            word_image_pertinence_score, dim=3)

        return word_image_max_pertinence_score
