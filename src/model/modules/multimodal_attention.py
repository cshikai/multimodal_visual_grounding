

from typing import Dict, Tuple

import torch
import pytorch_lightning as pl
import torch.nn.functional as F


class MultimodalAttention(pl.LightningModule):
    SPLIT_VA_ACROSS_GPU = 1

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
            image_feature, sentence_feature, batch_size)

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
        # #reshape to (B,B'M,M,T,1,D) repeated along dim 0

        reshaped_word_feature = word_feature.unsqueeze(1).unsqueeze(1).unsqueeze(
            4).unsqueeze(0).expand(batch_size, -1, self.M, self.M, -1, -1, -1)

        # reshape to (B,B'M,M,T,L,D) repeated along dim 1
        reshaped_image_feature = image_feature.unsqueeze(3).unsqueeze(
            1).expand(-1, batch_size, -1, -1, max_word_len, -1, -1)

        reshaped_image_feature = torch.split(
            reshaped_image_feature, reshaped_image_feature.shape[5]//self.SPLIT_VA_ACROSS_GPU, dim=5)

        reshaped_image_feature = tuple([
            f.to('cuda:{:d}'.format(gpu_id)) for gpu_id, f in enumerate(reshaped_image_feature)])

        # heatmap dims (B,B',M,M,T,1)
        similarity_heatmap = []

        for gpu_id in range(self.SPLIT_VA_ACROSS_GPU):
            device = 'cuda:{:d}'.format(gpu_id)
            similarity_heatmap_l = F.relu(F.cosine_similarity(
                reshaped_word_feature.to(device), reshaped_image_feature[gpu_id].to(device), dim=6))

            similarity_heatmap.append(similarity_heatmap_l)
        return similarity_heatmap

    def get_pertinence_scores(self, image_feature: torch.Tensor,
                              text_feature: torch.Tensor,
                              similarity_heatmap: torch.Tensor, batch_size: int):
        max_word_len = text_feature.shape[1]
        print('max-word-len', max_word_len)
        # collapse image width and heigh dimensions into single dim for weighted summing via matrix mult

        # similarity_heatmap_flat = torch.flatten(
        #     similarity_heatmap, start_dim=2, end_dim=3).permute(0, 1, 3, 4, 2).unsqueeze(4)
        # (B,B',T,L,MXM,D)
        # collapse width and height dims for image_feature for weighted summing via matrix mult
        image_feature_flat = torch.flatten(image_feature, start_dim=1, end_dim=2).permute(
            0, 2, 1, 3).unsqueeze(1).unsqueeze(1).expand(-1, batch_size, max_word_len, -1, -1, -1)

        image_feature_flat_shape = image_feature_flat.shape
        va_shape = image_feature_flat.shape[0:3] + image_feature_flat.shape[5:]

        image_feature_flat = torch.split(
            image_feature_flat, image_feature_flat.shape[3]//self.SPLIT_VA_ACROSS_GPU, dim=3)

        # 8, 8, 23, 4, 1024

        visual_word_attention = []

        # reshape for bmm

        for gpu_id in range(self.SPLIT_VA_ACROSS_GPU):
            device = 'cuda:{:d}'.format(gpu_id)
            # reshape heatmap so that dimension to sum across is at the end
            # (B,B' T, l, 1, MXM)

            visual_word_attention_l = torch.matmul(torch.flatten(
                similarity_heatmap[gpu_id], start_dim=2, end_dim=3).permute(0, 1, 3, 4, 2).unsqueeze(4), image_feature_flat[gpu_id].to(device)).squeeze(4)
            print(visual_word_attention_l.shape)
            visual_word_attention.append(visual_word_attention_l)

        # for gpu_id in range(self.SPLIT_VA_ACROSS_GPU):
        #     device = 'cuda:{:d}'.format(gpu_id)
        #     visual_word_attention_l = torch.ones(
        #         va_shape, device=device)
        #     for b in range(image_feature_flat_shape[0]):
        #         for b2 in range(image_feature_flat_shape[1]):
        #             for t in range(image_feature_flat_shape[2]):
        #                 visual_word_attention_l[b, b2, t, ...] = torch.bmm(torch.flatten(
        #                     similarity_heatmap[gpu_id], start_dim=2, end_dim=3).permute(0, 1, 3, 4, 2).unsqueeze(4)[b,
        #                                                                                                             b2, t, ...].to(device),
        #                     image_feature_flat[gpu_id][b, b2, t, ...].to(device)).squeeze(1)
        #     visual_word_attention.append(visual_word_attention_l)

        # for gpu_id in range(self.SPLIT_VA_ACROSS_GPU):
        #     device = 'cuda:{:d}'.format(gpu_id)
        #     # reshape heatmap so that dimension to sum across is at the end
        #     # (B,B' T, l, 1, MXM)

        #     visual_word_attention_l = torch.matmul(torch.flatten(
        #         similarity_heatmap[gpu_id], start_dim=2, end_dim=3).permute(0, 1, 3, 4, 2).unsqueeze(4), image_feature_flat[gpu_id].to(device)).squeeze(4)
        #     visual_word_attention.append(visual_word_attention_l)

        # visual_word_attention = torch.matmul(
        #     similarity_heatmap_flat, image_feature_flat).squeeze(4)

        # visual_word_attention = torch.utils.checkpoint.checkpoint(
        #     torch.matmul, similarity_heatmap_flat, image_feature_flat).squeeze(4)
        # print('visual word', visual_word_attention.shape)

        # (B,B',T,L,D)
        # visual_word_attention = torch.nn.functional.normalize(
        #     visual_word_attention, p=2, dim=4)

        # (B,B',T,L)
        word_image_pertinence_score = []

        if self.SPLIT_VA_ACROSS_GPU > 1:

            text_feature = text_feature.unsqueeze(
                1).expand(-1, batch_size, -1, -1)

            for gpu_id in range(self.SPLIT_VA_ACROSS_GPU):
                device = 'cuda:{:d}'.format(gpu_id)
                print('text_feat', text_feature.shape)
                print('vw attention', visual_word_attention[gpu_id].shape)
                word_image_pertinence_score.append(F.cosine_similarity(
                    text_feature.to(device), visual_word_attention[gpu_id].squeeze(3).to(device), dim=3))

            word_image_pertinence_score = torch.stack(
                [w.to('cuda') for w in word_image_pertinence_score], 3)

        else:
            text_feature = text_feature.unsqueeze(2).unsqueeze(
                1).expand(-1, batch_size, -1, self.L, -1)

            for gpu_id in range(self.SPLIT_VA_ACROSS_GPU):
                device = 'cuda:{:d}'.format(gpu_id)
                print('text_feat', text_feature.shape)
                print('vw attention', visual_word_attention[gpu_id].shape)
                word_image_pertinence_score.append(F.cosine_similarity(
                    text_feature.to(device), visual_word_attention[gpu_id].squeeze(3).to(device), dim=4))

            print('wroib4', word_image_pertinence_score[0].shape)
            word_image_pertinence_score = word_image_pertinence_score[0]

        # (B,B',T,)
        word_image_max_pertinence_score, _ = torch.max(
            word_image_pertinence_score, dim=3)

        return word_image_max_pertinence_score
