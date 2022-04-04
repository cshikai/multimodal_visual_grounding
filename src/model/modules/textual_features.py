from typing import Dict, List,  Tuple

import torch
import pytorch_lightning as pl


class TextualFeatures(pl.LightningModule):
    """
    """

    def __init__(self, cfg: Dict) -> None:
        """
        """
        super().__init__()

        # self.input_dim = cfg['model']['textual']['lstm']['input_dims']
        # # self.hidden_dim = cfg['model']['feature_hidden_dimension']
        # self.hidden_dim = 128
        # self.hidden_dim1 = 1024
        # assert self.hidden_dim % 2 == 0
        # self.lstm_hidden_dim = self.hidden_dim//2
        # self.num_layers = cfg['model']['textual']['lstm']['num_layers']
        # self.dropout = cfg['model']['textual']['lstm']['dropout']
        # self.relu_alpha = cfg['model']['leaky_relu_alpha']
        # self.lstm_1 = torch.nn.LSTM(self.input_dim, self.lstm_hidden_dim,
        #                             num_layers=1, dropout=self.dropout, bidirectional=True, batch_first=True,)
        # self.lstm_2 = torch.nn.LSTM(self.hidden_dim, self.lstm_hidden_dim,
        #                             num_layers=1, dropout=self.dropout, bidirectional=True, batch_first=True,)

        # self.leaky_relu = torch.nn.LeakyReLU(self.relu_alpha, inplace=True)
        # self.sentence_fc = torch.nn.Sequential(torch.nn.Linear(
        #     self.hidden_dim, self.hidden_dim), self.leaky_relu, torch.nn.Linear(self.hidden_dim, self.hidden_dim1), self.leaky_relu)
        # self.word_fc = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, self.hidden_dim), self.leaky_relu, torch.nn.Linear(
        #     self.hidden_dim, self.hidden_dim1), self.leaky_relu)

        # self.word_linear_comb = torch.nn.Linear(3, 1)
        # self.sentence_linear_comb = torch.nn.Linear(2, 1)

        # self.word_pre_fc = torch.nn.Linear(1024, self.hidden_dim)

    def forward(self, x: torch.Tensor, seq_len: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        """
        # packed_x = torch.nn.utils.rnn.pack_padded_sequence(
        #     x, seq_len, batch_first=True)
        # packed_output_1, (hidden_1, cell_1) = self.lstm_1(packed_x)
        # packed_output_2, (hidden_2, cell_2) = self.lstm_2(packed_output_1)

        # output_1, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #     packed_output_1, batch_first=True)
        # output_2, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #     packed_output_2, batch_first=True)

        # word_feature = self.word_linear_comb(
        #     torch.stack([self.word_pre_fc(x), output_1, output_2, ], -1)).squeeze(-1)
        # word_feature = self.word_fc(word_feature)

        # # start embedding taken from the backward lstm
        # # sentence_features_backwards = [output_1[:,0,self.hidden_dim:],output_1[:,0,self.hidden_dim:]],-1)

        # # end of sentence embedding taken from the forward lstm
        # sentence_end_1 = []
        # sentence_end_2 = []
        # for i, end in enumerate(seq_len):
        #     sentence_end_1.append(output_1[i, end-1, :self.lstm_hidden_dim])
        #     sentence_end_2.append(output_2[i, end-1, :self.lstm_hidden_dim])
        # # sentence_features_forward = torch.cat([torch.stack(sentence_end_1,0),torch.stack(sentence_end_1,0)],-1)

        # first_layer_sentence_feature = torch.cat(
        #     [torch.stack(sentence_end_1, 0), output_1[:, 0, self.lstm_hidden_dim:]], -1)
        # second_layer_sentence_feature = torch.cat(
        #     [torch.stack(sentence_end_2, 0), output_2[:, 0, self.lstm_hidden_dim:]], -1)
        # sentence_feature = self.sentence_linear_comb(torch.stack(
        #     [first_layer_sentence_feature, second_layer_sentence_feature], -1)).squeeze(-1)
        # sentence_feature = self.sentence_fc(sentence_feature)

        # print('word_feature', word_feature.shape)
        # print('sentence_feature', sentence_feature.shape)

        return torch.ones(x.shape[0], x.shape[1], 1024).cuda(), torch.ones(x.shape[0], 1024).cuda()
        # return word_feature, sentence_feature
