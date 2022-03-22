import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

from .context_combination import ContextCombination


class LSTMDecoder(pl.LightningModule):
    """
    Decoder Object

    Note that the forward method iterates through all time steps
    It expects an input of [batch]
    """

    def __init__(self, lstm_hid_dim, n_layers, lstm_dropout, decoder_trans_output_dim,
                 dim_final_feedforward_1,
                 dim_final_feedforward_2,
                 dim_final_feedforward_3,
                 attention_dim,
                 mode3_encoder,
                 callsign_encoder):
        """
        Initialises the decoder object.
        :param output_dim: number of classes to predict
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param dropout: dropout ratio for decoder
        :param attention: attention object to used (initialized in seq2seq)
        """
        super().__init__()
        # self.output_dim = output_dim
        # self.hid_dim = hid_dim
        # self.n_layers = n_layers
        # self.dropout = nn.Dropout(dropout)
        # self.attention = attention
        # self.fc_out = nn.Linear(
        #     hid_dim*2 + hid_dim + output_dim + hid_dim + hid_dim, output_dim).double()

        self.mode3_encoder = mode3_encoder
        self.callsign_encoder = callsign_encoder

        self.decoder_start_token = 2
        lstm_input_dim = decoder_trans_output_dim + \
            self.mode3_encoder.id_embed_dim + self.callsign_encoder.id_embed_dim + 1

        # includes output from previous ts

        context_input_dim = lstm_hid_dim
        self.lstm_hid_dim = lstm_hid_dim
        self.num_lstm_layers = n_layers

        self.lat_lstm = nn.LSTM(lstm_input_dim, lstm_hid_dim,
                                n_layers, dropout=lstm_dropout, batch_first=True).double()

        self.lon_lstm = nn.LSTM(lstm_input_dim, lstm_hid_dim,
                                n_layers, dropout=lstm_dropout, batch_first=True).double()

        self.alt_lstm = nn.LSTM(lstm_input_dim, lstm_hid_dim,
                                n_layers, dropout=lstm_dropout, batch_first=True).double()

        self.lat_cc = ContextCombination(context_input_dim,
                                         dim_final_feedforward_1,
                                         dim_final_feedforward_2,
                                         dim_final_feedforward_3,
                                         attention_dim)
        self.lon_cc = ContextCombination(context_input_dim,
                                         dim_final_feedforward_1,
                                         dim_final_feedforward_2,
                                         dim_final_feedforward_3,
                                         attention_dim)
        self.alt_cc = ContextCombination(context_input_dim,
                                         dim_final_feedforward_1,
                                         dim_final_feedforward_2,
                                         dim_final_feedforward_3,
                                         attention_dim)

    def forward(self, x_raw, decoder_trans_output, mode3_input, callsign_input,  seq_lens):
        """
        Forward propagation.
        :param input: label of dataset at each timestamp (tensor) [batch_size]
        :param hidden_cell: hidden state from previous timestamp (tensor) ([batch_size,n_layer,hid_dim],[batch_size,n_layer,hid_dim])
        :param encoder_output: used to measure similiarty in states in attention [batch_size,sequence_len,hid_dim]
        :param mask: mask to filter out the paddings in attention object [batch_size,sequence_len]
        :return: normalized output probabilities for each timestamp - softmax (tensor) [batch_size,sequence_len,num_outputs]
        """

        mode3_embedding = self.mode3_encoder(mode3_input)
        callsign_embedding = self.callsign_encoder(callsign_input)

        if len(decoder_trans_output.shape) == 2:
            partial_lstm_input = torch.cat((decoder_trans_output, mode3_embedding,
                                            callsign_embedding), dim=1)
        else:
            seq_len = decoder_trans_output.shape[1]

            partial_lstm_input = torch.cat((decoder_trans_output, mode3_embedding.unsqueeze(1).repeat(1, seq_len, 1),
                                            callsign_embedding.unsqueeze(1).repeat(1, seq_len, 1)), dim=2)

        batch_size = x_raw.shape[0]
        max_len = x_raw.shape[1]
        mask = self.get_len_mask(batch_size, max_len, seq_lens).cuda()
        predictions = torch.zeros(
            [batch_size, max_len, 3, ], requires_grad=True).to('cuda')

        previous_decoder_output = (torch.ones(
            [batch_size, 1, 3], dtype=torch.long) * self.decoder_start_token).to('cuda')

        lat_hidden_cell = (torch.zeros([self.num_lstm_layers,
                                        batch_size, self.lstm_hid_dim], requires_grad=True).double().to('cuda'), torch.zeros([self.num_lstm_layers,
                                                                                                                              batch_size, self.lstm_hid_dim], requires_grad=True).double().to('cuda'))
        lon_hidden_cell = (torch.zeros([self.num_lstm_layers,
                                        batch_size, self.lstm_hid_dim], requires_grad=True).double().to('cuda'), torch.zeros([self.num_lstm_layers,
                                                                                                                              batch_size, self.lstm_hid_dim], requires_grad=True).double().to('cuda'))
        alt_hidden_cell = (torch.zeros([self.num_lstm_layers,
                                        batch_size, self.lstm_hid_dim], requires_grad=True).double().to('cuda'), torch.zeros([self.num_lstm_layers,
                                                                                                                              batch_size, self.lstm_hid_dim], requires_grad=True).double().to('cuda'))

        for i in range(max_len):
            lat_lstm_input = torch.cat(
                (partial_lstm_input[:, i:i+1, :], previous_decoder_output[:, :, 0:1]), dim=2)
            lon_lstm_input = torch.cat(
                (partial_lstm_input[:, i:i+1, :], previous_decoder_output[:, :, 1:2]), dim=2)
            alt_lstm_input = torch.cat(
                (partial_lstm_input[:, i:i+1, :], previous_decoder_output[:, :, 2:3]), dim=2)

            lat_t_lstm_output, lat_hidden_cell = self.lat_lstm(
                lat_lstm_input,  lat_hidden_cell)
            lon_t_lstm_output, lon_hidden_cell = self.lon_lstm(
                lon_lstm_input,  lon_hidden_cell)
            alt_t_lstm_output, alt_hidden_cell = self.alt_lstm(
                alt_lstm_input,  alt_hidden_cell)

            previous_decoder_output[:, :, 0:1] = self.lat_cc(
                x_raw[:, :, 0:1], lat_t_lstm_output, mask[:, i:i+1, :])
            previous_decoder_output[:, :, 1:2] = self.lon_cc(
                x_raw[:, :, 1:2], lon_t_lstm_output, mask[:, i:i+1, :])

            previous_decoder_output[:, :, 2:3] = self.alt_cc(
                x_raw[:, :, 2:3], alt_t_lstm_output, mask[:, i:i+1, :])
            predictions[:, i, :] = torch.clone(
                previous_decoder_output[:, 0, :])

        return predictions

    def get_len_mask(self, batch_size, max_len, seq_lens):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        block = torch.ones(batch_size, max_len, max_len)
        for i in range(batch_size):
            seq_len = seq_lens[i]
            block[i, :seq_len, :seq_len] = torch.zeros(seq_len, seq_len)
        return block.bool()
