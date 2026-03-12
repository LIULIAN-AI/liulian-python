"""
Non-stationary Transformer: Exploring the Stationarity in Time Series Forecasting

Paper: https://openreview.net/pdf?id=ucNDIDRNjjv
Original Implementation: Time-Series-Library
https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py

Uses De-stationary Attention with learned tau (scaling) and delta (shift)
factors to handle non-stationary time series. A lightweight MLP-based
``Projector`` module learns these de-stationary factors from the raw input
statistics (mean and standard deviation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from liulian.models.torch.layers.transformer_blocks import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)
from liulian.models.torch.layers.attention import DSAttention, AttentionLayer
from liulian.models.torch.layers.embed import DataEmbedding


class Projector(nn.Module):
    """MLP to learn the De-stationary factors (tau and delta).

    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len, out_channels=1,
            kernel_size=kernel_size, padding=padding,
            padding_mode='circular', bias=False,
        )

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)        # B x 2E
        y = self.backbone(x)              # B x O
        return y


class Model(nn.Module):
    """Non-stationary Transformer with De-stationary Attention.

    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq,
            configs.dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Decoder (forecast tasks only)
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.dec_embedding = DataEmbedding(
                configs.dec_in, configs.d_model, configs.embed, configs.freq,
                configs.dropout,
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            DSAttention(
                                True, configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model, configs.n_heads,
                        ),
                        AttentionLayer(
                            DSAttention(
                                False, configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model, configs.n_heads,
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for _l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class,
            )

        # De-stationary factor learners
        self.tau_learner = Projector(
            enc_in=configs.enc_in, seq_len=configs.seq_len,
            hidden_dims=configs.p_hidden_dims,
            hidden_layers=configs.p_hidden_layers,
            output_dim=1,
        )
        self.delta_learner = Projector(
            enc_in=configs.enc_in, seq_len=configs.seq_len,
            hidden_dims=configs.p_hidden_dims,
            hidden_layers=configs.p_hidden_layers,
            output_dim=configs.seq_len,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        # Learned de-stationary factors
        tau = self.tau_learner(x_raw, std_enc)
        tau = torch.clamp(tau, max=80.0).exp()
        delta = self.delta_learner(x_raw, mean_enc)

        x_dec_new = torch.cat(
            [x_enc[:, -self.label_len:, :],
             torch.zeros_like(x_dec[:, -self.pred_len:, :])],
            dim=1,
        ).to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_raw = x_enc.clone().detach()

        mean_enc = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        mean_enc = mean_enc.unsqueeze(1).detach()
        x_enc = x_enc - mean_enc
        x_enc = x_enc.masked_fill(mask == 0, 0)
        std_enc = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        std_enc = std_enc.unsqueeze(1).detach()
        x_enc /= std_enc

        tau = self.tau_learner(x_raw, std_enc)
        tau = torch.clamp(tau, max=80.0).exp()
        delta = self.delta_learner(x_raw, mean_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.projection(enc_out)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def anomaly_detection(self, x_enc):
        x_raw = x_enc.clone().detach()

        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        tau = self.tau_learner(x_raw, std_enc)
        tau = torch.clamp(tau, max=80.0).exp()
        delta = self.delta_learner(x_raw, mean_enc)

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.projection(enc_out)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_raw = x_enc.clone().detach()

        mean_enc = x_enc.mean(1, keepdim=True).detach()
        std_enc = torch.sqrt(
            torch.var(x_enc - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()

        tau = self.tau_learner(x_raw, std_enc)
        tau = torch.clamp(tau, max=80.0).exp()
        delta = self.delta_learner(x_raw, mean_enc)

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
