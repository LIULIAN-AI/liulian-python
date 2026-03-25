from .base_adapter import TorchModelAdapter
import torch
import torch.nn as nn
from .layers.embed import DataEmbedding
from .layers.etsformer_blocks import EncoderLayer, Encoder, DecoderLayer, Decoder, Transform


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2202.01381
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        assert configs.e_layers == configs.d_layers, "Encoder and decoder layers must be equal"

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, getattr(configs, 'embed', 'timeF'), getattr(configs, 'freq', 'h'), configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.enc_in, configs.seq_len, self.pred_len, getattr(configs, 'top_k', 5),
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=getattr(configs, 'activation', 'sigmoid'),
                ) for _ in range(configs.e_layers)
            ]
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, self.pred_len,
                    dropout=configs.dropout,
                ) for _ in range(configs.d_layers)
            ],
        )
        self.transform = Transform(sigma=0.2)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None

class ETSformerAdapter(TorchModelAdapter):
    def _build_model(self) -> nn.Module:
        return Model(self.config)
