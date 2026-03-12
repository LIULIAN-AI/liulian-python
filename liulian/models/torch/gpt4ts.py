"""
GPT4TS (One Fits All): Power General Time Series Analysis by Pretrained LM

Paper: https://arxiv.org/abs/2302.11939
Reference Implementation: https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All

Fine-tunes only the LayerNorm and positional embedding of a frozen GPT-2
backbone for time series forecasting, imputation, anomaly detection, and
classification. Simpler than TimeLLM (no reprogramming or text prototypes)
— uses patching + linear projection into/out of the GPT-2 latent space.
"""

from os import PathLike
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from liulian.models.torch.layers.embed import DataEmbedding


class Model(nn.Module):
    """GPT4TS — frozen GPT-2 backbone with fine-tuned LayerNorm for time series.

    Paper: https://arxiv.org/abs/2302.11939

    The model:
    1. Patchifies the input time series with stride == patch_len (non-overlap).
    2. Projects each patch to GPT-2's hidden dimension via a linear layer.
    3. Passes through frozen GPT-2 layers (only LayerNorm params are trainable).
    4. Projects the concatenated output patches to the prediction length.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_ff = configs.d_ff

        self.patch_size = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'gpt4ts_stride', self.patch_size)
        self.gpt_layers = getattr(configs, 'gpt_layers', 6)
        self.d_model = getattr(configs, 'd_model', 768)

        self.cache_dir: Union[str, PathLike, None] = getattr(configs, 'cache_dir', None)

        # -- Load frozen GPT-2 backbone -----------------------------------
        from transformers import GPT2Config, GPT2Model

        gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
        gpt2_config.num_hidden_layers = self.gpt_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True

        try:
            self.gpt2 = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                local_files_only=True,
                config=gpt2_config,
            )
        except EnvironmentError:
            print('Local GPT-2 files not found. Downloading from HuggingFace...')
            self.gpt2 = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                local_files_only=False,
                config=gpt2_config,
            )

        self.gpt2_dim = self.gpt2.config.n_embd  # 768 for GPT-2 base

        # Freeze everything, then unfreeze LayerNorm + positional embedding
        for param in self.gpt2.parameters():
            param.requires_grad = False
        for layer in self.gpt2.h:
            for name, param in layer.named_parameters():
                if 'ln' in name:  # ln_1 and ln_2 (LayerNorm)
                    param.requires_grad = True
        # Also unfreeze final LayerNorm
        for param in self.gpt2.ln_f.parameters():
            param.requires_grad = True
        # Unfreeze positional embedding (wpe)
        self.gpt2.wpe.weight.requires_grad = True

        # -- Patch embedding: project each patch to GPT-2 dim ----
        self.num_patches = (self.seq_len - self.patch_size) // self.stride + 1
        # Instance normalization before patching
        self.ln_proj = nn.LayerNorm(self.d_ff)

        self.in_layer = nn.Linear(self.patch_size, self.gpt2_dim)

        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.out_layer = nn.Linear(
                self.gpt2_dim * self.num_patches, self.pred_len,
            )
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(self.gpt2_dim, self.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(self.gpt2_dim, self.seq_len)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.out_layer = nn.Linear(
                self.gpt2_dim * self.num_patches, configs.num_class,
            )

    def _patchify(self, x):
        """Convert [B, L, C] → [B*C, num_patches, patch_size]."""
        B, L, C = x.shape
        # Instance normalization (per-channel)
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdev

        # [B, L, C] → [B, C, L] → unfold → [B, C, num_patches, patch_size]
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # [B, C, num_patches, patch_size] → [B*C, num_patches, patch_size]
        x = x.reshape(B * C, self.num_patches, self.patch_size)
        return x, means, stdev, B, C

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x, means, stdev, B, C = self._patchify(x_enc)

        x = self.in_layer(x)               # [B*C, num_patches, gpt2_dim]
        outputs = self.gpt2(inputs_embeds=x).last_hidden_state

        # [B*C, num_patches, gpt2_dim] → [B*C, num_patches * gpt2_dim]
        outputs = outputs.reshape(B * C, -1)
        outputs = self.out_layer(outputs)   # [B*C, pred_len]
        outputs = outputs.reshape(B, C, -1).permute(0, 2, 1)  # [B, pred_len, C]

        # De-normalize
        outputs = outputs * stdev + means
        return outputs

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x, means, stdev, B, C = self._patchify(x_enc)

        x = self.in_layer(x)
        outputs = self.gpt2(inputs_embeds=x).last_hidden_state

        outputs = self.out_layer(outputs[:, -1:, :])  # [B*C, 1, seq_len]
        outputs = outputs.squeeze(1)                    # [B*C, seq_len]
        outputs = outputs.reshape(B, C, -1).permute(0, 2, 1)  # [B, L, C]

        outputs = outputs * stdev + means
        return outputs

    def anomaly_detection(self, x_enc):
        x, means, stdev, B, C = self._patchify(x_enc)

        x = self.in_layer(x)
        outputs = self.gpt2(inputs_embeds=x).last_hidden_state

        outputs = self.out_layer(outputs[:, -1:, :]).squeeze(1)
        outputs = outputs.reshape(B, C, -1).permute(0, 2, 1)

        outputs = outputs * stdev + means
        return outputs

    def classification(self, x_enc, x_mark_enc):
        x, _means, _stdev, B, C = self._patchify(x_enc)

        x = self.in_layer(x)
        outputs = self.gpt2(inputs_embeds=x).last_hidden_state

        # Pool per-channel, then average across channels
        outputs = outputs.reshape(B * C, -1)
        outputs = self.out_layer(outputs)   # [B*C, num_class]
        outputs = outputs.reshape(B, C, -1).mean(dim=1)  # [B, num_class]
        return outputs

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
