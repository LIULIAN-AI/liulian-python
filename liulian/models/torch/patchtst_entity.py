"""PatchTST with patch-level entity (station/channel) embeddings.

This model addresses the architectural incompatibility between PatchTST's
instance normalisation and the time-step-level entity embeddings used by
:class:`ChannelEntityWrapper`.

**Problem with time-step-level embedding:**
PatchTST's ``forecast()`` applies RevIN-style instance normalisation
(mean-subtraction + variance-scaling) before patching.  The existing
``ChannelEntityWrapper`` injects a constant per-station bias at the
time-step level — but mean-subtraction removes this constant offset
entirely, rendering the entity signal useless and introducing noise
through the projection layer.

**Solution — patch-level additive embedding:**
This model adds entity embeddings *after* patching and patch projection,
directly in ``d_model`` space — the same representation level as
positional encoding:

.. code-block:: text

    x_enc (B, T, N)
      ▼ instance normalisation (means, stdev)
      ▼ permute → (B, N, T)
      ▼ PatchEmbedding: pad → unfold → Linear(patch_len→d_model) + pos_emb
    enc_out (B*N, P, d_model)         ← patch tokens
      ▼ + entity_embedding(channel_ids)   ← (B*N, 1, d_model) broadcast over P
    enc_out (B*N, P, d_model)         ← entity-aware patch tokens
      ▼ Transformer Encoder
      ▼ FlattenHead → De-norm

The entity embedding survives because it is injected *after* normalisation
and patching.  It acts as a learned per-station bias in the transformer's
representational space, analogous to positional encoding.

**References:**

* Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
  with Transformers", ICLR 2023. (PatchTST)
* Vaswani et al., "Attention is All You Need", NeurIPS 2017.
  (additive positional encoding pattern)
* Swiss River Network Benchmark — station embeddings fused before input
  projection in Transformer models (no instance norm applied).

**Usage:**

Set ``model: patchtst_entity`` in the experiment config.  Entity
embeddings are always active (the model is purpose-built for multi-channel
station differentiation).  Use ``identifier_mode: none`` — the framework's
external wrappers are not needed.

To compare approaches:

* ``model: patchtst``, ``identifier_mode: none``
  → plain PatchTST (no entity info)
* ``model: patchtst``, ``identifier_mode: embedding``
  → PatchTST + time-step embedding (ChannelEntityWrapper — known to be
  ineffective due to instance normalisation)
* ``model: patchtst_entity``, ``identifier_mode: none``
  → PatchTST + patch-level entity embedding (this model)
"""

import torch
from torch import nn
from typing import Dict, Any

from liulian.models.torch.patchtst import Model as PatchTSTModel
from liulian.models.torch.base_adapter import TorchModelAdapter


class Model(PatchTSTModel):
    """PatchTST with patch-level entity embeddings.

    Inherits the full PatchTST architecture (patch embedding, transformer
    encoder, flatten head) and overrides the forward methods to inject
    per-channel entity embeddings after patching.

    The entity embedding is an ``nn.Embedding(enc_in, d_model)`` table
    that maps channel index → d_model vector, added to every patch token
    for that channel.  This is structurally identical to how positional
    encoding works in PatchTST.

    Parameters
    ----------
    configs : SimpleNamespace
        Same as :class:`PatchTSTModel` plus:
        - ``enc_in``: Number of channels/stations (used as vocabulary size).
        - ``d_model``: Model dimension (embedding lives in this space).
    patch_len : int
        Patch length (default: 16).
    stride : int
        Stride for patching (default: 8).
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__(configs, patch_len=patch_len, stride=stride)
        num_stations = getattr(configs, 'enc_in', 28)
        self.entity_embedding = nn.Embedding(num_stations, configs.d_model)

    # ------------------------------------------------------------------
    # Entity injection helper
    # ------------------------------------------------------------------

    def _inject_entity(
        self,
        enc_out: torch.Tensor,
        n_vars: int,
    ) -> torch.Tensor:
        """Add per-channel entity embeddings to patch representations.

        Parameters
        ----------
        enc_out : torch.Tensor
            Patch token tensor of shape ``(B*N, P, d_model)``.
        n_vars : int
            Number of channels/stations ``N``.

        Returns
        -------
        torch.Tensor
            ``(B*N, P, d_model)`` with entity embeddings added.
        """
        B = enc_out.shape[0] // n_vars
        # Channel indices: [0, 1, ..., N-1] repeated B times
        ids = torch.arange(n_vars, device=enc_out.device).repeat(B)  # (B*N,)
        emb = self.entity_embedding(ids)  # (B*N, d_model)
        # Broadcast over all patches: (B*N, 1, d_model)
        return enc_out + emb.unsqueeze(1)

    # ------------------------------------------------------------------
    # Overridden forward methods with entity injection
    # ------------------------------------------------------------------

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # ── Inject entity embeddings at patch level ──
        enc_out = self._inject_entity(enc_out, n_vars)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # ── Inject entity embeddings at patch level ──
        enc_out = self._inject_entity(enc_out, n_vars)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # ── Inject entity embeddings at patch level ──
        enc_out = self._inject_entity(enc_out, n_vars)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        return dec_out

    # classification is inherited — no entity injection needed
    # (classification is not used for multi-channel stations)


class PatchTSTEntityAdapter(TorchModelAdapter):
    """Adapter for PatchTST with patch-level entity embeddings.

    This adapter does NOT use :class:`EntityAwareMixin` because the entity
    embedding is handled internally by the model at the patch level.
    ``identifier_mode`` should be ``'none'`` to prevent the pipeline from
    double-wrapping with ``ChannelEntityWrapper``.

    Expected config parameters:
        - seq_len: Input sequence length
        - pred_len: Prediction sequence length
        - enc_in: Number of input features/channels/stations
        - d_model: Model dimension (default: 128)
        - n_heads: Number of attention heads (default: 16)
        - e_layers: Number of encoder layers (default: 3)
        - d_ff: Feed-forward dimension (default: 256)
        - dropout: Dropout rate (default: 0.2)
        - activation: Activation function (default: 'gelu')
        - factor: Attention factor (default: 1)
        - patch_len: Patch length (default: 16)
        - stride: Patch stride (default: 8)
        - task_name: Task type (default: 'long_term_forecast')
    """

    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'activation': 'gelu',
            'factor': 1,
            'patch_len': 16,
            'stride': 8,
            'task_name': 'long_term_forecast',
        }
        default_config.update(config)

        model = Model(
            self._dict_to_namespace(default_config),
            patch_len=default_config['patch_len'],
            stride=default_config['stride'],
        )
        super().__init__(model, default_config)
        # No _init_entity_support — entity embedding is internal

    def _prepare_model_inputs(self, inputs: Dict[str, torch.Tensor]) -> tuple:
        """Prepare inputs for PatchTST forward pass."""
        x_enc = inputs['x_enc']
        batch_size, seq_len, n_features = x_enc.shape

        x_mark_enc = inputs.get(
            'x_mark_enc',
            torch.zeros(batch_size, seq_len, 1, device=x_enc.device),
        )
        x_dec = inputs.get(
            'x_dec',
            torch.zeros(
                batch_size,
                self.config['pred_len'],
                n_features,
                device=x_enc.device,
            ),
        )
        x_mark_dec = inputs.get(
            'x_mark_dec',
            torch.zeros(
                batch_size, self.config['pred_len'], 1, device=x_enc.device
            ),
        )

        return (x_enc, x_mark_enc, x_dec, x_mark_dec)
