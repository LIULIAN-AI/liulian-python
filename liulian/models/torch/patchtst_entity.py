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

Set ``model: patchtst``, ``identifier_mode: patch_embedding`` in the
experiment config.  The pipeline will import this ``Model`` instead of
the base ``patchtst.Model`` and will **not** wrap with
``ChannelEntityWrapper``.

To compare approaches:

* ``model: patchtst``, ``identifier_mode: none``
  → plain PatchTST (no entity info)
* ``model: patchtst``, ``identifier_mode: embedding``
  → PatchTST + time-step embedding (ChannelEntityWrapper — known to be
  ineffective due to instance normalisation)
* ``model: patchtst``, ``identifier_mode: patch_embedding``
  → PatchTST + patch-level entity embedding (this module)
"""

import torch
from torch import nn

from liulian.models.torch.patchtst import Model as PatchTSTModel


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
