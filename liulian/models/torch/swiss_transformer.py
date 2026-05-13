"""General encoder-only Transformer models for the liulian framework.

Works with **any** dataset — traffic, ETT, weather, swiss-river, etc. —
with configurable output dimension (``c_out``) and full entity-identifier
support (embedding, one-hot, coordinates, sinusoidal, descriptors, or none).

Core models:

* :class:`SwissTransformerEmbeddingModel` — encoder-only Transformer with
  optional station/entity embedding, causal masking, multiple positional
  encoding modes (sinusoidal / learnable / RoPE), optional mask_embedding
  for missing values, and extrapolation via learnable future-step embeddings.
* :class:`SwissTransformerModel` — thin wrapper without embedding.
* :class:`TransformerEntityFeatureModel` — variant that concatenates
  pre-computed entity features at forward time.

Adapters mirror the entity-identifier modes from
:func:`~liulian.data.ts.timeseriesdataset.make_entity_features`:

* ``'none'``             — no entity features (default).
* ``'embedding'``        — learn ``nn.Embedding``; IDs from ``x_mark_enc``.
* ``'onehot'`` etc.      — already in ``x_enc``; transparent pass-through.
* ``'feature_concat'``   — separate ``entity_features`` tensor in batch.

Originally adapted from swiss-river-network-benchmark ``model.py``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from liulian.models.torch.base_adapter import TorchModelAdapter


# ======================================================================
# Positional encodings
# ======================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


# ======================================================================
# Entity-identifier constants (shared with swiss_lstm)
# ======================================================================

_TRANSPARENT_MODES = frozenset(
    {
        'onehot',
        'coordinates',
        'sinusoidal',
        'random',
        'descriptors',
        'numeric_id',
    }
)
_EMBEDDING_MODE = 'embedding'
_FEATURE_CONCAT_MODE = 'feature_concat'


# ======================================================================
# Transformer models
# ======================================================================


class SwissTransformerEmbeddingModel(nn.Module):
    """Encoder-only Transformer with optional entity embeddings.

    Supports three positional encoding modes:

    * ``"sinusoidal"`` — fixed Vaswani PE
    * ``"learnable"``  — learnable per-position vectors
    * ``"rope"``       — HuggingFace ``RoFormerModel`` (requires ``transformers``)

    Args:
        input_size: Number of input features per time step.
        num_embeddings: Vocabulary size for entity embedding (0 = none).
        embedding_size: Entity embedding dimension.
        num_heads: Attention heads.
        num_layers: Transformer encoder layers.
        dim_feedforward: FFN intermediate size.
        dropout: Dropout rate.
        d_model: Hidden dimension.
        ratio_heads_to_d_model: Fallback ratio when ``d_model`` is None.
        max_len: Max sequence length for positional encoding.
        positional_encoding: ``"sinusoidal"`` / ``"learnable"`` / ``"rope"``.
        missing_value_method: ``"mask_embedding"`` / ``None``.
        use_current_x: If *False*, use future-step embeddings (extrapolation).
        future_steps: Horizon (only when ``use_current_x=False``).
        d_future_emb: Future-step embedding dimension.
        c_out: Number of output channels (default ``1``).
    """

    def __init__(
        self,
        input_size: int,
        num_embeddings: int = 0,
        embedding_size: int = 10,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        d_model: Optional[int] = None,
        ratio_heads_to_d_model: int = 8,
        max_len: int = 500,
        positional_encoding: str = 'sinusoidal',
        missing_value_method: Optional[str] = None,
        use_current_x: bool = True,
        future_steps: int = 1,
        d_future_emb: int = 32,
        c_out: int = 1,
    ):
        super().__init__()
        self.use_current_x = use_current_x
        self.use_mask_embedding = missing_value_method == 'mask_embedding'
        self.positional_encoding = positional_encoding
        self.future_steps = future_steps

        # Entity embedding (integer-ID mode)
        self.embedding = (
            nn.Embedding(num_embeddings, embedding_size) if num_embeddings > 0 else None
        )

        # d_model resolution
        if d_model is None:
            d_model = int(num_heads * ratio_heads_to_d_model)
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        emb_dim = embedding_size if self.embedding else 0
        self.input_proj = nn.Linear(input_size + emb_dim, d_model)

        # Positional encoding
        self._rope_mode = False
        if positional_encoding == 'rope':
            try:
                from transformers import RoFormerModel, RoFormerConfig  # type: ignore

                cfg = RoFormerConfig(
                    vocab_size=1,
                    hidden_size=d_model,
                    num_attention_heads=num_heads,
                    num_hidden_layers=num_layers,
                    intermediate_size=dim_feedforward,
                    hidden_dropout_prob=dropout,
                    attention_probs_dropout_prob=dropout,
                    max_position_embeddings=max_len,
                    is_decoder=False,
                    rotary_value=False,
                )
                self.transformer = RoFormerModel(cfg)
                self._rope_mode = True
            except ImportError:
                positional_encoding = 'sinusoidal'
                self.positional_encoding = positional_encoding

        if not self._rope_mode:
            if positional_encoding == 'learnable':
                self.pos_embedding = LearnablePositionalEncoding(d_model, max_len)
            else:
                self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output head
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(d_model, c_out))

        # Mask embedding
        if self.use_mask_embedding:
            self.mask_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        # Extrapolation mode
        self._target_postprocessor = None
        if not use_current_x:
            self.future_step_embedding = nn.Parameter(
                torch.zeros(1, future_steps, d_future_emb)
            )
            proj_in = (
                (embedding_size + d_future_emb) if self.embedding else d_future_emb
            )
            self.future_proj = (
                nn.Identity() if proj_in == d_model else nn.Linear(proj_in, d_model)
            )
            self._target_postprocessor = lambda t: t[:, -future_steps:, :]

    def forward(
        self,
        e: Optional[torch.Tensor],
        x: torch.Tensor,
        time_masks: Optional[torch.Tensor] = None,
        pad_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            e: Entity IDs ``(B, seq_len)`` or *None*.
            x: Input features ``(B, seq_len, input_size)``.
            time_masks: Boolean missing-value mask ``(B, seq_len)``.
            pad_masks: Boolean padding mask ``(B, seq_len)``.
        """
        if self.use_current_x:
            if self.embedding is not None and e is not None:
                emb = self.embedding(e)
                x = torch.cat([emb, x], dim=-1)
            x = self.input_proj(x)
            seq_len = x.size(1)
        else:
            x_hist = x[:, : -self.future_steps, :]
            x_fut = self.future_step_embedding.expand(x.size(0), -1, -1)
            if self.embedding is not None and e is not None:
                emb = self.embedding(e)
                x_hist = torch.cat([emb[:, : -self.future_steps, :], x_hist], -1)
                x_fut = torch.cat([emb[:, -self.future_steps :, :], x_fut], -1)
            x_hist = self.input_proj(x_hist)
            x_fut = self.future_proj(x_fut)
            x = torch.cat([x_hist, x_fut], dim=1)
            seq_len = x.size(1)

        # Positional encoding
        if not self._rope_mode:
            x = self.pos_embedding(x)

        # Mask embedding for missing values
        if time_masks is not None and self.use_mask_embedding:
            x = x + time_masks.unsqueeze(-1) * self.mask_embedding

        # Causal / full mask
        if self.use_current_x:
            causal = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
        else:
            causal = torch.zeros(seq_len, seq_len, device=x.device).bool()

        # Transformer
        if self._rope_mode:
            hf_mask = (~causal).bool().unsqueeze(0)
            if time_masks is not None and not self.use_mask_embedding:
                hf_mask = hf_mask & (~time_masks).bool().unsqueeze(1)
            if pad_masks is not None:
                hf_mask = hf_mask & (~pad_masks).bool().unsqueeze(1)
            out = self.transformer(
                inputs_embeds=x, attention_mask=hf_mask
            ).last_hidden_state
        else:
            skp = None if self.use_mask_embedding else time_masks
            if pad_masks is not None:
                skp = pad_masks if skp is None else (skp | pad_masks)
            out = self.transformer(x, mask=causal, src_key_padding_mask=skp)

        target = self.linear(out)
        if self._target_postprocessor is not None:
            target = self._target_postprocessor(target)
        return target


class SwissTransformerModel(SwissTransformerEmbeddingModel):
    """Transformer **without** entity embedding — thin wrapper."""

    def __init__(self, **kwargs: Any):
        kwargs['num_embeddings'] = 0
        super().__init__(**kwargs)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        time_masks: Optional[torch.Tensor] = None,
        pad_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(None, x, time_masks, pad_masks)


class TransformerEntityFeatureModel(nn.Module):
    """Encoder-only Transformer that concatenates pre-computed entity features.

    For use with ``identifier_mode='feature_concat'``: the data layer provides
    entity features as a separate tensor that gets concatenated to ``x`` at
    forward time.

    Args:
        input_size: Raw feature dimension (excl. entity features).
        entity_dim: Dimension of the entity feature vector.
        num_heads, num_layers, dim_feedforward, dropout, d_model,
        max_len, positional_encoding, c_out: same as
        :class:`SwissTransformerEmbeddingModel`.
    """

    def __init__(
        self,
        input_size: int,
        entity_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        d_model: Optional[int] = None,
        max_len: int = 500,
        positional_encoding: str = 'sinusoidal',
        c_out: int = 1,
    ):
        super().__init__()
        if d_model is None:
            d_model = int(num_heads * 8)
        assert d_model % num_heads == 0

        self.input_proj = nn.Linear(input_size + entity_dim, d_model)

        if positional_encoding == 'learnable':
            self.pos_embedding = LearnablePositionalEncoding(d_model, max_len)
        else:
            self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(d_model, c_out))

    def forward(
        self,
        x: torch.Tensor,
        entity_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features ``(B, T, input_size)``.
            entity_features: Entity features ``(B, T, entity_dim)``.
        """
        x = torch.cat([x, entity_features], dim=-1)
        x = self.input_proj(x)
        x = self.pos_embedding(x)
        seq_len = x.size(1)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        out = self.transformer(x, mask=causal)
        return self.linear(out)


# ======================================================================
# Adapters
# ======================================================================


class _TransformerBaseAdapter(TorchModelAdapter):
    """Shared adapter for Transformer variants.

    Entity handling mirrors :class:`~liulian.models.torch.swiss_lstm._LSTMBaseAdapter`.
    """

    _entity_mode: str = 'none'
    _entity_id_col: int = 0

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self._entity_mode = config.get('identifier_mode', 'none')
        self._entity_id_col = config.get('entity_id_col', 0)

    def _forward_torch_model(
        self, torch_batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        import numpy as np

        def _to(v: Any) -> Optional[torch.Tensor]:
            if v is None:
                return None
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            return v.float().to(self.device) if isinstance(v, torch.Tensor) else v

        def _get(*keys: str) -> Any:
            for k in keys:
                v = torch_batch.get(k)
                if v is not None:
                    return v
            return None

        x_enc = _to(_get('x_enc', 'X'))
        mode = self._entity_mode

        if mode == _EMBEDDING_MODE:
            x_mark = _to(_get('x_mark_enc', 'X_mark'))
            if x_mark is not None and x_mark.ndim == 3:
                e = x_mark[:, :, self._entity_id_col].long()
            else:
                e = torch.zeros(
                    x_enc.size(0),
                    x_enc.size(1),
                    dtype=torch.long,
                    device=self.device,
                )
            output = self._model(e, x_enc)

        elif mode == _FEATURE_CONCAT_MODE:
            ef = _to(_get('entity_features'))
            if ef is None:
                output = self._model(x_enc)
            else:
                output = self._model(x_enc, ef)

        else:
            # 'none' or transparent modes
            output = self._model(x_enc)

        if isinstance(output, tuple):
            output = output[0]
        return {'predictions': output}


class TransformerEncoderAdapter(_TransformerBaseAdapter):
    """General-purpose encoder-only Transformer adapter.

    Config keys:
        enc_in (int): Input feature dimension.
        c_out (int): Output channels (default ``1``).
        n_heads (int): Attention heads (default ``4``).
        e_layers (int): Encoder layers (default ``2``).
        d_ff (int): FFN dimension (default ``128``).
        d_model (int | None): Hidden dimension.
        dropout (float): Dropout (default ``0.1``).
        positional_encoding (str): ``'sinusoidal'``, ``'learnable'``,
            or ``'rope'``.
        use_current_x (bool): Nowcast mode (default ``True``).
        pred_len / future_steps (int): Prediction horizon.
        identifier_mode (str): Entity identifier mode (default ``'none'``).
        entity_id_col (int): Column in x_mark_enc for embedding IDs.
        entity_dim (int): Entity feature dim for ``feature_concat`` mode.
        num_embeddings (int): Vocab for ``embedding`` mode.
        embedding_size (int): Embedding dim for ``embedding`` mode.

    Examples::

        # Traffic dataset (862 channels)
        TransformerEncoderAdapter(
            {
                'enc_in': 862,
                'c_out': 862,
                'n_heads': 8,
                'e_layers': 3,
                'd_model': 64,
            }
        )

        # Swiss-river with station embedding
        TransformerEncoderAdapter(
            {
                'enc_in': 1,
                'c_out': 1,
                'identifier_mode': 'embedding',
                'num_embeddings': 50,
                'embedding_size': 10,
            }
        )
    """

    def __init__(self, config: Dict[str, Any]):
        mode = config.get('identifier_mode', 'none')
        enc_in = config.get('enc_in', 1)
        c_out = config.get('c_out', 1)
        n_heads = config.get('n_heads', 4)
        n_layers = config.get('e_layers', 2)
        d_ff = config.get('d_ff', 128)
        dropout = config.get('dropout', 0.1)
        d_model = config.get('d_model', None)
        pe = config.get('positional_encoding', 'sinusoidal')
        use_current = config.get('use_current_x', True)
        future = config.get('future_steps', config.get('pred_len', 7))

        if mode == _EMBEDDING_MODE:
            model = SwissTransformerEmbeddingModel(
                input_size=enc_in,
                num_embeddings=config.get('num_embeddings', 50),
                embedding_size=config.get('embedding_size', 10),
                num_heads=n_heads,
                num_layers=n_layers,
                dim_feedforward=d_ff,
                dropout=dropout,
                d_model=d_model,
                positional_encoding=pe,
                missing_value_method=config.get('missing_value_method'),
                use_current_x=use_current,
                future_steps=future,
                d_future_emb=config.get('d_future_emb', 32),
                c_out=c_out,
            )
        elif mode == _FEATURE_CONCAT_MODE:
            entity_dim = config.get('entity_dim', 16)
            model = TransformerEntityFeatureModel(
                input_size=enc_in,
                entity_dim=entity_dim,
                num_heads=n_heads,
                num_layers=n_layers,
                dim_feedforward=d_ff,
                dropout=dropout,
                d_model=d_model,
                positional_encoding=pe,
                c_out=c_out,
            )
        else:
            model = SwissTransformerModel(
                input_size=enc_in,
                num_heads=n_heads,
                num_layers=n_layers,
                dim_feedforward=d_ff,
                dropout=dropout,
                d_model=d_model,
                positional_encoding=pe,
                use_current_x=use_current,
                future_steps=future,
                c_out=c_out,
            )

        full_cfg = dict(config)
        full_cfg.setdefault('identifier_mode', mode)
        super().__init__(model, full_cfg)


# ======================================================================
# Backward-compatible aliases
# ======================================================================

SwissTransformerAdapter = TransformerEncoderAdapter


class SwissTransformerEmbeddingAdapter(TransformerEncoderAdapter):
    """Backward-compatible alias — forces ``identifier_mode='embedding'``."""

    def __init__(self, config: Dict[str, Any]):
        cfg = dict(config)
        cfg.setdefault('identifier_mode', 'embedding')
        super().__init__(cfg)
