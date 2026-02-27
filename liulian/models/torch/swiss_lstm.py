"""General LSTM models for the liulian framework.

Provides flexible LSTM model variants that work with **any** dataset —
traffic, ETT, weather, swiss-river, etc. — with configurable output
dimension and entity-identifier support.

Core models:

* :class:`SwissLstmModel` — basic LSTM with configurable output head.
* :class:`ExtrapoLstmModelLIMO` — Last-Input-Multiple-Output extrapolation.
* :class:`ExtrapoLstmModelFEmbed` — learnable future-step embeddings.
* :class:`LstmEmbeddingModel` — LSTM with integrated ``nn.Embedding`` for
  entity/station identifiers supplied as integer IDs.

Adapters handle the mapping from liulian's batch format to model inputs
and support **all** entity identifier modes defined in
:func:`~liulian.data.ts.timeseriesdataset.make_entity_features`:

* ``none``          — no entity features (default).
* ``embedding``     — learn ``nn.Embedding`` inside the model; integer IDs
  extracted from ``x_mark_enc``.
* ``onehot`` / ``coordinates`` / ``sinusoidal`` / ``descriptors`` /
  ``numeric_id`` — pre-computed entity features already concatenated into
  ``x_enc`` by the data layer; the adapter passes them through transparently
  (make sure ``enc_in`` accounts for the extra dimensions).
* ``feature_concat`` — separate entity feature tensor provided under
  ``entity_features`` key in the batch dict; concatenated to ``x_enc`` at
  forward time (``enc_in`` should be raw feature count only — the adapter
  auto-adjusts ``input_size``).

Originally adapted from swiss-river-network-benchmark ``model.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from liulian.models.torch.base_adapter import TorchModelAdapter


# ======================================================================
# Core LSTM models
# ======================================================================


class SwissLstmModel(nn.Module):
    """Basic LSTM → Linear forecaster.

    Args:
        input_size: Number of input features per time step.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        c_out: Number of output channels (default ``1``).
        dropout: LSTM dropout between layers (applied when ``num_layers > 1``).

    Input:  ``(B, seq_len, input_size)``
    Output: ``(B, seq_len, c_out)``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        c_out: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.linear(out)


class ExtrapoLstmModelLIMO(nn.Module):
    """LSTM extrapolation — Last Input Multiple Output.

    The last hidden state is projected to ``future_steps * c_out`` outputs.

    Args:
        input_size: Number of input features.
        hidden_size: LSTM hidden dimension.
        num_layers: Stacked LSTM layers.
        future_steps: Prediction horizon.
        c_out: Output channels per time step (default ``1``).
        dropout: LSTM dropout.

    Input:  ``(B, seq_len, input_size)``  (includes ``future_steps`` padding)
    Output: ``(B, future_steps, c_out)``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        future_steps: int = 1,
        c_out: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.c_out = c_out
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, future_steps * c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, : -self.future_steps]
        out, _ = self.lstm(x)
        target = self.linear(out[:, -1, :])  # (B, future_steps * c_out)
        return target.view(-1, self.future_steps, self.c_out)


class ExtrapoLstmModelFEmbed(nn.Module):
    """LSTM extrapolation with learnable future-step embeddings.

    Args:
        input_size: Number of input features.
        hidden_size: LSTM hidden dimension.
        num_layers: Stacked LSTM layers.
        future_steps: Prediction horizon.
        d_future_emb: Future-step embedding dimension.
        c_out: Output channels per time step (default ``1``).
        dropout: LSTM dropout.

    Input:  ``(B, seq_len, input_size)``  (includes ``future_steps`` padding)
    Output: ``(B, future_steps, c_out)``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        future_steps: int = 1,
        d_future_emb: int = 32,
        c_out: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.input_proj = (
            nn.Linear(input_size, d_future_emb)
            if input_size != d_future_emb
            else nn.Identity()
        )
        self.future_step_embedding = nn.Parameter(
            torch.zeros(1, future_steps, d_future_emb)
        )
        self.lstm = nn.LSTM(
            input_size=d_future_emb,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hist = x[:, : -self.future_steps, :]
        x_hist = self.input_proj(x_hist)
        x_fut = self.future_step_embedding.expand(x.size(0), -1, -1)
        x_cat = torch.cat([x_hist, x_fut], dim=1)
        out, _ = self.lstm(x_cat)
        target = self.linear(out)
        return target[:, -self.future_steps :, :]  # (B, future_steps, c_out)


class LstmEmbeddingModel(nn.Module):
    """LSTM with entity/station embedding (``nn.Embedding``).

    For use when entity identifiers are supplied as **integer IDs**
    (``identifier_mode='embedding'``).

    Args:
        input_size: Number of raw input features per time step.
        num_embeddings: Vocabulary size for entity embedding.
        embedding_size: Entity embedding dimension.
        hidden_size: LSTM hidden dimension.
        num_layers: Stacked LSTM layers.
        c_out: Output channels (default ``1``).
        dropout: LSTM dropout.

    Forward:
        e: ``(B, seq_len)`` — integer entity IDs.
        x: ``(B, seq_len, input_size)``.
    Output: ``(B, seq_len, c_out)``
    """

    def __init__(
        self,
        input_size: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        c_out: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(
            input_size=input_size + embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, c_out))

    def forward(self, e: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(e)
        x = torch.cat([emb, x], dim=2)
        out, _ = self.lstm(x)
        return self.linear(out)


class LstmEntityFeatureModel(nn.Module):
    """LSTM that accepts **pre-computed** entity features.

    For modes like ``onehot``, ``coordinates``, ``sinusoidal``, or
    ``descriptors`` where the data layer provides continuous entity
    feature vectors rather than integer IDs.

    Args:
        input_size: Number of raw input features (excl. entity features).
        entity_dim: Dimension of the entity feature vector.
        hidden_size: LSTM hidden dimension.
        num_layers: Stacked LSTM layers.
        c_out: Output channels (default ``1``).
        dropout: LSTM dropout.

    Forward:
        x: ``(B, seq_len, input_size)`` — raw input features.
        entity_features: ``(B, seq_len, entity_dim)`` — entity features.
    Output: ``(B, seq_len, c_out)``
    """

    def __init__(
        self,
        input_size: int,
        entity_dim: int,
        hidden_size: int,
        num_layers: int,
        c_out: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size + entity_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, c_out))

    def forward(
        self,
        x: torch.Tensor,
        entity_features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([x, entity_features], dim=-1)
        out, _ = self.lstm(x)
        return self.linear(out)


# ======================================================================
# Entity-identifier modes understood by the adapters
# ======================================================================

# Modes where the entity features are already concatenated into x_enc by
# the data layer — the adapter doesn't need to do anything special.
_TRANSPARENT_MODES = frozenset(
    {
        'onehot',
        'coordinates',
        'sinusoidal',
        'descriptors',
        'numeric_id',
    }
)

# The mode that needs nn.Embedding inside the model.
_EMBEDDING_MODE = 'embedding'

# The mode providing a separate entity_features tensor in the batch.
_FEATURE_CONCAT_MODE = 'feature_concat'


# ======================================================================
# Adapters
# ======================================================================


class _LSTMBaseAdapter(TorchModelAdapter):
    """Shared adapter logic for LSTM variants.

    Maps the liulian 4-tensor / batch-dict call to the model's forward.

    Entity handling (configured via ``identifier_mode`` in config):

    * ``'none'`` (default) — standard ``forward(x)``.
    * ``'embedding'`` — integer entity IDs extracted from
      ``x_mark_enc[:, :, entity_id_col]`` (default col 0) → passed to
      ``forward(e, x)``.
    * ``'onehot'`` / ``'coordinates'`` / etc. — entity features are
      already part of ``x_enc``; adapter does nothing special.
    * ``'feature_concat'`` — entity features from batch key
      ``'entity_features'`` → ``forward(x, entity_features)``.
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

        def _to_tensor(v: Any) -> Optional[torch.Tensor]:
            if v is None:
                return None
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor):
                return v.float().to(self.device)
            return v

        def _get(*keys: str) -> Any:
            for k in keys:
                v = torch_batch.get(k)
                if v is not None:
                    return v
            return None

        x_enc = _to_tensor(_get('x_enc', 'X'))

        mode = self._entity_mode

        if mode == _EMBEDDING_MODE:
            # Extract integer IDs from marks
            x_mark = _to_tensor(_get('x_mark_enc', 'X_mark'))
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
            # Separate entity features tensor in batch
            ef = _to_tensor(_get('entity_features'))
            if ef is None:
                # Fallback: use model as plain LSTM
                output = self._model(x_enc)
            else:
                output = self._model(x_enc, ef)

        else:
            # 'none' or transparent modes (features already in x_enc)
            output = self._model(x_enc)

        if isinstance(output, tuple):
            output = output[0]
        return {'predictions': output}


class LSTMAdapter(_LSTMBaseAdapter):
    """General-purpose LSTM adapter — works with any dataset.

    Config keys:
        enc_in (int): Input feature dimension.
        c_out (int): Output channels (default mirrors ``enc_in`` for
            channel-preserving, or ``1`` for single-target).
        d_model (int): LSTM hidden dimension (default ``64``).
        e_layers (int): Number of LSTM layers (default ``2``).
        dropout (float): LSTM dropout (default ``0.0``).
        identifier_mode (str): Entity identifier mode (default ``'none'``).
        entity_id_col (int): Column in x_mark_enc for embedding IDs
            (default ``0``).
        entity_dim (int): Entity feature dimension for ``feature_concat``
            mode.
        num_embeddings (int): Vocab size for ``embedding`` mode.
        embedding_size (int): Embedding dim for ``embedding`` mode.

    Examples::

        # Traffic dataset (862 channels, no entity IDs)
        LSTMAdapter({'enc_in': 862, 'c_out': 862, 'd_model': 128})

        # Swiss-river (1 channel, with station embedding)
        LSTMAdapter(
            {
                'enc_in': 1,
                'c_out': 1,
                'd_model': 64,
                'identifier_mode': 'embedding',
                'num_embeddings': 50,
                'embedding_size': 10,
            }
        )

        # ETT dataset with one-hot entity features (already in x_enc)
        LSTMAdapter(
            {'enc_in': 7 + 10, 'c_out': 7, 'd_model': 64, 'identifier_mode': 'onehot'}
        )
    """

    def __init__(self, config: Dict[str, Any]):
        mode = config.get('identifier_mode', 'none')
        enc_in = config.get('enc_in', 1)
        c_out = config.get('c_out', 1)
        hidden = config.get('d_model', 64)
        n_layers = config.get('e_layers', 2)
        dropout = config.get('dropout', 0.0)

        if mode == _EMBEDDING_MODE:
            model = LstmEmbeddingModel(
                input_size=enc_in,
                num_embeddings=config.get('num_embeddings', 50),
                embedding_size=config.get('embedding_size', 10),
                hidden_size=hidden,
                num_layers=n_layers,
                c_out=c_out,
                dropout=dropout,
            )
        elif mode == _FEATURE_CONCAT_MODE:
            entity_dim = config.get('entity_dim', 16)
            model = LstmEntityFeatureModel(
                input_size=enc_in,
                entity_dim=entity_dim,
                hidden_size=hidden,
                num_layers=n_layers,
                c_out=c_out,
                dropout=dropout,
            )
        else:
            # 'none' or transparent (features already in x_enc)
            model = SwissLstmModel(
                input_size=enc_in,
                hidden_size=hidden,
                num_layers=n_layers,
                c_out=c_out,
                dropout=dropout,
            )

        full_cfg = dict(config)
        full_cfg.setdefault('identifier_mode', mode)
        super().__init__(model, full_cfg)


class ExtrapoLSTMAdapter(_LSTMBaseAdapter):
    """General-purpose extrapolation LSTM adapter.

    Config keys:
        enc_in, c_out, d_model, e_layers, dropout — same as :class:`LSTMAdapter`.
        pred_len / future_steps (int): Prediction horizon.
        extrapo_mode (str): ``'limo'`` or ``'fembed'``.
        d_future_emb (int): Future-step embedding dim for ``fembed`` mode.
    """

    def __init__(self, config: Dict[str, Any]):
        enc_in = config.get('enc_in', 1)
        c_out = config.get('c_out', 1)
        hidden = config.get('d_model', 64)
        n_layers = config.get('e_layers', 2)
        dropout = config.get('dropout', 0.0)
        future = config.get('future_steps', config.get('pred_len', 7))
        extrapo = config.get('extrapo_mode', 'limo')

        if extrapo == 'fembed':
            model = ExtrapoLstmModelFEmbed(
                input_size=enc_in,
                hidden_size=hidden,
                num_layers=n_layers,
                future_steps=future,
                d_future_emb=config.get('d_future_emb', 32),
                c_out=c_out,
                dropout=dropout,
            )
        else:
            model = ExtrapoLstmModelLIMO(
                input_size=enc_in,
                hidden_size=hidden,
                num_layers=n_layers,
                future_steps=future,
                c_out=c_out,
                dropout=dropout,
            )

        full_cfg = dict(config)
        full_cfg.setdefault('identifier_mode', 'none')
        super().__init__(model, full_cfg)


# ======================================================================
# Backward-compatible aliases
# ======================================================================

# Old adapter names kept for existing code / tests
SwissLSTMAdapter = LSTMAdapter
SwissExtrapoLSTMAdapter = ExtrapoLSTMAdapter


class SwissLSTMEmbeddingAdapter(LSTMAdapter):
    """Backward-compatible alias — forces ``identifier_mode='embedding'``."""

    def __init__(self, config: Dict[str, Any]):
        cfg = dict(config)
        cfg.setdefault('identifier_mode', 'embedding')
        super().__init__(cfg)
