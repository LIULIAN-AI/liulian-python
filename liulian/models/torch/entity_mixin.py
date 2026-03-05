"""Entity-aware mixin and wrapper for TSL model adapters.

Provides ``EntityAwareMixin``, ``EntityWrapper``, and
``ChannelEntityWrapper`` to enable entity-identifier support on any
Time-Series-Library model adapter with minimal changes.

Supported identifier modes
---------------------------
* ``'none'``         — no entity features; pass-through.
* Transparent modes (``'onehot'``, ``'coordinates'``, ``'sinusoidal'``,
  ``'descriptors'``, ``'numeric_id'``) — entity features are concatenated
  into ``x_enc`` by the data layer; no model changes needed other than setting
  ``enc_in`` to include the extra dimensions.
* ``'embedding'``    — **per_entity mode**: integer station IDs are extracted
  from ``x_mark_enc[:, :, entity_id_col]`` and looked up via
  ``nn.Embedding``; the resulting embedding vector is concatenated to
  ``x_enc`` and then projected back to the original ``enc_in`` dimensions
  via a learned linear layer before calling the inner model.
  **multi_channel mode**: a ``ChannelEntityWrapper`` injects per-channel
  station embeddings.

Usage in an adapter
-------------------
::

    from liulian.models.torch.entity_mixin import EntityAwareMixin


    class DLinearAdapter(EntityAwareMixin, TorchModelAdapter):
        def __init__(self, config):
            # _entity_model_config is identity — no enc_in change needed
            model = Model(self._dict_to_namespace(config))
            super().__init__(model, config)
            self._init_entity_support(config)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


# Modes where the data layer already concatenated entity features into x_enc.
_TRANSPARENT_MODES = frozenset(
    {'onehot', 'coordinates', 'sinusoidal', 'descriptors', 'numeric_id'}
)


class EntityWrapper(nn.Module):
    """Wraps a TSL ``nn.Module`` to inject entity embeddings into *x_enc*.

    For ``'embedding'`` mode the wrapper:
    1. Extracts integer station IDs from ``x_mark_enc[:, :, entity_id_col]``.
    2. Looks them up in an ``nn.Embedding`` table.
    3. Concatenates the resulting vector to ``x_enc`` along the feature axis.
    4. Projects ``(enc_in + embedding_size) → enc_in`` via a learned linear
       layer so the inner model sees the original dimensions.
    5. Repeats 2-4 for ``x_dec`` (if present) to keep encoder-decoder models
       consistent.

    The inner model is constructed with the **original** ``enc_in`` — no
    config adjustment is needed.

    Parameters
    ----------
    inner_model : nn.Module
        The original TSL model (e.g. ``DLinear.Model``).
    enc_in : int
        Number of input features the inner model expects.
    num_embeddings : int
        Number of distinct entities (vocabulary size).
    embedding_size : int
        Dimensionality of each entity embedding vector.
    entity_id_col : int
        Column index in ``x_mark_enc`` that holds the integer station ID.
    """

    def __init__(
        self,
        inner_model: nn.Module,
        enc_in: int,
        num_embeddings: int = 50,
        embedding_size: int = 10,
        entity_id_col: int = 0,
    ) -> None:
        super().__init__()
        self.inner = inner_model
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.entity_id_col = entity_id_col
        # Projection layers: (enc_in + embedding_size) → enc_in
        self.enc_proj = nn.Linear(enc_in + embedding_size, enc_in)
        self.dec_proj = nn.Linear(enc_in + embedding_size, enc_in)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        entity_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with entity embedding injection.

        Signature matches the TSL model convention:
        ``model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)``,
        extended with an optional ``entity_ids`` kwarg.

        Parameters
        ----------
        entity_ids : torch.Tensor | None
            Integer station indices of shape ``(B,)``.  When provided,
            these are used directly for the embedding lookup.  When
            *None*, falls back to reading station IDs from
            ``x_mark_enc[:, :, entity_id_col]`` (legacy path).

        Entity embeddings are injected into both ``x_enc`` and ``x_dec``
        (when present), then projected back to the original feature
        dimension so the inner model sees no dimension change.
        """
        # Resolve entity indices
        e_ids: torch.Tensor | None = None
        if entity_ids is not None:
            # Direct entity_ids tensor: (B,) → expand to (B, T_enc)
            e_ids = entity_ids.unsqueeze(1).expand(-1, x_enc.size(1))  # (B, T_enc)
        elif x_mark_enc is not None and x_mark_enc.ndim >= 2:
            # Legacy: read from mark tensor
            col = self.entity_id_col
            if x_mark_enc.ndim == 3:
                e_ids = x_mark_enc[:, : x_enc.size(1), col].long()
            else:
                e_ids = x_mark_enc[:, : x_enc.size(1)].long()

        if e_ids is not None:
            emb = self.embedding(e_ids)  # (B, T_enc, embedding_size)
            x_enc = self.enc_proj(torch.cat([x_enc, emb], dim=-1))

            # Also augment x_dec for encoder-decoder models
            if x_dec is not None:  # todo: is this correct?
                T_dec = x_dec.size(1)
                e_id_scalar = e_ids[:, 0:1]  # (B, 1)
                dec_emb = self.embedding(
                    e_id_scalar.expand(-1, T_dec)
                )  # (B, T_dec, embedding_size)
                x_dec = self.dec_proj(torch.cat([x_dec, dec_emb], dim=-1))

        # Only pass mask if explicitly provided — many TSL models
        # (e.g. LSTM) don't accept a mask parameter.
        if mask is not None:
            return self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec)


class ChannelEntityWrapper(nn.Module):
    """Per-channel entity embedding for multi_channel mode.

    In multi_channel mode every sample contains all stations as separate
    channels: ``x_enc`` has shape ``(B, T, N)`` where ``N`` is the number
    of stations.  This wrapper adds a learnable embedding per channel
    (station), enabling the model to learn station-specific behaviour.

    Architecture:

    1. Look up all station embeddings: ``(N, d)``.
    2. Reshape ``x_enc`` from ``(B, T, N)`` to ``(B, T, N, 1)``.
    3. Broadcast embeddings to ``(B, T, N, d)``.
    4. Concatenate: ``(B, T, N, 1+d)``.
    5. Project per-element: ``Linear(1+d, 1)`` → ``(B, T, N)``.
    6. The inner model receives original shape ``(B, T, N)``.

    The same procedure is applied to ``x_dec`` when present
    (encoder-decoder models).

    Parameters
    ----------
    inner_model : nn.Module
        The original TSL model (e.g. ``DLinear.Model``).
    num_stations : int
        Number of distinct stations / channels.
    embedding_size : int
        Dimensionality of each station embedding vector.
    """

    def __init__(
        self,
        inner_model: nn.Module,
        num_stations: int,
        embedding_size: int = 10,
    ) -> None:
        super().__init__()
        self.inner = inner_model
        self.station_embedding = nn.Embedding(num_stations, embedding_size)
        # Per-element projection: (value + embedding) → value
        self.enc_proj = nn.Linear(1 + embedding_size, 1)
        self.dec_proj = nn.Linear(1 + embedding_size, 1)
        # Fixed station indices [0, 1, ..., N-1]
        self.register_buffer(
            'station_ids', torch.arange(num_stations, dtype=torch.long)
        )

    def _augment(
        self,
        x: torch.Tensor,
        proj: nn.Linear,
    ) -> torch.Tensor:
        """Inject station embeddings channel-wise.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, N)``.
        proj : nn.Linear
            Projection layer ``Linear(1+d, 1)``.

        Returns
        -------
        torch.Tensor
            Augmented tensor of shape ``(B, T, N)``.
        """
        B, T, N = x.shape
        # (N, d) station embeddings
        emb = self.station_embedding(self.station_ids)  # (N, d)
        # Broadcast: (1, 1, N, d) → (B, T, N, d)
        emb = emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        # Reshape x: (B, T, N) → (B, T, N, 1)
        x_4d = x.unsqueeze(-1)
        # Concat: (B, T, N, 1+d)
        augmented = torch.cat([x_4d, emb], dim=-1)
        # Project: (B, T, N, 1+d) → (B, T, N, 1) → (B, T, N)
        return proj(augmented).squeeze(-1)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        entity_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with per-channel station embedding injection.

        Signature matches the TSL model convention plus optional
        ``entity_ids`` (accepted but ignored — station indices are
        pre-registered internally).
        """
        x_enc = self._augment(x_enc, self.enc_proj)

        if x_dec is not None:
            x_dec = self._augment(x_dec, self.dec_proj)

        if mask is not None:
            return self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec)


class EntityAwareMixin:
    """Mixin that adds entity-identifier support to any ``TorchModelAdapter``.

    Put this mixin **before** ``TorchModelAdapter`` in the MRO::

        class MyAdapter(EntityAwareMixin, TorchModelAdapter): ...

    Call helpers in ``__init__``:

    1. ``model_cfg = self._entity_model_config(config)``
       → for the projection-based approach this returns the config
       unchanged (the EntityWrapper handles dimension matching internally).
    2. After ``super().__init__(model, config)``:
       ``self._init_entity_support(config)``
       → for ``'embedding'`` mode this wraps ``self._model`` in
       :class:`EntityWrapper`.

    Config keys
    -----------
    identifier_mode : str
        One of ``'none'``, ``'embedding'``, ``'patch_embedding'``,
        ``'onehot'``, ``'coordinates'``, ``'sinusoidal'``,
        ``'descriptors'``, ``'numeric_id'``.
        ``'patch_embedding'`` is handled by the model internally
        (e.g. ``patchtst_entity.Model``); no wrapper is applied.
    entity_id_col : int
        Column index in ``x_mark_enc`` for integer station IDs
        (default ``0``).
    num_embeddings : int
        Vocabulary size for ``'embedding'`` mode (default ``50``).
    embedding_size : int
        Embedding vector dimension for ``'embedding'`` mode (default ``10``).
    """

    _entity_mode: str = 'none'
    _entity_id_col: int = 0

    # ------------------------------------------------------------------
    # Static helpers — usable before super().__init__
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Return config for building the inner model.

        With the projection-based :class:`EntityWrapper`, ``enc_in`` does
        **not** need adjustment — the wrapper projects augmented features
        back to original dimensions internally.  This method is kept for
        API compatibility and returns *config* unchanged.
        """
        return config

    # ------------------------------------------------------------------
    # Instance helpers — call after super().__init__
    # ------------------------------------------------------------------

    def _init_entity_support(self, config: Dict[str, Any]) -> None:
        """Finalise entity support after the model has been built.

        For ``'embedding'`` mode this wraps ``self._model`` inside an
        :class:`EntityWrapper` (per_entity mode) or :class:`ChannelEntityWrapper`
        (multi_channel mode) and moves it to ``self.device``.
        """
        self._entity_mode = config.get('identifier_mode', 'none')
        self._entity_id_col = config.get('entity_id_col', 0)

        if self._entity_mode == 'embedding':
            if config.get('split_mode') == 'multi_channel':
                self._model = ChannelEntityWrapper(
                    self._model,
                    num_stations=config.get('num_embeddings', 50),
                    embedding_size=config.get('embedding_size', 10),
                )
            else:
                enc_in = config.get('enc_in', 1)
                self._model = EntityWrapper(
                    self._model,
                    enc_in=enc_in,
                    num_embeddings=config.get('num_embeddings', 50),
                    embedding_size=config.get('embedding_size', 10),
                    entity_id_col=self._entity_id_col,
                )
            # Ensure the wrapper is on the correct device
            self._model = self._model.to(self.device)
