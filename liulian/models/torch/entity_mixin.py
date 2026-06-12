"""Entity-aware mixin and wrapper for TSL model adapters.

Provides ``EntityAwareMixin``, ``EntityWrapper``, and
``ChannelEntityWrapper`` to enable entity-identifier support on any
Time-Series-Library model adapter with minimal changes.

Supported identifier modes
---------------------------
* ``'none'``         — no entity features; pass-through.
* Transparent modes (``'onehot'``, ``'coordinates'``, ``'sinusoidal'``,
  ``'random'``, ``'descriptors'``, ``'numeric_id'``) — entity features are concatenated
  into ``x_enc`` by the data layer; no model changes needed other than setting
  ``enc_in`` to include the extra dimensions.
* ``'embedding'``    — **per_entity mode**: integer station IDs are extracted
  from ``x_mark_enc[:, :, entity_id_col]`` and looked up via
  ``nn.Embedding``; the resulting embedding vector is concatenated to
  ``x_enc`` and then projected back to the original ``enc_in`` dimensions
  via a learned linear layer before calling the inner model.
  **multi_channel mode**: a ``ChannelEntityWrapper`` injects per-channel
    station embeddings for ``id_integration='concat_to_x'``. PatchTST can
    instead handle ``id_integration='add_after_patch'`` internally.

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
_TRANSPARENT_MODES = frozenset({'onehot', 'coordinates', 'sinusoidal', 'random', 'descriptors', 'numeric_id'})


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
            # Direct entity_ids tensor: (B,) → expand to (B, T_enc)  todo: test this
            e_ids = entity_ids.unsqueeze(1).expand(-1, x_enc.size(1))  # (B, T_enc)
        elif x_mark_enc is not None and x_mark_enc.ndim >= 2:
            # Legacy: read from mark tensor  todo: test this
            col = self.entity_id_col
            if x_mark_enc.ndim == 3:
                e_ids = x_mark_enc[:, : x_enc.size(1), col].long()
            else:
                e_ids = x_mark_enc[:, : x_enc.size(1)].long()

        if e_ids is not None:
            emb = self.embedding(e_ids)  # (B, T_enc, embedding_size)
            x_enc = self.enc_proj(torch.cat([x_enc, emb], dim=-1))

            # Also augment x_dec for encoder-decoder models  # todo: this is not needed for non-enc-dec models
            if x_dec is not None:  # todo: test this part
                T_dec = x_dec.size(1)
                e_id_scalar = e_ids[:, 0:1]  # (B, 1)
                dec_emb = self.embedding(e_id_scalar.expand(-1, T_dec))  # (B, T_dec, embedding_size)
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
        self.register_buffer('station_ids', torch.arange(num_stations, dtype=torch.long))

    def _augment(
        self,
        x: torch.Tensor,
        proj: nn.Linear,
        channel_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inject station embeddings channel-wise.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, N)``.
        proj : nn.Linear
            Projection layer ``Linear(1+d, 1)``.
        channel_ids : torch.Tensor | None
            Optional channel-to-station mapping of shape ``(N,)`` or
            ``(B, N)``.  When provided, these indices are used for the
            embedding lookup instead of the default ``arange(N)`` buffer.
            This makes the wrapper safe under channel permutation.

        Returns
        -------
        torch.Tensor
            Augmented tensor of shape ``(B, T, N)``.
        """
        B, T, N = x.shape
        # Resolve channel → station mapping
        ids = channel_ids if channel_ids is not None else self.station_ids
        # (N, d) or (B, N, d) station embeddings
        emb = self.station_embedding(ids)
        if emb.ndim == 2:
            # (N, d) → broadcast to (B, T, N, d)
            emb = emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        else:
            # (B, N, d) → (B, 1, N, d) → (B, T, N, d)
            emb = emb.unsqueeze(1).expand(-1, T, -1, -1)
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
        ``entity_ids``.

        Parameters
        ----------
        entity_ids : torch.Tensor | None
            When provided (shape ``(N,)`` or ``(B, N)``), used as the
            channel-to-station mapping for the embedding lookup instead
            of the default ``arange(N)`` buffer.  This makes the wrapper
            safe when channel order may be permuted (e.g. by data
            augmentation or non-standard loaders).
        """
        x_enc = self._augment(x_enc, self.enc_proj, channel_ids=entity_ids)

        if x_dec is not None:
            x_dec = self._augment(x_dec, self.dec_proj, channel_ids=entity_ids)

        if mask is not None:
            return self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec)


def _build_channel_features(
    mode: str,
    num_stations: int,
    *,
    sinusoidal_dim: int = 16,
    random_dim: int = 16,
    random_seed: int = 2026,
    coordinates: dict[str, tuple[float, float]] | None = None,
    station_ids: list[str] | None = None,
) -> torch.Tensor:
    """Build a ``(N, D)`` feature matrix for all channels.

    Each row is the transparent identifier vector for one channel/station.

    Args:
        mode: Identifier mode (``onehot``, ``sinusoidal``, ``random``,
            ``coordinates``, ``numeric_id``).
        num_stations: Number of channels/stations.
        sinusoidal_dim: Dimension for sinusoidal encoding.
        random_dim: Dimension for random encoding.
        random_seed: Base seed for random identifiers.
        coordinates: Station-name to ``(lat, lon)`` mapping.
        station_ids: Ordered station name list (index-aligned with channels).

    Returns:
        Feature tensor of shape ``(N, D)``.
    """
    import hashlib

    import numpy as np

    ids = station_ids or [str(i) for i in range(num_stations)]

    if mode == 'onehot':
        return torch.eye(num_stations, dtype=torch.float32)

    if mode == 'numeric_id':
        denom = max(num_stations - 1, 1)
        vals = [i / denom for i in range(num_stations)]
        return torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)

    if mode == 'sinusoidal':
        rows = []
        for idx in range(num_stations):
            pos = torch.zeros(sinusoidal_dim, dtype=torch.float32)
            half = sinusoidal_dim // 2
            freqs = torch.exp(
                torch.arange(0, half, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
            )
            pos[:half] = torch.sin(idx * freqs)
            pos[half : 2 * half] = torch.cos(idx * freqs)
            rows.append(pos)
        return torch.stack(rows)

    if mode == 'random':
        rows = []
        for idx in range(num_stations):
            key = f'{random_seed}:{ids[idx]}'
            digest = hashlib.sha256(key.encode('utf-8')).digest()
            seed = int.from_bytes(digest[:8], byteorder='little', signed=False) % (2**32)
            rng = np.random.default_rng(seed)
            vec = torch.tensor(
                rng.standard_normal(random_dim).astype(np.float32),
                dtype=torch.float32,
            )
            norm = torch.linalg.norm(vec)
            if torch.isfinite(norm) and float(norm) > 0:
                vec = vec / norm
            rows.append(vec)
        return torch.stack(rows)

    if mode == 'coordinates':
        coords = coordinates or {}
        missing = [ids[idx] for idx in range(num_stations) if ids[idx] not in coords]
        if missing:
            # No silent zero fallback: zero vectors FAKE the identifier
            # (all channels identical) while the run still "succeeds" —
            # this is what invalidated the pre-2026-06-11 multi_channel
            # swiss coordinate cells. Inject the station->(x, y) mapping
            # via config['coordinates'] (not wired in the pipeline yet).
            raise ValueError(
                "identifier_mode='coordinates' requires a coordinate for "
                f'every channel; missing for {missing[:5]!r}'
                f'{"..." if len(missing) > 5 else ""}.'
            )
        arr = torch.tensor(
            [coords[ids[idx]] for idx in range(num_stations)],
            dtype=torch.float32,
        )
        # Min-max normalize per dimension (raw CH1903 meters are ~1e5-1e6,
        # which would dwarf the scaled inputs).
        lo = arr.min(dim=0).values
        span = (arr.max(dim=0).values - lo).clamp_min(1e-12)
        return (arr - lo) / span

    raise ValueError(f'Unsupported transparent mode for channel wrapper: {mode!r}')


class EntityTransparentWrapper(nn.Module):
    """Per-sample transparent (non-learned) feature injection — per_entity mode.

    Model-layer equivalent of the data-layer bake-in
    (``liulian.data.ts.timeseriesdataset.make_entity_features``): looks up a
    FIXED ``(num_stations, D)`` feature table by per-sample entity index,
    tiles it over time and CONCATENATES it to ``x_enc`` — no projection, no
    learnable parameters. The inner model must therefore be built with
    ``enc_in = base_features + D``.

    The table rows are built by calling the data-layer
    ``make_entity_features`` per station, so the injected values are
    BITWISE-IDENTICAL to what the data-layer bake-in would have produced
    (see tests/models/torch/test_transparent_injection_equivalence.py).
    This is what keeps wrapper-injected runs comparable with the
    data-injected baselines.

    Unlike :class:`EntityWrapper` (embedding mode) there is no projection
    back to the original width — transparency means the raw fixed vector IS
    the input. ``x_dec`` is passed through untouched (decoder injection is
    not needed for the current per_entity matrix; LSTM ignores ``x_dec``).

    Parameters
    ----------
    inner_model : nn.Module
        TSL-style model built with ``enc_in = base + feature_dim``.
    mode : str
        Transparent identifier mode (``onehot`` / ``coordinates`` /
        ``sinusoidal`` / ``random`` / ``numeric_id``).
    station_ids : list[str]
        Ordered station ids; row ``i`` of the table belongs to
        ``station_ids[i]`` (must match the dataset's entity indexing).
    coordinates : dict | None
        Station-name to ``(x, y)`` mapping (required for ``coordinates``).
    sinusoidal_dim / random_dim / random_seed
        Forwarded to ``make_entity_features``.
    """

    def __init__(
        self,
        inner_model: nn.Module,
        mode: str,
        station_ids: list[str],
        *,
        coordinates: dict[str, tuple[float, float]] | None = None,
        sinusoidal_dim: int = 16,
        random_dim: int = 16,
        random_seed: int = 2026,
    ) -> None:
        super().__init__()
        # Data-layer feature builder reused on purpose: constructive
        # equivalence with the bake-in path (NOT make_channel_features,
        # whose sinusoidal formula differs — see ledger 2026-06-12).
        from liulian.data.ts.timeseriesdataset import make_entity_features

        self.inner = inner_model
        self.mode = str(mode)
        self.station_ids = [str(s) for s in station_ids]
        self._coordinates = coordinates
        self.sinusoidal_dim = int(sinusoidal_dim)
        self.random_dim = int(random_dim)
        self.random_seed = int(random_seed)

        rows = []
        for name in self.station_ids:
            block = make_entity_features(
                name,
                self.station_ids,
                self.mode,
                seq_len=1,
                coordinates=coordinates,
                sinusoidal_dim=self.sinusoidal_dim,
                random_dim=self.random_dim,
                random_seed=self.random_seed,
            )
            if block is None:
                raise ValueError(
                    f'mode {mode!r} produces no transparent features — EntityTransparentWrapper is not applicable.'
                )
            rows.append(block[0])
        self.register_buffer('feature_table', torch.stack(rows))  # (N, D)

    @property
    def feature_dim(self) -> int:
        """Width ``D`` of the injected feature block."""
        return int(self.feature_table.shape[-1])

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        entity_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concat the per-sample fixed feature block to ``x_enc``.

        ``entity_ids``: integer station indices of shape ``(B,)`` — supplied
        by the trainer from the loader's per-sample entity index.
        """
        if entity_ids is None:
            raise ValueError(
                'EntityTransparentWrapper requires entity_ids (per-sample '
                'station indices); check the trainer wiring '
                '(pass_entity_ids).'
            )
        rows = self.feature_table[entity_ids.long()]  # (B, D)
        ent = rows.unsqueeze(1).expand(-1, x_enc.size(1), -1)  # (B, T, D)
        # Same layout as the data-layer bake-in: identifier block LAST.
        x_enc = torch.cat([x_enc, ent.to(x_enc.dtype)], dim=-1)
        if mask is not None:
            return self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec)


class ChannelTransparentWrapper(nn.Module):
    """Per-channel transparent (non-learned) feature injection for multi_channel mode.

    Similar to :class:`ChannelEntityWrapper` but uses pre-computed feature
    vectors (onehot, sinusoidal, random, coordinates, numeric_id) instead of
    learned ``nn.Embedding``.

    The feature matrix is registered as a buffer so it moves with the model
    to GPU and is included in ``state_dict``.

    Architecture:

    1. Pre-compute ``(N, D)`` feature matrix at init.
    2. Reshape ``x_enc`` from ``(B, T, N)`` to ``(B, T, N, 1)``.
    3. Broadcast features to ``(B, T, N, D)``.
    4. Concatenate: ``(B, T, N, 1+D)``.
    5. Project per-element: ``Linear(1+D, 1)`` -> ``(B, T, N)``.
    6. Inner model receives original shape ``(B, T, N)``.

    Parameters
    ----------
    inner_model : nn.Module
        The original TSL model.
    mode : str
        Transparent identifier mode.
    num_stations : int
        Number of channels/stations.
    sinusoidal_dim : int
        Dimension for sinusoidal encoding.
    random_dim : int
        Dimension for random encoding.
    random_seed : int
        Base seed for random identifiers.
    coordinates : dict or None
        Station-to-coordinate mapping.
    station_ids : list[str] or None
        Ordered station name list.
    """

    def __init__(
        self,
        inner_model: nn.Module,
        mode: str,
        num_stations: int,
        *,
        sinusoidal_dim: int = 16,
        random_dim: int = 16,
        random_seed: int = 2026,
        coordinates: dict[str, tuple[float, float]] | None = None,
        station_ids: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.inner = inner_model
        features = _build_channel_features(
            mode,
            num_stations,
            sinusoidal_dim=sinusoidal_dim,
            random_dim=random_dim,
            random_seed=random_seed,
            coordinates=coordinates,
            station_ids=station_ids,
        )
        self.register_buffer('channel_features', features)  # (N, D)
        feat_dim = features.shape[1]
        self.enc_proj = nn.Linear(1 + feat_dim, 1)
        self.dec_proj = nn.Linear(1 + feat_dim, 1)

    def _augment(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        B, T, N = x.shape
        feats = self.channel_features  # (N, D)
        feats = feats.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)  # (B,T,N,D)
        x_4d = x.unsqueeze(-1)  # (B, T, N, 1)
        augmented = torch.cat([x_4d, feats], dim=-1)  # (B, T, N, 1+D)
        return proj(augmented).squeeze(-1)  # (B, T, N)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        entity_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        One of ``'none'``, ``'embedding'``,
        ``'onehot'``, ``'coordinates'``, ``'sinusoidal'``, ``'random'``,
        ``'descriptors'``, ``'numeric_id'``.
        ``'embedding'`` + ``id_integration='add_after_patch'`` is handled
        by PatchTST internally; no wrapper is applied.
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
        :class:`EntityWrapper` (per_entity) or :class:`ChannelEntityWrapper`
        (multi_channel) and moves it to ``self.device``.

        For transparent modes (``onehot``, ``sinusoidal``, ``random``,
        ``coordinates``, ``numeric_id``) in ``multi_channel`` split, this
        wraps ``self._model`` in :class:`ChannelTransparentWrapper` so that
        per-channel features are injected at forward time without requiring
        ``station_name`` in the data layer.
        """
        self._entity_mode = config.get('identifier_mode', 'none')
        self._entity_id_col = config.get('entity_id_col', 0)

        if self._entity_mode == 'none':
            return

        is_multi_channel = config.get('split_mode') == 'multi_channel'
        is_add_after_patch = config.get('id_integration') == 'add_after_patch'

        if self._entity_mode == 'embedding' and not is_add_after_patch:
            if is_multi_channel:
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
            self._model = self._model.to(self.device)

        elif self._entity_mode in _TRANSPARENT_MODES and is_multi_channel:
            station_ids = config.get('station_ids')
            if isinstance(station_ids, list):
                station_ids_list = [str(s) for s in station_ids]
            else:
                station_ids_list = None
            num_stations = config.get('num_embeddings', config.get('enc_in', 50))
            self._model = ChannelTransparentWrapper(
                self._model,
                mode=self._entity_mode,
                num_stations=num_stations,
                sinusoidal_dim=config.get('sinusoidal_dim', 16),
                random_dim=config.get('random_identifier_dim', 16),
                random_seed=config.get('random_identifier_seed', 2026),
                coordinates=config.get('coordinates'),
                station_ids=station_ids_list,
            )
            self._model = self._model.to(self.device)
