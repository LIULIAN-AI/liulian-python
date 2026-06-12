"""Configurable scalers for time-series normalisation.

Provides a unified API for the normalisation strategies used across
reference projects:

* **StandardScaler** — zero-mean / unit-variance (TimeLLM default).
* **MinMaxScaler** — scale to [0, 1] (swiss-river isolated models).
* **DimSplitScaler** — per-dimension sub-scaler extracted from a fitted
  global scaler (swiss-river).
* **StationSplitScaler** — per-station sub-scaler identified by feature
  suffix (swiss-river spatio-temporal / GNN models).
* **EntityScaler** — high-level helper that fits independent scalers
  per entity per feature on a multi-entity DataFrame and handles
  batch-level inverse-transform for model predictions.  Generalised
  successor to ``PerStationScaler`` (backward-compatible alias kept).
* **NoScaler** — identity pass-through.

All scalers follow the sklearn interface (``fit``, ``transform``,
``inverse_transform``) and work with 2-D numpy arrays.

The factory :func:`get_scaler` creates the right scaler from a name string,
making it easy to configure via YAML / CLI::

    scaler = get_scaler('standard')  # StandardScaler
    scaler = get_scaler('minmax')  # MinMaxScaler
    scaler = get_scaler('none')  # NoScaler
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.utils.validation import check_is_fitted

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# -----------------------------------------------------------------------
# NoScaler
# -----------------------------------------------------------------------


class NoScaler:
    """Identity scaler — does nothing, for API compatibility."""

    def fit(self, X: np.ndarray, y: Any = None) -> 'NoScaler':
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


# -----------------------------------------------------------------------
# DimSplitScaler
# -----------------------------------------------------------------------


class DimSplitScaler:
    """Per-dimension sub-scaler extracted from a fitted global scaler.

    Adapted from ``swissrivernetwork.util.scaler.DimSplitScaler``.

    Args:
        global_scaler: A **fitted** ``MinMaxScaler`` or ``StandardScaler``.
        dims_kept: Indices of dimensions to keep.  ``None`` = all.
    """

    def __init__(
        self,
        global_scaler: Any,
        dims_kept: Optional[np.ndarray] = None,
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError('DimSplitScaler requires scikit-learn.')
        check_is_fitted(global_scaler)

        if dims_kept is None:
            dims_kept = np.arange(global_scaler.n_features_in_)
        self.dims_kept = dims_kept
        self.n_dim = len(dims_kept)
        self.scalers = [self._single(global_scaler, d) for d in dims_kept]
        self._cur: Optional[int] = None

    # -- indexing ---------------------------------------------------------

    def __getitem__(self, idx: int) -> 'DimSplitScaler':
        if idx < 0 or idx >= self.n_dim:
            raise IndexError(f'Index {idx} out of range [0, {self.n_dim})')
        self._cur = idx
        return self

    # -- transform --------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._cur is None:
            raise ValueError('Set dimension via indexing first: scaler[dim].')
        return self.scalers[self._cur].transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._cur is None:
            raise ValueError('Set dimension via indexing first: scaler[dim].')
        return self.scalers[self._cur].inverse_transform(X)

    # -- helper -----------------------------------------------------------

    @staticmethod
    def _single(global_scaler: Any, dim: int) -> Any:
        """Build a 1-D sklearn scaler for *dim* from *global_scaler*."""
        if isinstance(global_scaler, MinMaxScaler):
            s = MinMaxScaler()
        elif isinstance(global_scaler, StandardScaler):
            s = StandardScaler()
        else:
            raise TypeError(f'Unsupported scaler type: {type(global_scaler)}')

        for attr in (
            'min_',
            'scale_',
            'data_min_',
            'data_max_',
            'data_range_',
            'mean_',
            'var_',
        ):
            if hasattr(global_scaler, attr):
                val = getattr(global_scaler, attr)
                if val is not None:
                    setattr(s, attr, np.array([val[dim]]))

        s.n_features_in_ = 1
        if hasattr(global_scaler, 'n_samples_seen_'):
            s.n_samples_seen_ = global_scaler.n_samples_seen_
        return s


# -----------------------------------------------------------------------
# StationSplitScaler
# -----------------------------------------------------------------------


class StationSplitScaler:
    """Per-station sub-scaler identified by column-name suffix.

    Adapted from ``swissrivernetwork.util.scaler.StationSplitScaler``.

    Args:
        global_scaler: A **fitted** scaler with ``feature_names_in_``.
        feat_suffix: Suffix used to identify station columns (e.g. ``"_wt"``).
    """

    def __init__(
        self,
        global_scaler: Any,
        feat_suffix: str = '_wt',
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError('StationSplitScaler requires scikit-learn.')
        check_is_fitted(global_scaler)

        names = np.asarray(global_scaler.feature_names_in_, dtype=str)
        mask = np.char.endswith(names, feat_suffix)
        feat_idx = np.nonzero(mask)[0]
        feat_keys = [n.rstrip(feat_suffix) if feat_suffix else n for n in names[feat_idx]]

        self.feat_keys = feat_keys
        self._scalers: Dict[str, Any] = {}
        for key, idx in zip(feat_keys, feat_idx):
            self._scalers[key] = DimSplitScaler._single(global_scaler, idx)
        self._cur: Optional[str] = None

    def __getitem__(self, key: str) -> 'StationSplitScaler':
        if key not in self._scalers:
            raise KeyError(f"Station '{key}' not found. Available: {self.feat_keys}")
        self._cur = key
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._cur is None:
            raise ValueError("Set station via indexing first: scaler['station'].")
        return self._scalers[self._cur].transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._cur is None:
            raise ValueError("Set station via indexing first: scaler['station'].")
        return self._scalers[self._cur].inverse_transform(X)


# -----------------------------------------------------------------------
# EntityScaler  (was PerStationScaler)
# -----------------------------------------------------------------------


class EntityScaler:
    """Per-entity normalisation for multi-entity DataFrames.

    Fits an independent scaler per entity per feature suffix on **training
    data only**, matching the reference project's
    ``normalize_isolated_station`` approach.

    A global target scaler (fitted on all entities' target values pooled)
    is also maintained for batch-level ``inverse_transform`` of model
    predictions where entity identity is not available.

    Parameters
    ----------
    entity_ids : list[str]
        Entity identifier strings (e.g. station ids ``['2091', '2143']``,
        sensor ids, etc.).
    scaler_type : str
        ``'minmax'``, ``'standard'``, or ``'none'``.
    feature_suffixes : list[str]
        Column suffixes for feature columns (e.g. ``['_at']``).
    target_suffixes : list[str]
        Column suffixes for target columns (e.g. ``['_wt']``).

    Example
    -------
    ::

        scaler = EntityScaler(entity_ids, 'minmax')
        scaler.fit(train_df)
        scaler.transform(train_df)
        scaler.transform(val_df)
        scaler.transform(test_df)

        # Batch-level denorm (in trainer):
        raw_preds = scaler.inverse_transform(normalized_preds)

        # Exact per-entity denorm:
        raw = scaler.target_scalers['2091'].inverse_transform(x)
    """

    def __init__(
        self,
        entity_ids: Sequence[str],
        scaler_type: str = 'minmax',
        feature_suffixes: Sequence[str] = ('_at',),
        target_suffixes: Sequence[str] = ('_wt',),
        create_global_scaler: bool = False,
    ) -> None:
        self.entity_ids = list(entity_ids)
        self.scaler_type = scaler_type.strip().lower()
        self.feature_suffixes = list(feature_suffixes)
        self.target_suffixes = list(target_suffixes)
        self.create_global_scaler = create_global_scaler

        # Per-entity, per-suffix fitted scalers
        # key = (entity_id, suffix) → scaler
        self._scalers: Dict[tuple[str, str], Any] = {}
        self._global_target_scaler: Optional[Any] = None
        self._fitted = False

    # -- backward compatibility -------------------------------------------

    @property
    def station_ids(self) -> list[str]:
        """Alias kept for backward compatibility."""
        return self.entity_ids

    @station_ids.setter
    def station_ids(self, value: list[str]) -> None:
        self.entity_ids = value

    # -- fit --------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> 'EntityScaler':
        """Fit per-entity scalers on training data.

        NaN values are excluded when fitting (matching reference project
        behaviour of ``dropna`` before scaler fit).
        """  # todo: is it better to use a single scaler with multiple features for all entities?
        if self.scaler_type == 'none':
            self._fitted = True
            return self

        all_suffixes = self.feature_suffixes + self.target_suffixes
        all_target_vals: list[np.ndarray] = []

        for entity in self.entity_ids:
            for suffix in all_suffixes:
                col = f'{entity}{suffix}'
                if col not in train_df.columns:
                    continue
                scaler = get_scaler(self.scaler_type)
                vals = train_df[col].values.reshape(-1, 1)
                mask = ~np.isnan(vals.ravel())
                if mask.any():
                    scaler.fit(vals[mask])
                else:
                    scaler.fit(np.zeros((1, 1)))  # degenerate fallback
                self._scalers[(entity, suffix)] = scaler

                # Collect target values for optional global scaler
                if self.create_global_scaler and suffix in self.target_suffixes and mask.any():
                    all_target_vals.append(vals[mask])

        # Optional global target scaler (pooled over all entities)
        if self.create_global_scaler and all_target_vals:
            global_vals = np.concatenate(all_target_vals, axis=0)
            self._global_target_scaler = get_scaler(self.scaler_type)
            self._global_target_scaler.fit(global_vals)

        self._fitted = True
        return self

    # -- transform --------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame in-place using fitted scalers.

        Columns matching ``{station}{suffix}`` are normalized.
        Returns the same DataFrame for chaining.
        """
        if self.scaler_type == 'none' or not self._fitted:
            return df

        all_suffixes = self.feature_suffixes + self.target_suffixes
        for entity in self.entity_ids:
            for suffix in all_suffixes:
                col = f'{entity}{suffix}'
                key = (entity, suffix)
                if col not in df.columns or key not in self._scalers:
                    continue
                vals = df[col].values.reshape(-1, 1)
                if not np.isnan(vals).all():
                    df[col] = self._scalers[key].transform(vals).ravel()
        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *train_df* and transform it in-place."""
        return self.fit(train_df).transform(train_df)

    # -- inverse_transform ------------------------------------------------

    def inverse_transform(
        self,
        data: Any,
        entity_ids: list[str] | None = None,
        **kwargs,
    ) -> Any:
        """Inverse-transform normalized target predictions per entity.

        Parameters
        ----------
        data : numpy.ndarray or torch.Tensor
            Shape ``(B, T, C)`` or ``(N, C)``.
        entity_ids : list[str] | None
            Per-sample entity identifiers (length ``B``).  When provided,
            each sample is inverse-transformed using its own entity's
            fitted scaler via :meth:`inverse_transform_entity` — this is
            the *correct* approach for TS-mode data where every sample
            belongs to exactly one entity.

            When *None* (legacy / ST-mode), the method falls back to
            column-based mapping.
        **kwargs
            Extra keyword arguments (e.g. ``timestamps``) forwarded
            from the trainer.  Currently unused by the base scaler but
            available for subclass overrides.

        Returns
        -------
        Same type and shape as *data*, in original scale.
        """
        if not self._fitted or self.scaler_type == 'none':
            return data

        try:
            import torch as _torch

            is_tensor = isinstance(data, _torch.Tensor)
        except ImportError:
            is_tensor = False

        if is_tensor:
            device = data.device
            arr = data.detach().cpu().numpy().copy()  # todo: can this part use torch operations instead of numpy?
        else:
            arr = np.array(data, dtype=np.float64)

        orig_shape = arr.shape

        # ----- per-sample entity_ids path (per-entity mode) ---------------------
        if entity_ids is not None:
            n_samples = arr.shape[0]
            if len(entity_ids) != n_samples:
                raise ValueError(f'entity_ids length ({len(entity_ids)}) != batch size ({n_samples})')
            result = np.empty_like(arr)
            for i, eid in enumerate(entity_ids):
                sample = arr[i]  # shape (T, C) or (C,)
                was_2d = sample.ndim == 2
                n_cols = sample.shape[-1]
                for c in range(n_cols):
                    suffix = (
                        self.target_suffixes[c] if c < len(self.target_suffixes) else self.target_suffixes[0]
                    )  # todo: Is this correct? Should have a kwarg input for this
                    col_data = sample[:, c : c + 1] if was_2d else sample[c : c + 1]
                    result_col = self.inverse_transform_entity(col_data, eid, suffix)
                    if was_2d:
                        result[i, :, c : c + 1] = result_col if result_col.ndim == 2 else result_col.reshape(-1, 1)
                    else:
                        result[i, c : c + 1] = result_col.ravel()[:1]
            if is_tensor:
                return _torch.from_numpy(result.astype(np.float32)).to(device)
            return result

        # ----- multi-channel-mode column mapping (entity_ids not provided) ---------
        # Build ordered list of per-entity target scalers
        entity_target_scalers: list[Any] = []
        for entity in self.entity_ids:
            for suffix in self.target_suffixes:
                key = (entity, suffix)
                if key in self._scalers:
                    entity_target_scalers.append(self._scalers[key])

        if not entity_target_scalers:
            # No per-entity scalers fitted — return data unchanged.
            if is_tensor:
                return _torch.from_numpy(arr.astype(np.float32)).to(device)
            return arr

        n_cols = arr.shape[-1]
        arr_2d = arr.reshape(-1, n_cols)

        if n_cols == len(entity_target_scalers):
            # multi-channel mode: each column corresponds to one entity target
            for col_idx, scaler in enumerate(entity_target_scalers):
                col = arr_2d[:, col_idx : col_idx + 1]
                arr_2d[:, col_idx : col_idx + 1] = scaler.inverse_transform(col)
        else:
            # per_entity mode without entity_ids: column count doesn't match.
            # Raise so that callers (e.g. predict()) can fall back gracefully
            # to normalised-scale predictions.
            raise ValueError(
                f'Column count ({n_cols}) does not match number of entity '
                f'target scalers ({len(entity_target_scalers)}). '
                f'Pass entity_ids for per-entity inverse transform.'
            )

        arr = arr_2d.reshape(orig_shape)

        if is_tensor:
            return _torch.from_numpy(arr.astype(np.float32)).to(device)
        return arr

    def inverse_transform_entity(
        self,
        data: np.ndarray,
        entity_id: str,
        suffix: str = '_wt',
    ) -> np.ndarray:
        """Exact per-entity inverse-transform.

        Parameters
        ----------
        data : numpy.ndarray
            Shape ``(N, 1)`` or ``(N,)``.
        entity_id : str
            Entity identifier.
        suffix : str
            Column suffix (default ``'_wt'``).
        """
        key = (entity_id, suffix)
        if key not in self._scalers:
            raise KeyError(
                f'No scaler for entity={entity_id!r}, suffix={suffix!r}. Available entities: {self.entity_ids}'
            )
        was_1d = data.ndim == 1
        arr = data.reshape(-1, 1) if was_1d else data
        result = self._scalers[key].inverse_transform(arr)
        return result.ravel() if was_1d else result

    # -- accessors --------------------------------------------------------

    @property
    def target_scalers(self) -> Dict[str, Any]:
        """Per-entity target scalers: ``entity_id → scaler``."""
        return {
            entity: self._scalers[(entity, suffix)]
            for entity in self.entity_ids
            for suffix in self.target_suffixes
            if (entity, suffix) in self._scalers
        }

    @property
    def feature_scalers(self) -> Dict[str, Any]:
        """Per-entity feature scalers: ``entity_id → scaler``."""
        return {
            entity: self._scalers[(entity, suffix)]
            for entity in self.entity_ids
            for suffix in self.feature_suffixes
            if (entity, suffix) in self._scalers
        }

    @property
    def global_target_scaler(self) -> Any:
        """Global target scaler (pooled over all entities), or ``None``."""
        return self._global_target_scaler


# Backward-compatible alias
PerStationScaler = EntityScaler


# -----------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {}


def _ensure_registry() -> None:
    global _REGISTRY
    if _REGISTRY:
        return
    _REGISTRY['none'] = NoScaler
    if _SKLEARN_AVAILABLE:
        _REGISTRY['standard'] = StandardScaler
        _REGISTRY['minmax'] = MinMaxScaler


def get_scaler(name: str = 'standard', **kwargs: Any) -> Any:
    """Create a scaler by name.

    Args:
        name: One of ``"standard"``, ``"minmax"``, ``"none"``.
        **kwargs: Forwarded to the scaler constructor.

    Returns:
        A fresh (unfitted) scaler instance.
    """
    _ensure_registry()
    key = name.strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown scaler '{name}'. Available: {sorted(_REGISTRY)}")
    return cls(**kwargs)
