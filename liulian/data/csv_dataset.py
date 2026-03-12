"""CSV-based spatiotemporal datasets built on :class:`SpatialTempoDataset`.

Provides a unified hierarchy that replaces the duplicated TSLib-style
dataset classes (``ETTHourDataset``, ``ETTMinuteDataset``,
``CustomCSVDataset``, ``DatasetCustom``) while remaining fully
backward-compatible with the 4-tuple contract
``(seq_x, seq_y, seq_x_mark, seq_y_mark)`` expected by all models.

Class hierarchy::

    BaseDataset (ABC)
    └── TimeSeriesDataset
        └── SpatialTempoDataset  (adds graph_mode / edge_index / adj_matrix)
            └── CSVTimeSeriesDataset  (adds CSV loading, borders, time features)
                ├── ETTHourDataset   (12/4/4-month hourly borders)
                ├── ETTMinuteDataset (12/4/4-month 15-min borders)
                └── CustomCSVDataset (ratio-based borders — traffic, electricity, …)

Key features:

* Inherits all windowing / gap / noise / entity logic from
  :class:`TimeSeriesDataset`.
* Adds ``scaler_type`` support for configurable normalization.
* ``label_len`` (decoder warmup) is forwarded to :class:`TimeSeriesSplit`
  for Transformer-style models.
* Time features are generated and passed as the time column so that
  ``TimeSeriesSplit.__getitem__`` returns proper ``x_mark`` / ``y_mark``.

seq_len / label_len / pred_len
------------------------------
These three parameters define the sliding-window geometry, following the
Time-Series-Library (TSL) convention:

* ``seq_len``   — encoder input length (look-back context).
* ``pred_len``  — prediction horizon length.
* ``label_len`` — decoder warm-up overlap (Transformer models only).

Window layout in ``__getitem__``::

    |--------------seq_len--------------|
                        |---label_len---|---pred_len---|
                                        ^
                               prediction starts here

    seq_x = data[i : i + seq_len]                   # encoder input
    seq_y = data[i + seq_len - label_len :
                 i + seq_len + pred_len]             # decoder target

The last ``label_len`` rows of the encoder input overlap with the first
``label_len`` rows of the decoder target.  At training time, the
experiment code constructs:

    dec_inp = cat([batch_y[:, :label_len, :], zeros(pred_len)])

This gives the Transformer decoder ground-truth warm-up tokens followed
by zero placeholders that it must learn to predict.

**Encoder-only models** (DLinear, PatchTST, etc.) accept ``x_dec`` in
their ``forward()`` signature for API compatibility, but **completely
ignore it**.  Setting ``label_len=0`` for these models is recommended to
avoid constructing unused decoder inputs.  There is no data-leakage risk
either way — the overlap region is already seen by the encoder.

features modes ('M' / 'MS' / 'S')
----------------------------------
See the inline comments in :meth:`CSVTimeSeriesDataset._load_and_split`
and in :func:`liulian.pipeline.build_model` for detailed explanations.

Migration guide::

    # Old (TSLib):
    dataset = ETTHourDataset(root_path, data_path, flag='train', ...)
    item = dataset[0]

    # New (unified):
    dataset = ETTHourDataset(root_path, data_path, ...)
    split = dataset.get_split('train')  # torch Dataset
    item = split[0]

    # Or via data_factory (unchanged):
    loader = create_dataloader('ETTh1', root_path, data_path, flag='train', ...)
"""

from __future__ import annotations

import os
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from liulian.data.st.spatialtempodataset import SpatialTempoDataset

try:
    from liulian.utils.timefeatures import time_features
except ImportError:
    time_features = None


# ---------------------------------------------------------------------------
# Time feature helpers
# ---------------------------------------------------------------------------


def _time_features_from_dates(
    dates: pd.DatetimeIndex,
    freq: str = 'h',
    timeenc: int = 0,
) -> np.ndarray:
    """Extract time features from a datetime index.

    Two encoding modes are supported, matching the Time-Series-Library
    (TSL) ``data_factory.py`` convention:

    ``timeenc=1`` (*default in TSL* — used when ``embed='timeF'``)
        Delegates to :func:`liulian.utils.timefeatures.time_features`,
        which produces **continuous features normalized to [-0.5, 0.5]**
        via GluonTS-style feature extractors.  For hourly data this
        yields 4 dimensions: HourOfDay, DayOfWeek, DayOfMonth,
        DayOfYear.  This is the **recommended** encoding and the one
        used by all standard TSL benchmark scripts.

    ``timeenc=0`` (categorical / manual encoding)
        Produces hand-crafted normalized features in [-0.5, 0.5]:
        ``month/12-0.5``, ``day/31-0.5``, ``weekday/6-0.5``,
        ``hour/23-0.5`` (plus ``minute/59-0.5`` for minutely data).

        .. note::

           TSL's ``timeenc=0`` uses **raw integers** (month 1-12,
           day 1-31 etc.) instead of normalized values. Our version
           normalizes to [-0.5, 0.5] for better neural-network
           compatibility.  This is an intentional deviation — see
           ``docs/datasets.md`` for details.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Datetime values.
    freq : str
        Frequency hint (``'h'``, ``'t'``/``'min'``, etc.).
    timeenc : int
        0 = categorical (normalized), 1 = ``time_features()`` (default).

    Returns
    -------
    np.ndarray
        Shape ``(len(dates), n_features)``.
    """
    if timeenc == 1 and time_features is not None:
        tf = time_features(pd.to_datetime(dates), freq=freq)
        if isinstance(tf, torch.Tensor):
            return tf.transpose(1, 0).numpy()
        return tf.transpose(1, 0)

    # Categorical encoding (timeenc == 0 or fallback).
    # Normalized to [-0.5, 0.5] — see docstring for TSL deviation note.
    features = [
        dates.month / 12.0 - 0.5,
        dates.day / 31.0 - 0.5,
        dates.weekday / 6.0 - 0.5,
        dates.hour / 23.0 - 0.5,
    ]
    if freq in ('t', 'min'):
        features.append(dates.minute / 59.0 - 0.5)
    return np.column_stack(features).astype(np.float32)


# ---------------------------------------------------------------------------
# CSVTimeSeriesDataset — base for all CSV datasets
# ---------------------------------------------------------------------------


class CSVTimeSeriesDataset(SpatialTempoDataset):
    """Base class for CSV-backed spatiotemporal datasets.

    Inherits from :class:`SpatialTempoDataset`, which adds graph /
    spatial structure (``graph_mode``, ``edge_index``, ``adj_matrix``)
    on top of :class:`TimeSeriesDataset`.

    Loads a CSV file, splits into train/val/test based on row borders,
    optionally applies scaling, and produces time-mark features.
    Subclasses only need to override :meth:`_compute_borders` to define
    split boundaries.

    Parameters
    ----------
    root_path : str
        Directory containing the CSV file.
    data_path : str
        CSV filename.
    size : tuple[int, int, int] | None
        ``(seq_len, label_len, pred_len)``.
    features : str
        ``'M'``, ``'S'``, or ``'MS'``.
    target : str
        Target column name.
    scale : bool
        Apply StandardScaler.
    scaler_type : str
        Scaler name (``'standard'``, ``'minmax'``, ``'none'``).
        ``scale=True`` defaults to ``'standard'``; ``scale=False``
        forces ``'none'``.
    timeenc : int
        Time encoding mode (0 or 1).
    freq : str
        Frequency string for time features.
    graph_mode : str
        Passed to :class:`SpatialTempoDataset`.
        ``'none'`` | ``'edge_index'`` | ``'adj_matrix'``.
    graph_metadata : dict | None
        Passed to :class:`SpatialTempoDataset`.

    Any additional keyword arguments are forwarded through the
    ``SpatialTempoDataset → TimeSeriesDataset`` chain.
    """

    domain: str = 'spatiotemporal'
    version: str = '2.0'

    def __init__(
        self,
        root_path: str,
        data_path: str,
        *,
        size: Tuple[int, int, int] | None = None,
        features: str = 'M',
        target: str = 'OT',
        scale: bool = True,
        scaler_type: str | None = None,
        timeenc: int = 0,
        freq: str = 'h',
        **kwargs,
    ) -> None:
        # Parse size
        if size is None:
            seq_len, label_len, pred_len = 96, 48, 96
        else:
            seq_len, label_len, pred_len = size

        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        # Store seq_len early so _compute_borders can use it
        self.seq_len = seq_len
        # Store label_len for later
        self._label_len = label_len

        # Resolve scaler_type: explicit > inferred from scale flag
        if scaler_type is not None:
            resolved_scaler = scaler_type
        elif scale:
            resolved_scaler = 'standard'
        else:
            resolved_scaler = 'none'

        # Load and split data
        splits, feature_cols, target_cols, time_col, tf_cols = self._load_and_split()

        # Auto-detect station_ids from feature columns if not provided.
        #
        # For multi-entity CSV datasets (traffic, electricity, etc.),
        # each data column represents a sensor / client / entity.  In
        # features='M' or 'MS' mode, *all* non-date columns (including
        # the target column 'OT') are treated as data channels.
        #
        # TSL convention: enc_in = dec_in = c_out = num_data_columns.
        # For traffic: 861 sensor columns + 1 'OT' column = 862 channels.
        # 'OT' is simply the last column (a road-occupancy sensor like
        # the others) designated as the "target" for MS mode, but in M
        # mode all 862 channels are predicted equally.
        #
        # The station_ids list is used for:
        #   1. Entity embedding lookup (num_embeddings = len(station_ids))
        #   2. Per-entity inverse transform
        #   3. Per-channel reporting in visualization
        if 'station_ids' not in kwargs:
            if len(feature_cols) > 1:
                kwargs['station_ids'] = feature_cols

        # Initialize parent (handles scaler + split building)
        super().__init__(
            splits=splits,
            time_col=time_col,
            feature_cols=feature_cols,
            target_cols=target_cols,
            seq_len=seq_len,
            pred_len=pred_len,
            task='forecast',
            label_len=label_len,
            scaler_type=resolved_scaler,
            time_feature_cols=tf_cols,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------

    def _load_and_split(
        self,
    ) -> Tuple[dict[str, pd.DataFrame], list[str], list[str], str, list[str]]:
        """Load CSV and create split DataFrames.

        Returns
        -------
        splits : dict[str, DataFrame]
        feature_cols : list[str]
        target_cols : list[str]
        time_col : str
        time_feature_cols : list[str]
        """
        filepath = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(filepath)

        # Ensure 'date' column exists
        if 'date' not in df_raw.columns:
            raise ValueError(
                f'CSV must contain a "date" column. '
                f'Found columns: {list(df_raw.columns)}'
            )

        # Reorder columns: date, ...features, target
        cols = list(df_raw.columns)
        cols.remove('date')
        if self.target in cols:  # Ensure target is last for convenience (not strictly required)
            cols.remove(self.target)
            cols.append(self.target)
        df_raw = df_raw[['date'] + cols]

        # ── Feature / target column selection ──────────────────────────
        #
        # TSL features modes (from Time-Series-Library data_loader.py):
        #
        #   'M'  (Multivariate → Multivariate):
        #        Input = ALL non-date columns.  Output = ALL columns.
        #        enc_in = dec_in = c_out = len(data_cols).
        #        Example: traffic (862 cols → predict all 862).
        #
        #   'MS' (Multivariate → Single):
        #        Input = ALL non-date columns.  Output = only target col.
        #        enc_in = dec_in = c_out = len(data_cols) — the model
        #        still outputs all channels, but the trainer uses
        #        f_dim=-1 to select only the last column (target) for
        #        the loss.  This is the standard TSL convention.
        #        Example: electricity (321 cols in → predict only 'OT').
        #
        #   'S'  (Single → Single):
        #        Input = only target column.  Output = target column.
        #        enc_in = dec_in = c_out = 1.
        #        Example: univariate forecasting of 'OT' only.
        #
        # In all modes, the target column is reordered to be LAST in
        # the DataFrame (see column reordering above), so f_dim=-1
        # always selects it correctly in MS mode.
        if self.features in ('M', 'MS'):
            data_cols = [c for c in df_raw.columns if c != 'date']
        elif self.features == 'S':
            data_cols = [self.target]
        else:
            raise ValueError(f'Unsupported features mode: {self.features}')

        # Compute split borders (subclass-specific)
        n = len(df_raw)
        border1s, border2s = self._compute_borders(n)

        # Build time features
        dates = pd.DatetimeIndex(pd.to_datetime(df_raw['date']))
        time_feats = _time_features_from_dates(
            dates,
            freq=self.freq,
            timeenc=self.timeenc,
        )
        time_feat_cols = [f'_tf_{i}' for i in range(time_feats.shape[1])]

        # Build per-split DataFrames
        splits: dict[str, pd.DataFrame] = {}
        split_names = ['train', 'val', 'test']
        for i, split_name in enumerate(split_names):
            b1, b2 = border1s[i], border2s[i]
            sub = df_raw.iloc[b1:b2].copy().reset_index(drop=True)
            # Add monotonic time index for gap detection
            sub['_time_idx'] = np.arange(b1, b2)
            # Add time features
            tf_sub = time_feats[b1:b2]
            for j, tc in enumerate(time_feat_cols):
                sub[tc] = tf_sub[:, j]
            splits[split_name] = sub

        # In TSLib convention, feature_cols = target_cols = all data columns
        # Models decide which is target at inference time
        feature_cols = data_cols
        target_cols = data_cols

        return splits, feature_cols, target_cols, '_time_idx', time_feat_cols

    def _compute_borders(
        self,
        n: int,
    ) -> Tuple[list[int], list[int]]:
        """Return ``(border1s, border2s)`` for train/val/test splits.

        Each split is defined by a half-open interval
        ``[border1[i], border2[i])`` of row indices into the original
        DataFrame.

        **Border offset trick** (from TSL ``Dataset_Custom``):

        Val and test splits start at ``X - seq_len`` instead of ``X``
        so that the first sliding window in each split can look back
        ``seq_len`` rows into the *preceding* split's data for context.
        Without this offset the first few windows would lack sufficient
        history::

            |<------ train ------>|<------ val ------>|<--- test --->|
            0             num_train                              n
                          ↑
                    val border1 = num_train - seq_len
                    (allows first val window to see seq_len context)

        Must be overridden by subclasses with specific split logic.
        Default: 70/10/20 ratio split.
        """
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
        border2s = [num_train, n - num_test, n]
        return border1s, border2s


# ---------------------------------------------------------------------------
# ETTHourDataset
# ---------------------------------------------------------------------------


class ETTHourDataset(CSVTimeSeriesDataset):
    """ETT hour-level dataset (ETTh1, ETTh2).

    Hardcoded 12/4/4-month borders at hourly granularity.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str = 'ETTh1.csv',
        *,
        flag: str | None = None,  # accepted for backward compat; ignored
        size: Tuple[int, int, int] | None = None,
        features: str = 'M',
        target: str = 'OT',
        scale: bool = True,
        scaler_type: str | None = None,
        timeenc: int = 0,
        freq: str = 'h',
        **kwargs,
    ) -> None:
        super().__init__(
            root_path=root_path,
            data_path=data_path,
            size=size,
            features=features,
            target=target,
            scale=scale,
            scaler_type=scaler_type,
            timeenc=timeenc,
            freq=freq,
            **kwargs,
        )

    def _compute_borders(self, n: int):
        # ETT hourly: fixed 12/4/4-month calendar split.
        # At 24 hours/day × 30 days/month:
        #   train = rows [0, 8640)           — months 1–12
        #   val   = rows [8640-seq_len, 12480) — months 13–16 (with context)
        #   test  = rows [12480-seq_len, 17280) — months 17–20 (with context)
        b = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        # Cap at actual data length
        b = [min(v, n) for v in b]
        border1s = [b[0], max(b[1] - self.seq_len, 0), max(b[2] - self.seq_len, 0)]
        border2s = [b[1], b[2], b[3]]
        return border1s, border2s


# ---------------------------------------------------------------------------
# ETTMinuteDataset
# ---------------------------------------------------------------------------


class ETTMinuteDataset(CSVTimeSeriesDataset):
    """ETT minute-level dataset (ETTm1, ETTm2).

    Hardcoded 12/4/4-month borders at 15-minute granularity.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str = 'ETTm1.csv',
        *,
        flag: str | None = None,
        size: Tuple[int, int, int] | None = None,
        features: str = 'M',
        target: str = 'OT',
        scale: bool = True,
        scaler_type: str | None = None,
        timeenc: int = 0,
        freq: str = 't',
        **kwargs,
    ) -> None:
        super().__init__(
            root_path=root_path,
            data_path=data_path,
            size=size,
            features=features,
            target=target,
            scale=scale,
            scaler_type=scaler_type,
            timeenc=timeenc,
            freq=freq,
            **kwargs,
        )

    def _compute_borders(self, n: int):
        # ETT 15-minute: fixed 12/4/4-month calendar split.
        # At 4 samples/hour × 24 hours/day × 30 days/month = 2880/month:
        #   train = rows [0, 34560)            — months 1–12
        #   val   = rows [34560-seq_len, 46080) — months 13–16
        #   test  = rows [46080-seq_len, 57600) — months 17–20
        m = 4  # 4x hourly
        b = [
            0,
            12 * 30 * 24 * m,
            12 * 30 * 24 * m + 4 * 30 * 24 * m,
            12 * 30 * 24 * m + 8 * 30 * 24 * m,
        ]
        b = [min(v, n) for v in b]
        border1s = [b[0], max(b[1] - self.seq_len, 0), max(b[2] - self.seq_len, 0)]
        border2s = [b[1], b[2], b[3]]
        return border1s, border2s


# ---------------------------------------------------------------------------
# CustomCSVDataset
# ---------------------------------------------------------------------------


class CustomCSVDataset(CSVTimeSeriesDataset):
    """Generic CSV dataset with ratio-based splits (70/20/10).

    Works for weather, electricity, traffic, exchange_rate, illness,
    solar, and any other CSV with a ``date`` column.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        *,
        flag: str | None = None,
        size: Tuple[int, int, int] | None = None,
        features: str = 'S',
        target: str = 'OT',
        scale: bool = True,
        scaler_type: str | None = None,
        timeenc: int = 0,
        freq: str = 'h',
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        **kwargs,
    ) -> None:
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        super().__init__(
            root_path=root_path,
            data_path=data_path,
            size=size,
            features=features,
            target=target,
            scale=scale,
            scaler_type=scaler_type,
            timeenc=timeenc,
            freq=freq,
            **kwargs,
        )

    def _compute_borders(self, n: int):
        # Ratio-based split matching TSL Dataset_Custom:
        #   num_vali = n - num_train - num_test  (the remainder)
        #   train: [0, num_train)
        #   val:   [num_train - seq_len, num_train + num_vali)
        #   test:  [n - num_test - seq_len, n)
        #
        # The -seq_len offset on val/test border1 allows the first
        # sliding window in each split to have seq_len context rows
        # from the preceding split.  See base class docstring.
        num_train = int(n * self.train_ratio)
        num_test = int(n * self.test_ratio)
        border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
        border2s = [num_train, n - num_test, n]
        return border1s, border2s


# Backward-compatible alias
DatasetCustom = CustomCSVDataset
