"""CSV-based time-series datasets built on :class:`TimeSeriesDataset`.

Provides a unified hierarchy that replaces the duplicated TSLib-style
dataset classes (``ETTHourDataset``, ``ETTMinuteDataset``,
``CustomCSVDataset``, ``DatasetCustom``) while remaining fully
backward-compatible with the 4-tuple contract
``(seq_x, seq_y, seq_x_mark, seq_y_mark)`` expected by all models.

Key features:

* Inherits all windowing / gap / noise / entity logic from
  :class:`TimeSeriesDataset`.
* Adds ``scaler_type`` support for configurable normalisation.
* ``label_len`` (decoder warmup) is forwarded to :class:`TimeSeriesSplit`
  for Transformer-style models.
* Time features are generated and passed as the time column so that
  ``TimeSeriesSplit.__getitem__`` returns proper ``x_mark`` / ``y_mark``.

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

from liulian.data.ts.timeseriesdataset import TimeSeriesDataset, TimeSeriesSplit

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
    """Extract time features from datetime index.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Datetime values.
    freq : str
        Frequency hint (``'h'``, ``'t'``/``'min'``, etc.).
    timeenc : int
        0 = categorical encoding, 1 = ``time_features()`` from utils.

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

    # Categorical encoding (timeenc == 0 or fallback)
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


class CSVTimeSeriesDataset(TimeSeriesDataset):
    """Base class for CSV-backed time-series datasets.

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
    """

    domain: str = 'timeseries'
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
                f"CSV must contain a 'date' column. "
                f'Found columns: {list(df_raw.columns)}'
            )

        # Reorder columns: date, ...features, target
        cols = list(df_raw.columns)
        cols.remove('date')
        if self.target in cols:
            cols.remove(self.target)
            cols.append(self.target)
        df_raw = df_raw[['date'] + cols]

        # Feature / target selection
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
        dates = pd.to_datetime(df_raw['date'])
        time_feats = _time_features_from_dates(
            dates.dt, freq=self.freq, timeenc=self.timeenc,
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
        self, n: int,
    ) -> Tuple[list[int], list[int]]:
        """Return ``(border1s, border2s)`` for train/val/test.

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
        # 12 months train, 4 months val, 4 months test (hourly)
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
        # 12 months train, 4 months val, 4 months test (15-min)
        m = 4  # 4x hourly
        b = [0, 12*30*24*m, 12*30*24*m + 4*30*24*m, 12*30*24*m + 8*30*24*m]
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
        num_train = int(n * self.train_ratio)
        num_test = int(n * self.test_ratio)
        border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
        border2s = [num_train, n - num_test, n]
        return border1s, border2s


# Backward-compatible alias
DatasetCustom = CustomCSVDataset
