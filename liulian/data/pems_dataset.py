"""PEMS traffic sensor datasets built on :class:`TimeSeriesDataset`.

Loads PEMS03/04/07/08 from ``.npz`` files containing shape ``(T, N, F)``
arrays (time steps x sensors x features).  Each ``.npz`` file has a
single ``'data'`` key.

| Dataset | Sensors | Features | Time steps |
|---------|---------|----------|-----------|
| PEMS03  |     358 |        1 |    26 208 |
| PEMS04  |     307 |        3 |    16 992 |
| PEMS07  |     883 |        1 |    28 224 |
| PEMS08  |     170 |        3 |    17 856 |

The first feature (flow) is used by default.  All sensors are treated
as channels (enc_in = N).

Backward-compatible: ``_StandardScaler`` is kept as a private helper for
reference, but the dataset now uses the unified ``get_scaler`` factory.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from liulian.data.ts.timeseriesdataset import TimeSeriesDataset, TimeSeriesSplit


class _StandardScaler:
    """Minimal z-score scaler (no sklearn dependency).

    Kept for reference/testing.  The dataset itself now uses
    :func:`~liulian.data.scalers.get_scaler`.
    """

    def fit(self, data: np.ndarray) -> '_StandardScaler':
        self.mean_ = data.mean(axis=0)
        self.scale_ = data.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean_) / self.scale_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.scale_ + self.mean_


class PEMSDataset(TimeSeriesDataset):
    """PEMS traffic sensor dataset built on :class:`TimeSeriesDataset`.

    Parameters
    ----------
    root_path : str
        Directory containing ``.npz`` files (e.g. ``dataset/PEMS``).
    data_path : str
        Filename (e.g. ``'PEMS03.npz'``).
    size : tuple[int, int, int] | None
        ``(seq_len, label_len, pred_len)``.  Defaults to ``(96, 48, 96)``.
    features : str
        ``'M'`` (multivariate -> multivariate).
    target : str
        Ignored — all sensors are used.
    scale : bool
        Apply StandardScaler per sensor.
    scaler_type : str | None
        Explicit scaler name.  Overrides *scale* flag.
    timeenc : int
        Time encoding mode (only 0 supported).
    freq : str
        Frequency string (unused; PEMS has no timestamps).
    flag : str | None
        Accepted for backward compat with data_factory; ignored since
        all splits are created at once.
    """

    # 6:2:2 split
    _SPLIT_RATIOS = (0.6, 0.2, 0.2)

    domain: str = 'traffic'
    version: str = '2.0'

    def __init__(
        self,
        root_path: str,
        data_path: str = 'PEMS03.npz',
        *,
        flag: str | None = None,
        size: Optional[Tuple[int, int, int]] = None,
        features: str = 'M',
        target: str = 'OT',
        scale: bool = True,
        scaler_type: str | None = None,
        timeenc: int = 0,
        freq: str = 'h',
        **kwargs,
    ) -> None:
        if size is None:
            seq_len, label_len, pred_len = 96, 48, 96
        else:
            seq_len, label_len, pred_len = size

        # Resolve scaler
        if scaler_type is not None:
            resolved_scaler = scaler_type
        elif scale:
            resolved_scaler = 'standard'
        else:
            resolved_scaler = 'none'

        self._pems_root = root_path
        self._pems_file = data_path

        # Load .npz and build split DataFrames
        splits, feature_cols, tf_cols = self._load_npz(
            root_path,
            data_path,
            seq_len,
        )

        super().__init__(
            splits=splits,
            time_col='_time_idx',
            feature_cols=feature_cols,
            target_cols=feature_cols,  # all sensors are both input and target
            seq_len=seq_len,
            pred_len=pred_len,
            task='forecast',
            label_len=label_len,
            scaler_type=resolved_scaler,
            time_feature_cols=tf_cols,
            **kwargs,
        )

    def _load_npz(
        self,
        root_path: str,
        data_path: str,
        seq_len: int,
    ) -> Tuple[dict[str, pd.DataFrame], list[str], list[str]]:
        """Load PEMS .npz and return split DataFrames."""
        raw: np.ndarray = np.load(
            os.path.join(root_path, data_path),
            allow_pickle=True,
        )['data'].astype(np.float32)

        # Use first feature only (flow) → (T, N)
        if raw.ndim == 3:
            data = raw[:, :, 0]
        else:
            data = raw

        T, N = data.shape

        # Split boundaries
        train_end = int(T * self._SPLIT_RATIOS[0])
        val_end = train_end + int(T * self._SPLIT_RATIOS[1])

        borders = {
            'train': (0, train_end),
            'val': (train_end - seq_len, val_end),
            'test': (val_end - seq_len, T),
        }

        # Column names for sensors
        sensor_cols = [f'sensor_{i}' for i in range(N)]

        # Build DataFrames per split
        splits: dict[str, pd.DataFrame] = {}
        for split_name, (lo, hi) in borders.items():
            chunk = data[lo:hi]
            df = pd.DataFrame(chunk, columns=sensor_cols)
            # Monotonic time index for gap detection
            df['_time_idx'] = np.arange(lo, hi)
            # Synthetic time features
            idx_norm = np.arange(lo, hi, dtype=np.float32) / max(T, 1)
            df['_tf_0'] = idx_norm
            df['_tf_1'] = np.sin(idx_norm * 2 * np.pi)
            df['_tf_2'] = np.cos(idx_norm * 2 * np.pi)
            df['_tf_3'] = 0.0
            splits[split_name] = df

        tf_cols = ['_tf_0', '_tf_1', '_tf_2', '_tf_3']
        return splits, sensor_cols, tf_cols
