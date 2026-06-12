"""Float64-aligned dataloader to match TSL's internal data handling.

This module creates a dataloader that exactly matches TSL's data_loader.py
behavior, using float64 internally (like sklearn.StandardScaler) and only
converting to float32 at training time.

This is an EXPERIMENTAL module - changes here should NOT be merged to the
main pipeline without explicit user approval.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def time_features_from_dates(dates: pd.DatetimeIndex, freq: str = 'h') -> np.ndarray:
    """Extract time features exactly matching TSL's timefeatures.py.

    Returns float64 array matching TSL's behavior.
    
    TSL feature sets by frequency:
    - 'h' (hour): HourOfDay, DayOfWeek, DayOfMonth, DayOfYear (4 features)
    - 't' (minute): MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear (5 features)
    """
    features = []
    freq_lower = freq.lower().strip()
    
    if freq_lower == 't':
        # Minute frequency - 5 features
        features.append((dates.minute - 29.5) / 59)  # MinuteOfHour
        features.append((dates.hour - 11.5) / 23)  # HourOfDay
        features.append((dates.dayofweek - 3) / 6)  # DayOfWeek
        features.append((dates.day - 15.5) / 30)  # DayOfMonth
        features.append((dates.dayofyear - 182.5) / 365)  # DayOfYear
    else:
        # Hourly and other frequencies - 4 features (default)
        features.append((dates.hour - 11.5) / 23)  # HourOfDay
        features.append((dates.dayofweek - 3) / 6)  # DayOfWeek
        features.append((dates.day - 15.5) / 30)  # DayOfMonth
        features.append((dates.dayofyear - 182.5) / 365)  # DayOfYear

    return np.column_stack(features)  # float64 by default


class TSLAlignedDataset(Dataset):
    """Dataset that exactly matches TSL's Dataset_ETT_hour behavior.

    Key differences from Liulian's default:
    - Uses float64 internally (like sklearn.StandardScaler)
    - Matches TSL's exact border calculations
    - Returns numpy arrays, converted to tensors by DataLoader
    """

    def __init__(
        self,
        root_path: str,
        data_path: str = 'ETTh1.csv',
        flag: str = 'train',
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        features: str = 'M',
        target: str = 'OT',
        scale: bool = True,
        timeenc: int = 1,
        freq: str = 'h',
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # Read data
        df_raw = pd.read_csv(os.path.join(root_path, data_path))

        # Determine dataset type for borders
        if 'ETTh' in data_path:
            self._dataset_type = 'etth'
        elif 'ETTm' in data_path:
            self._dataset_type = 'ettm'
        else:
            self._dataset_type = 'custom'

        self._load_data(df_raw)

    def _compute_borders(self, n: int) -> Tuple[list, list]:
        """Compute borders matching TSL's exact implementation."""
        if self._dataset_type == 'etth':
            # ETT hourly: 12/4/4 months at hourly granularity
            border1s = [
                0,
                12 * 30 * 24 - self.seq_len,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ]
        elif self._dataset_type == 'ettm':
            # ETT minute: 12/4/4 months at 15-min granularity
            border1s = [
                0,
                12 * 30 * 24 * 4 - self.seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]
        else:
            # Custom: 70/10/20 split
            num_train = int(n * 0.7)
            num_test = int(n * 0.2)
            num_vali = n - num_train - num_test
            border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, n]
        return border1s, border2s

    def _load_data(self, df_raw: pd.DataFrame):
        """Load and preprocess data exactly matching TSL."""
        n = len(df_raw)
        border1s, border2s = self._compute_borders(n)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Feature columns
        if self.features in ('M', 'MS'):
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f'Unknown features mode: {self.features}')

        # Scale using training data (FLOAT64 - matching sklearn default)
        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)  # values are float64
            data = self.scaler.transform(df_data.values)  # returns float64
        else:
            data = df_data.values  # float64

        # Time features (FLOAT64)
        df_stamp = df_raw[['date']][border1:border2]
        dates = pd.to_datetime(df_stamp['date'].values)

        if self.timeenc == 0:
            # Manual encoding (TSL Dataset_ETT_hour default)
            df_stamp = df_stamp.copy()
            df_stamp['date'] = dates
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp[['month', 'day', 'weekday', 'hour']].values
        else:
            # Frequency-based encoding (timeenc=1)
            data_stamp = time_features_from_dates(
                pd.DatetimeIndex(dates), freq=self.freq
            )
            data_stamp = data_stamp  # Already transposed correctly

        # Store as FLOAT64 (matching TSL)
        self.data_x = data[border1:border2]  # float64
        self.data_y = data[border1:border2]  # float64
        self.data_stamp = data_stamp  # float64

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        """Return (seq_x, seq_y, seq_x_mark, seq_y_mark) as FLOAT64 numpy arrays.

        This matches TSL's Dataset_ETT_hour.__getitem__ exactly.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Return numpy arrays (float64) - TSL converts to tensor in DataLoader
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def create_tsl_aligned_dataloader(
    root_path: str,
    data_path: str,
    flag: str = 'train',
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 96,
    features: str = 'M',
    target: str = 'OT',
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    freq: str = 'h',
) -> DataLoader:
    """Create a DataLoader that exactly matches TSL's data loading behavior.

    Returns DataLoader with:
    - Float64 data internally (like TSL)
    - Exact same border calculations
    - Exact same time feature computation
    """
    dataset = TSLAlignedDataset(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features=features,
        target=target,
        freq=freq,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return loader


if __name__ == '__main__':
    # Quick test
    import random

    # Set seed
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)

    root_path = '../../dataset/ETT-small/'
    data_path = 'ETTh1.csv'

    loader = create_tsl_aligned_dataloader(
        root_path=root_path,
        data_path=data_path,
        flag='train',
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )

    batch = next(iter(loader))
    seq_x, seq_y, seq_x_mark, seq_y_mark = batch

    print(f'seq_x dtype: {seq_x.dtype}, shape: {seq_x.shape}')
    print(f'seq_x sum: {seq_x.sum().item():.10f}')
    print(f'First values: {seq_x[0, 0, :3]}')
