"""
Data factory for creating PyTorch DataLoaders with proper dataset selection.

Supports two dataset styles:

1. **TimeSeriesDataset subclasses** (ETTHour, ETTMinute, CustomCSV, PEMS):
   multi-split containers where ``get_split(flag)`` returns a torch Dataset.
2. **Standalone torch Datasets** (M4Dataset): single-flag per instance.

The factory transparently handles both styles.

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_factory.py

MIT License
"""

from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from liulian.data.csv_dataset import (
    ETTHourDataset,
    ETTMinuteDataset,
    CustomCSVDataset,
)
from liulian.data.m4_dataset import M4Dataset
from liulian.data.pems_dataset import PEMSDataset
from liulian.data.base import BaseDataset


# Dataset registry mapping names to classes
DATASET_REGISTRY: Dict[str, type] = {
    # ETT family
    'ETTh1': ETTHourDataset,
    'ETTh2': ETTHourDataset,
    'ETTm1': ETTMinuteDataset,
    'ETTm2': ETTMinuteDataset,
    # Generic CSV
    'custom': CustomCSVDataset,
    # M4 competition
    'm4': M4Dataset,
    # Standard benchmarks (all loaded via CustomCSVDataset)
    'weather': CustomCSVDataset,
    'electricity': CustomCSVDataset,
    'traffic': CustomCSVDataset,
    'exchange_rate': CustomCSVDataset,
    'illness': CustomCSVDataset,
    'solar': CustomCSVDataset,
    # PEMS traffic sensors
    'PEMS03': PEMSDataset,
    'PEMS04': PEMSDataset,
    'PEMS07': PEMSDataset,
    'PEMS08': PEMSDataset,
}

# Cache for multi-split datasets to avoid re-loading per flag
_DATASET_CACHE: Dict[str, BaseDataset] = {}


def create_dataloader(
    data_name: str,
    root_path: str,
    data_path: str,
    flag: str = 'train',
    size: Optional[tuple] = None,
    features: str = 'M',
    target: str = 'OT',
    scale: bool = True,
    timeenc: int = 0,
    freq: str = 'h',
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
    **kwargs,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for time series forecasting.

    Args:
        data_name (str): Dataset name (e.g., `ETTh1`, `ETTm1`, `custom`).
        root_path (str): Root directory containing the data file.
        data_path (str): CSV filename or dataset identifier.
        flag (str): Split type - one of `train`, `val`, or `test`.
        size (tuple or None): Tuple `(seq_len, label_len, pred_len)` defining
            input and prediction window sizes. `None` to use dataset defaults.
        features (str): Feature mode. One of:  todo: make sure of the explanation.
            - `M`: multivariate input and multivariate output (use all variables).
            - `S`: univariate input and output (single series / target only).
            - `MS`: multivariate input with a single target for prediction.
        target (str): Name of the target column to predict.
        scale (bool): If True, apply StandardScaler normalization to inputs.
        timeenc (int): Time encoding mode (commonly `0` or `1`).
        freq (str): Frequency string used to build time features (e.g., `h`, `t`).
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
        shuffle (bool): Whether to shuffle the dataset (typically True for `train`).
        drop_last (bool): Whether to drop the last incomplete batch.
        **kwargs: Additional dataset-specific arguments passed to the dataset ctor.

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader for the requested split.

    Raises:
        ValueError: If `data_name` is not registered in the dataset registry.

    Examples:
        >>> train_loader = create_dataloader(
        ...     data_name='ETTh1',
        ...     root_path='./data/ETT',
        ...     data_path='ETTh1.csv',
        ...     flag='train',
        ...     size=(96, 48, 96),
        ...     batch_size=32,
        ...     shuffle=True,
        ... )

        >>> val_loader = create_dataloader(
        ...     data_name='custom',
        ...     root_path='./data',
        ...     data_path='my_data.csv',
        ...     flag='val',
        ...     features='S',
        ...     target='value',
        ...     batch_size=32,
        ...     shuffle=False,
        ... )
    """
    # Get dataset class from registry
    if data_name not in DATASET_REGISTRY:
        raise ValueError(
            f'Unknown dataset: {data_name}. '
            f'Available datasets: {list(DATASET_REGISTRY.keys())}'
        )

    dataset_class = DATASET_REGISTRY[data_name]

    # Build kwargs for constructor
    ctor_kwargs = dict(
        root_path=root_path,
        data_path=data_path,
        size=size,
        features=features,
        target=target,
        scale=scale,
        **kwargs,
    )

    # Multi-split datasets (TimeSeriesDataset subclasses)
    if isinstance(dataset_class, type) and issubclass(dataset_class, BaseDataset):
        # Cache key to avoid re-loading per flag
        cache_key = (
            f'{data_name}:{root_path}:{data_path}:{size}:{features}:{target}:{scale}'
        )
        if cache_key not in _DATASET_CACHE:
            # Filter out flag — these datasets create all splits at once
            ctor_kwargs.pop('flag', None)
            # Add timeenc/freq only for CSV datasets
            if hasattr(dataset_class, 'timeenc'):
                ctor_kwargs['timeenc'] = timeenc
                ctor_kwargs['freq'] = freq
            _DATASET_CACHE[cache_key] = dataset_class(**ctor_kwargs)

        container = _DATASET_CACHE[cache_key]
        dataset = container.get_split(flag)
    else:
        # Standalone datasets (M4Dataset etc.)
        ctor_kwargs['flag'] = flag
        ctor_kwargs['timeenc'] = timeenc
        ctor_kwargs['freq'] = freq
        dataset = dataset_class(**ctor_kwargs)

    # Determine shuffle based on flag if not explicitly set
    if flag == 'train' and shuffle is None:
        shuffle = True
    elif flag in ['val', 'test'] and shuffle is None:
        shuffle = False

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return dataloader


def create_dataloaders(
    data_name: str,
    root_path: str,
    data_path: str,
    size: Optional[tuple] = None,
    features: str = 'M',
    target: str = 'OT',
    scale: bool = True,
    timeenc: int = 0,
    freq: str = 'h',
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for a dataset.

    Convenience function to create all three splits at once with consistent settings.

    Args:
        data_name: Dataset name (e.g., 'ETTh1', 'ETTm1', 'custom')
        root_path: Root directory containing the data file
        data_path: CSV filename
        size: Tuple of (seq_len, label_len, pred_len)
        features: Feature mode - 'M', 'S', or 'MS'
        target: Target column name
        scale: Whether to apply StandardScaler normalization
        timeenc: Time encoding mode (0 or 1)
        freq: Frequency string for time features
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        **kwargs: Additional dataset-specific arguments

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataLoaders

    Examples:
        >>> loaders = create_dataloaders(
        ...     data_name='ETTh1',
        ...     root_path='./data/ETT',
        ...     data_path='ETTh1.csv',
        ...     size=(96, 48, 96),
        ...     batch_size=32,
        ... )
        >>> train_loader = loaders['train']
        >>> val_loader = loaders['val']
        >>> test_loader = loaders['test']
    """
    loaders = {}

    for flag in ['train', 'val', 'test']:
        # Train shuffles, val/test don't
        shuffle = flag == 'train'

        loaders[flag] = create_dataloader(
            data_name=data_name,
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            size=size,
            features=features,
            target=target,
            scale=scale,
            timeenc=timeenc,
            freq=freq,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
            **kwargs,
        )

    return loaders


def register_dataset(name: str, dataset_class: type):
    """
    Register a new dataset class in the registry.

    Allows extending the factory with custom dataset classes.

    Args:
        name: Name to register the dataset under
        dataset_class: Dataset class (must inherit from torch.utils.data.Dataset)

    Examples:
        >>> class MyCustomDataset(Dataset):
        ...     def __init__(self, root_path, data_path, flag, **kwargs):
        ...         pass
        >>> register_dataset('my_data', MyCustomDataset)
        >>> loader = create_dataloader('my_data', ...)
    """
    DATASET_REGISTRY[name] = dataset_class
