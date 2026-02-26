"""Data layer — dataset abstractions, manifest management, and topology specs."""

from liulian.data.backend import (
    ArrayBackend,
    NumpyBackend,
    TorchBackend,
    get_backend,
    register_backend,
    with_backend,
)
from liulian.data.base import BaseDataset, DataSplit
from liulian.data.manifest import load_manifest, validate_manifest
from liulian.data.noise import add_noise_to_array, add_noise_to_dataframe
from liulian.data.prompt_bank import load_content
from liulian.data.scalers import (
    DimSplitScaler,
    EntityScaler,
    NoScaler,
    PerStationScaler,
    StationSplitScaler,
    get_scaler,
)
from liulian.data.spec import FieldSpec, TopologySpec

# Middle interfaces — always available (numpy-only)
from liulian.data.ts.timeseriesdataset import TimeSeriesDataset, TimeSeriesSplit
from liulian.data.st.spatialtempodataset import SpatialTempoDataset

# CSV / PEMS datasets (TimeSeriesDataset subclasses)
from liulian.data.csv_dataset import (
    CSVTimeSeriesDataset,
    CustomCSVDataset,
    ETTHourDataset,
    ETTMinuteDataset,
)
from liulian.data.pems_dataset import PEMSDataset

# Backward-compat alias
DatasetCustom = CustomCSVDataset

# Torch-dependent modules — import lazily to avoid hard dependency
try:
    from liulian.data.swiss_river import SwissRiverDataset
    from liulian.data.seq_dataset import (
        SequenceDataset,
        SequenceFullDataset,
        SequenceWindowedDataset,
        add_noise_to_array,
    )
except ImportError:  # torch not installed
    pass

__all__ = [
    # Backend
    'ArrayBackend',
    'NumpyBackend',
    'TorchBackend',
    'get_backend',
    'register_backend',
    'with_backend',
    # Core
    'BaseDataset',
    'DataSplit',
    'FieldSpec',
    'TopologySpec',
    'TimeSeriesDataset',
    'TimeSeriesSplit',
    'SpatialTempoDataset',
    'SwissRiverDataset',
    # CSV / PEMS datasets
    'CSVTimeSeriesDataset',
    'CustomCSVDataset',
    'ETTHourDataset',
    'ETTMinuteDataset',
    'PEMSDataset',
    'DatasetCustom',
    # Sequence datasets
    'SequenceDataset',
    'SequenceFullDataset',
    'SequenceWindowedDataset',
    # Scalers
    'EntityScaler',
    'PerStationScaler',
    'StationSplitScaler',
    'DimSplitScaler',
    'NoScaler',
    'get_scaler',
    # Utilities
    'add_noise_to_array',
    'add_noise_to_dataframe',
    'load_manifest',
    'validate_manifest',
    'load_content',
]
