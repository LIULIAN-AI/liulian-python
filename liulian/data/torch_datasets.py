"""Backward-compatibility shim for TSLib-style dataset classes.

All classes have been migrated to :mod:`liulian.data.csv_dataset`,
which builds on the unified :class:`TimeSeriesDataset` hierarchy.
This module re-exports them under the original names so that existing
imports continue to work.
"""

from liulian.data.csv_dataset import (  # noqa: F401
    CSVTimeSeriesDataset,
    CustomCSVDataset,
    DatasetCustom,
    ETTHourDataset,
    ETTMinuteDataset,
)
