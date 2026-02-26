"""Backward-compatibility shim for DatasetCustom.

The class has been merged into :mod:`liulian.data.csv_dataset` as
:class:`CustomCSVDataset`.  ``DatasetCustom`` is an alias.
"""

from liulian.data.csv_dataset import (  # noqa: F401
    CustomCSVDataset,
    DatasetCustom,
)
