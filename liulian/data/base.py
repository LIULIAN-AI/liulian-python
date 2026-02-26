"""Abstract base classes for datasets and data splits.

Every concrete dataset adapter (core or plugin) inherits from
:class:`BaseDataset` and returns :class:`DataSplit` instances for
train / val / test partitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from liulian.data.backend import ArrayBackend, get_backend
from liulian.data.spec import FieldSpec, TopologySpec


class DataSplit:
    """A single partition (train / val / test) of a dataset.

    Holds the feature and target arrays and provides batch sampling.
    Arrays may be NumPy arrays or PyTorch tensors depending on the
    active :class:`~liulian.data.backend.ArrayBackend`.

    Attributes:
        X: Feature array with shape ``(n_samples, n_timesteps, n_features)``.
        y: Target array with shape ``(n_samples, horizon, n_targets)``.
        name: Split name (``"train"``, ``"val"``, ``"test"``).
    """

    def __init__(
        self,
        X: Any,
        y: Any,
        name: str = 'train',
    ) -> None:
        self.X = X
        self.y = y
        self.name = name

    def get_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random batch of ``(X, y)`` pairs.

        Args:
            batch_size: Number of samples to return.  Clamped to dataset size.

        Returns:
            Tuple of ``(X_batch, y_batch)`` arrays.
        """
        n = min(batch_size, len(self.X))
        idx = np.random.choice(len(self.X), size=n, replace=False)
        return self.X[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:
        return f"DataSplit(name='{self.name}', samples={len(self)})"


class BaseDataset(ABC):
    """Abstract dataset interface.

    Subclasses must set ``domain`` and ``version`` and implement
    :meth:`get_split`.

    Attributes:
        domain: Short identifier for the domain (e.g. ``"hydrology"``).
        version: Semantic version of the dataset.
        manifest: Parsed manifest dictionary (provenance info).
        topology: Optional spatial/graph topology.
        fields: List of field specifications.
    """

    domain: str = ''
    version: str = ''

    def __init__(
        self,
        manifest: Optional[Dict[str, Any]] = None,
        topology: Optional[TopologySpec] = None,
        fields: Optional[List[FieldSpec]] = None,
        backend: Union[str, ArrayBackend] = 'numpy',
    ) -> None:
        self.manifest = manifest or {}
        self.topology = topology
        self.fields = fields or []
        self._backend = get_backend(backend)

    # --- Array backend ---------------------------------------------------

    @property
    def B(self) -> ArrayBackend:
        """Active array backend."""
        return self._backend

    @property
    def backend_name(self) -> str:
        """Name of the active backend (``'numpy'`` or ``'torch'``)."""
        return self._backend.name

    def _finalize_split(self, split: DataSplit) -> DataSplit:
        """Convert *split* arrays to the active backend format."""
        if self._backend.name == 'numpy':
            return split
        return DataSplit(
            X=self._backend.asarray(split.X),
            y=self._backend.asarray(split.y),
            name=split.name,
        )

    @abstractmethod
    def get_split(self, split_name: str) -> DataSplit:
        """Return the data split for the given partition name.

        Args:
            split_name: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            A :class:`DataSplit` instance.

        Raises:
            KeyError: If *split_name* is not available.
        """

    # --- Scaler / inverse transform --------------------------------------

    def inverse_transform(self, data):
        """Inverse-transform normalized predictions back to original scale.

        The default implementation is an identity pass-through.  Subclasses
        that apply normalisation should override this method to delegate to
        their scaler.

        Parameters
        ----------
        data : numpy.ndarray or torch.Tensor
            Model predictions (any shape).

        Returns
        -------
        Same type and shape, in original (un-normalized) scale.
        """
        return data

    def info(self) -> Dict[str, Any]:
        """Return dataset metadata summary.

        Returns:
            Dictionary with keys such as ``"domain"``, ``"version"``,
            ``"fields"``, ``"topology"``.
        """
        return {
            'domain': self.domain,
            'version': self.version,
            'fields': [f._asdict() for f in self.fields],
            'has_topology': self.topology is not None,
        }
