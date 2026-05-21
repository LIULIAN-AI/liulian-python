"""Array backend abstraction for transparent numpy / torch interop.

Template pattern
----------------
* :class:`ArrayBackend`  — abstract template defining array operations.
* :class:`NumpyBackend`  — concrete NumPy implementation.
* :class:`TorchBackend`  — concrete PyTorch implementation.
* :func:`with_backend`   — class decorator for transparent backend selection.

The ``@with_backend`` decorator injects a ``backend`` keyword argument
into any class's ``__init__``.  The chosen backend is accessible as
``self.B``::

    @with_backend
    class MyDataset:
        def forward(self):
            return self.B.zeros((3, 4))  # numpy or torch, transparently


    ds = MyDataset(backend='numpy')  # self.B → NumpyBackend
    ds = MyDataset(backend='torch')  # self.B → TorchBackend

For the :class:`~liulian.data.base.BaseDataset` hierarchy the backend
parameter is integrated directly into the class constructors and flows
through the inheritance chain via ``super().__init__(..., backend=backend)``.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Abstract template
# ---------------------------------------------------------------------------


class ArrayBackend(ABC):
    """Template defining array operations.

    Concrete subclasses (:class:`NumpyBackend`, :class:`TorchBackend`)
    implement these methods using either NumPy or PyTorch, so that
    downstream logic is independent of the array library.
    """

    name: str = ''

    @abstractmethod
    def asarray(self, data: Any, dtype: str = 'float32') -> Any:
        """Create an array / tensor from *data*."""

    @abstractmethod
    def zeros(self, shape: Union[int, tuple], dtype: str = 'float32') -> Any:
        """Zero-filled array / tensor."""

    @abstractmethod
    def ones(self, shape: Union[int, tuple], dtype: str = 'float32') -> Any:
        """Ones-filled array / tensor."""

    @abstractmethod
    def empty(self, shape: Union[int, tuple], dtype: str = 'float32') -> Any:
        """Uninitialized array / tensor."""

    @abstractmethod
    def stack(self, arrays: Sequence, axis: int = 0) -> Any:
        """Stack arrays along a new axis."""

    @abstractmethod
    def concatenate(self, arrays: Sequence, axis: int = 0) -> Any:
        """Concatenate arrays along an existing axis."""

    @abstractmethod
    def pad(
        self,
        array: Any,
        pad_width: Sequence,
        constant_values: float = 0,
    ) -> Any:
        """Pad with constant values (NumPy-style *pad_width*)."""

    @abstractmethod
    def arange(self, *args: Any) -> Any:
        """Evenly spaced values."""

    @abstractmethod
    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert to NumPy (for interop with NumPy-only code)."""

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------

_NP_DTYPES = {
    'float32': np.float32,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64,
    'bool': np.bool_,
}


class NumpyBackend(ArrayBackend):
    """Array operations implemented with NumPy."""

    name = 'numpy'

    @staticmethod
    def _dtype(s: str):
        return _NP_DTYPES.get(s, np.float32)

    def asarray(self, data, dtype='float32'):
        return np.asarray(data, dtype=self._dtype(dtype))

    def zeros(self, shape, dtype='float32'):
        return np.zeros(shape, dtype=self._dtype(dtype))

    def ones(self, shape, dtype='float32'):
        return np.ones(shape, dtype=self._dtype(dtype))

    def empty(self, shape, dtype='float32'):
        return np.empty(shape, dtype=self._dtype(dtype))

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def pad(self, array, pad_width, constant_values=0):
        return np.pad(array, pad_width, constant_values=constant_values)

    def arange(self, *args):
        return np.arange(*args)

    def to_numpy(self, array):
        return np.asarray(array)


class TorchBackend(ArrayBackend):
    """Array operations implemented with PyTorch.

    Raises :class:`ImportError` at instantiation time if ``torch`` is
    not installed.
    """

    name = 'torch'

    def __init__(self) -> None:
        try:
            import torch as _torch
        except ImportError as exc:
            raise ImportError('TorchBackend requires PyTorch. Install with: pip install torch') from exc
        self._torch = _torch

    def _dtype(self, s: str):
        mapping = {
            'float32': self._torch.float32,
            'float64': self._torch.float64,
            'int32': self._torch.int32,
            'int64': self._torch.int64,
            'bool': self._torch.bool,
        }
        return mapping.get(s, self._torch.float32)

    @staticmethod
    def _ensure_shape(shape):
        return (shape,) if isinstance(shape, int) else tuple(shape)

    def asarray(self, data, dtype='float32'):
        if isinstance(data, self._torch.Tensor):
            return data.to(dtype=self._dtype(dtype))
        return self._torch.tensor(
            np.asarray(data),
            dtype=self._dtype(dtype),
        )

    def zeros(self, shape, dtype='float32'):
        return self._torch.zeros(
            self._ensure_shape(shape),
            dtype=self._dtype(dtype),
        )

    def ones(self, shape, dtype='float32'):
        return self._torch.ones(
            self._ensure_shape(shape),
            dtype=self._dtype(dtype),
        )

    def empty(self, shape, dtype='float32'):
        return self._torch.empty(
            self._ensure_shape(shape),
            dtype=self._dtype(dtype),
        )

    def stack(self, arrays, axis=0):
        tensors = [a if isinstance(a, self._torch.Tensor) else self.asarray(a) for a in arrays]
        return self._torch.stack(tensors, dim=axis)

    def concatenate(self, arrays, axis=0):
        tensors = [a if isinstance(a, self._torch.Tensor) else self.asarray(a) for a in arrays]
        return self._torch.cat(tensors, dim=axis)

    def pad(self, array, pad_width, constant_values=0):
        import torch.nn.functional as F

        if not isinstance(array, self._torch.Tensor):
            array = self.asarray(array)
        # Convert numpy pad_width [(before0, after0), (before1, after1), …]
        # to torch format [lastDimL, lastDimR, …, firstDimL, firstDimR]
        torch_pad: list[int] = []
        for pw in reversed(pad_width):
            torch_pad.extend(pw)
        return F.pad(array, torch_pad, value=constant_values)

    def arange(self, *args):
        return self._torch.arange(*args)

    def to_numpy(self, array):
        if isinstance(array, self._torch.Tensor):
            return array.detach().cpu().numpy()
        return np.asarray(array)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[ArrayBackend]] = {
    'numpy': NumpyBackend,
    'torch': TorchBackend,
}


def get_backend(name: Union[str, ArrayBackend]) -> ArrayBackend:
    """Retrieve a backend by name, or pass through an existing instance.

    Parameters
    ----------
    name : str or ArrayBackend
        ``'numpy'``, ``'torch'``, or an :class:`ArrayBackend` instance.

    Returns
    -------
    ArrayBackend
    """
    if isinstance(name, ArrayBackend):
        return name
    if name not in _BACKENDS:
        raise ValueError(f'Unknown backend {name!r}. Available: {sorted(_BACKENDS)}')
    return _BACKENDS[name]()


def register_backend(name: str, cls: type[ArrayBackend]) -> None:
    """Register a custom backend globally."""
    _BACKENDS[name] = cls


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def with_backend(cls=None, *, default: str = 'numpy'):
    """Class decorator adding transparent array-backend selection.

    Adds a ``backend`` keyword argument to ``__init__`` and exposes the
    active backend as ``self.B``.

    Can be used bare or with arguments::

        @with_backend  # default='numpy'
        class A: ...


        @with_backend(default='torch')  # default='torch'
        class B: ...
    """

    def _wrap(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, backend=default, **kwargs):
            self._backend = get_backend(backend)
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init

        # Property accessors (skip if already defined by a parent)
        if 'B' not in cls.__dict__:
            cls.B = property(
                lambda self: self._backend,
                doc='Active array backend.',
            )
        if 'backend_name' not in cls.__dict__:
            cls.backend_name = property(
                lambda self: self._backend.name,
                doc='Name of the active backend.',
            )

        return cls

    if cls is not None:
        # Bare @with_backend (no arguments)
        return _wrap(cls)
    # Called as @with_backend(default='torch')
    return _wrap
