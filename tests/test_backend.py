"""Tests for the array backend system and its integration with datasets."""

from __future__ import annotations

import numpy as np
import pytest

from liulian.data.backend import (
    ArrayBackend,
    NumpyBackend,
    TorchBackend,
    get_backend,
    register_backend,
    with_backend,
)
from liulian.data.base import BaseDataset, DataSplit
from liulian.data.spec import TopologySpec


# ---------------------------------------------------------------------------
# Backend unit tests
# ---------------------------------------------------------------------------


class TestNumpyBackend:
    """NumpyBackend operations."""

    def setup_method(self):
        self.B = NumpyBackend()

    def test_name(self):
        assert self.B.name == 'numpy'

    def test_asarray(self):
        arr = self.B.asarray([[1, 2], [3, 4]], dtype='float32')
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (2, 2)

    def test_zeros(self):
        z = self.B.zeros((3, 4), dtype='float32')
        assert z.shape == (3, 4)
        assert (z == 0).all()

    def test_ones(self):
        o = self.B.ones((2,), dtype='int64')
        assert o.dtype == np.int64
        assert (o == 1).all()

    def test_empty(self):
        e = self.B.empty((5, 3))
        assert e.shape == (5, 3)

    def test_stack(self):
        a, b = np.ones((2, 3)), np.zeros((2, 3))
        s = self.B.stack([a, b], axis=0)
        assert s.shape == (2, 2, 3)

    def test_concatenate(self):
        a, b = np.ones((2, 3)), np.zeros((1, 3))
        c = self.B.concatenate([a, b], axis=0)
        assert c.shape == (3, 3)

    def test_pad(self):
        a = np.ones((2, 3))
        p = self.B.pad(a, [(0, 1), (0, 2)], constant_values=0)
        assert p.shape == (3, 5)

    def test_arange(self):
        r = self.B.arange(5)
        np.testing.assert_array_equal(r, np.arange(5))

    def test_to_numpy(self):
        arr = np.array([1, 2, 3])
        out = self.B.to_numpy(arr)
        assert isinstance(out, np.ndarray)


class TestTorchBackend:
    """TorchBackend operations."""

    def setup_method(self):
        torch = pytest.importorskip('torch')
        self.torch = torch
        self.B = TorchBackend()

    def test_name(self):
        assert self.B.name == 'torch'

    def test_asarray(self):
        t = self.B.asarray([[1, 2], [3, 4]], dtype='float32')
        assert isinstance(t, self.torch.Tensor)
        assert t.dtype == self.torch.float32
        assert t.shape == (2, 2)

    def test_asarray_from_tensor(self):
        src = self.torch.ones(3, dtype=self.torch.float64)
        t = self.B.asarray(src, dtype='float32')
        assert t.dtype == self.torch.float32

    def test_zeros(self):
        z = self.B.zeros((3, 4), dtype='float32')
        assert z.shape == (3, 4)
        assert (z == 0).all()

    def test_zeros_int(self):
        z = self.B.zeros(5, dtype='float32')
        assert z.shape == (5,)

    def test_ones(self):
        o = self.B.ones((2,), dtype='int64')
        assert o.dtype == self.torch.int64

    def test_empty(self):
        e = self.B.empty((5, 3))
        assert e.shape == (5, 3)

    def test_empty_zero_dim(self):
        e = self.B.empty((2, 0), dtype='int64')
        assert e.shape == (2, 0)

    def test_stack(self):
        a = self.torch.ones(2, 3)
        b = self.torch.zeros(2, 3)
        s = self.B.stack([a, b], axis=0)
        assert s.shape == (2, 2, 3)

    def test_concatenate(self):
        a = self.torch.ones(2, 3)
        b = self.torch.zeros(1, 3)
        c = self.B.concatenate([a, b], axis=0)
        assert c.shape == (3, 3)

    def test_pad(self):
        a = self.torch.ones(2, 3)
        p = self.B.pad(a, [(0, 1), (0, 2)], constant_values=0)
        assert p.shape == (3, 5)

    def test_arange(self):
        r = self.B.arange(5)
        assert isinstance(r, self.torch.Tensor)
        assert len(r) == 5

    def test_to_numpy(self):
        t = self.torch.tensor([1.0, 2.0])
        arr = self.B.to_numpy(t)
        assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_get_numpy(self):
        b = get_backend('numpy')
        assert isinstance(b, NumpyBackend)

    def test_get_torch(self):
        pytest.importorskip('torch')
        b = get_backend('torch')
        assert isinstance(b, TorchBackend)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown backend'):
            get_backend('jax')

    def test_passthrough_instance(self):
        b = NumpyBackend()
        assert get_backend(b) is b

    def test_register_custom(self):
        class DummyBackend(ArrayBackend):
            name = 'dummy'

            def asarray(self, data, dtype='float32'):
                return np.asarray(data)

            def zeros(self, shape, dtype='float32'):
                return np.zeros(shape)

            def ones(self, shape, dtype='float32'):
                return np.ones(shape)

            def empty(self, shape, dtype='float32'):
                return np.empty(shape)

            def stack(self, arrays, axis=0):
                return np.stack(arrays, axis=axis)

            def concatenate(self, arrays, axis=0):
                return np.concatenate(arrays, axis=axis)

            def pad(self, array, pad_width, constant_values=0):
                return np.pad(array, pad_width)

            def arange(self, *args):
                return np.arange(*args)

            def to_numpy(self, array):
                return np.asarray(array)

        register_backend('dummy', DummyBackend)
        b = get_backend('dummy')
        assert isinstance(b, DummyBackend)
        assert b.name == 'dummy'


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------


class TestWithBackendDecorator:
    """Test the @with_backend class decorator."""

    def test_bare_decorator_default_numpy(self):
        @with_backend
        class Foo:
            def __init__(self):
                pass

        obj = Foo()
        assert isinstance(obj.B, NumpyBackend)
        assert obj.backend_name == 'numpy'

    def test_decorator_with_default_torch(self):
        pytest.importorskip('torch')

        @with_backend(default='torch')
        class Bar:
            def __init__(self):
                pass

        obj = Bar()
        assert isinstance(obj.B, TorchBackend)
        assert obj.backend_name == 'torch'

    def test_override_at_instantiation(self):
        pytest.importorskip('torch')

        @with_backend
        class Baz:
            def __init__(self, x):
                self.x = x

        obj = Baz(42, backend='torch')
        assert obj.backend_name == 'torch'
        assert obj.x == 42

    def test_original_args_preserved(self):
        @with_backend
        class Qux:
            def __init__(self, a, b, c=10):
                self.a = a
                self.b = b
                self.c = c

        obj = Qux(1, 2, c=99)
        assert obj.a == 1
        assert obj.b == 2
        assert obj.c == 99
        assert obj.backend_name == 'numpy'

    def test_inheritance_through_kwargs(self):
        """Subclasses can pass backend through **kwargs."""
        pytest.importorskip('torch')

        @with_backend
        class Parent:
            def __init__(self, val):
                self.val = val

        class Child(Parent):
            def __init__(self, val, extra, **kwargs):
                super().__init__(val, **kwargs)
                self.extra = extra

        obj = Child(10, 'hi', backend='torch')
        assert obj.val == 10
        assert obj.extra == 'hi'
        assert obj.backend_name == 'torch'


# ---------------------------------------------------------------------------
# BaseDataset backend integration
# ---------------------------------------------------------------------------


class TestBaseDatasetBackend:
    """Backend integration in BaseDataset hierarchy."""

    def test_default_backend_is_numpy(self):
        from liulian.data.ts.timeseriesdataset import TimeSeriesDataset

        import pandas as pd

        df = pd.DataFrame({'epoch_day': range(100), 'x': range(100), 'y': range(100)})
        ds = TimeSeriesDataset(
            splits={'train': df},
            time_col='epoch_day',
            feature_cols=['x'],
            target_cols=['y'],
            seq_len=10,
        )
        assert ds.backend_name == 'numpy'
        split = ds.get_split('train')
        # TimeSeriesSplit is always torch-first
        import torch

        assert isinstance(split.X, torch.Tensor)

    def test_torch_backend(self):
        torch = pytest.importorskip('torch')
        from liulian.data.ts.timeseriesdataset import TimeSeriesDataset

        import pandas as pd

        df = pd.DataFrame({'epoch_day': range(100), 'x': range(100), 'y': range(100)})
        ds = TimeSeriesDataset(
            splits={'train': df},
            time_col='epoch_day',
            feature_cols=['x'],
            target_cols=['y'],
            seq_len=10,
            backend='torch',
        )
        assert ds.backend_name == 'torch'
        split = ds.get_split('train')
        assert isinstance(split.X, torch.Tensor)
        assert isinstance(split.y, torch.Tensor)

    def test_finalize_split_identity_for_numpy(self):
        s = DataSplit(X=np.zeros((2, 3, 1)), y=np.zeros((2, 3, 1)), name='test')
        from liulian.data.ts.timeseriesdataset import TimeSeriesDataset
        import pandas as pd

        ds = TimeSeriesDataset(
            splits={'x': pd.DataFrame({'epoch_day': [1]})},
            backend='numpy',
        )
        out = ds._finalize_split(s)
        assert out is s  # identity when backend is numpy


# ---------------------------------------------------------------------------
# SpatialTempoDataset backend integration
# ---------------------------------------------------------------------------


class TestSpatialTempoGraphBackend:
    """Graph arrays respect the active backend."""

    def _make_ds(self, backend='numpy'):
        import pandas as pd
        from liulian.data.st.spatialtempodataset import SpatialTempoDataset

        topo = TopologySpec(
            node_ids=['A', 'B', 'C'],
            edges=[('A', 'B'), ('B', 'C')],
            coordinates={'A': (1.0, 2.0), 'B': (3.0, 4.0), 'C': (5.0, 6.0)},
        )
        ds = SpatialTempoDataset(
            splits={
                'train': pd.DataFrame(
                    {'epoch_day': range(50), 'x': range(50), 'y': range(50)}
                )
            },
            time_col='epoch_day',
            feature_cols=['x'],
            target_cols=['y'],
            seq_len=5,
            topology=topo,
            graph_mode='edge_index',
            station_ids=['A', 'B', 'C'],
            backend=backend,
        )
        return ds

    def test_numpy_graph_arrays(self):
        ds = self._make_ds('numpy')
        assert isinstance(ds.edge_index, np.ndarray)
        assert isinstance(ds.adj_matrix, np.ndarray)
        assert isinstance(ds.node_coordinates, np.ndarray)
        assert ds.edge_index.shape == (2, 2)
        assert ds.adj_matrix.shape == (3, 3)
        assert ds.node_coordinates.shape == (3, 2)

    def test_torch_graph_arrays(self):
        torch = pytest.importorskip('torch')
        ds = self._make_ds('torch')
        assert isinstance(ds.edge_index, torch.Tensor)
        assert isinstance(ds.adj_matrix, torch.Tensor)
        assert isinstance(ds.node_coordinates, torch.Tensor)
        assert ds.edge_index.shape == (2, 2)
        assert ds.adj_matrix.shape == (3, 3)

    def test_no_topology_returns_none(self):
        import pandas as pd
        from liulian.data.st.spatialtempodataset import SpatialTempoDataset

        ds = SpatialTempoDataset(
            splits={
                'train': pd.DataFrame(
                    {'epoch_day': range(50), 'x': range(50), 'y': range(50)}
                )
            },
            time_col='epoch_day',
            feature_cols=['x'],
            target_cols=['y'],
            seq_len=5,
            graph_mode='none',
            backend='numpy',
        )
        assert ds.edge_index is None
        assert ds.adj_matrix is None


# ---------------------------------------------------------------------------
# SwissRiverDataset — graph optional
# ---------------------------------------------------------------------------

_SWISS_ROOT = 'dataset/swiss_river'


def _swiss_data_available() -> bool:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / _SWISS_ROOT
    return (root / 'swiss-1990_train.csv').exists()


@pytest.mark.skipif(
    not _swiss_data_available(), reason='Swiss River data not available'
)
class TestSwissRiverGraphOptional:
    """SwissRiverDataset works without graph data."""

    def test_no_graph_mode(self):
        from liulian.data.swiss_river import SwissRiverDataset

        ds = SwissRiverDataset(
            data_name='swiss-river-1990',
            graph_mode='none',
            seq_len=10,
            max_samples=5,
        )
        assert ds.topology is None
        assert ds.edge_index is None
        assert ds.adj_matrix is None
        split = ds.get_split('train')
        assert len(split) > 0

    def test_with_graph_mode(self):
        from liulian.data.swiss_river import SwissRiverDataset

        ds = SwissRiverDataset(
            data_name='swiss-river-1990',
            graph_mode='edge_index',
            seq_len=10,
            max_samples=5,
        )
        assert ds.topology is not None
        assert ds.edge_index is not None

    def test_torch_backend(self):
        torch = pytest.importorskip('torch')
        from liulian.data.swiss_river import SwissRiverDataset

        ds = SwissRiverDataset(
            data_name='swiss-river-1990',
            graph_mode='edge_index',
            seq_len=10,
            max_samples=5,
            backend='torch',
        )
        assert isinstance(ds.edge_index, torch.Tensor)
        split = ds.get_split('train')
        assert isinstance(split.X, torch.Tensor)

    def test_numpy_backend(self):
        from liulian.data.swiss_river import SwissRiverDataset

        ds = SwissRiverDataset(
            data_name='swiss-river-1990',
            graph_mode='edge_index',
            seq_len=10,
            max_samples=5,
            backend='numpy',
        )
        assert isinstance(ds.edge_index, np.ndarray)
        split = ds.get_split('train')
        # TimeSeriesSplit is always torch-first (graph arrays follow backend)
        import torch

        assert isinstance(split.X, torch.Tensor)


# ---------------------------------------------------------------------------
# Standalone torch datasets — @with_backend
# ---------------------------------------------------------------------------


class TestStandaloneTorchDatasets:
    """Verify that @with_backend on torch datasets doesn't break them."""

    def test_dataset_custom_has_backend(self):
        pytest.importorskip('torch')
        pytest.importorskip('sklearn')
        from liulian.data.dataset_custom import DatasetCustom

        # Check class has B property
        assert hasattr(DatasetCustom, 'B')

    def test_sequence_dataset_has_backend(self):
        pytest.importorskip('torch')
        from liulian.data.seq_dataset import SequenceDataset

        assert hasattr(SequenceDataset, 'B')

    def test_sequence_full_inherits_backend(self):
        """SequenceFullDataset inherits backend from SequenceDataset."""
        torch = pytest.importorskip('torch')
        import pandas as pd
        from liulian.data.seq_dataset import SequenceFullDataset

        df = pd.DataFrame({'epoch_day': range(50), 'x': range(50), 'y': range(50)})
        ds = SequenceFullDataset(
            df,
            time_col='epoch_day',
            feature_cols=['x'],
            target_cols=['y'],
            backend='torch',
        )
        assert ds.backend_name == 'torch'

    def test_sequence_windowed_inherits_backend(self):
        torch = pytest.importorskip('torch')
        import pandas as pd
        from liulian.data.seq_dataset import SequenceWindowedDataset

        df = pd.DataFrame({'epoch_day': range(50), 'x': range(50), 'y': range(50)})
        ds = SequenceWindowedDataset(
            window_len=10,
            df=df,
            time_col='epoch_day',
            feature_cols=['x'],
            target_cols=['y'],
            backend='numpy',
        )
        assert ds.backend_name == 'numpy'
