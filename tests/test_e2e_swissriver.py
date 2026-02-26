"""End-to-end tests using the real Swiss River dataset.

These tests are marked with ``@pytest.mark.main_branch`` so they only
run on the *main* branch (e.g. via ``pytest -m main_branch``).

Requires the Swiss River CSV data under ``dataset/swiss_river/``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

# Mark the entire module so it's easy to skip in CI feature branches
pytestmark = pytest.mark.main_branch

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), '..', 'dataset', 'swiss_river'
)
SKIP_REASON = 'Swiss River CSV data not found under dataset/swiss_river'


def _data_available() -> bool:
    return os.path.isdir(DATASET_ROOT)


@pytest.fixture(scope='module')
def swiss_dataset():
    """Build a small SwissRiverDataset for testing."""
    pytest.importorskip('torch')
    if not _data_available():
        pytest.skip(SKIP_REASON)
    from liulian.data.swiss_river import SwissRiverDataset

    return SwissRiverDataset(
        data_name='swiss-river-2010',
        root_path=DATASET_ROOT,
        split_mode='ts',
        seq_len=16,
        pred_len=4,
        task='forecast',
        use_full_history=False,
        identifier_mode='none',
    )


# ---------------------------------------------------------------------------
# Dataset loading tests
# ---------------------------------------------------------------------------


class TestSwissRiverDatasetLoading:
    @pytest.mark.skipif(not _data_available(), reason=SKIP_REASON)
    def test_train_split_has_samples(self, swiss_dataset):
        ts_split = swiss_dataset.get_split('train')
        assert len(ts_split) > 0, 'Train split should have samples'

    @pytest.mark.skipif(not _data_available(), reason=SKIP_REASON)
    def test_val_split_has_samples(self, swiss_dataset):
        ts_split = swiss_dataset.get_split('val')
        assert len(ts_split) > 0, 'Val split should have samples'

    @pytest.mark.skipif(not _data_available(), reason=SKIP_REASON)
    def test_sample_shape(self, swiss_dataset):
        ts_split = swiss_dataset.get_split('train')
        sample = ts_split[0]
        # Should be (x, y, x_mark, y_mark[, entity_id]) tuple
        assert len(sample) in (4, 5), f'Expected 4- or 5-tuple, got {len(sample)}'
        x, y, xm, ym = sample[:4]
        assert x.shape[0] == 16, f'seq_len mismatch: {x.shape}'
        assert y.shape[0] == 4, f'pred_len mismatch: {y.shape}'

    @pytest.mark.skipif(not _data_available(), reason=SKIP_REASON)
    def test_values_are_finite(self, swiss_dataset):
        import torch
        ts_split = swiss_dataset.get_split('train')
        x, y, xm, ym = ts_split[0][:4]
        assert torch.all(torch.isfinite(x)), 'x contains non-finite values'
        assert torch.all(torch.isfinite(y)), 'y contains non-finite values'


# ---------------------------------------------------------------------------
# E2E training test (short, 1 epoch)
# ---------------------------------------------------------------------------


class TestSwissRiverE2ETraining:
    @pytest.mark.skipif(not _data_available(), reason=SKIP_REASON)
    def test_one_epoch_train(self, swiss_dataset, tmp_path):
        torch = pytest.importorskip('torch')
        import torch.nn as nn
        from torch.utils.data import DataLoader

        from liulian.runtime.trainer import ForecastTrainer

        train_split = swiss_dataset.get_split('train')
        val_split = swiss_dataset.get_split('val')

        train_loader = DataLoader(train_split, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_split, batch_size=8, shuffle=False)

        # Determine input features from a sample
        x0, y0 = train_split[0][0], train_split[0][1]
        c_in = x0.shape[-1]

        # Simple linear model
        class _LinModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(16, 4)

            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                return self.proj(x_enc.transpose(1, 2)).transpose(1, 2)

        config = {
            'seq_len': 16,
            'pred_len': 4,
            'label_len': 0,
            'enc_in': c_in,
            'train_epochs': 1,
            'learning_rate': 0.01,
            'patience': 3,
            'max_train_iters': 5,
            'max_eval_iters': 5,
            'show_progress': False,
            'loss': 'mse',
            'nan_mask_loss': True,
            'metrics': ['rmse', 'mae', 'nse'],
        }
        trainer = ForecastTrainer(
            config=config,
            checkpoint_dir=str(tmp_path / 'ckpt'),
        )
        model = _LinModel()
        result = trainer.fit(model, train_loader, val_loader)

        assert result['epochs_run'] == 1
        h = result['history'][0]
        assert np.isfinite(h['train_loss']), f'train_loss={h['train_loss']}'
        assert np.isfinite(h['val_mse']), f'val_mse={h['val_mse']}'
        assert np.isfinite(h['val_rmse'])
        assert np.isfinite(h['val_mae'])

    @pytest.mark.skipif(not _data_available(), reason=SKIP_REASON)
    def test_predict_and_aggregate(self, swiss_dataset, tmp_path):
        torch = pytest.importorskip('torch')
        import torch.nn as nn
        from torch.utils.data import DataLoader

        from liulian.runtime.trainer import ForecastTrainer
        from liulian.viz.prediction_aggregator import aggregate_predictions

        test_split = swiss_dataset.get_split('test')
        test_loader = DataLoader(test_split, batch_size=8, shuffle=False)

        x0, y0 = test_split[0][0], test_split[0][1]
        c_in = x0.shape[-1]

        class _LinModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(16, 4)

            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                return self.proj(x_enc.transpose(1, 2)).transpose(1, 2)

        config = {
            'seq_len': 16,
            'pred_len': 4,
            'label_len': 0,
            'enc_in': c_in,
            'show_progress': False,
            'max_eval_iters': 10,
        }
        trainer = ForecastTrainer(
            config=config,
            checkpoint_dir=str(tmp_path / 'ckpt'),
        )
        model = _LinModel()
        pred_result = trainer.predict(model, test_loader, max_iters=10)

        assert pred_result['preds'].shape[1] == 4
        assert pred_result['trues'].shape[1] == 4

        # Aggregate with mean
        result = aggregate_predictions(
            pred_result['preds'],
            pred_result['trues'],
            pred_result['times'],
            method='mean',
        )
        assert len(result['time']) > 0
        assert result['pred'].shape[0] == len(result['time'])
