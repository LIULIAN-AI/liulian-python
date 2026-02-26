"""End-to-end tests using synthetic (dummy) data.

Tests the full Experiment → ForecastTrainer → predict → visualize
pipeline on a small deterministic synthetic dataset so that:
  - shapes of outputs are correct
  - metrics are finite and reasonable
  - visualization files are actually created
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from liulian.data.base import BaseDataset, DataSplit
from liulian.runtime import Experiment, ExperimentSpec
from liulian.runtime.trainer import ForecastTrainer
from liulian.tasks.base import PredictionRegime, PredictionTask


# ---------------------------------------------------------------------------
# Tiny deterministic model (linear projection)
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """A trivial model for testing: just a Linear from seq_len to pred_len."""

    def __init__(self, args: SimpleNamespace):
        super().__init__()
        self._args = args
        c_in = getattr(args, 'enc_in', 1)
        seq_len = getattr(args, 'seq_len', 16)
        pred_len = getattr(args, 'pred_len', 4)
        self.proj = nn.Linear(seq_len, pred_len)
        self.channel_proj = nn.Linear(c_in, c_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: (B, seq_len, C)
        out = self.proj(x_enc.transpose(1, 2)).transpose(1, 2)  # (B, pred_len, C)
        return out


# ---------------------------------------------------------------------------
# Synthetic dataset + data loaders
# ---------------------------------------------------------------------------


class _SyntheticDataset(BaseDataset):
    """Deterministic synthetic dataset for e2e tests."""
    domain = 'test'
    version = '0.0.1'

    def get_split(self, split_name: str) -> DataSplit:
        rng = np.random.default_rng(42)
        n = 32
        X = rng.normal(size=(n, 16, 1)).astype(np.float32)
        y = rng.normal(size=(n, 4, 1)).astype(np.float32)
        return DataSplit(X=X, y=y, name=split_name)


def _make_loaders(
    n_samples: int = 32,
    seq_len: int = 16,
    pred_len: int = 4,
    n_features: int = 1,
    batch_size: int = 8,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create deterministic train/val/test loaders."""
    rng = np.random.default_rng(seed)
    loaders = {}
    for split in ('train', 'val', 'test'):
        X = rng.normal(size=(n_samples, seq_len, n_features)).astype(np.float32)
        y = X[:, -pred_len:, :]  # target = last pred_len steps of input
        xm = np.zeros((n_samples, seq_len, 1), dtype=np.float32)
        ym = np.arange(seq_len, dtype=np.float32)[None, :, None].repeat(n_samples, axis=0)
        ym = ym[:, :pred_len, :]  # time marks for pred_len
        # Expand ym to full seq_len for marks
        ym_full = np.arange(seq_len, dtype=np.float32)[None, :, None].repeat(n_samples, axis=0)

        ds = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(xm),
            torch.from_numpy(ym_full[:, :pred_len, :]),
        )
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'))
    return loaders


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_args():
    return SimpleNamespace(
        enc_in=1,
        seq_len=16,
        pred_len=4,
        label_len=0,
    )


@pytest.fixture
def config():
    return {
        'seq_len': 16,
        'pred_len': 4,
        'label_len': 0,
        'enc_in': 1,
        'train_epochs': 2,
        'learning_rate': 0.01,
        'patience': 5,
        'batch_size': 8,
        'show_progress': False,
        'loss': 'mse',
        'metrics': ['rmse', 'mae', 'nse'],
    }


@pytest.fixture
def loaders():
    return _make_loaders()


# ---------------------------------------------------------------------------
# ForecastTrainer standalone
# ---------------------------------------------------------------------------


class TestForecastTrainerDummy:
    """Test ForecastTrainer directly with synthetic data."""

    def test_fit_produces_metrics(self, config, loaders, model_args, tmp_path):
        config['show_progress'] = False
        trainer = ForecastTrainer(
            config=config,
            checkpoint_dir=str(tmp_path / 'ckpt'),
        )
        model = _TinyModel(model_args)
        result = trainer.fit(model, loaders['train'], loaders['val'], loaders['test'])

        assert 'best_val_mse' in result
        assert result['best_val_mse'] < float('inf')
        assert result['epochs_run'] == 2
        assert len(result['history']) == 2

        # History entries should have configured metrics
        h = result['history'][0]
        assert 'train_loss' in h
        assert 'val_mse' in h
        assert 'val_rmse' in h
        assert 'val_mae' in h
        assert 'val_nse' in h
        assert all(np.isfinite(v) for v in [h['val_mse'], h['val_rmse'], h['val_mae']])

    def test_evaluate_returns_all_metrics(self, config, loaders, model_args, tmp_path):
        trainer = ForecastTrainer(
            config=config,
            checkpoint_dir=str(tmp_path / 'ckpt'),
        )
        model = _TinyModel(model_args)
        metrics = trainer.evaluate(model, loaders['test'])
        # Default metrics: mse, rmse, mae, nse
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'nse' in metrics
        assert all(np.isfinite(v) for k, v in metrics.items() if k != 'nse')

    def test_predict_shapes(self, config, loaders, model_args, tmp_path):
        trainer = ForecastTrainer(
            config=config,
            checkpoint_dir=str(tmp_path / 'ckpt'),
        )
        model = _TinyModel(model_args)
        pred_result = trainer.predict(model, loaders['test'])
        assert 'preds' in pred_result
        assert 'trues' in pred_result
        assert 'times' in pred_result
        assert pred_result['preds'].shape[1] == 4  # pred_len
        assert pred_result['trues'].shape[1] == 4
        assert pred_result['preds'].shape == pred_result['trues'].shape

    def test_configurable_loss(self, config, loaders, model_args, tmp_path):
        """Different loss names should work."""
        for loss_name in ('mse', 'mae', 'rmse'):
            cfg = {**config, 'loss': loss_name}
            trainer = ForecastTrainer(
                config=cfg,
                checkpoint_dir=str(tmp_path / f'ckpt_{loss_name}'),
            )
            model = _TinyModel(model_args)
            result = trainer.fit(model, loaders['train'], loaders['val'])
            assert result['epochs_run'] >= 1

    def test_rmse_metric_values(self, config, loaders, model_args, tmp_path):
        """RMSE should be approximately sqrt(MSE).

        Because evaluate() averages per-batch RMSE and per-batch MSE
        independently, sqrt(mean(MSE)) != mean(RMSE) in general.
        We use a loose tolerance.
        """
        trainer = ForecastTrainer(
            config=config,
            checkpoint_dir=str(tmp_path / 'ckpt'),
        )
        model = _TinyModel(model_args)
        metrics = trainer.evaluate(model, loaders['test'])
        # RMSE and sqrt(MSE) should be close but not identical due to
        # Jensen's inequality (mean of sqrt != sqrt of mean).
        np.testing.assert_allclose(
            metrics['rmse'],
            np.sqrt(metrics['mse']),
            rtol=5e-2,  # 5% tolerance for batch-averaging artifact
        )


# ---------------------------------------------------------------------------
# Full Experiment pipeline
# ---------------------------------------------------------------------------


class TestExperimentE2E:
    """Test the full Experiment lifecycle with dummy data."""

    def test_run_train_eval(self, config, loaders, model_args, tmp_path):
        spec = ExperimentSpec(
            name='test_e2e',
            task={'type': 'PredictionTask'},
            dataset={'type': 'Synthetic'},
            model={'type': 'TinyModel'},
        )
        task = PredictionTask(
            regime=PredictionRegime(
                horizon=4,
                context_length=16,
            )
        )
        dataset = _SyntheticDataset()
        model = _TinyModel(model_args)

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=dataset,
            model=None,
            torch_model=model,
            data_loaders=loaders,
            config=config,
        )
        summary = exp.run(train=True, eval=True)

        assert summary['status'] == 'ok'
        assert 'training' in summary['metrics']
        assert summary['metrics']['training']['epochs_run'] == 2
        assert 'predictions' in summary
        assert summary['predictions']['preds'].shape[1] == 4

    def test_predict_and_visualize(self, config, loaders, model_args, tmp_path):
        """Full pipeline: train → predict → visualize → files created."""
        spec = ExperimentSpec(
            name='test_viz_e2e',
            task={'type': 'PredictionTask'},
            dataset={'type': 'Synthetic'},
            model={'type': 'TinyModel'},
        )
        task = PredictionTask(
            regime=PredictionRegime(horizon=4, context_length=16)
        )
        dataset = _SyntheticDataset()
        model = _TinyModel(model_args)

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=dataset,
            model=None,
            torch_model=model,
            data_loaders=loaders,
            config=config,
        )
        summary = exp.run(train=True, eval=True)

        # Generate viz from predictions
        viz_dir = str(tmp_path / 'viz')
        paths = exp.visualize(summary, output_dir=viz_dir, method='mean')
        assert len(paths) > 0
        for p in paths.values():
            assert os.path.isfile(p)

    def test_metrics_are_finite(self, config, loaders, model_args, tmp_path):
        """Final test metrics should all be finite numbers."""
        spec = ExperimentSpec(
            name='test_finite',
            task={'type': 'PredictionTask'},
            dataset={'type': 'Synthetic'},
            model={'type': 'TinyModel'},
        )
        task = PredictionTask(
            regime=PredictionRegime(horizon=4, context_length=16)
        )
        dataset = _SyntheticDataset()
        model = _TinyModel(model_args)

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=dataset,
            model=None,
            torch_model=model,
            data_loaders=loaders,
            config=config,
        )
        summary = exp.run(train=True, eval=True)
        final = summary['metrics'].get('final_test', {})
        for k, v in final.items():
            if k != 'nse':  # NSE can be NaN for constant targets
                assert np.isfinite(v), f'{k}={v} is not finite'
