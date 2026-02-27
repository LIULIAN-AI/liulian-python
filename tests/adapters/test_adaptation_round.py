"""Tests for new features added in the 10-item adaptation round.

Covers:
- Ray HPO integration (make_trainable, RayOptimizer ASHA config)
- Entity identifier modes (descriptors, StationEmbedding)
- Channel-independent data arrangement
- wandb logging auto-creation
- Aggregator new methods (best, worst, single)
- Viz integration (prediction range plots, Experiment.visualize)
- CLI enhancements (train, predict, viz, hparam subcommands)
- Experiment HPO branch
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure liulian is importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_loaders():
    """Create tiny DataLoaders for testing (4-tuple: x, y, x_mark, y_mark)."""
    B, seq_len, pred_len, C = 8, 16, 4, 2
    x = torch.randn(B, seq_len, C)
    y = torch.randn(B, pred_len + seq_len, 1)
    xm = torch.arange(seq_len).float().unsqueeze(0).expand(B, -1).unsqueeze(-1)
    ym = (
        torch.arange(pred_len + seq_len)
        .float()
        .unsqueeze(0)
        .expand(B, -1)
        .unsqueeze(-1)
    )
    ds = TensorDataset(x, y, xm, ym)
    loader = DataLoader(ds, batch_size=4)
    return {
        'train': loader,
        'val': loader,
        'test': loader,
    }


@pytest.fixture
def dummy_config():
    return {
        'seq_len': 16,
        'pred_len': 4,
        'label_len': 0,
        'features': 'M',
        'train_epochs': 2,
        'learning_rate': 0.01,
        'patience': 2,
        'loss': 'mse',
        'metrics': ['rmse', 'mae'],
        'show_progress': False,
        'enc_in': 2,
        'd_model': 16,
        'd_ff': 16,
        'n_heads': 2,
        'e_layers': 1,
        'd_layers': 1,
        'dropout': 0.1,
        'c_out': 1,
    }


class _TinyModel(nn.Module):
    """Minimal model for testing: linear projection."""

    def __init__(self, enc_in=2, pred_len=4, c_out=1, **_kw):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(enc_in, c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        out = self.proj(x_enc[:, : self.pred_len, :])
        return out


# ===========================================================================
# 1. Ray HPO tests
# ===========================================================================


class TestRayOptimizer:
    def test_asha_config(self):
        from liulian.optim.ray_optimizer import RayOptimizer

        opt = RayOptimizer(
            config={
                'scheduler': 'asha',
                'grace_period': 2,
                'reduction_factor': 4,
            }
        )
        assert opt.config['scheduler'] == 'asha'
        assert opt.config['grace_period'] == 2
        assert opt.config['reduction_factor'] == 4

    def test_fallback_grid_sweep(self):
        from liulian.optim.ray_optimizer import RayOptimizer

        opt = RayOptimizer(config={'num_samples': 2})
        result = opt.run(
            spec=None,
            search_space={'lr': [0.001, 0.01], 'dropout': [0.1, 0.2]},
        )
        # With Ray installed: grid_search × num_samples gives more trials;
        # without Ray: fallback caps at num_samples.
        assert result.n_trials >= 2
        assert 'lr' in result.best_config

    def test_make_trainable_factory(self):
        from liulian.optim.ray_optimizer import make_trainable

        loaders = {
            'train': DataLoader(
                TensorDataset(
                    torch.randn(4, 16, 2),
                    torch.randn(4, 20, 1),
                    torch.zeros(4, 16, 1),
                    torch.zeros(4, 20, 1),
                ),
                batch_size=2,
            ),
            'val': DataLoader(
                TensorDataset(
                    torch.randn(4, 16, 2),
                    torch.randn(4, 20, 1),
                    torch.zeros(4, 16, 1),
                    torch.zeros(4, 20, 1),
                ),
                batch_size=2,
            ),
        }
        base_config = {
            'seq_len': 16,
            'pred_len': 4,
            'label_len': 0,
            'features': 'M',
            'train_epochs': 1,
            'learning_rate': 0.01,
            'patience': 1,
            'loss': 'mse',
            'show_progress': False,
        }
        args = SimpleNamespace(enc_in=2, pred_len=4, c_out=1)
        trainable = make_trainable(_TinyModel, args, loaders, base_config)
        assert callable(trainable)


# ===========================================================================
# 2. Entity identifiers
# ===========================================================================


class TestEntityIdentifiers:
    def test_descriptors_mode(self):
        from liulian.data.ts.timeseriesdataset import make_entity_features

        descriptors = {
            'station_a': [0.1, 0.2, 0.3, 0.4],
            'station_b': [0.5, 0.6, 0.7, 0.8],
        }
        result = make_entity_features(
            'station_a',
            ['station_a', 'station_b'],
            mode='descriptors',
            seq_len=10,
            descriptors=descriptors,
        )
        assert result is not None
        assert result.shape == (10, 4)
        assert torch.allclose(result[0], torch.tensor([0.1, 0.2, 0.3, 0.4]))

    def test_descriptors_fallback(self):
        from liulian.data.ts.timeseriesdataset import make_entity_features

        descriptors = {'station_a': [1.0, 2.0]}
        result = make_entity_features(
            'unknown',
            ['station_a'],
            mode='descriptors',
            seq_len=5,
            descriptors=descriptors,
        )
        assert result is not None
        assert result.shape == (5, 2)
        assert (result == 0).all()

    def test_station_embedding(self):
        from liulian.data.ts.timeseriesdataset import StationEmbedding

        emb = StationEmbedding(num_stations=10, embed_dim=4)
        x = torch.randn(3, 16, 5)
        ids = torch.tensor([0, 3, 7])
        out = emb(x, ids)
        assert out.shape == (3, 16, 9)  # 5 + 4

    def test_all_identifier_modes(self):
        from liulian.data.ts.timeseriesdataset import make_entity_features

        ids = ['s1', 's2', 's3']
        for mode in ['embedding_idx', 'onehot', 'numeric_id', 'sinusoidal']:
            result = make_entity_features('s1', ids, mode=mode, seq_len=8)
            assert result is not None
            assert result.shape[0] == 8

    def test_none_mode(self):
        from liulian.data.ts.timeseriesdataset import make_entity_features

        result = make_entity_features('s1', ['s1'], mode='none', seq_len=8)
        assert result is None


# ===========================================================================
# 3. Channel-independent data arrangement
# ===========================================================================


class TestChannelIndependent:
    def test_wrapping_increases_length(self):
        from liulian.data.ts.channel_independent import ChannelIndependentDataset

        base_ds = TensorDataset(
            torch.randn(10, 16, 3),  # feat
            torch.randn(10, 4, 1),  # target
            torch.arange(16).float().unsqueeze(0).expand(10, -1),  # time
        )
        ci = ChannelIndependentDataset(base_ds, n_channels=3)
        assert len(ci) == 30  # 10 * 3

    def test_single_channel_output(self):
        from liulian.data.ts.channel_independent import ChannelIndependentDataset

        base_ds = TensorDataset(
            torch.randn(5, 16, 4),
            torch.randn(5, 4, 2),
            torch.arange(16).float().unsqueeze(0).expand(5, -1),
        )
        ci = ChannelIndependentDataset(base_ds, n_channels=4)
        feat, target, time = ci[0]
        assert feat.shape == (16, 1)
        # Target unchanged (not match_target_channel)
        assert target.shape == (4, 2)

    def test_match_target_channel(self):
        from liulian.data.ts.channel_independent import ChannelIndependentDataset

        n_ch = 3
        base_ds = TensorDataset(
            torch.randn(5, 16, n_ch),
            torch.randn(5, 4, n_ch),
            torch.arange(16).float().unsqueeze(0).expand(5, -1),
        )
        ci = ChannelIndependentDataset(
            base_ds,
            n_channels=n_ch,
            match_target_channel=True,
        )
        feat, target, time = ci[1]  # channel 1
        assert feat.shape == (16, 1)
        assert target.shape == (4, 1)

    def test_auto_detect_channels(self):
        from liulian.data.ts.channel_independent import ChannelIndependentDataset

        base_ds = TensorDataset(
            torch.randn(3, 8, 5),
            torch.randn(3, 4, 1),
            torch.arange(8).float().unsqueeze(0).expand(3, -1),
        )
        ci = ChannelIndependentDataset(base_ds)
        assert ci.n_channels == 5
        assert len(ci) == 15


# ===========================================================================
# 4. wandb logging auto-creation
# ===========================================================================


class TestWandbLogging:
    def test_auto_create_local_logger(self, dummy_config, dummy_loaders):
        """When no wandb_project is set, Experiment creates LocalFileLogger."""
        from liulian.runtime import Experiment, ExperimentSpec
        from liulian.tasks.base import PredictionRegime, PredictionTask
        from liulian.loggers.local_logger import LocalFileLogger

        task = PredictionTask(
            regime=PredictionRegime(horizon=4, context_length=16),
        )
        spec = ExperimentSpec(name='test_log', task={}, dataset={}, model={})
        model = _TinyModel()

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=MagicMock(),
            model=None,
            torch_model=model,
            data_loaders=dummy_loaders,
            config=dummy_config,
        )
        summary = exp.run(train=True)
        assert isinstance(exp.exp_logger, LocalFileLogger)
        assert summary['status'] == 'ok'
        assert 'metrics' in summary
        assert 'training' in summary['metrics']

    def test_dev_run_disables_wandb(self, dummy_config):
        """dev_run=True should use LocalFileLogger even with wandb_project."""
        from liulian.runtime import Experiment, ExperimentSpec
        from liulian.tasks.base import PredictionRegime, PredictionTask
        from liulian.loggers.local_logger import LocalFileLogger

        cfg = {**dummy_config, 'wandb_project': 'test', 'dev_run': True}
        task = PredictionTask(
            regime=PredictionRegime(horizon=4, context_length=16),
        )
        spec = ExperimentSpec(name='test_dev', task={}, dataset={}, model={})
        model = _TinyModel()

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=MagicMock(),
            model=None,
            torch_model=model,
            data_loaders={
                'train': DataLoader(
                    TensorDataset(
                        torch.randn(4, 16, 2),
                        torch.randn(4, 20, 1),
                        torch.zeros(4, 16, 1),
                        torch.zeros(4, 20, 1),
                    ),
                    batch_size=2,
                ),
                'val': DataLoader(
                    TensorDataset(
                        torch.randn(4, 16, 2),
                        torch.randn(4, 20, 1),
                        torch.zeros(4, 16, 1),
                        torch.zeros(4, 20, 1),
                    ),
                    batch_size=2,
                ),
            },
            config=cfg,
        )
        summary = exp.run(train=True)
        assert isinstance(exp.exp_logger, LocalFileLogger)
        assert summary['status'] == 'ok'
        assert 'metrics' in summary


# ===========================================================================


class TestAggregatorEnhancements:
    @pytest.fixture
    def sample_data(self):
        """Create sample sliding-window prediction data."""
        N, pred_len, C = 10, 4, 1
        preds = np.random.randn(N, pred_len, C).astype(np.float32)
        trues = np.random.randn(N, pred_len, C).astype(np.float32)
        # times: each window starts at i, so times = [i, i+1, ..., i+pred_len-1+seq_len-1]
        win_len = 16 + pred_len  # seq_len + pred_len
        times = np.array([np.arange(i, i + win_len) for i in range(N)])
        return preds, trues, times

    def test_best_method(self, sample_data):
        from liulian.viz.prediction_aggregator import aggregate_predictions

        preds, trues, times = sample_data
        result = aggregate_predictions(preds, trues, times, method='best')
        assert 'time' in result
        assert 'pred' in result
        assert len(result['time']) > 0

    def test_worst_method(self, sample_data):
        from liulian.viz.prediction_aggregator import aggregate_predictions

        preds, trues, times = sample_data
        result = aggregate_predictions(preds, trues, times, method='worst')
        assert len(result['time']) > 0

    def test_single_method(self, sample_data):
        from liulian.viz.prediction_aggregator import aggregate_predictions

        preds, trues, times = sample_data
        result = aggregate_predictions(preds, trues, times, method='single')
        assert len(result['time']) > 0
        # 'single' should have fewer or equal time points (no overlap)
        result_mean = aggregate_predictions(preds, trues, times, method='mean')
        assert len(result['time']) <= len(result_mean['time'])

    def test_all_methods(self, sample_data):
        from liulian.viz.prediction_aggregator import aggregate_predictions

        preds, trues, times = sample_data
        for method in [
            'longest_history',
            'last',
            'mean',
            'median',
            'best',
            'worst',
            'single',
        ]:
            result = aggregate_predictions(preds, trues, times, method=method)
            assert result['pred'].ndim == 2
            assert result['true'].ndim == 2

    def test_unknown_method_raises(self, sample_data):
        from liulian.viz.prediction_aggregator import aggregate_predictions

        preds, trues, times = sample_data
        with pytest.raises(ValueError, match='Unknown aggregation'):
            aggregate_predictions(preds, trues, times, method='invalid')


# ===========================================================================
# 6. Viz: prediction range plots
# ===========================================================================


class TestPredictionRangePlots:
    def test_plot_prediction_range(self, tmp_path):
        from liulian.viz.plots import plot_prediction_range

        N, pred_len, C = 10, 4, 1
        preds = np.random.randn(N, pred_len, C).astype(np.float32)
        trues = np.random.randn(N, pred_len, C).astype(np.float32)
        win_len = 16 + pred_len
        times = np.array([np.arange(i, i + win_len) for i in range(N)])

        save_path = str(tmp_path / 'range.png')
        plot_prediction_range(
            preds,
            trues,
            times,
            save_path=save_path,
        )
        assert os.path.exists(save_path)

    def test_save_prediction_plots_includes_range(self, tmp_path):
        from liulian.viz.plots import save_prediction_plots

        N, pred_len, C = 10, 4, 1
        preds = np.random.randn(N, pred_len, C).astype(np.float32)
        trues = np.random.randn(N, pred_len, C).astype(np.float32)
        win_len = 16 + pred_len
        times = np.array([np.arange(i, i + win_len) for i in range(N)])

        paths = save_prediction_plots(
            preds,
            trues,
            times,
            output_dir=str(tmp_path),
        )
        assert 'pred_vs_gt' in paths
        assert 'pred_range' in paths
        assert os.path.exists(paths['pred_range'])


# ===========================================================================
# 7. CLI enhancements
# ===========================================================================


class TestCLIEnhancements:
    def test_train_subcommand_exists(self):
        from liulian.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(['train', '--help'])
        assert exc_info.value.code == 0

    def test_predict_subcommand_exists(self):
        from liulian.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(['predict', '--help'])
        assert exc_info.value.code == 0

    def test_viz_subcommand_exists(self):
        from liulian.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(['viz', '--help'])
        assert exc_info.value.code == 0

    def test_tune_subcommand_exists(self):
        from liulian.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(['tune', '--help'])
        assert exc_info.value.code == 0

    def test_info_still_works(self, capsys):
        from liulian.cli import main

        main(['info'])
        captured = capsys.readouterr()
        assert 'liulian' in captured.out

    def test_run_still_works_help(self):
        from liulian.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(['run', '--help'])
        assert exc_info.value.code == 0


# ===========================================================================
# 8. Experiment HPO branch
# ===========================================================================


class TestExperimentHPO:
    def test_hpo_branch_with_fallback(self, dummy_config, dummy_loaders):
        """Test that HPO branch works with fallback grid sweep."""
        from liulian.optim.ray_optimizer import RayOptimizer
        from liulian.runtime import Experiment, ExperimentSpec
        from liulian.tasks.base import PredictionRegime, PredictionTask

        cfg = {
            **dummy_config,
            'search_space': {
                'learning_rate': [0.001, 0.01],
            },
        }
        optimizer = RayOptimizer(
            config={
                'num_samples': 2,
                'max_epochs': 1,
            }
        )
        task = PredictionTask(
            regime=PredictionRegime(horizon=4, context_length=16),
        )
        spec = ExperimentSpec(name='test_hpo', task={}, dataset={}, model={})
        model = _TinyModel()

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=MagicMock(),
            model=None,
            torch_model=model,
            optimizer=optimizer,
            data_loaders=dummy_loaders,
            config=cfg,
        )
        summary = exp.run(train=True)
        assert summary['status'] == 'ok'
        assert 'hpo' in summary['metrics']
        # With Ray installed, grid_search([0.001, 0.01]) × num_samples=2 = 4;
        # without Ray the fallback grid sweep caps at num_samples=2.
        assert summary['metrics']['hpo']['n_trials'] >= 2


# ===========================================================================
# 9. Experiment auto-viz
# ===========================================================================


class TestExperimentAutoViz:
    def test_auto_viz_generates_plots(self, dummy_config, dummy_loaders, tmp_path):
        from liulian.runtime import Experiment, ExperimentSpec
        from liulian.tasks.base import PredictionRegime, PredictionTask

        cfg = {**dummy_config, 'auto_viz': True}
        task = PredictionTask(
            regime=PredictionRegime(horizon=4, context_length=16),
        )
        spec = ExperimentSpec(name='test_viz', task={}, dataset={}, model={})
        model = _TinyModel()

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=MagicMock(),
            model=None,
            torch_model=model,
            data_loaders=dummy_loaders,
            config=cfg,
        )
        summary = exp.run(train=True)
        assert summary['status'] == 'ok'
        assert 'metrics' in summary
        # Predictions must be present for auto_viz to produce plots
        assert 'predictions' in summary, (
            'Expected predictions in summary for auto_viz test'
        )
        assert 'viz_paths' in summary


# ===========================================================================
# 10. E2E with dummy data
# ===========================================================================


class TestE2EDummy:
    """End-to-end test with synthetic dummy data through the full pipeline."""

    def test_full_pipeline(self):
        from liulian.runtime import Experiment, ExperimentSpec
        from liulian.tasks.base import PredictionRegime, PredictionTask

        B, seq_len, pred_len, C = 16, 16, 4, 2
        x = torch.randn(B, seq_len, C)
        y = torch.randn(B, pred_len + seq_len, 1)
        xm = torch.zeros(B, seq_len, 1)
        ym = torch.zeros(B, pred_len + seq_len, 1)

        ds = TensorDataset(x, y, xm, ym)
        loader = DataLoader(ds, batch_size=4)
        loaders = {'train': loader, 'val': loader, 'test': loader}

        config = {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'label_len': 0,
            'features': 'M',
            'train_epochs': 2,
            'learning_rate': 0.01,
            'patience': 2,
            'loss': 'mse',
            'metrics': ['rmse', 'mae'],
            'show_progress': False,
        }

        task = PredictionTask(
            regime=PredictionRegime(horizon=pred_len, context_length=seq_len),
        )
        spec = ExperimentSpec(
            name='e2e_dummy',
            task={},
            dataset={},
            model={},
        )
        model = _TinyModel(enc_in=C, pred_len=pred_len)

        exp = Experiment(
            spec=spec,
            task=task,
            dataset=MagicMock(),
            model=None,
            torch_model=model,
            data_loaders=loaders,
            config=config,
        )
        summary = exp.run(train=True, eval=True)

        assert summary['status'] == 'ok'
        assert 'training' in summary['metrics']
        assert summary['metrics']['training']['epochs_run'] == 2
        assert 'final_test' in summary['metrics']
        assert summary['metrics']['training']['epochs_run'] == 2
        assert 'predictions' in summary
        assert summary['predictions']['preds'].shape[1] == pred_len
