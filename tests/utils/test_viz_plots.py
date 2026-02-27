"""Tests for liulian.viz.plots — plotting and format_metrics_table.

Uses matplotlib with Agg backend (headless). Verifies file creation,
return values, and basic content—not pixel-level rendering.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from liulian.viz.plots import (
    format_metrics_table,
    plot_prediction_range,
    plot_predictions,
    plot_prediction_summary,
    save_prediction_plots,
)


# ---------------------------------------------------------------------------
# format_metrics_table
# ---------------------------------------------------------------------------


class TestFormatMetricsTable:
    def test_basic(self):
        out = format_metrics_table({'mse': 0.01, 'mae': 0.05})
        assert 'mse' in out
        assert 'mae' in out
        assert '0.010000' in out

    def test_empty(self):
        out = format_metrics_table({})
        assert 'no metrics' in out.lower()

    def test_custom_title(self):
        out = format_metrics_table({'rmse': 0.1}, title='Val Metrics')
        assert 'Val Metrics' in out


# ---------------------------------------------------------------------------
# plot_predictions
# ---------------------------------------------------------------------------


class TestPlotPredictions:
    @pytest.fixture
    def tmpdir(self, tmp_path):
        return str(tmp_path)

    def test_save_creates_file(self, tmpdir):
        T, C = 20, 2
        time = np.arange(T)
        true = np.random.randn(T, C).astype(np.float32)
        pred = true + 0.1 * np.random.randn(T, C).astype(np.float32)
        save_path = os.path.join(tmpdir, 'preds.png')
        plot_predictions(time, true, pred, save_path=save_path)
        assert os.path.isfile(save_path)
        assert os.path.getsize(save_path) > 100  # not empty

    def test_univariate(self, tmpdir):
        T = 15
        time = np.arange(T)
        true = np.sin(time * 0.5).astype(np.float32).reshape(T, 1)
        pred = (true + 0.05).reshape(T, 1)
        save_path = os.path.join(tmpdir, 'uni.png')
        plot_predictions(time, true, pred, save_path=save_path)
        assert os.path.isfile(save_path)

    def test_custom_labels(self, tmpdir):
        T, C = 10, 2
        time = np.arange(T)
        true = np.zeros((T, C), dtype=np.float32)
        pred = np.zeros((T, C), dtype=np.float32)
        save_path = os.path.join(tmpdir, 'labels.png')
        plot_predictions(
            time,
            true,
            pred,
            target_names=['Discharge', 'Level'],
            title='Custom Plot',
            xlabel='Day',
            ylabel='m3/s',
            save_path=save_path,
        )
        assert os.path.isfile(save_path)


# ---------------------------------------------------------------------------
# plot_prediction_summary
# ---------------------------------------------------------------------------


class TestPlotPredictionSummary:
    def test_save(self, tmp_path):
        T, C = 20, 1
        # Simulate 4 entities
        results = []
        for i in range(4):
            results.append(
                {
                    'time': np.arange(T),
                    'pred': np.random.randn(T, C).astype(np.float32),
                    'true': np.random.randn(T, C).astype(np.float32),
                }
            )
        save_path = str(tmp_path / 'summary.png')
        plot_prediction_summary(results, save_path=save_path)
        assert os.path.isfile(save_path)

    def test_empty_results(self):
        """Empty input should return without error and not create files."""
        result = plot_prediction_summary([])
        # plot_prediction_summary returns None on empty input
        assert result is None


# ---------------------------------------------------------------------------
# plot_prediction_range
# ---------------------------------------------------------------------------


class TestPlotPredictionRange:
    def test_save_file(self, tmp_path):
        N, pred_len, C = 6, 3, 1
        ctx = 5
        win_len = ctx + pred_len
        preds = np.random.randn(N, pred_len, C).astype(np.float32)
        trues = np.random.randn(N, pred_len, C).astype(np.float32)
        times = np.array([np.arange(i, i + win_len) for i in range(N)])
        save_path = str(tmp_path / 'range.png')
        plot_prediction_range(preds, trues, times, save_path=save_path)
        assert os.path.isfile(save_path)
        assert os.path.getsize(save_path) > 100

    def test_multivariate(self, tmp_path):
        N, pred_len, C = 4, 2, 3
        ctx = 4
        win_len = ctx + pred_len
        preds = np.random.randn(N, pred_len, C).astype(np.float32)
        trues = np.random.randn(N, pred_len, C).astype(np.float32)
        times = np.array([np.arange(i, i + win_len) for i in range(N)])
        save_path = str(tmp_path / 'range_mv.png')
        plot_prediction_range(
            preds,
            trues,
            times,
            target_names=['Q', 'H', 'T'],
            save_path=save_path,
        )
        assert os.path.isfile(save_path)


# ---------------------------------------------------------------------------
# save_prediction_plots (high-level)
# ---------------------------------------------------------------------------


class TestSavePredictionPlots:
    def test_returns_paths(self, tmp_path):
        N, pred_len, C = 6, 3, 1
        ctx = 5
        win_len = ctx + pred_len
        preds = np.random.randn(N, pred_len, C).astype(np.float32)
        trues = np.random.randn(N, pred_len, C).astype(np.float32)
        times = np.array([np.arange(i, i + win_len) for i in range(N)])
        out_dir = str(tmp_path / 'viz_out')
        paths = save_prediction_plots(
            preds,
            trues,
            times,
            method='mean',
            output_dir=out_dir,
        )
        assert 'pred_vs_gt' in paths
        assert 'pred_range' in paths
        for p in paths.values():
            assert os.path.isfile(p)

    def test_all_methods(self, tmp_path):
        """save_prediction_plots should work with every aggregation method."""
        N, pred_len, C = 8, 3, 1
        ctx = 5
        win_len = ctx + pred_len
        preds = np.random.randn(N, pred_len, C).astype(np.float32)
        trues = np.random.randn(N, pred_len, C).astype(np.float32)
        times = np.array([np.arange(i, i + win_len) for i in range(N)])
        for method in (
            'mean',
            'median',
            'last',
            'longest_history',
            'best',
            'worst',
            'single',
        ):
            out_dir = str(tmp_path / f'viz_{method}')
            paths = save_prediction_plots(
                preds,
                trues,
                times,
                method=method,
                output_dir=out_dir,
            )
            for p in paths.values():
                assert os.path.isfile(p), f'Missing file for method={method}'
