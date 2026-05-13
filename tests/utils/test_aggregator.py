"""Tests for liulian.viz.prediction_aggregator — all 7 aggregation methods.

Each test uses deterministic data so we can assert exact values, not just shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from liulian.viz.prediction_aggregator import aggregate_predictions


# ---------------------------------------------------------------------------
# Helpers — build tiny synthetic sliding-window data
# ---------------------------------------------------------------------------


def _make_windows(
    n_windows: int = 4,
    pred_len: int = 3,
    context_len: int = 5,
    n_features: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic overlapping sliding-window data.

    Returns (preds, trues, times) shaped:
        preds: (N, pred_len, C)
        trues: (N, pred_len, C)
        times: (N, context_len + pred_len)
    """
    rng = np.random.default_rng(seed)
    win_len = context_len + pred_len
    # Simulate sliding windows with stride=1 for maximal overlap
    times = np.array([np.arange(i, i + win_len) for i in range(n_windows)])
    # Predictions: each window gets slightly different values
    preds = rng.normal(size=(n_windows, pred_len, n_features)).astype(np.float32)
    # Ground truth: deterministic from the time index so the same timestep
    # always has the same true value regardless of which window contains it.
    trues = np.zeros_like(preds)
    for n in range(n_windows):
        for p in range(pred_len):
            t_idx = times[n, context_len + p]
            trues[n, p, :] = float(t_idx) * 0.1  # simple deterministic GT
    return preds, trues, times


# ---------------------------------------------------------------------------
# Test: mean method
# ---------------------------------------------------------------------------


class TestMean:
    def test_output_keys_and_shapes(self):
        preds, trues, times = _make_windows()
        result = aggregate_predictions(preds, trues, times, method='mean')
        assert set(result.keys()) == {'time', 'pred', 'true'}
        T = len(result['time'])
        assert result['pred'].shape == (T, 1)
        assert result['true'].shape == (T, 1)

    def test_exact_values(self):
        """With known data, verify that mean is computed correctly."""
        # 2 windows, pred_len=2, 1 feature, stride-1 → overlap at t=6
        preds = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
        trues = np.array([[[0.5], [0.6]], [[0.5], [0.7]]], dtype=np.float32)
        # Window 0: context=[0,1,2,3,4], predict=[5,6]
        # Window 1: context=[1,2,3,4,5], predict=[6,7]
        times = np.array([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])
        result = aggregate_predictions(preds, trues, times, method='mean')
        # Unique prediction times: 5, 6, 7
        np.testing.assert_array_equal(result['time'], [5, 6, 7])
        # t=5: only window 0 → pred=1.0
        # t=6: window 0 pred=2.0, window 1 pred=3.0 → mean=2.5
        # t=7: only window 1 → pred=4.0
        expected_pred = np.array([[1.0], [2.5], [4.0]], dtype=np.float32)
        np.testing.assert_allclose(result['pred'], expected_pred)

    def test_multivariate(self):
        preds, trues, times = _make_windows(n_features=3)
        result = aggregate_predictions(preds, trues, times, method='mean')
        assert result['pred'].shape[1] == 3


# ---------------------------------------------------------------------------
# Test: median method
# ---------------------------------------------------------------------------


class TestMedian:
    def test_exact_values(self):
        # 3 overlapping windows, pred_len=2
        # Window 0: pred_times=[6,7], preds=[10,11]
        # Window 1: pred_times=[7,8], preds=[20,21]
        # Window 2: pred_times=[8,9], preds=[30,31]
        preds = np.array(
            [
                [[10.0], [11.0]],
                [[20.0], [21.0]],
                [[30.0], [31.0]],
            ],
            dtype=np.float32,
        )
        trues = np.ones((3, 2, 1), dtype=np.float32)
        times = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 9],
            ]
        )
        result = aggregate_predictions(preds, trues, times, method='median')
        # t=7: w0 pred=11, w1 pred=20 → median of [11,20] = 15.5
        idx_t7 = np.where(result['time'] == 7)[0][0]
        assert result['pred'][idx_t7, 0] == pytest.approx(15.5)
        # t=8: w1 pred=21, w2 pred=30 → median of [21,30] = 25.5
        idx_t8 = np.where(result['time'] == 8)[0][0]
        assert result['pred'][idx_t8, 0] == pytest.approx(25.5)

    def test_shape_matches_mean(self):
        preds, trues, times = _make_windows()
        r_mean = aggregate_predictions(preds, trues, times, method='mean')
        r_median = aggregate_predictions(preds, trues, times, method='median')
        assert r_mean['pred'].shape == r_median['pred'].shape


# ---------------------------------------------------------------------------
# Test: longest_history (keep first occurrence)
# ---------------------------------------------------------------------------


class TestLongestHistory:
    def test_keeps_first(self):
        # t=6 appears in window 0 (first) and window 1 (second)
        preds = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
        trues = np.ones((2, 2, 1), dtype=np.float32)
        times = np.array([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])
        result = aggregate_predictions(preds, trues, times, method='longest_history')
        idx_t6 = np.where(result['time'] == 6)[0][0]
        # Window 0 predicts 2.0 at t=6 — should be kept
        assert result['pred'][idx_t6, 0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test: last (keep last occurrence)
# ---------------------------------------------------------------------------


class TestLast:
    def test_keeps_last(self):
        preds = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
        trues = np.ones((2, 2, 1), dtype=np.float32)
        times = np.array([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])
        result = aggregate_predictions(preds, trues, times, method='last')
        idx_t6 = np.where(result['time'] == 6)[0][0]
        # Window 1 predicts 3.0 at t=6 — should be kept
        assert result['pred'][idx_t6, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Test: best (closest to truth)
# ---------------------------------------------------------------------------


class TestBest:
    def test_picks_closest(self):
        # t=6: window 0 pred=2.0, window 1 pred=3.0, true=2.1 → best=2.0
        preds = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
        trues = np.array([[[0.0], [2.1]], [[0.0], [2.1]]], dtype=np.float32)
        times = np.array([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])
        result = aggregate_predictions(preds, trues, times, method='best')
        idx_t6 = np.where(result['time'] == 6)[0][0]
        # Window 0 pred 2.0 has error |2.0 - 2.1| = 0.1
        # Window 1 pred 3.0 has error |3.0 - 2.1| = 0.9
        assert result['pred'][idx_t6, 0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test: worst (farthest from truth)
# ---------------------------------------------------------------------------


class TestWorst:
    def test_picks_farthest(self):
        preds = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
        trues = np.array([[[0.0], [2.1]], [[0.0], [2.1]]], dtype=np.float32)
        times = np.array([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])
        result = aggregate_predictions(preds, trues, times, method='worst')
        idx_t6 = np.where(result['time'] == 6)[0][0]
        # Window 1 pred 3.0 has larger error → should be picked
        assert result['pred'][idx_t6, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Test: single (no-overlap, stride by pred_len)
# ---------------------------------------------------------------------------


class TestSingle:
    def test_no_overlap(self):
        """Windows should be strided by pred_len → no duplicate timesteps."""
        preds, trues, times = _make_windows(n_windows=6, pred_len=3)
        result = aggregate_predictions(preds, trues, times, method='single')
        # Each selected window contributes 3 unique time-steps; stride=3
        # so we should see only windows 0 and 3 (indices 0, 3)
        # Window 0 pred_times = [5,6,7], Window 3 pred_times = [8,9,10]
        unique_times = result['time']
        assert len(unique_times) == len(np.unique(unique_times)), (
            'single method should produce unique timesteps'
        )

    def test_values_from_correct_window(self):
        """Values should come from the selected (strided) windows."""
        preds = np.array(
            [
                [[10.0], [11.0]],
                [[20.0], [21.0]],
                [[30.0], [31.0]],
                [[40.0], [41.0]],
            ],
            dtype=np.float32,
        )
        trues = np.ones((4, 2, 1), dtype=np.float32)
        times = np.array(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
            ]
        )
        result = aggregate_predictions(preds, trues, times, method='single')
        # pred_len=2, stride=2 → select windows 0 and 2
        # Window 0 pred: [10, 11], Window 2 pred: [30, 31]
        assert 10.0 in result['pred'][:, 0]
        assert 30.0 in result['pred'][:, 0]


# ---------------------------------------------------------------------------
# Test: multi-entity windows (per-entity split)
# ---------------------------------------------------------------------------


class TestMultiEntity:
    def test_longest_history_is_entity_aware_when_entity_ids_given(self):
        """Overlaps should be resolved per entity, then averaged for global viz."""
        # Two entities (A/B), each with two overlapping windows:
        # A t=6: [2, 3] -> longest_history picks 2
        # B t=6: [20, 30] -> longest_history picks 20
        # Global aggregate should therefore use mean([2, 20]) = 11.
        preds = np.array(
            [
                [[1.0], [2.0]],   # A window 0  -> times 5,6
                [[3.0], [4.0]],   # A window 1  -> times 6,7
                [[10.0], [20.0]], # B window 0  -> times 5,6
                [[30.0], [40.0]], # B window 1  -> times 6,7
            ],
            dtype=np.float32,
        )
        trues = np.ones_like(preds, dtype=np.float32)
        times = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5, 6, 7],
                [0, 1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5, 6, 7],
            ]
        )
        entity_ids = np.array(['A', 'A', 'B', 'B'])

        result = aggregate_predictions(
            preds,
            trues,
            times,
            method='longest_history',
            entity_ids=entity_ids,
        )
        idx_t6 = np.where(result['time'] == 6)[0][0]
        assert result['pred'][idx_t6, 0] == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# Test: unknown method raises
# ---------------------------------------------------------------------------


def test_unknown_method_raises():
    preds, trues, times = _make_windows()
    with pytest.raises(ValueError, match='Unknown aggregation'):
        aggregate_predictions(preds, trues, times, method='nonexistent')


# ---------------------------------------------------------------------------
# Test: torch tensor inputs
# ---------------------------------------------------------------------------


def test_torch_tensor_input():
    """aggregate_predictions should accept torch.Tensor inputs."""
    torch = pytest.importorskip('torch')
    preds_np, trues_np, times_np = _make_windows()
    preds = torch.from_numpy(preds_np)
    trues = torch.from_numpy(trues_np)
    times = torch.from_numpy(times_np.astype(np.float32))
    result = aggregate_predictions(preds, trues, times, method='mean')
    assert isinstance(result['pred'], np.ndarray)
    T = len(result['time'])
    assert result['pred'].shape == (T, 1)


# ---------------------------------------------------------------------------
# Test: explicit pred_len override
# ---------------------------------------------------------------------------


def test_pred_len_override():
    """pred_len kwarg should be respected."""
    preds, trues, times = _make_windows(pred_len=3)
    # Only aggregate 2 of 3 prediction steps
    result = aggregate_predictions(preds, trues, times, method='mean', pred_len=2)
    # Should have fewer unique times than when using all 3
    result_full = aggregate_predictions(preds, trues, times, method='mean')
    assert len(result['time']) <= len(result_full['time'])


# ---------------------------------------------------------------------------
# Test: chronological ordering
# ---------------------------------------------------------------------------


def test_output_is_chronological():
    """Returned time axis should be sorted."""
    preds, trues, times = _make_windows()
    for method in (
        'mean',
        'median',
        'last',
        'longest_history',
        'best',
        'worst',
        'single',
    ):
        result = aggregate_predictions(preds, trues, times, method=method)
        assert np.all(np.diff(result['time']) >= 0), (
            f'method={method!r}: time not sorted'
        )
