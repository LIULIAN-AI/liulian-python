"""Prediction aggregation middleware for overlapping sliding-window outputs.

When a model produces predictions over a sliding window, each future
time-step may be predicted by multiple windows (with different amounts
of look-back history).  This module aggregates those overlapping
predictions into a single time-series per target column.

Supported aggregation methods (from the reference project):

* ``'longest_history'`` — keep only the prediction from the window with
  the longest available history (i.e. the first time a day appears).
* ``'last'``            — keep only the prediction from the *last* window
  that covers each time-step.
* ``'mean'``            — arithmetic mean of all overlapping predictions.
* ``'median'``          — median of all overlapping predictions.
* ``'best'``            — keep the prediction closest to ground truth
  (lowest absolute error) at each time-step.
* ``'worst'``           — keep the prediction farthest from ground truth
  (highest absolute error) at each time-step.
* ``'single'``          — no-overlap mode: take only the last pred_len
  non-overlapping windows (stride = pred_len).

Usage::

    from liulian.viz.prediction_aggregator import aggregate_predictions

    result = aggregate_predictions(preds, trues, times, method='mean')
    # result['time']  — unique sorted time indices  (T,)
    # result['pred']  — aggregated predictions       (T, C)
    # result['true']  — ground-truth values           (T, C)
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import torch


AggMethod = Literal[
    'longest_history',
    'last',
    'mean',
    'median',
    'best',
    'worst',
    'single',
]


def aggregate_predictions(
    preds: torch.Tensor | np.ndarray,
    trues: torch.Tensor | np.ndarray,
    times: torch.Tensor | np.ndarray,
    *,
    method: AggMethod = 'mean',
    pred_len: int | None = None,
) -> Dict[str, np.ndarray]:
    """Aggregate overlapping sliding-window predictions into one time-series.

    Parameters
    ----------
    preds : array-like, shape ``(N, pred_len, C)``
        Model outputs for *N* windows.
    trues : array-like, shape ``(N, pred_len, C)``
        Ground-truth targets aligned with *preds*.
    times : array-like, shape ``(N, win_len)``
        Epoch-day (or other numeric time index) for each position in the
        full window.  Only the **last** ``pred_len`` entries are used as
        the target time indices.
    method : str
        Aggregation strategy (see module docstring).
    pred_len : int or None
        If *None*, inferred from ``preds.shape[1]``.

    Returns
    -------
    dict
        ``time`` — ``(T,)``  unique sorted time indices.
        ``pred`` — ``(T, C)`` aggregated predictions.
        ``true`` — ``(T, C)`` ground-truth values.
    """
    # todo: fixme: this function may be incorrect if there are multiple stations. It might influence viz as well.
    preds = _to_numpy(preds)
    trues = _to_numpy(trues)
    times = _to_numpy(times)

    N = preds.shape[0]
    if pred_len is None:
        pred_len = preds.shape[1]
    C = preds.shape[2]

    # Extract time indices corresponding to the prediction horizon
    # times has shape (N, win_len); we need the last pred_len entries.
    pred_times = times[:, -pred_len:]  # (N, pred_len)

    # Flatten
    flat_times = pred_times.reshape(-1).astype(np.int64)
    flat_preds = preds.reshape(-1, C)
    flat_trues = trues.reshape(-1, C)

    if method == 'longest_history':
        return _agg_longest_history(flat_times, flat_preds, flat_trues, C)
    elif method == 'last':
        return _agg_last(flat_times, flat_preds, flat_trues, C)
    elif method == 'mean':
        return _agg_reduce(flat_times, flat_preds, flat_trues, C, np.mean)
    elif method == 'median':
        return _agg_reduce(flat_times, flat_preds, flat_trues, C, np.median)
    elif method == 'best':
        return _agg_best_worst(flat_times, flat_preds, flat_trues, C, best=True)
    elif method == 'worst':
        return _agg_best_worst(flat_times, flat_preds, flat_trues, C, best=False)
    elif method == 'single':
        return _agg_single(preds, trues, times, pred_len, C)
    else:
        raise ValueError(f'Unknown aggregation method: {method!r}')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _agg_longest_history(
    flat_times: np.ndarray,
    flat_preds: np.ndarray,
    flat_trues: np.ndarray,
    C: int,
) -> Dict[str, np.ndarray]:
    """Keep first occurrence (= longest history) for each time step."""  # todo, fixme: I don't think this is correct.
    unique_times, first_idx = np.unique(flat_times, return_index=True)
    return {
        'time': unique_times,
        'pred': flat_preds[first_idx],
        'true': flat_trues[first_idx],
    }


def _agg_last(
    flat_times: np.ndarray,
    flat_preds: np.ndarray,
    flat_trues: np.ndarray,
    C: int,
) -> Dict[str, np.ndarray]:
    """Keep the last occurrence for each time step."""
    # Reverse, then use np.unique (which takes first occurrence)
    rev_times = flat_times[::-1]
    rev_preds = flat_preds[::-1]
    rev_trues = flat_trues[::-1]
    unique_times, first_idx = np.unique(rev_times, return_index=True)
    # Sort back to chronological order
    order = np.argsort(unique_times)
    return {
        'time': unique_times[order],
        'pred': rev_preds[first_idx][order],
        'true': rev_trues[first_idx][order],
    }


def _agg_reduce(
    flat_times: np.ndarray,
    flat_preds: np.ndarray,
    flat_trues: np.ndarray,
    C: int,
    reduce_fn,
) -> Dict[str, np.ndarray]:
    """Aggregate by grouping on time index and applying *reduce_fn*."""
    unique_times = np.unique(flat_times)
    unique_times.sort()
    T = len(unique_times)
    time_to_idx = {int(t): i for i, t in enumerate(unique_times)}

    # Group predictions
    pred_groups: list[list[np.ndarray]] = [[] for _ in range(T)]
    true_groups: list[list[np.ndarray]] = [[] for _ in range(T)]

    for i, t in enumerate(flat_times):
        idx = time_to_idx[int(t)]
        pred_groups[idx].append(flat_preds[i])
        true_groups[idx].append(flat_trues[i])

    agg_pred = np.zeros((T, C), dtype=np.float32)
    agg_true = np.zeros((T, C), dtype=np.float32)
    for i in range(T):
        agg_pred[i] = reduce_fn(pred_groups[i], axis=0)
        agg_true[i] = reduce_fn(true_groups[i], axis=0)

    return {
        'time': unique_times,
        'pred': agg_pred,
        'true': agg_true,
    }


def _agg_best_worst(
    flat_times: np.ndarray,
    flat_preds: np.ndarray,
    flat_trues: np.ndarray,
    C: int,
    *,
    best: bool = True,
) -> Dict[str, np.ndarray]:
    """Keep the prediction closest to (best) or farthest from (worst) truth."""
    unique_times = np.unique(flat_times)
    unique_times.sort()
    T = len(unique_times)
    time_to_idx = {int(t): i for i, t in enumerate(unique_times)}

    pred_groups: list[list[np.ndarray]] = [[] for _ in range(T)]
    true_groups: list[list[np.ndarray]] = [[] for _ in range(T)]

    for i, t in enumerate(flat_times):
        idx = time_to_idx[int(t)]
        pred_groups[idx].append(flat_preds[i])
        true_groups[idx].append(flat_trues[i])

    agg_pred = np.zeros((T, C), dtype=np.float32)
    agg_true = np.zeros((T, C), dtype=np.float32)

    for i in range(T):
        preds_arr = np.array(pred_groups[i])  # (K, C)
        trues_arr = np.array(true_groups[i])  # (K, C)
        errors = np.mean(np.abs(preds_arr - trues_arr), axis=1)  # (K,)
        if best:
            pick = int(np.argmin(errors))
        else:
            pick = int(np.argmax(errors))
        agg_pred[i] = preds_arr[pick]
        agg_true[i] = trues_arr[pick]

    return {
        'time': unique_times,
        'pred': agg_pred,
        'true': agg_true,
    }


def _agg_single(
    preds: np.ndarray,
    trues: np.ndarray,
    times: np.ndarray,
    pred_len: int,
    C: int,
) -> Dict[str, np.ndarray]:
    """No-overlap aggregation: stride by pred_len, take non-overlapping windows.

    Selects every ``pred_len``-th window so that predictions don't overlap.
    """
    N = preds.shape[0]
    stride = max(pred_len, 1)

    selected_indices = list(range(0, N, stride))
    if not selected_indices:
        selected_indices = [0]

    sel_preds = preds[selected_indices]  # (S, pred_len, C)
    sel_trues = trues[selected_indices]  # (S, pred_len, C)
    sel_times = times[selected_indices, -pred_len:]  # (S, pred_len)

    # Flatten to a single time-series
    flat_times = sel_times.reshape(-1).astype(np.int64)
    flat_preds = sel_preds.reshape(-1, C)
    flat_trues = sel_trues.reshape(-1, C)

    # Deduplicate (shouldn't overlap, but handle edge cases)
    unique_times, first_idx = np.unique(flat_times, return_index=True)
    order = np.argsort(unique_times)

    return {
        'time': unique_times[order],
        'pred': flat_preds[first_idx][order],
        'true': flat_trues[first_idx][order],
    }
