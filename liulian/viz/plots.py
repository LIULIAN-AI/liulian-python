"""Plotting utilities for experiment results.

Provides:

* :func:`format_metrics_table` — ASCII metric table for console output.
* :func:`plot_predictions`     — actual vs predicted line chart.
* :func:`plot_prediction_summary` — mean ± std across multiple entities.
* :func:`plot_prediction_range`   — min/max/mean prediction envelope.
* :func:`save_prediction_plots`   — high-level: aggregates + plots + saves.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np


def format_metrics_table(metrics: Dict[str, float], title: str = 'Metrics') -> str:
    """Format a metrics dictionary as a simple ASCII table.

    Args:
        metrics: Metric name → scalar value mapping.
        title: Table title printed as the header line.

    Returns:
        Multi-line string suitable for ``print()``.
    """
    if not metrics:
        return f'{title}: (no metrics)'

    # Determine column widths
    name_width = max(len(k) for k in metrics)
    lines = [title, '-' * (name_width + 14)]
    for name, value in metrics.items():
        lines.append(f'  {name:<{name_width}}  {value:>10.6f}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# matplotlib helpers
# ---------------------------------------------------------------------------

def _import_mpl():
    """Import matplotlib with Agg backend (safe for headless servers)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_predictions(
    time: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    *,
    title: str = 'Predictions vs Ground Truth',
    xlabel: str = 'Time (epoch day)',
    ylabel: str = 'Value',
    target_names: Sequence[str] | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> None:
    """Plot actual vs predicted time-series.

    Parameters
    ----------
    time : (T,) array
        Time indices (x-axis).
    true : (T,) or (T, C) array
        Ground-truth values.
    pred : (T,) or (T, C) array
        Predicted values.
    title, xlabel, ylabel : str
        Plot labels.
    target_names : list of str or None
        Per-column labels (for multi-target).  Falls back to
        ``Target 0, Target 1, …``.
    save_path : str or None
        When provided the figure is saved to disk and closed.
    figsize : tuple
        Figure dimensions in inches.
    """
    plt = _import_mpl()

    true = np.atleast_2d(true) if true.ndim == 1 else true
    pred = np.atleast_2d(pred) if pred.ndim == 1 else pred
    if true.ndim == 1:
        true = true[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]
    C = true.shape[1]
    if target_names is None:
        target_names = [f'Target {i}' for i in range(C)]

    fig, axes = plt.subplots(C, 1, figsize=(figsize[0], figsize[1] * C),
                             squeeze=False, sharex=True)

    for c in range(C):
        ax = axes[c, 0]
        ax.plot(time, true[:, c], label='Ground Truth', linewidth=1.0, alpha=0.85)
        ax.plot(time, pred[:, c], label='Prediction', linewidth=1.0, alpha=0.85)
        ax.set_ylabel(target_names[c])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlabel(xlabel)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_prediction_summary(
    results: List[Dict[str, np.ndarray]],
    *,
    title: str = 'Mean ± Std Across Entities',
    xlabel: str = 'Time (epoch day)',
    ylabel: str = 'Value',
    save_path: str | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> None:
    """Plot mean ± std of predictions across multiple entity results.

    Each element of *results* is a dict with ``time``, ``pred``, ``true``
    (as returned by :func:`aggregate_predictions`).  All must share the
    same time axis.

    Parameters
    ----------
    results : list of dict
        Per-entity aggregation results.
    """
    if not results:
        return
    plt = _import_mpl()

    time = results[0]['time']
    preds = np.stack([r['pred'] for r in results])  # (E, T, C)
    trues = np.stack([r['true'] for r in results])

    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    true_mean = trues.mean(axis=0)

    C = pred_mean.shape[1] if pred_mean.ndim == 2 else 1
    if pred_mean.ndim == 1:
        pred_mean = pred_mean[:, None]
        pred_std = pred_std[:, None]
        true_mean = true_mean[:, None]

    fig, axes = plt.subplots(C, 1, figsize=(figsize[0], figsize[1] * C),
                             squeeze=False, sharex=True)
    for c in range(C):
        ax = axes[c, 0]
        ax.plot(time, true_mean[:, c], label='GT Mean', linewidth=1.2, color='#1f77b4')
        ax.plot(time, pred_mean[:, c], label='Pred Mean', linewidth=1.2, color='#ff7f0e')
        ax.fill_between(
            time,
            pred_mean[:, c] - pred_std[:, c],
            pred_mean[:, c] + pred_std[:, c],
            alpha=0.2, color='#ff7f0e', label='Pred ±1σ',
        )
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlabel(xlabel)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_prediction_range(
    preds: np.ndarray,
    trues: np.ndarray,
    times: np.ndarray,
    *,
    pred_len: int | None = None,
    title: str = 'Prediction Range (Min/Max/Mean)',
    xlabel: str = 'Time (epoch day)',
    ylabel: str = 'Value',
    target_names: Sequence[str] | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> None:
    """Plot prediction envelope: min, max, mean across overlapping windows.

    For each time-step that appears in multiple windows, shows the range
    of predictions as a shaded region and the mean as a line.

    Parameters
    ----------
    preds : (N, pred_len, C) array
        Raw sliding-window predictions.
    trues : (N, pred_len, C) array
        Ground-truth aligned with *preds*.
    times : (N, win_len) array
        Time marks for each window.
    pred_len : int or None
        Prediction horizon (inferred if None).
    title, xlabel, ylabel : str
        Plot labels.
    target_names : list of str or None
        Per-column labels.
    save_path : str or None
        When provided the figure is saved and closed.
    figsize : tuple
        Figure dimensions.
    """
    from liulian.viz.prediction_aggregator import _to_numpy

    plt = _import_mpl()

    preds = _to_numpy(preds)
    trues = _to_numpy(trues)
    times = _to_numpy(times)

    N = preds.shape[0]
    if pred_len is None:
        pred_len = preds.shape[1]
    C = preds.shape[2]

    pred_times = times[:, -pred_len:]  # (N, pred_len)

    # Group predictions by time
    unique_times = np.unique(pred_times.reshape(-1).astype(np.int64))
    unique_times.sort()
    T = len(unique_times)
    time_to_idx = {int(t): i for i, t in enumerate(unique_times)}

    pred_groups: list[list[np.ndarray]] = [[] for _ in range(T)]
    true_vals = np.zeros((T, C), dtype=np.float32)
    true_count = np.zeros(T, dtype=np.int32)

    for n in range(N):
        for p in range(pred_len):
            t_val = int(pred_times[n, p])
            idx = time_to_idx[t_val]
            pred_groups[idx].append(preds[n, p])
            true_vals[idx] += trues[n, p]
            true_count[idx] += 1

    for i in range(T):
        if true_count[i] > 0:
            true_vals[i] /= true_count[i]

    pred_mean = np.zeros((T, C), dtype=np.float32)
    pred_min = np.zeros((T, C), dtype=np.float32)
    pred_max = np.zeros((T, C), dtype=np.float32)
    for i in range(T):
        arr = np.array(pred_groups[i])  # (K, C)
        pred_mean[i] = arr.mean(axis=0)
        pred_min[i] = arr.min(axis=0)
        pred_max[i] = arr.max(axis=0)

    if target_names is None:
        target_names = [f'Target {i}' for i in range(C)]

    fig, axes = plt.subplots(
        C, 1, figsize=(figsize[0], figsize[1] * C), squeeze=False, sharex=True,
    )
    for c in range(C):
        ax = axes[c, 0]
        ax.plot(unique_times, true_vals[:, c], label='Ground Truth',
                linewidth=1.2, color='#1f77b4', alpha=0.9)
        ax.plot(unique_times, pred_mean[:, c], label='Pred Mean',
                linewidth=1.0, color='#ff7f0e', alpha=0.9)
        ax.fill_between(
            unique_times, pred_min[:, c], pred_max[:, c],
            alpha=0.2, color='#ff7f0e', label='Pred Range',
        )
        ax.set_ylabel(target_names[c])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlabel(xlabel)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def save_prediction_plots(
    preds: np.ndarray,
    trues: np.ndarray,
    times: np.ndarray,
    *,
    method: str = 'mean',
    pred_len: int | None = None,
    output_dir: str = 'artifacts',
    title_prefix: str = '',
    target_names: Sequence[str] | None = None,
) -> Dict[str, str]:
    """High-level helper: aggregate → plot → save.

    Parameters
    ----------
    preds : (N, pred_len, C)
        Raw sliding-window predictions.
    trues : (N, pred_len, C)
        Ground-truth aligned with *preds*.
    times : (N, win_len)
        Time marks for each window.
    method : str
        Aggregation method (see :mod:`~liulian.viz.prediction_aggregator`).
    pred_len : int or None
        Prediction horizon (inferred from *preds* if None).
    output_dir : str
        Directory where figure PNGs are saved.
    title_prefix : str
        Prepended to plot titles.
    target_names : list of str or None
        Column labels.

    Returns
    -------
    dict
        Mapping of plot name → file path for each saved figure.
    """
    from liulian.viz.prediction_aggregator import aggregate_predictions

    result = aggregate_predictions(
        preds, trues, times, method=method, pred_len=pred_len,
    )

    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    # Full time-series plot
    ts_path = os.path.join(output_dir, 'pred_vs_gt.png')
    plot_predictions(
        result['time'], result['true'], result['pred'],
        title=f'{title_prefix}Predictions vs Ground Truth ({method})',
        target_names=target_names,
        save_path=ts_path,
    )
    paths['pred_vs_gt'] = ts_path

    # Prediction range plot (min/max/mean envelope)
    range_path = os.path.join(output_dir, 'pred_range.png')
    plot_prediction_range(
        preds, trues, times,
        pred_len=pred_len,
        title=f'{title_prefix}Prediction Range (Min/Max/Mean)',
        target_names=target_names,
        save_path=range_path,
    )
    paths['pred_range'] = range_path

    return paths
