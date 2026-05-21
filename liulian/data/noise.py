"""Reusable noise injection utilities for time-series data.

Provides a unified API for adding synthetic noise to numpy arrays.
Used by both :mod:`liulian.data.ts.timeseriesdataset` and
:mod:`liulian.data.seq_dataset` to decouple noise logic from dataset
construction.

Supported noise types
---------------------
* ``'gaussian'`` / ``'gaussian_a'`` — Additive Gaussian noise.
* ``'impulse'`` / ``'impulse_a'`` — Random spike noise.
* ``'quantization'`` — Quantize values to discrete levels.

Adapted from:
- refer_projects/swiss-river-network-benchmark/swissrivernetwork/benchmark/dataset.py
  (``add_gaussian_noise``, ``add_impulse_noise``)
- liulian/data/seq_dataset.py (``add_noise_to_array``)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core noise generators
# ---------------------------------------------------------------------------


def add_gaussian_noise(
    data: np.ndarray,
    noise_level: float = 0.01,
    std: float | None = None,
) -> np.ndarray:
    """Add Gaussian noise to *data*.

    Parameters
    ----------
    data:
        Original signal (any shape).
    noise_level:
        Standard deviation of noise as a *fraction* of ``data.std()``.
        Ignored when *std* is given explicitly.
    std:
        Absolute standard deviation to use directly.

    Returns
    -------
    np.ndarray
        Noisy copy of *data*.
    """
    if std is None:
        sigma = float(np.std(data)) * noise_level
    else:
        sigma = std
    if sigma <= 1e-9:
        return data.copy()
    return data + np.random.normal(0, sigma, data.shape).astype(data.dtype)


def add_impulse_noise(
    data: np.ndarray,
    probability: float = 0.01,
    scale_factor: float = 5.0,
    magnitude: float | None = None,
) -> np.ndarray:
    """Add impulse (spike) noise to *data*.

    Parameters
    ----------
    data:
        Original signal.
    probability:
        Fraction of data points affected by spikes.
    scale_factor:
        Spike magnitude as a multiple of ``data.std()``.
        Ignored when *magnitude* is given.
    magnitude:
        Absolute spike magnitude.

    Returns
    -------
    np.ndarray
        Noisy copy of *data*.
    """
    if probability <= 1e-9:
        return data.copy()
    noisy = data.copy()
    n_samples = data.size
    n_spikes = max(1, int(n_samples * probability))
    if magnitude is None:
        magnitude = float(np.std(data)) * scale_factor
    if magnitude <= 1e-9:
        return noisy
    spike_idx = np.random.choice(n_samples, n_spikes, replace=False)
    flat = noisy.ravel()
    flat[spike_idx] += np.random.choice([-magnitude, magnitude], size=n_spikes).astype(flat.dtype)
    return noisy


def add_quantization_noise(
    data: np.ndarray,
    levels: int = 100,
) -> np.ndarray:
    """Quantize *data* to a fixed number of discrete levels.

    Parameters
    ----------
    data:
        Original signal.
    levels:
        Number of quantization bins.

    Returns
    -------
    np.ndarray
        Quantized copy of *data*.
    """
    mn, mx = float(data.min()), float(data.max())
    if mx - mn < 1e-8:
        return data.copy()
    step = (mx - mn) / levels
    return (np.round((data - mn) / step) * step + mn).astype(data.dtype)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

# Alias map so callers from the reference project (``gaussian_a``,
# ``impulse_a``) work transparently alongside the canonical names.
_ALIAS_MAP = {
    'gaussian_a': 'gaussian',
    'impulse_a': 'impulse',
}


def add_noise_to_array(
    arr: np.ndarray,
    noise_type: str,
    noise_kwargs: dict | None = None,
) -> np.ndarray:
    """Apply noise to *arr* (any shape).

    Parameters
    ----------
    arr:
        Input array.
    noise_type:
        Noise family — ``'gaussian'``, ``'gaussian_a'``, ``'impulse'``,
        ``'impulse_a'``, or ``'quantization'``.
    noise_kwargs:
        Extra keyword arguments forwarded to the noise generator.

    Returns
    -------
    np.ndarray
        Noisy array (same shape and dtype as *arr*).
    """
    if noise_kwargs is None:
        noise_kwargs = {}

    canonical = _ALIAS_MAP.get(noise_type, noise_type)

    if canonical == 'gaussian':
        return add_gaussian_noise(arr, **noise_kwargs)

    if canonical == 'impulse':
        return add_impulse_noise(arr, **noise_kwargs)

    if canonical == 'quantization':
        return add_quantization_noise(arr, **noise_kwargs)

    raise ValueError(f'Unknown noise_type: {noise_type!r}')


def add_noise_to_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    noise_type: str,
    noise_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Apply noise **in-place** to selected *columns* of *df*.

    Parameters
    ----------
    df:
        DataFrame to modify.
    columns:
        Column names to inject noise into.
    noise_type:
        See :func:`add_noise_to_array`.
    noise_kwargs:
        Forwarded to :func:`add_noise_to_array`.

    Returns
    -------
    pd.DataFrame
        The same *df* (modified in place) for chaining convenience.
    """
    for col in columns:
        df[col] = add_noise_to_array(df[col].values, noise_type, noise_kwargs)
    return df
