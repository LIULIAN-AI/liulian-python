"""Post-HPO checkpoint trimming utility.

Keeps only the best *N* trials and removes ``.pth`` files from the rest to
free disk space.  Adapted from the Swiss-River-Network-Benchmark project's
``trim_checkpoints`` routine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def trim_checkpoints(
    root_path: str | Path,
    keep_best_n: int = 10,
    anchor_metric: str = 'loss',
    mode: str = 'min',
    if_trim_best_n: bool = True,
    keep_best_for_trimmed_trials: bool = True,
    keep_last_for_trimmed_trials: bool = False,
) -> Tuple[int, float]:
    """Trim ``.pth`` checkpoint files from non-best Ray Tune trials.

    This can reclaim 80%+ of disk space after a large HPO sweep.

    Args:
        root_path: Directory containing the Ray Tune experiment
            (must contain ``experiment_state-*.json``).
        keep_best_n: Number of best trials whose checkpoints to preserve
            (the best and last checkpoints are always kept for these).
        anchor_metric: Metric used to rank trials.
        mode: ``"min"`` or ``"max"`` — direction for *anchor_metric*.
        if_trim_best_n: If ``True``, also trim intermediate checkpoints
            from the top-*N* trials (keeping only best + last).
        keep_best_for_trimmed_trials: Keep the single best checkpoint
            for trials outside the top *N*.
        keep_last_for_trimmed_trials: Keep the last checkpoint for trials
            outside the top *N*.

    Returns:
        ``(n_files_removed, mb_freed)`` — count and megabytes freed.
    """
    root_path = Path(root_path)
    if not root_path.is_dir():
        logger.warning('trim_checkpoints: %s is not a directory — skipping.', root_path)
        return 0, 0.0

    # If the directory itself is an experiment, process it.
    # Otherwise, recurse one level (multi-experiment layout).
    children = list(root_path.iterdir())
    has_state = any(
        c.is_file() and c.name.startswith('experiment_state-') and c.name.endswith('.json')
        for c in children
    )
    if not has_state:
        # Try one level deeper (e.g. storage_path/run_name/experiment_state-*.json)
        total_removed = 0
        total_freed = 0.0
        for child in children:
            if child.is_dir():
                nr, mf = trim_checkpoints(
                    child,
                    keep_best_n=keep_best_n,
                    anchor_metric=anchor_metric,
                    mode=mode,
                    if_trim_best_n=if_trim_best_n,
                    keep_best_for_trimmed_trials=keep_best_for_trimmed_trials,
                    keep_last_for_trimmed_trials=keep_last_for_trimmed_trials,
                )
                total_removed += nr
                total_freed += mf
        return total_removed, total_freed

    try:
        from ray.tune import ExperimentAnalysis
    except ImportError:
        logger.warning('ray[tune] not installed — cannot trim checkpoints.')
        return 0, 0.0

    try:
        analysis = ExperimentAnalysis(str(root_path))
    except Exception as exc:
        logger.warning('Could not load ExperimentAnalysis from %s: %s', root_path, exc)
        return 0, 0.0

    trials = list(analysis.trials)
    if len(trials) <= keep_best_n:
        logger.info(
            'Only %d trials (<= keep_best_n=%d) — no trimming needed.',
            len(trials),
            keep_best_n,
        )
        return 0, 0.0

    # Rank trials by anchor_metric
    import numpy as np

    metric_vals = []
    for t in trials:
        ma = t.metric_analysis.get(anchor_metric, {})
        metric_vals.append(ma.get(mode, float('inf') if mode == 'min' else float('-inf')))

    idx_sorted = np.argsort(metric_vals)
    if mode == 'max':
        idx_sorted = idx_sorted[::-1]

    idx_best_n = set(idx_sorted[:keep_best_n].tolist())

    total_removed = 0
    total_freed = 0.0
    files_trimmed: List[str] = []

    for idx, trial in enumerate(trials):
        is_best = idx in idx_best_n
        if is_best and not if_trim_best_n:
            continue  # Skip best trials entirely

        keep_best = True if is_best else keep_best_for_trimmed_trials
        keep_last = True if is_best else keep_last_for_trimmed_trials

        nr, freed, trimmed = _trim_one_trial(
            trial, analysis, root_path, anchor_metric, mode, keep_best, keep_last
        )
        total_removed += nr
        total_freed += freed
        files_trimmed.extend(trimmed)

    if files_trimmed:
        trimmed_log = root_path / 'trimmed_files.txt'
        with open(trimmed_log, 'a') as f:
            for fp in files_trimmed:
                f.write(f'{fp}\n')

    mb_freed = total_freed / (1024 ** 2)
    logger.info(
        'Trimmed %d checkpoint files, freed %.2f MB from %s',
        total_removed,
        mb_freed,
        root_path,
    )
    return total_removed, mb_freed


def _trim_one_trial(
    trial,
    analysis,
    root_path: Path,
    anchor_metric: str,
    mode: str,
    keep_best_checkpoint: bool,
    keep_last_checkpoint: bool,
) -> Tuple[int, float, List[str]]:
    """Remove ``.pth`` files from a single trial's checkpoints."""
    trial_path = Path(trial.path)
    if not trial_path.is_dir():
        return 0, 0.0, []

    # Identify checkpoint dirs to skip
    dirs_to_keep: set = set()
    try:
        if keep_best_checkpoint:
            best_ck = analysis.get_best_checkpoint(trial, metric=anchor_metric, mode=mode)
            if best_ck is not None:
                dirs_to_keep.add(Path(best_ck.path).name)
        if keep_last_checkpoint:
            last_ck = analysis.get_last_checkpoint(trial)
            if last_ck is not None:
                dirs_to_keep.add(Path(last_ck.path).name)
    except Exception:
        # If checkpoint tracking is unavailable, skip trimming this trial
        return 0, 0.0, []

    n_removed = 0
    freed = 0.0
    trimmed: List[str] = []

    for ck_dir in sorted(trial_path.iterdir()):
        if not ck_dir.is_dir() or ck_dir.name in dirs_to_keep:
            continue
        # Only consider dirs that look like checkpoints
        if not ck_dir.name.startswith('checkpoint_'):
            continue
        for child in ck_dir.iterdir():
            if child.is_file() and child.suffix == '.pth':
                size = child.stat().st_size
                child.unlink()
                freed += size
                n_removed += 1
                try:
                    rel = str(ck_dir.relative_to(root_path) / child.name)
                except ValueError:
                    rel = str(child)
                trimmed.append(rel)

    if n_removed > 0:
        logger.debug(
            'Trial %s: removed %d files, freed %.2f MB',
            trial.trial_id,
            n_removed,
            freed / (1024 ** 2),
        )

    return n_removed, freed, trimmed
