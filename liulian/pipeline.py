"""Pipeline — reusable building blocks for the experiment lifecycle.

This module extracts all orchestration logic that used to live inside
``experiments/swiss_river/run.py`` into framework-level, dataset-agnostic
functions.  Any experiment script (or the CLI) can build a full pipeline
in a handful of calls::

    from liulian.pipeline import run_experiment
    from liulian.config import load_config

    cfg = load_config('experiments/swiss_river/default_config.yaml')
    summary = run_experiment(cfg)

Individual building blocks are also exposed for fine-grained control::

    dataset = build_dataset(cfg)
    model = build_model(cfg, dataset)
    loaders = build_loaders(dataset, cfg)
    exp = build_experiment(cfg, dataset, model, loaders)
    summary = exp.run(train=True)
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np

import liulian.utils.log_tags  # noqa: F401  — registers logger.ok / logger.hint
from liulian.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility (random, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info('Random seed set to %d', seed)


def parse_metric_names(value: str | list[str]) -> list[str]:
    """Normalize metric configuration into a list of lowercase names.

    Accepts a comma-separated string (``"rmse,mae,nse"``) or a list.
    Falls back to ``['rmse', 'mae', 'nse']`` when empty.
    """
    if isinstance(value, str):
        metrics = [token.strip().lower() for token in value.split(',') if token.strip()]
    else:
        metrics = [str(token).strip().lower() for token in value if str(token).strip()]
    return metrics or ['rmse', 'mae', 'nse']


def build_hpo_experiment_name(config: Dict[str, Any]) -> str:
    """Build a Ray Tune experiment name following the naming convention.

    Pattern: ``{data}_{model}_{task}_{split_mode}[_{extra}]_{timestamp}``
    """
    import datetime

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    parts = [
        config.get('data', 'data'),
        config.get('model', 'model'),
        config.get('task', 'forecast'),
        config.get('split_mode', 'per_entity'),
    ]
    extra = config.get('hpo_experiment_name')
    if extra:
        parts.append(extra)
    parts.append(ts)
    return '_'.join(parts)


# ── Noise ───────────────────────────────────────────────────────────────


def build_noise_kwargs(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pack noise-related config keys into a kwargs dict for the dataset.

    Returns ``None`` when no noise is configured.
    """
    if config.get('noise_type') is None:
        return None
    return {
        'noise_level': config.get('noise_level', 0.01),
        'probability': config.get('noise_probability', 0.01),
        'scale_factor': config.get('noise_scale_factor', 5.0),
    }


# ── Dataset ─────────────────────────────────────────────────────────────


def build_dataset(config: Dict[str, Any]) -> Any:
    """Construct a dataset from the config.

    Supports ``SwissRiverDataset`` (for ``swiss-river-*`` data names).
    Extend via the ``elif`` ladder for other dataset families.

    Returns:
        A dataset object with ``get_split()`` and ``get_data_loaders()``.
    """
    noise_kwargs = build_noise_kwargs(config)
    data_name = config['data']

    if data_name.startswith('swiss-river'):
        from liulian.data.swiss_river import SwissRiverDataset

        dataset = SwissRiverDataset(
            data_name=data_name,
            split_mode=config['split_mode'],
            seq_len=config['seq_len'],
            pred_len=config['pred_len'],
            task=config['task'],
            train_split=config['train_split'],
            scaler_type=config.get('scaler', 'none'),
            use_current_x=config['use_current_x'],
            use_full_history=config['use_full_history'],
            short_subsequence_method=config['short_subsequence_method'],
            gap_mode=config['gap_mode'],
            max_mask_consecutive=config['max_mask_consecutive'],
            noise_type=config['noise_type'],
            noise_kwargs=noise_kwargs,
            include_historical_y=config['include_historical_y'],
            include_historical_predicted_y=config['include_historical_predicted_y'],
            identifier_mode=config['identifier_mode'],
            id_integration=config['id_integration'],
            graph_mode=config['graph_mode'],
            graphlet_num_hops=config['graphlet_num_hops'],
            max_samples=config.get('max_samples'),
        )
    else:
        raise ValueError(
            f'Unknown dataset: {data_name!r}. '
            f"Supported prefixes: 'swiss-river-*'. "
            f'Add new datasets by extending build_dataset().'
        )
    return dataset


def print_dataset_summary(dataset: Any) -> None:
    """Print a compact summary of the dataset splits and configuration."""
    info = dataset.info()
    logger.info(
        'Dataset: %s  (mode=%s, graph=%s)',
        info.get('data_name'),
        info.get('split_mode'),
        info.get('graph_name', 'none'),
    )
    logger.info(
        '  stations=%d, task=%s, seq_len=%d, pred_len=%d',
        info.get('num_stations', 0),
        info.get('task', ''),
        info.get('seq_len', 0),
        info.get('pred_len', 0),
    )
    logger.info(
        '  noise=%s, identifier=%s [%s], gap=%s',
        info.get('noise_type', 'none'),
        info.get('identifier_mode', 'none'),
        info.get('id_integration', ''),
        info.get('gap_mode', 'split'),
    )
    for name in ('train', 'val', 'test'):
        split = dataset.get_split(name)
        logger.info(
            '  split %-5s  samples=%d  feat=%d  targ=%d',
            name,
            len(split),
            split.feat_dim,
            split.targ_dim,
        )


def auto_detect_enc_in(dataset: Any) -> int:
    """Derive ``enc_in`` from the training split's feature dimension."""
    split = dataset.get_split('train')
    return split.feat_dim if split.feat_dim > 0 else 1


# ── Model ───────────────────────────────────────────────────────────────


def _load_prompt_content(config: Dict[str, Any]) -> str:
    """Load prompt text file for LLM-based models (e.g. TimeLLM).

    Falls back to a generic description when no file is found.
    """
    prompt_map = {
        'swiss-river-1990': 'wt-swiss-1990',
        'swiss-river-2010': 'wt-swiss-2010',
        'swiss-river-zurich': 'wt-zurich',
    }
    fname = prompt_map.get(config.get('data', ''), config.get('data', ''))
    prompt_path = os.path.join(PROJECT_ROOT, 'dataset', 'prompt_bank', f'{fname}.txt')
    if os.path.exists(prompt_path):
        with open(prompt_path) as fh:
            return fh.read()
    return (
        'Time series dataset for forecasting. '
        'Data includes monitoring stations with periodic observations.'
    )


def build_model(config: Dict[str, Any], dataset: Any = None) -> Any:
    """Instantiate a forecasting model by name and wrap if needed.

    All models are loaded via dynamic import from
    ``liulian.models.torch.<model_name>``.  Special pre-processing
    (e.g. prompt loading for TimeLLM) is handled transparently.

        Handles:
        - Model instantiation via ``liulian.models.torch.<name>.Model``
        - EntityWrapper wrapping when ``identifier_mode='embedding'``
            (uses ChannelEntityWrapper for ``split_mode='multi_channel'``)
        - PatchTST internal patch-level entity integration when
            ``identifier_mode='embedding'`` and ``id_integration='add_after_patch'``
        - Auto enc_in detection from dataset

    Args:
        config: Full experiment config dict.
        dataset: Dataset (used for auto enc_in and station count).

    Returns:
        A PyTorch ``nn.Module`` (potentially wrapped in EntityWrapper).
    """
    import importlib
    from types import SimpleNamespace

    model_name = config['model']

    # Auto-detect enc_in
    if config.get('enc_in') is None and dataset is not None:
        config['enc_in'] = auto_detect_enc_in(dataset)
        logger.info('Auto-detected enc_in=%d from training data', config['enc_in'])

    ns = SimpleNamespace(**config)

    # Pre-processing for LLM-based models
    if model_name == 'timellm':
        ns.content = _load_prompt_content(config)

    if (
        config.get('id_integration') == 'add_after_patch'
        and not (model_name == 'patchtst' and config.get('identifier_mode') == 'embedding')
    ):
        raise ValueError(
            "id_integration='add_after_patch' is only supported for model='patchtst' with identifier_mode='embedding'."
        )

    # Dynamic import for all models
    _module_name = model_name

    try:
        mod = importlib.import_module(f'liulian.models.torch.{_module_name}')
        model = mod.Model(ns).float()
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise ValueError(
            f'Unknown model: {model_name!r}. Available: lstm, dlinear, timellm, '
            f'or any module under liulian.models.torch.*.'
        ) from exc

    # Wrap with entity embedding when configured.
    # PatchTST + add_after_patch is handled internally by the model.
    if (
        config.get('identifier_mode') == 'embedding'
        and config.get('id_integration') != 'add_after_patch'
    ):
        num_emb = config.get('num_embeddings')
        if num_emb is None and dataset is not None:
            num_emb = len(dataset.station_ids)
        if num_emb is None:
            raise ValueError('num_embeddings must be specified when identifier_mode=embedding.')
        config['num_embeddings'] = num_emb

        emb_size = config.get('embedding_size', 10)

        if config.get('split_mode') == 'multi_channel':
            # Multi-channel mode: per-channel entity embedding
            from liulian.models.torch.entity_mixin import ChannelEntityWrapper

            model = ChannelEntityWrapper(
                inner_model=model,
                num_stations=num_emb,
                embedding_size=emb_size,
            )
            logger.info(
                'ChannelEntityWrapper: num_stations=%d, embedding_size=%d',
                num_emb,
                emb_size,
            )
        else:
            # Per-entity mode: per-sample entity embedding
            from liulian.models.torch.entity_mixin import EntityWrapper

            model = EntityWrapper(
                inner_model=model,
                enc_in=config['enc_in'],
                num_embeddings=num_emb,
                embedding_size=emb_size,
            )
            logger.info(
                'EntityWrapper: num_embeddings=%d, embedding_size=%d',
                num_emb,
                emb_size,
            )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info('Model: %s  (%.1fK params)', model_name, n_params / 1e3)

    return model


# ── Data loaders ────────────────────────────────────────────────────────


def build_loaders(dataset: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create data loaders from the dataset.

    Delegates to ``dataset.get_data_loaders()`` which is the standard
    interface for liulian datasets.
    """
    return dataset.get_data_loaders(
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 0),
    )


# ── HPO optimizer ───────────────────────────────────────────────────────


def build_optimizer(config: Dict[str, Any]) -> Optional[Any]:
    """Build a RayOptimizer from HPO config keys.

    Returns ``None`` if HPO is disabled or Ray is not installed.
    """
    if not config.get('hpo', False):
        return None

    try:
        import ray  # noqa: F401
        from liulian.optim.ray_optimizer import RayOptimizer
    except ImportError:
        logger.info('ray[tune] not installed — HPO disabled.')
        return None

    optimizer = RayOptimizer(
        config={
            'num_samples': config.get('hpo_num_samples', 200),
            'max_epochs': config.get('train_epochs', 30),
            'metric': 'loss',
            'mode': 'min',
            'scheduler': config.get('hpo_scheduler', 'asha'),
            'grace_period': config.get('hpo_grace_period', 5),
            'reduction_factor': config.get('hpo_reduction_factor', 1.5),
            'storage_path': config.get('hpo_storage_path'),
            'resources_per_trial': {
                'cpu': config.get('hpo_resources_cpu', 1),
                'gpu': config.get('hpo_resources_gpu', 0),
            },
            'num_cpus': config.get('hpo_num_cpus'),
            'max_concurrent_trials': config.get('hpo_max_concurrent'),
            'resume': config.get('hpo_resume', False),
            'save_checkpoints': config.get('hpo_save_checkpoints', True),
            'trim_checkpoints': config.get('hpo_trim_checkpoints', True),
            'keep_best_n': config.get('hpo_keep_best_n', 10),
            'trim_best_n': config.get('hpo_trim_best_n', True),
            'trim_keep_best': config.get('hpo_trim_keep_best', True),
            'trim_keep_last': config.get('hpo_trim_keep_last', False),
            'experiment_name': build_hpo_experiment_name(config),
            'local_mode': config.get('hpo_local_mode', False),
            # Pass through for experiment_name building
            'data': config.get('data'),
            'model': config.get('model'),
            'task': config.get('task'),
            'mode_tag': config.get('split_mode'),
        },
    )

    # Default search space — resolved from search_spaces.py registry
    if 'search_space' not in config:
        from liulian.optim.search_spaces import resolve_search_space

        config['search_space'] = resolve_search_space(
            model=config.get('model', ''),
            data=config.get('data', ''),
            identifier_mode=config.get('identifier_mode', 'none'),
            id_integration=config.get('id_integration', 'concat_to_x'),
        )
        logger.info(
            'Resolved search space for model=%s, data=%s: %s',
            config.get('model'),
            config.get('data'),
            list(config['search_space'].keys()),
        )

    logger.info(
        'HPO enabled: %d samples, grace=%s, reduction=%s, storage=%s',
        config.get('hpo_num_samples', 200),
        config.get('hpo_grace_period', 5),
        config.get('hpo_reduction_factor', 1.5),
        config.get('hpo_storage_path', 'artifacts/ray_results'),
    )
    return optimizer


# ── Logger ──────────────────────────────────────────────────────────────


def build_logger(config: Dict[str, Any]) -> Optional[Any]:
    """Build experiment logger (wandb or None).

    Returns ``None`` if wandb is not configured or dev_run is set.
    The :class:`Experiment` will auto-create a local logger as fallback.
    """
    if not config.get('wandb_project') or config.get('dev_run', False):
        return None
    try:
        from liulian.loggers.wandb_logger import WandbLogger

        exp_logger = WandbLogger(
            project=config['wandb_project'],
            entity=config.get('wandb_entity'),
            config=config,
        )
        logger.info('wandb logging enabled → project=%s', config['wandb_project'])
        return exp_logger
    except Exception as exc:
        logger.warning('wandb init failed (%s); using local logging.', exc)
        return None


# ── Experiment assembly ─────────────────────────────────────────────────


def build_experiment(
    config: Dict[str, Any],
    dataset: Any = None,
    model: Any = None,
    loaders: Optional[Dict[str, Any]] = None,
) -> Any:
    """Assemble a full :class:`Experiment` from config.

    If *dataset*, *model*, or *loaders* are not provided, they are built
    automatically from *config*.

    Returns:
        An :class:`Experiment` instance ready to ``run()``.
    """
    from liulian.runtime import Experiment, ExperimentSpec
    from liulian.tasks.base import PredictionRegime, PredictionTask

    # Build components if not provided
    if dataset is None:
        dataset = build_dataset(config)
        print_dataset_summary(dataset)

    if model is None:
        model = build_model(config, dataset)

    if loaders is None:
        loaders = build_loaders(dataset, config)

    # Task
    task = PredictionTask(
        regime=PredictionRegime(
            horizon=config['pred_len'],
            context_length=config['seq_len'],
        ),
        loss_name=config.get('loss', 'mse'),
        metrics=parse_metric_names(config.get('metrics', 'rmse,mae,nse')),
    )

    # Spec
    model_name = config.get('model', 'model')
    spec = ExperimentSpec(
        name=f'{config.get("data", "data")}_{model_name}'
        f'_{config.get("task", "forecast")}_{config.get("split_mode", "per_entity")}',
        task={
            'type': 'PredictionTask',
            'pred_len': config['pred_len'],
            'task': config.get('task', 'forecast'),
            'loss': config.get('loss', 'mse'),
            'metrics': parse_metric_names(config.get('metrics', 'rmse,mae,nse')),
        },
        dataset={
            'type': type(dataset).__name__,
            'data': config.get('data'),
            'split_mode': config.get('split_mode'),
            'graph_mode': config.get('graph_mode'),
            'identifier_mode': config.get('identifier_mode'),
            'noise_type': config.get('noise_type'),
        },
        model={
            'type': model_name,
            'd_model': config.get('d_model'),
            'enc_in': config.get('enc_in'),
        },
        metadata={'seed': config.get('seed')},
    )

    # Optimizer
    optimizer = build_optimizer(config)

    # Logger
    exp_logger = build_logger(config)

    # Viz config
    config['auto_viz'] = config.get('auto_viz', True)
    config['viz_method'] = config.get('viz_method', 'mean')

    return Experiment(
        spec=spec,
        task=task,
        dataset=dataset,
        model=None,
        torch_model=model,
        optimizer=optimizer,
        exp_logger=exp_logger,
        data_loaders=loaders,
        config=config,
    )


# ── Full pipeline runner ────────────────────────────────────────────────


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full experiment pipeline: seed → build → run → viz → report.

    This is the single function that replaces the entire ``main()`` in
    ``experiments/swiss_river/run.py``.  It produces identical behaviour
    and all the same intermediate / final outputs.

    Args:
        config: Merged config dict (from :func:`load_config`).

    Returns:
        Experiment summary dictionary.
    """
    from liulian.config import apply_quick_test

    t0 = time.time()

    # ── Seed ────────────────────────────────────────────────────────
    seed_everything(config.get('seed', 2026))

    # ── Quick-test overrides ────────────────────────────────────────
    if config.get('quick_test', False):
        apply_quick_test(config)

    # ── Build ───────────────────────────────────────────────────────
    dataset = build_dataset(config)
    print_dataset_summary(dataset)
    model = build_model(config, dataset)
    loaders = build_loaders(dataset, config)
    exp = build_experiment(config, dataset, model, loaders)

    # ── Run ─────────────────────────────────────────────────────────
    summary = exp.run(train=not config.get('eval_only', False))

    elapsed = time.time() - t0

    # ── Post-run visualisation ──────────────────────────────────────
    artifacts_dir = summary.get('artifacts_dir', 'artifacts')
    pred_result = summary.get('predictions')
    viz_method = config.get('viz_method', 'mean')

    if pred_result is not None and 'viz_paths' not in summary:
        try:
            from liulian.viz.plots import save_prediction_plots

            viz_dir = os.path.join(artifacts_dir, 'figures')

            # Derive target names from dataset when available
            target_names = None
            if dataset is not None:
                try:
                    info = dataset.info()
                    target_names = info.get('target_names')
                except Exception:
                    pass

            plot_paths = save_prediction_plots(
                preds=pred_result['preds'],
                trues=pred_result['trues'],
                times=pred_result['times'],
                method=viz_method,
                output_dir=viz_dir,
                title_prefix=f'{config["data"]} / {config["model"]} — ',
                target_names=target_names,
            )
            logger.ok('Saved prediction plots:')
            for name, path in plot_paths.items():
                logger.info('  %s → %s', name, path)
        except Exception as exc:
            logger.warning('Visualisation failed: %s', exc)
    elif 'viz_paths' in summary:
        logger.info('Auto-viz plots already generated by Experiment.')

    # Save raw predictions as .npz
    if pred_result is not None:
        npz_path = os.path.join(artifacts_dir, 'predictions.npz')
        np.savez_compressed(
            npz_path,
            preds=pred_result['preds'].numpy(),
            trues=pred_result['trues'].numpy(),
            times=pred_result['times'].numpy(),
        )
        logger.ok('Raw predictions saved → %s', npz_path)

    # ── Console report ──────────────────────────────────────────────
    print_report(config, summary, elapsed)

    # ── Cleanup logger ──────────────────────────────────────────────
    exp_logger = getattr(exp, 'exp_logger', None)
    if exp_logger is not None:
        try:
            exp_logger.finish()
        except Exception:
            pass

    return summary


# ── Report ──────────────────────────────────────────────────────────────


def print_report(
    config: Dict[str, Any],
    summary: Dict[str, Any],
    elapsed: float,
) -> None:
    """Print a human-readable summary to stdout (matches run.py output)."""
    spec_name = (
        f'{config.get("data", "data")}_{config.get("model", "model")}'
        f'_{config.get("task", "forecast")}_{config.get("split_mode", "per_entity")}'
    )
    print(f'\n{"=" * 60}')
    print(f'Experiment : {spec_name}')
    print(f'Data       : {config["data"]}  (split_mode={config["split_mode"]})')
    print(
        f'Task       : {config["task"]}  '
        f'(seq={config["seq_len"]}, pred={config["pred_len"]}, '
        f'full_hist={config["use_full_history"]})'
    )
    print(f'Graph      : {config["graph_mode"]}')
    print(f'Noise      : {config.get("noise_type") or "none"}')
    print(f'Identifiers: {config["identifier_mode"]} [{config["id_integration"]}]')
    print(
        f'Gaps       : {config["gap_mode"]} '
        f'(short_subseq={config["short_subsequence_method"]})'
    )
    print(f'Hist. y    : {config["include_historical_y"]}')
    print(f'Elapsed    : {elapsed:.1f}s')
    print(f'Artifacts  : {summary.get("artifacts_dir", "N/A")}')
    for key, val in summary.get('metrics', {}).items():
        if key == 'history':
            continue
        print(f'  {key}: {val}')
    if 'hpo' in summary.get('metrics', {}):
        hpo = summary['metrics']['hpo']
        print(f'  HPO best: {hpo["best_value"]:.6f}  trials: {hpo["n_trials"]}')
    print(f'{"=" * 60}')
