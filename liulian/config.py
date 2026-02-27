"""Experiment configuration — defaults, YAML loading, CLI override merge.

The configuration system follows a three-layer priority model::

    CLI args  >  YAML file  >  DEFAULT_CONFIG

Usage::

    from liulian.config import load_config

    # From YAML only
    cfg = load_config('experiments/swiss_river/default_config.yaml')

    # YAML + CLI overrides
    cfg = load_config(
        'config.yaml', cli_overrides={'model': 'lstm', 'quick_test': True}
    )

    # Defaults only (no YAML)
    cfg = load_config()
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Project paths ───────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')


# ── Default configuration ──────────────────────────────────────────────
# Every key used by run.py / pipeline.py must appear here with its
# default value.  This is the single source of truth for defaults.

DEFAULT_CONFIG: Dict[str, Any] = {
    # ── General ─────────────────────────────────────────────────────
    'model': 'lstm',
    'seed': 2026,
    'quick_test': False,
    'eval_only': False,
    # ── Data ────────────────────────────────────────────────────────
    'data': 'swiss-river-1990',
    'seq_len': 90,
    'pred_len': 7,
    'features': 'M',
    'label_len': 0,
    'train_split': 0.8,
    'max_samples': None,
    # ── Task & mode ─────────────────────────────────────────────────
    'task': 'forecast',
    'split_mode': 'ts',
    'scaler': 'minmax',
    'use_current_x': True,
    'use_full_history': False,
    # ── Gap handling ────────────────────────────────────────────────
    'short_subsequence_method': 'drop',
    'gap_mode': 'split',
    'max_mask_consecutive': 10,
    # ── Noise injection ─────────────────────────────────────────────
    'noise_type': None,
    'noise_level': 0.01,
    'noise_probability': 0.01,
    'noise_scale_factor': 5.0,
    # ── Historical target ───────────────────────────────────────────
    'include_historical_y': 'none',
    'include_historical_predicted_y': False,
    # ── Entity identifiers ──────────────────────────────────────────
    'identifier_mode': 'embedding',
    'id_integration': 'concat_to_x',
    'embedding_size': 10,
    'num_embeddings': None,
    # ── Graph / spatial ─────────────────────────────────────────────
    'graph_mode': 'none',
    'graphlet_num_hops': 1,
    # ── Model architecture ──────────────────────────────────────────
    'enc_in': None,
    'dec_in': 1,
    'c_out': 1,
    'd_model': 64,
    'd_ff': 32,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'dropout': 0.1,
    'patch_len': 16,
    'stride': 8,
    # ── LLM backbone (TimeLLM) ─────────────────────────────────────
    'llm_model': 'GPT2',
    'llm_dim': 768,
    'llm_layers': 6,
    'prompt_domain': 0,
    'cache_dir': os.path.join(PROJECT_ROOT, 'cache'),
    # ── Training ────────────────────────────────────────────────────
    'train_epochs': 30,
    'batch_size': 8,
    'learning_rate': 0.001,
    'loss': 'mse',
    'metrics': 'rmse,mae,nse',
    'eval_denorm': True,
    'show_progress': True,
    'patience': 10,
    'lradj': 'none',
    'pct_start': 0.2,
    'cos_T_max': 20,
    'cos_eta_min': 1e-8,
    'cos_T_0': 10,
    'cos_T_mult': 2,
    'step_size': 10,
    'gamma': 0.5,
    'milestones': '30,60,90',
    'sched_patience': 5,
    'sched_factor': 0.5,
    'num_workers': 0,
    # ── Logging ─────────────────────────────────────────────────────
    'wandb_project': None,
    'wandb_entity': None,
    'dev_run': False,
    # ── HPO (Ray Tune) ──────────────────────────────────────────────
    'hpo': True,
    'hpo_num_samples': 200,
    'hpo_scheduler': 'asha',
    'hpo_grace_period': 5,
    'hpo_reduction_factor': 1.5,
    'hpo_storage_path': None,
    'hpo_resources_cpu': 1,
    'hpo_resources_gpu': 0.25,
    'hpo_num_cpus': None,
    'hpo_max_concurrent': None,
    'hpo_resume': False,
    'hpo_save_checkpoints': True,
    'hpo_trim_checkpoints': True,
    'hpo_keep_best_n': 10,
    'hpo_trim_best_n': True,
    'hpo_trim_keep_best': True,
    'hpo_trim_keep_last': False,
    'hpo_experiment_name': None,
    'hpo_local_mode': False,
    # ── Visualisation ───────────────────────────────────────────────
    'auto_viz': True,
    'viz_method': 'mean',
    # ── Misc model keys (Time-LLM compat) ──────────────────────────
    'embed': 'timeF',
    'activation': 'gelu',
    'output_attention': False,
    'moving_avg': 25,
    'factor': 1,
    'task_name': 'long_term_forecast',
}


# ── Model-specific default overrides ───────────────────────────────────

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'lstm': {
        'dropout': 0.0,
        'batch_size': 256,
        'lradj': 'none',
    },
}

# ── Quick-test overrides ───────────────────────────────────────────────

QUICK_TEST_OVERRIDES: Dict[str, Any] = {
    'train_epochs': 2,
    'batch_size': 4,
    'patience': 2,
    'max_train_iters': 100,
    'max_eval_iters': 50,
    'num_workers': 0,
    'identifier_mode': 'embedding',
    'hpo_num_samples': 2,
    'hpo_local_mode': True,
}


# ── Public API ─────────────────────────────────────────────────────────


def load_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a merged configuration dict.

    Priority: *cli_overrides* > *yaml_path* > :data:`DEFAULT_CONFIG`.

    Args:
        yaml_path: Path to a YAML config file.  ``None`` → use defaults.
        cli_overrides: Dict of key-value pairs from CLI arguments.

    Returns:
        Merged configuration dictionary.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # Layer 2: YAML file
    if yaml_path is not None:
        yaml_cfg = _load_yaml(yaml_path)
        cfg.update(yaml_cfg)

    # Layer 3: CLI overrides (highest priority)
    if cli_overrides:
        # Filter out None values so that un-set CLI flags don't clobber YAML
        for k, v in cli_overrides.items():
            if v is not None:
                cfg[k] = v

    return cfg


def apply_quick_test(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply quick-test overrides in-place and return the config."""
    config.update(QUICK_TEST_OVERRIDES)
    logger.info(
        'Quick test: %d epochs, %d iter/epoch, batch_size=%d',
        config['train_epochs'],
        config.get('max_train_iters', '∞'),
        config['batch_size'],
    )
    return config


def apply_model_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply model-specific defaults (only for keys not already overridden).

    Model defaults have lower priority than user-set values.
    """
    model_name = config.get('model', '')
    defaults = MODEL_DEFAULTS.get(model_name, {})
    for k, v in defaults.items():
        if k not in config or config[k] == DEFAULT_CONFIG.get(k):
            config[k] = v
    return config


# ── Internal ───────────────────────────────────────────────────────────


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    logger.info('Loaded config from %s (%d keys)', path, len(data))
    return data
