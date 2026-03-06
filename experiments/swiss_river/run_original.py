#!/usr/bin/env python
"""Swiss River forecasting experiment — liulian framework.

This is the **recommended** way to run Swiss River experiments.
All training, evaluation, checkpointing, and logging are handled by the
liulian framework — the experiment script only sets up configuration and
calls framework APIs.

Usage::

    # ----------------------------------------------------------------
    # Basic runs
    # ----------------------------------------------------------------

    # LSTM quick test  (~2 s)
    python experiments/swiss_river/run.py --model lstm --quick_test

    # LSTM full training
    python experiments/swiss_river/run.py --model lstm --train_epochs 30

    # TimeLLM (slow, needs GPU)
    python experiments/swiss_river/run.py --model timellm

    # Evaluate from checkpoint (no training)
    python experiments/swiss_river/run.py --model lstm --eval_only

    # ----------------------------------------------------------------
    # Nowcasting  (predict y_n at t_n given x_1…x_n, y_1…y_{n-1})
    # ----------------------------------------------------------------

    python experiments/swiss_river/run.py --model lstm --task nowcast

    # ----------------------------------------------------------------
    # Full-history LSTM  (no sliding window — entire segment as input)
    # ----------------------------------------------------------------

    python experiments/swiss_river/run.py --model lstm --use_full_history

    # ----------------------------------------------------------------
    # Noise injection
    # ----------------------------------------------------------------

    python experiments/swiss_river/run.py --model lstm --noise_type gaussian_a --noise_level 0.05

    # ----------------------------------------------------------------
    # Entity identifiers
    # ----------------------------------------------------------------

    python experiments/swiss_river/run.py --model lstm --identifier_mode onehot --id_integration concat_to_x
    python experiments/swiss_river/run.py --model lstm --identifier_mode sinusoidal

    # ----------------------------------------------------------------
    # Gap handling
    # ----------------------------------------------------------------

    python experiments/swiss_river/run.py --model lstm --gap_mode mask_pad --max_mask_consecutive 10

    # ----------------------------------------------------------------
    # Spatial-temporal mode with graph info
    # ----------------------------------------------------------------

    python experiments/swiss_river/run.py --model lstm --split_mode st --graph_mode edge_index
    python experiments/swiss_river/run.py --model lstm --split_mode ts --graph_mode graphlet_features

    # ----------------------------------------------------------------
    # Historical target inclusion
    # ----------------------------------------------------------------

    python experiments/swiss_river/run.py --model lstm --include_historical_y gt
    python experiments/swiss_river/run.py --model lstm --include_historical_predicted_y

    # ----------------------------------------------------------------
    # Zurich sub-graph
    # ----------------------------------------------------------------

    python experiments/swiss_river/run.py --model lstm --data swiss-river-zurich --quick_test

See ``experiments/swiss_river/run_experiment.py`` for a lower-level script
that follows the Time-LLM ``run_main.py`` loop directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import ray
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_DATASET_ROOT = os.path.join(_PROJECT_ROOT, 'dataset')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _DATASET_ROOT not in sys.path:
    sys.path.insert(0, _DATASET_ROOT)

# ---------------------------------------------------------------------------
# liulian imports
# ---------------------------------------------------------------------------
from liulian.data.swiss_river import SwissRiverDataset
from liulian.runtime import Experiment, ExperimentSpec
from liulian.tasks.base import PredictionRegime, PredictionTask

from liulian.utils.log_tags import setup_logging as _setup_logging

_setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_metric_names(value: str | list[str]) -> list[str]:
    """Normalize metric configuration from CLI into a list."""
    if isinstance(value, str):
        metrics = [token.strip().lower() for token in value.split(',') if token.strip()]
    else:
        metrics = [str(token).strip().lower() for token in value if str(token).strip()]
    return metrics or ['rmse', 'mae', 'nse']


def _build_hpo_experiment_name(config: dict, model_name: str) -> str:
    """Build a Ray Tune experiment name following the singleton pattern.

    Pattern: ``{data}_{model}_{task}_{split_mode}[_{extra}]_{timestamp}``

    If the user set ``hpo_experiment_name`` it is inserted as an extra
    distinguisher before the timestamp (not used as the full name).
    """
    import datetime

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    parts = [
        choices=['concat_to_x', 'add_to_x', 'add_after_patch'],
        model_name,
        config.get('task', 'forecast'),
        config.get('split_mode', 'ts'),
    ]
    extra = config.get('hpo_experiment_name')
    if extra:
        parts.append(extra)
    parts.append(ts)
    return '_'.join(parts)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(name: str, cfg: dict) -> torch.nn.Module:
    """Instantiate a forecasting model by name.

    Args:
        name: ``"lstm"`` or ``"timellm"``.
        cfg: Flat config dict (converted to namespace internally).

    Returns:
        A PyTorch ``nn.Module``.
    """
    from types import SimpleNamespace

    ns = SimpleNamespace(**cfg)

    if name == 'lstm':
        from liulian.models.torch.lstm import Model

        return Model(ns).float()  # todo: should be float (same as timellm)?

    if name == 'timellm':
        from liulian.models.torch.timellm import Model

        # Load prompt text for the dataset
        prompt_map = {
            'swiss-river-1990': 'wt-swiss-1990',
            'swiss-river-2010': 'wt-swiss-2010',
            'swiss-river-zurich': 'wt-zurich',
        }
        fname = prompt_map.get(cfg.get('data', ''), cfg.get('data', ''))
        prompt_path = os.path.join(
            _PROJECT_ROOT, 'dataset', 'prompt_bank', f'{fname}.txt'
        )
        if os.path.exists(prompt_path):
            with open(prompt_path) as fh:
                ns.content = fh.read()
        else:
            ns.content = (
                'Swiss River Network water temperature dataset. '
                'Daily water and air temperature from monitoring stations.'
            )
        return Model(ns).float()  # todo: float is from the original code.

    raise ValueError(f"Unknown model: {name!r}. Use 'lstm' or 'timellm'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Swiss River experiment (liulian)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model', default='lstm', choices=['lstm', 'timellm'])
    p.add_argument(
        '--quick_test',
        action='store_true',
        help='Quick smoke-test: 2 epochs, tiny batch, capped iters.',
    )
    p.add_argument(
        '--eval_only',
        action='store_true',
        help='Evaluate from checkpoint without training.',
    )
    p.add_argument('--seed', type=int, default=2026)

    # ----- Data ----------------------------------------------------------
    g = p.add_argument_group('Data')
    g.add_argument(
        '--data',
        default='swiss-river-1990',
        choices=['swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich'],
        help='Dataset variant.',
    )
    g.add_argument(
        '--seq_len',
        type=int,
        default=90,
        help='Look-back window size (0 → full history).',
    )
    g.add_argument(
        '--pred_len', type=int, default=7, help='Forecast horizon (num future steps).'
    )
    g.add_argument(
        '--features',
        default='M',
        choices=['M', 'S', 'MS'],  # is this useful?
        help='Feature selection method. M=multivariate in/out, S=univariate in/out, '
        'MS=multivariate in with single target.',
    )
    g.add_argument('--label_len', type=int, default=0)
    g.add_argument(
        '--train_split',
        type=float,
        default=0.8,
        help='Fraction of train CSV used for training (rest → val).',
    )
    g.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Cap per-split samples (for quick dev iterations).',
    )

    # ----- Task / mode ---------------------------------------------------
    g = p.add_argument_group('Task & mode')
    g.add_argument(
        '--task',
        default='forecast',
        choices=['forecast', 'nowcast'],
        help='Prediction task type.',
    )
    g.add_argument(
        '--split_mode',
        default='ts',
        choices=['ts', 'st', 'channel_independent'],
        help='ts = per-station TS, st = multi-station spatial-temporal, '
        'channel_independent = per-feature-channel univariate.',
    )
    g.add_argument(
        '--scaler',
        default='minmax',
        choices=['none', 'minmax', 'standard'],
        help='Per-station normalisation: none = raw values, '
        'minmax = [0,1] scaling, standard = z-score.',
    )
    g.add_argument(
        '--use_current_x',
        action='store_true',
        default=True,
        help='Include current time-step features as model input. '
        'Only relevant for nowcasting (predict y_n at t_n given x_1…x_n, y_1…y_{n-1}).',
    )
    g.add_argument(
        '--no_current_x',
        action='store_false',
        dest='use_current_x',
        help='Exclude current time-step features (strict forecasting).',
    )
    g.add_argument(
        '--use_full_history',
        action='store_true',
        help='LSTM-like full-history mode (no sliding window).',
    )

    # ----- Gap / break handling ------------------------------------------
    g = p.add_argument_group('Gap handling')
    g.add_argument(
        '--short_subsequence_method',
        default='drop',
        choices=['drop', 'pad'],
        help='How to handle sub-sequences shorter than seq_len.',
    )
    g.add_argument(
        '--gap_mode',
        default='split',
        choices=['split', 'mask_pad'],
        help='split = treat gaps as segment boundaries; '
        'mask_pad = fill small gaps with zeros + mask.',
    )
    g.add_argument(
        '--max_mask_consecutive',
        type=int,
        default=10,
        help='Max gap size to fill when gap_mode=mask_pad.',
    )

    # ----- Noise injection -----------------------------------------------
    g = p.add_argument_group('Noise injection')
    g.add_argument(
        '--noise_type',
        default=None,
        choices=[
            None,
            'gaussian',
            'gaussian_a',
            'impulse',
            'impulse_a',
            'quantization',
        ],
        help='Noise family to inject into input features.',
    )
    g.add_argument(
        '--noise_level',
        type=float,
        default=0.01,
        help='Gaussian noise level (fraction of data std).',
    )
    g.add_argument(
        '--noise_probability',
        type=float,
        default=0.01,
        help='Impulse noise probability per data point.',
    )
    g.add_argument(
        '--noise_scale_factor',
        type=float,
        default=5.0,
        help='Impulse noise spike magnitude (× data std).',
    )

    # ----- Historical target inclusion -----------------------------------
    g = p.add_argument_group('Historical target')
    g.add_argument(
        '--include_historical_y',
        default='none',
        choices=['none', 'gt', 'predicted'],
        help='Include historical ground-truth or predicted y.',
    )
    g.add_argument(
        '--include_historical_predicted_y',
        action='store_true',
        help='Append predicted y of neighbours as extra features.',
    )

    # ----- Entity identifiers -------------------------------------------
    g = p.add_argument_group('Entity identifiers')
    g.add_argument(
        '--identifier_mode',
        default='embedding',
        choices=[
            'none',
            'embedding',
            'embedding_idx',
            'onehot',
            'numeric_id',
            'coordinates',
            'sinusoidal',
            'descriptors',
        ],
        help='Station identifier strategy. "embedding" uses a '
        'learnable nn.Embedding (as in the Swiss River '
        'reference project).',
    )
    g.add_argument(
        '--id_integration',
        default='concat_to_x',
        choices=['concat_to_x', 'add_to_x'],
        help='How entity features are merged with input x '
        '(ignored when identifier_mode="embedding").',
    )
    g.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help='Embedding vector dimension per entity '
        '(only used when identifier_mode="embedding").',
    )
    g.add_argument(
        '--num_embeddings',
        type=int,
        default=None,
        help='Number of distinct entities for embedding table. '
        'Auto-detected from dataset if None '
        '(only used when identifier_mode="embedding").',
    )

    # ----- Graph / spatial -----------------------------------------------
    g = p.add_argument_group('Graph / spatial')
    g.add_argument(
        '--graph_mode',
        default='none',
        choices=['none', 'edge_index', 'adj_matrix', 'graphlet_features'],
        help='How graph info is exposed to the model.',
    )
    g.add_argument(
        '--graphlet_num_hops',
        type=int,
        default=1,
        help='Neighbourhood radius for graphlet features.',
    )

    # ----- Model architecture -------------------------------------------
    g = p.add_argument_group('Model architecture')
    g.add_argument(
        '--enc_in',
        type=int,
        default=None,
        help='Encoder input dim (auto-detected if None).',
    )
    g.add_argument('--dec_in', type=int, default=1)
    g.add_argument('--c_out', type=int, default=1)
    g.add_argument('--d_model', type=int, default=64)
    g.add_argument('--d_ff', type=int, default=32)
    g.add_argument('--n_heads', type=int, default=8)
    g.add_argument('--e_layers', type=int, default=2)
    g.add_argument('--d_layers', type=int, default=1)
    g.add_argument('--dropout', type=float, default=0.1)
    g.add_argument('--patch_len', type=int, default=16)
    g.add_argument('--stride', type=int, default=8)

    # ----- LLM (TimeLLM only) -------------------------------------------
    g = p.add_argument_group('LLM backbone')
    g.add_argument('--llm_model', default='GPT2')
    g.add_argument('--llm_dim', type=int, default=768)
    g.add_argument('--llm_layers', type=int, default=6)
    g.add_argument('--prompt_domain', type=int, default=0)
    g.add_argument('--cache-dir', default=os.path.join(_PROJECT_ROOT, 'cache'))

    # ----- Training ------------------------------------------------------
    g = p.add_argument_group('Training')
    g.add_argument('--train_epochs', type=int, default=30)
    g.add_argument('--batch_size', type=int, default=8)
    g.add_argument('--learning_rate', type=float, default=0.001)
    g.add_argument(
        '--loss',
        default='mse',
        choices=['mse', 'mae', 'rmse'],
        help='Training loss used by ForecastTrainer.',
    )
    g.add_argument(
        '--metrics',
        default='rmse,mae,nse',
        help='Comma-separated eval metrics. Supported: mse, mae, rmse, nse.',
    )
    g.add_argument(
        '--eval_denorm',
        action='store_true',
        default=True,
        help='Compute metrics in original data scale (after inverse transform).',
    )
    g.add_argument(
        '--no_eval_denorm',
        action='store_false',
        dest='eval_denorm',
        help='Disable denormalized metric computation.',
    )
    g.add_argument(
        '--show_progress',
        action='store_true',
        default=True,
        help='Show tqdm progress bars for train/val/test loops.',
    )
    g.add_argument(
        '--no_progress',
        action='store_false',
        dest='show_progress',
        help='Disable tqdm progress bars.',
    )
    g.add_argument('--patience', type=int, default=10)
    g.add_argument(
        '--lradj',
        default='none',
        choices=[
            'none',
            'constant',
            'type1',
            'type2',
            'type3',
            'PEMS',
            'COS',
            'cosine_warmup',
            'onecycle',
            'step',
            'multistep',
            'exponential',
            'plateau',
            'TST',
        ],
        help='LR schedule type. Swiss River LSTM uses "none" '
        '(constant Adam LR). See liulian.optim.lr_schedulers '
        'for all options.',
    )
    g.add_argument(
        '--pct_start', type=float, default=0.2, help='OneCycleLR warm-up fraction.'
    )
    g.add_argument('--cos_T_max', type=int, default=20, help='CosineAnnealingLR T_max.')
    g.add_argument(
        '--cos_eta_min',
        type=float,
        default=1e-8,
        help='CosineAnnealingLR / CosineWarmRestarts eta_min.',
    )
    g.add_argument(
        '--cos_T_0', type=int, default=10, help='CosineAnnealingWarmRestarts T_0.'
    )
    g.add_argument(
        '--cos_T_mult', type=int, default=2, help='CosineAnnealingWarmRestarts T_mult.'
    )
    g.add_argument(
        '--step_size', type=int, default=10, help='StepLR step_size (epochs).'
    )
    g.add_argument(
        '--gamma',
        type=float,
        default=0.5,
        help='Decay factor for StepLR/ExponentialLR/MultiStepLR.',
    )
    g.add_argument(
        '--milestones',
        default='30,60,90',
        help='Comma-separated epoch milestones for MultiStepLR.',
    )
    g.add_argument(
        '--sched_patience', type=int, default=5, help='ReduceLROnPlateau patience.'
    )
    g.add_argument(
        '--sched_factor', type=float, default=0.5, help='ReduceLROnPlateau factor.'
    )
    g.add_argument('--num_workers', type=int, default=0)

    # ----- Logging -------------------------------------------------------
    g = p.add_argument_group('Logging')
    g.add_argument(
        '--wandb_project',
        default=None,
        help='Enable wandb logging with this project name.',
    )
    g.add_argument('--wandb_entity', default=None, help='wandb team/user entity.')
    g.add_argument(
        '--dev_run',
        action='store_true',
        help='Dev mode: disables wandb even if project is set.',
    )

    # ----- HPO -----------------------------------------------------------
    g = p.add_argument_group('Hyperparameter optimisation')
    g.add_argument(
        '--hpo',
        action='store_true',
        default=True,
        help='Enable Ray Tune hyperparameter search.',
    )
    g.add_argument(
        '--hpo_num_samples',
        type=int,
        default=200,
        help='Number of HPO trials (Swiss River ref default: 200).',
    )
    g.add_argument(
        '--hpo_scheduler',
        default='asha',
        choices=['asha', 'none'],
        help='Ray Tune scheduler.',
    )
    g.add_argument(
        '--hpo_grace_period',
        type=int,
        default=5,
        help='ASHA min epochs before pruning '
        '(Swiss River ref: 3 for vanilla, 5 for embedding).',
    )
    g.add_argument(
        '--hpo_reduction_factor',
        type=float,
        default=1.5,
        help='ASHA bracket reduction factor '
        '(Swiss River ref: 2 for vanilla, 1.5 for embedding).',
    )
    g.add_argument(
        '--hpo_storage_path',
        type=str,
        default=None,
        help='Directory for Ray Tune output (default: artifacts/ray_results).',
    )
    g.add_argument(
        '--hpo_resources_cpu', type=int, default=1, help='CPUs per trial (default: 1).'
    )
    g.add_argument(
        '--hpo_resources_gpu',
        type=float,
        default=0.25,
        help='GPUs per trial — use fractional (e.g. 0.25) '
        'to share one GPU across trials (default: 0).',
    )
    g.add_argument(
        '--hpo_num_cpus',
        type=int,
        default=None,
        help='Total CPUs for ray.init(). Negative = cpu_count - N. None = auto.',
    )
    g.add_argument(
        '--hpo_max_concurrent',
        type=int,
        default=None,
        help='Max concurrent trials (default: unlimited).',
    )
    g.add_argument(
        '--hpo_resume',
        action='store_true',
        default=False,
        help='Resume a previously interrupted HPO run.',
    )
    g.add_argument(
        '--hpo_save_checkpoints',
        action='store_true',
        default=True,
        help='Save .pth checkpoints per trial (default: True).',
    )
    g.add_argument(
        '--hpo_no_save_checkpoints',
        action='store_false',
        dest='hpo_save_checkpoints',
        help='Disable checkpoint saving.',
    )
    g.add_argument(
        '--hpo_trim_checkpoints',
        action='store_true',
        default=True,
        help='Trim .pth files after HPO to save disk (default: True).',
    )
    g.add_argument(
        '--hpo_no_trim',
        action='store_false',
        dest='hpo_trim_checkpoints',
        help='Disable post-HPO checkpoint trimming.',
    )
    g.add_argument(
        '--hpo_keep_best_n',
        type=int,
        default=10,
        help='Number of best trials to keep when trimming (default: 10).',
    )
    g.add_argument(
        '--hpo_trim_best_n',
        action='store_true',
        default=True,
        help='Also trim checkpoints within the best-N trials (default: True).',
    )
    g.add_argument(
        '--hpo_no_trim_best_n',
        action='store_false',
        dest='hpo_trim_best_n',
        help='Keep all checkpoints for the best-N trials.',
    )
    g.add_argument(
        '--hpo_trim_keep_best',
        action='store_true',
        default=True,
        help='Keep best-epoch checkpoint for trimmed trials (default: True).',
    )
    g.add_argument(
        '--hpo_no_trim_keep_best',
        action='store_false',
        dest='hpo_trim_keep_best',
        help='Remove all checkpoints for trimmed trials.',
    )
    g.add_argument(
        '--hpo_trim_keep_last',
        action='store_true',
        default=False,
        help='Keep last-epoch checkpoint for trimmed trials (default: False).',
    )
    g.add_argument(
        '--hpo_experiment_name',
        type=str,
        default=None,
        help='Extra distinguisher appended to the auto-generated '
        'Ray experiment name.  Default name pattern: '
        '{data}_{model}_{task}_{split_mode}[_{extra}]_{timestamp}.',
    )
    g.add_argument(
        '--hpo_local_mode',
        action='store_true',
        default=False,
        help='Debug mode: forces num_cpus=1 so all Ray tasks run '
        'in a single process.  Breakpoints and IDE debuggers '
        'work normally.  (Replaces the old Ray local_mode.)',
    )

    # ----- Visualisation -------------------------------------------------
    g = p.add_argument_group('Visualisation')
    g.add_argument(
        '--auto_viz',
        action='store_true',
        default=True,
        help='Auto-generate plots after training.',
    )
    g.add_argument(
        '--no_viz',
        action='store_false',
        dest='auto_viz',
        help='Disable auto-visualisation.',
    )
    g.add_argument(
        '--viz_method',
        default='mean',
        choices=[
            'mean',
            'median',
            'last',
            'longest_history',
            'best',
            'worst',
            'single',
        ],
        help='Aggregation method for prediction plots.',
    )

    # ----- Misc model keys expected by Time-LLM models --------------------
    g = p.add_argument_group('Misc')
    g.add_argument('--embed', default='timeF')
    g.add_argument('--activation', default='gelu')
    g.add_argument('--output_attention', action='store_true')
    g.add_argument('--moving_avg', type=int, default=25)
    g.add_argument('--factor', type=int, default=1)
    g.add_argument('--task_name', default='long_term_forecast')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_noise_kwargs(args: argparse.Namespace) -> dict | None:
    """Pack noise-related CLI args into a kwargs dict for the dataset."""
    if args.noise_type is None:
        return None
    return {
        'noise_level': args.noise_level,
        'probability': args.noise_probability,
        'scale_factor': args.noise_scale_factor,
    }


def _auto_enc_in(dataset: SwissRiverDataset) -> int:
    """Derive ``enc_in`` from the training split's feature dim."""
    split = dataset.get_split('train')
    return split.feat_dim if split.feat_dim > 0 else 1


def _print_dataset_summary(dataset: SwissRiverDataset) -> None:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Configuration ----
    config = vars(args).copy()
    if args.quick_test:  # fixme test: should remove not
        config.update(
            train_epochs=2,
            batch_size=4,
            patience=2,
            max_train_iters=100,
            max_eval_iters=50,
            num_workers=0,
            identifier_mode='embedding',
            hpo_num_samples=2,
            hpo_local_mode=True,
        )
        logger.info('Quick test: 2 epochs, 100 iter/epoch, batch_size=4')

    # Set settings for lstm (Swiss River benchmark uses no LR scheduling):
    if args.model == 'lstm':
        config.update(
            dropout=0.0,
            batch_size=256,
            lradj='none',  # Swiss River LSTM: constant Adam LR
        )

    # ---- Dataset ----
    noise_kwargs = _build_noise_kwargs(args)
    dataset = SwissRiverDataset(
        data_name=config['data'],
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

    _print_dataset_summary(dataset)

    # ---- Auto-detect enc_in from actual data shape ----
    if config['enc_in'] is None:
        config['enc_in'] = _auto_enc_in(dataset)
        logger.info('Auto-detected enc_in=%d from training data', config['enc_in'])

    # ---- Task ----
    task = PredictionTask(
        regime=PredictionRegime(
            horizon=config['pred_len'],
            context_length=config['seq_len'],
        ),
        loss_name=config['loss'],
        metrics=_parse_metric_names(config['metrics']),
    )

    # ---- Model ----
    model = build_model(args.model, config)

    # ---- Wrap with EntityWrapper for learnable embedding mode ----
    if config['identifier_mode'] == 'embedding':
        from liulian.models.torch.entity_mixin import EntityWrapper

        num_emb = config.get('num_embeddings') or len(dataset.station_ids)
        config['num_embeddings'] = num_emb
        model = EntityWrapper(
            inner_model=model,
            enc_in=config['enc_in'],
            num_embeddings=num_emb,
            embedding_size=config.get('embedding_size', 10),
        )
        logger.info(
            'EntityWrapper: num_embeddings=%d, embedding_size=%d',
            num_emb,
            config['embedding_size'],
        )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info('Model: %s  (%.1fK params)', args.model, n_params / 1e3)

    # ---- Spec ----
    spec = ExperimentSpec(
        name=f'swiss_river_{args.model}_{config["task"]}_{config["split_mode"]}',
        task={
            'type': 'PredictionTask',
            'pred_len': config['pred_len'],
            'task': config['task'],
            'loss': config['loss'],
            'metrics': _parse_metric_names(config['metrics']),
        },
        dataset={
            'type': 'SwissRiverDataset',
            'data': config['data'],
            'split_mode': config['split_mode'],
            'graph_mode': config['graph_mode'],
            'identifier_mode': config['identifier_mode'],
            'noise_type': config['noise_type'],
        },
        model={
            'type': args.model,
            'd_model': config['d_model'],
            'enc_in': config['enc_in'],
        },
        metadata={'seed': args.seed},
    )

    # ---- Data loaders ----
    loaders = dataset.get_data_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    # ---- Logger (wandb or local) ----
    exp_logger = None
    if config.get('wandb_project') and not config.get('dev_run'):
        try:
            from liulian.loggers.wandb_logger import WandbLogger

            exp_logger = WandbLogger(
                project=config['wandb_project'],
                entity=config.get('wandb_entity'),
                config=config,
            )
            logger.info('wandb logging enabled → project=%s', config['wandb_project'])
        except Exception as exc:
            logger.warning('wandb init failed (%s); using local logging.', exc)

    # ---- HPO optimizer ----
    optimizer = None
    if config.get('hpo'):
        from liulian.optim.ray_optimizer import RayOptimizer

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
                'experiment_name': _build_hpo_experiment_name(config, args.model),
                'local_mode': config.get('hpo_local_mode', False),
            }
        )
        # Default search space for LSTM / TimeLLM
        config['search_space'] = config.get(
            'search_space',
            {
                # 'batch_size': randint(32, 256 + 1),
                'embedding_size': ray.tune.randint(1, 30 + 1),
                'd_model': ray.tune.randint(16, 128 + 1),  # 128
                'e_layers': ray.tune.randint(1, 3 + 1),  # more layers!
                'learning_rate': ray.tune.uniform(
                    0.00001, 0.01
                ),  # [0.0001, 0.0005, 0.001, 0.005],  # 10x less
                # 'dropout': [0.05, 0.1, 0.2],
            },
        )
        logger.info(
            'HPO enabled: %d samples, grace_period=%s, reduction_factor=%s, '
            'storage=%s, resources_per_trial=%s',
            config.get('hpo_num_samples', 200),
            config.get('hpo_grace_period', 5),
            config.get('hpo_reduction_factor', 1.5),
            config.get('hpo_storage_path', 'artifacts/ray_results'),
            {
                'cpu': config.get('hpo_resources_cpu', 1),
                'gpu': config.get('hpo_resources_gpu', 0),
            },
        )

    # ---- Auto-viz config ----
    config['auto_viz'] = config.get('auto_viz', True)
    config['viz_method'] = config.get('viz_method', 'mean')

    # ---- Experiment ----
    exp = Experiment(
        spec=spec,
        task=task,
        dataset=dataset,
        model=None,  # not using liulian adapter
        torch_model=model,
        optimizer=optimizer,
        exp_logger=exp_logger,
        data_loaders=loaders,
        config=config,
    )

    # ---- Run! ----
    summary = exp.run(train=not args.eval_only)

    elapsed = time.time() - t0

    # ---- Visualisation ----
    artifacts_dir = summary.get('artifacts_dir', 'artifacts')
    pred_result = summary.get('predictions')
    viz_method = config.get('viz_method', 'mean')

    # If auto_viz is disabled or predictions were already visualized by
    # Experiment.visualize(), skip the manual plots.
    if pred_result is not None and 'viz_paths' not in summary:
        from liulian.viz.plots import save_prediction_plots

        try:
            viz_dir = os.path.join(artifacts_dir, 'figures')
            plot_paths = save_prediction_plots(
                preds=pred_result['preds'],
                trues=pred_result['trues'],
                times=pred_result['times'],
                method=viz_method,
                output_dir=viz_dir,
                title_prefix=f'{config["data"]} / {args.model} — ',
                target_names=['water_temperature'],
            )
            logger.ok('Saved prediction plots:')
            for name, path in plot_paths.items():
                logger.info('  %s → %s', name, path)
        except Exception as exc:
            logger.warning('Visualisation failed: %s', exc)
    elif 'viz_paths' in summary:
        logger.info('Auto-viz plots already generated by Experiment.')

    # Also save raw predictions as .npz for later analysis  todo: as pth?
    if pred_result is not None:
        npz_path = os.path.join(artifacts_dir, 'predictions.npz')
        np.savez_compressed(
            npz_path,
            preds=pred_result['preds'].numpy(),
            trues=pred_result['trues'].numpy(),
            times=pred_result['times'].numpy(),
        )
        logger.ok('Raw predictions saved → %s', npz_path)

    # ---- Report ----
    print(f'\n{"=" * 60}')
    print(f'Experiment : {spec.name}')
    print(f'Data       : {config["data"]}  (split_mode={config["split_mode"]})')
    print(
        f'Task       : {config["task"]}  (seq={config["seq_len"]}, '
        f'pred={config["pred_len"]}, full_hist={config["use_full_history"]})'
    )
    print(f'Graph      : {config["graph_mode"]}')
    print(f'Noise      : {config["noise_type"] or "none"}')
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
            continue  # too verbose for console
        print(f'  {key}: {val}')
    if 'hpo' in summary.get('metrics', {}):
        hpo = summary['metrics']['hpo']
        print(f'  HPO best: {hpo["best_value"]:.6f}  trials: {hpo["n_trials"]}')
    print(f'{"=" * 60}')

    # ---- Cleanup ----
    if exp_logger is not None:
        try:
            exp_logger.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
