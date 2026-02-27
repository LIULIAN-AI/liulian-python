#!/usr/bin/env python3
"""Run benchmark experiments from YAML configs.

Usage::

    # Run all configs in a directory
    python tools/run_benchmark.py --config-dir experiments/configs/long_term/

    # Run a single config
    python tools/run_benchmark.py --config experiments/configs/long_term/dlinear_etth1_H96.yaml

    # Resume (skip completed)
    python tools/run_benchmark.py --config-dir experiments/configs/long_term/ --resume

    # Multiple seeds
    python tools/run_benchmark.py --config-dir experiments/configs/long_term/ --seeds 1,2,3

    # Dry run
    python tools/run_benchmark.py --config-dir experiments/configs/long_term/ --dry-run

    # Limit GPU
    CUDA_VISIBLE_DEVICES=0 python tools/run_benchmark.py --config-dir experiments/configs/long_term/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liulian.utils.log_tags import setup_logging as _setup_logging

_setup_logging(
    level=logging.INFO, fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

RESULTS_DIR = ROOT / 'experiments' / 'results'


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def result_path(config_path: Path, seed: int, results_dir: Path) -> Path:
    """Compute the output path for a given config + seed."""
    # Extract group from parent dir name
    group = config_path.parent.name
    stem = config_path.stem
    return results_dir / group / f'{stem}_seed{seed}.json'


def is_completed(config_path: Path, seed: int, results_dir: Path) -> bool:
    """Check if this experiment has already been run."""
    rp = result_path(config_path, seed, results_dir)
    return rp.exists()


def run_single_experiment(
    config: Dict[str, Any],
    seed: int,
    output_path: Path,
    max_train_iters: Optional[int] = None,
    max_eval_iters: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single experiment and save results.

    Returns the result dict.
    """
    import torch
    import numpy as np

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Override iteration limits if provided (for smoke testing)
    if max_train_iters is not None:
        config['max_train_iters'] = max_train_iters
    if max_eval_iters is not None:
        config['max_eval_iters'] = max_eval_iters

    model_name = config['model']
    dataset_name = config.get('dataset', 'unknown')

    t0 = time.time()

    try:
        # Build model
        model, model_adapter = _build_model(config)

        # Build data loaders
        train_loader, val_loader, test_loader = _build_dataloaders(config)

        # Build trainer
        from liulian.runtime.trainer import ForecastTrainer

        trainer = ForecastTrainer(config)

        # Train
        summary = trainer.fit(model, train_loader, val_loader, test_loader)

        elapsed = time.time() - t0

        result = {
            'model': model_name,
            'dataset': dataset_name,
            'seed': seed,
            'config': {
                k: v
                for k, v in config.items()
                if not isinstance(v, (type,)) and k != 'model_class'
            },
            'final_test': summary.get('final_test', {}),
            'best_val_score': summary.get('best_val_score', float('inf')),
            'epochs_run': summary.get('epochs_run', 0),
            'elapsed_seconds': elapsed,
            'status': 'success',
        }

    except Exception as e:
        elapsed = time.time() - t0
        result = {
            'model': model_name,
            'dataset': dataset_name,
            'seed': seed,
            'config': {k: v for k, v in config.items() if not isinstance(v, (type,))},
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'elapsed_seconds': elapsed,
        }
        log.error('FAILED: %s on %s (seed %d): %s', model_name, dataset_name, seed, e)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


def _build_model(config: Dict[str, Any]):
    """Instantiate the model from config."""
    import torch

    model_name = config['model']

    # TSL models via adapter
    from liulian.models.torch import (
        DLinearAdapter,
        TransformerAdapter,
        InformerAdapter,
        AutoformerAdapter,
        FEDformerAdapter,
        iTransformerAdapter,
        PatchTSTAdapter,
        TimesNetAdapter,
        TimeMixerAdapter,
        TimeXerAdapter,
        MambaAdapter,
        LSTMAdapter,
        ExtrapoLSTMAdapter,
        TransformerEncoderAdapter,
    )

    adapter_map = {
        'DLinear': DLinearAdapter,
        'Transformer': TransformerAdapter,
        'Informer': InformerAdapter,
        'Autoformer': AutoformerAdapter,
        'FEDformer': FEDformerAdapter,
        'iTransformer': iTransformerAdapter,
        'PatchTST': PatchTSTAdapter,
        'TimesNet': TimesNetAdapter,
        'TimeMixer': TimeMixerAdapter,
        'TimeXer': TimeXerAdapter,
        'Mamba': MambaAdapter,
        'LSTMAdapter': LSTMAdapter,
        'ExtrapoLSTMAdapter': ExtrapoLSTMAdapter,
        'TransformerEncoderAdapter': TransformerEncoderAdapter,
    }

    # Optional-dependency models (require transformers)
    if model_name in ('TimeLLM', 'TimeMoE'):
        try:
            if model_name == 'TimeLLM':
                from liulian.models.torch.timellm import TimeLLMAdapter

                adapter_map['TimeLLM'] = TimeLLMAdapter
            else:
                from liulian.models.torch.timemoe import TimeMoEAdapter

                adapter_map['TimeMoE'] = TimeMoEAdapter
        except ImportError:
            raise ImportError(f'{model_name} requires transformers library')

    if model_name not in adapter_map:
        raise ValueError(f'Unknown model: {model_name}')

    adapter_cls = adapter_map[model_name]
    adapter = adapter_cls(config)
    model = adapter._model

    return model, adapter


def _build_dataloaders(config: Dict[str, Any]):
    """Build train/val/test dataloaders from config."""
    from liulian.data.data_factory import create_dataloader

    data_name = config.get('data_name', config.get('dataset', 'custom'))
    root_path = config.get('root_path', '')
    data_path = config.get('data_path', '')
    seq_len = config.get('seq_len', 96)
    label_len = config.get('label_len', 48)
    pred_len = config.get('pred_len', 96)
    size = (seq_len, label_len, pred_len)

    common = dict(
        data_name=data_name,
        root_path=root_path,
        data_path=data_path,
        size=size,
        features=config.get('features', 'M'),
        target=config.get('target', 'OT'),
        scale=config.get('scale', True),
        timeenc=config.get('timeenc', 0),
        freq=config.get('freq', 'h'),
        batch_size=config.get('batch_size', 32),
    )

    train_loader = create_dataloader(
        flag='train', shuffle=True, drop_last=True, **common
    )
    val_loader = create_dataloader(flag='val', shuffle=False, drop_last=False, **common)
    test_loader = create_dataloader(
        flag='test', shuffle=False, drop_last=False, **common
    )

    return train_loader, val_loader, test_loader


def gather_configs(config_dir: Path) -> List[Path]:
    """Collect all .yaml files in config_dir."""
    configs = sorted(config_dir.glob('*.yaml'))
    if not configs:
        configs = sorted(config_dir.glob('**/*.yaml'))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description='Run benchmark experiments')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, help='Single config YAML file')
    group.add_argument('--config-dir', type=str, help='Directory of config YAML files')
    parser.add_argument(
        '--seeds',
        type=str,
        default='42',
        help='Comma-separated list of random seeds (default: 42)',
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=str(RESULTS_DIR),
        help='Output directory for results',
    )
    parser.add_argument(
        '--resume', action='store_true', help='Skip configs whose results already exist'
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='List configs without running'
    )
    parser.add_argument(
        '--max-train-iters',
        type=int,
        default=None,
        help='Cap training iterations per epoch (for smoke testing)',
    )
    parser.add_argument(
        '--max-eval-iters',
        type=int,
        default=None,
        help='Cap eval iterations (for smoke testing)',
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    results_dir = Path(args.results_dir)

    # Collect configs
    if args.config:
        configs = [Path(args.config)]
    else:
        configs = gather_configs(Path(args.config_dir))

    if not configs:
        log.error('No config files found')
        sys.exit(1)

    total_runs = len(configs) * len(seeds)
    log.info(
        'Found %d configs × %d seeds = %d total runs',
        len(configs),
        len(seeds),
        total_runs,
    )

    if args.dry_run:
        for c in configs:
            for s in seeds:
                skip = (
                    'SKIP' if args.resume and is_completed(c, s, results_dir) else 'RUN'
                )
                print(f'  [{skip}] {c.stem} (seed {s})')
        return

    completed = 0
    skipped = 0
    failed = 0

    for i, config_path in enumerate(configs):
        cfg = load_config(config_path)
        for seed in seeds:
            run_id = f'[{i + 1}/{len(configs)}] {config_path.stem} seed={seed}'

            if args.resume and is_completed(config_path, seed, results_dir):
                log.info('SKIP (exists): %s', run_id)
                skipped += 1
                continue

            log.info('START: %s', run_id)
            out_path = result_path(config_path, seed, results_dir)

            result = run_single_experiment(
                cfg,
                seed,
                out_path,
                max_train_iters=args.max_train_iters,
                max_eval_iters=args.max_eval_iters,
            )

            if result['status'] == 'success':
                completed += 1
                test_metrics = result.get('final_test', {})
                metric_str = ', '.join(f'{k}={v:.6f}' for k, v in test_metrics.items())
                log.info(
                    'DONE: %s — %s (%.1fs)',
                    run_id,
                    metric_str,
                    result['elapsed_seconds'],
                )
            else:
                failed += 1

    log.info('=' * 60)
    log.info(
        'Completed: %d | Skipped: %d | Failed: %d | Total: %d',
        completed,
        skipped,
        failed,
        completed + skipped + failed,
    )


if __name__ == '__main__':
    main()
