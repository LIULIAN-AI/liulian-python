#!/usr/bin/env python
"""Unified experiment entry point for LIULIAN.

Delegates to ``liulian.pipeline.run_experiment`` via the three-layer
config system (DEFAULT_CONFIG < YAML < CLI overrides).  Point ``--config``
at any dataset/model YAML under ``experiments/`` to run that experiment.

Usage
-----
Quick start (Swiss River DLinear):

    python experiments/run.py --config experiments/swiss_river/default_config.yaml --quick_test

Standard benchmarks:

    python experiments/run.py --config experiments/traffic/patchtst_config.yaml
    python experiments/run.py --config experiments/electricity/lstm_config.yaml
    python experiments/run.py --config experiments/exchange_rate/dlinear_config.yaml
    python experiments/run.py --config experiments/pems/patchtst_config.yaml --data PEMS04

Swiss River models:

    python experiments/run.py --config experiments/swiss_river/patchtst_config.yaml
    python experiments/run.py --config experiments/swiss_river/dlinear_config.yaml

CLI overrides (any config key):

    python experiments/run.py --config experiments/traffic/patchtst_config.yaml \\
        --pred_len 192 --train_epochs 50 --batch_size 64

Evaluation only:

    python experiments/run.py --config experiments/swiss_river/default_config.yaml --eval_only

Available config files
~~~~~~~~~~~~~~~~~~~~~~
::

    experiments/
    ├── electricity/   dlinear_config.yaml  lstm_config.yaml  patchtst_config.yaml
    ├── exchange_rate/ dlinear_config.yaml  lstm_config.yaml  patchtst_config.yaml
    ├── pems/          dlinear_config.yaml  lstm_config.yaml  patchtst_config.yaml
    ├── swiss_river/   default_config.yaml  dlinear_config.yaml  patchtst_config.yaml
    └── traffic/       dlinear_config.yaml  lstm_config.yaml  patchtst_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Path setup — ensure project root is importable regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from liulian.config import DEFAULT_CONFIG, load_config  # noqa: E402
from liulian.pipeline import run_experiment  # noqa: E402
from liulian.utils.log_tags import setup_logging  # noqa: E402

setup_logging(level=logging.INFO)


def _build_parser() -> argparse.ArgumentParser:
    """Build an argument parser with every DEFAULT_CONFIG key as a flag."""
    p = argparse.ArgumentParser(
        description='LIULIAN experiment runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Any key in the YAML config can be overridden from the CLI.\n'
            'Examples:\n'
            '  %(prog)s --config experiments/traffic/patchtst_config.yaml --quick_test\n'
            '  %(prog)s --config experiments/pems/lstm_config.yaml --data PEMS08 --pred_len 24\n'
        ),
    )
    p.add_argument(
        '--config', '-c',
        # required=True,
        default='etth1/patchtst_config.yaml',
        help='Path to YAML config file (e.g. experiments/traffic/patchtst_config.yaml).',
    )

    # Auto-generate --key flags from DEFAULT_CONFIG for full flexibility
    for key, default in DEFAULT_CONFIG.items():
        flag = f'--{key}'
        if isinstance(default, bool):
            p.add_argument(flag, action='store_true', default=None)
            p.add_argument(f'--no_{key}', action='store_false', dest=key, default=None)
        elif isinstance(default, int):
            p.add_argument(flag, type=int, default=None)
        elif isinstance(default, float):
            p.add_argument(flag, type=float, default=None)
        else:
            p.add_argument(flag, default=None)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Collect non-None CLI overrides
    skip = {'config'}
    cli_overrides = {
        k: v for k, v in vars(args).items() if k not in skip and v is not None
    }

    cfg = load_config(yaml_path=args.config, cli_overrides=cli_overrides)

    # # Set up quick test mode to True:
    # cfg['quick_test'] = True
    # cfg['hpo'] = False

    run_experiment(cfg)


if __name__ == '__main__':
    main()
