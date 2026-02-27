#!/usr/bin/env python
"""Swiss River forecasting experiment — thin wrapper.

Delegates entirely to ``liulian.pipeline.run_experiment`` via the
three-layer config system (DEFAULT_CONFIG < YAML < CLI overrides).

Usage::

    # LSTM quick test
    python experiments/swiss_river/run.py --quick_test

    # LSTM full training (from default config)
    python experiments/swiss_river/run.py

    # Custom YAML + CLI overrides
    python experiments/swiss_river/run.py --config my.yaml --model timellm --train_epochs 50

    # Evaluate from checkpoint
    python experiments/swiss_river/run.py --eval_only

    # Previous run.py is preserved as run_original.py in this directory.

See ``liulian run --help`` for the full CLI interface.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Path setup — ensure project root is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from liulian.config import DEFAULT_CONFIG, load_config  # noqa: E402
from liulian.pipeline import run_experiment  # noqa: E402
from liulian.utils.log_tags import setup_logging  # noqa: E402

setup_logging(level=logging.INFO)


def main() -> None:
    p = argparse.ArgumentParser(
        description='Swiss River experiment (liulian pipeline)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--config',
        default=os.path.join(_SCRIPT_DIR, 'default_config.yaml'),
        help='Path to YAML config file.',
    )
    # Generate --key args from DEFAULT_CONFIG for full backward compat
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

    args = p.parse_args()

    # Collect non-None CLI overrides
    skip = {'config'}
    cli_overrides = {
        k: v for k, v in vars(args).items() if k not in skip and v is not None
    }

    cfg = load_config(yaml_path=args.config, cli_overrides=cli_overrides)
    run_experiment(cfg)


if __name__ == '__main__':
    main()
