#!/usr/bin/env python
"""Swiss River experiment — convenience wrapper.

This is a thin convenience wrapper that delegates to the unified
``experiments/run.py`` entry point.  It defaults to the Swiss River
LSTM config (``default_config.yaml``).

Prefer using the unified entry point directly::

    python experiments/run.py --config experiments/swiss_river/default_config.yaml
    python experiments/run.py --config experiments/swiss_river/patchtst_config.yaml --quick_test

Legacy usage (still works)::

    python experiments/swiss_river/run.py                   # defaults to default_config.yaml
    python experiments/swiss_river/run.py --quick_test
    python experiments/swiss_river/run.py --config experiments/swiss_river/patchtst_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

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

    skip = {'config'}
    cli_overrides = {
        k: v for k, v in vars(args).items() if k not in skip and v is not None
    }

    cfg = load_config(yaml_path=args.config, cli_overrides=cli_overrides)
    run_experiment(cfg)


if __name__ == '__main__':
    main()
