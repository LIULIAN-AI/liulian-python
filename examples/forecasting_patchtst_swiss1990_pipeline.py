"""Run PatchTST on Swiss River 1990 through the unified LIULIAN pipeline.

Usage:
    python examples/forecasting_patchtst_swiss1990_pipeline.py --quick-test
    python examples/forecasting_patchtst_swiss1990_pipeline.py
"""

from __future__ import annotations

import argparse
import json

from liulian.config import load_config
from liulian.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PatchTST Swiss River 1990 demo runner')
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Enable quick-test overrides for a fast smoke run.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(
        yaml_path='experiments/swiss_river/patchtst_config.yaml',
        cli_overrides={
            'quick_test': args.quick_test,
            'hpo': False,
        },
    )
    summary = run_experiment(config)

    # Print a small deterministic tail summary for docs/tutorial users.
    payload = {
        'artifacts_dir': summary.get('artifacts_dir'),
        'metrics_keys': sorted((summary.get('metrics') or {}).keys()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
