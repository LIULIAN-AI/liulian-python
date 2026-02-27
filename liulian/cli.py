"""Command-line interface for liulian.

Thin CLI layer that delegates to :mod:`liulian.pipeline` for all heavy
lifting.  Every subcommand loads configuration via the three-layer
priority model (``DEFAULT_CONFIG < YAML < CLI``) and calls pipeline
functions.

Subcommands
-----------
``liulian info``
    Print version and project tagline.

``liulian run <config.yaml> [overrides…]``
    Full pipeline — seed → build → train → eval → viz → report.

``liulian train <config.yaml> [overrides…]``
    Training only (no final test evaluation).

``liulian eval <config.yaml> [overrides…]``
    Evaluate from checkpoint (no training).

``liulian predict <config.yaml> [overrides…]``
    Inference / prediction from checkpoint.

``liulian tune <config.yaml> [overrides…]``
    Hyperparameter search via Ray Tune.

``liulian viz <config.yaml> [overrides…]``
    Generate prediction plots from checkpoint.

CLI override syntax
-------------------
Any ``DEFAULT_CONFIG`` key can be overridden on the command line::

    liulian run config.yaml --model lstm --train_epochs 50 --batch_size 16

Boolean flags use ``--flag`` / ``--no_flag`` (e.g. ``--quick_test``
/ ``--no_quick_test``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict

from liulian import __version__
from liulian.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# ── Viz method choices (shared across subcommands) ──────────────────────
_VIZ_METHODS = [
    'mean',
    'median',
    'last',
    'longest_history',
    'best',
    'worst',
    'single',
]


# ── Config helpers ──────────────────────────────────────────────────────


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML config + CLI overrides via :func:`liulian.config.load_config`.

    The ``args`` namespace **must** contain ``config`` (YAML path, may be
    ``None``).  All other attributes are treated as CLI overrides.
    """
    from liulian.config import load_config

    yaml_path = getattr(args, 'config', None)

    # Collect all non-None CLI overrides (skip argparse internal keys)
    skip = {'config', 'command', 'func', 'verbose'}
    cli_overrides = {
        k: v for k, v in vars(args).items() if k not in skip and v is not None
    }
    return load_config(yaml_path=yaml_path, cli_overrides=cli_overrides)


# ── Subcommands ─────────────────────────────────────────────────────────


def cmd_info(_args: argparse.Namespace) -> None:
    """Print version information."""
    print(f'liulian {__version__}')
    print('Liquid Intelligence and Unified Logic for Interactive Adaptive Networks')
    print('"Where Space and Time Converge in Intelligence"')


def cmd_run(args: argparse.Namespace) -> None:
    """Full pipeline — train + eval + viz + report."""
    from liulian.pipeline import run_experiment

    cfg = _build_config(args)
    run_experiment(cfg)


def cmd_train(args: argparse.Namespace) -> None:
    """Train a model (no final test evaluation)."""
    from liulian.pipeline import build_experiment, seed_everything
    from liulian.config import apply_model_defaults, apply_quick_test

    cfg = _build_config(args)
    seed_everything(cfg.get('seed', 2026))
    if cfg.get('quick_test'):
        apply_quick_test(cfg)
    apply_model_defaults(cfg)

    exp = build_experiment(cfg)
    summary = exp.run(train=True, eval=False)
    _print_summary(summary)


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate from checkpoint (no training)."""
    from liulian.pipeline import build_experiment, seed_everything
    from liulian.config import apply_model_defaults

    cfg = _build_config(args)
    cfg['eval_only'] = True
    seed_everything(cfg.get('seed', 2026))
    apply_model_defaults(cfg)

    exp = build_experiment(cfg)
    summary = exp.run(train=False, eval=True)
    _print_summary(summary)


def cmd_predict(args: argparse.Namespace) -> None:
    """Run inference from a trained checkpoint."""
    from liulian.pipeline import build_experiment, seed_everything
    from liulian.config import apply_model_defaults

    cfg = _build_config(args)
    seed_everything(cfg.get('seed', 2026))
    apply_model_defaults(cfg)

    exp = build_experiment(cfg)
    summary = exp.run(train=False, eval=True, infer=True)
    _print_summary(summary)
    if 'predictions' in summary:
        preds = summary['predictions']
        print(f'  Predictions shape: {list(preds["preds"].shape)}')
        print(f'  Ground truth shape: {list(preds["trues"].shape)}')


def cmd_tune(args: argparse.Namespace) -> None:
    """Hyperparameter search via Ray Tune."""
    from liulian.pipeline import build_experiment, seed_everything
    from liulian.config import apply_model_defaults, apply_quick_test

    cfg = _build_config(args)
    cfg['hpo'] = True  # ensure HPO is on
    seed_everything(cfg.get('seed', 2026))
    if cfg.get('quick_test'):
        apply_quick_test(cfg)
    apply_model_defaults(cfg)

    exp = build_experiment(cfg)
    summary = exp.run(train=True, eval=True)
    _print_summary(summary)
    if 'hpo' in summary.get('metrics', {}):
        hpo = summary['metrics']['hpo']
        print('\n  HPO Results:')
        print(f'    Best value: {hpo["best_value"]:.6f}')
        print(f'    Trials: {hpo["n_trials"]}')
        print(f'    Best config: {hpo["best_config"]}')


def cmd_viz(args: argparse.Namespace) -> None:
    """Generate prediction visualisations from checkpoint."""
    from liulian.pipeline import build_experiment, seed_everything
    from liulian.config import apply_model_defaults

    cfg = _build_config(args)
    cfg['auto_viz'] = True
    seed_everything(cfg.get('seed', 2026))
    apply_model_defaults(cfg)

    exp = build_experiment(cfg)
    summary = exp.run(train=False, eval=True)
    if 'predictions' in summary:
        paths = exp.visualize(summary, method=cfg.get('viz_method', 'mean'))
        print(f'\n  Saved {len(paths)} plots:')
        for name, path in paths.items():
            print(f'    {name}: {path}')
    else:
        print('  No predictions available for visualization.')


def _print_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print experiment summary."""
    print(f'\n{"=" * 50}')
    print(f'  Status: {summary["status"]}')
    print(f'  Run ID: {summary.get("run_id", "N/A")}')
    if 'training' in summary.get('metrics', {}):
        m = summary['metrics']['training']
        print(f'  Epochs: {m.get("epochs_run", "?")}')
        best = m.get('best_val_mse')
        if best is not None:
            print(f'  Best Val MSE: {best:.6f}')
    if 'final_test' in summary.get('metrics', {}):
        t = summary['metrics']['final_test']
        for k, v in t.items():
            if isinstance(v, float):
                print(f'  Test {k.upper()}: {v:.6f}')
    print(f'  Artifacts: {summary.get("artifacts_dir", "N/A")}')
    print(f'{"=" * 50}')


# ── Argument building ──────────────────────────────────────────────────


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add config path and universal override arguments to *parser*.

    Every subcommand (except ``info``) gets:
    - Positional ``config`` (YAML path)
    - All DEFAULT_CONFIG keys as ``--key`` overrides
    """
    parser.add_argument(
        'config',
        nargs='?',
        default=None,
        help='Path to YAML config file (optional; uses defaults if omitted).',
    )

    # Dynamically generate --key args from DEFAULT_CONFIG
    for key, default in DEFAULT_CONFIG.items():
        flag = f'--{key}'
        if isinstance(default, bool):
            # Boolean: --flag / --no_flag
            parser.add_argument(flag, action='store_true', default=None)
            parser.add_argument(
                f'--no_{key}', action='store_false', dest=key, default=None
            )
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=None)
        elif isinstance(default, float):
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, default=None)


# ── Main entry point ───────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``liulian`` CLI."""
    parser = argparse.ArgumentParser(
        prog='liulian',
        description='LIULIAN — Research OS for spatiotemporal model experimentation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'examples:\n'
            '  liulian run config.yaml\n'
            '  liulian run config.yaml --model lstm --train_epochs 50\n'
            '  liulian tune config.yaml --hpo_num_samples 100\n'
            '  liulian eval config.yaml --eval_only\n'
            '  liulian viz config.yaml --viz_method median\n'
        ),
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'liulian {__version__}',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging.',
    )

    subparsers = parser.add_subparsers(dest='command', help='Available subcommands')

    # info
    sp_info = subparsers.add_parser('info', help='Show version and project info')
    sp_info.set_defaults(func=cmd_info)

    # run  — full pipeline
    sp_run = subparsers.add_parser(
        'run',
        help='Full pipeline (train + eval + viz + report)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(sp_run)
    sp_run.set_defaults(func=cmd_run)

    # train
    sp_train = subparsers.add_parser(
        'train',
        help='Train a model (no final test)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(sp_train)
    sp_train.set_defaults(func=cmd_train)

    # eval
    sp_eval = subparsers.add_parser(
        'eval',
        help='Evaluate from checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(sp_eval)
    sp_eval.set_defaults(func=cmd_eval)

    # predict
    sp_predict = subparsers.add_parser(
        'predict',
        help='Run inference from checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(sp_predict)
    sp_predict.set_defaults(func=cmd_predict)

    # tune  (renamed from hparam — follows YOLO / Ray Tune convention)
    sp_tune = subparsers.add_parser(
        'tune',
        help='Hyperparameter search (Ray Tune)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(sp_tune)
    sp_tune.set_defaults(func=cmd_tune)

    # viz
    sp_viz = subparsers.add_parser(
        'viz',
        help='Generate prediction plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(sp_viz)
    sp_viz.set_defaults(func=cmd_viz)

    args = parser.parse_args(argv)

    # Logging
    if args.verbose:
        from liulian.utils.log_tags import setup_logging as _setup_logging

        _setup_logging(level=logging.DEBUG, fmt='%(name)s %(message)s')
    else:
        from liulian.utils.log_tags import setup_logging as _setup_logging

        _setup_logging(level=logging.INFO)

    if hasattr(args, 'func'):
        try:
            args.func(args)
        except FileNotFoundError as exc:
            print(f'Error: {exc}', file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print('\nInterrupted.', file=sys.stderr)
            sys.exit(130)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
