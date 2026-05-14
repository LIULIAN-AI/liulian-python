#!/usr/bin/env python3
"""Run the entity-identifier matrix through experiments/run.py."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if __package__ in (None, ''):
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiments.run import run_with_config
from experiments.entity_identifier.compare import generate_reports
from experiments.entity_identifier.matrix import (
    DATASETS,
    MODELS,
    DEFAULT_SEEDS,
    MODES,
    build_cli_overrides,
    config_hash,
    entity_identifier_label,
    is_mode_applicable,
    iter_jobs,
    resolve_config,
    validate_tsl_hparam_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _timestamp_tag() -> str:
    """Build a compact timestamp tag for run directory names."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _phase_defaults(phase: str) -> dict[str, Any]:
    """Return execution defaults for each matrix phase.

    Args:
        phase: One of ``dry``, ``smoke``, ``dev``, ``full``.

    Returns:
        Dictionary containing execution-mode defaults for the phase.
    """
    if phase == 'dry':
        return {
            'execute': False,
            'hpo': False,
            'quick_test': False,
            'train_epochs': None,
            'hpo_num_samples': None,
        }
    if phase == 'smoke':
        return {
            'execute': True,
            'hpo': False,
            'quick_test': True,
            'train_epochs': 2,
            'hpo_num_samples': None,
        }
    if phase == 'dev':
        return {
            'execute': True,
            'hpo': False,
            'quick_test': False,
            'train_epochs': 5,
            'hpo_num_samples': None,
        }
    if phase == 'full':
        return {
            'execute': True,
            'hpo': True,
            'quick_test': False,
            'train_epochs': None,
            # 50 is the lower bound for paper-grade HPO with ASHA scheduler;
            # 10 is too few to find good hparams across (model, mode) cells.
            'hpo_num_samples': 50,
        }
    raise ValueError(f'Unsupported phase: {phase}')


def build_run_command(
    *,
    python_bin: str,
    run_script: Path,
    config_path: Path,
    overrides: dict[str, Any],
) -> list[str]:
    """Build equivalent CLI command text for reproducibility logs.

    Args:
        python_bin: Python executable path.
        run_script: Entry script path.
        config_path: Base YAML config path.
        overrides: CLI overrides.

    Returns:
        Tokenized command list that reproduces the in-process invocation.
    """
    cmd = [python_bin, str(run_script), '--config', str(config_path)]
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cmd.append(f'--{key}' if value else f'--no_{key}')
            continue
        cmd.extend([f'--{key}', str(value)])
    return cmd


class _TeeStream:
    """Write stdout/stderr to both terminal and log file."""

    def __init__(self, *streams: Any) -> None:
        self._streams = list(streams)
        self.encoding = getattr(streams[0], 'encoding', 'utf-8') if streams else 'utf-8'

    def write(self, text: str) -> int:
        active_streams: list[Any] = []
        for stream in self._streams:
            try:
                stream.write(text)
                stream.flush()
                active_streams.append(stream)
            except ValueError:
                # Some background loggers can emit after a wrapped stream is
                # closed during teardown; drop closed streams to avoid
                # cascading logging errors.
                continue
        self._streams = active_streams
        return len(text)

    def flush(self) -> None:
        active_streams: list[Any] = []
        for stream in self._streams:
            try:
                stream.flush()
                active_streams.append(stream)
            except ValueError:
                continue
        self._streams = active_streams

    def isatty(self) -> bool:
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self._streams)

    def __getattr__(self, name: str) -> Any:
        if not self._streams:
            raise AttributeError(name)
        return getattr(self._streams[0], name)


@contextlib.contextmanager
def _working_directory(path: Path) -> Any:
    """Temporarily switch current working directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _TimeoutExpiredError(TimeoutError):
    """Raised when in-process execution exceeds timeout."""


@contextlib.contextmanager
def _timeout_guard(seconds: int) -> Any:
    """Enforce a hard timeout using SIGALRM on POSIX."""
    if seconds <= 0:
        yield
        return

    if os.name != 'posix':
        raise RuntimeError('Timeout is only supported on POSIX for in-process mode.')

    def _handler(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
        del signum, frame
        raise _TimeoutExpiredError(f'Timed out after {seconds} seconds')

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _load_latest_status_by_job(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load latest manifest record per ``job_key``.

    Args:
        manifest_path: Path to ``manifest.jsonl``.

    Returns:
        Mapping from job key to latest record payload.
    """
    records: dict[str, dict[str, Any]] = {}
    if not manifest_path.exists():
        return records
    with manifest_path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            job_key = str(record.get('job_key', ''))
            if not job_key:
                continue
            records[job_key] = record
    return records


def _append_manifest(manifest_path: Path, record: dict[str, Any]) -> None:
    """Append a JSON record to the matrix manifest.

    Args:
        manifest_path: Path to ``manifest.jsonl``.
        record: Single run record to append.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


def _run_in_process(
    *,
    config_path: Path,
    overrides: dict[str, Any],
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> tuple[int, float]:
    """Execute one experiment in-process and tee logs to file + stdout.

    Args:
        config_path: Base YAML config path.
        overrides: CLI-style overrides passed to ``run_with_config``.
        cwd: Working directory for this job.
        log_path: Destination run log file path.
        timeout_seconds: Maximum allowed runtime (0 disables timeout).

    Returns:
        Tuple ``(return_code, elapsed_seconds)`` where:
            * ``0`` = success
            * ``-9`` = timeout
            * ``1`` = uncaught exception
    """
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log_fh:
        tee = _TeeStream(sys.stdout, log_fh)
        try:
            with (
                _working_directory(cwd),
                contextlib.redirect_stdout(tee),
                contextlib.redirect_stderr(tee),
                _timeout_guard(timeout_seconds),
            ):
                run_with_config(
                    config_path=str(config_path),
                    cli_overrides=overrides,
                )
        except _TimeoutExpiredError:
            print(f'[error] Timeout after {timeout_seconds} seconds', file=tee)
            elapsed = time.time() - start
            return -9, elapsed
        except Exception:
            traceback.print_exc(file=tee)
            elapsed = time.time() - start
            return 1, elapsed
    elapsed = time.time() - start
    return 0, elapsed


def _find_latest_results_json(job_dir: Path) -> Path | None:
    """Find the newest ``results.json`` under a job directory."""
    artifacts_root = job_dir / 'artifacts'
    if not artifacts_root.exists():
        return None
    candidates = list(artifacts_root.glob('**/results.json'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


SUPPORTED_PLOT_AGGREGATIONS: tuple[str, ...] = (
    'longest_history',
    'last',
    'mean',
    'median',
    'best',
    'worst',
    'single',
)


def create_pred_vs_true_plots(
    *,
    npz_path: Path,
    output_dir: Path,
    aggregations: list[str],
    formats: list[str],
    file_stem: str = 'pred_vs_gt',
) -> list[Path]:
    """Create predicted-vs-ground-truth plots via the shared viz module.

    Args:
        npz_path: Path to ``predictions.npz``.
        output_dir: Directory for plot files.
        aggregations: Aggregation methods; each creates one curve pair.
        formats: Figure formats, e.g. ``['png', 'svg', 'pdf']``.
        file_stem: Base file name prefix.

    Returns:
        List of created plot paths.
    """
    from liulian.viz.plots import save_prediction_plots

    with np.load(npz_path, allow_pickle=True) as data:
        preds = np.asarray(data['preds'])
        trues = np.asarray(data['trues'])
        times = np.asarray(data['times']) if 'times' in data else None
        entity_ids = np.asarray(data['entity_ids']) if 'entity_ids' in data else None

    normalized_aggs = [agg.strip().lower() for agg in aggregations if agg.strip()]
    normalized_formats = [fmt.strip().lower().lstrip('.') for fmt in formats if fmt.strip()]
    if not normalized_aggs:
        raise ValueError('At least one plot aggregation is required.')
    if not normalized_formats:
        raise ValueError('At least one plot format is required.')

    created_paths: list[Path] = []
    seen_paths: set[Path] = set()
    output_dir.mkdir(parents=True, exist_ok=True)

    for aggregation in normalized_aggs:
        if aggregation not in SUPPORTED_PLOT_AGGREGATIONS:
            raise ValueError(
                f'Unsupported aggregation: {aggregation!r}. '
                f'Use one of {SUPPORTED_PLOT_AGGREGATIONS}.'
            )
        plot_paths = save_prediction_plots(
            preds=preds,
            trues=trues,
            times=times,
            entity_ids=entity_ids,
            method=aggregation,
            output_dir=str(output_dir),
            formats=normalized_formats,
            plot_stem=f'{file_stem}_{aggregation}',
            range_stem=f'{file_stem}_{aggregation}_range',
        )
        for raw_path in plot_paths.values():
            path = Path(raw_path)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            created_paths.append(path)

    return created_paths


def _extract_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Extract report metrics from a results JSON payload."""
    metrics = results.get('metrics', {})
    test_metrics = metrics.get('test', {}) if isinstance(metrics, dict) else {}
    hpo = results.get('hpo') or metrics.get('hpo') or {}
    return {
        'test_mse': test_metrics.get('mse'),
        'test_rmse': test_metrics.get('rmse'),
        'test_mae': test_metrics.get('mae'),
        'test_nse': test_metrics.get('nse'),
        'hpo_best_value': hpo.get('best_value') if isinstance(hpo, dict) else None,
        'hpo_trials': hpo.get('n_trials') if isinstance(hpo, dict) else None,
    }


_MIN_TRAIN_EPOCHS_FULL = 5
_MIN_TRAIN_SAMPLES_FULL = 1000
_MIN_HPO_NUM_SAMPLES_FULL = 10
_MIN_HPO_NUM_SAMPLES_PAPER = 50


def _assert_no_dev_caps_in_full(
    args: argparse.Namespace, phase_cfg: dict[str, Any]
) -> None:
    """Fail loudly if --phase=full was invoked with dev/smoke caps.

    A real production bug once let a dev YAML silently cap full-phase
    runs to ``train_epochs=1`` and ``max_train_samples=100``, producing
    undertrained baselines that looked plausible. This guard makes the
    failure mode loud instead of silent.
    """
    if args.phase != 'full':
        return
    te = args.train_epochs if args.train_epochs is not None else phase_cfg.get(
        'train_epochs'
    )
    if te is not None and te < _MIN_TRAIN_EPOCHS_FULL:
        raise ValueError(
            f'--phase=full but train_epochs={te} (< {_MIN_TRAIN_EPOCHS_FULL}). '
            f'This is a dev/smoke value. Either drop --train-epochs (use per-experiment '
            f'config default), drop --config <dev-yaml>, or pass a real value (e.g. 30).'
        )
    mts = args.max_train_samples
    if mts is not None and mts < _MIN_TRAIN_SAMPLES_FULL:
        raise ValueError(
            f'--phase=full but max_train_samples={mts} (< {_MIN_TRAIN_SAMPLES_FULL}). '
            f'This is a dev/smoke cap. Drop --max-train-samples to use all data, '
            f'or pass a real value.'
        )
    hns = args.hpo_num_samples if args.hpo_num_samples is not None else phase_cfg.get(
        'hpo_num_samples'
    )
    if hns is not None and hns < _MIN_HPO_NUM_SAMPLES_FULL:
        raise ValueError(
            f'--phase=full but hpo_num_samples={hns} (< {_MIN_HPO_NUM_SAMPLES_FULL}). '
            f'This is a dev/smoke value. Use >= {_MIN_HPO_NUM_SAMPLES_PAPER} for paper baselines.'
        )
    if hns is not None and hns < _MIN_HPO_NUM_SAMPLES_PAPER:
        import warnings

        warnings.warn(
            f'--phase=full with hpo_num_samples={hns} is below the recommended '
            f'{_MIN_HPO_NUM_SAMPLES_PAPER} for paper-grade HPO. Results may be noisy.',
            stacklevel=2,
        )


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    """Run the entity-identifier matrix according to CLI options.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Summary dictionary containing run counts and report paths.
    """
    phase_cfg = _phase_defaults(args.phase)
    _assert_no_dev_caps_in_full(args, phase_cfg)
    run_root = Path(args.output_root).resolve() / args.run_tag
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / 'manifest.jsonl'
    run_script = PROJECT_ROOT / 'experiments' / 'run.py'

    jobs = iter_jobs(
        datasets=args.datasets,
        models=args.models,
        modes=args.modes,
        seeds=args.seeds,
    )
    requested_datasets = tuple(args.datasets) if args.datasets else DATASETS
    requested_models = tuple(args.models) if args.models else MODELS
    requested_modes = tuple(args.modes) if args.modes else MODES
    inapplicable = [
        (dataset, model, mode)
        for dataset in requested_datasets
        for model in requested_models
        for mode in requested_modes
        if not is_mode_applicable(dataset, model, mode)
    ]
    if inapplicable:
        combo_preview = ', '.join(
            f'{dataset}/{model}/{mode}' for dataset, model, mode in inapplicable[:8]
        )
        if len(inapplicable) > 8:
            combo_preview += f' ... (+{len(inapplicable) - 8} more)'
        print(
            '  - skipped inapplicable dataset/model/mode combinations: '
            f'{combo_preview}'
        )
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    previous = _load_latest_status_by_job(manifest_path)
    summary = {
        'run_root': str(run_root),
        'total_jobs': len(jobs),
        'executed': 0,
        'skipped': 0,
        'failed': 0,
    }

    for index, job in enumerate(jobs, start=1):
        print(f'[{index}/{len(jobs)}] {job.job_key}')
        if args.resume and previous.get(job.job_key, {}).get('status') == 'ok':
            summary['skipped'] += 1
            print('  - skipped (already completed with status=ok)')
            continue

        job_dir = run_root / job.folder_name
        job_dir.mkdir(parents=True, exist_ok=True)
        hpo_storage = job_dir / 'ray_results' if phase_cfg['hpo'] else None
        overrides = build_cli_overrides(
            job,
            hpo=phase_cfg['hpo'],
            quick_test=phase_cfg['quick_test'],
            max_train_samples=args.max_train_samples,
            train_epochs=(
                args.train_epochs
                if args.train_epochs is not None
                else phase_cfg['train_epochs']
            ),
            hpo_num_samples=(
                args.hpo_num_samples
                if args.hpo_num_samples is not None
                else phase_cfg.get('hpo_num_samples')
            ),
            hpo_storage_path=hpo_storage,
            hpo_max_concurrent=args.hpo_max_concurrent,
            hpo_resources_gpu=args.hpo_resources_gpu,
            hpo_resume=bool(args.resume),
        )
        config = resolve_config(project_root=PROJECT_ROOT, job=job, overrides=overrides)
        tsl_hparams = validate_tsl_hparam_coverage(config, job)
        cfg_hash = config_hash(config)
        if phase_cfg['hpo']:
            print(
                '  - HPO settings: '
                f"num_samples={config.get('hpo_num_samples')} "
                f"max_concurrent={config.get('hpo_max_concurrent')} "
                f"gpu_per_trial={config.get('hpo_resources_gpu')} "
                f"patience={config.get('patience')} "
                f"disable_early_stopping={config.get('disable_early_stopping', False)}"
            )
        if config.get('max_train_samples') is not None:
            print(
                '  - Data cap: '
                f"max_train_samples={config.get('max_train_samples')}"
            )

        config_snapshot_path = job_dir / 'resolved_config.yaml'
        with config_snapshot_path.open('w', encoding='utf-8') as fh:
            yaml.safe_dump(config, fh, sort_keys=True, allow_unicode=False)

        command = build_run_command(
            python_bin=args.python_bin,
            run_script=run_script,
            config_path=(PROJECT_ROOT / job.config_path).resolve(),
            overrides=overrides,
        )
        (job_dir / 'command.txt').write_text(' '.join(command), encoding='utf-8')

        record: dict[str, Any] = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'phase': args.phase,
            'job_key': job.job_key,
            'run_group': job.run_group,
            'dataset': job.dataset,
            'model': job.model,
            'mode': job.mode,
            'entity_identifier': entity_identifier_label(job.mode),
            'seed': job.seed,
            'tsl_alignment': job.tsl_alignment,
            'tsl_hparams_checked': tsl_hparams,
            'config_hash': cfg_hash,
            'config_path': str(job.config_path),
            'run_dir': str(job_dir),
            'command': command,
            'execution_mode': 'in_process_import',
            'hpo_num_samples': (
                config.get('hpo_num_samples') if phase_cfg['hpo'] else None
            ),
            'hpo_max_concurrent': (
                config.get('hpo_max_concurrent') if phase_cfg['hpo'] else None
            ),
            'hpo_resources_gpu': (
                config.get('hpo_resources_gpu') if phase_cfg['hpo'] else None
            ),
            'disable_early_stopping': (
                config.get('disable_early_stopping', False) if phase_cfg['hpo'] else None
            ),
            'patience': config.get('patience') if phase_cfg['hpo'] else None,
            'plot_aggregations': args.plot_aggregations,
            'plot_formats': args.plot_formats,
        }

        if not phase_cfg['execute']:
            record['status'] = 'dry-run'
            _append_manifest(manifest_path, record)
            print('  - dry-run command emitted')
            continue

        summary['executed'] += 1
        log_path = job_dir / 'run.log'
        returncode, elapsed = _run_in_process(
            config_path=(PROJECT_ROOT / job.config_path).resolve(),
            overrides=overrides,
            cwd=job_dir,
            log_path=log_path,
            timeout_seconds=args.timeout_seconds,
        )
        record['elapsed_seconds'] = round(elapsed, 3)
        record['returncode'] = returncode

        if returncode != 0:
            record['status'] = 'failed'
            summary['failed'] += 1
            _append_manifest(manifest_path, record)
            print(f'  - failed with return code {returncode}')
            if args.strict:
                raise RuntimeError(f'Job failed: {job.job_key}')
            continue

        result_json_path = _find_latest_results_json(job_dir)
        if result_json_path is None:
            record['status'] = 'incomplete'
            summary['failed'] += 1
            _append_manifest(manifest_path, record)
            print('  - completed process but results.json not found')
            if args.strict:
                raise RuntimeError(f'Results missing for {job.job_key}')
            continue

        with result_json_path.open('r', encoding='utf-8') as fh:
            results = json.load(fh)
        record.update(_extract_metrics(results))
        record['results_json'] = str(result_json_path)
        record['artifacts_dir'] = results.get('artifacts_dir')

        npz_path = result_json_path.parent / 'predictions.npz'
        if npz_path.exists():
            try:
                plot_paths = create_pred_vs_true_plots(
                    npz_path=npz_path,
                    output_dir=job_dir,
                    aggregations=args.plot_aggregations,
                    formats=args.plot_formats,
                    file_stem=args.plot_file_stem,
                )
                if plot_paths:
                    # Keep a canonical single path for backward compatibility.
                    record['plot_path'] = str(plot_paths[0])
                record['plot_paths'] = [str(path) for path in plot_paths]
            except Exception as exc:  # pragma: no cover - defensive reporting
                record['plot_error'] = f'{type(exc).__name__}: {exc}'
        else:
            record['plot_error'] = 'predictions.npz not found'

        record['status'] = 'ok'
        _append_manifest(manifest_path, record)
        print('  - ok')

    if not args.skip_compare:
        report_paths = generate_reports(run_root)
        summary['summary_csv'] = str(report_paths['summary_csv'])
        summary['comparison_md'] = str(report_paths['comparison_md'])
    return summary


def _parse_args(
        parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """
    Parse command line arguments.

    If "--config" is given, parse the experiment configuration file to override.

    Args:
        parser: Argument parser to use for parsing CLI arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    args = parser.parse_args()
    if not args.config:
        return args
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != 'help'
    }
    # Read yaml file and override args with config values:
    try:
        import yaml
        config_path = Path(args.config)
        if not config_path.is_absolute():
            cwd_candidate = (Path.cwd() / config_path).resolve()
            if cwd_candidate.exists():
                config_path = cwd_candidate
            else:
                config_path = (
                    PROJECT_ROOT / 'experiments' / 'entity_identifier' / config_path
                ).resolve()
        with config_path.open('r', encoding='utf-8') as fh:
            config_dict = yaml.safe_load(fh)
        for key, value in config_dict.items():
            if value is None or not hasattr(args, key):
                continue
            # Precedence: explicit CLI args > config file > parser defaults.
            if getattr(args, key) == defaults.get(key):
                setattr(args, key, value)
    except Exception as exc:
        raise RuntimeError(f'Failed to parse config file {args.config}: {exc}') from exc
    return args


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for matrix execution.

    Returns:
        argparse.ArgumentParser: CLI parser for matrix execution.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Run Swiss1990/Traffic/Electricity x LSTM/PatchTST/DLinear '
            'entity-identifier matrix via experiments/run.py.'
        ),
    )
    exec_group = parser.add_argument_group('Execution')
    matrix_group = parser.add_argument_group('Matrix scope')
    output_group = parser.add_argument_group('Outputs & resume')
    hpo_group = parser.add_argument_group('HPO')
    data_group = parser.add_argument_group('Data capping')
    plot_group = parser.add_argument_group('Prediction plotting')

    exec_group.add_argument(
        '--config',
        type=str,
        default=None,
        help=(
            'Optional matrix-runner config YAML. '
            'When omitted, built-in CLI defaults are used. '
            'WARNING: do NOT default to a dev/smoke YAML — those cap '
            'train_epochs and max_train_samples and will silently '
            'invalidate any full-phase run.'
        ),
    )
    exec_group.add_argument(
        '--phase',
        choices=('dry', 'smoke', 'dev', 'full'),
        default='dry',
        help='Execution phase preset.',
    )
    exec_group.add_argument(
        '--timeout-seconds',
        type=int,
        default=0,
        help='Per-job timeout in seconds (0 disables timeout).',
    )
    exec_group.add_argument(
        '--max-jobs',
        type=int,
        default=None,
        help='Run only first N jobs after expansion.',
    )
    exec_group.add_argument(
        '--strict',
        action='store_true',
        help='Stop immediately on first failed/incomplete run.',
    )

    output_group.add_argument(
        '--run-tag',
        default=_timestamp_tag(),
        help='Run tag under artifacts/entity_identifier/.',
    )
    output_group.add_argument(
        '--output-root',
        default=str(PROJECT_ROOT / 'artifacts' / 'entity_identifier'),
        help='Root directory for matrix outputs.',
    )
    output_group.add_argument(
        '--resume',
        action='store_true',
        help='Skip jobs already recorded as ok in manifest.',
    )
    output_group.add_argument(
        '--skip-compare',
        action='store_true',
        help='Skip post-run comparison report generation.',
    )
    output_group.add_argument(
        '--python-bin',
        default=str(PROJECT_ROOT / '.venv' / 'bin' / 'python'),
        help='Python executable shown in saved equivalent CLI command snapshots.',
    )

    matrix_group.add_argument(
        '--datasets',
        nargs='*',
        default=None,
        help='Subset of datasets (default: all in matrix).',
    )
    matrix_group.add_argument(
        '--models',
        nargs='*',
        default=None,
        help='Subset of models (default: all in matrix).',
    )
    matrix_group.add_argument(
        '--modes',
        nargs='*',
        default=None,
        help=f'Subset of modes (default: {", ".join(MODES)}).',
    )
    matrix_group.add_argument(
        '--seeds',
        nargs='*',
        type=int,
        default=list(DEFAULT_SEEDS),
        help='Seeds for matrix expansion.',
    )

    hpo_group.add_argument(
        '--hpo-num-samples',
        type=int,
        default=None,
        help='Override hpo_num_samples (default for full phase: 10).',
    )
    hpo_group.add_argument(
        '--hpo-max-concurrent',
        type=int,
        default=None,
        help='Override max concurrent Ray trials (auto policy when unset).',
    )
    hpo_group.add_argument(
        '--hpo-resources-gpu',
        type=float,
        default=None,
        help='Override GPU fraction per Ray trial (auto policy when unset).',
    )
    hpo_group.add_argument(
        '--train-epochs',
        type=int,
        default=None,
        help='Override train_epochs for all jobs.',
    )

    data_group.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Cap training datapoints per job for fast pipeline checks.',
    )

    plot_group.add_argument(
        '--plot-aggregations',
        nargs='*',
        default=['mean', 'median', 'best', 'last'],
        help=(
            'Aggregation methods for pred-vs-gt plots. '
            'Supported: '
            f"{', '.join(SUPPORTED_PLOT_AGGREGATIONS)}."
        ),
    )
    plot_group.add_argument(
        '--plot-formats',
        nargs='*',
        default=['png'],
        help='Plot formats to export (e.g., png svg pdf).',
    )
    plot_group.add_argument(
        '--plot-file-stem',
        default='pred_vs_gt',
        help='Base filename prefix for generated prediction plots.',
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = _parse_args(parser)
    summary = run_matrix(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
