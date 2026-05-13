"""Aggregate entity-identifier matrix run artifacts into comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.entity_identifier.matrix import TSL_ALIGNMENT_STATUS, entity_identifier_label


def _to_float(value: Any) -> float | None:
    """Convert a value to ``float`` with ``None`` fallback."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float | None]) -> float | None:
    """Compute mean over non-null values."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _std(values: list[float | None]) -> float | None:
    """Compute population std deviation over non-null values."""
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None
    mu = sum(valid) / len(valid)
    return (sum((x - mu) ** 2 for x in valid) / len(valid)) ** 0.5


def _fmt_num(value: float | None, digits: int = 6) -> str:
    """Format optional float with fixed precision."""
    if value is None:
        return 'N/A'
    return f'{value:.{digits}f}'


def _fmt_mean_std(
    mean: float | None, std: float | None, digits: int = 6
) -> str:
    """Format mean +/- std, falling back to mean-only when std is absent."""
    if mean is None:
        return 'N/A'
    if std is None:
        return f'{mean:.{digits}f}'
    return f'{mean:.{digits}f} +/- {std:.{digits}f}'


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    """Format optional percentage value."""
    if value is None:
        return 'N/A'
    return f'{value:.{digits}f}%'


def _latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the latest manifest record for each job key."""
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get('job_key', ''))
        if not key:
            continue
        by_key[key] = record
    return list(by_key.values())


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load jsonl run records from manifest."""
    records: list[dict[str, Any]] = []
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
            if isinstance(record, dict):
                records.append(record)
    return records


def _build_mode_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-job records into per-mode comparison rows."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        dataset = str(record.get('dataset', ''))
        model = str(record.get('model', ''))
        mode = str(record.get('mode', ''))
        if not dataset or not model or not mode:
            continue
        grouped[(dataset, model, mode)].append(record)

    pair_keys = sorted(
        {(dataset, model) for dataset, model, _ in grouped},
        key=lambda item: (item[0], item[1]),
    )
    rows: list[dict[str, Any]] = []
    preferred_mode_order = {
        'none': 0,
        'embedding': 1,
        'onehot': 2,
        'coordinates': 3,
        'sinusoidal': 4,
        'random': 5,
    }

    for dataset, model in pair_keys:
        baseline_none = [
            r
            for r in grouped.get((dataset, model, 'none'), [])
            if str(r.get('status', '')).lower() == 'ok'
        ]
        baseline_none_mse = _mean([_to_float(r.get('test_mse')) for r in baseline_none])

        pair_modes = sorted(
            {mode for (ds, md, mode) in grouped if ds == dataset and md == model},
            key=lambda mode: (
                preferred_mode_order.get(mode.lower(), 99),
                entity_identifier_label(mode),
            ),
        )

        for mode in pair_modes:
            ok_records = [
                r
                for r in grouped.get((dataset, model, mode), [])
                if str(r.get('status', '')).lower() == 'ok'
            ]
            mse_vals = [_to_float(r.get('test_mse')) for r in ok_records]
            rmse_vals = [_to_float(r.get('test_rmse')) for r in ok_records]
            mae_vals = [_to_float(r.get('test_mae')) for r in ok_records]
            nse_vals = [_to_float(r.get('test_nse')) for r in ok_records]

            mse = _mean(mse_vals)
            rmse = _mean(rmse_vals)
            mae = _mean(mae_vals)
            nse = _mean(nse_vals)
            hpo_best = _mean([_to_float(r.get('hpo_best_value')) for r in ok_records])

            mse_std = _std(mse_vals)
            rmse_std = _std(rmse_vals)
            mae_std = _std(mae_vals)
            nse_std = _std(nse_vals)

            delta_mse_vs_none = None
            delta_mse_vs_none_pct = None
            if mse is not None and baseline_none_mse is not None:
                delta_mse_vs_none = mse - baseline_none_mse
                if abs(baseline_none_mse) > 1e-12:
                    delta_mse_vs_none_pct = 100.0 * delta_mse_vs_none / baseline_none_mse
                elif mode.lower() == 'none':
                    delta_mse_vs_none_pct = 0.0

            rows.append(
                {
                    'dataset': dataset,
                    'model': model,
                    'mode': mode,
                    'entity_identifier': entity_identifier_label(mode),
                    'tsl_alignment': TSL_ALIGNMENT_STATUS.get(
                        (dataset, model), 'no canonical TSL comparison'
                    ),
                    'n_runs': len(ok_records),
                    'mse': mse,
                    'mse_std': mse_std,
                    'rmse': rmse,
                    'rmse_std': rmse_std,
                    'mae': mae,
                    'mae_std': mae_std,
                    'nse': nse,
                    'nse_std': nse_std,
                    'hpo_best': hpo_best,
                    'delta_mse_vs_none': delta_mse_vs_none,
                    'delta_mse_vs_none_pct': delta_mse_vs_none_pct,
                }
            )
    return rows


def _write_summary_csv(records: list[dict[str, Any]], out_csv: Path) -> None:
    """Write flat per-job summary CSV for downstream analysis."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'job_key',
        'dataset',
        'model',
        'mode',
        'entity_identifier',
        'seed',
        'status',
        'elapsed_seconds',
        'tsl_alignment',
        'test_mse',
        'test_rmse',
        'test_mae',
        'test_nse',
        'hpo_best_value',
        'hpo_trials',
        'hpo_num_samples',
        'hpo_max_concurrent',
        'hpo_resources_gpu',
        'disable_early_stopping',
        'patience',
        'plot_path',
        'results_json',
        'run_dir',
    ]
    with out_csv.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            row['entity_identifier'] = entity_identifier_label(str(record.get('mode', '')))
            writer.writerow(row)


def _write_pair_markdown(
    records: list[dict[str, Any]],
    mode_rows: list[dict[str, Any]],
    out_md: Path,
) -> None:
    """Write human-readable markdown report from aggregate rows."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    status_counter = Counter(str(record.get('status', 'unknown')) for record in records)
    failed = [
        record
        for record in records
        if str(record.get('status', '')).lower() in {'failed', 'incomplete'}
    ]

    lines: list[str] = []
    lines.append('# Entity Identifier Comparison')
    lines.append('')
    lines.append('## Status Counts')
    lines.append('')
    lines.append('| Status | Count |')
    lines.append('|---|---:|')
    for status, count in sorted(status_counter.items()):
        lines.append(f'| {status} | {count} |')
    lines.append('')

    lines.append('## Mode-Level Comparison')
    lines.append('')
    lines.append(
        '| Dataset | Model | Entity identifier | TSL alignment | n_runs | '
        'MSE | RMSE | MAE | NSE | HPO best | dMSE vs none | dMSE vs none % |'
    )
    lines.append(
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'
    )
    for row in mode_rows:
        lines.append(
            '| '
            + f"{row['dataset']} | {row['model']} | {row['entity_identifier']} | "
            + f"{row['tsl_alignment']} | {row['n_runs']} | "
            + f"{_fmt_mean_std(row['mse'], row.get('mse_std'))} | "
            + f"{_fmt_mean_std(row['rmse'], row.get('rmse_std'))} | "
            + f"{_fmt_mean_std(row['mae'], row.get('mae_std'))} | "
            + f"{_fmt_mean_std(row['nse'], row.get('nse_std'))} | "
            + f"{_fmt_num(row['hpo_best'])} | "
            + f"{_fmt_num(row['delta_mse_vs_none'])} | "
            + f"{_fmt_pct(row['delta_mse_vs_none_pct'])} |"
        )
    lines.append('')

    lines.append('## Failed / Incomplete Runs')
    lines.append('')
    if failed:
        lines.append('| Job key | Status | Return code |')
        lines.append('|---|---|---:|')
        for record in failed:
            lines.append(
                f"| {record.get('job_key')} | {record.get('status')} | "
                f"{record.get('returncode', 'N/A')} |"
            )
    else:
        lines.append('None.')
    lines.append('')

    out_md.write_text('\n'.join(lines), encoding='utf-8')


def generate_reports(run_root: Path) -> dict[str, Path]:
    """Generate summary CSV and markdown comparison from a run root."""
    manifest_path = run_root / 'manifest.jsonl'
    latest = _latest_records(load_manifest(manifest_path))
    mode_rows = _build_mode_rows(latest)

    summary_csv = run_root / 'summary.csv'
    comparison_md = run_root / 'comparison.md'
    _write_summary_csv(latest, summary_csv)
    _write_pair_markdown(latest, mode_rows, comparison_md)
    return {
        'summary_csv': summary_csv,
        'comparison_md': comparison_md,
        'manifest': manifest_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for comparison report generation."""
    parser = argparse.ArgumentParser(
        description='Generate entity-identifier matrix comparison tables.',
    )
    parser.add_argument(
        '--run-root',
        required=True,
        help='Run root directory (e.g. artifacts/entity_identifier/<run_tag>).',
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    outputs = generate_reports(run_root)
    print(f"Summary CSV: {outputs['summary_csv']}")
    print(f"Comparison MD: {outputs['comparison_md']}")


if __name__ == '__main__':
    main()
