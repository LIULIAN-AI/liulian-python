#!/usr/bin/env python3
"""Aggregate experiment results and generate benchmark report.

Usage::

    # Generate markdown report
    python tools/aggregate_results.py --results-dir experiments/results/

    # Generate LaTeX tables
    python tools/aggregate_results.py --results-dir experiments/results/ --format latex

    # Output to specific file
    python tools/aggregate_results.py --results-dir experiments/results/ --output docs/benchmark_results.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / 'experiments' / 'results'


def load_results(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all JSON result files grouped by experiment group."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for json_file in sorted(results_dir.rglob('*.json')):
        group = json_file.parent.name
        try:
            with open(json_file) as f:
                data = json.load(f)
            data['_file'] = str(json_file)
            groups[group].append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f'Warning: Cannot load {json_file}: {e}', file=sys.stderr)
    return dict(groups)


def aggregate_by_model_dataset(
    results: List[Dict[str, Any]],
    metric: str = 'mse',
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Aggregate results for each (model, dataset) pair across seeds.

    Returns {(model, dataset): {"mean": ..., "std": ..., "n": ...}}
    """
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in results:
        if r.get('status') != 'success':
            continue
        model = r.get('model', '?')
        dataset = r.get('dataset', '?')
        metrics = r.get('metrics', {})
        test_metrics = metrics.get('test', r.get('final_test', {}))
        val = test_metrics.get(metric)
        if val is not None and np.isfinite(val):
            buckets[(model, dataset)].append(val)

    agg = {}
    for key, values in buckets.items():
        agg[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'n': len(values),
        }
    return agg


def aggregate_entity_ablation(
    results: List[Dict[str, Any]],
    metric: str = 'mse',
) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Aggregate entity ablation results by (model, dataset, mode)."""
    buckets: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in results:
        if r.get('status') != 'success':
            continue
        model = r.get('model', '?')
        dataset = r.get('dataset', '?')
        cfg = r.get('config', {})
        mode = cfg.get('identifier_mode', 'none')
        metrics = r.get('metrics', {})
        test_metrics = metrics.get('test', r.get('final_test', {}))
        val = test_metrics.get(metric)
        if val is not None and np.isfinite(val):
            buckets[(model, dataset, mode)].append(val)

    agg = {}
    for key, values in buckets.items():
        agg[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'n': len(values),
        }
    return agg


# ======================================================================
# Report generators
# ======================================================================


def _fmt(val: float, fmt: str = '.4f') -> str:
    """Format a float value."""
    if np.isnan(val) or np.isinf(val):
        return '—'
    return f'{val:{fmt}}'


def generate_markdown_report(
    groups: Dict[str, List[Dict[str, Any]]],
    output_path: Optional[Path] = None,
) -> str:
    """Generate a full markdown benchmark report."""
    lines: List[str] = []
    lines.append('# Liulian Benchmark Results\n')
    lines.append(
        f'*Generated from {sum(len(v) for v in groups.values())} result files*\n'
    )

    # --- Summary ---
    lines.append('## Summary\n')
    lines.append('| Experiment Group | Runs | Success | Failed |')
    lines.append('|------------------|------|---------|--------|')
    for gname, gresults in sorted(groups.items()):
        n_success = sum(1 for r in gresults if r.get('status') == 'success')
        n_failed = sum(1 for r in gresults if r.get('status') != 'success')
        lines.append(f'| {gname} | {len(gresults)} | {n_success} | {n_failed} |')
    lines.append('')

    # --- Long-term forecasting ---
    if 'long_term' in groups:
        lines.append('## Core Benchmark: Long-term Forecasting (E.2.1)\n')
        for metric in ['mse', 'mae']:
            lines.append(f'### {metric.upper()}\n')
            agg = aggregate_by_model_dataset(groups['long_term'], metric)
            lines.extend(_model_dataset_table(agg, metric))
            lines.append('')

    # --- Entity ablation ---
    if 'entity' in groups:
        lines.append('## Entity Identifier Ablation (E.2.2)\n')
        agg = aggregate_entity_ablation(groups['entity'], 'mse')
        lines.extend(_entity_table(agg))
        lines.append('')

    # --- Nowcasting ---
    if 'nowcasting' in groups:
        lines.append('## Nowcasting — Swiss-river (E.2.3)\n')
        for metric in ['mse', 'nse']:
            lines.append(f'### {metric.upper()}\n')
            agg = aggregate_by_model_dataset(groups['nowcasting'], metric)
            lines.extend(_model_dataset_table(agg, metric))
            lines.append('')

    # --- M4 ---
    if 'm4' in groups:
        lines.append('## Short-term Forecasting — M4 (E.2.4)\n')
        for metric in ['smape', 'mase']:
            agg = aggregate_by_model_dataset(groups['m4'], metric)
            if agg:
                lines.append(f'### {metric.upper()}\n')
                lines.extend(_model_dataset_table(agg, metric))
                lines.append('')

    # --- Spatial ---
    if 'spatial' in groups:
        lines.append('## Spatial-Temporal Forecasting (E.2.5)\n')
        agg = aggregate_by_model_dataset(groups['spatial'], 'mse')
        lines.extend(_model_dataset_table(agg, 'mse'))
        lines.append('')

    # --- Ablation studies ---
    for abl_name, abl_title in [
        ('ablation_norm', 'Normalization Effect (E.2.6A)'),
        ('ablation_aug', 'Augmentation Effect (E.2.6B)'),
        ('ablation_seqlen', 'Input Length Sensitivity (E.2.6C)'),
        ('ablation_tf', 'Teacher Forcing Effect (E.2.6D)'),
    ]:
        if abl_name in groups:
            lines.append(f'## {abl_title}\n')
            agg = aggregate_by_model_dataset(groups[abl_name], 'mse')
            lines.extend(_model_dataset_table(agg, 'mse'))
            lines.append('')

    # --- Failed experiments ---
    all_failed = []
    for gname, gresults in groups.items():
        for r in gresults:
            if r.get('status') != 'success':
                all_failed.append((gname, r))
    if all_failed:
        lines.append('## Failed Experiments\n')
        lines.append('| Group | Model | Dataset | Seed | Error |')
        lines.append('|-------|-------|---------|------|-------|')
        for gname, r in all_failed[:50]:  # Limit to 50
            lines.append(
                f'| {gname} | {r.get("model", "?")} | {r.get("dataset", "?")} | '
                f'{r.get("seed", "?")} | {str(r.get("error", "?"))[:80]} |'
            )
        if len(all_failed) > 50:
            lines.append(f'\n*... and {len(all_failed) - 50} more*\n')
        lines.append('')

    report = '\n'.join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f'Report written to {output_path}')

    return report


def _model_dataset_table(
    agg: Dict[Tuple[str, str], Dict[str, float]],
    metric: str,
) -> List[str]:
    """Generate a markdown table from (model, dataset) aggregation."""
    if not agg:
        return ['*No data available*\n']

    models = sorted(set(m for m, _ in agg))
    datasets = sorted(set(d for _, d in agg))

    lines = []
    header = '| Model | ' + ' | '.join(datasets) + ' |'
    sep = '|-------|' + '|'.join(['-------'] * len(datasets)) + '|'
    lines.append(header)
    lines.append(sep)

    # Find best per dataset for bolding
    best_per_ds: Dict[str, float] = {}
    for ds in datasets:
        vals = [(m, agg[(m, ds)]['mean']) for m in models if (m, ds) in agg]
        if vals:
            if metric in ('nse',):
                best_per_ds[ds] = max(v for _, v in vals)
            else:
                best_per_ds[ds] = min(v for _, v in vals)

    for model in models:
        cells = []
        for ds in datasets:
            if (model, ds) in agg:
                val = agg[(model, ds)]['mean']
                std = agg[(model, ds)]['std']
                cell = _fmt(val)
                if std > 0:
                    cell += f' ±{_fmt(std, ".3f")}'
                # Bold best
                is_best = ds in best_per_ds and abs(val - best_per_ds[ds]) < 1e-8
                if is_best:
                    cell = f'**{cell}**'
                cells.append(cell)
            else:
                cells.append('—')
        lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')

    return lines


def _entity_table(
    agg: Dict[Tuple[str, str, str], Dict[str, float]],
) -> List[str]:
    """Generate entity ablation table."""
    if not agg:
        return ['*No data available*\n']

    models = sorted(set(m for m, _, _ in agg))
    datasets = sorted(set(d for _, d, _ in agg))
    modes = sorted(set(mode for _, _, mode in agg))

    lines = []
    for ds in datasets:
        lines.append(f'\n### {ds}\n')
        header = '| Model | ' + ' | '.join(modes) + ' |'
        sep = '|-------|' + '|'.join(['-------'] * len(modes)) + '|'
        lines.append(header)
        lines.append(sep)

        for model in models:
            cells = []
            for mode in modes:
                if (model, ds, mode) in agg:
                    val = agg[(model, ds, mode)]['mean']
                    cells.append(_fmt(val))
                else:
                    cells.append('—')
            lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')

    return lines


def generate_latex_tables(
    groups: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Generate LaTeX table files for academic papers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'long_term' in groups:
        for metric in ['mse', 'mae']:
            agg = aggregate_by_model_dataset(groups['long_term'], metric)
            if agg:
                latex = _to_latex(agg, metric, f'Core Benchmark — {metric.upper()}')
                with open(output_dir / f'long_term_{metric}.tex', 'w') as f:
                    f.write(latex)

    if 'entity' in groups:
        agg = aggregate_entity_ablation(groups['entity'], 'mse')
        if agg:
            # One table per dataset
            datasets = sorted(set(d for _, d, _ in agg))
            for ds in datasets:
                sub_agg = {(m, mode): v for (m, d, mode), v in agg.items() if d == ds}
                latex = _to_latex_entity(sub_agg, ds)
                with open(output_dir / f'entity_{ds.lower()}.tex', 'w') as f:
                    f.write(latex)

    print(f'LaTeX tables written to {output_dir}')


def _to_latex(
    agg: Dict[Tuple[str, str], Dict[str, float]],
    metric: str,
    caption: str,
) -> str:
    """Convert model×dataset aggregation to LaTeX table."""
    models = sorted(set(m for m, _ in agg))
    datasets = sorted(set(d for _, d in agg))

    n_cols = len(datasets) + 1
    lines = []
    lines.append(f'\\begin{{table}}[htbp]')
    lines.append(f'\\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\begin{{tabular}}{{{"l" + "c" * len(datasets)}}}')
    lines.append(f'\\toprule')
    lines.append('Model & ' + ' & '.join(datasets) + ' \\\\')
    lines.append('\\midrule')

    # Find best per dataset
    best_per_ds: Dict[str, float] = {}
    for ds in datasets:
        vals = [agg[(m, ds)]['mean'] for m in models if (m, ds) in agg]
        if vals:
            best_per_ds[ds] = min(vals)

    for model in models:
        cells = []
        for ds in datasets:
            if (model, ds) in agg:
                val = agg[(model, ds)]['mean']
                cell = _fmt(val)
                if ds in best_per_ds and abs(val - best_per_ds[ds]) < 1e-8:
                    cell = f'\\textbf{{{cell}}}'
                cells.append(cell)
            else:
                cells.append('—')
        lines.append(f'{model} & ' + ' & '.join(cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append(f'\\end{{tabular}}')
    lines.append(f'\\end{{table}}')
    return '\n'.join(lines)


def _to_latex_entity(
    agg: Dict[Tuple[str, str], Dict[str, float]],
    dataset: str,
) -> str:
    """Convert entity ablation to LaTeX table for one dataset."""
    models = sorted(set(m for m, _ in agg))
    modes = sorted(set(mode for _, mode in agg))

    lines = []
    lines.append(f'\\begin{{table}}[htbp]')
    lines.append(f'\\centering')
    lines.append(f'\\caption{{Entity identifier ablation on {dataset}}}')
    lines.append(f'\\begin{{tabular}}{{{"l" + "c" * len(modes)}}}')
    lines.append(f'\\toprule')
    lines.append('Model & ' + ' & '.join(modes) + ' \\\\')
    lines.append('\\midrule')

    for model in models:
        cells = []
        for mode in modes:
            if (model, mode) in agg:
                cells.append(_fmt(agg[(model, mode)]['mean']))
            else:
                cells.append('—')
        lines.append(f'{model} & ' + ' & '.join(cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append(f'\\end{{tabular}}')
    lines.append(f'\\end{{table}}')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate benchmark results')
    parser.add_argument(
        '--results-dir',
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help='Directory containing result JSON files',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: stdout for markdown)',
    )
    parser.add_argument(
        '--format',
        type=str,
        default='markdown',
        choices=['markdown', 'latex'],
        help='Output format',
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f'Error: Results directory {results_dir} does not exist', file=sys.stderr)
        sys.exit(1)

    groups = load_results(results_dir)
    if not groups:
        print('No result files found', file=sys.stderr)
        sys.exit(1)

    print(
        f'Loaded results: {", ".join(f"{k}({len(v)})" for k, v in sorted(groups.items()))}'
    )

    if args.format == 'latex':
        output_dir = Path(args.output) if args.output else results_dir / 'tables'
        generate_latex_tables(groups, output_dir)
    else:
        output_path = Path(args.output) if args.output else None
        report = generate_markdown_report(groups, output_path)
        if not output_path:
            print(report)


if __name__ == '__main__':
    main()
