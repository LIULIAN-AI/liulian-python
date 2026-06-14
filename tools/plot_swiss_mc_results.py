#!/usr/bin/env python3
"""Plot swiss multi_channel (dlinear/patchtst) entity-identifier results.

Reads denorm test RMSE from real results.json (no hardcoded numbers) and
emits one grouped-bar figure per model (3 datasets × 6 modes) with the
`none` baseline drawn as a reference line. Also writes a CSV.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'entity_identifier'
OUT = ROOT / 'docs' / 'research' / 'figures' / 'swiss-mc-2026-06-14'
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ['swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich']
MODELS = ['dlinear', 'patchtst']
MODES = ['none', 'embedding', 'onehot', 'random', 'sinusoidal', 'coordinates']
TAG = {d: f'swiss-mc-{d.split("-")[-1]}-20260614' for d in DATASETS}
SHORT = {'swiss-river-1990': '1990 (28)', 'swiss-river-2010': '2010 (63)',
         'swiss-river-zurich': 'zurich (15)'}
COLORS = {'none': '#9aa0a6', 'embedding': '#1a73e8', 'onehot': '#34a853',
          'random': '#a142f4', 'sinusoidal': '#ea8600', 'coordinates': '#d93025'}


def rmse(tag, ds, model, mode):
    hits = glob.glob(f'{ART}/{tag}/{ds}-{model}-{mode}-*/**/results.json', recursive=True)
    if not hits:
        return None
    m = json.load(open(hits[0]))['metrics']['test']
    return float(m.get('denorm_rmse', m.get('rmse')))


data = {model: {ds: {mode: rmse(TAG[ds], ds, model, mode) for mode in MODES}
                for ds in DATASETS} for model in MODELS}

with open(OUT / 'swiss-mc-rmse.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['model', 'dataset', *MODES])
    for model in MODELS:
        for ds in DATASETS:
            cells = data[model][ds]
            row = [f'{cells[m]:.4f}' if cells[m] is not None else '' for m in MODES]
            w.writerow([model, ds, *row])
print('wrote', OUT / 'swiss-mc-rmse.csv')

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
for ax, model in zip(axes, MODELS):
    n = len(MODES)
    bw = 0.82 / n
    for j, mode in enumerate(MODES):
        xs = [i + (j - n / 2 + 0.5) * bw for i in range(len(DATASETS))]
        ys = [data[model][ds][mode] for ds in DATASETS]
        ax.bar(xs, ys, bw, label=mode, color=COLORS[mode], edgecolor='white', linewidth=0.5)
    # none reference line per dataset (thin) — visual anchor
    for i, ds in enumerate(DATASETS):
        nb = data[model][ds]['none']
        ax.plot([i - 0.45, i + 0.45], [nb, nb], color='#444', lw=0.8, ls='--', alpha=0.6)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([SHORT[d] for d in DATASETS])
    ax.set_title(f'{model}  (multi_channel)')
    ax.set_ylabel('Test RMSE (°C)')
    ax.grid(axis='y', alpha=0.3)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axes[0].legend(ncol=6, fontsize=8, loc='upper center', bbox_to_anchor=(1.05, -0.12), frameon=False)
fig.suptitle('swiss-river × {dlinear, patchtst}: entity-identifier modes '
             '(single seed 2026, 50-trial HPO; dashed = none baseline)', fontsize=11)
fig.tight_layout()
fig.savefig(OUT / 'swiss-mc-rmse-by-mode.png', dpi=150, bbox_inches='tight')
print('wrote', OUT / 'swiss-mc-rmse-by-mode.png')
