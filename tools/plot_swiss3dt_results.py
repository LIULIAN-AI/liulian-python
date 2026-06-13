#!/usr/bin/env python3
"""Plot the swiss3dt entity-identifier results from real results.json files.

Reads denorm test RMSE for every (dataset, mode) cell straight off disk —
no hardcoded numbers (code-verifier: numbers must come from the run, not a
literal). Emits two figures + a machine-readable CSV next to the docs.

  fig1  swiss3dt-rmse-by-mode.png   grouped bars, 3 datasets x 6 modes
  fig2  swiss3dt-coordinates-fix.png  old (zero-vector) vs new coordinates
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
OUT = ROOT / 'docs' / 'research' / 'figures' / 'swiss3dt-2026-06-13'
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ['swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich']
MODES = ['none', 'embedding', 'onehot', 'random', 'sinusoidal', 'coordinates']
NEW_TAG = {'swiss-river-1990': 'swiss3dt-1990-20260612',
           'swiss-river-2010': 'swiss3dt-2010-20260612',
           'swiss-river-zurich': 'swiss3dt-zurich-20260612'}
# The 2026-05-15 advisor slide's numbers come from this run (its onehot
# 1.1717 matches the slide's 1.171), NOT the earlier under-trained
# fullmatrix-0511 (none 3.88). Use it so the coordinate comparison is on the
# same footing as the slide.
OLD_TAG = 'swissriver-lstm-REAL-20260512-084128'  # swiss-river-1990 only


def rmse(tag: str, ds: str, mode: str) -> float | None:
    hits = glob.glob(f'{ART}/{tag}/{ds}-lstm-{mode}-*/**/results.json', recursive=True)
    if not hits:
        return None
    m = json.load(open(hits[0]))['metrics']['test']
    return float(m.get('denorm_rmse', m.get('rmse')))


# ── gather ──────────────────────────────────────────────────────────────
new = {ds: {mode: rmse(NEW_TAG[ds], ds, mode) for mode in MODES} for ds in DATASETS}

csv_path = OUT / 'swiss3dt-rmse.csv'
with open(csv_path, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['dataset', *MODES])
    for ds in DATASETS:
        w.writerow([ds, *[f'{new[ds][m]:.4f}' if new[ds][m] is not None else '' for m in MODES]])
print('wrote', csv_path)

# ── fig 1: grouped bars ─────────────────────────────────────────────────
COLORS = {
    'none': '#9aa0a6', 'embedding': '#1a73e8', 'onehot': '#34a853',
    'random': '#a142f4', 'sinusoidal': '#ea8600', 'coordinates': '#d93025',
}
fig, ax = plt.subplots(figsize=(9, 4.8))
n_modes = len(MODES)
group_w = 0.82
bar_w = group_w / n_modes
short = {'swiss-river-1990': '1990 (28 st.)', 'swiss-river-2010': '2010 (63 st.)',
         'swiss-river-zurich': 'zurich (15 st.)'}
for j, mode in enumerate(MODES):
    xs = [i + (j - n_modes / 2 + 0.5) * bar_w for i in range(len(DATASETS))]
    ys = [new[ds][mode] for ds in DATASETS]
    ax.bar(xs, ys, bar_w, label=mode, color=COLORS[mode], edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels([short[d] for d in DATASETS])
ax.set_ylabel('Test RMSE (°C, denormalized)')
ax.set_title('swiss-river × LSTM: entity-identifier modes (single seed 2026, 50-trial HPO)')
ax.legend(ncol=6, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
ax.grid(axis='y', alpha=0.3)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'swiss3dt-rmse-by-mode.png', dpi=150, bbox_inches='tight')
print('wrote', OUT / 'swiss3dt-rmse-by-mode.png')

# ── fig 2: coordinates old vs new (1990 only) ───────────────────────────
old_coord = rmse(OLD_TAG, 'swiss-river-1990', 'coordinates')
old_none = rmse(OLD_TAG, 'swiss-river-1990', 'none')
new_coord = new['swiss-river-1990']['coordinates']
new_none = new['swiss-river-1990']['none']

fig2, (axL, axR) = plt.subplots(1, 2, figsize=(8, 4.2))
for ax, (title, none_v, coord_v, sub) in zip(
    (axL, axR),
    [('OLD (2026-05-12)\nzero-vector bug', old_none, old_coord, 'coord worse than none (+%.1f%%)'),
     ('NEW (2026-06-13)\nreal + normalized', new_none, new_coord, 'coord beats none (%.1f%%)')],
):
    delta = (coord_v - none_v) / none_v * 100.0
    bars = ax.bar(['none', 'coordinates'], [none_v, coord_v],
                  color=['#9aa0a6', '#d93025'], edgecolor='white')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Test RMSE (°C)')
    ax.set_ylim(0, max(none_v, coord_v) * 1.30)  # headroom for the annotation
    ax.text(0.5, 0.95, sub % delta, transform=ax.transAxes, ha='center', fontsize=9,
            color=('#d93025' if delta > 0 else '#188038'))
    for b, v in zip(bars, [none_v, coord_v]):
        ax.text(b.get_x() + b.get_width() / 2, v + max(none_v, coord_v) * 0.02,
                f'{v:.3f}', ha='center', fontsize=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig2.suptitle('Coordinates identifier — swiss-river-1990 × LSTM', fontsize=11)
fig2.tight_layout()
fig2.savefig(OUT / 'swiss3dt-coordinates-fix.png', dpi=150, bbox_inches='tight')
print('wrote', OUT / 'swiss3dt-coordinates-fix.png')
print(f'OLD coord/none = {old_coord:.3f}/{old_none:.3f}  '
      f'NEW coord/none = {new_coord:.3f}/{new_none:.3f}')
