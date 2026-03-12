#!/usr/bin/env python3
"""Merge new ILI results into the full comparison results JSON."""
import json

ROOT = "/media/linlin/New Volume/projects/2026.02.07_liulian/codes/liulian-python"

# Read the original full results (16 matched + 2 old ILI)
with open(f'{ROOT}/experiments/adapt_tsl_lib/tsl_comparison_results.json') as f:
    original = json.load(f)

# Read the new ILI results
with open(f'{ROOT}/artifacts/tsl_comparison_results.json') as f:
    new_ili = json.load(f)

# Merge: replace ILI entries in original with new ones
new_names = {e['name'] for e in new_ili}
merged = [e for e in original if e['name'] not in new_names] + new_ili

# Write merged back to both locations
for path in [
    f'{ROOT}/artifacts/tsl_comparison_results.json',
    f'{ROOT}/experiments/adapt_tsl_lib/tsl_comparison_results.json',
]:
    with open(path, 'w') as f:
        json.dump(merged, f, indent=2)

print(f'Merged {len(merged)} results')
for e in merged:
    print(f"  {e['name']}: {e['status']}")
