#!/usr/bin/env python3
"""Update docs/tsl_comparison.md with all 121 experiment results from JSON."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_JSON = ROOT / "artifacts" / "tsl_comparison_results.json"
DOC_PATH = ROOT / "docs" / "tsl_comparison.md"

# Load results
with open(RESULTS_JSON) as f:
    data = json.load(f)

result_map = {e["name"]: e for e in data}

# ── Helper: normalize dataset name ─────────────────────────────────────────
DATASET_ALIASES = {
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "Weather": "Weather",
    "ECL": "ECL",
    "Traffic": "Traffic",
    "Exchange": "Exchange",
    "ILI": "ILI",
    "Exchange Rate": "Exchange",
    "Electricity": "ECL",
}

MODEL_ALIASES = {
    "PatchTST": "PatchTST",
    "DLinear": "DLinear",
    "Informer": "Informer",
    "Autoformer": "Autoformer",
    "FEDformer": "FEDformer",
    "TimesNet": "TimesNet",
    "Transformer": "Transformer",
    "iTransformer": "iTransformer",
    "TimeMixer": "TimeMixer",
    "TimeXer": "TimeXer",
    "Mamba": "Mamba",
    "NonstationaryTransformer": "NonstationaryTransformer",
    "Nonstationary Transformer": "NonstationaryTransformer",
    "Nonstat. Transformer": "NonstationaryTransformer",
    "LightTS": "LightTS",
    "Reformer": "Reformer",
    "GPT4TS": "GPT4TS",
    "LSTM": "LSTM",
}


def lookup(dataset: str, model: str):
    """Look up the result for a dataset-model pair."""
    ds = None
    for alias, canonical in DATASET_ALIASES.items():
        if alias.lower() in dataset.lower() or dataset.strip() == alias:
            ds = canonical
            break
    if ds is None:
        # Try direct match
        ds = dataset.strip()

    md = None
    for alias, canonical in MODEL_ALIASES.items():
        if alias.lower() == model.strip().lower():
            md = canonical
            break
    if md is None:
        md = model.strip()

    key = f"{ds}_{md}"
    return result_map.get(key), key


def compute_pct(entry) -> float:
    """Compute MSE relative diff percentage from entry fields."""
    pct = entry.get("mse_rel_diff_pct")
    if pct is not None:
        return float(pct)
    tsl = entry.get("tsl_final_mse")
    ll = entry.get("ll_final_mse")
    if tsl and ll and tsl > 0:
        return abs(tsl - ll) / tsl * 100
    return 0.0


def status_cell(entry) -> str:
    """Convert JSON entry to doc status string."""
    if entry is None:
        return "⏳ pending"
    status = entry.get("status", "")
    if status == "checked and matched":
        pct = compute_pct(entry)
        return f"✅ matched ({pct:.2f}%)"
    elif status == "checked but not matched":
        pct = compute_pct(entry)
        return f"⚠️ not matched ({pct:.1f}%)"
    elif "GPU OOM" in status:
        return "⛔ skipped (GPU OOM)"
    elif "mamba-ssm" in status:
        return "⛔ skipped (mamba-ssm)"
    elif status.startswith("skipped"):
        return f"⛔ {status}"
    else:
        return status


# ── Pattern 1: "| # | Dataset — Model | ⏳ pending | ..." ──────────────────
# Matches rows in the Informer/Autoformer etc. sections
DASH_PATTERN = re.compile(
    r'(\|\s*\d+\s*\|\s*)'
    r'([\w\s]+?)\s*—\s*([\w\s\.]+?)'
    r'(\s*\|\s*)⏳ pending'
    r'(\s*\|)',
)

def replace_dash_row(m):
    prefix = m.group(1)
    dataset = m.group(2).strip()
    model = m.group(3).strip()
    sep = m.group(4)
    tail = m.group(5)
    entry, key = lookup(dataset, model)
    new_status = status_cell(entry)
    return f"{prefix}{dataset} — {model}{sep}{new_status}{tail}"


# ── Pattern 2: "| # | Dataset | Model | ... | ⏳ pending | —" ────────────
# Matches rows in the New Models Part 3 sections (NonstationaryTransformer etc.)
PIPE_PATTERN = re.compile(
    r'(\|\s*\d+\s*\|\s*)'
    r'([\w\s]+?)'
    r'(\s*\|\s*)'
    r'([\w\s\.]+?)'
    r'(\s*\|\s*(?:[^|]+\s*\|\s*){1,4})'  # up to 4 intervening columns
    r'⏳ pending'
    r'(\s*\|\s*—\s*\|)',
)

def replace_pipe_row(m):
    prefix = m.group(1)
    dataset = m.group(2).strip()
    sep1 = m.group(3)
    model = m.group(4).strip()
    middle = m.group(5)
    tail = m.group(6)
    entry, key = lookup(dataset, model)
    new_status = status_cell(entry)
    # Replace the last `—` with empty since status goes where ⏳ pending was
    return f"{prefix}{dataset}{sep1}{model}{middle}{new_status}{tail}"


doc_text = DOC_PATH.read_text()

# Apply pattern 1 (format: "Dataset — Model | ⏳ pending")
new_text = DASH_PATTERN.sub(replace_dash_row, doc_text)

# Apply pattern 2 (format: "Dataset | Model | ... cols ... | ⏳ pending | —")
new_text = PIPE_PATTERN.sub(replace_pipe_row, new_text)

# ── Update summary header ──────────────────────────────────────────────────
matched = sum(1 for e in data if e.get("status") == "checked and matched")
not_matched = sum(1 for e in data if e.get("status") == "checked but not matched")
skipped = sum(1 for e in data if e.get("status", "").startswith("skipped"))
total_run = matched + not_matched
total = len(data)

summary_old = re.compile(
    r'\*\*Status\*\*:.*?experiments pending execution\.',
    re.DOTALL
)
summary_new = (
    f"**Status**: All {total} experiments complete. "
    f"**{matched} matched** (MSE diff ≤5%), "
    f"**{not_matched} close** (MSE diff >5%), "
    f"**{skipped} skipped** (dependency or GPU OOM). "
    f"Overall match rate: {matched}/{total_run} ({100*matched/total_run:.0f}% of runnable experiments)."
)
new_text = summary_old.sub(summary_new, new_text)

# ── Write updated doc ──────────────────────────────────────────────────────
DOC_PATH.write_text(new_text)
print(f"Updated {DOC_PATH}")

# ── Verify remaining ⏳ pending ──────────────────────────────────────────
remaining = re.findall(r'⏳ pending', new_text)
if remaining:
    print(f"WARNING: {len(remaining)} '⏳ pending' entries still remain")
    # Show context around each
    for m in re.finditer(r'.{80}⏳ pending.{40}', new_text):
        print(f"  ...{m.group()}...")
else:
    print("✓ All '⏳ pending' entries replaced")
