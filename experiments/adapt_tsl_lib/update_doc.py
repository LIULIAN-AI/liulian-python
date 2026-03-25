#!/usr/bin/env python3
"""Update docs/tsl_comparison.md from comparison JSON results.

Supports:
- classic section/checklist updates for PatchTST + DLinear sections
- full-table status refresh across all models
- master results table enrichment with a Special Settings column
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_JSON_CANDIDATES = [
    ROOT / "experiments" / "adapt_tsl_lib" / "tsl_comparison_results.json",
    ROOT / "artifacts" / "tsl_comparison_results.json",
]
DOC_PATH = ROOT / "docs" / "tsl_comparison.md"

# Map JSON keys to section numbers in the doc
SECTION_MAP = {
    "ETTh1_PatchTST": 1,
    "ETTh2_PatchTST": 2,
    "ETTm1_PatchTST": 3,
    "ETTm2_PatchTST": 4,
    "Weather_PatchTST": 5,
    "ECL_PatchTST": 6,
    "Traffic_PatchTST": 7,
    "Exchange_PatchTST": 8,
    "ILI_PatchTST": 9,
    "ETTh1_DLinear": 10,
    "ETTh2_DLinear": 11,
    "ETTm1_DLinear": 12,
    "ETTm2_DLinear": 13,
    "Weather_DLinear": 14,
    "ECL_DLinear": 15,
    "Traffic_DLinear": 16,
    "Exchange_DLinear": 17,
    "ILI_DLinear": 18,
}

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
    "NS-Transformer": "NonstationaryTransformer",
    "Nonstat. Transformer": "NonstationaryTransformer",
    "LightTS": "LightTS",
    "Reformer": "Reformer",
    "GPT4TS": "GPT4TS",
    "LSTM": "LSTM",
}


def format_mse(val: float) -> str:
    """Format MSE value for display."""
    return f"{val:.4f}"


def format_mae(val: float) -> str:
    """Format MAE value for display."""
    return f"{val:.4f}"


def build_result_block(entry: dict) -> str:
    """Build the verified result markdown block for a given entry."""
    large = entry.get("large_dataset", False)
    epoch_limit = entry.get("epoch_limit")
    tsl_mse = entry["tsl_final_mse"]
    tsl_mae = entry["tsl_final_mae"]
    ll_mse = entry["liulian_final_mse"]
    ll_mae = entry["liulian_final_mae"]
    tsl_ep = entry["tsl_epochs_run"]
    ll_ep = entry["liulian_epochs_run"]
    tsl_time = entry["tsl_time_s"]
    ll_time = entry["liulian_time_s"]
    special_applied = bool(entry.get("special_settings_applied", False))
    special_tags = entry.get("special_settings_tags", []) or []

    if large and epoch_limit:
        header = f"**Verified result** ({epoch_limit}-epoch comparison):"
    else:
        header = f"**Verified result** (full training, TSL {tsl_ep}ep / LL {ll_ep}ep):"

    mse_diff = abs(tsl_mse - ll_mse)
    mse_pct = mse_diff / tsl_mse * 100 if tsl_mse != 0 else 0

    lines = [
        header,
        "| Source | MSE | MAE |",
        "|--------|-----|-----|",
        f"| TSL | {format_mse(tsl_mse)} | {format_mae(tsl_mae)} |",
        f"| Liulian | {format_mse(ll_mse)} | {format_mae(ll_mae)} |",
        f"| Diff | {format_mse(mse_diff)} ({mse_pct:.2f}%) | {format_mae(abs(tsl_mae - ll_mae))} |",
        "",
        f"Training time: TSL {tsl_time:.1f}s / Liulian {ll_time:.1f}s",
    ]
    if special_applied:
        lines.append(f"Special settings: {', '.join(special_tags) if special_tags else 'yes'}")
    return "\n".join(lines)


def status_emoji(status: str) -> str:
    if "matched" in status and "not" not in status:
        return "✅"
    elif "not matched" in status:
        return "❌"
    else:
        return "⏳"


def update_section_results(content: str, section_num: int, entry: dict) -> str:
    """Update a specific section with verified results."""
    status = entry["status"]
    emoji = status_emoji(status)
    result_block = build_result_block(entry)

    # Pattern to find the section header like "### 6. ECL (Electricity) — PatchTST  ✅ checked and revised"
    # We need to update the header emoji + status text
    header_pattern = rf"(### {section_num}\. .+? — (?:PatchTST|DLinear)\s+)[✅⏳❌]\s+.*"
    header_replacement = rf"\g<1>{emoji} {status}"
    content = re.sub(header_pattern, header_replacement, content)

    # Now handle the result block. Two cases:
    # Case 1: Section already has "**Verified result**" — replace it
    # Case 2: Section has "**Status**: Awaiting run..." — replace with result block

    # Find the section boundaries (between ### N. and the next --- or ### or ## )
    section_header_match = re.search(
        rf"### {section_num}\. .+? — (?:PatchTST|DLinear)", content
    )
    if not section_header_match:
        print(f"  WARNING: Could not find section header for #{section_num}")
        return content

    section_start = section_header_match.start()

    # Find the end of this section (next --- on its own line, or next ##)
    rest = content[section_start:]
    # Find the separator --- that ends this section
    sep_match = re.search(r"\n---\s*\n", rest)
    if sep_match:
        section_end = section_start + sep_match.start()
    else:
        section_end = len(content)

    section_text = content[section_start:section_end]

    # Replace existing "Verified result" block if present
    verified_pattern = r"\*\*Verified result\*\*[^\n]*\n\| Source \| MSE \| MAE \|\n\|[-|]+\|\n\| TSL \|[^\n]+\n\| Liulian \|[^\n]+\n(?:\| Diff \|[^\n]+\n)?(?:\nTraining time:[^\n]*)?"
    if re.search(verified_pattern, section_text):
        new_section = re.sub(verified_pattern, result_block, section_text)
        content = content[:section_start] + new_section + content[section_end:]
    else:
        # Replace "**Status**: Awaiting run..." with result block
        status_pattern = r"\*\*Status\*\*:\s*Awaiting run[^\n]*"
        if re.search(status_pattern, section_text):
            new_section = re.sub(status_pattern, result_block, section_text)
            content = content[:section_start] + new_section + content[section_end:]
        else:
            print(
                f"  WARNING: Could not find result block or status line in section #{section_num}"
            )

    return content


def normalize_dataset_name(dataset: str) -> str:
    text = (dataset or "").strip()
    for alias, canonical in DATASET_ALIASES.items():
        if alias.lower() == text.lower():
            return canonical
    for alias, canonical in DATASET_ALIASES.items():
        if alias.lower() in text.lower():
            return canonical
    return text


def normalize_model_name(model: str) -> str:
    text = (model or "").strip()
    for alias, canonical in MODEL_ALIASES.items():
        if alias.lower() == text.lower():
            return canonical
    return text


def make_pair_key(dataset: str, model: str) -> str:
    return f"{normalize_dataset_name(dataset)}_{normalize_model_name(model)}"


def compute_pct(entry: dict) -> float:
    pct = entry.get("mse_rel_diff_pct")
    if pct is not None:
        return float(pct)
    tsl = entry.get("tsl_final_mse")
    ll = entry.get("liulian_final_mse")
    if ll is None:
        ll = entry.get("ll_final_mse")
    if tsl and ll and tsl > 0:
        return abs(tsl - ll) / tsl * 100
    return 0.0


def status_cell(entry: dict | None) -> str:
    """Convert a result entry to doc status cell text."""
    if entry is None:
        return "⏳ pending"
    status = entry.get("status", "")
    special = bool(entry.get("special_settings_applied", False))
    special_suffix = " 🛠" if special else ""
    if status == "checked and matched":
        pct = compute_pct(entry)
        return f"✅ matched ({pct:.2f}%){special_suffix}"
    if status == "checked but not matched":
        pct = compute_pct(entry)
        return f"⚠️ not matched ({pct:.1f}%){special_suffix}"
    if "GPU OOM" in status:
        return f"⛔ skipped (GPU OOM){special_suffix}"
    if "mamba-ssm" in status:
        return f"⛔ skipped (mamba-ssm){special_suffix}"
    if status.startswith("skipped"):
        return f"⛔ {status}{special_suffix}"
    return f"{status}{special_suffix}" if status else f"⏳ pending{special_suffix}"


def lookup_result(results: dict[str, dict], dataset: str, model: str) -> tuple[dict | None, str]:
    key = make_pair_key(dataset, model)
    if key in results:
        return results.get(key), key
    for entry in results.values():
        if not isinstance(entry, dict):
            continue
        if make_pair_key(entry.get("dataset", ""), entry.get("model", "")) == key:
            return entry, key
    return None, key


def _replace_dash_row(match: re.Match, results: dict[str, dict]) -> str:
    prefix = match.group(1)
    dataset = match.group(2).strip()
    model = match.group(3).strip()
    sep = match.group(4)
    tail = match.group(5)
    entry, _ = lookup_result(results, dataset, model)
    return f"{prefix}{dataset} — {model}{sep}{status_cell(entry)}{tail}"


def _replace_pipe_row(match: re.Match, results: dict[str, dict]) -> str:
    prefix = match.group(1)
    dataset = match.group(2).strip()
    sep1 = match.group(3)
    model = match.group(4).strip()
    middle = match.group(5)
    tail = match.group(6)
    entry, _ = lookup_result(results, dataset, model)
    return f"{prefix}{dataset}{sep1}{model}{middle}{status_cell(entry)}{tail}"


def update_pending_status_tables(content: str, results: dict[str, dict]) -> str:
    """Update doc tables where status cells are still '⏳ pending'."""
    dash_pattern = re.compile(
        r'(\|\s*\d+\s*\|\s*)'
        r'([\w\s]+?)\s*—\s*([\w\s\.]+?)'
        r'(\s*\|\s*)⏳ pending'
        r'(\s*\|)',
    )
    pipe_pattern = re.compile(
        r'(\|\s*\d+\s*\|\s*)'
        r'([\w\s]+?)'
        r'(\s*\|\s*)'
        r'([\w\s\.\-]+?)'
        r'(\s*\|\s*(?:[^|]+\s*\|\s*){1,4})'
        r'⏳ pending'
        r'(\s*\|\s*—\s*\|)',
    )

    content = dash_pattern.sub(lambda m: _replace_dash_row(m, results), content)
    content = pipe_pattern.sub(lambda m: _replace_pipe_row(m, results), content)
    return content


def update_summary_header(content: str, results: dict[str, dict]) -> str:
    matched = sum(1 for e in results.values() if e.get("status") == "checked and matched")
    not_matched = sum(1 for e in results.values() if e.get("status") == "checked but not matched")
    skipped = sum(1 for e in results.values() if str(e.get("status", "")).startswith("skipped"))
    total = len(results)
    runnable = matched + not_matched
    rate = (100 * matched / runnable) if runnable else 0.0

    summary_pattern = re.compile(r'\*\*Status\*\*:[^\n]*')
    summary_line = (
        f"**Status**: All {total} experiments complete. "
        f"**{matched} matched** (MSE diff ≤5%), "
        f"**{not_matched} not matched** (MSE diff >5%), "
        f"**{skipped} skipped** (dependency or GPU OOM). "
        f"Overall match rate: {matched}/{runnable} ({rate:.0f}% of runnable experiments)."
    )
    return summary_pattern.sub(summary_line, content, count=1)


def special_settings_cell(entry: dict | None) -> str:
    if not entry:
        return "—"
    if not entry.get("special_settings_applied", False):
        return "—"
    tags = entry.get("special_settings_tags", []) or []
    return ", ".join(tags) if tags else "yes"


def update_master_results_table(content: str, results: dict[str, dict]) -> str:
    """Ensure master table has 'Special Settings' and update row values."""
    section_title = "## Master Results Table (All 121 Experiments)"
    section_start = content.find(section_title)
    if section_start < 0:
        print("  WARNING: Master results section not found")
        return content

    sub = content[section_start:]
    table_start_rel = sub.find("\n| #")
    summary_rel = sub.find("\n### Summary by Model")
    if table_start_rel < 0 or summary_rel < 0 or table_start_rel >= summary_rel:
        print("  WARNING: Master results table boundaries not found")
        return content

    table_start = section_start + table_start_rel + 1
    summary_start = section_start + summary_rel
    table_text = content[table_start:summary_start]
    lines = table_text.splitlines()
    if len(lines) < 3:
        print("  WARNING: Master results table is too short")
        return content

    rows_block = "\n".join(lines[2:])
    new_header = (
        "| #   | Model          | Dataset  | Script | Status                   | Special Settings | TSL MSE | LL MSE |\n"
        "| --- | -------------- | -------- | :----: | ------------------------ | ---------------- | ------: | -----: |\n"
    )

    def mse_cell(entry: dict | None, key: str) -> str:
        if not entry:
            return "—"
        val = entry.get(key)
        if val is None:
            return "—"
        return f"{float(val):.4f}"

    new_rows: list[str] = []
    for line in rows_block.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().split("|")[1:-1]]
        if len(cells) < 7:
            continue

        row_num = cells[0]
        if row_num == "#":
            continue

        model = cells[1]
        dataset = cells[2]
        script = cells[3]
        entry, _ = lookup_result(results, dataset, model)
        status = status_cell(entry)
        tsl_mse = mse_cell(entry, "tsl_final_mse")
        ll_mse = mse_cell(entry, "liulian_final_mse")
        special = special_settings_cell(entry)
        new_rows.append(
            f"| {row_num} | {model} | {dataset} | {script} | {status} | {special} | {tsl_mse} | {ll_mse} |"
        )

    rebuilt = new_header + "\n".join(new_rows) + "\n"
    return content[:table_start] + rebuilt + content[summary_start:]


def update_checklist(content: str, results: dict) -> str:
    """Update the checklist table at the bottom of the doc."""
    lines = []
    lines.append(
        "| # | Dataset | Model | Has TSL Script | Config Revised | Verified | Status | Special Settings | TSL ep | LL ep | TSL time | LL time |"
    )
    lines.append(
        "|---|---------|-------|---------------|----------------|----------|--------|------------------|--------|-------|----------|---------|"
    )

    for key, section_num in SECTION_MAP.items():
        entry = results.get(key)
        if not entry:
            continue
        dataset = entry["dataset"]
        model = entry["model"]
        has_script = "✅" if entry["has_tsl_script"] else "❌ (defaults)"
        status = entry["status"]
        emoji = status_emoji(status)

        mse_diff = abs(entry["tsl_final_mse"] - entry["liulian_final_mse"])
        mse_pct = (
            mse_diff / entry["tsl_final_mse"] * 100
            if entry["tsl_final_mse"] != 0
            else 0
        )

        if "not matched" in status:
            verified = f"❌ MSE gap {mse_pct:.1f}%"
        elif "matched" in status:
            verified = f"✅ MSE diff {mse_pct:.2f}%"
        else:
            verified = "⏳"

        tsl_ep = entry.get("tsl_epochs_run", "?")
        ll_ep = entry.get("liulian_epochs_run", "?")
        tsl_time = entry.get("tsl_time_s", "?")
        ll_time = entry.get("liulian_time_s", "?")
        tsl_time_str = f"{tsl_time:.0f}s" if isinstance(tsl_time, (int, float)) else str(tsl_time)
        ll_time_str = f"{ll_time:.0f}s" if isinstance(ll_time, (int, float)) else str(ll_time)
        special_tags = entry.get("special_settings_tags", []) or []
        special_cell = "—"
        if entry.get("special_settings_applied", False):
            special_cell = ", ".join(special_tags) if special_tags else "yes"

        lines.append(
            f"| {section_num} | {dataset} | {model} | {has_script} | ✅ | {verified} | {status} | {special_cell} | {tsl_ep} | {ll_ep} | {tsl_time_str} | {ll_time_str} |"
        )

    # Find and replace the existing checklist table — match both old (7-col) and new (11-col) formats
    checklist_header_pattern = r"\| # \| Dataset \| Model \| Has TSL Script \| Config Revised \| Verified \| Status \|[^\n]*\n\|[-| ]+\|\n(?:\|[^\n]+\|\n)+"
    new_table = "\n".join(lines) + "\n"

    if re.search(checklist_header_pattern, content):
        content = re.sub(checklist_header_pattern, new_table, content)
    else:
        print("  WARNING: Could not find checklist table to update")

    return content


def normalize_results(raw: list[dict]) -> dict[str, dict]:
    """Convert list of results to dict keyed by name, with normalized field names."""
    results = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue

        status = entry.get("status", "")
        if status == "dry-run":
            continue

        required = ["name", "dataset", "model", "tsl_epochs_run", "tsl_time_s"]
        if any(k not in entry for k in required):
            continue

        name = entry["name"]
        dataset = entry["dataset"]
        model = entry["model"]
        results[name] = {
            "name": name,
            "dataset": dataset,
            "model": model,
            "has_tsl_script": entry["has_tsl_script"],
            "large_dataset": entry.get("large", entry.get("large_dataset", False)),
            "epoch_limit": entry.get("epoch_limit"),
            "tsl_epochs_run": entry["tsl_epochs_run"],
            "liulian_epochs_run": entry.get("ll_epochs_run", entry.get("liulian_epochs_run")),
            "tsl_time_s": entry["tsl_time_s"],
            "liulian_time_s": entry.get("ll_time_s", entry.get("liulian_time_s")),
            "tsl_final_mse": entry["tsl_final_mse"],
            "tsl_final_mae": entry["tsl_final_mae"],
            "liulian_final_mse": entry.get("ll_final_mse", entry.get("liulian_final_mse")),
            "liulian_final_mae": entry.get("ll_final_mae", entry.get("liulian_final_mae")),
            "status": entry["status"],
            "special_settings_applied": entry.get("special_settings_applied", False),
            "special_settings_tags": entry.get("special_settings_tags", []),
        }

    return results


def load_results() -> dict[str, dict]:
    chosen = None
    for path in RESULTS_JSON_CANDIDATES:
        if path.exists():
            chosen = path
            break
    if chosen is None:
        raise FileNotFoundError(
            f"No results JSON found in candidates: {[str(p) for p in RESULTS_JSON_CANDIDATES]}"
        )

    print(f"Reading results from: {chosen}")
    with open(chosen) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return normalize_results(raw)
    return raw


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Update docs/tsl_comparison.md from compare JSON")
    parser.add_argument(
        "--mode",
        choices=["classic", "full", "all"],
        default="all",
        help=(
            "classic: update PatchTST/DLinear sections + checklist; "
            "full: update pending status tables + summary + master table; "
            "all: run both (default)."
        ),
    )
    args = parser.parse_args(argv)

    results = load_results()

    print(f"Reading doc from: {DOC_PATH}")
    content = DOC_PATH.read_text()

    # Count matched/unmatched
    matched = sum(
        1 for e in results.values() if "matched" in e["status"] and "not" not in e["status"]
    )
    not_matched = sum(1 for e in results.values() if "not matched" in e["status"])
    print(f"Results: {matched} matched, {not_matched} not matched, {len(results)} total")

    if args.mode in {"classic", "all"}:
        for key, section_num in SECTION_MAP.items():
            if key not in results:
                print(f"  SKIP: {key} not in results")
                continue
            entry = results[key]
            print(f"  Updating section #{section_num}: {key} -> {entry['status']}")
            content = update_section_results(content, section_num, entry)

        print("Updating checklist table...")
        content = update_checklist(content, results)

    if args.mode in {"full", "all"}:
        print("Updating pending status tables...")
        content = update_pending_status_tables(content, results)
        print("Updating summary header...")
        content = update_summary_header(content, results)
        print("Updating master results table (Special Settings)...")
        content = update_master_results_table(content, results)

    # Write back
    DOC_PATH.write_text(content)
    print(f"Updated doc written to: {DOC_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
