#!/usr/bin/env python3
"""Read tsl_comparison_results.json and update docs/tsl_comparison.md with actual results."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_JSON = ROOT / "experiments" / "adapt_tsl_lib" / "tsl_comparison_results.json"
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

# Friendly dataset names for display
DATASET_DISPLAY = {
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "Weather": "Weather",
    "ECL": "ECL (Electricity)",
    "Traffic": "Traffic",
    "Exchange": "Exchange Rate",
    "ILI": "ILI (Illness)",
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


def update_checklist(content: str, results: dict) -> str:
    """Update the checklist table at the bottom of the doc."""
    lines = []
    lines.append(
        "| # | Dataset | Model | Has TSL Script | Config Revised | Verified | Status | TSL ep | LL ep | TSL time | LL time |"
    )
    lines.append(
        "|---|---------|-------|---------------|----------------|----------|--------|--------|-------|----------|---------|"
    )

    for key, section_num in SECTION_MAP.items():
        entry = results[key]
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

        lines.append(
            f"| {section_num} | {dataset} | {model} | {has_script} | ✅ | {verified} | {status} | {tsl_ep} | {ll_ep} | {tsl_time_str} | {ll_time_str} |"
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
        name = entry["name"]
        results[name] = {
            "dataset": entry["dataset"],
            "model": entry["model"],
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
        }
    return results


def main():
    print(f"Reading results from: {RESULTS_JSON}")
    with open(RESULTS_JSON) as f:
        raw = json.load(f)

    results = normalize_results(raw) if isinstance(raw, list) else raw

    print(f"Reading doc from: {DOC_PATH}")
    content = DOC_PATH.read_text()

    # Count matched/unmatched
    matched = sum(
        1 for e in results.values() if "matched" in e["status"] and "not" not in e["status"]
    )
    not_matched = sum(1 for e in results.values() if "not matched" in e["status"])
    print(f"Results: {matched} matched, {not_matched} not matched, {len(results)} total")

    # Update each section
    for key, section_num in SECTION_MAP.items():
        if key not in results:
            print(f"  SKIP: {key} not in results")
            continue
        entry = results[key]
        print(f"  Updating section #{section_num}: {key} -> {entry['status']}")
        content = update_section_results(content, section_num, entry)

    # Update checklist
    print("Updating checklist table...")
    content = update_checklist(content, results)

    # Write back
    DOC_PATH.write_text(content)
    print(f"Updated doc written to: {DOC_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
