#!/usr/bin/env python3
"""Run pending and/or unmatched TSL-vs-liulian pairs via compare_tsl_liulian.py.

Default behavior:
- Select pairs that are either:
  1) not yet completed in existing JSON results, or
  2) completed as "checked but not matched"
- Exclude PatchTST and DLinear models by default.
- Exclude pairs with no TSL counterpart.

Examples:
  python experiments/adapt_tsl_lib/run_pending_or_unmatched.py
  python experiments/adapt_tsl_lib/run_pending_or_unmatched.py --mode unmatched --print-only
  python experiments/adapt_tsl_lib/run_pending_or_unmatched.py --include-model PatchTST --mode pending
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import compare_tsl_liulian as comp


DEFAULT_EXCLUDED_MODELS = {"PatchTST", "DLinear"}
OOM_FALLBACK_ENV_VAR = "LIULIAN_COMPARE_RUNTIME_OVERRIDES"
OOM_FALLBACK_PROFILES: dict[str, dict] = {
    "Traffic_TimesNet": {
        "tsl_overrides": {"batch_size": 2, "use_amp": False},
        "liulian_cli_overrides": {"batch_size": 2},
    },
    "ILI_TimesNet": {
        "tsl_overrides": {"batch_size": 2, "use_amp": False},
        "liulian_cli_overrides": {"batch_size": 2},
    },
    "Traffic_TimeXer": {
        "tsl_overrides": {"batch_size": 2, "use_amp": False},
        "liulian_cli_overrides": {"batch_size": 2},
    },
}


def load_results_by_name(json_paths: list[Path]) -> dict[str, dict]:
    """Load and merge result entries by pair name from existing JSON files."""
    merged: dict[str, dict] = {}
    for json_path in json_paths:
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text())
        except Exception as exc:
            print(f"Warning: failed to parse {json_path}: {exc}")
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            merged[name] = entry
    return merged


def select_pairs(
    mode: str,
    exclude_models: set[str],
    results_by_name: dict[str, dict],
) -> list[str]:
    """Select pair names in EXPERIMENTS order for requested mode."""
    selected: list[str] = []

    for exp in comp.EXPERIMENTS:
        if exp.model in exclude_models:
            continue
        if not exp.tsl_comparable:
            continue

        entry = results_by_name.get(exp.name)
        is_completed = bool(entry and comp.is_completed_result(entry))
        status = entry.get("status", "") if entry else ""

        want_pending = mode in {"pending", "both"} and not is_completed
        want_unmatched = mode in {"unmatched", "both"} and status == "checked but not matched"

        if want_pending or want_unmatched:
            selected.append(exp.name)

    return selected


def chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def load_latest_results_by_name(json_paths: list[Path]) -> tuple[dict[str, dict], Path | None]:
    """Load results from the latest existing JSON result path."""
    existing_paths = [p for p in json_paths if p.exists()]
    if not existing_paths:
        return {}, None
    latest = max(existing_paths, key=lambda p: p.stat().st_mtime)
    return load_results_by_name([latest]), latest


def run_compare(
    pair_names: list[str],
    chunk_size: int,
    dry_run: bool,
    disable_es: bool,
    runtime_overrides: dict[str, dict] | None = None,
) -> int:
    """Invoke compare_tsl_liulian.py for selected pairs, optionally chunked."""
    compare_script = Path(__file__).with_name("compare_tsl_liulian.py")
    rc = 0

    for idx, batch in enumerate(chunked(pair_names, chunk_size), start=1):
        cmd = [
            comp.PYTHON,
            str(compare_script),
            "--pairs",
            *batch,
        ]
        if dry_run:
            cmd.append("--dry-run")
        if disable_es:
            cmd.append("--disable-es")

        print(f"\nBatch {idx}: {len(batch)} pair(s)")
        print(" ".join(cmd))

        env = None
        if runtime_overrides:
            env = dict(os.environ)
            env[OOM_FALLBACK_ENV_VAR] = json.dumps(runtime_overrides)
            print(
                f"Using runtime overrides via {OOM_FALLBACK_ENV_VAR} for "
                f"{len(runtime_overrides)} pair(s)."
            )

        proc = subprocess.run(cmd, cwd=str(comp.PROJECT_ROOT), env=env)
        if proc.returncode != 0:
            rc = proc.returncode
            print(f"Batch {idx} failed with exit code {proc.returncode}")
            break

    return rc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["pending", "unmatched", "both"],
        default="both",
        help="Selection mode for pair picking (default: both).",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=None,
        help=(
            "Model name to exclude (repeatable). "
            "Defaults to excluding PatchTST and DLinear."
        ),
    )
    parser.add_argument(
        "--include-model",
        action="append",
        default=None,
        help=(
            "Model name to include even if it is in the default exclusion list "
            "(repeatable)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Max number of pairs per compare_tsl_liulian.py invocation.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print selected pairs; do not execute compare script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass through --dry-run to compare_tsl_liulian.py.",
    )
    parser.add_argument(
        "--disable-es",
        action="store_true",
        help="Pass through --disable-es to compare_tsl_liulian.py.",
    )
    parser.add_argument(
        "--oom-fallback",
        dest="oom_fallback",
        action="store_true",
        help="Enable OOM fallback retry for targeted pairs (default: enabled).",
    )
    parser.add_argument(
        "--no-oom-fallback",
        dest="oom_fallback",
        action="store_false",
        help="Disable OOM fallback retry.",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Restrict initial selection to predefined OOM fallback pairs.",
    )
    parser.set_defaults(oom_fallback=True)
    args = parser.parse_args()

    if args.chunk_size <= 0:
        print("--chunk-size must be > 0")
        sys.exit(2)

    exclude_models = set(DEFAULT_EXCLUDED_MODELS)
    if args.exclude_model:
        exclude_models.update(args.exclude_model)
    if args.include_model:
        exclude_models.difference_update(args.include_model)

    results_by_name = load_results_by_name([Path(p) for p in comp.RESULTS_JSON_FILES])
    selected = select_pairs(
        mode=args.mode,
        exclude_models=exclude_models,
        results_by_name=results_by_name,
    )

    if args.fallback_only:
        selected = [name for name in selected if name in OOM_FALLBACK_PROFILES]

    print(f"Mode: {args.mode}")
    print(f"Excluded models: {sorted(exclude_models)}")
    print(f"Selected pairs: {len(selected)}")

    for name in selected:
        print(f"  - {name}")

    if not selected:
        return

    if args.print_only:
        return

    rc = run_compare(
        pair_names=selected,
        chunk_size=args.chunk_size,
        dry_run=args.dry_run,
        disable_es=args.disable_es,
    )

    should_run_fallback = (
        args.oom_fallback
        and not args.print_only
        and not args.dry_run
    )
    if should_run_fallback:
        targeted_pairs = [name for name in selected if name in OOM_FALLBACK_PROFILES]
        if targeted_pairs:
            latest_results_by_name, latest_json_path = load_latest_results_by_name(
                [Path(p) for p in comp.RESULTS_JSON_FILES]
            )
            if latest_json_path is not None:
                print(f"\nFallback check from latest results file: {latest_json_path}")

            retry_pairs: list[str] = []
            for name in targeted_pairs:
                status = latest_results_by_name.get(name, {}).get("status", "")
                if isinstance(status, str) and ("GPU OOM" in status or status == "run failed"):
                    retry_pairs.append(name)

            if retry_pairs:
                retry_overrides = {name: OOM_FALLBACK_PROFILES[name] for name in retry_pairs}
                print("\nRunning OOM fallback retries for pairs:")
                for name in retry_pairs:
                    print(f"  - {name}")
                retry_rc = run_compare(
                    pair_names=retry_pairs,
                    chunk_size=args.chunk_size,
                    dry_run=False,
                    disable_es=args.disable_es,
                    runtime_overrides=retry_overrides,
                )
                if rc == 0 and retry_rc != 0:
                    rc = retry_rc

                final_results_by_name, _ = load_latest_results_by_name(
                    [Path(p) for p in comp.RESULTS_JSON_FILES]
                )
                print("\nFinal statuses after OOM fallback retries:")
                for name in retry_pairs:
                    final_status = final_results_by_name.get(name, {}).get("status", "missing")
                    print(f"  - {name}: {final_status}")
            else:
                print("\nNo targeted pairs require OOM fallback retry.")
        else:
            print("\nNo selected pairs are configured for OOM fallback profiles.")

    sys.exit(rc)


if __name__ == "__main__":
    main()
