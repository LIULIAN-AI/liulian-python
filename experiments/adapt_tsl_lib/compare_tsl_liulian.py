#!/usr/bin/env python3
"""Compare TSL (Time-Series-Library) vs liulian benchmark results.

Runs both TSL and liulian for each dataset × model pair, compares metrics,
and writes structured results to ``artifacts/tsl_comparison_results.txt``.

Usage (from project root, with .venv activated):

    .venv/bin/python tools/compare_tsl_liulian.py

Options:
    --pairs PAIR [PAIR ...]   Run only selected pairs by name, e.g.
                              --pairs ETTh1_PatchTST ETTh2_DLinear
    --dry-run                 Print commands without executing them.

The script automatically:
  • Limits large datasets (ECL, Traffic) to 2 epochs and compares per-epoch
    test loss instead of final metrics.
  • Records wall-clock time and epochs for every experiment.
  • Writes per-pair "checked and matched" / "checked but not matched" verdicts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TSL_ROOT = PROJECT_ROOT / "refer_projects" / "Time-Series-Library"
PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
RESULTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_FILE = RESULTS_DIR / "tsl_comparison_results.txt"

# Tolerance for "matched" verdict
MSE_ABS_TOL = 0.010   # absolute MSE difference
MSE_REL_TOL = 0.05    # 5 % relative

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """One dataset × model comparison pair."""
    name: str                     # e.g. "ETTh1_PatchTST"
    dataset: str
    model: str                    # "PatchTST" or "DLinear"
    liulian_config: str           # path relative to PROJECT_ROOT
    has_tsl_script: bool          # True if a script exists in TSL
    large: bool = False           # limit to 2 epochs and compare per-epoch
    # Non-default TSL args (override defaults from run.py)
    tsl_overrides: dict = field(default_factory=dict)


# TSL default args (applied to every TSL run); scripts override some of these.
TSL_BASE_ARGS = {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "features": "M",
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 96,
    "e_layers": 2,
    "d_layers": 1,
    "factor": 3,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "des": "Exp",
    "itr": 1,
}

EXPERIMENTS: list[Experiment] = [
    # ── PatchTST ────────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_PatchTST",
        dataset="ETTh1", model="PatchTST",
        liulian_config="experiments/etth1/patchtst_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "e_layers": 1, "n_heads": 2,
        },
    ),
    Experiment(
        name="ETTh2_PatchTST",
        dataset="ETTh2", model="PatchTST",
        liulian_config="experiments/etth2/patchtst_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "e_layers": 3, "n_heads": 4,
        },
    ),
    Experiment(
        name="ETTm1_PatchTST",
        dataset="ETTm1", model="PatchTST",
        liulian_config="experiments/ettm1/patchtst_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "e_layers": 1, "n_heads": 2,
        },
    ),
    Experiment(
        name="ETTm2_PatchTST",
        dataset="ETTm2", model="PatchTST",
        liulian_config="experiments/ettm2/patchtst_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "e_layers": 3, "n_heads": 16,
        },
    ),
    Experiment(
        name="Weather_PatchTST",
        dataset="Weather", model="PatchTST",
        liulian_config="experiments/weather/patchtst_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "e_layers": 2, "n_heads": 4,
            "train_epochs": 3,
        },
    ),
    Experiment(
        name="ECL_PatchTST",
        dataset="ECL", model="PatchTST",
        liulian_config="experiments/electricity/patchtst_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "e_layers": 2,
            "batch_size": 16,
        },
    ),
    Experiment(
        name="Traffic_PatchTST",
        dataset="Traffic", model="PatchTST",
        liulian_config="experiments/traffic/patchtst_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "d_model": 512, "d_ff": 512,
            "e_layers": 2, "top_k": 5,
            "batch_size": 4,
        },
    ),
    Experiment(
        name="Exchange_PatchTST",
        dataset="Exchange", model="PatchTST",
        liulian_config="experiments/exchange_rate/patchtst_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "e_layers": 2,
        },
    ),
    Experiment(
        name="ILI_PatchTST",
        dataset="ILI", model="PatchTST",
        liulian_config="experiments/illness/patchtst_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "e_layers": 4, "n_heads": 4, "d_model": 1024,
        },
    ),
    # ── DLinear ─────────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_DLinear",
        dataset="ETTh1", model="DLinear",
        liulian_config="experiments/etth1/dlinear_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
        },
    ),
    Experiment(
        name="ETTh2_DLinear",
        dataset="ETTh2", model="DLinear",
        liulian_config="experiments/etth2/dlinear_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
        },
    ),
    Experiment(
        name="ETTm1_DLinear",
        dataset="ETTm1", model="DLinear",
        liulian_config="experiments/ettm1/dlinear_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
        },
    ),
    Experiment(
        name="ETTm2_DLinear",
        dataset="ETTm2", model="DLinear",
        liulian_config="experiments/ettm2/dlinear_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
        },
    ),
    Experiment(
        name="Weather_DLinear",
        dataset="Weather", model="DLinear",
        liulian_config="experiments/weather/dlinear_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
        },
    ),
    Experiment(
        name="ECL_DLinear",
        dataset="ECL", model="DLinear",
        liulian_config="experiments/electricity/dlinear_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
        },
    ),
    Experiment(
        name="Traffic_DLinear",
        dataset="Traffic", model="DLinear",
        liulian_config="experiments/traffic/dlinear_config.yaml",
        has_tsl_script=False,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
        },
    ),
    Experiment(
        name="Exchange_DLinear",
        dataset="Exchange", model="DLinear",
        liulian_config="experiments/exchange_rate/dlinear_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
        },
    ),
    Experiment(
        name="ILI_DLinear",
        dataset="ILI", model="DLinear",
        liulian_config="experiments/illness/dlinear_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
        },
    ),
]

# ---------------------------------------------------------------------------
# TSL command builder
# ---------------------------------------------------------------------------

def build_tsl_cmd(
    exp: Experiment,
    epoch_limit: Optional[int] = None,
    disable_es: bool = False,
) -> list[str]:
    """Build the TSL run.py command for an experiment."""
    args = dict(TSL_BASE_ARGS)
    args["model"] = exp.model
    args.update(exp.tsl_overrides)
    if epoch_limit is not None:
        args["train_epochs"] = epoch_limit
    if disable_es:
        args["patience"] = 9999  # effectively disable early stopping

    cmd = [PYTHON, "-u", "run.py"]
    for k, v in args.items():
        cmd.extend([f"--{k}", str(v)])
    return cmd


def build_liulian_cmd(
    exp: Experiment,
    epoch_limit: Optional[int] = None,
    disable_es: bool = False,
) -> list[str]:
    """Build the liulian experiments/run.py command."""
    config_path = str(PROJECT_ROOT / exp.liulian_config)
    cmd = [PYTHON, str(PROJECT_ROOT / "experiments" / "run.py"),
           "--config", config_path]
    if epoch_limit is not None:
        cmd.extend(["--train_epochs", str(epoch_limit)])
    if disable_es:
        cmd.append("--disable_early_stopping")
    return cmd

# ---------------------------------------------------------------------------
# Parsers for TSL output
# ---------------------------------------------------------------------------

# TSL per-epoch line:
#   Epoch: 1, Steps: 409 | Train Loss: 0.5010 Vali Loss: 0.3994 Test Loss: 0.3747
_TSL_EPOCH_RE = re.compile(
    r"Epoch:\s*(\d+),\s*Steps:\s*\d+\s*\|\s*"
    r"Train Loss:\s*([\d.]+)\s+"
    r"Vali Loss:\s*([\d.]+)\s+"
    r"Test Loss:\s*([\d.]+)"
)

# TSL final test line:
#   mse:0.37921, mae:0.39964, dtw:...
_TSL_FINAL_RE = re.compile(
    r"mse:([\d.]+),\s*mae:([\d.]+)"
)


def parse_tsl_output(output: str) -> dict:
    """Extract per-epoch and final metrics from TSL stdout."""
    result: dict = {"epochs": [], "final_mse": None, "final_mae": None}
    for m in _TSL_EPOCH_RE.finditer(output):
        result["epochs"].append({
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(2)),
            "vali_loss": float(m.group(3)),
            "test_loss": float(m.group(4)),   # this is test MSE
        })
    m = _TSL_FINAL_RE.search(output)
    if m:
        result["final_mse"] = float(m.group(1))
        result["final_mae"] = float(m.group(2))
    return result

# ---------------------------------------------------------------------------
# Parsers for liulian output
# ---------------------------------------------------------------------------

# Liulian per-epoch line (from logger.info):
#   Epoch 1 (12.3s) | Train MSE: 0.500000 | val_mse: 0.400000 | test_mse=0.380000, ...
_LL_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\([\d.]+s\)\s*\|.*?"
    r"test_mse=([\d.]+)"
)

# Liulian JSON final output — we look for "test": {"mse": ...}
_LL_JSON_MSE_RE = re.compile(r'"test":\s*\{[^}]*"mse":\s*([\d.eE+-]+)')
_LL_JSON_MAE_RE = re.compile(r'"test":\s*\{[^}]*"mae":\s*([\d.eE+-]+)')
_LL_JSON_EPOCHS_RE = re.compile(r'"epochs_run":\s*(\d+)')


def parse_liulian_output(output: str) -> dict:
    """Extract per-epoch and final metrics from liulian stdout/stderr."""
    result: dict = {"epochs": [], "final_mse": None, "final_mae": None,
                    "epochs_run": None}
    for m in _LL_EPOCH_RE.finditer(output):
        result["epochs"].append({
            "epoch": int(m.group(1)),
            "test_mse": float(m.group(2)),
        })

    # Parse JSON output at the end
    m = _LL_JSON_MSE_RE.search(output)
    if m:
        result["final_mse"] = float(m.group(1))
    m = _LL_JSON_MAE_RE.search(output)
    if m:
        result["final_mae"] = float(m.group(1))
    m = _LL_JSON_EPOCHS_RE.search(output)
    if m:
        result["epochs_run"] = int(m.group(1))
    return result

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], cwd: str, label: str) -> tuple[str, float]:
    """Run a command, return (combined_output, elapsed_seconds)."""
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  CWD: {cwd}")
    print(f"  CMD: {' '.join(cmd[:6])} ...")
    print(f"{'='*60}")

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=7200,  # 2 hour max per experiment
    )
    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.1f}s (exit code {proc.returncode})")
    return proc.stdout, elapsed


def compare_metrics(
    tsl: dict, ll: dict, large: bool,
) -> tuple[bool, str]:
    """Compare TSL vs liulian metrics. Returns (matched, detail_string)."""
    lines = []

    if large:
        # Compare per-epoch test loss
        if not tsl["epochs"] or not ll["epochs"]:
            return False, "FAIL: no per-epoch data extracted"
        # Compare each epoch present in both
        matched = True
        n = min(len(tsl["epochs"]), len(ll["epochs"]))
        for i in range(n):
            tsl_mse = tsl["epochs"][i]["test_loss"]
            ll_mse = ll["epochs"][i]["test_mse"]
            diff = abs(tsl_mse - ll_mse)
            rel = diff / max(tsl_mse, 1e-8)
            ok = diff < MSE_ABS_TOL or rel < MSE_REL_TOL
            mark = "✓" if ok else "✗"
            lines.append(
                f"  Epoch {i+1}: TSL test_loss={tsl_mse:.6f}  "
                f"liulian test_mse={ll_mse:.6f}  "
                f"diff={diff:.6f} ({rel*100:.2f}%) {mark}"
            )
            if not ok:
                matched = False
        return matched, "\n".join(lines)
    else:
        # Compare final MSE/MAE
        if tsl["final_mse"] is None:
            return False, "FAIL: no TSL final mse found"
        if ll["final_mse"] is None:
            return False, "FAIL: no liulian final mse found"

        mse_diff = abs(tsl["final_mse"] - ll["final_mse"])
        mse_rel = mse_diff / max(tsl["final_mse"], 1e-8)
        mae_diff = abs((tsl["final_mae"] or 0) - (ll["final_mae"] or 0))

        ok_mse = mse_diff < MSE_ABS_TOL or mse_rel < MSE_REL_TOL
        lines.append(
            f"  TSL  final: MSE={tsl['final_mse']:.6f}  "
            f"MAE={tsl['final_mae']:.6f}"
        )
        lines.append(
            f"  LL   final: MSE={ll['final_mse']:.6f}  "
            f"MAE={ll['final_mae']:.6f}"
        )
        lines.append(
            f"  Diff:       MSE={mse_diff:.6f} ({mse_rel*100:.2f}%)  "
            f"MAE={mae_diff:.6f}"
        )
        return ok_mse, "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Run only these experiment names (e.g. ETTh1_PatchTST)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--disable-es", action="store_true",
        help=(
            "Disable early stopping for both sides. "
            "TSL: sets --patience 9999 (effectively disabling ES). "
            "Liulian: passes --disable_early_stopping flag. "
            "By default, both TSL and liulian use their native early stopping "
            "(patience=3). Use this flag only for diagnostic purposes to "
            "isolate the effect of ES divergence on results."
        ),
    )
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.pairs:
        names = set(args.pairs)
        experiments = [e for e in EXPERIMENTS if e.name in names]
        if not experiments:
            print(f"No matching experiments for: {args.pairs}")
            print(f"Available: {[e.name for e in EXPERIMENTS]}")
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for exp in experiments:
        print(f"\n{'#'*70}")
        print(f"# {exp.name}  (large={exp.large}, has_script={exp.has_tsl_script})")
        print(f"{'#'*70}")

        epoch_limit = 2 if exp.large else None
        disable_es = getattr(args, 'disable_es', False)
        tsl_cmd = build_tsl_cmd(exp, epoch_limit=epoch_limit, disable_es=disable_es)
        ll_cmd = build_liulian_cmd(exp, epoch_limit=epoch_limit, disable_es=disable_es)

        if args.dry_run:
            print(f"  TSL cmd: {' '.join(tsl_cmd)}")
            print(f"  LL  cmd: {' '.join(ll_cmd)}")
            results.append({
                "name": exp.name, "status": "dry-run",
                "tsl_cmd": " ".join(tsl_cmd),
                "ll_cmd": " ".join(ll_cmd),
            })
            continue

        # --- Run TSL ---
        try:
            tsl_out, tsl_time = run_cmd(
                tsl_cmd, cwd=str(TSL_ROOT), label=f"TSL {exp.name}",
            )
        except subprocess.TimeoutExpired:
            tsl_out, tsl_time = "", -1
            print("  TSL TIMEOUT")

        tsl_metrics = parse_tsl_output(tsl_out)

        # --- Run liulian ---
        try:
            ll_out, ll_time = run_cmd(
                ll_cmd, cwd=str(PROJECT_ROOT), label=f"liulian {exp.name}",
            )
        except subprocess.TimeoutExpired:
            ll_out, ll_time = "", -1
            print("  liulian TIMEOUT")

        ll_metrics = parse_liulian_output(ll_out)

        # --- Compare ---
        matched, detail = compare_metrics(tsl_metrics, ll_metrics, exp.large)
        status = "checked and matched" if matched else "checked but not matched"

        tsl_epochs_run = len(tsl_metrics["epochs"])
        ll_epochs_run = ll_metrics.get("epochs_run") or len(ll_metrics["epochs"])

        r = {
            "name": exp.name,
            "dataset": exp.dataset,
            "model": exp.model,
            "has_tsl_script": exp.has_tsl_script,
            "large": exp.large,
            "epoch_limit": epoch_limit,
            "tsl_epochs_run": tsl_epochs_run,
            "ll_epochs_run": ll_epochs_run,
            "tsl_time_s": round(tsl_time, 1),
            "ll_time_s": round(ll_time, 1),
            "tsl_final_mse": tsl_metrics.get("final_mse"),
            "tsl_final_mae": tsl_metrics.get("final_mae"),
            "ll_final_mse": ll_metrics.get("final_mse"),
            "ll_final_mae": ll_metrics.get("final_mae"),
            "status": status,
            "detail": detail,
        }
        results.append(r)

        print(f"\n  ── Result: {status} ──")
        print(detail)
        print(f"  TSL: {tsl_epochs_run} epochs in {tsl_time:.1f}s")
        print(f"  LL:  {ll_epochs_run} epochs in {ll_time:.1f}s")

    # --- Write results file ---
    with open(RESULTS_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TSL vs Liulian Comparison Results\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"--- {r['name']} ---\n")
            if r.get("status") == "dry-run":
                f.write(f"  Status: dry-run\n")
                f.write(f"  TSL cmd: {r.get('tsl_cmd', '')}\n")
                f.write(f"  LL  cmd: {r.get('ll_cmd', '')}\n")
            else:
                f.write(f"  Dataset: {r.get('dataset', '')}\n")
                f.write(f"  Model: {r.get('model', '')}\n")
                f.write(f"  Has TSL script: {r.get('has_tsl_script', '')}\n")
                f.write(f"  Large dataset: {r.get('large', False)}\n")
                f.write(f"  Epoch limit: {r.get('epoch_limit', 'full')}\n")
                f.write(f"  TSL epochs run: {r.get('tsl_epochs_run', '?')}\n")
                f.write(f"  TSL time (s): {r.get('tsl_time_s', '?')}\n")
                f.write(f"  Liulian epochs run: {r.get('ll_epochs_run', '?')}\n")
                f.write(f"  Liulian time (s): {r.get('ll_time_s', '?')}\n")
                if r.get("tsl_final_mse") is not None:
                    f.write(f"  TSL final MSE: {r['tsl_final_mse']:.6f}\n")
                    f.write(f"  TSL final MAE: {r.get('tsl_final_mae', '?')}\n")
                if r.get("ll_final_mse") is not None:
                    f.write(f"  Liulian final MSE: {r['ll_final_mse']:.6f}\n")
                    f.write(f"  Liulian final MAE: {r.get('ll_final_mae', '?')}\n")
                f.write(f"  Detail:\n{r.get('detail', '')}\n")
                f.write(f"  Status: {r['status']}\n")
            f.write("\n")

        # Summary table
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Name':<25} {'TSL ep':>6} {'LL ep':>5} "
                f"{'TSL time':>9} {'LL time':>8} {'Status'}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            if r.get("status") == "dry-run":
                f.write(f"{r['name']:<25} {'---':>6} {'---':>5} "
                        f"{'---':>9} {'---':>8} dry-run\n")
            else:
                f.write(
                    f"{r['name']:<25} "
                    f"{r.get('tsl_epochs_run', '?'):>6} "
                    f"{r.get('ll_epochs_run', '?'):>5} "
                    f"{r.get('tsl_time_s', '?'):>8}s "
                    f"{r.get('ll_time_s', '?'):>7}s "
                    f"{r['status']}\n"
                )

    # Also write JSON for programmatic use
    json_file = RESULTS_DIR / "tsl_comparison_results.json"
    # Remove non-serialisable bits
    json_results = []
    for r in results:
        jr = dict(r)
        jr.pop("detail", None)
        json_results.append(jr)
    with open(json_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results written to:")
    print(f"  Text: {RESULTS_FILE}")
    print(f"  JSON: {json_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
