#!/usr/bin/env python3
"""Compare TSL (Time-Series-Library) vs liulian benchmark results.

Runs both TSL and liulian for each dataset × model pair, compares metrics,
and writes structured results to
``experiments/adapt_tsl_lib/tsl_comparison_results.txt``.

Usage (from project root, with .venv activated):

    .venv/bin/python tools/compare_tsl_liulian.py

Options:
    --pairs PAIR [PAIR ...]   Run only selected pairs by name, e.g.
                              --pairs ETTh1_PatchTST ETTh2_DLinear
    --remaining-only          Skip pairs that already have non-dry-run
                              results in an existing JSON results file.
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
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TSL_ROOT = PROJECT_ROOT / "refer_projects" / "Time-Series-Library"
PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "adapt_tsl_lib"
LEGACY_RESULTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_FILE = RESULTS_DIR / "tsl_comparison_results.txt"
RESULTS_JSON_FILES = [
    RESULTS_DIR / 'tsl_comparison_results.json',
    LEGACY_RESULTS_DIR / 'tsl_comparison_results.json',
]
RUNTIME_OVERRIDES_ENV_VAR = 'LIULIAN_COMPARE_RUNTIME_OVERRIDES'

OOM_FALLBACK_PROFILES: dict[str, dict[str, dict]] = {
    "Traffic_TimesNet": {
        "tsl_overrides": {
            "batch_size": 2,
            "use_amp": False,
        },
        "liulian_cli_overrides": {
            "batch_size": 2,
        },
    },
    "ILI_TimesNet": {
        "tsl_overrides": {
            "batch_size": 2,
            "use_amp": False,
        },
        "liulian_cli_overrides": {
            "batch_size": 2,
        },
    },
    "Traffic_TimeXer": {
        "tsl_overrides": {
            "batch_size": 2,
            "use_amp": False,
        },
        "liulian_cli_overrides": {
            "batch_size": 2,
        },
    },
}

COMPLETED_STATUSES = {
    'checked and matched',
    'checked but not matched',
}

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
    tsl_comparable: bool = True   # False if model has no TSL counterpart
    skip_reason: Optional[str] = None  # If set, pair is tracked but skipped.
    large: bool = False           # limit to 2 epochs and compare per-epoch
    # Non-default TSL args (override defaults from run.py)
    tsl_overrides: dict = field(default_factory=dict)


# TSL default args (applied to every TSL run); scripts override some of these.
TSL_BASE_ARGS = {
    "num_workers": 0,
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
    # ── Informer ────────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_Informer",
        dataset="ETTh1", model="Informer",
        liulian_config="experiments/etth1/informer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTh2_Informer",
        dataset="ETTh2", model="Informer",
        liulian_config="experiments/etth2/informer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm1_Informer",
        dataset="ETTm1", model="Informer",
        liulian_config="experiments/ettm1/informer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm2_Informer",
        dataset="ETTm2", model="Informer",
        liulian_config="experiments/ettm2/informer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Weather_Informer",
        dataset="Weather", model="Informer",
        liulian_config="experiments/weather/informer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ECL_Informer",
        dataset="ECL", model="Informer",
        liulian_config="experiments/electricity/informer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Traffic_Informer",
        dataset="Traffic", model="Informer",
        liulian_config="experiments/traffic/informer_config.yaml",
        has_tsl_script=False,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Exchange_Informer",
        dataset="Exchange", model="Informer",
        liulian_config="experiments/exchange_rate/informer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ILI_Informer",
        dataset="ILI", model="Informer",
        liulian_config="experiments/illness/informer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    # ── Autoformer ──────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_Autoformer",
        dataset="ETTh1", model="Autoformer",
        liulian_config="experiments/etth1/autoformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTh2_Autoformer",
        dataset="ETTh2", model="Autoformer",
        liulian_config="experiments/etth2/autoformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm1_Autoformer",
        dataset="ETTm1", model="Autoformer",
        liulian_config="experiments/ettm1/autoformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm2_Autoformer",
        dataset="ETTm2", model="Autoformer",
        liulian_config="experiments/ettm2/autoformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 1,
        },
    ),
    Experiment(
        name="Weather_Autoformer",
        dataset="Weather", model="Autoformer",
        liulian_config="experiments/weather/autoformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "train_epochs": 2,
        },
    ),
    Experiment(
        name="ECL_Autoformer",
        dataset="ECL", model="Autoformer",
        liulian_config="experiments/electricity/autoformer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Traffic_Autoformer",
        dataset="Traffic", model="Autoformer",
        liulian_config="experiments/traffic/autoformer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "train_epochs": 3,
        },
    ),
    Experiment(
        name="Exchange_Autoformer",
        dataset="Exchange", model="Autoformer",
        liulian_config="experiments/exchange_rate/autoformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ILI_Autoformer",
        dataset="ILI", model="Autoformer",
        liulian_config="experiments/illness/autoformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    # ── FEDformer ───────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_FEDformer",
        dataset="ETTh1", model="FEDformer",
        liulian_config="experiments/etth1/fedformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTh2_FEDformer",
        dataset="ETTh2", model="FEDformer",
        liulian_config="experiments/etth2/fedformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm1_FEDformer",
        dataset="ETTm1", model="FEDformer",
        liulian_config="experiments/ettm1/fedformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm2_FEDformer",
        dataset="ETTm2", model="FEDformer",
        liulian_config="experiments/ettm2/fedformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Weather_FEDformer",
        dataset="Weather", model="FEDformer",
        liulian_config="experiments/weather/fedformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ECL_FEDformer",
        dataset="ECL", model="FEDformer",
        liulian_config="experiments/electricity/fedformer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Traffic_FEDformer",
        dataset="Traffic", model="FEDformer",
        liulian_config="experiments/traffic/fedformer_config.yaml",
        has_tsl_script=False,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Exchange_FEDformer",
        dataset="Exchange", model="FEDformer",
        liulian_config="experiments/exchange_rate/fedformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ILI_FEDformer",
        dataset="ILI", model="FEDformer",
        liulian_config="experiments/illness/fedformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    # ── TimesNet ────────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_TimesNet",
        dataset="ETTh1", model="TimesNet",
        liulian_config="experiments/etth1/timesnet_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "d_model": 16, "d_ff": 32,
            "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5,
        },
    ),
    Experiment(
        name="ETTh2_TimesNet",
        dataset="ETTh2", model="TimesNet",
        liulian_config="experiments/etth2/timesnet_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "d_model": 32, "d_ff": 32,
            "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5,
        },
    ),
    Experiment(
        name="ETTm1_TimesNet",
        dataset="ETTm1", model="TimesNet",
        liulian_config="experiments/ettm1/timesnet_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "d_model": 64, "d_ff": 64,
            "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5,
        },
    ),
    Experiment(
        name="ETTm2_TimesNet",
        dataset="ETTm2", model="TimesNet",
        liulian_config="experiments/ettm2/timesnet_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "d_model": 32, "d_ff": 32,
            "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5,
        },
    ),
    Experiment(
        name="Weather_TimesNet",
        dataset="Weather", model="TimesNet",
        liulian_config="experiments/weather/timesnet_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "d_model": 32, "d_ff": 32,
            "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5,
        },
    ),
    Experiment(
        name="ECL_TimesNet",
        dataset="ECL", model="TimesNet",
        liulian_config="experiments/electricity/timesnet_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "d_model": 256, "d_ff": 512,
            "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5,
        },
    ),
    Experiment(
        name="Traffic_TimesNet",
        dataset="Traffic", model="TimesNet",
        liulian_config="experiments/traffic/timesnet_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "d_model": 512, "d_ff": 512,
            "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5,
        },
    ),
    Experiment(
        name="Exchange_TimesNet",
        dataset="Exchange", model="TimesNet",
        liulian_config="experiments/exchange_rate/timesnet_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "d_model": 64, "d_ff": 64,
            "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5,
        },
    ),
    Experiment(
        name="ILI_TimesNet",
        dataset="ILI", model="TimesNet",
        liulian_config="experiments/illness/timesnet_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "d_model": 768, "d_ff": 768,
            "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5,
        },
    ),
    # ── Transformer ─────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_Transformer",
        dataset="ETTh1", model="Transformer",
        liulian_config="experiments/etth1/transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTh2_Transformer",
        dataset="ETTh2", model="Transformer",
        liulian_config="experiments/etth2/transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm1_Transformer",
        dataset="ETTm1", model="Transformer",
        liulian_config="experiments/ettm1/transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm2_Transformer",
        dataset="ETTm2", model="Transformer",
        liulian_config="experiments/ettm2/transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 1,
        },
    ),
    Experiment(
        name="Weather_Transformer",
        dataset="Weather", model="Transformer",
        liulian_config="experiments/weather/transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "train_epochs": 3,
        },
    ),
    Experiment(
        name="ECL_Transformer",
        dataset="ECL", model="Transformer",
        liulian_config="experiments/electricity/transformer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "features": "S",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Traffic_Transformer",
        dataset="Traffic", model="Transformer",
        liulian_config="experiments/traffic/transformer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "train_epochs": 3,
        },
    ),
    Experiment(
        name="Exchange_Transformer",
        dataset="Exchange", model="Transformer",
        liulian_config="experiments/exchange_rate/transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ILI_Transformer",
        dataset="ILI", model="Transformer",
        liulian_config="experiments/illness/transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    # ── iTransformer ────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_iTransformer",
        dataset="ETTh1", model="iTransformer",
        liulian_config="experiments/etth1/itransformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "d_model": 128, "d_ff": 128,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTh2_iTransformer",
        dataset="ETTh2", model="iTransformer",
        liulian_config="experiments/etth2/itransformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "d_model": 128, "d_ff": 128,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm1_iTransformer",
        dataset="ETTm1", model="iTransformer",
        liulian_config="experiments/ettm1/itransformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "d_model": 128, "d_ff": 128,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm2_iTransformer",
        dataset="ETTm2", model="iTransformer",
        liulian_config="experiments/ettm2/itransformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "d_model": 128, "d_ff": 128,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Weather_iTransformer",
        dataset="Weather", model="iTransformer",
        liulian_config="experiments/weather/itransformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "d_model": 512, "d_ff": 512,
            "e_layers": 3, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ECL_iTransformer",
        dataset="ECL", model="iTransformer",
        liulian_config="experiments/electricity/itransformer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "d_model": 512, "d_ff": 512,
            "e_layers": 3, "d_layers": 1, "factor": 3,
            "batch_size": 16, "learning_rate": 0.0005,
        },
    ),
    Experiment(
        name="Traffic_iTransformer",
        dataset="Traffic", model="iTransformer",
        liulian_config="experiments/traffic/itransformer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "d_model": 512, "d_ff": 512,
            "e_layers": 4, "d_layers": 1, "factor": 3,
            "batch_size": 16, "learning_rate": 0.001,
        },
    ),
    Experiment(
        name="Exchange_iTransformer",
        dataset="Exchange", model="iTransformer",
        liulian_config="experiments/exchange_rate/itransformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "d_model": 128, "d_ff": 128,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ILI_iTransformer",
        dataset="ILI", model="iTransformer",
        liulian_config="experiments/illness/itransformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "d_model": 128, "d_ff": 128,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    # ── TimeMixer ───────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_TimeMixer",
        dataset="ETTh1", model="TimeMixer",
        liulian_config="experiments/etth1/timemixer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "label_len": 0,
            "d_model": 16, "d_ff": 32, "e_layers": 2,
            "batch_size": 128, "learning_rate": 0.01, "train_epochs": 10,
            "down_sampling_layers": 3, "down_sampling_method": "avg",
            "down_sampling_window": 2,
        },
    ),
    Experiment(
        name="ETTh2_TimeMixer",
        dataset="ETTh2", model="TimeMixer",
        liulian_config="experiments/etth2/timemixer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "label_len": 0,
            "d_model": 16, "d_ff": 32, "e_layers": 2,
            "learning_rate": 0.01,
            "down_sampling_layers": 3, "down_sampling_method": "avg",
            "down_sampling_window": 2,
        },
    ),
    Experiment(
        name="ETTm1_TimeMixer",
        dataset="ETTm1", model="TimeMixer",
        liulian_config="experiments/ettm1/timemixer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "label_len": 0,
            "d_model": 16, "d_ff": 32, "e_layers": 2,
            "batch_size": 16, "learning_rate": 0.01,
            "down_sampling_layers": 3, "down_sampling_method": "avg",
            "down_sampling_window": 2,
        },
    ),
    Experiment(
        name="ETTm2_TimeMixer",
        dataset="ETTm2", model="TimeMixer",
        liulian_config="experiments/ettm2/timemixer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "label_len": 0,
            "d_model": 32, "d_ff": 32, "e_layers": 2,
            "batch_size": 128, "learning_rate": 0.01,
            "down_sampling_layers": 3, "down_sampling_method": "avg",
            "down_sampling_window": 2,
        },
    ),
    Experiment(
        name="Weather_TimeMixer",
        dataset="Weather", model="TimeMixer",
        liulian_config="experiments/weather/timemixer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "label_len": 0,
            "d_model": 16, "d_ff": 32, "e_layers": 3, "d_layers": 1, "factor": 3,
            "batch_size": 128, "learning_rate": 0.01, "train_epochs": 20,
            "down_sampling_layers": 3, "down_sampling_method": "avg",
            "down_sampling_window": 2,
        },
    ),
    Experiment(
        name="ECL_TimeMixer",
        dataset="ECL", model="TimeMixer",
        liulian_config="experiments/electricity/timemixer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "label_len": 0,
            "d_model": 16, "d_ff": 32, "e_layers": 3, "d_layers": 1, "factor": 3,
            "batch_size": 32, "learning_rate": 0.01, "train_epochs": 20,
            "down_sampling_layers": 3, "down_sampling_method": "avg",
            "down_sampling_window": 2,
        },
    ),
    Experiment(
        name="Traffic_TimeMixer",
        dataset="Traffic", model="TimeMixer",
        liulian_config="experiments/traffic/timemixer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "label_len": 0,
            "d_model": 32, "d_ff": 64, "e_layers": 3, "d_layers": 1, "factor": 3,
            "batch_size": 8, "learning_rate": 0.01,
            "down_sampling_layers": 3, "down_sampling_method": "avg",
            "down_sampling_window": 2,
        },
    ),
    Experiment(
        name="Exchange_TimeMixer",
        dataset="Exchange", model="TimeMixer",
        liulian_config="experiments/exchange_rate/timemixer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "label_len": 0,
            "d_model": 16, "d_ff": 32, "e_layers": 2,
            "learning_rate": 0.01,
            "down_sampling_layers": 3, "down_sampling_method": "avg",
            "down_sampling_window": 2,
            "channel_independence": 1, "decomp_method": "moving_avg",
            "use_norm": 1,
        },
    ),
    Experiment(
        name="ILI_TimeMixer",
        dataset="ILI", model="TimeMixer",
        liulian_config="experiments/illness/timemixer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 0, "pred_len": 24,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
            "d_model": 16, "d_ff": 32, "e_layers": 2,
            "learning_rate": 0.01,
            "down_sampling_layers": 3, "down_sampling_method": "avg",
            "down_sampling_window": 2,
            "channel_independence": 1, "decomp_method": "moving_avg",
            "use_norm": 1,
        },
    ),
    # ── TimeXer ─────────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_TimeXer",
        dataset="ETTh1", model="TimeXer",
        liulian_config="experiments/etth1/timexer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "d_model": 256, "e_layers": 1, "factor": 3,
            "batch_size": 4,
        },
    ),
    Experiment(
        name="ETTh2_TimeXer",
        dataset="ETTh2", model="TimeXer",
        liulian_config="experiments/etth2/timexer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "d_model": 256, "d_ff": 1024,
            "e_layers": 1, "factor": 3,
            "batch_size": 16,
        },
    ),
    Experiment(
        name="ETTm1_TimeXer",
        dataset="ETTm1", model="TimeXer",
        liulian_config="experiments/ettm1/timexer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "d_model": 256, "e_layers": 1, "factor": 3,
            "batch_size": 4,
        },
    ),
    Experiment(
        name="ETTm2_TimeXer",
        dataset="ETTm2", model="TimeXer",
        liulian_config="experiments/ettm2/timexer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "d_model": 256, "e_layers": 1, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Weather_TimeXer",
        dataset="Weather", model="TimeXer",
        liulian_config="experiments/weather/timexer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "d_model": 256, "d_ff": 512,
            "e_layers": 1, "factor": 3,
            "batch_size": 4,
        },
    ),
    Experiment(
        name="ECL_TimeXer",
        dataset="ECL", model="TimeXer",
        liulian_config="experiments/electricity/timexer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "d_ff": 512, "e_layers": 4, "factor": 3,
            "batch_size": 4,
        },
    ),
    Experiment(
        name="Traffic_TimeXer",
        dataset="Traffic", model="TimeXer",
        liulian_config="experiments/traffic/timexer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "d_model": 512, "d_ff": 512,
            "e_layers": 3, "factor": 3,
            "batch_size": 16, "learning_rate": 0.001,
        },
    ),
    Experiment(
        name="Exchange_TimeXer",
        dataset="Exchange", model="TimeXer",
        liulian_config="experiments/exchange_rate/timexer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "d_model": 256, "e_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ILI_TimeXer",
        dataset="ILI", model="TimeXer",
        liulian_config="experiments/illness/timexer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
            "d_model": 256, "e_layers": 1, "factor": 3,
        },
    ),
    # ── Mamba ───────────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_Mamba",
        dataset="ETTh1", model="Mamba",
        liulian_config="experiments/etth1/mamba_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "d_model": 128, "d_ff": 16,
            "e_layers": 2, "d_layers": 1,
            "expand": 2, "d_conv": 4,
        },
    ),
    Experiment(
        name="ETTh2_Mamba",
        dataset="ETTh2", model="Mamba",
        liulian_config="experiments/etth2/mamba_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "d_model": 128, "d_ff": 16,
            "e_layers": 2,
            "expand": 2, "d_conv": 4,
        },
    ),
    Experiment(
        name="ETTm1_Mamba",
        dataset="ETTm1", model="Mamba",
        liulian_config="experiments/ettm1/mamba_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "d_model": 128, "d_ff": 16,
            "e_layers": 2,
            "expand": 2, "d_conv": 4,
        },
    ),
    Experiment(
        name="ETTm2_Mamba",
        dataset="ETTm2", model="Mamba",
        liulian_config="experiments/ettm2/mamba_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "d_model": 128, "d_ff": 16,
            "e_layers": 2,
            "expand": 2, "d_conv": 4,
        },
    ),
    Experiment(
        name="Weather_Mamba",
        dataset="Weather", model="Mamba",
        liulian_config="experiments/weather/mamba_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "d_model": 128, "d_ff": 16,
            "e_layers": 2, "d_layers": 1,
            "expand": 2, "d_conv": 4,
        },
    ),
    Experiment(
        name="ECL_Mamba",
        dataset="ECL", model="Mamba",
        liulian_config="experiments/electricity/mamba_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "d_model": 128, "d_ff": 16,
            "e_layers": 2, "d_layers": 1,
            "expand": 2, "d_conv": 4,
        },
    ),
    Experiment(
        name="Traffic_Mamba",
        dataset="Traffic", model="Mamba",
        liulian_config="experiments/traffic/mamba_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "d_model": 128, "d_ff": 16,
            "e_layers": 2, "d_layers": 1,
            "expand": 2, "d_conv": 4,
        },
    ),
    Experiment(
        name="Exchange_Mamba",
        dataset="Exchange", model="Mamba",
        liulian_config="experiments/exchange_rate/mamba_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "d_model": 128, "d_ff": 16,
            "e_layers": 2, "d_layers": 1,
            "expand": 2, "d_conv": 4,
        },
    ),

    # ── Nonstationary Transformer ────────────────────────────────────────
    Experiment(
        name="ETTh1_NonstationaryTransformer",
        dataset="ETTh1", model="Nonstationary_Transformer",
        liulian_config="experiments/etth1/nonstationary_transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "d_model": 128,
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "p_hidden_dims": "256 256", "p_hidden_layers": 2,
        },
    ),
    Experiment(
        name="ETTh2_NonstationaryTransformer",
        dataset="ETTh2", model="Nonstationary_Transformer",
        liulian_config="experiments/etth2/nonstationary_transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "p_hidden_dims": "256 256", "p_hidden_layers": 2,
        },
    ),
    Experiment(
        name="ETTm1_NonstationaryTransformer",
        dataset="ETTm1", model="Nonstationary_Transformer",
        liulian_config="experiments/ettm1/nonstationary_transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "p_hidden_dims": "16 16 16 16", "p_hidden_layers": 4,
        },
    ),
    Experiment(
        name="ETTm2_NonstationaryTransformer",
        dataset="ETTm2", model="Nonstationary_Transformer",
        liulian_config="experiments/ettm2/nonstationary_transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "p_hidden_dims": "256 256 256 256", "p_hidden_layers": 4,
        },
    ),
    Experiment(
        name="ECL_NonstationaryTransformer",
        dataset="ECL", model="Nonstationary_Transformer",
        liulian_config="experiments/electricity/nonstationary_transformer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "d_model": 2048,
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "p_hidden_dims": "256 256", "p_hidden_layers": 2,
        },
    ),
    Experiment(
        name="Weather_NonstationaryTransformer",
        dataset="Weather", model="Nonstationary_Transformer",
        liulian_config="experiments/weather/nonstationary_transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "train_epochs": 3,
            "p_hidden_dims": "256 256", "p_hidden_layers": 2,
        },
    ),
    Experiment(
        name="Traffic_NonstationaryTransformer",
        dataset="Traffic", model="Nonstationary_Transformer",
        liulian_config="experiments/traffic/nonstationary_transformer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "train_epochs": 3,
            "p_hidden_dims": "128 128", "p_hidden_layers": 2,
        },
    ),
    Experiment(
        name="Exchange_NonstationaryTransformer",
        dataset="Exchange", model="Nonstationary_Transformer",
        liulian_config="experiments/exchange_rate/nonstationary_transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "p_hidden_dims": "256 256", "p_hidden_layers": 2,
        },
    ),
    Experiment(
        name="ILI_NonstationaryTransformer",
        dataset="ILI", model="Nonstationary_Transformer",
        liulian_config="experiments/illness/nonstationary_transformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ILI_36_24",
            "enc_in": 7, "dec_in": 7, "c_out": 7,
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "e_layers": 2, "d_layers": 1, "factor": 3,
            "p_hidden_dims": "32 32", "p_hidden_layers": 2,
        },
    ),

    # ── LightTS ──────────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_LightTS",
        dataset="ETTh1", model="LightTS",
        liulian_config="experiments/etth1/lightts_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTh2_LightTS",
        dataset="ETTh2", model="LightTS",
        liulian_config="experiments/etth2/lightts_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm1_LightTS",
        dataset="ETTm1", model="LightTS",
        liulian_config="experiments/ettm1/lightts_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm2_LightTS",
        dataset="ETTm2", model="LightTS",
        liulian_config="experiments/ettm2/lightts_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ECL_LightTS",
        dataset="ECL", model="LightTS",
        liulian_config="experiments/electricity/lightts_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Weather_LightTS",
        dataset="Weather", model="LightTS",
        liulian_config="experiments/weather/lightts_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Traffic_LightTS",
        dataset="Traffic", model="LightTS",
        liulian_config="experiments/traffic/lightts_config.yaml",
        has_tsl_script=False,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Exchange_LightTS",
        dataset="Exchange", model="LightTS",
        liulian_config="experiments/exchange_rate/lightts_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ILI_LightTS",
        dataset="ILI", model="LightTS",
        liulian_config="experiments/illness/lightts_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ILI_36_24",
            "enc_in": 7, "dec_in": 7, "c_out": 7,
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),

    # ── Reformer ─────────────────────────────────────────────────────────
    Experiment(
        name="ETTh1_Reformer",
        dataset="ETTh1", model="Reformer",
        liulian_config="experiments/etth1/reformer_config.yaml",
        has_tsl_script=True,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTh2_Reformer",
        dataset="ETTh2", model="Reformer",
        liulian_config="experiments/etth2/reformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm1_Reformer",
        dataset="ETTm1", model="Reformer",
        liulian_config="experiments/ettm1/reformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm2_Reformer",
        dataset="ETTm2", model="Reformer",
        liulian_config="experiments/ettm2/reformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ECL_Reformer",
        dataset="ECL", model="Reformer",
        liulian_config="experiments/electricity/reformer_config.yaml",
        has_tsl_script=True,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Weather_Reformer",
        dataset="Weather", model="Reformer",
        liulian_config="experiments/weather/reformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Traffic_Reformer",
        dataset="Traffic", model="Reformer",
        liulian_config="experiments/traffic/reformer_config.yaml",
        has_tsl_script=False,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="Exchange_Reformer",
        dataset="Exchange", model="Reformer",
        liulian_config="experiments/exchange_rate/reformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),
    Experiment(
        name="ILI_Reformer",
        dataset="ILI", model="Reformer",
        liulian_config="experiments/illness/reformer_config.yaml",
        has_tsl_script=False,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ILI_36_24",
            "enc_in": 7, "dec_in": 7, "c_out": 7,
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "e_layers": 2, "d_layers": 1, "factor": 3,
        },
    ),

    # ── GPT4TS (liulian-native, no TSL counterpart) ────────────────────
    Experiment(
        name="ETTh1_GPT4TS",
        dataset="ETTh1", model="GPT4TS",
        liulian_config="experiments/etth1/gpt4ts_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "d_model": 768, "d_ff": 768,
            "e_layers": 6, "d_layers": 1,
        },
    ),
    Experiment(
        name="ETTh2_GPT4TS",
        dataset="ETTh2", model="GPT4TS",
        liulian_config="experiments/etth2/gpt4ts_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "d_model": 768, "d_ff": 768,
            "e_layers": 6, "d_layers": 1,
        },
    ),
    Experiment(
        name="ETTm1_GPT4TS",
        dataset="ETTm1", model="GPT4TS",
        liulian_config="experiments/ettm1/gpt4ts_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "d_model": 768, "d_ff": 768,
            "e_layers": 6, "d_layers": 1,
        },
    ),
    Experiment(
        name="ETTm2_GPT4TS",
        dataset="ETTm2", model="GPT4TS",
        liulian_config="experiments/ettm2/gpt4ts_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "d_model": 768, "d_ff": 768,
            "e_layers": 6, "d_layers": 1,
        },
    ),
    Experiment(
        name="Weather_GPT4TS",
        dataset="Weather", model="GPT4TS",
        liulian_config="experiments/weather/gpt4ts_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "d_model": 768, "d_ff": 768,
            "e_layers": 6, "d_layers": 1,
        },
    ),
    Experiment(
        name="ECL_GPT4TS",
        dataset="ECL", model="GPT4TS",
        liulian_config="experiments/electricity/gpt4ts_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "d_model": 768, "d_ff": 768,
            "e_layers": 6, "d_layers": 1,
        },
    ),
    Experiment(
        name="Traffic_GPT4TS",
        dataset="Traffic", model="GPT4TS",
        liulian_config="experiments/traffic/gpt4ts_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "d_model": 768, "d_ff": 768,
            "e_layers": 6, "d_layers": 1,
        },
    ),
    Experiment(
        name="Exchange_GPT4TS",
        dataset="Exchange", model="GPT4TS",
        liulian_config="experiments/exchange_rate/gpt4ts_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "d_model": 768, "d_ff": 768,
            "e_layers": 6, "d_layers": 1,
        },
    ),
    Experiment(
        name="ILI_GPT4TS",
        dataset="ILI", model="GPT4TS",
        liulian_config="experiments/illness/gpt4ts_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "d_model": 768, "d_ff": 768,
            "e_layers": 6, "d_layers": 1,
        },
    ),

    # ── TimeLLM (external reference, no bundled TSL long-term counterpart) ──
    Experiment(
        name="ETTh1_TimeLLM",
        dataset="ETTh1", model="TimeLLM",
        liulian_config="experiments/etth1/timellm_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no bundled TSL long-term counterpart (external Time-LLM repo)",
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
        },
    ),
    Experiment(
        name="ETTh2_TimeLLM",
        dataset="ETTh2", model="TimeLLM",
        liulian_config="experiments/etth2/timellm_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no bundled TSL long-term counterpart (external Time-LLM repo)",
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
        },
    ),
    Experiment(
        name="ETTm1_TimeLLM",
        dataset="ETTm1", model="TimeLLM",
        liulian_config="experiments/ettm1/timellm_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no bundled TSL long-term counterpart (external Time-LLM repo)",
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
        },
    ),
    Experiment(
        name="ETTm2_TimeLLM",
        dataset="ETTm2", model="TimeLLM",
        liulian_config="experiments/ettm2/timellm_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no bundled TSL long-term counterpart (external Time-LLM repo)",
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
        },
    ),
    Experiment(
        name="Weather_TimeLLM",
        dataset="Weather", model="TimeLLM",
        liulian_config="experiments/weather/timellm_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no bundled TSL long-term counterpart (external Time-LLM repo)",
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
        },
    ),
    Experiment(
        name="ECL_TimeLLM",
        dataset="ECL", model="TimeLLM",
        liulian_config="experiments/electricity/timellm_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no bundled TSL long-term counterpart (external Time-LLM repo)",
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
        name="Traffic_TimeLLM",
        dataset="Traffic", model="TimeLLM",
        liulian_config="experiments/traffic/timellm_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no bundled TSL long-term counterpart (external Time-LLM repo)",
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
        name="Exchange_TimeLLM",
        dataset="Exchange", model="TimeLLM",
        liulian_config="experiments/exchange_rate/timellm_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no bundled TSL long-term counterpart (external Time-LLM repo)",
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
        },
    ),
    Experiment(
        name="ILI_TimeLLM",
        dataset="ILI", model="TimeLLM",
        liulian_config="experiments/illness/timellm_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no bundled TSL long-term counterpart (external Time-LLM repo)",
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
        },
    ),

    # ── TimeMoE (TSL model exists, but comparison task mismatch) ───────────
    Experiment(
        name="ETTh1_TimeMoE",
        dataset="ETTh1", model="TimeMoE",
        liulian_config="experiments/etth1/timemoe_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="TimeMoE is zero-shot task in current code, not long_term_forecast",
        tsl_overrides={
            "task_name": "zero_shot_forecast",
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
        },
    ),
    Experiment(
        name="ETTh2_TimeMoE",
        dataset="ETTh2", model="TimeMoE",
        liulian_config="experiments/etth2/timemoe_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="TimeMoE is zero-shot task in current code, not long_term_forecast",
        tsl_overrides={
            "task_name": "zero_shot_forecast",
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
        },
    ),
    Experiment(
        name="ETTm1_TimeMoE",
        dataset="ETTm1", model="TimeMoE",
        liulian_config="experiments/ettm1/timemoe_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="TimeMoE is zero-shot task in current code, not long_term_forecast",
        tsl_overrides={
            "task_name": "zero_shot_forecast",
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
        },
    ),
    Experiment(
        name="ETTm2_TimeMoE",
        dataset="ETTm2", model="TimeMoE",
        liulian_config="experiments/ettm2/timemoe_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="TimeMoE is zero-shot task in current code, not long_term_forecast",
        tsl_overrides={
            "task_name": "zero_shot_forecast",
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
        },
    ),
    Experiment(
        name="Weather_TimeMoE",
        dataset="Weather", model="TimeMoE",
        liulian_config="experiments/weather/timemoe_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="TimeMoE is zero-shot task in current code, not long_term_forecast",
        tsl_overrides={
            "task_name": "zero_shot_forecast",
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
        },
    ),
    Experiment(
        name="ECL_TimeMoE",
        dataset="ECL", model="TimeMoE",
        liulian_config="experiments/electricity/timemoe_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="TimeMoE is zero-shot task in current code, not long_term_forecast",
        large=True,
        tsl_overrides={
            "task_name": "zero_shot_forecast",
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
        },
    ),
    Experiment(
        name="Traffic_TimeMoE",
        dataset="Traffic", model="TimeMoE",
        liulian_config="experiments/traffic/timemoe_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="TimeMoE is zero-shot task in current code, not long_term_forecast",
        large=True,
        tsl_overrides={
            "task_name": "zero_shot_forecast",
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
        },
    ),
    Experiment(
        name="Exchange_TimeMoE",
        dataset="Exchange", model="TimeMoE",
        liulian_config="experiments/exchange_rate/timemoe_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="TimeMoE is zero-shot task in current code, not long_term_forecast",
        tsl_overrides={
            "task_name": "zero_shot_forecast",
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
        },
    ),
    Experiment(
        name="ILI_TimeMoE",
        dataset="ILI", model="TimeMoE",
        liulian_config="experiments/illness/timemoe_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="TimeMoE is zero-shot task in current code, not long_term_forecast",
        tsl_overrides={
            "task_name": "zero_shot_forecast",
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
        },
    ),

    # ── ETSformer (TSL scripts exist for ETTh1/ECL; liulian adapter pending) ─
    Experiment(
        name="ETTh1_ETSformer",
        dataset="ETTh1", model="ETSformer",
        liulian_config="experiments/etth1/etsformer_config.yaml",
        has_tsl_script=True,
        # skip_reason omitted for test
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh1.csv",
            "data": "ETTh1",
            "model_id": "ETTh1_96_96",
            "e_layers": 2, "d_layers": 2, "factor": 3,
        },
    ),
    Experiment(
        name="ETTh2_ETSformer",
        dataset="ETTh2", model="ETSformer",
        liulian_config="experiments/etth2/etsformer_config.yaml",
        has_tsl_script=False,
        # skip_reason omitted for test
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTh2.csv",
            "data": "ETTh2",
            "model_id": "ETTh2_96_96",
            "e_layers": 2, "d_layers": 2, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm1_ETSformer",
        dataset="ETTm1", model="ETSformer",
        liulian_config="experiments/ettm1/etsformer_config.yaml",
        has_tsl_script=False,
        # skip_reason omitted for test
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm1.csv",
            "data": "ETTm1",
            "model_id": "ETTm1_96_96",
            "e_layers": 2, "d_layers": 2, "factor": 3,
        },
    ),
    Experiment(
        name="ETTm2_ETSformer",
        dataset="ETTm2", model="ETSformer",
        liulian_config="experiments/ettm2/etsformer_config.yaml",
        has_tsl_script=False,
        # skip_reason omitted for test
        tsl_overrides={
            "root_path": "./dataset/ETT-small/",
            "data_path": "ETTm2.csv",
            "data": "ETTm2",
            "model_id": "ETTm2_96_96",
            "e_layers": 2, "d_layers": 2, "factor": 3,
        },
    ),
    Experiment(
        name="Weather_ETSformer",
        dataset="Weather", model="ETSformer",
        liulian_config="experiments/weather/etsformer_config.yaml",
        has_tsl_script=False,
        # skip_reason omitted for test
        tsl_overrides={
            "root_path": "./dataset/weather/",
            "data_path": "weather.csv",
            "data": "custom",
            "model_id": "weather_96_96",
            "enc_in": 21, "dec_in": 21, "c_out": 21,
            "e_layers": 2, "d_layers": 2, "factor": 3,
        },
    ),
    Experiment(
        name="ECL_ETSformer",
        dataset="ECL", model="ETSformer",
        liulian_config="experiments/electricity/etsformer_config.yaml",
        has_tsl_script=True,
        # skip_reason omitted for test
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/",
            "data_path": "electricity.csv",
            "data": "custom",
            "model_id": "ECL_96_96",
            "enc_in": 321, "dec_in": 321, "c_out": 321,
            "e_layers": 2, "d_layers": 2, "factor": 3,
        },
    ),
    Experiment(
        name="Traffic_ETSformer",
        dataset="Traffic", model="ETSformer",
        liulian_config="experiments/traffic/etsformer_config.yaml",
        has_tsl_script=False,
        # skip_reason omitted for test
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/",
            "data_path": "traffic.csv",
            "data": "custom",
            "model_id": "traffic_96_96",
            "enc_in": 862, "dec_in": 862, "c_out": 862,
            "e_layers": 2, "d_layers": 2, "factor": 3,
        },
    ),
    Experiment(
        name="Exchange_ETSformer",
        dataset="Exchange", model="ETSformer",
        liulian_config="experiments/exchange_rate/etsformer_config.yaml",
        has_tsl_script=False,
        # skip_reason omitted for test
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/",
            "data_path": "exchange_rate.csv",
            "data": "custom",
            "model_id": "Exchange_96_96",
            "enc_in": 8, "dec_in": 8, "c_out": 8,
            "e_layers": 2, "d_layers": 2, "factor": 3,
        },
    ),
    Experiment(
        name="ILI_ETSformer",
        dataset="ILI", model="ETSformer",
        liulian_config="experiments/illness/etsformer_config.yaml",
        has_tsl_script=False,
        # skip_reason omitted for test
        tsl_overrides={
            "root_path": "./dataset/illness/",
            "data_path": "national_illness.csv",
            "data": "custom",
            "model_id": "ili_36_24",
            "seq_len": 36, "label_len": 18, "pred_len": 24,
            "e_layers": 2, "d_layers": 2, "factor": 3,
        },
    ),

    # ── Stationary (no canonical model in TSL/liulian registry) ────────────
    Experiment(
        name="ETTh1_Stationary",
        dataset="ETTh1", model="Stationary",
        liulian_config="experiments/etth1/stationary_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no canonical Stationary model in current TSL/liulian setup",
        tsl_overrides={
            "root_path": "./dataset/ETT-small/", "data_path": "ETTh1.csv", "data": "ETTh1", "model_id": "ETTh1_96_96",
        },
    ),
    Experiment(
        name="ETTh2_Stationary",
        dataset="ETTh2", model="Stationary",
        liulian_config="experiments/etth2/stationary_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no canonical Stationary model in current TSL/liulian setup",
        tsl_overrides={
            "root_path": "./dataset/ETT-small/", "data_path": "ETTh2.csv", "data": "ETTh2", "model_id": "ETTh2_96_96",
        },
    ),
    Experiment(
        name="ETTm1_Stationary",
        dataset="ETTm1", model="Stationary",
        liulian_config="experiments/ettm1/stationary_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no canonical Stationary model in current TSL/liulian setup",
        tsl_overrides={
            "root_path": "./dataset/ETT-small/", "data_path": "ETTm1.csv", "data": "ETTm1", "model_id": "ETTm1_96_96",
        },
    ),
    Experiment(
        name="ETTm2_Stationary",
        dataset="ETTm2", model="Stationary",
        liulian_config="experiments/ettm2/stationary_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no canonical Stationary model in current TSL/liulian setup",
        tsl_overrides={
            "root_path": "./dataset/ETT-small/", "data_path": "ETTm2.csv", "data": "ETTm2", "model_id": "ETTm2_96_96",
        },
    ),
    Experiment(
        name="Weather_Stationary",
        dataset="Weather", model="Stationary",
        liulian_config="experiments/weather/stationary_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no canonical Stationary model in current TSL/liulian setup",
        tsl_overrides={
            "root_path": "./dataset/weather/", "data_path": "weather.csv", "data": "custom", "model_id": "weather_96_96", "enc_in": 21, "dec_in": 21, "c_out": 21,
        },
    ),
    Experiment(
        name="ECL_Stationary",
        dataset="ECL", model="Stationary",
        liulian_config="experiments/electricity/stationary_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no canonical Stationary model in current TSL/liulian setup",
        large=True,
        tsl_overrides={
            "root_path": "./dataset/electricity/", "data_path": "electricity.csv", "data": "custom", "model_id": "ECL_96_96", "enc_in": 321, "dec_in": 321, "c_out": 321,
        },
    ),
    Experiment(
        name="Traffic_Stationary",
        dataset="Traffic", model="Stationary",
        liulian_config="experiments/traffic/stationary_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no canonical Stationary model in current TSL/liulian setup",
        large=True,
        tsl_overrides={
            "root_path": "./dataset/traffic/", "data_path": "traffic.csv", "data": "custom", "model_id": "traffic_96_96", "enc_in": 862, "dec_in": 862, "c_out": 862,
        },
    ),
    Experiment(
        name="Exchange_Stationary",
        dataset="Exchange", model="Stationary",
        liulian_config="experiments/exchange_rate/stationary_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no canonical Stationary model in current TSL/liulian setup",
        tsl_overrides={
            "root_path": "./dataset/exchange_rate/", "data_path": "exchange_rate.csv", "data": "custom", "model_id": "Exchange_96_96", "enc_in": 8, "dec_in": 8, "c_out": 8,
        },
    ),
    Experiment(
        name="ILI_Stationary",
        dataset="ILI", model="Stationary",
        liulian_config="experiments/illness/stationary_config.yaml",
        has_tsl_script=False,
        tsl_comparable=False,
        skip_reason="no canonical Stationary model in current TSL/liulian setup",
        tsl_overrides={
            "root_path": "./dataset/illness/", "data_path": "national_illness.csv", "data": "custom", "model_id": "ili_36_24", "seq_len": 36, "label_len": 18, "pred_len": 24,
        },
    ),

    # ── Additional missing datasets requested by history (PatchTST/DLinear) ──
    Experiment(
        name="Solar_PatchTST",
        dataset="Solar-Energy", model="PatchTST",
        liulian_config="experiments/solar_energy/patchtst_config.yaml",
        has_tsl_script=False,
        skip_reason="dataset/config not integrated in liulian long_term comparison",
        tsl_overrides={"root_path": "./dataset/Solar/", "data_path": "solar_AL.txt", "data": "Solar", "model_id": "Solar_96_96"},
    ),
    Experiment(
        name="Solar_DLinear",
        dataset="Solar-Energy", model="DLinear",
        liulian_config="experiments/solar_energy/dlinear_config.yaml",
        has_tsl_script=False,
        skip_reason="dataset/config not integrated in liulian long_term comparison",
        tsl_overrides={"root_path": "./dataset/Solar/", "data_path": "solar_AL.txt", "data": "Solar", "model_id": "Solar_96_96"},
    ),
    Experiment(
        name="PEMS03_PatchTST",
        dataset="PEMS03", model="PatchTST",
        liulian_config="experiments/pems03/patchtst_config.yaml",
        has_tsl_script=False,
        skip_reason="TSL PEMS scripts are short-term; long-term compare path not aligned",
        tsl_overrides={"data": "PEMS", "root_path": "./dataset/PEMS03/", "data_path": "PEMS03.npz", "model_id": "PEMS03_96_12"},
    ),
    Experiment(
        name="PEMS03_DLinear",
        dataset="PEMS03", model="DLinear",
        liulian_config="experiments/pems03/dlinear_config.yaml",
        has_tsl_script=False,
        skip_reason="TSL PEMS scripts are short-term; long-term compare path not aligned",
        tsl_overrides={"data": "PEMS", "root_path": "./dataset/PEMS03/", "data_path": "PEMS03.npz", "model_id": "PEMS03_96_12"},
    ),
    Experiment(
        name="PEMS04_PatchTST",
        dataset="PEMS04", model="PatchTST",
        liulian_config="experiments/pems04/patchtst_config.yaml",
        has_tsl_script=False,
        skip_reason="TSL PEMS scripts are short-term; long-term compare path not aligned",
        tsl_overrides={"data": "PEMS", "root_path": "./dataset/PEMS04/", "data_path": "PEMS04.npz", "model_id": "PEMS04_96_12"},
    ),
    Experiment(
        name="PEMS04_DLinear",
        dataset="PEMS04", model="DLinear",
        liulian_config="experiments/pems04/dlinear_config.yaml",
        has_tsl_script=False,
        skip_reason="TSL PEMS scripts are short-term; long-term compare path not aligned",
        tsl_overrides={"data": "PEMS", "root_path": "./dataset/PEMS04/", "data_path": "PEMS04.npz", "model_id": "PEMS04_96_12"},
    ),
    Experiment(
        name="PEMS07_PatchTST",
        dataset="PEMS07", model="PatchTST",
        liulian_config="experiments/pems07/patchtst_config.yaml",
        has_tsl_script=False,
        skip_reason="TSL PEMS scripts are short-term; long-term compare path not aligned",
        tsl_overrides={"data": "PEMS", "root_path": "./dataset/PEMS07/", "data_path": "PEMS07.npz", "model_id": "PEMS07_96_12"},
    ),
    Experiment(
        name="PEMS07_DLinear",
        dataset="PEMS07", model="DLinear",
        liulian_config="experiments/pems07/dlinear_config.yaml",
        has_tsl_script=False,
        skip_reason="TSL PEMS scripts are short-term; long-term compare path not aligned",
        tsl_overrides={"data": "PEMS", "root_path": "./dataset/PEMS07/", "data_path": "PEMS07.npz", "model_id": "PEMS07_96_12"},
    ),
    Experiment(
        name="PEMS08_PatchTST",
        dataset="PEMS08", model="PatchTST",
        liulian_config="experiments/pems08/patchtst_config.yaml",
        has_tsl_script=False,
        skip_reason="TSL PEMS scripts are short-term; long-term compare path not aligned",
        tsl_overrides={"data": "PEMS", "root_path": "./dataset/PEMS08/", "data_path": "PEMS08.npz", "model_id": "PEMS08_96_12"},
    ),
    Experiment(
        name="PEMS08_DLinear",
        dataset="PEMS08", model="DLinear",
        liulian_config="experiments/pems08/dlinear_config.yaml",
        has_tsl_script=False,
        skip_reason="TSL PEMS scripts are short-term; long-term compare path not aligned",
        tsl_overrides={"data": "PEMS", "root_path": "./dataset/PEMS08/", "data_path": "PEMS08.npz", "model_id": "PEMS08_96_12"},
    ),
    Experiment(
        name="CovidDeaths_PatchTST",
        dataset="Covid Deaths", model="PatchTST",
        liulian_config="experiments/covid_deaths/patchtst_config.yaml",
        has_tsl_script=False,
        skip_reason="no canonical TSL script/config mapping in current repository",
        tsl_overrides={"model_id": "covid_deaths_96_96"},
    ),
    Experiment(
        name="CovidDeaths_DLinear",
        dataset="Covid Deaths", model="DLinear",
        liulian_config="experiments/covid_deaths/dlinear_config.yaml",
        has_tsl_script=False,
        skip_reason="no canonical TSL script/config mapping in current repository",
        tsl_overrides={"model_id": "covid_deaths_96_96"},
    ),
    Experiment(
        name="NYCTaxi_PatchTST",
        dataset="NYC Taxi", model="PatchTST",
        liulian_config="experiments/nyc_taxi/patchtst_config.yaml",
        has_tsl_script=False,
        skip_reason="no canonical TSL script/config mapping in current repository",
        tsl_overrides={"model_id": "nyc_taxi_96_96"},
    ),
    Experiment(
        name="NYCTaxi_DLinear",
        dataset="NYC Taxi", model="DLinear",
        liulian_config="experiments/nyc_taxi/dlinear_config.yaml",
        has_tsl_script=False,
        skip_reason="no canonical TSL script/config mapping in current repository",
        tsl_overrides={"model_id": "nyc_taxi_96_96"},
    ),
    Experiment(
        name="NN5_PatchTST",
        dataset="NN5", model="PatchTST",
        liulian_config="experiments/nn5/patchtst_config.yaml",
        has_tsl_script=False,
        skip_reason="no canonical TSL script/config mapping in current repository",
        tsl_overrides={"model_id": "nn5_96_56"},
    ),
    Experiment(
        name="NN5_DLinear",
        dataset="NN5", model="DLinear",
        liulian_config="experiments/nn5/dlinear_config.yaml",
        has_tsl_script=False,
        skip_reason="no canonical TSL script/config mapping in current repository",
        tsl_overrides={"model_id": "nn5_96_56"},
    ),
    Experiment(
        name="FREDMD_PatchTST",
        dataset="FRED-MD", model="PatchTST",
        liulian_config="experiments/fred_md/patchtst_config.yaml",
        has_tsl_script=False,
        skip_reason="no canonical TSL script/config mapping in current repository",
        tsl_overrides={"model_id": "fred_md_96_96"},
    ),
    Experiment(
        name="FREDMD_DLinear",
        dataset="FRED-MD", model="DLinear",
        liulian_config="experiments/fred_md/dlinear_config.yaml",
        has_tsl_script=False,
        skip_reason="no canonical TSL script/config mapping in current repository",
        tsl_overrides={"model_id": "fred_md_96_96"},
    ),
]

# ---------------------------------------------------------------------------
# TSL command builder
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_liulian_config_backfills(config_rel_path: str) -> dict[str, object]:
    """Load dataset-facing backfills from the liulian YAML config.

    These are only used when a TSL script/pair definition leaves a setting
    implicit. The most important case is ``freq``: many bundled TSL scripts
    omit ``--freq``, which otherwise falls back to run.py's hourly default
    even for daily / weekly / minute datasets.
    """
    config_path = PROJECT_ROOT / config_rel_path
    try:
        data = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    backfills: dict[str, object] = {}
    freq = data.get('freq')
    if freq not in (None, ''):
        backfills['freq'] = normalize_tsl_freq(freq)
    return backfills


def normalize_tsl_freq(freq: object) -> object:
    """Normalize liulian frequency values to TSL's expected tokens.

    TSL's ``TimeFeatureEmbedding`` expects compact codes like ``h``, ``d``, ``t``,
    while liulian configs may use readable values such as ``15min``.
    """
    if not isinstance(freq, str):
        return freq
    f = freq.strip().lower()
    if not f:
        return freq

    # Minute-level aliases (TSL uses "t").
    if f in {'t', 'min', 'mins', 'minute', 'minutes'}:
        return 't'
    if re.fullmatch(r'\d+\s*(min|mins|minute|minutes)', f):
        return 't'

    # Common aliases for other granularities.
    alias_map = {
        'hour': 'h',
        'hourly': 'h',
        '1h': 'h',
        'day': 'd',
        'daily': 'd',
        '1d': 'd',
        'week': 'w',
        'weekly': 'w',
        'month': 'm',
        'monthly': 'm',
        'year': 'a',
        'yearly': 'a',
        'annual': 'a',
        'business': 'b',
        'businessday': 'b',
        'second': 's',
        'secondly': 's',
    }
    return alias_map.get(f, freq)


def build_tsl_cmd(
    exp: Experiment,
    epoch_limit: Optional[int] = None,
    disable_es: bool = False,
    extra_overrides: Optional[dict] = None,
) -> list[str]:
    """Build the TSL run.py command for an experiment."""
    args = dict(TSL_BASE_ARGS)
    args["model"] = exp.model
    args.update(exp.tsl_overrides)
    for key, value in load_liulian_config_backfills(exp.liulian_config).items():
        args.setdefault(key, value)
    if extra_overrides:
        args.update(extra_overrides)
    if epoch_limit is not None:
        args["train_epochs"] = epoch_limit
    if disable_es:
        args["patience"] = 9999  # effectively disable early stopping

    cmd = [PYTHON, "-u", "run.py"]
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
            continue

        if k == "use_amp":
            sv_amp = str(v).strip().lower()
            if sv_amp in {"1", "true", "yes", "on"}:
                cmd.append("--use_amp")
                continue
            if sv_amp in {"0", "false", "no", "off"}:
                continue

        sv = str(v)
        # Some TSL args are nargs='+' (e.g., --p_hidden_dims 256 256).
        # If the value contains spaces, split into separate arguments.
        if " " in sv:
            cmd.append(f"--{k}")
            cmd.extend(sv.split())
        else:
            cmd.extend([f"--{k}", sv])
    return cmd


def build_liulian_cmd(
    exp: Experiment,
    epoch_limit: Optional[int] = None,
    disable_es: bool = False,
    tsl_extra_overrides: Optional[dict] = None,
    cli_overrides: Optional[dict] = None,
) -> list[str]:
    """Build the liulian experiments/run.py command."""
    config_path = str(PROJECT_ROOT / exp.liulian_config)
    cmd = [PYTHON, str(PROJECT_ROOT / "experiments" / "run.py"),
           "--config", config_path]
    if epoch_limit is not None:
        cmd.extend(["--train_epochs", str(epoch_limit)])
    if disable_es:
        cmd.append("--disable_early_stopping")
    
    effective_tsl_overrides = dict(exp.tsl_overrides)
    if tsl_extra_overrides:
        effective_tsl_overrides.update(tsl_extra_overrides)

    # MATCH TSL RANDOM SEED (TSL uses fix_seed=2021 hardcoded in run.py)
    # Liulian defaults to seed=2026, so we override it to match TSL
    cmd.extend(["--seed", "2021"])

    # MATCH TSL SCALING (TSL defaults to inverse=False)
    if not effective_tsl_overrides.get("inverse", False):
        cmd.append("--no_eval_denorm")

    if cli_overrides:
        for key, value in cli_overrides.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
                else:
                    cmd.append(f'--no_{key}')
                continue
            cmd.extend([f'--{key}', str(value)])

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

def run_cmd(
    cmd: list[str],
    cwd: str,
    label: str,
    timeout_seconds: int = 7200,
    progress_interval_seconds: int = 30,
) -> tuple[str, float, int]:
    """Run a command, return (combined_output, elapsed_seconds, returncode)."""
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  CWD: {cwd}")
    print(f"  CMD: {' '.join(cmd[:6])} ...")
    print(f"{'='*60}")

    # Progress behaviour policy:
    # - TSL reference long-term pipeline (run.py + exp/*.py) does not expose
    #   a reusable global progress-bar API; it prints periodic textual logs.
    # - liulian uses tqdm in trainer loops, but those are local batch/epoch bars
    #   rather than one global per-run pipeline bar.
    # Therefore we keep a runner-level fallback progress bar based on timeout
    # budget, which is stable across both pipelines.
    print("  Progress mode: fallback runner progress bar")

    t0 = time.time()
    popen = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    while True:
        elapsed = time.time() - t0
        if elapsed >= timeout_seconds:
            popen.kill()
            out, _ = popen.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout_seconds, output=out)

        wait_s = min(progress_interval_seconds, max(timeout_seconds - elapsed, 0.1))
        try:
            out, _ = popen.communicate(timeout=wait_s)
            elapsed = time.time() - t0
            print(f"  Finished in {elapsed:.1f}s (exit code {popen.returncode})")
            return out, elapsed, popen.returncode
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            pct = min(99.9, (elapsed / timeout_seconds) * 100.0)
            bar_width = 24
            filled = int((pct / 100.0) * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"  ... {label} [{bar}] {pct:5.1f}% "
                f"({elapsed:.1f}s / {timeout_seconds}s timeout budget)"
            )


def _tail_lines(text: str, n: int = 40) -> str:
    """Return the last *n* lines of text for compact error reporting."""
    lines = text.strip().splitlines()
    if not lines:
        return ''
    return '\n'.join(lines[-n:])


def is_completed_result(entry: dict) -> bool:
    """Return True if a stored result counts as a completed comparison."""
    status = entry.get('status', '')
    return status in COMPLETED_STATUSES or (isinstance(status, str) and status.startswith('skipped'))


def parse_runtime_overrides_env() -> dict[str, dict]:
    """Parse optional pair runtime overrides from LIULIAN_COMPARE_RUNTIME_OVERRIDES.

    Expected JSON shape:
    {
      "PairName": {
        "tsl_overrides": {...},
        "liulian_cli_overrides": {...}
      }
    }
    """
    raw = os.getenv(RUNTIME_OVERRIDES_ENV_VAR)
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        print(
            f'Warning: failed to parse {RUNTIME_OVERRIDES_ENV_VAR}: {exc}. '
            'Ignoring runtime overrides.'
        )
        return {}

    if not isinstance(parsed, dict):
        print(
            f'Warning: {RUNTIME_OVERRIDES_ENV_VAR} must be a JSON object. '
            'Ignoring runtime overrides.'
        )
        return {}

    normalized: dict[str, dict] = {}
    for pair_name, pair_cfg in parsed.items():
        if not isinstance(pair_name, str) or not isinstance(pair_cfg, dict):
            continue

        tsl_overrides = pair_cfg.get('tsl_overrides', {})
        liulian_cli_overrides = pair_cfg.get('liulian_cli_overrides', {})
        if not isinstance(tsl_overrides, dict) or not isinstance(liulian_cli_overrides, dict):
            continue

        normalized[pair_name] = {
            'tsl_overrides': dict(tsl_overrides),
            'liulian_cli_overrides': dict(liulian_cli_overrides),
        }
    return normalized


def merge_runtime_overrides(
    base_overrides: dict[str, dict],
    extra_overrides: dict[str, dict],
) -> dict[str, dict]:
    """Merge pair runtime overrides with per-scope dict update semantics.

    Known scopes:
    - tsl_overrides
    - liulian_cli_overrides
    """
    merged: dict[str, dict] = {}

    for source in (base_overrides, extra_overrides):
        for pair_name, pair_cfg in source.items():
            if not isinstance(pair_name, str) or not isinstance(pair_cfg, dict):
                continue

            target = merged.setdefault(
                pair_name,
                {
                    'tsl_overrides': {},
                    'liulian_cli_overrides': {},
                },
            )

            tsl_overrides = pair_cfg.get('tsl_overrides')
            if isinstance(tsl_overrides, dict):
                target['tsl_overrides'].update(tsl_overrides)

            liulian_cli_overrides = pair_cfg.get('liulian_cli_overrides')
            if isinstance(liulian_cli_overrides, dict):
                target['liulian_cli_overrides'].update(liulian_cli_overrides)

    return merged


def build_smoke_tsl_data_overrides(exp: Experiment, max_rows: int) -> dict[str, str]:
    """Create a small CSV subset for TSL-side smoke testing.

    Returns TSL CLI overrides (`root_path`, `data_path`) when source paths are
    available for this experiment; otherwise returns an empty dict.
    """
    root_path = exp.tsl_overrides.get('root_path')
    data_path = exp.tsl_overrides.get('data_path')
    if not root_path or not data_path:
        return {}

    src_path = (TSL_ROOT / str(root_path) / str(data_path)).resolve()
    if not src_path.exists():
        print(f"Warning: smoke source CSV not found for {exp.name}: {src_path}")
        return {}

    smoke_dir = PROJECT_ROOT / 'artifacts' / 'tsl_smoke_data' / exp.name
    smoke_dir.mkdir(parents=True, exist_ok=True)
    dst_path = smoke_dir / src_path.name

    seq_len = int(exp.tsl_overrides.get('seq_len', TSL_BASE_ARGS.get('seq_len', 96)))
    pred_len = int(exp.tsl_overrides.get('pred_len', TSL_BASE_ARGS.get('pred_len', 96)))
    min_required_rows = max(seq_len + pred_len + 64, pred_len * 10 + 64)
    effective_rows = max(max_rows, min_required_rows)
    line_limit = max(2, effective_rows + 1)
    with src_path.open('r', encoding='utf-8') as src_f, dst_path.open('w', encoding='utf-8') as dst_f:
        for line_idx, line in enumerate(src_f):
            dst_f.write(line)
            if line_idx + 1 >= line_limit:
                break

    rel_root = os.path.relpath(smoke_dir, TSL_ROOT).replace(os.sep, '/')
    if not rel_root.endswith('/'):
        rel_root += '/'

    return {
        'root_path': f'./{rel_root}',
        'data_path': dst_path.name,
    }


def collect_special_settings(
    args: argparse.Namespace,
    exp: Experiment,
    tsl_runtime_overrides: dict,
    ll_cli_runtime_overrides: dict,
) -> dict[str, object]:
    tags: list[str] = []

    if args.oom_fallback and exp.name in OOM_FALLBACK_PROFILES:
        tags.append('oom_fallback')
    if args.smoke_test:
        tags.append('smoke_test')
    if args.single_pair_diagnostic:
        tags.append('single_pair_diagnostic')
    if args.disable_es:
        tags.append('disable_es')
    if args.timeout_seconds != 7200:
        tags.append('custom_timeout')
    if tsl_runtime_overrides or ll_cli_runtime_overrides:
        tags.append('runtime_overrides')

    return {
        'special_settings_applied': bool(tags),
        'special_settings_tags': tags,
        'special_settings_detail': {
            'tsl_overrides': dict(tsl_runtime_overrides),
            'liulian_cli_overrides': dict(ll_cli_runtime_overrides),
            'timeout_seconds': int(args.timeout_seconds),
            'progress_interval_seconds': int(args.progress_interval_seconds),
        },
    }


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
        '--remaining-only', action='store_true',
        help=(
            'Skip pairs that already have non-dry-run results in an existing '
            'JSON results file, and run only the remaining selected pairs.'
        ),
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
    parser.add_argument(
        "--oom-fallback", action="store_true",
        help=(
            "Apply built-in OOM fallback runtime overrides for heavy pairs "
            "(Traffic_TimesNet, ILI_TimesNet, Traffic_TimeXer): "
            "TSL and liulian both use aligned low-memory settings "
            "(batch_size/d_model/d_ff/n_heads/train_epochs; TimeXer learning_rate). "
            "No environment variable needed."
        ),
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=7200,
        help="Per-process timeout in seconds for each TSL/liulian run (default: 7200).",
    )
    parser.add_argument(
        "--progress-interval-seconds", type=int, default=30,
        help="Heartbeat print interval in seconds while a TSL/liulian process is running.",
    )
    parser.add_argument(
        "--single-pair-diagnostic", action="store_true",
        help=(
            "Require exactly one selected pair and run in diagnostic single-pair mode. "
            "Useful for heavy cases to isolate runtime issues and avoid multi-pair queues."
        ),
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=(
            "Run a real lightweight execution (not dry-run) on small data slices "
            "for quick pipeline sanity checks. "
            "TSL uses sampled CSV rows; liulian uses max_samples + quick_test."
        ),
    )
    parser.add_argument(
        "--smoke-max-rows", type=int, default=1024,
        help="Max data rows (excluding header) kept in TSL sampled CSV for smoke test.",
    )
    parser.add_argument(
        "--smoke-max-samples", type=int, default=64,
        help="Liulian --max_samples value for smoke test mode.",
    )
    parser.add_argument(
        "--smoke-train-epochs", type=int, default=1,
        help="Train epochs used in smoke test mode.",
    )
    args = parser.parse_args()
    runtime_overrides_by_pair = parse_runtime_overrides_env()
    if args.oom_fallback:
        runtime_overrides_by_pair = merge_runtime_overrides(
            OOM_FALLBACK_PROFILES,
            runtime_overrides_by_pair,
        )

    experiments = EXPERIMENTS
    if args.pairs:
        names = set(args.pairs)
        experiments = [e for e in EXPERIMENTS if e.name in names]
        if not experiments:
            print(f"No matching experiments for: {args.pairs}")
            print(f"Available: {[e.name for e in EXPERIMENTS]}")
            sys.exit(1)

    selected_experiments = experiments
    selected_names = {e.name for e in selected_experiments}

    if args.timeout_seconds <= 0:
        print("--timeout-seconds must be > 0")
        sys.exit(2)
    if args.progress_interval_seconds <= 0:
        print("--progress-interval-seconds must be > 0")
        sys.exit(2)

    if args.single_pair_diagnostic and len(experiments) != 1:
        print(
            "--single-pair-diagnostic requires exactly one selected pair. "
            "Use --pairs with a single name."
        )
        sys.exit(2)

    existing_results_by_name: dict[str, dict] = {}
    if args.remaining_only:
        for json_path in RESULTS_JSON_FILES:
            if not json_path.exists():
                continue
            try:
                with open(json_path) as f:
                    loaded = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f'Warning: failed to read existing results from {json_path}: {exc}')
                continue

            if not isinstance(loaded, list):
                print(f'Warning: ignoring non-list results file: {json_path}')
                continue

            loaded_count = 0
            for entry in loaded:
                if not isinstance(entry, dict):
                    continue
                name = entry.get('name')
                if not name or not is_completed_result(entry):
                    continue
                if name not in selected_names:
                    continue
                if name not in existing_results_by_name:
                    loaded_count += 1
                existing_results_by_name[name] = entry

            if loaded_count:
                print(f'Loaded {loaded_count} unique completed result(s) from {json_path}')

        remaining = [e for e in experiments if e.name not in existing_results_by_name]
        skipped = len(experiments) - len(remaining)
        print(
            f'Remaining-only mode: {skipped} already completed, '
            f'{len(remaining)} remaining.'
        )
        experiments = remaining

        if not experiments:
            print('No remaining experiments to run.')
            return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for exp in experiments:
        print(f"\n{'#'*70}")
        print(f"# {exp.name}  (large={exp.large}, has_script={exp.has_tsl_script})")
        print(f"{'#'*70}")

        epoch_limit = args.smoke_train_epochs if args.smoke_test else (2 if exp.large else None)
        disable_es = getattr(args, 'disable_es', False)

        if exp.skip_reason is not None or not exp.tsl_comparable:
            reason = exp.skip_reason
            if reason is None:
                reason = 'no tsl counterpart'
            status = f'skipped: {reason}'
            if not exp.tsl_comparable and exp.skip_reason is None:
                detail = (
                    f"Model {exp.model} has no counterpart in the bundled "
                    f"Time-Series-Library reference repo; pair is tracked but "
                    f"not directly comparable."
                )
            else:
                detail = (
                    f"Pair {exp.name} is tracked in comparison inventory but "
                    f"skipped: {reason}."
                )
            r = {
                "name": exp.name,
                "dataset": exp.dataset,
                "model": exp.model,
                "has_tsl_script": exp.has_tsl_script,
                "large": exp.large,
                "epoch_limit": epoch_limit,
                "tsl_epochs_run": 0,
                "ll_epochs_run": 0,
                "tsl_time_s": 0.0,
                "ll_time_s": 0.0,
                "tsl_final_mse": None,
                "tsl_final_mae": None,
                "ll_final_mse": None,
                "ll_final_mae": None,
                "status": status,
                "detail": detail,
            }
            results.append(r)
            print(f"\n  ── Result: {status} ──")
            print(detail)
            continue

        pair_runtime = runtime_overrides_by_pair.get(exp.name, {})
        tsl_runtime_overrides = dict(pair_runtime.get('tsl_overrides') or {})
        ll_cli_runtime_overrides = dict(pair_runtime.get('liulian_cli_overrides') or {})

        if args.smoke_test:
            tsl_runtime_overrides.update(build_smoke_tsl_data_overrides(exp, args.smoke_max_rows))
            tsl_runtime_overrides.update(
                {
                    'use_amp': False,
                    'batch_size': 1,
                }
            )

            if exp.model in {'TimesNet', 'TimeXer'}:
                tsl_runtime_overrides.update(
                    {
                        'd_model': 128,
                        'd_ff': 256,
                        'n_heads': 4,
                    }
                )

            ll_cli_runtime_overrides.update(
                {
                    'max_samples': args.smoke_max_samples,
                    'quick_test': True,
                    'batch_size': 1,
                }
            )

            if exp.model in {'TimesNet', 'TimeXer'}:
                ll_cli_runtime_overrides.update(
                    {
                        'd_model': 128,
                        'd_ff': 256,
                        'n_heads': 4,
                    }
                )

        special_settings = collect_special_settings(
            args=args,
            exp=exp,
            tsl_runtime_overrides=tsl_runtime_overrides,
            ll_cli_runtime_overrides=ll_cli_runtime_overrides,
        )

        tsl_cmd = build_tsl_cmd(
            exp,
            epoch_limit=epoch_limit,
            disable_es=disable_es,
            extra_overrides=tsl_runtime_overrides,
        )
        ll_cmd = build_liulian_cmd(
            exp,
            epoch_limit=epoch_limit,
            disable_es=disable_es,
            tsl_extra_overrides=tsl_runtime_overrides,
            cli_overrides=ll_cli_runtime_overrides,
        )

        if args.dry_run:
            print(f"  TSL cmd: {' '.join(tsl_cmd)}")
            print(f"  LL  cmd: {' '.join(ll_cmd)}")
            results.append({
                "name": exp.name,
                "dataset": exp.dataset,
                "model": exp.model,
                "has_tsl_script": exp.has_tsl_script,
                "large": exp.large,
                "epoch_limit": epoch_limit,
                "status": "dry-run",
                "tsl_cmd": " ".join(tsl_cmd),
                "ll_cmd": " ".join(ll_cmd),
                **special_settings,
            })
            continue

        # --- Run TSL ---
        try:
            tsl_out, tsl_time, tsl_rc = run_cmd(
                tsl_cmd,
                cwd=str(TSL_ROOT),
                label=f"TSL {exp.name}",
                timeout_seconds=args.timeout_seconds,
                progress_interval_seconds=args.progress_interval_seconds,
            )
        except subprocess.TimeoutExpired:
            tsl_out, tsl_time, tsl_rc = '', -1, -1
            print("  TSL TIMEOUT")

        tsl_metrics = parse_tsl_output(tsl_out)

        # --- Run liulian ---
        try:
            ll_out, ll_time, ll_rc = run_cmd(
                ll_cmd,
                cwd=str(PROJECT_ROOT),
                label=f"liulian {exp.name}",
                timeout_seconds=args.timeout_seconds,
                progress_interval_seconds=args.progress_interval_seconds,
            )
        except subprocess.TimeoutExpired:
            ll_out, ll_time, ll_rc = '', -1, -1
            print("  liulian TIMEOUT")

        ll_metrics = parse_liulian_output(ll_out)

        # --- Compare ---
        if tsl_rc != 0:
            status = 'run failed'
            detail = (
                f'TSL command failed with exit code {tsl_rc}.\n'
                f'Last output lines:\n{_tail_lines(tsl_out)}'
            )
        elif ll_rc != 0:
            status = 'run failed'
            detail = (
                f'liulian command failed with exit code {ll_rc}.\n'
                f'Last output lines:\n{_tail_lines(ll_out)}'
            )
        else:
            matched, detail = compare_metrics(tsl_metrics, ll_metrics, exp.large)
            status = 'checked and matched' if matched else 'checked but not matched'

        tsl_epochs_run = len(tsl_metrics["epochs"])
        ll_epochs_run = ll_metrics.get("epochs_run") or len(ll_metrics["epochs"])

        r = {
            "name": exp.name,
            "dataset": exp.dataset,
            "model": exp.model,
            "smoke_test": bool(args.smoke_test),
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
            **special_settings,
        }
        results.append(r)

        print(f"\n  ── Result: {status} ──")
        print(detail)
        print(f"  TSL: {tsl_epochs_run} epochs in {tsl_time:.1f}s")
        print(f"  LL:  {ll_epochs_run} epochs in {ll_time:.1f}s")

    results_to_write = results
    if args.remaining_only:
        merged_by_name = {
            name: dict(existing_results_by_name[name])
            for name in selected_names
            if name in existing_results_by_name
        }
        for r in results:
            merged_by_name[r['name']] = r
        results_to_write = [
            merged_by_name[e.name]
            for e in selected_experiments
            if e.name in merged_by_name
        ]
    else:
        # When running specific --pairs, merge new results with all prior completed
        # results so we never overwrite already-recorded data.
        all_existing: dict = {}
        for jf_path in RESULTS_JSON_FILES:
            p = Path(jf_path)
            if p.exists():
                try:
                    existing = json.loads(p.read_text())
                    if isinstance(existing, list):
                        for entry in existing:
                            name = entry.get('name')
                            if name and name not in all_existing:
                                all_existing[name] = entry
                except Exception:
                    pass
        # New results override existing entries of the same name
        for r in results:
            all_existing[r['name']] = r
        results_to_write = list(all_existing.values())

    # --- Write results file ---
    with open(RESULTS_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TSL vs Liulian Comparison Results\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for r in results_to_write:
            f.write(f"--- {r['name']} ---\n")
            if r.get("status") == "dry-run":
                f.write(f"  Status: dry-run\n")
                f.write(f"  TSL cmd: {r.get('tsl_cmd', '')}\n")
                f.write(f"  LL  cmd: {r.get('ll_cmd', '')}\n")
            else:
                f.write(f"  Dataset: {r.get('dataset', '')}\n")
                f.write(f"  Model: {r.get('model', '')}\n")
                f.write(f"  Special settings applied: {r.get('special_settings_applied', False)}\n")
                if r.get('special_settings_tags'):
                    f.write(f"  Special settings tags: {', '.join(r.get('special_settings_tags', []))}\n")
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
                f"{'TSL time':>9} {'LL time':>8} {'Special':>8} {'Status'}\n")
        f.write("-" * 80 + "\n")
        for r in results_to_write:
            if r.get("status") == "dry-run":
                f.write(f"{r['name']:<25} {'---':>6} {'---':>5} "
                        f"{'---':>9} {'---':>8} {'yes' if r.get('special_settings_applied') else 'no':>8} dry-run\n")
            else:
                f.write(
                    f"{r['name']:<25} "
                    f"{r.get('tsl_epochs_run', '?'):>6} "
                    f"{r.get('ll_epochs_run', '?'):>5} "
                    f"{r.get('tsl_time_s', '?'):>8}s "
                    f"{r.get('ll_time_s', '?'):>7}s "
                    f"{'yes' if r.get('special_settings_applied') else 'no':>8} "
                    f"{r['status']}\n"
                )

    # Also write JSON for programmatic use
    json_file = RESULTS_DIR / "tsl_comparison_results.json"
    # Remove non-serialisable bits
    json_results = []
    for r in results_to_write:
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
