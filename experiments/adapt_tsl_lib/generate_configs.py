#!/usr/bin/env python3
"""Generate per-dataset YAML configs for all model × dataset pairs.

Produces config files under ``experiments/{dataset}/{model}_config.yaml``
matching the liulian per-dataset schema (consumed by ``experiments/run.py``).

Hyperparameters are sourced from TSL benchmark scripts where available;
TSL defaults are used otherwise.

Usage (from project root):
    .venv/bin/python experiments/adapt_tsl_lib/generate_configs.py
    .venv/bin/python experiments/adapt_tsl_lib/generate_configs.py --dry-run
    .venv/bin/python experiments/adapt_tsl_lib/generate_configs.py --models informer autoformer
    .venv/bin/python experiments/adapt_tsl_lib/generate_configs.py --datasets etth1 electricity
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ═══════════════════════════════════════════════════════════════════════
# Dataset metadata
# ═══════════════════════════════════════════════════════════════════════

DATASETS: dict[str, dict] = {
    "etth1": {
        "data": "ETTh1",
        "channels": 7,
        "target": "OT",
        "freq": "h",
        "seq_len": 96, "pred_len": 96, "label_len": 48,
        "description": "ETT hour-level benchmark (ETTh1). 7 transformer features.",
    },
    "etth2": {
        "data": "ETTh2",
        "channels": 7,
        "target": "OT",
        "freq": "h",
        "seq_len": 96, "pred_len": 96, "label_len": 48,
        "description": "ETT hour-level benchmark (ETTh2). 7 transformer features.",
    },
    "ettm1": {
        "data": "ETTm1",
        "channels": 7,
        "target": "OT",
        "freq": "t",
        "seq_len": 96, "pred_len": 96, "label_len": 48,
        "description": "ETT 15-minute benchmark (ETTm1). 7 transformer features.",
    },
    "ettm2": {
        "data": "ETTm2",
        "channels": 7,
        "target": "OT",
        "freq": "t",
        "seq_len": 96, "pred_len": 96, "label_len": 48,
        "description": "ETT 15-minute benchmark (ETTm2). 7 transformer features.",
    },
    "electricity": {
        "data": "electricity",
        "channels": 321,
        "target": None,
        "freq": "h",
        "seq_len": 96, "pred_len": 96, "label_len": 48,
        "description": "Electricity dataset: 321 clients' hourly consumption.",
    },
    "weather": {
        "data": "weather",
        "channels": 21,
        "target": None,
        "freq": "t",
        "seq_len": 96, "pred_len": 96, "label_len": 48,
        "description": "Weather dataset: 21 meteorological indicators.",
    },
    "traffic": {
        "data": "traffic",
        "channels": 862,
        "target": None,
        "freq": "h",
        "seq_len": 96, "pred_len": 96, "label_len": 48,
        "description": "Traffic dataset: 862 sensors on San Francisco freeways.",
    },
    "exchange_rate": {
        "data": "exchange_rate",
        "channels": 8,
        "target": None,
        "freq": "d",
        "seq_len": 96, "pred_len": 96, "label_len": 48,
        "description": "Exchange Rate dataset: daily rates of 8 countries (1990-2016).",
    },
    "illness": {
        "data": "illness",
        "channels": 7,
        "target": None,
        "freq": "w",
        "seq_len": 36, "pred_len": 24, "label_len": 18,
        "description": "ILI (national illness) dataset. Weekly influenza-like illness data.",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# TSL defaults (used when script doesn't override)
# ═══════════════════════════════════════════════════════════════════════

TSL_DEFAULTS = {
    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "dropout": 0.1,
    "factor": 3,
    "train_epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.0001,
}

# ═══════════════════════════════════════════════════════════════════════
# Model-specific overrides (from TSL benchmark scripts, pred_len=96)
# Key: (model, dataset) → dict of overrides on top of TSL_DEFAULTS
# ═══════════════════════════════════════════════════════════════════════

# -- Informer: only ETTh1 and ECL have scripts; rest use defaults -------
INFORMER_OVERRIDES: dict[str, dict] = {
    "etth1":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "etth2":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "ettm1":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "ettm2":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "electricity": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "weather":     {"e_layers": 2, "d_layers": 1, "factor": 3},
    "traffic":     {"e_layers": 2, "d_layers": 1, "factor": 3},
    "exchange_rate": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "illness":     {"e_layers": 2, "d_layers": 1, "factor": 3},
}

# -- FEDformer: same pattern as Informer ---------------------------------
FEDFORMER_OVERRIDES: dict[str, dict] = {
    "etth1":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "etth2":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "ettm1":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "ettm2":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "electricity": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "weather":     {"e_layers": 2, "d_layers": 1, "factor": 3},
    "traffic":     {"e_layers": 2, "d_layers": 1, "factor": 3},
    "exchange_rate": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "illness":     {"e_layers": 2, "d_layers": 1, "factor": 3},
}

# -- Autoformer: full coverage -------------------------------------------
AUTOFORMER_OVERRIDES: dict[str, dict] = {
    "etth1":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "etth2":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "ettm1":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "ettm2":       {"e_layers": 2, "d_layers": 1, "factor": 1},
    "electricity": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "weather":     {"e_layers": 2, "d_layers": 1, "factor": 3, "train_epochs": 2},
    "traffic":     {"e_layers": 2, "d_layers": 1, "factor": 3, "train_epochs": 3},
    "exchange_rate": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "illness":     {"e_layers": 2, "d_layers": 1, "factor": 3},
}

# -- TimesNet: full coverage, varies d_model/d_ff, uses top_k=5 ----------
TIMESNET_OVERRIDES: dict[str, dict] = {
    "etth1":       {"d_model": 16, "d_ff": 32, "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5},
    "etth2":       {"d_model": 32, "d_ff": 32, "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5},
    "ettm1":       {"d_model": 64, "d_ff": 64, "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5},
    "ettm2":       {"d_model": 32, "d_ff": 32, "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5},
    "electricity": {"d_model": 256, "d_ff": 512, "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5},
    "weather":     {"d_model": 32, "d_ff": 32, "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5},
    "traffic":     {"d_model": 512, "d_ff": 512, "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5},
    "exchange_rate": {"d_model": 64, "d_ff": 64, "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5},
    "illness":     {"d_model": 768, "d_ff": 768, "e_layers": 2, "d_layers": 1, "factor": 3, "top_k": 5},
}

# -- Transformer: full coverage, ECL uses features=S (we keep M for consistency)
TRANSFORMER_OVERRIDES: dict[str, dict] = {
    "etth1":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "etth2":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "ettm1":       {"e_layers": 2, "d_layers": 1, "factor": 3},
    "ettm2":       {"e_layers": 2, "d_layers": 1, "factor": 1},
    "electricity": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "weather":     {"e_layers": 2, "d_layers": 1, "factor": 3, "train_epochs": 3},
    "traffic":     {"e_layers": 2, "d_layers": 1, "factor": 3, "train_epochs": 3},
    "exchange_rate": {"e_layers": 2, "d_layers": 1, "factor": 3},
    "illness":     {"e_layers": 2, "d_layers": 1, "factor": 3},
}

# -- iTransformer: 4 datasets with scripts, rest use sensible defaults ---
ITRANSFORMER_OVERRIDES: dict[str, dict] = {
    "etth1":       {"d_model": 128, "d_ff": 128, "e_layers": 2, "d_layers": 1, "factor": 3},
    "etth2":       {"d_model": 128, "d_ff": 128, "e_layers": 2, "d_layers": 1, "factor": 3},
    "ettm1":       {"d_model": 128, "d_ff": 128, "e_layers": 2, "d_layers": 1, "factor": 3},
    "ettm2":       {"d_model": 128, "d_ff": 128, "e_layers": 2, "d_layers": 1, "factor": 3},
    "electricity": {"d_model": 512, "d_ff": 512, "e_layers": 3, "d_layers": 1, "factor": 3, "batch_size": 16, "learning_rate": 0.0005},
    "weather":     {"d_model": 512, "d_ff": 512, "e_layers": 3, "d_layers": 1, "factor": 3},
    "traffic":     {"d_model": 512, "d_ff": 512, "e_layers": 4, "d_layers": 1, "factor": 3, "batch_size": 16, "learning_rate": 0.001},
    "exchange_rate": {"d_model": 128, "d_ff": 128, "e_layers": 2, "d_layers": 1, "factor": 3},
    "illness":     {"d_model": 128, "d_ff": 128, "e_layers": 2, "d_layers": 1, "factor": 3},
}

# -- TimeMixer: label_len=0, unique down_sampling args -------------------
TIMEMIXER_OVERRIDES: dict[str, dict] = {
    "etth1":       {"d_model": 16, "d_ff": 32, "e_layers": 2, "batch_size": 128, "learning_rate": 0.01, "train_epochs": 10, "down_sampling_layers": 3, "down_sampling_method": "avg", "down_sampling_window": 2},
    "etth2":       {"d_model": 16, "d_ff": 32, "e_layers": 2, "learning_rate": 0.01, "down_sampling_layers": 3, "down_sampling_method": "avg", "down_sampling_window": 2},
    "ettm1":       {"d_model": 16, "d_ff": 32, "e_layers": 2, "batch_size": 16, "learning_rate": 0.01, "down_sampling_layers": 3, "down_sampling_method": "avg", "down_sampling_window": 2},
    "ettm2":       {"d_model": 32, "d_ff": 32, "e_layers": 2, "batch_size": 128, "learning_rate": 0.01, "down_sampling_layers": 3, "down_sampling_method": "avg", "down_sampling_window": 2},
    "electricity": {"d_model": 16, "d_ff": 32, "e_layers": 3, "d_layers": 1, "factor": 3, "batch_size": 32, "learning_rate": 0.01, "train_epochs": 20, "down_sampling_layers": 3, "down_sampling_method": "avg", "down_sampling_window": 2},
    "weather":     {"d_model": 16, "d_ff": 32, "e_layers": 3, "d_layers": 1, "factor": 3, "batch_size": 128, "learning_rate": 0.01, "train_epochs": 20, "down_sampling_layers": 3, "down_sampling_method": "avg", "down_sampling_window": 2},
    "traffic":     {"d_model": 32, "d_ff": 64, "e_layers": 3, "d_layers": 1, "factor": 3, "batch_size": 8, "learning_rate": 0.01, "down_sampling_layers": 3, "down_sampling_method": "avg", "down_sampling_window": 2},
    "exchange_rate": {"d_model": 16, "d_ff": 32, "e_layers": 2, "learning_rate": 0.01, "down_sampling_layers": 3, "down_sampling_method": "avg", "down_sampling_window": 2},
    "illness":     {"d_model": 16, "d_ff": 32, "e_layers": 2, "learning_rate": 0.01, "down_sampling_layers": 3, "down_sampling_method": "avg", "down_sampling_window": 2},
}

# -- TimeXer: varies e_layers/d_model/d_ff per dataset -------------------
TIMEXER_OVERRIDES: dict[str, dict] = {
    "etth1":       {"d_model": 256, "e_layers": 1, "factor": 3, "batch_size": 4},
    "etth2":       {"d_model": 256, "d_ff": 1024, "e_layers": 1, "factor": 3, "batch_size": 16},
    "ettm1":       {"d_model": 256, "e_layers": 1, "factor": 3, "batch_size": 4},
    "ettm2":       {"d_model": 256, "e_layers": 1, "d_layers": 1, "factor": 3},
    "electricity": {"d_ff": 512, "e_layers": 4, "factor": 3, "batch_size": 4},
    "weather":     {"d_model": 256, "d_ff": 512, "e_layers": 1, "factor": 3, "batch_size": 4},
    "traffic":     {"d_model": 512, "d_ff": 512, "e_layers": 3, "factor": 3, "batch_size": 16, "learning_rate": 0.001},
    "exchange_rate": {"d_model": 256, "e_layers": 1, "factor": 3},
    "illness":     {"d_model": 256, "e_layers": 1, "factor": 3},
}

# -- Mamba: expand=2, d_conv=4, d_ff=16 (Mamba-specific), no dec_in ------
MAMBA_OVERRIDES: dict[str, dict] = {
    "etth1":       {"d_model": 128, "d_ff": 16, "e_layers": 2, "d_layers": 1, "expand": 2, "d_conv": 4},
    "etth2":       {"d_model": 128, "d_ff": 16, "e_layers": 2, "expand": 2, "d_conv": 4},
    "ettm1":       {"d_model": 128, "d_ff": 16, "e_layers": 2, "expand": 2, "d_conv": 4},
    "ettm2":       {"d_model": 128, "d_ff": 16, "e_layers": 2, "expand": 2, "d_conv": 4},
    "electricity": {"d_model": 128, "d_ff": 16, "e_layers": 2, "d_layers": 1, "expand": 2, "d_conv": 4},
    "weather":     {"d_model": 128, "d_ff": 16, "e_layers": 2, "d_layers": 1, "expand": 2, "d_conv": 4},
    "traffic":     {"d_model": 128, "d_ff": 16, "e_layers": 2, "d_layers": 1, "expand": 2, "d_conv": 4},
    "exchange_rate": {"d_model": 128, "d_ff": 16, "e_layers": 2, "d_layers": 1, "expand": 2, "d_conv": 4},
    "illness":     {"d_model": 128, "d_ff": 16, "e_layers": 2, "expand": 2, "d_conv": 4},
}

# -- LSTM: liulian-native, not in TSL. Uses own defaults ------------------
LSTM_OVERRIDES: dict[str, dict] = {
    "etth1":       {"d_model": 64, "d_ff": 32, "e_layers": 2, "d_layers": 1, "dropout": 0.0},
    "etth2":       {"d_model": 64, "d_ff": 32, "e_layers": 2, "d_layers": 1, "dropout": 0.0},
    "ettm1":       {"d_model": 64, "d_ff": 32, "e_layers": 2, "d_layers": 1, "dropout": 0.0},
    "ettm2":       {"d_model": 64, "d_ff": 32, "e_layers": 2, "d_layers": 1, "dropout": 0.0},
    "electricity": {"d_model": 64, "d_ff": 32, "e_layers": 2, "d_layers": 1, "dropout": 0.0},
    "weather":     {"d_model": 64, "d_ff": 32, "e_layers": 2, "d_layers": 1, "dropout": 0.0},
    "traffic":     {"d_model": 64, "d_ff": 32, "e_layers": 2, "d_layers": 1, "dropout": 0.0},
    "exchange_rate": {"d_model": 64, "d_ff": 32, "e_layers": 2, "d_layers": 1, "dropout": 0.0},
    "illness":     {"d_model": 64, "d_ff": 32, "e_layers": 2, "d_layers": 1, "dropout": 0.0},
}

# ═════════════════════════════════════════════════════════════════════
# Consolidated model registry
# ═════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, dict] = {
    # model_name → {
    #   "overrides": per-dataset overrides,
    #   "label_len_override": optional (e.g. TimeMixer uses 0),
    #   "extra_arch_keys": list of extra model-specific keys to emit,
    #   "is_tsl": whether it has a TSL counterpart (for comparison script),
    #   "tsl_model_name": TSL model class name (e.g. "Informer"),
    #   "seed": default seed (2021 for TSL-aligned, 2026 for liulian-native),
    #   "train_style": "tsl" or "lstm" (controls training defaults),
    # }
    "informer": {
        "overrides": INFORMER_OVERRIDES,
        "is_tsl": True, "tsl_model_name": "Informer",
        "seed": 2021, "train_style": "tsl",
    },
    "autoformer": {
        "overrides": AUTOFORMER_OVERRIDES,
        "is_tsl": True, "tsl_model_name": "Autoformer",
        "seed": 2021, "train_style": "tsl",
    },
    "fedformer": {
        "overrides": FEDFORMER_OVERRIDES,
        "is_tsl": True, "tsl_model_name": "FEDformer",
        "seed": 2021, "train_style": "tsl",
    },
    "timesnet": {
        "overrides": TIMESNET_OVERRIDES,
        "is_tsl": True, "tsl_model_name": "TimesNet",
        "seed": 2021, "train_style": "tsl",
    },
    "transformer": {
        "overrides": TRANSFORMER_OVERRIDES,
        "is_tsl": True, "tsl_model_name": "Transformer",
        "seed": 2021, "train_style": "tsl",
    },
    "itransformer": {
        "overrides": ITRANSFORMER_OVERRIDES,
        "is_tsl": True, "tsl_model_name": "iTransformer",
        "seed": 2021, "train_style": "tsl",
    },
    "timemixer": {
        "overrides": TIMEMIXER_OVERRIDES,
        "label_len_override": 0,
        "is_tsl": True, "tsl_model_name": "TimeMixer",
        "seed": 2021, "train_style": "tsl",
    },
    "timexer": {
        "overrides": TIMEXER_OVERRIDES,
        "is_tsl": True, "tsl_model_name": "TimeXer",
        "seed": 2021, "train_style": "tsl",
    },
    "mamba": {
        "overrides": MAMBA_OVERRIDES,
        "extra_arch_keys": ["expand", "d_conv"],
        "is_tsl": True, "tsl_model_name": "Mamba",
        "seed": 2021, "train_style": "tsl",
    },
    "lstm": {
        "overrides": LSTM_OVERRIDES,
        "is_tsl": False, "tsl_model_name": None,
        "seed": 2026, "train_style": "lstm",
    },
}

# TSL script availability matrix (True = .sh exists in TSL benchmark scripts)
TSL_SCRIPT_AVAILABLE: dict[str, dict[str, bool]] = {
    "informer":     {"etth1": True, "etth2": False, "ettm1": False, "ettm2": False, "electricity": True, "weather": False, "traffic": False, "exchange_rate": False, "illness": False},
    "fedformer":    {"etth1": True, "etth2": False, "ettm1": False, "ettm2": False, "electricity": True, "weather": False, "traffic": False, "exchange_rate": False, "illness": False},
    "autoformer":   {ds: True for ds in DATASETS},
    "timesnet":     {ds: True for ds in DATASETS},
    "transformer":  {ds: True for ds in DATASETS},
    "itransformer": {"etth1": False, "etth2": True, "ettm1": False, "ettm2": False, "electricity": True, "weather": True, "traffic": True, "exchange_rate": False, "illness": False},
    "timemixer":    {"etth1": True, "etth2": True, "ettm1": True, "ettm2": True, "electricity": True, "weather": True, "traffic": True, "exchange_rate": False, "illness": False},
    "timexer":      {"etth1": True, "etth2": True, "ettm1": True, "ettm2": True, "electricity": True, "weather": True, "traffic": True, "exchange_rate": False, "illness": False},
    "mamba":        {"etth1": True, "etth2": True, "ettm1": True, "ettm2": True, "electricity": True, "weather": True, "traffic": True, "exchange_rate": True, "illness": False},
    "lstm":         {ds: False for ds in DATASETS},
}

# ═══════════════════════════════════════════════════════════════════════
# Config template
# ═══════════════════════════════════════════════════════════════════════

def _render_config(
    model_name: str,
    dataset_key: str,
) -> str:
    """Render a complete YAML config string for model × dataset."""
    ds = DATASETS[dataset_key]
    reg = MODEL_REGISTRY[model_name]
    overrides = reg["overrides"].get(dataset_key, {})

    # Merge TSL defaults with per-dataset overrides
    params = dict(TSL_DEFAULTS)
    params.update(overrides)

    # label_len: dataset default unless model overrides
    label_len = reg.get("label_len_override", ds["label_len"])

    seed = reg["seed"]
    is_lstm = reg["train_style"] == "lstm"

    # Training style defaults
    train_epochs = params.get("train_epochs", 10)
    batch_size = params.get("batch_size", 32)
    learning_rate = params.get("learning_rate", 0.0001)
    dropout = params.get("dropout", 0.1)
    patience = 10 if is_lstm else 3
    lradj = "none" if is_lstm else "type1"
    identifier_mode = "embedding" if is_lstm else "none"
    id_integration = "concat_to_x"

    # Extra model-specific architecture keys
    extra_arch_lines = ""
    for key in reg.get("extra_arch_keys", []):
        val = params.get(key)
        if val is not None:
            extra_arch_lines += f"{key}: {val}\n"

    # TimeMixer down_sampling keys
    down_sampling_lines = ""
    if "down_sampling_layers" in params:
        down_sampling_lines = (
            f"down_sampling_layers: {params['down_sampling_layers']}\n"
            f"down_sampling_method: {params['down_sampling_method']}\n"
            f"down_sampling_window: {params['down_sampling_window']}\n"
        )

    # TimesNet top_k
    top_k_line = ""
    if "top_k" in params:
        top_k_line = f"top_k: {params['top_k']}\n"

    # Target/freq lines (only for ETT datasets which need them)
    target_line = f"target: {ds['target']}" if ds["target"] else ""
    freq_line = f"freq: {ds['freq']}" if ds["freq"] else ""
    target_freq_block = ""
    if target_line:
        target_freq_block += f"\n{target_line}"
    if freq_line:
        target_freq_block += f"\n{freq_line}"

    tsl_name = reg.get("tsl_model_name", model_name)
    header_comment = f"# {ds['description']}\n"
    if reg["is_tsl"]:
        has_script = TSL_SCRIPT_AVAILABLE.get(model_name, {}).get(dataset_key, False)
        script_note = "has TSL benchmark script" if has_script else "no TSL script — using TSL defaults"
        header_comment += f"# {tsl_name} ({script_note})\n"
    else:
        header_comment += f"# {model_name} (liulian-native model, no TSL counterpart)\n"

    config = textwrap.dedent(f"""\
{header_comment}#
# Usage:
#   liulian run experiments/{dataset_key}/{model_name}_config.yaml
#   liulian run experiments/{dataset_key}/{model_name}_config.yaml --quick_test

# ── General ─────────────────────────────────────────────────────────────
model: {model_name}
seed: {seed}
quick_test: false
eval_only: false

# ── Data ────────────────────────────────────────────────────────────────
data: {ds['data']}
seq_len: {ds['seq_len']}
pred_len: {ds['pred_len']}
features: M
label_len: {label_len}
train_split: 0.7
max_samples: null{target_freq_block}

# ── Task & mode ─────────────────────────────────────────────────────────
task: forecast
split_mode: multi_channel
scaler: standard
use_current_x: true
use_full_history: false

# ── Gap handling ────────────────────────────────────────────────────────
short_subsequence_method: drop
gap_mode: split
max_mask_consecutive: 10

# ── Noise injection ─────────────────────────────────────────────────────
noise_type: null
noise_level: 0.01
noise_probability: 0.01
noise_scale_factor: 5.0

# ── Historical target ──────────────────────────────────────────────────
include_historical_y: none
include_historical_predicted_y: false

# ── Entity identifiers ─────────────────────────────────────────────────
identifier_mode: {identifier_mode}
id_integration: {id_integration}
embedding_size: 10
num_embeddings: null

# ── Graph / spatial ─────────────────────────────────────────────────────
graph_mode: none
graphlet_num_hops: 1

# ── Model architecture ─────────────────────────────────────────────────
enc_in: null
dec_in: 1
c_out: 1
d_model: {params['d_model']}
d_ff: {params['d_ff']}
n_heads: {params['n_heads']}
e_layers: {params['e_layers']}
d_layers: {params['d_layers']}
dropout: {dropout}
{extra_arch_lines}{top_k_line}{down_sampling_lines}individual: false
moving_avg: 25
patch_len: 16
stride: 8

# ── Training ────────────────────────────────────────────────────────────
train_epochs: {train_epochs}
batch_size: {batch_size}
learning_rate: {learning_rate}
loss: mse
metrics: rmse,mae,mse
eval_denorm: true
show_progress: true
patience: {patience}
disable_early_stopping: false
lradj: {lradj}
pct_start: 0.2
cos_T_max: 20
cos_eta_min: 1.0e-8
cos_T_0: 10
cos_T_mult: 2
step_size: 10
gamma: 0.5
milestones: "30,60,90"
sched_patience: 5
sched_factor: 0.5
num_workers: 0

# ── HPO (Ray Tune) ─────────────────────────────────────────────────────
hpo: false
hpo_num_samples: 200
hpo_scheduler: asha
hpo_grace_period: 5
hpo_reduction_factor: 1.5
hpo_storage_path: null
hpo_resources_cpu: 1
hpo_resources_gpu: 0.25
hpo_num_cpus: null
hpo_max_concurrent: null
hpo_resume: false
hpo_save_checkpoints: true
hpo_trim_checkpoints: true
hpo_keep_best_n: 10
hpo_trim_best_n: true
hpo_trim_keep_best: true
hpo_trim_keep_last: false
hpo_experiment_name: null
hpo_local_mode: false

# ── Logging ─────────────────────────────────────────────────────────────
wandb_project: null
wandb_entity: null
dev_run: false

# ── Visualisation ───────────────────────────────────────────────────────
auto_viz: true
viz_method: mean

# ── Misc model keys (compat) ───────────────────────────────────────────
embed: timeF
activation: gelu
output_attention: false
factor: {params.get('factor', 1)}
task_name: long_term_forecast
""")
    return config


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-dataset configs")
    parser.add_argument("--dry-run", action="store_true", help="Print paths without writing")
    parser.add_argument("--models", nargs="+", help="Only generate for these models")
    parser.add_argument("--datasets", nargs="+", help="Only generate for these datasets")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing configs")
    args = parser.parse_args()

    models = args.models or list(MODEL_REGISTRY.keys())
    datasets = args.datasets or list(DATASETS.keys())

    created, skipped = 0, 0
    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            print(f"  ⚠ Unknown model '{model_name}', skipping")
            continue
        for ds_key in datasets:
            if ds_key not in DATASETS:
                print(f"  ⚠ Unknown dataset '{ds_key}', skipping")
                continue

            out_dir = PROJECT_ROOT / "experiments" / ds_key
            out_file = out_dir / f"{model_name}_config.yaml"

            if out_file.exists() and not args.overwrite:
                skipped += 1
                continue

            content = _render_config(model_name, ds_key)

            if args.dry_run:
                rel = out_file.relative_to(PROJECT_ROOT)
                print(f"  [dry-run] would create {rel}")
                created += 1
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            out_file.write_text(content, encoding="utf-8")
            created += 1

    verb = "would create" if args.dry_run else "created"
    print(f"\nDone: {verb} {created} configs, skipped {skipped} (already exist).")
    if skipped and not args.overwrite:
        print("  Use --overwrite to regenerate existing configs.")


if __name__ == "__main__":
    main()
