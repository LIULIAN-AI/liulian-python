#!/usr/bin/env python3
"""Generate all experiment YAML configs for the benchmark suite.

Usage::

    python tools/generate_configs.py [--output-dir experiments/configs]

Generates YAML config files for every experiment group defined in the
integration plan (Part E):

* ``long_term/``    — Core Forecasting Benchmark (E.2.1)
* ``entity/``       — Entity Identifier Ablation (E.2.2)
* ``nowcasting/``   — Nowcasting on Swiss-river (E.2.3)
* ``m4/``           — Short-term Forecasting (E.2.4)
* ``spatial/``      — Spatial-Temporal Forecasting (E.2.5)
* ``ablation_norm/`` — Normalization Effect (E.2.6A)
* ``ablation_aug/``  — Augmentation Effect (E.2.6B)
* ``ablation_seqlen/`` — Input Length Sensitivity (E.2.6C)
* ``ablation_tf/``   — Teacher Forcing Effect (E.2.6D)
"""

from __future__ import annotations

import argparse
import os
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import yaml


# ======================================================================
# Constants
# ======================================================================

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / 'experiments' / 'configs'

# --- Models ---

TSL_MODELS = [
    'DLinear',
    'Transformer',
    'Informer',
    'Autoformer',
    'FEDformer',
    'iTransformer',
    'PatchTST',
    'TimesNet',
    'TimeMixer',
    'TimeXer',
    'Mamba',
    'TimeLLM',
]

CUSTOM_MODELS = [
    'LSTMAdapter',
    'ExtrapoLSTMAdapter',
    'TransformerEncoderAdapter',
]

ALL_MODELS = TSL_MODELS + CUSTOM_MODELS

ENCODER_DECODER_MODELS = [
    'Transformer',
    'Informer',
    'Autoformer',
    'FEDformer',
]

# --- Datasets ---

LONG_TERM_DATASETS = {
    'ETTh1': {
        'root_path': 'dataset/ETT-small',
        'data_path': 'ETTh1.csv',
        'data_name': 'ETTh1',
        'freq': 'h',
        'enc_in': 7,
        'target': 'OT',
    },
    'ETTh2': {
        'root_path': 'dataset/ETT-small',
        'data_path': 'ETTh2.csv',
        'data_name': 'ETTh2',
        'freq': 'h',
        'enc_in': 7,
        'target': 'OT',
    },
    'ETTm1': {
        'root_path': 'dataset/ETT-small',
        'data_path': 'ETTm1.csv',
        'data_name': 'ETTm1',
        'freq': 'h',
        'enc_in': 7,
        'target': 'OT',
    },
    'ETTm2': {
        'root_path': 'dataset/ETT-small',
        'data_path': 'ETTm2.csv',
        'data_name': 'ETTm2',
        'freq': 'h',
        'enc_in': 7,
        'target': 'OT',
    },
    'Weather': {
        'root_path': 'dataset/weather',
        'data_path': 'weather.csv',
        'data_name': 'weather',
        'freq': 'h',
        'enc_in': 21,
        'target': 'OT',
    },
    'Electricity': {
        'root_path': 'dataset/electricity',
        'data_path': 'electricity.csv',
        'data_name': 'electricity',
        'freq': 'h',
        'enc_in': 321,
        'target': 'OT',
    },
    'Traffic': {
        'root_path': 'dataset/traffic',
        'data_path': 'traffic.csv',
        'data_name': 'traffic',
        'freq': 'h',
        'enc_in': 862,
        'target': 'OT',
    },
    'Exchange': {
        'root_path': 'dataset/exchange_rate',
        'data_path': 'exchange_rate.csv',
        'data_name': 'exchange_rate',
        'freq': 'd',
        'enc_in': 8,
        'target': 'OT',
    },
    'ILI': {
        'root_path': 'dataset/illness',
        'data_path': 'national_illness.csv',
        'data_name': 'illness',
        'freq': 'w',
        'enc_in': 7,
        'target': 'OT',
    },
    'Swiss1990': {
        'root_path': 'dataset/swiss_river',
        'data_path': '1990.csv',
        'data_name': 'custom',
        'freq': 'd',
        'enc_in': 64,
        'target': 'OT',
    },
    'Swiss2010': {
        'root_path': 'dataset/swiss_river',
        'data_path': '2010.csv',
        'data_name': 'custom',
        'freq': 'd',
        'enc_in': 64,
        'target': 'OT',
    },
    'SwissZurich': {
        'root_path': 'dataset/swiss_river',
        'data_path': 'zurich.csv',
        'data_name': 'custom',
        'freq': 'd',
        'enc_in': 15,
        'target': 'OT',
    },
}

ENTITY_DATASETS = {
    'Traffic': LONG_TERM_DATASETS['Traffic'],
    'Electricity': LONG_TERM_DATASETS['Electricity'],
    'Exchange': LONG_TERM_DATASETS['Exchange'],
    'Swiss1990': LONG_TERM_DATASETS['Swiss1990'],
    'Swiss2010': LONG_TERM_DATASETS['Swiss2010'],
    'SwissZurich': LONG_TERM_DATASETS['SwissZurich'],
    'PEMS03': {
        'root_path': 'dataset/PEMS',
        'data_path': 'PEMS03.npz',
        'data_name': 'PEMS03',
        'freq': 'h',
        'enc_in': 358,
    },
    'PEMS04': {
        'root_path': 'dataset/PEMS',
        'data_path': 'PEMS04.npz',
        'data_name': 'PEMS04',
        'freq': 'h',
        'enc_in': 307,
    },
    'PEMS07': {
        'root_path': 'dataset/PEMS',
        'data_path': 'PEMS07.npz',
        'data_name': 'PEMS07',
        'freq': 'h',
        'enc_in': 883,
    },
    'PEMS08': {
        'root_path': 'dataset/PEMS',
        'data_path': 'PEMS08.npz',
        'data_name': 'PEMS08',
        'freq': 'h',
        'enc_in': 170,
    },
}

SWISS_DATASETS = {
    'Swiss1990': LONG_TERM_DATASETS['Swiss1990'],
    'Swiss2010': LONG_TERM_DATASETS['Swiss2010'],
    'SwissZurich': LONG_TERM_DATASETS['SwissZurich'],
}

SPATIAL_DATASETS = {
    'PEMS03': ENTITY_DATASETS['PEMS03'],
    'PEMS04': ENTITY_DATASETS['PEMS04'],
    'PEMS07': ENTITY_DATASETS['PEMS07'],
    'PEMS08': ENTITY_DATASETS['PEMS08'],
    'Swiss1990': LONG_TERM_DATASETS['Swiss1990'],
    'Swiss2010': LONG_TERM_DATASETS['Swiss2010'],
    'SwissZurich': LONG_TERM_DATASETS['SwissZurich'],
}

M4_FREQS = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']

# --- Default hyperparameters per model ---

_DEFAULT_HPARAMS: Dict[str, Dict[str, Any]] = {
    'DLinear': {
        'd_model': 128,
        'd_ff': 256,
        'n_heads': 1,
        'e_layers': 1,
        'd_layers': 0,
        'dropout': 0.0,
        'learning_rate': 0.001,
        'batch_size': 32,
        'train_epochs': 50,
    },
    'Transformer': {
        'd_model': 512,
        'd_ff': 2048,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'train_epochs': 20,
    },
    'Informer': {
        'd_model': 512,
        'd_ff': 2048,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        'factor': 5,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'train_epochs': 20,
    },
    'Autoformer': {
        'd_model': 512,
        'd_ff': 2048,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        'moving_avg': 25,
        'factor': 1,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'train_epochs': 20,
    },
    'FEDformer': {
        'd_model': 512,
        'd_ff': 2048,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'dropout': 0.1,
        'moving_avg': 25,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'train_epochs': 20,
    },
    'iTransformer': {
        'd_model': 512,
        'd_ff': 2048,
        'n_heads': 8,
        'e_layers': 3,
        'd_layers': 0,
        'dropout': 0.1,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'train_epochs': 20,
    },
    'PatchTST': {
        'd_model': 128,
        'd_ff': 256,
        'n_heads': 16,
        'e_layers': 3,
        'd_layers': 0,
        'dropout': 0.2,
        'learning_rate': 0.0001,
        'batch_size': 128,
        'train_epochs': 100,
    },
    'TimesNet': {
        'd_model': 64,
        'd_ff': 64,
        'n_heads': 1,
        'e_layers': 2,
        'd_layers': 0,
        'dropout': 0.1,
        'top_k': 5,
        'num_kernels': 6,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'train_epochs': 20,
    },
    'TimeMixer': {
        'd_model': 64,
        'd_ff': 64,
        'n_heads': 1,
        'e_layers': 4,
        'd_layers': 0,
        'dropout': 0.1,
        'down_sampling_layers': 3,
        'down_sampling_window': 2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'train_epochs': 20,
    },
    'TimeXer': {
        'd_model': 256,
        'd_ff': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 0,
        'dropout': 0.1,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'train_epochs': 20,
    },
    'Mamba': {
        'd_model': 128,
        'd_ff': 256,
        'n_heads': 1,
        'e_layers': 2,
        'd_layers': 0,
        'dropout': 0.1,
        'd_state': 16,
        'expand': 2,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'train_epochs': 20,
    },
    'TimeLLM': {
        'd_model': 32,
        'd_ff': 128,
        'n_heads': 8,
        'e_layers': 1,
        'd_layers': 0,
        'dropout': 0.1,
        'learning_rate': 0.01,
        'batch_size': 16,
        'train_epochs': 30,
        'llm_model': 'GPT2',
        'llm_dim': 768,
    },
    'LSTMAdapter': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 64,
        'train_epochs': 50,
    },
    'ExtrapoLSTMAdapter': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 64,
        'train_epochs': 50,
    },
    'TransformerEncoderAdapter': {
        'dropout': 0.1,
        'num_t_heads': 4,
        'ratio_heads_to_d_model': 8,
        'dim_feedforward': 128,
        'num_layers': 2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'train_epochs': 50,
    },
}


# ======================================================================
# Config builder helpers
# ======================================================================


def _base_config(
    model: str,
    dataset_name: str,
    dataset_info: Dict[str, Any],
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 96,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a base experiment config dict."""
    hparams = dict(_DEFAULT_HPARAMS.get(model, {}))
    hparams.update(overrides)

    cfg: Dict[str, Any] = {
        'model': model,
        'dataset': dataset_name,
        'root_path': dataset_info.get('root_path', ''),
        'data_path': dataset_info.get('data_path', ''),
        'data_name': dataset_info.get('data_name', 'custom'),
        'features': 'M',
        'target': dataset_info.get('target', 'OT'),
        'freq': dataset_info.get('freq', 'h'),
        'seq_len': seq_len,
        'label_len': label_len,
        'pred_len': pred_len,
        'enc_in': dataset_info.get('enc_in', 7),
        'dec_in': dataset_info.get('enc_in', 7),
        'c_out': dataset_info.get('enc_in', 7),
        'patience': 5,
        'loss': 'mse',
        'metrics': ['mse', 'mae', 'rmse'],
    }
    cfg.update(hparams)
    return cfg


def _write_yaml(config: Dict[str, Any], filepath: Path) -> None:
    """Write config dict to a YAML file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ======================================================================
# Experiment generators
# ======================================================================


def generate_long_term(output_dir: Path) -> int:
    """E.2.1 — Core Benchmark: Long-term Forecasting."""
    count = 0
    for model, (ds_name, ds_info) in product(ALL_MODELS, LONG_TERM_DATASETS.items()):
        # ILI uses different horizons and input length
        if ds_name == 'ILI':
            horizons = [24, 36, 48, 60]
            seq_len, label_len = 36, 18
        elif ds_name.startswith('Swiss'):
            horizons = [7, 14, 30, 60]
            seq_len, label_len = 30, 0
        else:
            horizons = [96, 192, 336, 720]
            seq_len, label_len = 96, 48

        for h in horizons:
            cfg = _base_config(
                model,
                ds_name,
                ds_info,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=h,
            )
            stem = f'{model.lower()}_{ds_name.lower()}_H{h}'
            _write_yaml(cfg, output_dir / 'long_term' / f'{stem}.yaml')
            count += 1
    return count


def generate_entity(output_dir: Path) -> int:
    """E.2.2 — Entity Identifier Ablation (FULL)."""
    entity_modes = ['none', 'embedding', 'onehot', 'numeric_id', 'sinusoidal']
    count = 0
    for model, (ds_name, ds_info) in product(ALL_MODELS, ENTITY_DATASETS.items()):
        is_swiss = ds_name.startswith('Swiss')
        pred_len = 20 if is_swiss else (12 if ds_name.startswith('PEMS') else 96)
        seq_len = 30 if is_swiss else 96
        label_len = 0 if is_swiss else 48

        modes = list(entity_modes)
        if is_swiss:
            modes.append('coordinates')

        for mode in modes:
            cfg = _base_config(
                model,
                ds_name,
                ds_info,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                identifier_mode=mode,
            )
            stem = f'{model.lower()}_{ds_name.lower()}_{mode}'
            _write_yaml(cfg, output_dir / 'entity' / f'{stem}.yaml')
            count += 1
    return count


def generate_nowcasting(output_dir: Path) -> int:
    """E.2.3 — Nowcasting on Swiss-river."""
    windows = [1, 7, 14, 30]
    entity_modes = ['none', 'embedding']
    count = 0
    for model, (ds_name, ds_info) in product(ALL_MODELS, SWISS_DATASETS.items()):
        for window, mode in product(windows, entity_modes):
            cfg = _base_config(
                model,
                ds_name,
                ds_info,
                seq_len=30,
                label_len=0,
                pred_len=window,
                identifier_mode=mode,
                nan_mask_loss=True,
                teacher_forcing='zeros',
            )
            cfg['metrics'] = ['mse', 'mae', 'rmse', 'nse']
            stem = f'{model.lower()}_{ds_name.lower()}_W{window}_{mode}'
            _write_yaml(cfg, output_dir / 'nowcasting' / f'{stem}.yaml')
            count += 1
    return count


def generate_m4(output_dir: Path) -> int:
    """E.2.4 — Short-term Forecasting (M4)."""
    count = 0
    for model, freq in product(ALL_MODELS, M4_FREQS):
        cfg: Dict[str, Any] = {
            'model': model,
            'dataset': f'M4_{freq}',
            'data_name': 'm4',
            'seasonal_patterns': freq,
            'features': 'M',
            'loss': 'smape',
            'metrics': ['smape', 'mase'],
        }
        cfg.update(_DEFAULT_HPARAMS.get(model, {}))
        stem = f'{model.lower()}_m4_{freq.lower()}'
        _write_yaml(cfg, output_dir / 'm4' / f'{stem}.yaml')
        count += 1
    return count


def generate_spatial(output_dir: Path) -> int:
    """E.2.5 — Spatial-Temporal Forecasting."""
    count = 0
    for model, (ds_name, ds_info) in product(ALL_MODELS, SPATIAL_DATASETS.items()):
        is_swiss = ds_name.startswith('Swiss')
        pred_len = 20 if is_swiss else 12
        seq_len = 30 if is_swiss else 96
        label_len = 0 if is_swiss else 48

        for variant in ['baseline', 'entity']:
            mode = 'none' if variant == 'baseline' else 'embedding'
            cfg = _base_config(
                model,
                ds_name,
                ds_info,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                identifier_mode=mode,
            )
            stem = f'{model.lower()}_{ds_name.lower()}_{variant}'
            _write_yaml(cfg, output_dir / 'spatial' / f'{stem}.yaml')
            count += 1
    return count


def generate_ablation_norm(output_dir: Path) -> int:
    """E.2.6A — Normalization Effect."""
    datasets = {
        'ETTh1': LONG_TERM_DATASETS['ETTh1'],
        'Weather': LONG_TERM_DATASETS['Weather'],
        'Electricity': LONG_TERM_DATASETS['Electricity'],
    }
    scalers = ['none', 'standard', 'minmax']
    count = 0
    for model, (ds_name, ds_info), scaler in product(
        ALL_MODELS, datasets.items(), scalers
    ):
        scale = scaler != 'none'
        cfg = _base_config(
            model, ds_name, ds_info, pred_len=96, scale=scale, scaler_type=scaler
        )
        stem = f'{model.lower()}_{ds_name.lower()}_{scaler}'
        _write_yaml(cfg, output_dir / 'ablation_norm' / f'{stem}.yaml')
        count += 1
    return count


def generate_ablation_aug(output_dir: Path) -> int:
    """E.2.6B — Augmentation Effect."""
    datasets = {
        'ETTh1': LONG_TERM_DATASETS['ETTh1'],
        'Weather': LONG_TERM_DATASETS['Weather'],
        'Traffic': LONG_TERM_DATASETS['Traffic'],
    }
    aug_sets = {
        'none': [],
        'jitter': ['jitter'],
        'jitter_scaling': ['jitter', 'scaling'],
        'jitter_scaling_warp': ['jitter', 'scaling', 'window_warp'],
    }
    count = 0
    for model, (ds_name, ds_info), (aug_key, aug_list) in product(
        ALL_MODELS, datasets.items(), aug_sets.items()
    ):
        cfg = _base_config(model, ds_name, ds_info, pred_len=96)
        if aug_list:
            cfg['augmentation'] = aug_list
        stem = f'{model.lower()}_{ds_name.lower()}_{aug_key}'
        _write_yaml(cfg, output_dir / 'ablation_aug' / f'{stem}.yaml')
        count += 1
    return count


def generate_ablation_seqlen(output_dir: Path) -> int:
    """E.2.6C — Input Length Sensitivity."""
    datasets = {
        'ETTh1': LONG_TERM_DATASETS['ETTh1'],
        'Weather': LONG_TERM_DATASETS['Weather'],
        'Electricity': LONG_TERM_DATASETS['Electricity'],
    }
    input_lengths = [48, 96, 192, 336, 512]
    count = 0
    for model, (ds_name, ds_info), L in product(
        ALL_MODELS, datasets.items(), input_lengths
    ):
        cfg = _base_config(
            model, ds_name, ds_info, seq_len=L, label_len=L // 2, pred_len=96
        )
        stem = f'{model.lower()}_{ds_name.lower()}_L{L}'
        _write_yaml(cfg, output_dir / 'ablation_seqlen' / f'{stem}.yaml')
        count += 1
    return count


def generate_ablation_tf(output_dir: Path) -> int:
    """E.2.6D — Teacher Forcing Effect (encoder-decoder models only)."""
    datasets = {
        'ETTh1': LONG_TERM_DATASETS['ETTh1'],
        'Weather': LONG_TERM_DATASETS['Weather'],
        'Electricity': LONG_TERM_DATASETS['Electricity'],
    }
    tf_modes = ['label', 'zeros', 'none']
    count = 0
    for model, (ds_name, ds_info), tf in product(
        ENCODER_DECODER_MODELS, datasets.items(), tf_modes
    ):
        label_len = 48 if tf == 'label' else 0
        cfg = _base_config(
            model,
            ds_name,
            ds_info,
            pred_len=96,
            label_len=label_len,
            teacher_forcing=tf,
        )
        stem = f'{model.lower()}_{ds_name.lower()}_tf_{tf}'
        _write_yaml(cfg, output_dir / 'ablation_tf' / f'{stem}.yaml')
        count += 1
    return count


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate benchmark experiment configs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Output directory for YAML configs',
    )
    parser.add_argument(
        '--group',
        type=str,
        default='all',
        choices=[
            'all',
            'long_term',
            'entity',
            'nowcasting',
            'm4',
            'spatial',
            'ablation_norm',
            'ablation_aug',
            'ablation_seqlen',
            'ablation_tf',
        ],
        help='Generate only a specific experiment group',
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    generators = {
        'long_term': generate_long_term,
        'entity': generate_entity,
        'nowcasting': generate_nowcasting,
        'm4': generate_m4,
        'spatial': generate_spatial,
        'ablation_norm': generate_ablation_norm,
        'ablation_aug': generate_ablation_aug,
        'ablation_seqlen': generate_ablation_seqlen,
        'ablation_tf': generate_ablation_tf,
    }

    total = 0
    if args.group == 'all':
        for name, gen_fn in generators.items():
            count = gen_fn(output_dir)
            print(f'  {name}: {count} configs')
            total += count
    else:
        total = generators[args.group](output_dir)
        print(f'  {args.group}: {total} configs')

    print(f'\nTotal: {total} configs written to {output_dir}')


if __name__ == '__main__':
    main()
