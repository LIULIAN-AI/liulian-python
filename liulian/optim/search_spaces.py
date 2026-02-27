"""Pre-defined HPO search spaces matching reference projects.

Each function returns a ``dict`` whose values are **plain lists** (for the
fallback grid sweep) or can be wrapped with ``ray.tune.*`` helpers by the
caller.

Three families:

* **swiss-river** grids — from ``ray_tune.py`` in the swiss-river benchmark.
  Ref: https://github.com/RiverNetwork/swiss-river-network-benchmark
* **timellm** grids — derived from the shell-script hyper-parameter configs
  in the Time-LLM repository.
  Ref: https://github.com/KimMeen/Time-LLM
* **TSL model** grids — derived from the Time-Series-Library (TSLib) default
  configs and per-model training scripts.
  Ref: https://github.com/thuml/Time-Series-Library

Hyperparameter ranges are cross-referenced against each model's original
paper defaults and the TSLib canonical training scripts found under
``scripts/long_term_forecast/`` in the TSLib repository. TSLib argparse
defaults (``run.py``): d_model=512, d_ff=2048, n_heads=8, e_layers=2,
d_layers=1, dropout=0.1, factor=1, moving_avg=25, batch_size=32,
learning_rate=0.0001, train_epochs=10, patience=3, top_k=5, num_kernels=6,
expand=2, d_conv=4.

Usage::

    from liulian.optim.search_spaces import swiss_lstm_space

    space = swiss_lstm_space()

    # With Ray Tune:
    from ray import tune

    ray_space = {k: tune.choice(v) for k, v in space.items()}
"""

from __future__ import annotations

from typing import Any, Dict, List

# ======================================================================
# Swiss-river reference spaces
# Source: https://github.com/RiverNetwork/swiss-river-network-benchmark
# ======================================================================


def swiss_lstm_space() -> Dict[str, List[Any]]:
    """LSTM search space (swiss-river ``search_space_lstm``).

    Source: swiss-river-network-benchmark ``ray_tune.py``.
    """
    return {
        'batch_size': [32, 64, 128, 256],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'hidden_size': [16, 32, 64, 128],
        'num_layers': [1, 2, 3],
    }


def swiss_transformer_space() -> Dict[str, List[Any]]:
    """Transformer search space (swiss-river ``search_space_transformer``).

    Source: swiss-river-network-benchmark ``ray_tune.py``.
    """
    return {
        'batch_size': [32, 64, 128, 256],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
        'num_t_heads': [2, 4, 6, 8],
        'ratio_heads_to_d_model': [4, 8, 16],
        'dim_feedforward': [32, 64, 128, 256],
        'num_layers': [1, 2, 3],
    }


def swiss_lstm_embedding_space() -> Dict[str, List[Any]]:
    """LSTM + embedding search space (swiss-river ``search_space_lstm_embedding``).

    Source: swiss-river-network-benchmark ``ray_tune.py``.
    """
    return {
        'batch_size': [32, 64, 128, 256],
        'learning_rate': [0.00001, 0.0001, 0.0005, 0.001],
        'hidden_size': [16, 32, 64, 128],
        'num_layers': [1, 2, 3],
        'embedding_size': [1, 5, 10, 15, 30],
    }


def swiss_transformer_embedding_space() -> Dict[str, List[Any]]:
    """Transformer + embedding search space.

    Source: swiss-river-network-benchmark ``ray_tune.py``.
    """
    return {
        **swiss_transformer_space(),
        'embedding_size': [1, 5, 10, 15, 30],
        'learning_rate': [0.00001, 0.0001, 0.0005, 0.001],
    }


def swiss_stgnn_space() -> Dict[str, List[Any]]:
    """Spatio-temporal GNN search space (swiss-river ``search_space_stgnn``).

    Source: swiss-river-network-benchmark ``ray_tune.py``.
    """
    return {
        'batch_size': [1, 2, 3, 4],
        'learning_rate': [0.00001, 0.0001, 0.0005, 0.001],
        'hidden_size': [16, 32, 64, 128],
        'num_layers': [1, 2, 3],
        'gnn_conv': ['GCN', 'GIN', 'GraphSAGE', 'MPNN', 'GAT'],
        'num_convs': [1, 2, 3, 4, 5, 6],
    }


# ======================================================================
# Swiss-river ASHA scheduler presets
# Source: swiss-river-network-benchmark ``ray_tune.py``
# ======================================================================


def swiss_asha_default() -> Dict[str, Any]:
    """ASHA preset: ``max_t=200, grace_period=3, reduction_factor=2``."""
    return {
        'scheduler': 'asha',
        'max_epochs': 200,
        'grace_period': 3,
        'reduction_factor': 2,
    }


def swiss_asha_soft() -> Dict[str, Any]:
    """ASHA preset (soft): ``max_t=200, grace_period=5, reduction_factor=1.5``."""
    return {
        'scheduler': 'asha',
        'max_epochs': 200,
        'grace_period': 5,
        'reduction_factor': 1.5,
    }


def swiss_asha_single_soft() -> Dict[str, Any]:
    """ASHA preset (single model, soft): ``max_t=500, grace_period=5, rf=1.5``."""
    return {
        'scheduler': 'asha',
        'max_epochs': 500,
        'grace_period': 5,
        'reduction_factor': 1.5,
    }


def swiss_asha_single_hard() -> Dict[str, Any]:
    """ASHA preset (single model, hard): ``max_t=500, grace_period=3, rf=2``."""
    return {
        'scheduler': 'asha',
        'max_epochs': 500,
        'grace_period': 3,
        'reduction_factor': 2,
    }


# ======================================================================
# TimeLLM-derived spaces (from shell script configs)
# Paper: 'Time-LLM: Time Series Forecasting by Reprogramming Large
#         Language Models' (ICLR 2024)
# arXiv: https://arxiv.org/abs/2310.01728
# Code:  https://github.com/KimMeen/Time-LLM
# Configs: scripts/ directory in the Time-LLM repository
# ======================================================================


def timellm_etth1_space() -> Dict[str, List[Any]]:
    """TimeLLM hyper-parameter grid for ETTh1.

    Paper: https://arxiv.org/abs/2310.01728
    Config: https://github.com/KimMeen/Time-LLM/tree/main/scripts
    """
    return {
        'd_model': [16, 32],
        'd_ff': [32, 128],
        'learning_rate': [0.001, 0.01, 0.02],
        'batch_size': [16, 32],
        'lradj': ['type1', 'COS', 'TST'],
        'train_epochs': [30, 50],
    }


def timellm_etth2_space() -> Dict[str, List[Any]]:
    """TimeLLM hyper-parameter grid for ETTh2.

    Paper: https://arxiv.org/abs/2310.01728
    Config: https://github.com/KimMeen/Time-LLM/tree/main/scripts
    """
    return {
        'd_model': [16, 32],
        'd_ff': [32, 128],
        'learning_rate': [0.001, 0.01, 0.015],
        'batch_size': [16, 32],
        'lradj': ['type1', 'COS'],
        'train_epochs': [30, 50],
    }


def timellm_ettm_space() -> Dict[str, List[Any]]:
    """TimeLLM hyper-parameter grid for ETTm1 / ETTm2.

    Paper: https://arxiv.org/abs/2310.01728
    Config: https://github.com/KimMeen/Time-LLM/tree/main/scripts
    """
    return {
        'd_model': [16, 32],
        'd_ff': [32, 128],
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [8, 16, 32],
        'lradj': ['type1', 'COS'],
        'train_epochs': [10, 30],
    }


def timellm_weather_space() -> Dict[str, List[Any]]:
    """TimeLLM hyper-parameter grid for Weather.

    Paper: https://arxiv.org/abs/2310.01728
    Config: https://github.com/KimMeen/Time-LLM/tree/main/scripts
    """
    return {
        'd_model': [16, 32],
        'd_ff': [32, 128],
        'learning_rate': [0.005, 0.01],
        'batch_size': [16, 32],
        'lradj': ['type1'],
        'train_epochs': [10, 30],
    }


def timellm_electricity_space() -> Dict[str, List[Any]]:
    """TimeLLM hyper-parameter grid for Electricity.

    Paper: https://arxiv.org/abs/2310.01728
    Config: https://github.com/KimMeen/Time-LLM/tree/main/scripts
    """
    return {
        'd_model': [16, 32],
        'd_ff': [32, 128],
        'learning_rate': [0.001, 0.01],
        'batch_size': [16, 32],
        'lradj': ['type1'],
        'train_epochs': [10, 30],
    }


def timellm_swissriver_space() -> Dict[str, List[Any]]:
    """TimeLLM hyper-parameter grid for Swiss River benchmark.

    Paper: https://arxiv.org/abs/2310.01728
    Note: Adapted for Swiss River dataset; seq/pred/label lens match the
    benchmark defaults (seq=90, pred=7, label=0).
    """
    return {
        'd_model': [16, 32],
        'd_ff': [32, 128],
        'learning_rate': [0.001, 0.005],
        'batch_size': [8, 16],
        'lradj': ['type1'],
        'train_epochs': [30, 50],
        'seq_len': [90],
        'label_len': [0],
        'pred_len': [7],
    }


# ======================================================================
# TSL model spaces — general-purpose grids per model architecture
#
# Canonical source for all TSL models:
#   https://github.com/thuml/Time-Series-Library
#
# TSLib argparse defaults (run.py):
#   d_model=512, d_ff=2048, n_heads=8, e_layers=2, d_layers=1,
#   dropout=0.1, factor=1, moving_avg=25, batch_size=32, lr=0.0001,
#   train_epochs=10, patience=3, top_k=5, num_kernels=6, expand=2,
#   d_conv=4, down_sampling_layers=0, down_sampling_window=1,
#   patch_len=16.
#
# Per-model scripts override these defaults. Ranges below span the
# values seen across ETT, Weather, ECL, and Traffic scripts in TSLib.
# ======================================================================


def dlinear_space() -> Dict[str, List[Any]]:
    """DLinear search space (lightweight: only LR / batch).

    Paper: "Are Transformers Effective for Time Series Forecasting?"
           (AAAI 2023, Zeng et al.)
    arXiv: https://arxiv.org/abs/2205.13504
    Code:  https://github.com/cure-lab/LTSF-Linear
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh

    DLinear has no attention / d_model / d_ff — it uses simple
    decomposition + linear layers. The authors note "no model
    hyper-parameter tuning needed". Only batch_size and LR matter.
    TSLib scripts use the argparse defaults (lr=0.0001, batch=32,
    epochs=10) unchanged for DLinear.
    """
    return {
        'batch_size': [16, 32, 64, 128],  # TSLib default: 32
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],  # TSLib default: 0.0001
        'train_epochs': [10, 20, 50],  # TSLib default: 10
    }


def transformer_tsl_space() -> Dict[str, List[Any]]:
    """Vanilla Transformer (TSL) — encoder-decoder architecture.

    Paper: "Attention Is All You Need" (NeurIPS 2017, Vaswani et al.)
    arXiv: https://arxiv.org/abs/1706.03762
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/Transformer.sh

    TSLib scripts use the argparse defaults: d_model=512, d_ff=2048,
    n_heads=8, e_layers=2, d_layers=1, dropout=0.1.
    """
    return {
        'batch_size': [16, 32],  # TSLib default: 32
        'learning_rate': [0.0001, 0.0005, 0.001],  # TSLib default: 0.0001
        'd_model': [128, 256, 512],  # TSLib default: 512
        'd_ff': [256, 512, 2048],  # TSLib default: 2048
        'n_heads': [4, 8],  # TSLib default: 8
        'e_layers': [2, 3],  # TSLib default: 2
        'd_layers': [1, 2],  # TSLib default: 1
        'dropout': [0.0, 0.1, 0.2],  # TSLib default: 0.1
        'train_epochs': [10, 20, 50],  # TSLib default: 10
    }


def informer_space() -> Dict[str, List[Any]]:
    """Informer search space — ProbSparse attention.

    Paper: "Informer: Beyond Efficient Transformer for Long Sequence
           Time-Series Forecasting" (AAAI 2021 Best Paper, Zhou et al.)
    arXiv: https://arxiv.org/abs/2012.07436
    Code:  https://github.com/zhouhaoyi/Informer2020
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/Informer.sh

    TSLib scripts use argparse defaults: d_model=512, d_ff=2048,
    n_heads=8, e_layers=2, d_layers=1, factor=3 (overrides default 1).
    ``distil=True`` by default (argparse ``store_false``).
    """
    return {
        'batch_size': [16, 32],  # TSLib default: 32
        'learning_rate': [0.0001, 0.0005, 0.001],  # TSLib default: 0.0001
        'd_model': [256, 512],  # TSLib default: 512
        'd_ff': [512, 1024, 2048],  # TSLib default: 2048
        'n_heads': [4, 8],  # TSLib default: 8
        'e_layers': [2, 3],  # TSLib default: 2
        'd_layers': [1, 2],  # TSLib default: 1
        'dropout': [0.0, 0.1, 0.2],  # TSLib default: 0.1
        'factor': [1, 3, 5],  # TSLib scripts: 3; controls ProbSparse sampling
        'train_epochs': [10, 20, 50],  # TSLib default: 10
    }


def autoformer_space() -> Dict[str, List[Any]]:
    """Autoformer search space — auto-correlation + decomposition.

    Paper: "Autoformer: Decomposition Transformers with Auto-Correlation
           for Long-Term Series Forecasting" (NeurIPS 2021, Wu et al.)
    arXiv: https://arxiv.org/abs/2106.13008
    Code:  https://github.com/thuml/Autoformer
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/Autoformer.sh

    TSLib scripts use argparse defaults: d_model=512, d_ff=2048,
    n_heads=8, e_layers=2, d_layers=1, factor=3 (overrides default 1),
    moving_avg=25.
    """
    return {
        'batch_size': [16, 32],  # TSLib default: 32
        'learning_rate': [0.0001, 0.0005, 0.001],  # TSLib default: 0.0001
        'd_model': [256, 512],  # TSLib default: 512
        'd_ff': [512, 1024, 2048],  # TSLib default: 2048
        'n_heads': [4, 8],  # TSLib default: 8
        'e_layers': [2, 3],  # TSLib default: 2
        'd_layers': [1],  # TSLib default: 1
        'dropout': [0.0, 0.1, 0.2],  # TSLib default: 0.1
        'moving_avg': [13, 25],  # TSLib default: 25
        'factor': [1, 3],  # TSLib scripts: 3
        'train_epochs': [10, 20, 50],  # TSLib default: 10
    }


def fedformer_space() -> Dict[str, List[Any]]:
    """FEDformer search space — frequency-enhanced decomposition.

    Paper: "FEDformer: Frequency Enhanced Decomposed Transformer for
           Long-term Series Forecasting" (ICML 2022, Zhou et al.)
    Proc:  https://proceedings.mlr.press/v162/zhou22g.html
    Code:  https://github.com/MAZiqing/FEDformer
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/FEDformer.sh

    TSLib scripts use argparse defaults: d_model=512, d_ff=2048,
    e_layers=2, d_layers=1, factor=3, moving_avg=25.
    """
    return {
        'batch_size': [16, 32],  # TSLib default: 32
        'learning_rate': [0.0001, 0.0005, 0.001],  # TSLib default: 0.0001
        'd_model': [256, 512],  # TSLib default: 512
        'd_ff': [512, 1024, 2048],  # TSLib default: 2048
        'n_heads': [4, 8],  # TSLib default: 8
        'e_layers': [2, 3],  # TSLib default: 2
        'd_layers': [1],  # TSLib default: 1
        'dropout': [0.0, 0.1, 0.2],  # TSLib default: 0.1
        'moving_avg': [13, 25],  # TSLib default: 25
        'train_epochs': [10, 20, 50],  # TSLib default: 10
    }


def itransformer_space() -> Dict[str, List[Any]]:
    """iTransformer search space — inverted attention on variables.

    Paper: "iTransformer: Inverted Transformers Are Effective for Time
           Series Forecasting" (ICLR 2024 Spotlight, Liu et al.)
    arXiv: https://arxiv.org/abs/2310.06625
    Code:  https://github.com/thuml/iTransformer
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/iTransformer.sh

    TSLib Weather script overrides: d_model=512, d_ff=512, e_layers=3.
    Note: d_ff=512 is much smaller than the argparse default (2048) and
    equals d_model — this is intentional for the inverted architecture.
    """
    return {
        'batch_size': [16, 32],  # TSLib default: 32
        'learning_rate': [0.0001, 0.0005, 0.001],  # TSLib default: 0.0001
        'd_model': [128, 256, 512],  # TSLib Weather: 512
        'd_ff': [128, 256, 512],  # TSLib Weather: 512 (intentionally = d_model)
        'n_heads': [4, 8],  # TSLib default: 8
        'e_layers': [2, 3, 4],  # TSLib Weather: 3
        'dropout': [0.0, 0.1, 0.2],  # TSLib default: 0.1
        'train_epochs': [10, 20, 50],  # TSLib default: 10
    }


def patchtst_space() -> Dict[str, List[Any]]:
    """PatchTST search space — patch-based channel-independent.

    Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with
           Transformers" (ICLR 2023, Nie et al.)
    arXiv: https://arxiv.org/abs/2211.14730
    Code:  https://github.com/yuqinie98/PatchTST
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh

    TSLib scripts use argparse defaults (d_model=512, d_ff=2048) but
    override e_layers and n_heads per horizon: e_layers=1 for short
    horizons (96), n_heads=2/8/16 depending on pred_len.
    """
    return {
        'batch_size': [16, 32, 64, 128],  # TSLib default: 32
        'learning_rate': [0.0001, 0.0005, 0.001],  # TSLib default: 0.0001
        'd_model': [128, 256, 512],  # TSLib default: 512
        'd_ff': [256, 512, 2048],  # TSLib default: 2048
        'n_heads': [2, 4, 8, 16],  # TSLib scripts: 2/8/16 by pred_len
        'e_layers': [1, 2, 3],  # TSLib scripts: 1 (short), varies
        'dropout': [0.0, 0.1, 0.2],  # TSLib default: 0.1
        'train_epochs': [10, 50, 100],  # TSLib default: 10
    }


def timesnet_space() -> Dict[str, List[Any]]:
    """TimesNet search space — temporal 2D-variation modeling.

    Paper: "TimesNet: Temporal 2D-Variation Modeling for General Time
           Series Analysis" (ICLR 2023, Wu et al.)
    arXiv: https://arxiv.org/abs/2210.02186
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/TimesNet.sh

    TSLib Weather script overrides: d_model=16, d_ff=32, e_layers=2,
    top_k=5, num_kernels=6. These are MUCH smaller than the argparse
    defaults (d_model=512, d_ff=2048) — TimesNet uses tiny hidden dims
    because the 2D-variation mechanism is already parameter-heavy.
    """
    return {
        'batch_size': [16, 32],  # TSLib default: 32
        'learning_rate': [0.0001, 0.0005, 0.001],  # TSLib default: 0.0001
        'd_model': [16, 32, 64, 128],  # TSLib Weather: 16 (!)
        'd_ff': [32, 64, 128, 256],  # TSLib Weather: 32 (!)
        'e_layers': [2, 3],  # TSLib default: 2
        'top_k': [3, 5],  # TSLib default: 5
        'num_kernels': [3, 6],  # TSLib default: 6
        'dropout': [0.0, 0.1, 0.2],  # TSLib default: 0.1
        'train_epochs': [10, 20, 50],  # TSLib default: 10
    }


def timemixer_space() -> Dict[str, List[Any]]:
    """TimeMixer search space — decomposable multi-scale mixing.

    Paper: "TimeMixer: Decomposable Multiscale Mixing for Time Series
           Forecasting" (ICLR 2024, Wang et al.)
    OpenReview: https://openreview.net/pdf?id=7oLshfEIC2
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/TimeMixer.sh

    TSLib Weather script overrides: d_model=16, d_ff=32, e_layers=3,
    learning_rate=0.01, batch_size=128, down_sampling_layers=3,
    down_sampling_window=2, down_sampling_method=avg, train_epochs=20.
    Like TimesNet, TimeMixer uses very small d_model/d_ff values.
    """
    return {
        'batch_size': [16, 32, 128],  # TSLib Weather: 128
        'learning_rate': [0.0001, 0.001, 0.01],  # TSLib Weather: 0.01
        'd_model': [16, 32, 64, 128],  # TSLib Weather: 16
        'd_ff': [32, 64, 128, 256],  # TSLib Weather: 32
        'e_layers': [2, 3, 4],  # TSLib Weather: 3
        'dropout': [0.0, 0.1, 0.2],  # TSLib default: 0.1
        'down_sampling_layers': [2, 3],  # TSLib Weather: 3
        'down_sampling_window': [2],  # TSLib Weather: 2
        'train_epochs': [10, 20, 50],  # TSLib Weather: 20
    }


def timexer_space() -> Dict[str, List[Any]]:
    """TimeXer search space — exogenous variable integration via patches.

    Paper: "TimeXer: Empowering Transformers for Time Series Forecasting
           with Exogenous Variables" (NeurIPS 2024, Wang et al.)
    arXiv: https://arxiv.org/abs/2402.19072
    Code:  https://github.com/thuml/TimeXer
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/TimeXer.sh

    TSLib Weather script overrides per horizon: d_model=128/256,
    d_ff=512/1024, e_layers=1/3, patch_len=16 (argparse default).
    """
    return {
        'batch_size': [4, 16, 32],  # TSLib Weather: 4
        'learning_rate': [0.0001, 0.0005, 0.001],  # TSLib default: 0.0001
        'd_model': [128, 256, 512],  # TSLib Weather: 128/256
        'd_ff': [256, 512, 1024],  # TSLib Weather: 512/1024
        'n_heads': [4, 8],  # TSLib default: 8
        'e_layers': [1, 2, 3],  # TSLib Weather: 1/3
        'dropout': [0.0, 0.1, 0.2],  # TSLib default: 0.1
        'train_epochs': [10, 20, 50],  # TSLib default: 10
    }


def mamba_space() -> Dict[str, List[Any]]:
    """Mamba (S-Mamba) search space — selective state-space model.

    Paper: "Mamba: Linear-Time Sequence Modeling with Selective State
           Spaces" (Gu & Dao, 2024)
    arXiv: https://arxiv.org/abs/2312.00752
    Code:  https://github.com/state-spaces/mamba
    TSLib: https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/Mamba.sh

    TSLib Weather script overrides: d_model=128, d_ff=16, e_layers=2,
    expand=2, d_conv=4. Note: d_ff=16 is MUCH smaller than the argparse
    default of 2048 — Mamba relies on the selective SSM scan rather than
    large feed-forward layers. d_conv controls the local convolution
    kernel size in the Mamba block.
    """
    return {
        'batch_size': [16, 32],  # TSLib default: 32
        'learning_rate': [0.0001, 0.0005, 0.001],  # TSLib default: 0.0001
        'd_model': [64, 128, 256],  # TSLib Weather: 128
        'd_ff': [16, 32, 64],  # TSLib Weather: 16 (!); NOT the usual 2048
        'e_layers': [2, 4],  # TSLib Weather: 2
        'd_state': [16, 32],  # SSM state dimension
        'expand': [2],  # TSLib default: 2; Mamba expansion factor
        'd_conv': [4],  # TSLib default: 4; local conv kernel size
        'dropout': [0.0, 0.1],  # TSLib default: 0.1
        'train_epochs': [10, 20, 50],  # TSLib default: 10
    }


def lstm_general_space() -> Dict[str, List[Any]]:
    """Generalized LSTM adapter space (all datasets).

    Custom architecture (liulian LSTMAdapter). Search space adapted from
    the swiss-river LSTM space with broader ranges for general datasets.
    """
    return {
        'batch_size': [32, 64, 128, 256],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'hidden_size': [32, 64, 128, 256],
        'num_layers': [1, 2, 3],
        'dropout': [0.0, 0.1, 0.2, 0.3],
    }


def transformer_enc_general_space() -> Dict[str, List[Any]]:
    """Generalized Transformer-Encoder adapter space (all datasets).

    Custom architecture (liulian TransformerEncoderAdapter). Search space
    adapted from the swiss-river Transformer space with broader ranges.
    """
    return {
        'batch_size': [32, 64, 128, 256],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'dropout': [0.0, 0.1, 0.2, 0.3],
        'num_t_heads': [2, 4, 8],
        'ratio_heads_to_d_model': [4, 8, 16],
        'dim_feedforward': [64, 128, 256],
        'num_layers': [1, 2, 3],
    }


# ======================================================================
# ASHA presets for TSL models
# Ref: https://docs.ray.io/en/latest/tune/api/schedulers.html#asha
# ======================================================================


def asha_fast() -> Dict[str, Any]:
    """ASHA preset for fast models (DLinear, LSTM) — aggressive pruning.

    Suitable for models that converge quickly and have low per-epoch cost.
    """
    return {
        'scheduler': 'asha',
        'max_epochs': 100,
        'grace_period': 5,
        'reduction_factor': 3,
    }


def asha_medium() -> Dict[str, Any]:
    """ASHA preset for medium models (PatchTST, iTransformer, TimesNet).

    Balanced pruning for models with moderate training cost.
    """
    return {
        'scheduler': 'asha',
        'max_epochs': 50,
        'grace_period': 5,
        'reduction_factor': 2,
    }


def asha_slow() -> Dict[str, Any]:
    """ASHA preset for slow models (Transformer enc-dec, TimeLLM).

    Conservative pruning for expensive models that need more warm-up.
    """
    return {
        'scheduler': 'asha',
        'max_epochs': 30,
        'grace_period': 3,
        'reduction_factor': 2,
    }


# ======================================================================
# Registry / lookup
# ======================================================================

_SPACE_REGISTRY: Dict[str, Any] = {
    # Swiss-river model families
    'swiss_lstm': swiss_lstm_space,
    'swiss_transformer': swiss_transformer_space,
    'swiss_lstm_embedding': swiss_lstm_embedding_space,
    'swiss_transformer_embedding': swiss_transformer_embedding_space,
    'swiss_stgnn': swiss_stgnn_space,
    # TimeLLM dataset families
    'timellm_etth1': timellm_etth1_space,
    'timellm_etth2': timellm_etth2_space,
    'timellm_ettm': timellm_ettm_space,
    'timellm_weather': timellm_weather_space,
    'timellm_electricity': timellm_electricity_space,
    'timellm_swissriver': timellm_swissriver_space,
    # TSL model-centric spaces (general-purpose)
    'dlinear': dlinear_space,
    'transformer': transformer_tsl_space,
    'informer': informer_space,
    'autoformer': autoformer_space,
    'fedformer': fedformer_space,
    'itransformer': itransformer_space,
    'patchtst': patchtst_space,
    'timesnet': timesnet_space,
    'timemixer': timemixer_space,
    'timexer': timexer_space,
    'mamba': mamba_space,
    'lstm': lstm_general_space,
    'lstm_general': lstm_general_space,
    'transformer_encoder': transformer_enc_general_space,
    'transformer_enc': transformer_enc_general_space,
}

_ASHA_REGISTRY: Dict[str, Any] = {
    'default': swiss_asha_default,
    'soft': swiss_asha_soft,
    'single_soft': swiss_asha_single_soft,
    'single_hard': swiss_asha_single_hard,
    'fast': asha_fast,
    'medium': asha_medium,
    'slow': asha_slow,
}


def get_search_space(name: str) -> Dict[str, List[Any]]:
    """Look up a pre-defined search space by name.

    Args:
        name: Registry key (e.g. ``"swiss_lstm"``, ``"timellm_etth1"``).

    Returns:
        Search space dictionary (plain lists).
    """
    fn = _SPACE_REGISTRY.get(name.strip().lower())
    if fn is None:
        raise ValueError(
            f"Unknown search space '{name}'. Available: {sorted(_SPACE_REGISTRY)}"
        )
    return fn()


def get_asha_preset(name: str = 'default') -> Dict[str, Any]:
    """Look up an ASHA scheduler preset by name.

    Args:
        name: One of ``"default"``, ``"soft"``, ``"single_soft"``,
            ``"single_hard"``.

    Returns:
        Dict suitable for passing into ``RayOptimizer`` config.
    """
    fn = _ASHA_REGISTRY.get(name.strip().lower())
    if fn is None:
        raise ValueError(
            f"Unknown ASHA preset '{name}'. Available: {sorted(_ASHA_REGISTRY)}"
        )
    return fn()
