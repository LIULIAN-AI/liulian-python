# Search Spaces & Hyperparameter References

Pre-defined HPO search spaces for all supported model architectures. Each space is cross-referenced against the model's original paper and the canonical [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) training scripts.

## Usage

```python
from liulian.optim.search_spaces import get_search_space, get_asha_preset

# Look up a space by name
space = get_search_space("patchtst")

# With Ray Tune
from ray import tune
ray_space = {k: tune.choice(v) for k, v in space.items()}

# ASHA scheduler preset
asha = get_asha_preset("medium")
```

---

## TSLib Argparse Defaults

All TSL models inherit these defaults from `run.py` unless overridden by per-model scripts:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `d_model` | 512 | Model dimension |
| `d_ff` | 2048 | Feed-forward dimension |
| `n_heads` | 8 | Attention heads |
| `e_layers` | 2 | Encoder layers |
| `d_layers` | 1 | Decoder layers |
| `dropout` | 0.1 | Dropout rate |
| `factor` | 1 | Attention factor |
| `moving_avg` | 25 | Moving average window |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 0.0001 | Learning rate |
| `train_epochs` | 10 | Training epochs |
| `patience` | 3 | Early stopping patience |
| `top_k` | 5 | TimesNet: top-k frequencies |
| `num_kernels` | 6 | TimesNet: Inception kernels |
| `expand` | 2 | Mamba: expansion factor |
| `d_conv` | 4 | Mamba: conv kernel size |
| `patch_len` | 16 | TimeXer: patch length |

Source: [TSLib run.py](https://github.com/thuml/Time-Series-Library/blob/main/run.py)

---

## TSL Model Spaces

### DLinear

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 16, 32, 64, 128 | 32 |
| `learning_rate` | 0.0001, 0.0005, 0.001, 0.005 | 0.0001 |
| `train_epochs` | 10, 20, 50 | 10 |

DLinear has no model-specific hyperparameters (no attention, d_model, or d_ff). The authors note _"no model hyper-parameter tuning needed"_.

| | |
|-|-|
| **Paper** | _"Are Transformers Effective for Time Series Forecasting?"_ (AAAI 2023, Zeng et al.) |
| **arXiv** | [2205.13504](https://arxiv.org/abs/2205.13504) |
| **Code** | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) |
| **TSLib Script** | [DLinear_ETTh1.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh) |

---

### Transformer

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 16, 32 | 32 |
| `learning_rate` | 0.0001, 0.0005, 0.001 | 0.0001 |
| `d_model` | 128, 256, 512 | 512 |
| `d_ff` | 256, 512, 2048 | 2048 |
| `n_heads` | 4, 8 | 8 |
| `e_layers` | 2, 3 | 2 |
| `d_layers` | 1, 2 | 1 |
| `dropout` | 0.0, 0.1, 0.2 | 0.1 |
| `train_epochs` | 10, 20, 50 | 10 |

| | |
|-|-|
| **Paper** | _"Attention Is All You Need"_ (NeurIPS 2017, Vaswani et al.) |
| **arXiv** | [1706.03762](https://arxiv.org/abs/1706.03762) |
| **TSLib Script** | [Transformer.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/Transformer.sh) |

---

### Informer

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 16, 32 | 32 |
| `learning_rate` | 0.0001, 0.0005, 0.001 | 0.0001 |
| `d_model` | 256, 512 | 512 |
| `d_ff` | 512, 1024, 2048 | 2048 |
| `n_heads` | 4, 8 | 8 |
| `e_layers` | 2, 3 | 2 |
| `d_layers` | 1, 2 | 1 |
| `dropout` | 0.0, 0.1, 0.2 | 0.1 |
| `factor` | 1, 3, 5 | 3 (scripts) |
| `train_epochs` | 10, 20, 50 | 10 |

| | |
|-|-|
| **Paper** | _"Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"_ (AAAI 2021 Best Paper, Zhou et al.) |
| **arXiv** | [2012.07436](https://arxiv.org/abs/2012.07436) |
| **Code** | [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020) |
| **TSLib Script** | [Informer.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/Informer.sh) |

---

### Autoformer

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 16, 32 | 32 |
| `learning_rate` | 0.0001, 0.0005, 0.001 | 0.0001 |
| `d_model` | 256, 512 | 512 |
| `d_ff` | 512, 1024, 2048 | 2048 |
| `n_heads` | 4, 8 | 8 |
| `e_layers` | 2, 3 | 2 |
| `d_layers` | 1 | 1 |
| `dropout` | 0.0, 0.1, 0.2 | 0.1 |
| `moving_avg` | 13, 25 | 25 |
| `factor` | 1, 3 | 3 (scripts) |
| `train_epochs` | 10, 20, 50 | 10 |

| | |
|-|-|
| **Paper** | _"Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"_ (NeurIPS 2021, Wu et al.) |
| **arXiv** | [2106.13008](https://arxiv.org/abs/2106.13008) |
| **Code** | [thuml/Autoformer](https://github.com/thuml/Autoformer) |
| **TSLib Script** | [Autoformer.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/Autoformer.sh) |

---

### FEDformer

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 16, 32 | 32 |
| `learning_rate` | 0.0001, 0.0005, 0.001 | 0.0001 |
| `d_model` | 256, 512 | 512 |
| `d_ff` | 512, 1024, 2048 | 2048 |
| `n_heads` | 4, 8 | 8 |
| `e_layers` | 2, 3 | 2 |
| `d_layers` | 1 | 1 |
| `dropout` | 0.0, 0.1, 0.2 | 0.1 |
| `moving_avg` | 13, 25 | 25 |
| `train_epochs` | 10, 20, 50 | 10 |

| | |
|-|-|
| **Paper** | _"FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"_ (ICML 2022, Zhou et al.) |
| **Proceedings** | [PMLR v162](https://proceedings.mlr.press/v162/zhou22g.html) |
| **Code** | [MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer) |
| **TSLib Script** | [FEDformer.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ECL_script/FEDformer.sh) |

---

### iTransformer

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 16, 32 | 32 |
| `learning_rate` | 0.0001, 0.0005, 0.001 | 0.0001 |
| `d_model` | 128, 256, 512 | 512 |
| `d_ff` | 128, 256, 512 | 512 (!) |
| `n_heads` | 4, 8 | 8 |
| `e_layers` | 2, 3, 4 | 3 |
| `dropout` | 0.0, 0.1, 0.2 | 0.1 |
| `train_epochs` | 10, 20, 50 | 10 |

!!! note "d_ff = d_model"
    iTransformer uses d_ff=512 (same as d_model), much smaller than the TSLib argparse default of 2048. This is intentional for the inverted architecture where attention operates over variables rather than time steps.

| | |
|-|-|
| **Paper** | _"iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"_ (ICLR 2024 Spotlight, Liu et al.) |
| **arXiv** | [2310.06625](https://arxiv.org/abs/2310.06625) |
| **Code** | [thuml/iTransformer](https://github.com/thuml/iTransformer) |
| **TSLib Script** | [iTransformer.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/iTransformer.sh) |

---

### PatchTST

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 16, 32, 64, 128 | 32 |
| `learning_rate` | 0.0001, 0.0005, 0.001 | 0.0001 |
| `d_model` | 128, 256, 512 | 512 |
| `d_ff` | 256, 512, 2048 | 2048 |
| `n_heads` | 2, 4, 8, 16 | varies |
| `e_layers` | 1, 2, 3 | varies |
| `dropout` | 0.0, 0.1, 0.2 | 0.1 |
| `train_epochs` | 10, 50, 100 | 10 |

!!! note "Variable e_layers and n_heads"
    TSLib scripts use e_layers=1 for short horizons (96) and n_heads=2/8/16 depending on pred_len, rather than fixed values.

| | |
|-|-|
| **Paper** | _"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"_ (ICLR 2023, Nie et al.) |
| **arXiv** | [2211.14730](https://arxiv.org/abs/2211.14730) |
| **Code** | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) |
| **TSLib Script** | [PatchTST_ETTh1.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh) |

---

### TimesNet

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 16, 32 | 32 |
| `learning_rate` | 0.0001, 0.0005, 0.001 | 0.0001 |
| `d_model` | 16, 32, 64, 128 | **16** (!) |
| `d_ff` | 32, 64, 128, 256 | **32** (!) |
| `e_layers` | 2, 3 | 2 |
| `top_k` | 3, 5 | 5 |
| `num_kernels` | 3, 6 | 6 |
| `dropout` | 0.0, 0.1, 0.2 | 0.1 |
| `train_epochs` | 10, 20, 50 | 10 |

!!! warning "Tiny d_model / d_ff"
    TimesNet uses d_model=16 and d_ff=32 in TSLib scripts — much smaller than the argparse defaults (512 / 2048). This is because the 2D-variation mechanism with Inception convolutions is already parameter-heavy.

| | |
|-|-|
| **Paper** | _"TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"_ (ICLR 2023, Wu et al.) |
| **arXiv** | [2210.02186](https://arxiv.org/abs/2210.02186) |
| **TSLib Script** | [TimesNet.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/TimesNet.sh) |

---

### TimeMixer

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 16, 32, 128 | **128** |
| `learning_rate` | 0.0001, 0.001, 0.01 | **0.01** |
| `d_model` | 16, 32, 64, 128 | **16** |
| `d_ff` | 32, 64, 128, 256 | **32** |
| `e_layers` | 2, 3, 4 | 3 |
| `dropout` | 0.0, 0.1, 0.2 | 0.1 |
| `down_sampling_layers` | 2, 3 | 3 |
| `down_sampling_window` | 2 | 2 |
| `train_epochs` | 10, 20, 50 | 20 |

!!! warning "Non-standard defaults"
    TimeMixer scripts diverge significantly from TSLib argparse defaults: lr=0.01 (vs 0.0001), batch_size=128 (vs 32), d_model=16 (vs 512).

| | |
|-|-|
| **Paper** | _"TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"_ (ICLR 2024, Wang et al.) |
| **OpenReview** | [7oLshfEIC2](https://openreview.net/pdf?id=7oLshfEIC2) |
| **TSLib Script** | [TimeMixer.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/TimeMixer.sh) |

---

### TimeXer

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 4, 16, 32 | **4** |
| `learning_rate` | 0.0001, 0.0005, 0.001 | 0.0001 |
| `d_model` | 128, 256, 512 | 128/256 |
| `d_ff` | 256, 512, 1024 | 512/1024 |
| `n_heads` | 4, 8 | 8 |
| `e_layers` | 1, 2, 3 | 1/3 |
| `dropout` | 0.0, 0.1, 0.2 | 0.1 |
| `train_epochs` | 10, 20, 50 | 10 |

| | |
|-|-|
| **Paper** | _"TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"_ (NeurIPS 2024, Wang et al.) |
| **arXiv** | [2402.19072](https://arxiv.org/abs/2402.19072) |
| **Code** | [thuml/TimeXer](https://github.com/thuml/TimeXer) |
| **TSLib Script** | [TimeXer.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/TimeXer.sh) |

---

### Mamba (S-Mamba)

| Key | Values | TSLib Default |
|-----|--------|---------------|
| `batch_size` | 16, 32 | 32 |
| `learning_rate` | 0.0001, 0.0005, 0.001 | 0.0001 |
| `d_model` | 64, 128, 256 | 128 |
| `d_ff` | 16, 32, 64 | **16** (!) |
| `e_layers` | 2, 4 | 2 |
| `d_state` | 16, 32 | — |
| `expand` | 2 | 2 |
| `d_conv` | 4 | 4 |
| `dropout` | 0.0, 0.1 | 0.1 |
| `train_epochs` | 10, 20, 50 | 10 |

!!! warning "d_ff = 16"
    Mamba uses d_ff=16 in TSLib scripts — drastically smaller than the 2048 default. Mamba relies on the selective SSM scan mechanism rather than large feed-forward layers. The `d_conv` parameter controls the local convolution kernel size in each Mamba block.

| | |
|-|-|
| **Paper** | _"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"_ (Gu & Dao, 2024) |
| **arXiv** | [2312.00752](https://arxiv.org/abs/2312.00752) |
| **Code** | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| **TSLib Script** | [Mamba.sh](https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/Weather_script/Mamba.sh) |

---

## TimeLLM Spaces

Dataset-specific grids derived from the [Time-LLM](https://github.com/KimMeen/Time-LLM) training scripts.

| | |
|-|-|
| **Paper** | _"Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"_ (ICLR 2024) |
| **arXiv** | [2310.01728](https://arxiv.org/abs/2310.01728) |
| **Code** | [KimMeen/Time-LLM](https://github.com/KimMeen/Time-LLM) |

Available spaces: `timellm_etth1`, `timellm_etth2`, `timellm_ettm`, `timellm_weather`, `timellm_electricity`, `timellm_swissriver`.

---

## Swiss-River Spaces

Grids from the [swiss-river-network-benchmark](https://github.com/RiverNetwork/swiss-river-network-benchmark) `ray_tune.py`.

Available spaces: `swiss_lstm`, `swiss_transformer`, `swiss_lstm_embedding`, `swiss_transformer_embedding`, `swiss_stgnn`.

---

## Custom Model Spaces

### LSTM (General)

Adapted from the swiss-river LSTM space with broader ranges for general datasets.

Available as: `lstm` or `lstm_general`.

### Transformer Encoder (General)

Adapted from the swiss-river Transformer space with broader ranges.

Available as: `transformer_encoder` or `transformer_enc`.

---

## ASHA Scheduler Presets

| Preset | Max Epochs | Grace Period | Reduction Factor | Use Case |
|--------|-----------|-------------|-----------------|----------|
| `default` | 200 | 3 | 2 | Swiss-river default |
| `soft` | 200 | 5 | 1.5 | Swiss-river gentle pruning |
| `single_soft` | 500 | 5 | 1.5 | Single model, gentle |
| `single_hard` | 500 | 3 | 2 | Single model, aggressive |
| `fast` | 100 | 5 | 3 | DLinear, LSTM (fast models) |
| `medium` | 50 | 5 | 2 | PatchTST, iTransformer, TimesNet |
| `slow` | 30 | 3 | 2 | Transformer enc-dec, TimeLLM |

Reference: [Ray Tune ASHA Documentation](https://docs.ray.io/en/latest/tune/api/schedulers.html#asha)
