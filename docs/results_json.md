# Results JSON Reference

After each experiment run, Liulian saves a comprehensive `results.json`
file in the experiment's artifacts directory.  This document describes
every field in that file.

## Top-Level Structure

```json
{
  "experiment": { ... },
  "data": { ... },
  "model": { ... },
  "training": { ... },
  "hpo": { ... },
  "metrics": { ... },
  "history": { ... },
  "timing": { ... },
  "gpu": { ... },
  "artifacts_dir": "artifacts/..."
}
```

---

## `experiment`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Auto-generated identifier: `{data}_{model}_{task}_{split_mode}` |
| `timestamp` | `str` | ISO 8601 timestamp when the run completed |
| `seed` | `int` | Random seed used for reproducibility |
| `quick_test` | `bool` | Whether `--quick_test` mode was active |

## `data`

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | `str` | Dataset identifier (e.g. `traffic`, `swiss-river-1990`, `PEMS03`) |
| `features` | `str` | Feature mode: `M` (multivariate), `S` (single), `MS` (multi→single) |
| `seq_len` | `int` | Look-back window (input sequence length) |
| `pred_len` | `int` | Forecast horizon (prediction length) |
| `split_mode` | `str` | How channels are handled: `multi_channel`, `per_entity`, `channel_independent` |
| `scaler` | `str` | Normalisation strategy: `standard`, `minmax`, `none` |
| `train_split` | `float` | Fraction of data used for training (e.g. `0.7`) |
| `noise_type` | `str\|null` | Noise injection type, or `null` if none applied |
| `noise_level` | `float\|null` | Noise amplitude (only present when noise is active) |
| `identifier_mode` | `str` | Entity identifier strategy (`none`, `embedding`, `one_hot`, etc.) |
| `graph_mode` | `str` | Spatial graph mode (`none`, `edge_index`, `adj_matrix`, etc.) |

## `model`

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | Model architecture name (e.g. `patchtst`, `dlinear`, `lstm`) |
| `enc_in` | `int` | Number of input channels (auto-detected from data) |
| `d_model` | `int` | Hidden dimension size |
| `d_ff` | `int` | Feed-forward dimension |
| `n_heads` | `int` | Number of attention heads (Transformer models) |
| `e_layers` | `int` | Number of encoder layers |
| `d_layers` | `int` | Number of decoder layers |
| `dropout` | `float` | Dropout rate |
| `patch_len` | `int` | Patch length (PatchTST) |
| `stride` | `int` | Patch stride (PatchTST) |
| `individual` | `bool\|null` | Per-channel independence (DLinear) |
| `total_params` | `int` | Total number of model parameters |
| `trainable_params` | `int` | Number of trainable (non-frozen) parameters |

## `training`

| Field | Type | Description |
|-------|------|-------------|
| `epochs` | `int` | Maximum training epochs |
| `batch_size` | `int` | Training batch size |
| `learning_rate` | `float` | Initial learning rate |
| `lr_scheduler` | `str` | LR schedule type (e.g. `type1`, `cosine_warmup`, `none`) |
| `loss` | `str` | Loss function (`mse`, `mae`, `rmse`) |
| `patience` | `int` | Early-stopping patience (epochs without improvement) |
| `eval_denorm` | `bool` | Whether metrics are computed on denormalised predictions |

## `hpo`

Present only when HPO was enabled (`hpo: true`).  `null` otherwise.

| Field | Type | Description |
|-------|------|-------------|
| `best_value` | `float` | Best objective value found by HPO |
| `n_trials` | `int` | Total number of HPO trials completed |
| `best_hparams` | `dict` | Dictionary of best hyper-parameter values |

## `metrics`

A flat dictionary of final evaluation metrics.  The exact keys depend on
the `metrics` configuration.  Common fields:

| Field | Type | Description |
|-------|------|-------------|
| `mse` | `float` | Mean Squared Error |
| `rmse` | `float` | Root Mean Squared Error |
| `mae` | `float` | Mean Absolute Error |
| `nse` | `float` | Nash–Sutcliffe Efficiency (hydrology) |
| `r2` | `float` | Coefficient of Determination |
| `mape` | `float` | Mean Absolute Percentage Error |

> **Note**: Only the metrics specified in `config.metrics` are computed
> and stored.  Liulian does not compute metrics that were not requested.

## `history`

Per-split training loss summaries (if available).  `null` when not
recorded.

```json
{
  "train_loss": {
    "final": 0.023,
    "best": 0.019,
    "n_epochs": 30
  },
  "val_loss": {
    "final": 0.031,
    "best": 0.028,
    "n_epochs": 30
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `final` | `float` | Loss at the last completed epoch |
| `best` | `float` | Best (minimum) loss observed across all epochs |
| `n_epochs` | `int` | Total number of epochs recorded |

## `timing`

| Field | Type | Description |
|-------|------|-------------|
| `total_seconds` | `float` | Wall-clock time for the full pipeline (seconds) |
| `total_human` | `str` | Human-readable duration (e.g. `2m 34s`, `1h 12m 5s`) |

## `gpu`

| Field | Type | Description |
|-------|------|-------------|
| `device` | `str` | GPU device name (e.g. `NVIDIA GeForce RTX 3070`) or `CPU` |
| `cuda_version` | `str\|null` | CUDA toolkit version |
| `memory_gb` | `float\|null` | Total GPU memory in GB |

## `artifacts_dir`

| Field | Type | Description |
|-------|------|-------------|
| `artifacts_dir` | `str` | Path to the directory containing all experiment outputs |

---

## Example

```json
{
  "experiment": {
    "name": "traffic_patchtst_forecast_multi_channel",
    "timestamp": "2026-02-19T14:30:00",
    "seed": 2026,
    "quick_test": false
  },
  "data": {
    "dataset": "traffic",
    "features": "M",
    "seq_len": 96,
    "pred_len": 96,
    "split_mode": "multi_channel",
    "scaler": "standard",
    "train_split": 0.7,
    "noise_type": null,
    "noise_level": null,
    "identifier_mode": "none",
    "graph_mode": "none"
  },
  "model": {
    "type": "patchtst",
    "enc_in": 862,
    "d_model": 512,
    "d_ff": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "dropout": 0.1,
    "patch_len": 16,
    "stride": 8,
    "individual": null,
    "total_params": 2847234,
    "trainable_params": 2847234
  },
  "training": {
    "epochs": 30,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "lr_scheduler": "type1",
    "loss": "mse",
    "patience": 10,
    "eval_denorm": true
  },
  "hpo": null,
  "metrics": {
    "mse": 0.015234,
    "rmse": 0.123432,
    "mae": 0.084521
  },
  "history": {
    "train_loss": {
      "final": 0.018,
      "best": 0.015,
      "n_epochs": 30
    }
  },
  "timing": {
    "total_seconds": 1847.23,
    "total_human": "30m 47s"
  },
  "gpu": {
    "device": "NVIDIA GeForce RTX 3070",
    "cuda_version": "12.8",
    "memory_gb": 7.8
  },
  "artifacts_dir": "artifacts/traffic_patchtst_forecast_multi_channel_20260219_143000"
}
```
