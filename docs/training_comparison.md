# Training Settings Comparison

> **Purpose**: Compare training configurations across the three projects that
> inform liulian's design — Time-LLM-Revised (time-series library family),
> swiss-river-network-benchmark, and liulian itself — so we can identify
> settings that may cause bad results and divergences worth investigating.

## Side-by-Side Comparison

| Setting | **Time-LLM-Revised** | **swiss-river-benchmark** | **liulian** |
|---|---|---|---|
| **learning_rate** | `0.0001` (default); `0.001` (SRNB cfg); `0.01` (ETTh1) | Ray: `uniform(1e-5, 0.01)` | `0.001` |
| **train_epochs** | `10` (default); `30` (SRNB cfg); `100` (ETTh1) | `30` (fixed in search) | `30` |
| **batch_size** | `32` (default); `8` (SRNB cfg); `24` (ETTh1/Traffic) | LSTM: `randint(32, 257)`; STGNN: `randint(1, 5)` | `8` |
| **optimizer** | Adam | Adam | Adam |
| **weight_decay** | 0 (implicit) | 0 (implicit) | 0 (implicit) |
| **lr_scheduler** | `OneCycleLR` (pct_start=0.2) or `CosineAnnealingLR` (T_max=20, eta_min=1e-8). Multiple custom modes (type1 = halve each epoch, type2 = milestones, PEMS, TST, constant) | **None** | `OneCycleLR` (pct_start=0.2) or `CosineAnnealingLR` (T_max=20, eta_min=1e-8) |
| **loss** | `MSELoss` | `MSELoss` (NaN-masked) | Configurable: `mse`, `mae`, `rmse` |
| **early stopping** | Yes — patience `7`–`10`, monitors val loss | **No** — relies on ASHA scheduler pruning | Yes — patience `10`, monitors `val_{loss_name}` |
| **seq_len** | `96` (default); `90` (SRNB cfg); `512` (ETTh1) | `90` (window_len) | `90` |
| **label_len** | `48` (default); `0` (SRNB cfg) | N/A (no decoder) | `0` |
| **pred_len** | `96` (default); `7` (SRNB cfg) | `future_steps` varies | `7` |
| **gradient clipping** | Not used (AMP scaler when `use_amp`) | Not used | Not used |
| **warmup** | Implicit via OneCycleLR 20% ramp | None | Implicit via OneCycleLR 20% ramp |
| **dropout** | `0.1` | Transformer: `uniform(0, 0.5)`; LSTM: none | `0.1` |
| **eval metrics** | MSE + MAE | Train: MSE; Val: MSE + RMSE; Test: RMSE + MAE + NSE (de-normalised) | Configurable: default `rmse, mae, nse` |
| **normalisation** | `StandardScaler` (per dataset) | `MinMaxScaler` (per column per station) | Via `SwissRiverDataset` |
| **mixed precision** | Supported (DeepSpeed + Accelerate bf16) | No | No |
| **seed** | `2021` | Not set | `2026` |
| **train/val split** | Preset per-dataset files | `0.8` | `0.8` |
| **d_model** | `16`–`32` | LSTM hidden: varies; Transformer varies | `64` |
| **HPO** | Manual `--itr` repetitions | Ray Tune + ASHA (grace=3–5, reduction=1.5–2, max_t=200–500) | Optional Ray Tune + ASHA |
| **distributed** | Hugging Face Accelerate + DeepSpeed ZeRO-2 | Single GPU per Ray trial | No |
| **NaN handling** | None | Mask-based: loss on valid values only | None |

## Key Observations & Likely Causes of Bad Results

### 1. Learning Rate
- Swiss-river uses HPO over a **wide** range (`1e-5` to `0.01`), while
  liulian defaults to `0.001`. If the optimal LR for a particular
  station/model is significantly different, a fixed LR may underperform.
- **Recommendation**: Either use HPO or try `1e-4` to `5e-3` sweeps.

### 2. LR Scheduler vs No Scheduler
- Swiss-river uses **no LR scheduler** — flat LR through all epochs.
  liulian (inherited from Time-LLM) uses OneCycleLR by default.
- OneCycleLR can cause instability if `pct_start` or `max_lr` are wrong
  for the smaller swiss-river dataset. The 20% warmup ramps to a peak
  that may overshoot for small batch sizes.
- **Recommendation**: Try `lradj=constant` when comparing against
  swiss-river baselines to isolate this effect.

### 3. Batch Size
- Swiss-river LSTM sweeps `32–256`; liulian defaults to `8`.
  Very small batches introduce high gradient variance, which interacts
  poorly with OneCycleLR.
- **Recommendation**: Increase to `32` minimum for fair comparison.

### 4. Early Stopping vs ASHA Pruning
- Swiss-river does **not** do per-run early stopping. It trains for
  exactly `num_epochs` and relies on ASHA to kill bad trials. Liulian's
  early stopping with patience 10 may terminate promising runs too early
  if the validation metric is noisy.
- **Recommendation**: When comparing, disable early stopping
  (`patience=999`) and run for the full `train_epochs`.

### 5. Normalisation
- Swiss-river uses **MinMaxScaler** (per station per column); Time-LLM
  uses **StandardScaler**. Different normalisations change the loss
  landscape — MSE on MinMax-scaled data weights all features equally,
  while MSE on StandardScaler-scaled data effectively weighs by
  inverse variance.
- **Recommendation**: Ensure consistent normalisation when comparing
  metric values. Report metrics on **de-normalised** predictions.

### 6. NaN / Gap Handling
- Swiss-river masks NaN values in loss computation (`mask = ~isnan(y)`).
  Liulian does NOT mask NaNs — if any NaN leaks into predictions, the
  entire loss becomes NaN and training silently fails.
- **Recommendation**: Add NaN-masking to `ForecastTrainer` loss or
  ensure data is fully clean before training.

### 7. Decoder Architecture
- Swiss-river models predict **directly** (no encoder-decoder split).
  liulian (like Time-LLM) constructs a decoder input
  `[label_len zeros ∥ pred_len zeros]`. With `label_len=0` this is
  harmless, but if `label_len > 0` it feeds ground truth into the
  decoder during training — a form of teacher forcing that inflates
  training metrics.
- **Recommendation**: Keep `label_len=0` for swiss-river experiments
  to match the reference architecture.

### 8. Evaluation Metric Mismatch
- Swiss-river computes test metrics on **de-normalised** data.
  Time-LLM computes metrics on normalised data. If liulian computes
  metrics on normalised data while comparing against swiss-river
  de-normalised results, numbers won't be comparable.
- **Recommendation**: Always de-normalise before computing final
  test metrics when comparing across projects.

### 9. d_model Size
- Liulian uses `d_model=64` vs Time-LLM's `16`–`32`. A larger d_model
  increases capacity but may overfit on the small swiss-river dataset.
- **Recommendation**: Sweep `d_model ∈ {16, 32, 64}` to find the best
  for each dataset.

### 10. HPO Strategy
- Swiss-river's ASHA scheduler is critical for finding good configs.
  Without HPO, liulian uses fixed defaults — which may be far from
  optimal for a specific station or model.
- **Recommendation**: Always run at least a small HPO sweep (10–50
  trials) before concluding that a model performs poorly.

## Resolution Status

All ten gaps identified above have been addressed in liulian's codebase.

| # | Gap | Resolution | Module |
|---|---|---|---|
| 1 | Learning Rate | HPO search spaces cover `1e-5` to `0.01` per model/dataset | `optim/search_spaces.py` |
| 2 | LR Scheduler vs None | `lradj` config already supports `constant`; docs now call it out | `runtime/trainer.py` |
| 3 | Batch Size | Per-model HPO spaces include batch_size ranges matching reference | `optim/search_spaces.py` |
| 4 | Early Stopping vs ASHA | `disable_early_stopping` flag, auto-set when Ray ASHA is active | `runtime/trainer.py`, `optim/ray_optimizer.py` |
| 5 | Normalisation | `get_scaler()` factory: `standard`, `minmax`, `none`; `DimSplitScaler`, `StationSplitScaler` wrappers | `data/scalers.py` |
| 6 | NaN / Gap Handling | `nan_mask_loss` config → `_masked_loss()` masks NaN targets before loss | `runtime/trainer.py` |
| 7 | Decoder Architecture | `teacher_forcing` config: `label` (default), `zeros`, `none` with `_build_decoder_input()` | `runtime/trainer.py` |
| 8 | Evaluation Metric Mismatch | `eval_denorm` config + `inverse_transform_fn` → reports both normalised and `denorm_` prefixed metrics | `runtime/trainer.py` |
| 9 | d_model Size | HPO grids include d_model sweeps per model | `optim/search_spaces.py` |
| 10 | HPO Strategy | Pre-built search spaces + ASHA presets matching reference projects | `optim/search_spaces.py` |

Additionally, LSTM and Transformer model families from swiss-river-network-benchmark
have been ported as liulian adapters (`models/torch/swiss_lstm.py`,
`models/torch/swiss_transformer.py`), and optional HF Accelerate + DeepSpeed
integration has been added (`runtime/accelerator.py`).

## Source Files

| Project | Key Files |
|---|---|
| Time-LLM-Revised | [run_main.py](https://github.com/jajupmochi/liulian-python/blob/main/refer_projects/Time-LLM-Revised/run_main.py), [utils/tools.py](https://github.com/jajupmochi/liulian-python/blob/main/refer_projects/Time-LLM-Revised/utils/tools.py), [configs/srnb.yaml](https://github.com/jajupmochi/liulian-python/blob/main/refer_projects/Time-LLM-Revised/configs/srnb.yaml), [scripts/TimeLLM_ETTh1.sh](https://github.com/jajupmochi/liulian-python/blob/main/refer_projects/Time-LLM-Revised/scripts/TimeLLM_ETTh1.sh) |
| swiss-river-benchmark | [benchmark/training.py](https://github.com/jajupmochi/liulian-python/blob/main/refer_projects/swiss-river-network-benchmark/swissrivernetwork/benchmark/training.py), [benchmark/ray_tune.py](https://github.com/jajupmochi/liulian-python/blob/main/refer_projects/swiss-river-network-benchmark/swissrivernetwork/benchmark/ray_tune.py), [benchmark/train_single_model.py](https://github.com/jajupmochi/liulian-python/blob/main/refer_projects/swiss-river-network-benchmark/swissrivernetwork/benchmark/train_single_model.py) |
| liulian | [runtime/trainer.py](https://github.com/jajupmochi/liulian-python/blob/main/liulian/runtime/trainer.py), [experiments/swiss_river/run.py](https://github.com/jajupmochi/liulian-python/blob/main/experiments/swiss_river/run.py) |
