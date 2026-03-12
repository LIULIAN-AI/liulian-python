# TSL vs Liulian Benchmark Comparison

This document records the systematic comparison between the
[Time-Series-Library (TSL)](https://github.com/thuml/Time-Series-Library)
reference implementation and liulian for **15 model architectures** on
standard long-term forecasting benchmarks (9 datasets, 121 experiments).

**Models**: PatchTST, DLinear, Informer, Autoformer, FEDformer, TimesNet,
Transformer, iTransformer, TimeMixer, TimeXer, Mamba, Nonstationary Transformer,
LightTS, Reformer, GPT4TS + LSTM (native).

**Status**: PatchTST and DLinear fully compared (17/18 matched). All other
models have configs generated and comparison script entries ready —
experiments pending execution.

**Goal**: Ensure liulian produces results matching TSL when configured with the
same hyperparameters and data splits.

**Methodology**:
1. Read the exact TSL script for each dataset/model pair.
2. Align the liulian YAML config to match every non-default parameter.
3. Run both, compare test MSE and MAE.
4. For large datasets (ECL 321-channel, Traffic 862-channel), compare
   per-epoch test loss after 1–2 epochs instead of full training.

**Early stopping handling**:
- TSL uses early stopping with `patience=3` by default; the comparison script
  runs TSL as-is (no modifications to its early stopping behaviour).
- Liulian also uses `patience=3` by default (matching TSL), with early stopping
  enabled unless `disable_early_stopping: true` is set in the config.
- For 5 pairs where early stopping caused >3% divergence (due to different
  random states triggering ES at different epochs), a second comparison
  was run with the `--disable-es` flag, which sets TSL `patience=9999` and
  passes `--disable_early_stopping` to liulian. These results are documented
  separately in the "ILI Gap Analysis" section below.
- The primary results in each per-dataset section use the **ES-disabled** run
  for the 5 re-run pairs (ECL PatchTST, Exchange PatchTST, ILI PatchTST,
  Exchange DLinear, ILI DLinear) and the **default ES** run for all other pairs.

---

## Summary of Root Causes (found during ETTh1 investigation)

| # | Cause | Impact | Fix |
|---|-------|--------|-----|
| 1 | **Wrong dataset class** — ETT datasets routed to `CustomCSVDataset` (70/10/20 ratio split) instead of `ETTHourDataset`/`ETTMinuteDataset` (12/4/4-month fixed borders) | MSE gap ~0.08 | `pipeline.py`: conditional routing for ETTh1/h2→`ETTHourDataset`, ETTm1/m2→`ETTMinuteDataset` |
| 2 | **Wrong default hyperparameters** — liulian `MODEL_DEFAULTS['patchtst']` uses d_model=128, d_ff=256, n_heads=16, e_layers=3, dropout=0.2 instead of TSL defaults d_model=512, d_ff=2048, n_heads=8, e_layers=2, dropout=0.1 | Large MSE gap | Override in every per-dataset YAML config |
| 3 | **Seed mismatch** — liulian default seed=2026 vs TSL hardcoded seed=2021 | Minor metric variance | Set `seed: 2021` in every config |
| 4 | **Metric aggregation** (**fixed**) — TSL computes global metrics by concatenating all predictions; liulian previously computed equal per-batch averages | ILI: 17.19% gap → 3.18% | Changed `evaluate()` to sample-weighted averaging (`np.average(vals, weights=batch_sizes)`) |

---

## TSL Default Parameters (from `run.py` argparse)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `seed` | 2021 | Hardcoded in `run.py` L10 (`fix_seed = 2021`) |
| `d_model` | 512 | |
| `d_ff` | 2048 | |
| `n_heads` | 8 | |
| `e_layers` | 2 | |
| `d_layers` | 1 | |
| `dropout` | 0.1 | |
| `batch_size` | 32 | |
| `learning_rate` | 0.0001 | |
| `train_epochs` | 10 | |
| `patience` | 3 | Early stopping |
| `lradj` | type1 | Halve LR each epoch after first |
| `seq_len` | 96 | |
| `label_len` | 48 | |
| `pred_len` | 96 | |
| `individual` | False | DLinear: shared vs per-channel linear |
| `moving_avg` | 25 | DLinear kernel size |
| `factor` | 1 | ProbAttention factor (unused by PatchTST) |

---

## PatchTST Comparisons

### 1. ETTh1 — PatchTST  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh` | `--e_layers 1 --n_heads 2 --factor 3` |
| Dataset class | Fixed | `ETTHourDataset` (12/4/4 month borders) |
| Hyperparams | Fixed | d_model=512, d_ff=2048, n_heads=2, e_layers=1, dropout=0.1 |
| Seed | Fixed | Changed 2026 → 2021 |
| Training | Matched | epochs=10, batch=32, lr=0.0001, patience=3, lradj=type1 |
| Config file | `experiments/etth1/patchtst_config.yaml` | |

**Changes made**:
- `pipeline.py`: Route ETTh1 → `ETTHourDataset` instead of `CustomCSVDataset`
- Config: d_model 128→512, d_ff 256→2048, n_heads 16→2, e_layers 3→1, dropout 0.2→0.1, seed 2026→2021

**Verified result** (full training, TSL 10ep / LL 10ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.3792 | 0.3996 |
| Liulian | 0.3797 | 0.3976 |
| Diff | 0.0005 (0.14%) | 0.0020 |

Training time: TSL 53.6s / Liulian 52.6s
---

### 2. ETTh2 — PatchTST  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/ETT_script/PatchTST_ETTh2.sh` | `--e_layers 3 --n_heads 4 --factor 3` (pred_len=96) |
| Dataset class | Fixed | `ETTHourDataset` (12/4/4 month borders) |
| Hyperparams | Fixed | d_model=512, d_ff=2048, n_heads=4, e_layers=3, dropout=0.1 |
| Seed | Set | 2021 |
| Training | Matched | epochs=10, batch=32, lr=0.0001, patience=3, lradj=type1 |
| Config file | `experiments/etth2/patchtst_config.yaml` | |

**Changes made**:
- `pipeline.py`: Route ETTh2 → `ETTHourDataset`
- Config: hyperparams aligned, seed=2021

**Verified result** (full training, TSL 4ep / LL 7ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.2913 | 0.3464 |
| Liulian | 0.2971 | 0.3463 |
| Diff | 0.0058 (2.00%) | 0.0001 |

Training time: TSL 54.0s / Liulian 86.9s
---

### 3. ETTm1 — PatchTST  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/ETT_script/PatchTST_ETTm1.sh` | `--e_layers 1 --n_heads 2 --factor 3` |
| Dataset class | Fixed | `ETTMinuteDataset` (12/4/4 month borders) |
| Hyperparams | Fixed | d_model=512, d_ff=2048, n_heads=2, e_layers=1, dropout=0.1 |
| Seed | Set | 2021 |
| Training | Matched | epochs=10, batch=32, lr=0.0001, patience=3, lradj=type1 |
| Config file | `experiments/ettm1/patchtst_config.yaml` | |

**Changes made**:
- `pipeline.py`: Route ETTm1 → `ETTMinuteDataset`
- Config: hyperparams aligned, seed=2021

**Verified result** (full training, TSL 10ep / LL 10ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.3231 | 0.3646 |
| Liulian | 0.3236 | 0.3647 |
| Diff | 0.0005 (0.15%) | 0.0000 |

Training time: TSL 182.8s / Liulian 186.1s
---

### 4. ETTm2 — PatchTST  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/ETT_script/PatchTST_ETTm2.sh` | `--e_layers 3 --n_heads 16 --factor 3` |
| Dataset class | Fixed | `ETTMinuteDataset` (12/4/4 month borders) |
| Hyperparams | Fixed | d_model=512, d_ff=2048, n_heads=16, e_layers=3, dropout=0.1 |
| Seed | Set | 2021 |
| Training | Matched | epochs=10, batch=32, lr=0.0001, patience=3, lradj=type1 |
| Config file | `experiments/ettm2/patchtst_config.yaml` | |

**Changes made**:
- `pipeline.py`: Route ETTm2 → `ETTMinuteDataset`
- Config: hyperparams aligned, seed=2021

**Verified result** (full training, TSL 4ep / LL 5ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.1834 | 0.2684 |
| Liulian | 0.1811 | 0.2645 |
| Diff | 0.0023 (1.25%) | 0.0039 |

Training time: TSL 227.7s / Liulian 285.3s
---

### 5. Weather — PatchTST  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/Weather_script/PatchTST.sh` | `--e_layers 2 --n_heads 4 --train_epochs 3 --data custom` (21 channels) |
| Dataset class | OK | Both use `Dataset_Custom` / `CustomCSVDataset` with 70/10/20 split |
| Hyperparams | Fixed | d_model=512, d_ff=2048, n_heads=4, e_layers=2, dropout=0.1 |
| Seed | Set | 2021 |
| Training | Matched | epochs=3, batch=32, lr=0.0001, patience=3, lradj=type1 |
| Config file | `experiments/weather/patchtst_config.yaml` | |

**Changes made**:
- Config: hyperparams aligned, seed=2021, epochs reduced to 3

**Verified result** (full training, TSL 3ep / LL 3ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.1761 | 0.2197 |
| Liulian | 0.1761 | 0.2170 |
| Diff | 0.0000 (0.01%) | 0.0027 |

Training time: TSL 258.8s / Liulian 262.3s
---

### 6. ECL (Electricity) — PatchTST  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/ECL_script/PatchTST.sh` | `--e_layers 2 --batch_size 16 --data custom` (321 channels) |
| Dataset class | OK | Both use `Dataset_Custom` / `CustomCSVDataset` with 70/10/20 split |
| Hyperparams | Fixed | d_model=512, d_ff=2048, n_heads=8, e_layers=2, dropout=0.1 |
| Seed | Set | 2021 |
| Training | Matched | epochs=10, batch=16, lr=0.0001, patience=3, lradj=type1 |
| Config file | `experiments/electricity/patchtst_config.yaml` | |

**Changes made**:
- Config: hyperparams aligned, batch_size=16, seed=2021

**Note**: Large dataset (321 channels). Comparison script limits to 2 epochs and compares per-epoch test loss.

**Verified result** (2-epoch comparison):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.1908 | 0.2848 |
| Liulian | 0.1973 | 0.2938 |
| Diff | 0.0065 (3.39%) | 0.0090 |

Training time: TSL 1587.7s / Liulian 1687.5s

---

### 7. Traffic — PatchTST  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/Traffic_script/PatchTST.sh` | `--e_layers 2 --d_ff 512 --batch_size 4 --data custom` (862 channels) |
| Dataset class | OK | Both use `Dataset_Custom` / `CustomCSVDataset` with 70/10/20 split |
| Hyperparams | Fixed | d_model=512, **d_ff=512**, n_heads=8, e_layers=2, dropout=0.1 |
| Seed | Set | 2021 |
| Training | Matched | epochs=10, batch=4, lr=0.0001, patience=3, lradj=type1 |
| Config file | `experiments/traffic/patchtst_config.yaml` | |

**Changes made**:
- Config: d_ff set to 512 (not default 2048), batch_size=4, seed=2021

**Note**: Largest dataset (862 channels). Comparison script limits to 2 epochs and compares per-epoch test loss.

**Verified result** (2-epoch comparison):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.4942 | 0.3440 |
| Liulian | 0.5000 | 0.3379 |
| Diff | 0.0058 (1.17%) | 0.0061 |

Training time: TSL 1225.3s / Liulian 1264.4s

---

### 8. Exchange Rate — PatchTST  ❌ checked but not matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/Exchange_script/PatchTST.sh` | `--e_layers 2 --data custom` (8 channels) |
| Dataset class | OK | Both use `Dataset_Custom` / `CustomCSVDataset` with 70/10/20 split |
| Hyperparams | Fixed | d_model=512, d_ff=2048, n_heads=8, e_layers=2, dropout=0.1 |
| Seed | Set | 2021 |
| Training | Matched | epochs=10, batch=32, lr=0.0001, patience=3, lradj=type1 |
| Config file | `experiments/exchange_rate/patchtst_config.yaml` | |

**Changes made**:
- Config: hyperparams aligned, seed=2021

**Verified result** (full training, TSL 10ep / LL 10ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.0961 | 0.2147 |
| Liulian | 0.0868 | 0.2046 |
| Diff | 0.0092 (9.62%) | 0.0100 |

Training time: TSL 73.9s / Liulian 66.2s

---

### 9. ILI (Illness) — PatchTST  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/ILI_script/PatchTST.sh` | `--seq_len 36 --pred_len 24 --label_len 18 --d_model 1024 --e_layers 4 --n_heads 4` (7 channels) |
| Dataset class | OK | Both use `Dataset_Custom` / `CustomCSVDataset` with 70/10/20 split |
| Hyperparams | Fixed | **d_model=1024**, d_ff=2048, **n_heads=4**, **e_layers=4**, dropout=0.1 |
| Sequence lengths | Fixed | **seq_len=36**, **pred_len=24**, **label_len=18** |
| Seed | Set | 2021 |
| Training | Matched | epochs=10, batch=32, lr=0.0001, patience=3, lradj=type1 |
| Config file | `experiments/illness/patchtst_config.yaml` | |

**Changes made**:
- Config: d_model=1024, e_layers=4, n_heads=4, seq_len=36, pred_len=24, seed=2021

**Verified result** (full training, TSL 10ep / LL 10ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 2.2176 | 0.8806 |
| Liulian | 2.1708 | 0.8884 |
| Diff | 0.0468 (2.11%) | 0.0078 |

Training time: TSL 25.0s / Liulian 18.4s

---

## DLinear Comparisons

### Architecture note

DLinear uses only `individual`, `moving_avg`, `enc_in`, `seq_len`, `pred_len`.
Parameters like `d_model`, `d_ff`, `n_heads`, `e_layers` are **irrelevant**
for DLinear (it uses two simple linear layers). Only **training parameters**
(seed, epochs, batch_size, lr, patience) affect results.

### 10. ETTh1 — DLinear  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh` | All defaults (enc_in=7, --data ETTh1) |
| Dataset class | Fixed | `ETTHourDataset` |
| Training params | Matched | seed=2021, epochs=10, batch=32, lr=0.0001, patience=3 |
| Config file | `experiments/etth1/dlinear_config.yaml` | |

**Verified result** (full training, TSL 9ep / LL 10ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.3962 | 0.4108 |
| Liulian | 0.3961 | 0.4103 |
| Diff | 0.0000 (0.01%) | 0.0006 |

Training time: TSL 19.6s / Liulian 13.2s
---

### 11. ETTh2 — DLinear  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | **None** — using TSL defaults | |
| Dataset class | Fixed | `ETTHourDataset` |
| Training params | Fixed → TSL defaults | seed 2026→2021, epochs 50→10, lr 0.001→0.0001, patience 5→3 |
| Config file | `experiments/etth2/dlinear_config.yaml` | |

**Changes made**:
- seed: 2026 → 2021
- train_epochs: 50 → 10
- learning_rate: 0.001 → 0.0001
- patience: 5 → 3

**Verified result** (full training, TSL 5ep / LL 8ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.3414 | 0.3953 |
| Liulian | 0.3467 | 0.4000 |
| Diff | 0.0053 (1.54%) | 0.0047 |

Training time: TSL 13.7s / Liulian 15.0s

---

### 12. ETTm1 — DLinear  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | **None** — using TSL defaults | |
| Dataset class | Fixed | `ETTMinuteDataset` |
| Training params | Fixed → TSL defaults | seed 2026→2021, epochs 50→10, lr 0.001→0.0001, patience 5→3 |
| Config file | `experiments/ettm1/dlinear_config.yaml` | |

**Changes made**: Same as ETTh2 DLinear above.

**Verified result** (full training, TSL 6ep / LL 7ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.3459 | 0.3737 |
| Liulian | 0.3445 | 0.3718 |
| Diff | 0.0014 (0.41%) | 0.0019 |

Training time: TSL 33.7s / Liulian 38.7s

---

### 13. ETTm2 — DLinear  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | **None** — using TSL defaults | |
| Dataset class | Fixed | `ETTMinuteDataset` |
| Training params | Fixed → TSL defaults | seed 2026→2021, epochs 50→10, lr 0.001→0.0001, patience 5→3 |
| Config file | `experiments/ettm2/dlinear_config.yaml` | |

**Changes made**: Same as ETTh2 DLinear above.

**Verified result** (full training, TSL 7ep / LL 6ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.1934 | 0.2928 |
| Liulian | 0.1921 | 0.2913 |
| Diff | 0.0013 (0.66%) | 0.0015 |

Training time: TSL 37.4s / Liulian 30.5s

---

### 14. Weather — DLinear  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | **None** — using TSL defaults | |
| Dataset class | OK | `CustomCSVDataset` with 70/10/20 split |
| Training params | Fixed → TSL defaults | seed 2026→2021, epochs 50→10, lr 0.001→0.0001, patience 5→3 |
| Config file | `experiments/weather/dlinear_config.yaml` | |

**Changes made**: Same as ETTh2 DLinear above.

**Verified result** (full training, TSL 7ep / LL 9ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.1962 | 0.2561 |
| Liulian | 0.1954 | 0.2551 |
| Diff | 0.0008 (0.41%) | 0.0010 |

Training time: TSL 52.0s / Liulian 43.9s

---

### 15. ECL (Electricity) — DLinear  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | `scripts/long_term_forecast/ECL_script/DLinear.sh` | All defaults (enc_in=321, --data custom) |
| Dataset class | OK | `CustomCSVDataset` with 70/10/20 split |
| Training params | Matched | seed=2021, epochs=10, batch=32, lr=0.0001, patience=3 |
| Config file | `experiments/electricity/dlinear_config.yaml` | |

**Note**: Large dataset (321 channels). Comparison script limits to 2 epochs.

**Verified result** (2-epoch comparison):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.2229 | 0.3168 |
| Liulian | 0.2231 | 0.3170 |
| Diff | 0.0003 (0.11%) | 0.0003 |

Training time: TSL 39.1s / Liulian 97.6s

---

### 16. Traffic — DLinear  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | **None** — using TSL defaults | |
| Dataset class | OK | `CustomCSVDataset` with 70/10/20 split |
| Training params | Fixed → TSL defaults | seed 2026→2021, epochs 30→10, batch 16→32, patience 10→3 |
| identifier_mode | Fixed | `embedding` → `none` (TSL has no entity embeddings) |
| Config file | `experiments/traffic/dlinear_config.yaml` | |

**Changes made**:
- seed: 2026 → 2021
- train_epochs: 30 → 10
- batch_size: 16 → 32
- patience: 10 → 3
- identifier_mode: embedding → none (critical — this changes model architecture)

**Note**: Largest dataset (862 channels). Comparison script limits to 2 epochs.

**Verified result** (2-epoch comparison):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.7290 | 0.4476 |
| Liulian | 0.7275 | 0.4472 |
| Diff | 0.0014 (0.19%) | 0.0004 |

Training time: TSL 54.6s / Liulian 96.8s

---

### 17. Exchange Rate — DLinear  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | **None** — using TSL defaults | |
| Dataset class | OK | `CustomCSVDataset` with 70/10/20 split |
| Training params | Fixed → TSL defaults | seed 2026→2021, epochs 30→10, patience 10→3 |
| identifier_mode | Fixed | `embedding` → `none` (TSL has no entity embeddings) |
| Config file | `experiments/exchange_rate/dlinear_config.yaml` | |

**Changes made**:
- seed: 2026 → 2021
- train_epochs: 30 → 10
- patience: 10 → 3
- identifier_mode: embedding → none

**Verified result** (full training, TSL 10ep / LL 10ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 0.0944 | 0.2274 |
| Liulian | 0.0940 | 0.2269 |
| Diff | 0.0003 (0.35%) | 0.0006 |

Training time: TSL 18.6s / Liulian 11.1s

---

### 18. ILI (Illness) — DLinear  ✅ checked and matched

| Aspect | Checked | Details |
|--------|---------|---------|
| TSL script | **None** — using TSL defaults | |
| Dataset class | OK | `CustomCSVDataset` with 70/10/20 split |
| Training params | Fixed → TSL defaults | seed 2026→2021, epochs 50→10, lr 0.001→0.0001, patience 5→3 |
| Sequence lengths | OK | seq_len=36, pred_len=24 (matches ILI convention) |
| Config file | `experiments/illness/dlinear_config.yaml` | |

**Changes made**: Same as ETTh2 DLinear above.

**Verified result** (full training, TSL 10ep / LL 10ep):
| Source | MSE | MAE |
|--------|-----|-----|
| TSL | 4.7815 | 1.6558 |
| Liulian | 4.7805 | 1.6559 |
| Diff | 0.0010 (0.02%) | 0.0001 |

Training time: TSL 13.1s / Liulian 5.6s

---

## Informer Comparisons

**Paper**: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021 Best Paper, Zhou et al.)

TSL scripts use: d_model=512, d_ff=2048, n_heads=8, e_layers=2, d_layers=1,
factor=3, distil=True. Learning rate: 0.0001. Seed: 2021.

| # | Dataset | Status | Config |
|---|---------|--------|--------|
| 19 | ETTh1 — Informer | ⏳ pending | `experiments/etth1/informer_config.yaml` |
| 20 | ETTh2 — Informer | ⏳ pending | `experiments/etth2/informer_config.yaml` |
| 21 | ETTm1 — Informer | ⏳ pending | `experiments/ettm1/informer_config.yaml` |
| 22 | ETTm2 — Informer | ⏳ pending | `experiments/ettm2/informer_config.yaml` |
| 23 | Weather — Informer | ⏳ pending | `experiments/weather/informer_config.yaml` |
| 24 | ECL — Informer | ⏳ pending | `experiments/electricity/informer_config.yaml` |
| 25 | Traffic — Informer | ⏳ pending | `experiments/traffic/informer_config.yaml` |
| 26 | Exchange — Informer | ⏳ pending | `experiments/exchange_rate/informer_config.yaml` |
| 27 | ILI — Informer | ⏳ pending | `experiments/illness/informer_config.yaml` |

---

## Autoformer Comparisons

**Paper**: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" (NeurIPS 2021, Wu et al.)

TSL scripts use: d_model=512, d_ff=2048, n_heads=8, e_layers=2, d_layers=1,
factor=3, moving_avg=25. Learning rate: 0.0001. Seed: 2021.

| # | Dataset | Status | Config |
|---|---------|--------|--------|
| 28 | ETTh1 — Autoformer | ⏳ pending | `experiments/etth1/autoformer_config.yaml` |
| 29 | ETTh2 — Autoformer | ⏳ pending | `experiments/etth2/autoformer_config.yaml` |
| 30 | ETTm1 — Autoformer | ⏳ pending | `experiments/ettm1/autoformer_config.yaml` |
| 31 | ETTm2 — Autoformer | ⏳ pending | `experiments/ettm2/autoformer_config.yaml` |
| 32 | Weather — Autoformer | ⏳ pending | `experiments/weather/autoformer_config.yaml` |
| 33 | ECL — Autoformer | ⏳ pending | `experiments/electricity/autoformer_config.yaml` |
| 34 | Traffic — Autoformer | ⏳ pending | `experiments/traffic/autoformer_config.yaml` |
| 35 | Exchange — Autoformer | ⏳ pending | `experiments/exchange_rate/autoformer_config.yaml` |
| 36 | ILI — Autoformer | ⏳ pending | `experiments/illness/autoformer_config.yaml` |

---

## FEDformer Comparisons

**Paper**: "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting" (ICML 2022, Zhou et al.)

TSL scripts use: d_model=512, d_ff=2048, n_heads=8, e_layers=2, d_layers=1,
factor=3, moving_avg=25. Learning rate: 0.0001. Seed: 2021.

| # | Dataset | Status | Config |
|---|---------|--------|--------|
| 37 | ETTh1 — FEDformer | ⏳ pending | `experiments/etth1/fedformer_config.yaml` |
| 38 | ETTh2 — FEDformer | ⏳ pending | `experiments/etth2/fedformer_config.yaml` |
| 39 | ETTm1 — FEDformer | ⏳ pending | `experiments/ettm1/fedformer_config.yaml` |
| 40 | ETTm2 — FEDformer | ⏳ pending | `experiments/ettm2/fedformer_config.yaml` |
| 41 | Weather — FEDformer | ⏳ pending | `experiments/weather/fedformer_config.yaml` |
| 42 | ECL — FEDformer | ⏳ pending | `experiments/electricity/fedformer_config.yaml` |
| 43 | Traffic — FEDformer | ⏳ pending | `experiments/traffic/fedformer_config.yaml` |
| 44 | Exchange — FEDformer | ⏳ pending | `experiments/exchange_rate/fedformer_config.yaml` |
| 45 | ILI — FEDformer | ⏳ pending | `experiments/illness/fedformer_config.yaml` |

---

## TimesNet Comparisons

**Paper**: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (ICLR 2023, Wu et al.)

TSL scripts use MUCH smaller d_model/d_ff than other models. Per-dataset
overrides: d_model=16–64, d_ff=32–128, e_layers=2, top_k=5, num_kernels=6.

| # | Dataset | Status | Config | d_model | d_ff |
|---|---------|--------|--------|---------|------|
| 46 | ETTh1 — TimesNet | ⏳ pending | `experiments/etth1/timesnet_config.yaml` | 16 | 32 |
| 47 | ETTh2 — TimesNet | ⏳ pending | `experiments/etth2/timesnet_config.yaml` | 32 | 64 |
| 48 | ETTm1 — TimesNet | ⏳ pending | `experiments/ettm1/timesnet_config.yaml` | 32 | 64 |
| 49 | ETTm2 — TimesNet | ⏳ pending | `experiments/ettm2/timesnet_config.yaml` | 32 | 64 |
| 50 | Weather — TimesNet | ⏳ pending | `experiments/weather/timesnet_config.yaml` | 16 | 32 |
| 51 | ECL — TimesNet | ⏳ pending | `experiments/electricity/timesnet_config.yaml` | 32 | 32 |
| 52 | Traffic — TimesNet | ⏳ pending | `experiments/traffic/timesnet_config.yaml` | 32 | 32 |
| 53 | Exchange — TimesNet | ⏳ pending | `experiments/exchange_rate/timesnet_config.yaml` | 32 | 64 |
| 54 | ILI — TimesNet | ⏳ pending | `experiments/illness/timesnet_config.yaml` | 64 | 128 |

---

## Transformer Comparisons

**Paper**: "Attention Is All You Need" (NeurIPS 2017, Vaswani et al.)

TSL Transformer uses argparse defaults throughout: d_model=512, d_ff=2048,
n_heads=8, e_layers=2, d_layers=1. Learning rate: 0.0001.

| # | Dataset | Status | Config |
|---|---------|--------|--------|
| 55 | ETTh1 — Transformer | ⏳ pending | `experiments/etth1/transformer_config.yaml` |
| 56 | ETTh2 — Transformer | ⏳ pending | `experiments/etth2/transformer_config.yaml` |
| 57 | ETTm1 — Transformer | ⏳ pending | `experiments/ettm1/transformer_config.yaml` |
| 58 | ETTm2 — Transformer | ⏳ pending | `experiments/ettm2/transformer_config.yaml` |
| 59 | Weather — Transformer | ⏳ pending | `experiments/weather/transformer_config.yaml` |
| 60 | ECL — Transformer | ⏳ pending | `experiments/electricity/transformer_config.yaml` |
| 61 | Traffic — Transformer | ⏳ pending | `experiments/traffic/transformer_config.yaml` |
| 62 | Exchange — Transformer | ⏳ pending | `experiments/exchange_rate/transformer_config.yaml` |
| 63 | ILI — Transformer | ⏳ pending | `experiments/illness/transformer_config.yaml` |

**Note**: TSL ECL Transformer script uses `--features S` (univariate), but
liulian config uses `features: M` (multivariate) for consistency across the
benchmark. This may cause a legitimate MSE difference.

---

## iTransformer Comparisons

**Paper**: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024 Spotlight, Liu et al.)

TSL scripts use: d_model=512, d_ff=512 (smaller than usual), e_layers=3–4.
iTransformer is encoder-only (no d_layers/decoder). TSL scripts only cover
ETTh2, ECL, Weather, Traffic; other datasets use argparse defaults.

| # | Dataset | Status | Config | Notes |
|---|---------|--------|--------|-------|
| 64 | ETTh1 — iTransformer | ⏳ pending | `experiments/etth1/itransformer_config.yaml` | Defaults |
| 65 | ETTh2 — iTransformer | ⏳ pending | `experiments/etth2/itransformer_config.yaml` | TSL script |
| 66 | ETTm1 — iTransformer | ⏳ pending | `experiments/ettm1/itransformer_config.yaml` | Defaults |
| 67 | ETTm2 — iTransformer | ⏳ pending | `experiments/ettm2/itransformer_config.yaml` | Defaults |
| 68 | Weather — iTransformer | ⏳ pending | `experiments/weather/itransformer_config.yaml` | TSL script |
| 69 | ECL — iTransformer | ⏳ pending | `experiments/electricity/itransformer_config.yaml` | TSL script |
| 70 | Traffic — iTransformer | ⏳ pending | `experiments/traffic/itransformer_config.yaml` | TSL script |
| 71 | Exchange — iTransformer | ⏳ pending | `experiments/exchange_rate/itransformer_config.yaml` | Defaults |
| 72 | ILI — iTransformer | ⏳ pending | `experiments/illness/itransformer_config.yaml` | Defaults |

---

## TimeMixer Comparisons

**Paper**: "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting" (ICLR 2024, Wang et al.)

TSL scripts use very small d_model/d_ff (like TimesNet). Unique params:
label_len=0, down_sampling_layers=3, down_sampling_window=2,
down_sampling_method=avg. Learning rate varies (Weather: 0.01).
TSL scripts available for 7 datasets (no Exchange, no ILI).

| # | Dataset | Status | Config | Notes |
|---|---------|--------|--------|-------|
| 73 | ETTh1 — TimeMixer | ⏳ pending | `experiments/etth1/timemixer_config.yaml` | d_model=16, lr=0.01 |
| 74 | ETTh2 — TimeMixer | ⏳ pending | `experiments/etth2/timemixer_config.yaml` | d_model=16, lr=0.01 |
| 75 | ETTm1 — TimeMixer | ⏳ pending | `experiments/ettm1/timemixer_config.yaml` | d_model=16, lr=0.01 |
| 76 | ETTm2 — TimeMixer | ⏳ pending | `experiments/ettm2/timemixer_config.yaml` | d_model=16, lr=0.01 |
| 77 | Weather — TimeMixer | ⏳ pending | `experiments/weather/timemixer_config.yaml` | d_model=16, lr=0.01 |
| 78 | ECL — TimeMixer | ⏳ pending | `experiments/electricity/timemixer_config.yaml` | d_model=32, lr=0.001 |
| 79 | Traffic — TimeMixer | ⏳ pending | `experiments/traffic/timemixer_config.yaml` | d_model=32, lr=0.001 |

---

## TimeXer Comparisons

**Paper**: "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables" (NeurIPS 2024, Wang et al.)

TSL scripts vary d_model/d_ff/e_layers per dataset. patch_len=16 (argparse
default). TSL scripts available for 7 datasets (no Exchange, no ILI).

| # | Dataset | Status | Config | d_model | d_ff | e_layers |
|---|---------|--------|--------|---------|------|----------|
| 80 | ETTh1 — TimeXer | ⏳ pending | `experiments/etth1/timexer_config.yaml` | 256 | 512 | 2 |
| 81 | ETTh2 — TimeXer | ⏳ pending | `experiments/etth2/timexer_config.yaml` | 128 | 256 | 2 |
| 82 | ETTm1 — TimeXer | ⏳ pending | `experiments/ettm1/timexer_config.yaml` | 128 | 256 | 1 |
| 83 | ETTm2 — TimeXer | ⏳ pending | `experiments/ettm2/timexer_config.yaml` | 128 | 256 | 1 |
| 84 | Weather — TimeXer | ⏳ pending | `experiments/weather/timexer_config.yaml` | 256 | 512 | 3 |
| 85 | ECL — TimeXer | ⏳ pending | `experiments/electricity/timexer_config.yaml` | 256 | 512 | 2 |
| 86 | Traffic — TimeXer | ⏳ pending | `experiments/traffic/timexer_config.yaml` | 512 | 512 | 3 |

---

## Mamba Comparisons

**Paper**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2024)

TSL scripts use: d_model=128, d_ff=16 (!), e_layers=2, expand=2, d_conv=4.
d_ff is much smaller than the argparse default because the SSM scan replaces
the feed-forward role. TSL scripts available for 8 datasets (no ILI).

| # | Dataset | Status | Config |
|---|---------|--------|--------|
| 87 | ETTh1 — Mamba | ⏳ pending | `experiments/etth1/mamba_config.yaml` |
| 88 | ETTh2 — Mamba | ⏳ pending | `experiments/etth2/mamba_config.yaml` |
| 89 | ETTm1 — Mamba | ⏳ pending | `experiments/ettm1/mamba_config.yaml` |
| 90 | ETTm2 — Mamba | ⏳ pending | `experiments/ettm2/mamba_config.yaml` |
| 91 | Weather — Mamba | ⏳ pending | `experiments/weather/mamba_config.yaml` |
| 92 | ECL — Mamba | ⏳ pending | `experiments/electricity/mamba_config.yaml` |
| 93 | Traffic — Mamba | ⏳ pending | `experiments/traffic/mamba_config.yaml` |
| 94 | Exchange — Mamba | ⏳ pending | `experiments/exchange_rate/mamba_config.yaml` |

---

## LSTM Comparisons (liulian-native, no TSL reference)

LSTM is a liulian-native model (not from TSL). No TSL reference scripts exist.
Included for completeness — configs use seed=2026, patience=10, lradj=none.

| # | Dataset | Status | Config |
|---|---------|--------|--------|
| 95 | ETTh1 — LSTM | ⏳ pending | `experiments/etth1/lstm_config.yaml` |
| 96 | ETTh2 — LSTM | ⏳ pending | `experiments/etth2/lstm_config.yaml` |
| 97 | ETTm1 — LSTM | ⏳ pending | `experiments/ettm1/lstm_config.yaml` |
| 98 | ETTm2 — LSTM | ⏳ pending | `experiments/ettm2/lstm_config.yaml` |
| 99 | Weather — LSTM | ⏳ pending | `experiments/weather/lstm_config.yaml` |
| 100 | ECL — LSTM | — | `experiments/electricity/lstm_config.yaml` (existing) |
| 101 | Traffic — LSTM | — | `experiments/traffic/lstm_config.yaml` (existing) |
| 102 | Exchange — LSTM | — | `experiments/exchange_rate/lstm_config.yaml` (existing) |
| 103 | ILI — LSTM | ⏳ pending | `experiments/illness/lstm_config.yaml` |

---

## Datasets Without TSL Scripts

The following datasets have **no TSL long-term-forecast scripts** for either
PatchTST or DLinear. They are excluded from this comparison:

| Dataset | Reason |
|---------|--------|
| Solar-Energy | No TSL script; no liulian config |
| PEMS03/04/07/08 | No TSL script (TSL only has short-term-forecast PEMS scripts) |
| M4 | Short-term forecast task; different from long-term benchmark |

---

## Code Changes Summary

### `liulian/pipeline.py`

ETT dataset routing fix (applied earlier):
```python
# Before: all CSV datasets routed to CustomCSVDataset
# After: ETTh1/ETTh2 → ETTHourDataset, ETTm1/ETTm2 → ETTMinuteDataset
```
This ensures the 12/4/4-month fixed borders are used instead of 70/10/20 ratio
splits, matching TSL's `Dataset_ETT_hour` and `Dataset_ETT_minute`.

### Config YAML files updated

**PatchTST** (all 9 datasets):
All configs set to match the exact parameters from the corresponding TSL script,
overriding liulian's `MODEL_DEFAULTS['patchtst']` which uses wrong values
(d_model=128, d_ff=256, n_heads=16, e_layers=3, dropout=0.2).

**DLinear** (all 9 datasets):
- ETTh1, ECL: configs already matched TSL scripts
- ETTh2, ETTm1, ETTm2, Weather, ILI: seed→2021, epochs→10, lr→0.0001, patience→3
- Traffic: additionally batch_size→32, identifier_mode→none
- Exchange: additionally identifier_mode→none

---

## Known Minor Differences

1. **Metric aggregation** (**fixed**): TSL concatenates all test predictions
   then computes global MSE/MAE. Liulian previously averaged per-batch metrics
   with equal weight, which inflated metrics when the last batch was smaller
   (especially on ILI with 170 samples → 5×32 + 1×10, last batch got 16.7%
   weight instead of 5.9%). **Fix**: Changed `evaluate()` in `trainer.py` to
   use sample-weighted averaging (`np.average(vals, weights=batch_sizes)`),
   which is mathematically equivalent to TSL's global metric. This fixed
   ILI PatchTST from 17.19% gap → 3.18% gap.

2. **Early stopping divergence**: Both TSL and liulian use `patience=3`
   early stopping by default, but different random states lead to different
   convergence trajectories, causing ES to trigger at different epochs
   (e.g., TSL DLinear at epoch 4 vs liulian at epoch 10). For affected pairs,
   a second run with ES disabled (`--disable-es` flag: TSL gets
   `--patience 9999`, liulian gets `--disable_early_stopping`) was performed
   to isolate this effect. Results: ILI DLinear 5.54% → 0.02%;
   Exchange DLinear 4.15% → 0.35%. The `disable_early_stopping` config key
   is available in liulian for users who want to disable ES.

3. **`factor` parameter**: TSL scripts set `--factor 3` (ProbAttention sparsity).
   Liulian configs use `factor: 1`. This parameter is **unused by PatchTST**
   (which uses standard full attention) and **unused by DLinear**. No impact.

4. **`label_len`**: Some liulian PatchTST configs have `label_len: 0` vs TSL
   default 48. PatchTST is encoder-only and does not use `label_len`. No impact.

---

## ILI Gap Analysis

The ILI (Illness) dataset is the smallest benchmark (966 rows, 7 channels,
weekly frequency). Its small size amplifies minor implementation differences.

### Root Causes Identified

| # | Cause | Impact | Fix |
|---|-------|--------|-----|
| 1 | **Metric aggregation bias** — Per-batch MSE averaging gave equal weight to each batch; last batch (10/170 samples) was overweighted by 2.8× | ILI PatchTST: 17.19% gap → 3.18%; ILI DLinear: 10.64% → 5.54% | Changed `evaluate()` to sample-weighted averaging in `trainer.py` |
| 2 | **Early stopping divergence** — Different random states cause different convergence and different ES-triggered epochs (both TSL and liulian use `patience=3` by default) | TSL DLinear early-stopped at epoch 4, liulian trained 10; Exchange DLinear: TSL 5 ep vs LL 9 ep | Comparison re-run with ES disabled (`--disable-es`): TSL `patience=9999`, liulian `--disable_early_stopping` |
| 3 | **Random state divergence** — Liulian builds dataset before model (consuming RNG), so `torch.manual_seed(2021)` produces different initial weights | Different optimization trajectories | Not fixable without major refactoring of initialization order |
| 4 | **`num_workers` difference** — TSL uses `num_workers=10`, liulian uses `num_workers=0` | Shuffle order within epochs may differ slightly | Subsumed by cause #3 |

### ILI PatchTST — Progressive Fix Results

| Metric | Original | After Metric Fix | After ES Disabled |
|--------|----------|-----------------|-------------------|
| TSL MSE | 2.2421 | 2.2421 | 2.2176 |
| Liulian MSE | 2.6276 | 2.1708 | 2.1708 |
| Gap | 17.19% | 3.18% | **2.11%** |
| TSL epochs | 9 | 9 | 10 |
| LL epochs | 10 | 10 | 10 |
| Status | ❌ not matched | ✅ matched | ✅ matched |

### ILI DLinear — Progressive Fix Results

| Metric | Original | After Metric Fix | After ES Disabled |
|--------|----------|-----------------|-------------------|
| TSL MSE | 5.0608 | 5.0608 | 4.7815 |
| Liulian MSE | 4.5225 | 4.7805 | 4.7805 |
| Gap | 10.64% | 5.54% | **0.02%** |
| TSL epochs | 4 | 4 | 10 |
| LL epochs | 10 | 10 | 10 |
| Status | ❌ not matched | ❌ not matched | ✅ matched |

### Exchange Rate — Before and After ES Disabled

| Pair | ES On (Diff%) | ES Off (Diff%) | Notes |
|------|-------------|---------------|-------|
| Exchange PatchTST | 8.79% | 9.62% | TSL: 4→10 ep, LL: 5→10 ep; gap widened — marked as **not matched** |
| Exchange DLinear | 4.15% | **0.35%** | TSL: 5→10 ep, LL: 9→10 ep; dramatic improvement |

**Conclusion**: Disabling early stopping isolated the ES-divergence effect.
The ILI DLinear gap dropped from 5.54% to 0.02%, and
Exchange DLinear from 4.15% to 0.35%. Exchange PatchTST remained at 9.62%
(marked as not matched). **17 of 18 pairs match; 1 (Exchange PatchTST) does
not**, likely due to pure random-state divergence on this small dataset
(8 channels, only 7588 rows).

---

## Checklist

> **Note on early stopping**: Rows 6, 8, 9, 17, 18 (ECL PatchTST, Exchange PatchTST, ILI PatchTST,
> Exchange DLinear, ILI DLinear) were re-run with early stopping disabled on both sides
> (`--disable-es`: TSL `patience=9999`, liulian `--disable_early_stopping`).
> All other rows use default early stopping (`patience=3` on both TSL and liulian).

| # | Dataset | Model | Has TSL Script | Config Revised | Verified | Status | TSL ep | LL ep | TSL time | LL time |
|---|---------|-------|---------------|----------------|----------|--------|--------|-------|----------|---------|
| 1 | ETTh1 | PatchTST | ✅ | ✅ | ✅ MSE diff 0.14% | checked and matched | 10 | 10 | 54s | 53s |
| 2 | ETTh2 | PatchTST | ✅ | ✅ | ✅ MSE diff 2.00% | checked and matched | 4 | 7 | 54s | 87s |
| 3 | ETTm1 | PatchTST | ✅ | ✅ | ✅ MSE diff 0.15% | checked and matched | 10 | 10 | 183s | 186s |
| 4 | ETTm2 | PatchTST | ✅ | ✅ | ✅ MSE diff 1.25% | checked and matched | 4 | 5 | 228s | 285s |
| 5 | Weather | PatchTST | ✅ | ✅ | ✅ MSE diff 0.01% | checked and matched | 3 | 3 | 259s | 262s |
| 6 | ECL | PatchTST | ✅ | ✅ | ✅ MSE diff 3.39% | checked and matched | 2 | 2 | 1588s | 1688s |
| 7 | Traffic | PatchTST | ✅ | ✅ | ✅ MSE diff 1.17% | checked and matched | 2 | 2 | 1225s | 1264s |
| 8 | Exchange | PatchTST | ✅ | ✅ | ❌ MSE gap 9.6% | checked but not matched | 10 | 10 | 74s | 66s |
| 9 | ILI | PatchTST | ✅ | ✅ | ✅ MSE diff 2.11% | checked and matched | 10 | 10 | 25s | 18s |
| 10 | ETTh1 | DLinear | ✅ | ✅ | ✅ MSE diff 0.01% | checked and matched | 9 | 10 | 20s | 13s |
| 11 | ETTh2 | DLinear | ❌ (defaults) | ✅ | ✅ MSE diff 1.54% | checked and matched | 5 | 8 | 14s | 15s |
| 12 | ETTm1 | DLinear | ❌ (defaults) | ✅ | ✅ MSE diff 0.41% | checked and matched | 6 | 7 | 34s | 39s |
| 13 | ETTm2 | DLinear | ❌ (defaults) | ✅ | ✅ MSE diff 0.66% | checked and matched | 7 | 6 | 37s | 30s |
| 14 | Weather | DLinear | ❌ (defaults) | ✅ | ✅ MSE diff 0.41% | checked and matched | 7 | 9 | 52s | 44s |
| 15 | ECL | DLinear | ✅ | ✅ | ✅ MSE diff 0.11% | checked and matched | 2 | 2 | 39s | 98s |
| 16 | Traffic | DLinear | ❌ (defaults) | ✅ | ✅ MSE diff 0.19% | checked and matched | 2 | 2 | 55s | 97s |
| 17 | Exchange | DLinear | ❌ (defaults) | ✅ | ✅ MSE diff 0.35% | checked and matched | 10 | 10 | 19s | 11s |
| 18 | ILI | DLinear | ❌ (defaults) | ✅ | ✅ MSE diff 0.02% | checked and matched | 10 | 10 | 13s | 6s |

---

## New Models (Part 3)

The following 4 models were identified as missing and implemented in Part 3.
Comparison entries are ready; experiments pending execution.

### Nonstationary Transformer

**Paper**: [Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting](https://openreview.net/pdf?id=ucNDIDRNjjv) (NeurIPS 2022)

**Architecture**: Encoder-decoder Transformer with De-Stationary Attention (DSAttention).
Two Projector MLPs learn `tau` (scaling) and `delta` (shift) statistics from
input sequences to restore non-stationary information lost by instance
normalization. The projectors use Conv1d + stacked linear layers.

**Key hyperparameters**: `p_hidden_dims` (list of hidden dimensions for
projectors), `p_hidden_layers` (number of projector layers). Values vary
significantly per dataset (e.g., `[256,256]` for ETTh1 vs `[16,16,16,16]` for
ETTm1).

**TSL scripts**: All 9 datasets have dedicated scripts.

**Liulian adapter**: `liulian/models/torch/nonstationary_transformer.py`

| # | Dataset | Model | Has TSL Script | Config Revised | Verified | Status |
|---|---------|-------|---------------|----------------|----------|--------|
| 19 | ETTh1 | Nonstat. Transformer | ✅ | ✅ | ⏳ pending | — |
| 20 | ETTh2 | Nonstat. Transformer | ✅ | ✅ | ⏳ pending | — |
| 21 | ETTm1 | Nonstat. Transformer | ✅ | ✅ | ⏳ pending | — |
| 22 | ETTm2 | Nonstat. Transformer | ✅ | ✅ | ⏳ pending | — |
| 23 | ECL | Nonstat. Transformer | ✅ | ✅ | ⏳ pending | — |
| 24 | Weather | Nonstat. Transformer | ✅ | ✅ | ⏳ pending | — |
| 25 | Traffic | Nonstat. Transformer | ✅ | ✅ | ⏳ pending | — |
| 26 | Exchange | Nonstat. Transformer | ✅ | ✅ | ⏳ pending | — |
| 27 | ILI | Nonstat. Transformer | ✅ | ✅ | ⏳ pending | — |

### LightTS

**Paper**: [Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures](https://arxiv.org/abs/2207.01186) (arXiv 2022)

**Architecture**: Pure MLP model using Iterative Enhancement Blocks (IEBlock).
Two sampling strategies—continuous (first/last chunks) and interval
(even/odd indices)—extract temporal features. Autoregressive skip connection
preserves low-frequency information. No attention or embedding layers.

**Key hyperparameters**: Standard `enc_in`, `seq_len`, `pred_len`. The model
is self-contained and does not use `d_model` or `e_layers`.

**TSL scripts**: ETTh1 and ECL only; other datasets use TSL defaults.

**Liulian adapter**: `liulian/models/torch/lightts.py`

| # | Dataset | Model | Has TSL Script | Config Revised | Verified | Status |
|---|---------|-------|---------------|----------------|----------|--------|
| 28 | ETTh1 | LightTS | ✅ | ✅ | ⏳ pending | — |
| 29 | ETTh2 | LightTS | ❌ (defaults) | ✅ | ⏳ pending | — |
| 30 | ETTm1 | LightTS | ❌ (defaults) | ✅ | ⏳ pending | — |
| 31 | ETTm2 | LightTS | ❌ (defaults) | ✅ | ⏳ pending | — |
| 32 | ECL | LightTS | ✅ | ✅ | ⏳ pending | — |
| 33 | Weather | LightTS | ❌ (defaults) | ✅ | ⏳ pending | — |
| 34 | Traffic | LightTS | ❌ (defaults) | ✅ | ⏳ pending | — |
| 35 | Exchange | LightTS | ❌ (defaults) | ✅ | ⏳ pending | — |
| 36 | ILI | LightTS | ❌ (defaults) | ✅ | ⏳ pending | — |

### Reformer

**Paper**: [Reformer: The Efficient Transformer](https://openreview.net/forum?id=rkgNKkHtvB) (ICLR 2020)

**Architecture**: Encoder-only Transformer with Locality-Sensitive Hashing
(LSH) self-attention for $O(L \log L)$ complexity. Input is padded to
multiples of `2 * bucket_size` for the hashing scheme. For forecasting,
the encoder input is `[x_enc; x_dec_placeholder]` concatenated along the
time axis, and the output is sliced to `[-pred_len:]`.

**Key hyperparameters**: Standard transformer params (`d_model`, `e_layers`,
`n_heads`). LSH-specific: `bucket_size` (default 4), `n_hashes` (default 4).

**Dependencies**: `reformer-pytorch>=1.4.0`, `local-attention>=1.11.0`

**TSL scripts**: ETTh1 and ECL only; other datasets use TSL defaults.

**Liulian adapter**: `liulian/models/torch/reformer.py`,
`liulian/models/torch/layers/attention.py` (ReformerLayer)

| # | Dataset | Model | Has TSL Script | Config Revised | Verified | Status |
|---|---------|-------|---------------|----------------|----------|--------|
| 37 | ETTh1 | Reformer | ✅ | ✅ | ⏳ pending | — |
| 38 | ETTh2 | Reformer | ❌ (defaults) | ✅ | ⏳ pending | — |
| 39 | ETTm1 | Reformer | ❌ (defaults) | ✅ | ⏳ pending | — |
| 40 | ETTm2 | Reformer | ❌ (defaults) | ✅ | ⏳ pending | — |
| 41 | ECL | Reformer | ✅ | ✅ | ⏳ pending | — |
| 42 | Weather | Reformer | ❌ (defaults) | ✅ | ⏳ pending | — |
| 43 | Traffic | Reformer | ❌ (defaults) | ✅ | ⏳ pending | — |
| 44 | Exchange | Reformer | ❌ (defaults) | ✅ | ⏳ pending | — |
| 45 | ILI | Reformer | ❌ (defaults) | ✅ | ⏳ pending | — |

### GPT4TS (One Fits All)

**Paper**: [One Fits All: Power General Time Series Analysis by Pretrained LM](https://arxiv.org/abs/2302.11939) (NeurIPS 2023)

**Architecture**: Frozen pre-trained GPT-2 backbone with fine-tuned
LayerNorm and positional embedding. Input is patch-based: the time series
is normalized (instance norm), segmented into patches of length `patch_len`
with stride `stride`, then projected to GPT-2's hidden dimension (768).
Output projection maps back to `pred_len`.

**Key hyperparameters**: `gpt_layers` (number of GPT-2 layers to use,
default 6), `patch_len` (default 16), `stride` (default 8). The model's
`d_model` is fixed at 768 (GPT-2 base).

**Dependencies**: `transformers` (HuggingFace, for GPT-2 weights)

**TSL scripts**: None — no TSL counterpart. Implemented from paper.
No comparison experiments are included (liulian-only).

**Liulian adapter**: `liulian/models/torch/gpt4ts.py`

### Note on TS-LLM

"TS-LLM" was identified as a potentially missing model name during the gap
analysis, but it does not correspond to a single well-defined paper or TSL
implementation. It may refer to a family of LLM-based time series methods.
No adapter was created. The GPT4TS model above covers the primary
"pretrained LLM for time series" approach from the TSL ecosystem.
