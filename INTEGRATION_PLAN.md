# Liulian Integration & Benchmark Plan

**Date:** 2026-02-17 (revised)  
**Scope:** Full cross-model × cross-dataset integration of all features, comprehensive tests, documentation, and a reproducible academic benchmark.

---

## Progress Summary

### Part 1 — TSL Comparison Infrastructure (COMPLETED, commit `04d4f28`)

- 17/18 PatchTST + DLinear benchmarks matched (≤5% MSE gap)
- Fixed: metric aggregation bias, dataset class routing, hyperparameter defaults
- Created comparison script with automated TSL vs liulian execution

### Part 2 — Multi-Model Expansion (IN PROGRESS)

| Deliverable | Status | Details |
|-------------|--------|---------|
| Config generator | ✅ Done | `experiments/adapt_tsl_lib/generate_configs.py` |
| Per-dataset configs | ✅ Done | 114 configs (12 models × 9 datasets, 87 new) |
| Comparison script | ✅ Done | 94 experiments across 11 models |
| HPO search spaces | ✅ Done | All 12 models have search spaces + TimeLLM resolution fix |
| Mamba import shim | ✅ Done | `liulian/models/torch/mamba.py` re-exports from `mamba_model` |
| E2E anchor tests | ✅ Done | 48 test classes (12 models × 4 scenarios), baselines pending |
| Documentation | ✅ Done | 103 experiments documented in `docs/tsl_comparison.md` |
| Missing models | ⏳ Future | Nonstationary_Transformer, LightTS, Reformer, GPT4TS, TS-LLM |

---

## Part A — Feature Inventory

### A.1 Models (16 architectures)

| # | Model | Adapter | Entity IDs | c_out | Decoder | Dropout | Task Coverage |
|---|-------|---------|:----------:|:-----:|:-------:|:-------:|:-------------:|
| 1 | DLinear | `DLinearAdapter` | **No** | implicit | No | No | Full (4 tasks) |
| 2 | Transformer | `TransformerAdapter` | **No** | Yes | Yes | Yes | Full |
| 3 | Informer | `InformerAdapter` | **No** | Yes | Yes | Yes | Full |
| 4 | Autoformer | `AutoformerAdapter` | **No** | Yes | Yes | Yes | Full |
| 5 | FEDformer | `FEDformerAdapter` | **No** | Yes | Yes | Yes | Full |
| 6 | iTransformer | `iTransformerAdapter` | **No** | implicit | No | Yes | Full |
| 7 | PatchTST | `PatchTSTAdapter` | **No** | implicit | No | Yes | Full |
| 8 | TimesNet | `TimesNetAdapter` | **No** | Yes | No | Yes | Full |
| 9 | TimeMixer | `TimeMixerAdapter` | **No** | Yes | No | Yes | Full |
| 10 | TimeXer | `TimeXerAdapter` | **No** | Yes | No | Yes | Forecast only |
| 11 | Mamba | `MambaAdapter` | **No** | Yes | No | Yes | Forecast only |
| 12 | LSTM (general) | `LSTMAdapter` | **All 7** | Yes | No | Yes | Nowcast |
| 13 | ExtrapoLSTM | `ExtrapoLSTMAdapter` | Partial | Yes | No | Yes | Extrapolation |
| 14 | Transformer-Enc | `TransformerEncoderAdapter` | **All 7** | Yes | No | Yes | Nowcast/Extrap |
| 15 | TimeLLM | `TimeLLMAdapter` (exported) | **EntityAwareMixin** | implicit | No | Yes | Forecast only |
| 16 | TimeMoE | `TimeMoEAdapter` (exported) | **EntityAwareMixin** | implicit | No | No | Zero-shot |

**Entity identifier modes:** `none`, `embedding`, `onehot`, `coordinates`, `sinusoidal`, `descriptors`, `numeric_id`, `feature_concat` — supported by all TSL adapters (models 1–16) via EntityAwareMixin. Full 7-mode support on models 12–14.

### A.2 Dataset Entity Analysis

#### A.2.1 Which dataset features are actually entities?

| # | Dataset | Cols | Features = entities? | Reasoning |
|---|---------|:----:|:--------------------:|-----------|
| 1 | **Traffic** | 862 | **Yes** | Columns 0–861 are occupancy rates from 862 road sensors across the San Francisco Bay Area. Each column is an independent sensor measuring the same physical quantity (road occupancy %). They are interchangeable entities with spatial locations. |
| 2 | **Electricity** | 321 | **Yes** | Columns 0–320 are hourly kWh consumption from 321 Portuguese households/clients. Each column is an independent metered client, measuring the same quantity. Interchangeable entities. |
| 3 | **PEMS03** | 358 | **Yes** | 358 traffic flow sensors. Shape `(T, 358, 1)` — explicitly indexed by sensor. Interchangeable entities with known adjacency graph. |
| 4 | **PEMS04** | 307 | **Yes** | 307 traffic flow sensors. Shape `(T, 307, 3)` — 3 features per sensor (flow, occupancy, speed). Interchangeable entities. |
| 5 | **PEMS07** | 883 | **Yes** | 883 traffic flow sensors. Same structure as PEMS03. |
| 6 | **PEMS08** | 170 | **Yes** | 170 traffic flow sensors. Same structure as PEMS04. |
| 7 | **Swiss-river** | varies | **Yes** | Columns are `{station_id}_wt` / `{station_id}_at`. Each station is an entity with coordinates and graph topology. Already supports entity identifiers natively. |
| 8 | **Exchange Rate** | 8 | **Borderline** | 8 national currency exchange rates (to USD). Each column is a country's rate — arguably entities (countries), but only 8, so embedding has limited value. The rates may co-move but are heterogeneous (different scales, dynamics). |
| 9 | **ETTh1/h2** | 7 | **No** | Columns are HUFL (High UseFul Load), HULL (High UseLess Load), MUFL, MULL, LUFL, LULL, OT (Oil Temperature). These are **heterogeneous physical quantities** from a single transformer station — different units, different semantics (load vs. temperature). Not interchangeable. |
| 10 | **ETTm1/m2** | 7 | **No** | Same as ETTh, at 15-minute resolution. Not entities. |
| 11 | **Weather** | 21 | **No** | Columns are pressure (mbar), temperature (°C), humidity (%), wind speed (m/s), radiation (W/m²), etc. **Heterogeneous physical quantities** from a single weather station. Not interchangeable. |
| 12 | **ILI** | 7 | **No** | Columns are %WEIGHTED ILI, %UNWEIGHTED ILI, AGE 0-4, AGE 5-24, ILITOTAL, NUM.PROVIDERS, OT. Heterogeneous epidemiological statistics. Not entities. |
| 13 | **M4** | 1 | **No** | Univariate series. No entity concept. |

#### A.2.2 When do entity identifiers help vs. multivariate features?

**Entity identifiers are beneficial when:**

1. **Features are homogeneous.** Traffic sensors all measure road occupancy %; electricity clients all measure kWh. A model that learns per-entity biases/offsets via embeddings can capture individual baseline differences (e.g., a highway sensor vs. a residential street sensor) without needing the model to learn this from the raw values alone.

2. **Features are interchangeable / exchangeable.** If you permute the sensor columns, the forecasting problem shouldn't fundamentally change. Entity embeddings allow the model to treat inputs as "samples from a distribution of entities" rather than fixed positional features, enabling better generalization (transfer to unseen sensors).

3. **Number of entities is large.** With 862 sensors (traffic) or 321 clients (electricity), the model has hundreds of parallel time-series. Entity embeddings provide a compact learned representation (e.g., dim=16) instead of a 862-wide one-hot, enabling parameter sharing and regularization.

4. **Spatial/relational structure exists.** Swiss-river stations and PEMS sensors have known geographic coordinates and adjacency graphs. Entity identifiers (especially `coordinates` and `embedding` modes) can encode this structure, allowing the model to learn spatial correlations.

**Entity identifiers are NOT helpful when:**

1. **Features are heterogeneous.** ETT's HUFL (load in MW) and OT (temperature in °C) measure different physical quantities. Treating them as interchangeable entities would lose the semantic distinction. The model needs to learn separate dynamics for each feature.

2. **Features interact semantically.** Weather features (pressure, temperature, humidity) have physical relationships (e.g., temperature affects humidity). The model benefits from seeing them as a multivariate input vector, not as independent entity time-series.

3. **Very few features.** ETT (7), ILI (7), Exchange (8) — embedding overhead isn't justified. The model can directly learn per-variate patterns with so few dimensions.

**Conclusion for experiments:** We will test entity identifiers on **Traffic, Electricity, PEMS, Swiss-river** (true entity datasets), include Exchange as a borderline case, and run the heterogeneous datasets (ETT, Weather, ILI) **without** entity identifiers as controls. The entity ablation experiment will quantify the benefit.

### A.2.3 Dataset Summary

| # | Dataset | Class | Rows | Channels | Entities? | Time Features | Topology |
|---|---------|-------|------|----------|:---------:|:-------------:|:--------:|
| 1 | ETTh1 | `ETTHourDataset` | 17,420 | 7 | No — heterogeneous | Yes | No |
| 2 | ETTh2 | `ETTHourDataset` | 17,420 | 7 | No — heterogeneous | Yes | No |
| 3 | ETTm1 | `ETTMinuteDataset` | 69,680 | 7 | No — heterogeneous | Yes | No |
| 4 | ETTm2 | `ETTMinuteDataset` | 69,680 | 7 | No — heterogeneous | Yes | No |
| 5 | Weather | `CustomCSVDataset` | 52,696 | 21 | No — heterogeneous | Yes | No |
| 6 | Electricity | `CustomCSVDataset` | 26,304 | 321 | **Yes** — clients | Yes | No |
| 7 | Traffic | `CustomCSVDataset` | 17,544 | 862 | **Yes** — sensors | Yes | No |
| 8 | Exchange Rate | `CustomCSVDataset` | ~7,588 | 8 | Borderline — countries | Yes | No |
| 9 | ILI (illness) | `CustomCSVDataset` | ~966 | 7 | No — heterogeneous | Yes | No |
| 10 | M4 (6 freqs) | `M4Dataset` | varies | 1 | No — univariate | No | No |
| 11 | Swiss-1990 | `SwissRiverDataset` | ~11k | 1–2/stn | **Yes** — stations | epoch-day | **Yes** |
| 12 | Swiss-2010 | `SwissRiverDataset` | ~4k | 1–2/stn | **Yes** — stations | epoch-day | **Yes** |
| 13 | Swiss-Zurich | `SwissRiverDataset` | ~3k | 1–2/stn | **Yes** — stations | epoch-day | **Yes** |
| 14 | PEMS (03/04/07/08) | *(not wired)* | varies | 170–883 | **Yes** — sensors | No | **Yes** |

### A.3 Training & Runtime Features

| Feature | Location | Current Scope |
|---------|----------|---------------|
| Early stopping (patience) | `ForecastTrainer` | All torch models |
| Disable early stopping (ASHA) | `ForecastTrainer` | All torch models |
| NaN mask loss | `ForecastTrainer` | All torch models |
| Teacher forcing (label/zeros/none) | `ForecastTrainer` | Encoder-decoder models |
| Denormalized metrics | `ForecastTrainer` | When `inverse_transform_fn` provided |
| LR scheduling (OneCycle/Cosine) | `ForecastTrainer` | All torch models |
| Configurable loss (MSE/MAE/RMSE) | `ForecastTrainer` | All torch models |
| Accelerator (DDP/DeepSpeed/fp16) | `accelerator.py` | All torch models |
| HPO (Ray ASHA + grid fallback) | `RayOptimizer` | All torch models |
| 11 search spaces + 4 ASHA presets | `search_spaces.py` | Swiss + TimeLLM families |
| 5 scaler types | `scalers.py` | Swiss-river primarily |
| 8 augmentation techniques | `augmentation.py` | Manual; not in trainer loop |
| 7 aggregation methods | `prediction_aggregator.py` | All models |
| WandB + local logging | `loggers/` | All experiments |
| Auto-viz (pred plots, range) | `viz/plots.py` | All experiments |
| CLI (7 subcommands) | `cli.py` | All experiments |
| Data factory registry | `data_factory.py` | ETT + custom + M4 |

### A.4 Integration Gap Summary

| Gap | Impact |
|-----|--------|
| **Entity identifiers on TSL models (#1–11, 15–16)** | 13 of 16 models have zero entity-ID support |
| **PEMS dataset not wired** | 4 graph-topology datasets unused |
| **Swiss-river E2E tests broken** (3-tuple vs 4-tuple) | 4 failing tests |
| **TimeLLM/TimeMoE not exported** in `__init__.py` | Cannot import via standard path |
| **Search spaces only for Swiss + TimeLLM families** | No HPO configs for DLinear/PatchTST/iTransformer/etc. on benchmark datasets |
| **Augmentation not integrated in trainer** | Must be called manually outside training loop |
| **data_factory missing** weather/traffic/electricity/illness/exchange/swiss entries | Only ETT + custom + M4 registered |
| **No unified experiment YAML configs** for reproducible benchmarks | Only 1 example YAML exists |
| **Ray Tune API broken** (metric/mode duplication) | 2 failing tests |

---

## Part B — Integration Plan

### Phase 1: Fix Pre-existing Failures (estimated: 1 session)

| Item | Task | Files |
|------|------|-------|
| 1.1 | **Fix Swiss-river 3-tuple → 4-tuple** — the `TimeSeriesSplit.__getitem__` returns `(feat, targ, time)` but tests/trainer expect `(x, y, x_mark, y_mark)`. Align by splitting time into x_mark/y_mark or adding a 4th element. | `data/ts/timeseriesdataset.py`, `tests/test_e2e_swissriver.py` |
| 1.2 | **Fix Ray Tune metric/mode duplication** — do not pass `metric`/`mode` to `tune.run()` when the scheduler already has them. | `optim/ray_optimizer.py`, `tests/test_adaptation_round.py` |
| 1.3 | **Fix DLinear CUDA→numpy** — add `.cpu()` before `.numpy()` in the test or adapter. | `tests/models/torch/test_dlinear.py` or `models/torch/dlinear.py` |
| 1.4 | **Fix augmentation grid_sample** — fix `_time_warp_linear` output shape for `grid_sample`. | `utils/augmentation.py` |

### Phase 2: Entity Identifier Support for TSL Models (estimated: 2 sessions)

**Goal:** Enable entity identifiers on the 11 Time-Series-Library models (DLinear through Mamba) via a **mixin/wrapper** approach—no invasive rewrite of each model.

| Item | Task | Design |
|------|------|--------|
| 2.1 | **Create `EntityAwareMixin`** — a reusable mixin for `TorchModelAdapter` subclasses that handles entity-ID dispatch before calling `_forward_torch_model()`. | New file: `models/torch/entity_mixin.py`. Extracts the pattern from `_LSTMBaseAdapter._forward_torch_model()` into a reusable mixin. |
| 2.2 | **`EntityWrapper` nn.Module** — wraps any `nn.Module` model, prepending entity features to `x_enc` (for transparent modes) or concatenating embeddings. The inner model sees augmented `x_enc` with `enc_in + entity_dim` channels. | Inside `entity_mixin.py`. |
| 2.3 | **Update each TSL adapter** to inherit `EntityAwareMixin` — For each of the 11 adapters, add the mixin and adjust `enc_in` when entity features are present. | `dlinear.py`, `transformer.py`, `informer.py`, `autoformer.py`, `fedformer.py`, `itransformer.py`, `patchtst.py`, `timesnet.py`, `timemixer.py`, `timexer.py`, `mamba_model.py` |
| 2.4 | **Update data factory** to pass entity features through the standard 4-tuple format. For TSL datasets (`ETTHour`, `CustomCSV`, etc.), entity_id_col = sensor index in multi-entity scenarios (traffic = 862 sensors, electricity = 321 clients). | `data_factory.py`, `torch_datasets.py` |
| 2.5 | **Tests** — add parametrized tests for each model × entity mode (at least `none`, `embedding`, `onehot`). | `tests/models/torch/test_entity_integration.py` |

### Phase 3: Wire Missing Datasets (estimated: 1 session)

| Item | Task |
|------|------|
| 3.1 | **Register weather, traffic, electricity, illness, exchange_rate** in `data_factory.py` `DATASET_REGISTRY`. |
| 3.2 | **Wire PEMS datasets** — create `PEMSDataset` class loading `.npz` files with adjacency matrix; add to registry. Wire to `SpatialTempoDataset` for topology. |
| 3.3 | **Add swiss-river variants** to data_factory (swiss-1990, swiss-2010, swiss-zurich). |
| 3.4 | **Prompt bank** — add prompt text for traffic, electricity, exchange_rate, illness, PEMS (for TimeLLM). |
| 3.5 | **Tests** — smoke-test each newly registered dataset loads and returns correct shapes. |

### Phase 4: Search Spaces & HPO for All Combinations (estimated: 1 session)

| Item | Task |
|------|------|
| 4.1 | **Add search spaces** for all 11 TSL models × the 5 main benchmark datasets (ETTh1, weather, electricity, traffic, exchange_rate). ~55 new space entries. |
| 4.2 | **Per-model ASHA presets** with sensible `grace_period` and `reduction_factor` tuned to each model's convergence speed. |
| 4.3 | **Tests** — verify all new search spaces are loadable and produce valid configs. |

### Phase 5: Augmentation in Training Loop (estimated: 0.5 session)

| Item | Task |
|------|------|
| 5.1 | **Add `augmentation` config key** to `ForecastTrainer`. If present, apply `apply_augmentations(x_enc, aug_list)` during training batches. |
| 5.2 | **Tests** — verify augmented training works and doesn't change eval behavior. |

### Phase 6: Export TimeLLM/TimeMoE + Unified Experiment Configs (estimated: 0.5 session)

| Item | Task |
|------|------|
| 6.1 | **Export TimeLLM + TimeMoE** in `models/torch/__init__.py` with proper optional-dependency guards. |
| 6.2 | **Create experiment YAML templates** for all benchmark combinations under `experiments/configs/`. |

---

## Part C — Test Plan

### C.1 New Test Files

| Test File | Coverage | Est. Tests |
|-----------|----------|------------|
| `tests/test_entity_integration.py` | Entity mixin × all 16 models × 3 modes (none, embedding, onehot) | ~48 |
| `tests/test_dataset_registry.py` | All datasets in data_factory: load, shape, inverse_transform | ~20 |
| `tests/test_pems_dataset.py` | PEMS loading, adjacency matrix, SpatialTempoDataset integration | ~10 |
| `tests/test_augmentation_trainer.py` | Augmentation in training loop, no effect on eval | ~6 |
| `tests/test_search_spaces_full.py` | All ~55 new search spaces load + validate | ~55 |
| `tests/test_experiment_configs.py` | Each YAML config parses and runs 1-epoch smoke test | ~20 |

### C.2 Fix Existing Broken Tests

| Test File | Current Failures | Fix Required |
|-----------|:----------------:|--------------|
| `test_e2e_swissriver.py` | 4 | Tuple alignment (Phase 1.1) |
| `test_adaptation_round.py` | 2 | Ray Tune API fix (Phase 1.2) |
| `test_dlinear.py` | 1 | CUDA→CPU fix (Phase 1.3) |
| `test_augmentation.py` | 1 | grid_sample fix (Phase 1.4) |

### C.3 Target Test Counts

| Status | Current | After Plan |
|--------|---------|------------|
| Total | 502 | ~660 |
| Passing | 478 | ~655 |
| Failing | 8 | 0 |
| Skipped | 15 | ~5 (optional deps) |

---

## Part D — Documentation Plan

| Document | Content | Location |
|----------|---------|----------|
| `docs/entity_identifiers.md` | Tutorial: how to use entity identifiers with any model. All 7 modes explained with code examples. | New |
| `docs/datasets.md` | Catalog of all datasets with shapes, features, loading instructions. | New |
| `docs/benchmark_results.md` | Full benchmark tables and analysis (generated from Part E). | New |
| `docs/models/index.md` | Model catalog — architecture, supported features, config keys. | Update existing |
| `docs/training_comparison.md` | Update resolution status table to reflect all fixes. | Update |
| `docs/cli.md` | Add examples for all newly supported dataset/model combos. | Update |

---

## Part E — Academic Benchmark Experiment

### E.1 Objective

Produce a comprehensive, reproducible benchmark comparing **all models × all datasets × key feature variations**, following the protocol of Long-term Time-series Forecasting benchmarks (Wu et al., 2023; Zeng et al., 2023; Nie et al., 2023). All experiment code and configs will be generated; only minimal pipeline tests will be run during development. Full experiments are run by the user.

### E.2 Experiment Matrix

#### E.2.1 Core Benchmark: Long-term Forecasting

**Models (15):** DLinear, Transformer, Informer, Autoformer, FEDformer, iTransformer, PatchTST, TimesNet, TimeMixer, TimeXer, Mamba, TimeLLM, LSTMAdapter, ExtrapoLSTMAdapter, TransformerEncoderAdapter

**Datasets (12):** ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, Exchange Rate, ILI, Swiss-1990, Swiss-2010, Swiss-Zurich

**Prediction horizons (4):** {96, 192, 336, 720}  
(ILI uses {24, 36, 48, 60} due to weekly granularity and short length; Swiss uses {7, 14, 30, 60})

**Input length:** 96 (standard; ILI uses 36; Swiss uses 30)

**Total runs:** 15 models × 12 datasets × 4 horizons = **720 runs**

**Metrics:** MSE, MAE (primary), plus RMSE, RSE, CORR

#### E.2.2 Entity Identifier Ablation (FULL)

**Models (15):** DLinear, Transformer, Informer, Autoformer, FEDformer, iTransformer, PatchTST, TimesNet, TimeMixer, TimeXer, Mamba, TimeLLM, LSTMAdapter, ExtrapoLSTMAdapter, TransformerEncoderAdapter

**Datasets with natural entities (10):**
- Traffic (862 sensors) — entity = sensor ID
- Electricity (321 clients) — entity = client ID
- Exchange Rate (8 countries) — borderline; included for completeness
- Swiss-1990 (stations) — entity = station ID, with coordinates + topology
- Swiss-2010 (stations) — entity = station ID, with coordinates + topology
- Swiss-Zurich (stations) — entity = station ID, with coordinates + topology
- PEMS03 (358 sensors) — entity = sensor ID, with adjacency graph
- PEMS04 (307 sensors) — entity = sensor ID, with adjacency graph
- PEMS07 (883 sensors) — entity = sensor ID, with adjacency graph
- PEMS08 (170 sensors) — entity = sensor ID, with adjacency graph

**Entity modes (5):** none, embedding, onehot, numeric_id, sinusoidal  
(coordinates only for Swiss variants where lat/lon are available)

**Horizon:** 96 (Exchange: 96; Swiss: 20; PEMS: 12)

**Total runs:** 15 models × 10 datasets × 5 modes = **750 runs**  
(+ 15 models × 3 datasets [Swiss] × 1 extra mode [coordinates] = **795 runs**)

#### E.2.3 Nowcasting on Swiss-river (NEW)

**Goal:** Evaluate all possible models on the Swiss-river nowcasting task, where the model predicts current water temperature given current air temperature as input (no future look-ahead).

**Models (all that support nowcasting/forecasting — 15):**
DLinear, Transformer, Informer, Autoformer, FEDformer, iTransformer, PatchTST, TimesNet, TimeMixer, TimeXer, Mamba, TimeLLM, LSTMAdapter, ExtrapoLSTMAdapter, TransformerEncoderAdapter

**Datasets (3):** Swiss-1990, Swiss-2010, Swiss-Zurich

**Prediction windows:** {1, 7, 14, 30} days  
(1-day = true nowcasting; 7/14/30 = short-term forecasting for comparison)

**Entity mode:** embedding (default for stations) + none (as baseline)

**Total runs:** 15 models × 3 datasets × 4 windows × 2 entity modes = **360 runs**

**Metrics:** MSE, MAE, RMSE, NSE (Nash-Sutcliffe — standard in hydrology)

#### E.2.4 Short-term Forecasting (M4)

**Models (15):** DLinear, Transformer, Informer, Autoformer, FEDformer, iTransformer, PatchTST, TimesNet, TimeMixer, TimeXer, Mamba, TimeLLM, LSTMAdapter, ExtrapoLSTMAdapter, TransformerEncoderAdapter

**Datasets (6):** M4-Yearly, M4-Quarterly, M4-Monthly, M4-Weekly, M4-Daily, M4-Hourly

**Metrics:** SMAPE, MASE, OWA (M4 standard)

**Total runs:** 15 × 6 = **90 runs**

#### E.2.5 Spatial-Temporal Forecasting (PEMS + Swiss-river)

**Models (all 15):** DLinear, Transformer, Informer, Autoformer, FEDformer, iTransformer, PatchTST, TimesNet, TimeMixer, TimeXer, Mamba, TimeLLM, LSTMAdapter, ExtrapoLSTMAdapter, TransformerEncoderAdapter

**Datasets (7):** PEMS03, PEMS04, PEMS07, PEMS08, Swiss-1990, Swiss-2010, Swiss-Zurich

**Feature variations (2):**
- Without entity embeddings (baseline multivariate)
- With entity embeddings

**Horizon:** 12 (PEMS standard), 20 (Swiss-river)

**Total runs:** 15 × 7 × 2 = **210 runs**

#### E.2.6 Feature Effect Studies

##### Study A: Normalization Effect
- **Models (all 15):** DLinear, Transformer, Informer, Autoformer, FEDformer, iTransformer, PatchTST, TimesNet, TimeMixer, TimeXer, Mamba, TimeLLM, LSTMAdapter, ExtrapoLSTMAdapter, TransformerEncoderAdapter
- **Datasets:** ETTh1, Weather, Electricity
- **Scalers:** none, standard, minmax
- **Total:** 15 × 3 × 3 = **135 runs**

##### Study B: Augmentation Effect
- **Models (all 15)**
- **Datasets:** ETTh1, Weather, Traffic
- **Augmentation sets:** none, {jitter}, {jitter+scaling}, {jitter+scaling+window_warp}
- **Total:** 15 × 3 × 4 = **180 runs**

##### Study C: Input Length Sensitivity
- **Models (all 15)**
- **Datasets:** ETTh1, Weather, Electricity
- **Input lengths:** {48, 96, 192, 336, 512}
- **Horizon:** 96
- **Total:** 15 × 3 × 5 = **225 runs**

##### Study D: Teacher Forcing Effect (encoder-decoder models only)
- **Models (4):** Transformer, Informer, Autoformer, FEDformer
- **Datasets:** ETTh1, Weather, Electricity
- **TF modes:** label (with label_len), zeros, none
- **Total:** 4 × 3 × 3 = **36 runs**

### E.3 Experiment Execution Sub-plan

#### E.3.1 Config Generation

A Python script `tools/generate_configs.py` will auto-generate all YAML configs:

```
experiments/configs/
├─ long_term/{model}_{dataset}_H{horizon}.yaml         (720 files)
├─ entity/{model}_{dataset}_{mode}.yaml                 (795 files)
├─ nowcasting/{model}_{swiss_variant}_W{window}_{entity}.yaml  (360 files)
├─ m4/{model}_{freq}.yaml                               (90 files)  
├─ spatial/{model}_{dataset}_{variant}.yaml              (210 files)
├─ ablation_norm/{model}_{dataset}_{scaler}.yaml         (135 files)
├─ ablation_aug/{model}_{dataset}_{aug}.yaml             (180 files)
├─ ablation_seqlen/{model}_{dataset}_L{len}.yaml         (225 files)
└─ ablation_tf/{model}_{dataset}_{mode}.yaml             (36 files)
```

#### E.3.2 Experiment Runner Script

`tools/run_benchmark.py` — orchestrates execution:
- `--config-dir experiments/configs/long_term/` — run a group
- `--config experiments/configs/long_term/dlinear_etth1_H96.yaml` — run one
- `--max-parallel N` — parallel workers (via subprocess, not Ray)
- `--resume` — skip configs whose results already exist
- `--dry-run` — list configs without running
- `--seeds 1,2,3` — multiple random seeds per config
- Results collected to `experiments/results/{group}/{config_stem}.json`

#### E.3.3 Results Aggregator Script

`tools/aggregate_results.py` — post-processing:
- Load all JSON result files from a group
- Generate LaTeX tables (per-experiment-group)
- Generate comparison figures (bar charts, heatmaps, critical difference diagrams)
- Statistical significance tests (Friedman + Nemenyi post-hoc)
- Output to `experiments/results/tables/` and `docs/benchmark_results.md`

#### E.3.4 How to Run Experiments (User Instructions)

```bash
# 1. Activate environment
cd /path/to/liulian-python
source .venv/bin/activate

# 2. Generate all configs (one-time)
python tools/generate_configs.py

# 3. Run a specific experiment group
#    (start with core benchmark; others can run in parallel on different GPUs)

# Core benchmark (720 runs)
python tools/run_benchmark.py --config-dir experiments/configs/long_term/ --seeds 1,2,3

# Entity ablation (795 runs)
python tools/run_benchmark.py --config-dir experiments/configs/entity/ --seeds 1,2,3

# Nowcasting (360 runs)
python tools/run_benchmark.py --config-dir experiments/configs/nowcasting/ --seeds 1,2,3

# M4 short-term (90 runs)
python tools/run_benchmark.py --config-dir experiments/configs/m4/ --seeds 1,2,3

# Spatial-temporal (210 runs)
python tools/run_benchmark.py --config-dir experiments/configs/spatial/ --seeds 1,2,3

# Ablation studies (576 runs total)
python tools/run_benchmark.py --config-dir experiments/configs/ablation_norm/ --seeds 1,2,3
python tools/run_benchmark.py --config-dir experiments/configs/ablation_aug/ --seeds 1,2,3
python tools/run_benchmark.py --config-dir experiments/configs/ablation_seqlen/ --seeds 1,2,3
python tools/run_benchmark.py --config-dir experiments/configs/ablation_tf/ --seeds 1,2,3

# 4. Multi-GPU parallel execution (optional)
#    Run different groups on different GPUs:
CUDA_VISIBLE_DEVICES=0 python tools/run_benchmark.py --config-dir experiments/configs/long_term/ &
CUDA_VISIBLE_DEVICES=1 python tools/run_benchmark.py --config-dir experiments/configs/entity/ &
CUDA_VISIBLE_DEVICES=2 python tools/run_benchmark.py --config-dir experiments/configs/nowcasting/ &
CUDA_VISIBLE_DEVICES=3 python tools/run_benchmark.py --config-dir experiments/configs/spatial/ &

# 5. Resume after interruption (skips completed configs)
python tools/run_benchmark.py --config-dir experiments/configs/long_term/ --resume --seeds 1,2,3

# 6. Aggregate results and generate report
python tools/aggregate_results.py --results-dir experiments/results/ --output docs/benchmark_results.md

# 7. Generate LaTeX tables for paper
python tools/aggregate_results.py --results-dir experiments/results/ --format latex --output experiments/results/tables/
```

#### E.3.5 Execution Priority Order

| Step | Group | Runs | Est. GPU-hours (1 GPU) |
|------|-------|:----:|:----------------------:|
| 1 | Core benchmark (E.2.1) | 720 | 180–720 |
| 2 | Nowcasting (E.2.3) | 360 | 60–180 |
| 3 | Entity ablation (E.2.2) | 795 | 199–398 |
| 4 | Spatial-temporal (E.2.5) | 210 | 50–100 |
| 5 | Input length (E.2.6C) | 225 | 56–112 |
| 6 | Augmentation (E.2.6B) | 180 | 45–90 |
| 7 | Normalization (E.2.6A) | 135 | 34–68 |
| 8 | M4 short-term (E.2.4) | 90 | 8–23 |
| 9 | Teacher forcing (E.2.6D) | 36 | 9–18 |
| **Total** | | **2,751** | **~654–1,709** |

With 4 GPUs running in parallel: **~7–18 days** for the full suite.

### E.4 Report Structure

The final report (`docs/benchmark_results.md` + `BENCHMARK_REPORT.md`) will follow this structure:

```
1. Introduction
   1.1 Motivation — unified framework comparison
   1.2 Contributions — entity identifiers across all models, comprehensive ablations

2. Experimental Setup
   2.1 Datasets — table with statistics (rows, channels, frequency, domain, entity nature)
   2.2 Models — table with architecture summary
   2.3 Implementation details — hardware, training protocol, seeds, hyperparameters
   2.4 Evaluation protocol — metrics, significance testing

3. Results
   3.1 Main Results — Long-term Forecasting (E.2.1)
       Table 1: MSE/MAE for 15 models × 12 datasets × 4 horizons
       Finding: Best model per dataset family
       
   3.2 Entity Identifier Analysis (E.2.2)
       Table 2: Effect of 5 entity modes on 10 entity-datasets × 15 models
       Figure 1: Heatmap of entity mode × model improvement (% change from none)
       Finding: When entity identifiers help vs. hurt; which modes are best
       
   3.3 Nowcasting — Swiss River (E.2.3)
       Table 3: MSE/MAE/NSE for 15 models × 3 swiss variants × 4 prediction windows
       Figure 2: NSE vs. prediction window per model family
       Finding: Best models for hydrological nowcasting; entity embedding impact
       
   3.4 Short-term Forecasting (E.2.4)
       Table 4: SMAPE/MASE/OWA for 15 models × 6 M4 frequencies
       
   3.5 Spatial-Temporal Forecasting (E.2.5)
       Table 5: PEMS + Swiss-river with/without entity embeddings
       Finding: Value of entity embeddings on spatial datasets

4. Ablation Studies
   4.1 Input Length Sensitivity (E.2.6C)
       Figure 3: MSE vs. input length curves per model
       Finding: Optimal input lengths per model family
       
   4.2 Normalization Effect (E.2.6A)
       Table 6: Standard vs. MinMax vs. None for all 15 models
       Finding: When normalization matters
       
   4.3 Data Augmentation Effect (E.2.6B)
       Table 7: Progressive augmentation results for all 15 models
       Finding: Augmentation ROI
       
   4.4 Teacher Forcing (E.2.6D)
       Table 8: Label vs. zeros vs. none for encoder-decoder models
       Finding: Teacher forcing impact quantified

5. Discussion
   5.1 Model recommendations per dataset type
   5.2 Entity identifier guidelines — when to use, which mode
   5.3 Homogeneous-entity vs. heterogeneous-feature datasets
   5.4 Feature interaction effects
   5.5 Computational cost analysis (time per model × dataset)
   
6. Conclusion

Appendix A: Full hyperparameter tables
Appendix B: Per-horizon detailed results
Appendix C: Training curves (selected)
Appendix D: Dataset entity analysis rationale
```

### E.5 Computational Budget Estimate

| Experiment Group | Runs | Est. Time/Run | Total GPU-hours |
|------------------|------|---------------|-----------------|
| Core benchmark (E.2.1) | 720 | 15–60 min | 180–720 h |
| Entity ablation (E.2.2) | 795 | 15–30 min | 199–398 h |
| Nowcasting (E.2.3) | 360 | 10–30 min | 60–180 h |
| M4 short-term (E.2.4) | 90 | 5–15 min | 8–23 h |
| Spatial-temporal (E.2.5) | 210 | 15–30 min | 53–105 h |
| Normalization (E.2.6A) | 135 | 15 min | 34 h |
| Augmentation (E.2.6B) | 180 | 15–20 min | 45–60 h |
| Input length (E.2.6C) | 225 | 15 min | 56 h |
| Teacher forcing (E.2.6D) | 36 | 15 min | 9 h |
| **Total** | **2,751** | — | **~652–1,585 h** |

With a single GPU, the full suite takes ~27–66 days. With 4 GPUs in parallel, ~7–17 days.

---

## Part F — Execution Phases & Dependencies

```
Phase 1: Fix pre-existing failures          (prerequisite for all)
    │
    ├─► Phase 2: Entity ID mixin            (prerequisite for E.2.2, E.2.3, E.2.5)
    │       │
    │       └─► Phase 4: Search spaces      (can start after Phase 2)
    │
    ├─► Phase 3: Wire datasets              (prerequisite for E.2.1, E.2.3, E.2.4, E.2.5)
    │       │
    │       └─► Phase 6: Experiment configs + runner scripts (after Phase 3 + 4)
    │
    └─► Phase 5: Augmentation in trainer    (prerequisite for E.2.6B)

After Phases 1–6 (code complete):
    User runs: experiments using tools/run_benchmark.py (E.3.4)
    User runs: tools/aggregate_results.py to generate report (E.4)
    Phase 7: Write documentation (Part D)
```

---

## Part G — Deliverables Checklist

### Code deliverables (implemented by agent)
- [ ] All 8 pre-existing test failures fixed (Phase 1)
- [ ] `EntityAwareMixin` + `EntityWrapper` (Phase 2)
- [ ] Entity ID support in all 16 model adapters (Phase 2)
- [ ] All datasets registered in data_factory (Phase 3)
- [ ] PEMS dataset class (Phase 3)
- [ ] Search spaces for all model × dataset combos (Phase 4)
- [ ] Augmentation in training loop (Phase 5)
- [x] TimeLLM/TimeMoE exported (Phase 6)
- [ ] `tools/generate_configs.py` — generates all 2,751 experiment YAML configs
- [ ] `tools/run_benchmark.py` — experiment runner with resume, parallel, seeds
- [ ] `tools/aggregate_results.py` — results aggregator with LaTeX/figures/significance
- [ ] Minimum pipeline tests (smoke tests for config generation, runner, aggregation)
- [ ] Documentation updates (Part D)

### User-run deliverables
- [ ] Run 2,751 experiments across 9 groups (E.3.4 instructions)
- [ ] Aggregate results and generate benchmark report (E.4)
