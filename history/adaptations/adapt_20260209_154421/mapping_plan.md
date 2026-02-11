# Adaptation Mapping Plan

**Run ID:** adapt_20260209_154421  
**Target Project:** liulian-python  
**Source Projects:** Time-LLM, Time-Series-Library  
**Date:** 2026-02-09  
**Strategy:** Minimal Wrapper Approach - Wrap external models in ExecutableModel interfaces

---

## OVERVIEW

Total Planned Adaptations: 46 items across 3 categories
- **Category 1 (PHASE 1):** Core Layers & Utilities - 15 items
- **Category 2 (PHASE 2):** High-Value Models - 25 items  
- **Category 3 (PHASE 3):** Task-Specific Handlers - 6 items

---

## CATEGORY 1: CORE LAYERS & UTILITIES (PHASE 1)

### Shared Layers (available in both projects)
These can be consolidated and improved with Time-Series-Library's more advanced implementations.

#### Item 1.1: Enhanced Embedding Module
- **Source:** Time-Series-Library/layers/Embed.py
- **Target Path:** `liulian/timeseries_models/layers/embed.py`
- **Adaptation Type:** CREATE_NEW_MODULE
- **Key Changes:**
  - Consolidate patch embedding, temporal embedding, value embedding
  - Add support for moment decomposition embeddings
  - Enhance normalization options
- **Risk Level:** LOW
- **Estimated LOC:** 250
- **Dependencies:** torch, numpy
- **Tests:** 5 new tests

#### Item 1.2: AutoCorrelation Layer
- **Source:** Time-Series-Library/layers/AutoCorrelation.py  
- **Target Path:** `liulian/timeseries_models/layers/autocorrelation.py`
- **Adaptation Type:** CREATE_NEW_MODULE
- **Key Changes:**
  - Maintain compatibility with Time-LLM version
  - Use Time-Series-Library's enhanced FFT-based implementation
  - Add masking support
- **Risk Level:** LOW
- **Estimated LOC:** 180
- **Dependencies:** torch, torch.fft
- **Tests:** 4 new tests

#### Item 1.3: FourierCorrelation Layer (NEW)
- **Source:** Time-Series-Library/layers/FourierCorrelation.py
- **Target Path:** `liulian/timeseries_models/layers/fourier_correlation.py`
- **Adaptation Type:** CREATE_NEW_MODULE
- **Key Changes:**
  - Frequency-domain correlation computation
  - Requires torch >= 1.9 (fft module)
- **Risk Level:** MEDIUM
- **Estimated LOC:** 150
- **Dependencies:** torch.fft
- **Tests:** 4 new tests

#### Item 1.4: MultiWaveletCorrelation Layer (NEW)
- **Source:** Time-Series-Library/layers/MultiWaveletCorrelation.py
- **Target Path:** `liulian/timeseries_models/layers/multiwavelet_correlation.py`
- **Adaptation Type:** CREATE_NEW_MODULE
- **Key Changes:**
  - Wavelet-based correlation computation
  - CWT-FWT hybrid approach
- **Risk Level:** MEDIUM
- **Estimated LOC:** 200
- **Dependencies:** torch, scipy.signal (optional)
- **Tests:** 4 new tests

#### Item 1.5: Standard Normalization Layer
- **Source:** Time-Series-Library/layers/StandardNorm.py
- **Target Path:** `liulian/timeseries_models/layers/standard_norm.py`
- **Adaptation Type:** CREATE_NEW_MODULE
- **Key Changes:**
  - Stateful normalization for time series
  - Instance, channel, batch normalization modes
- **Risk Level:** LOW
- **Estimated LOC:** 120
- **Dependencies:** torch
- **Tests:** 3 new tests

#### Item 1.6-1.10: Encoder-Decoder Modules
Consolidate 5 architectural patterns:
- `Transformer_EncDec.py` → `liulian/timeseries_models/layers/transformer_encdec.py` (MEDIUM risk, 200 LOC)
- `Autoformer_EncDec.py` → `liulian/timeseries_models/layers/autoformer_encdec.py` (MEDIUM risk, 220 LOC)
- `Crossformer_EncDec.py` → `liulian/timeseries_models/layers/crossformer_encdec.py` (HIGH risk, 250 LOC)
- `ETSformer_EncDec.py` → `liulian/timeseries_models/layers/etsformer_encdec.py` (HIGH risk, 280 LOC)
- `Pyraformer_EncDec.py` → `liulian/timeseries_models/layers/pyraformer_encdec.py` (HIGH risk, 260 LOC)

Risk: These use different attention mechanisms; consolidation needed.
Tests: 3 tests each = 15 total

#### Item 1.11: Conv Blocks Module
- **Source:** Time-Series-Library/layers/Conv_Blocks.py
- **Target Path:** `liulian/timeseries_models/layers/conv_blocks.py`
- **Adaptation Type:** CREATE_NEW_MODULE
- **Risk Level:** LOW
- **Estimated LOC:** 180
- **Tests:** 3 new tests

#### Item 1.12-1.15: Utility Modules (4 items)
Core utilities consolidated from Time-LLM and Time-Series-Library:

1. **Losses Module** → `liulian/timeseries_models/utils/losses.py`
   - MSE, MAE, MAPE, RMSE, SMAPE losses
   - Custom time-series losses (seasonal, trend-aware)
   - Risk: LOW, 150 LOC

2. **Metrics Module** → `liulian/timeseries_models/utils/metrics.py`
   - MAE, MSE, RMSE, MAPE, SMAPE, NMAE
   - Horizon-wise evaluation
   - Risk: LOW, 200 LOC

3. **Time Features Module** → `liulian/timeseries_models/utils/timefeatures.py`
   - Month-of-year, day-of-week, hour-of-day features
   - Support for multiple calendar systems
   - Risk: LOW, 180 LOC

4. **Masking & Preprocessing** → `liulian/timeseries_models/utils/masking.py`
   - Attention masking utilities
   - Data preprocessing helpers
   - Risk: LOW, 140 LOC

**Category 1 Summary:**
- Total items: 15
- Total estimated LOC: 2,995
- Average risk: MEDIUM-LOW
- Expected time: 3-4 days
- Pre-requisite for: All Category 2 and 3 items

---

## CATEGORY 2: HIGH-VALUE MODELS (PHASE 2)

Target: 25 models wrapped in ExecutableModel adapters.

### Tier 1: Foundational Models (5 models)
Essential architectures appearing across multiple research papers.

#### Item 2.1: TimeLLM Model Adapter
- **Source:** Time-LLM/models/TimeLLM.py + Time-LLM/layers
- **Target Path:** `liulian/timeseries_models/timellm_adapter.py`
- **Adapter Class:** `TimeLLMExecutableAdapter(ExecutableModel)`
- **Key Changes:**
  - Wrap torch.nn.Module in ExecutableModel interface
  - Support flexible input shapes
  - LLM prompt integration (optional feature flag)
- **Configuration Mapping:**
  ```
  Original args → New config dict:
  - seq_len → input_seq_len
  - label_len → label_len  
  - pred_len → output_seq_len
  - d_model → hidden_dim
  - n_heads → num_heads
  - n_layers → num_layers
  ```
- **Risk Level:** MEDIUM (new wrapper, existing model stable)
- **Estimated LOC:** 180 (adapter) + 50 (tests)
- **Dependencies:** torch, LLM (optional)

#### Item 2.2: Informer Model Adapter
- **Source:** Time-Series-Library/models/Informer.py + layers
- **Target Path:** `liulian/timeseries_models/informer_adapter.py`
- **Adapter Class:** `InformerExecutableAdapter(ExecutableModel)`
- **Key Differences:** Sparse attention (ProbSparse), canonical ensemble
- **Risk Level:** MEDIUM
- **Estimated LOC:** 200

#### Item 2.3: Autoformer Model Adapter
- **Source:** Both projects (Time-Series-Library version is more complete)
- **Target Path:** `liulian/timeseries_models/autoformer_adapter.py`
- **Adapter Class:** `AutoformerExecutableAdapter(ExecutableModel)`
- **Key Differences:** Trend decomposition, seasonal-trend separation
- **Risk Level:** MEDIUM
- **Estimated LOC:** 220

#### Item 2.4: DLinear Model Adapter
- **Source:** Time-LLM/models/DLinear.py (lighter) or Time-Series-Library version
- **Target Path:** `liulian/timeseries_models/dlinear_adapter.py`
- **Adapter Class:** `DLinearExecutableAdapter(ExecutableModel)`
- **Key Differences:** Simple linear model baseline
- **Risk Level:** LOW
- **Estimated LOC:** 100

#### Item 2.5: FEDformer Model Adapter
- **Source:** Time-Series-Library/models/FEDformer.py
- **Target Path:** `liulian/timeseries_models/fedformer_adapter.py`
- **Adapter Class:** `FEDformerExecutableAdapter(ExecutableModel)`
- **Key Differences:** Fourier enhanced decomposition
- **Requires:** Item 1.3 (FourierCorrelation)
- **Risk Level:** HIGH
- **Estimated LOC:** 240

### Tier 2: Advanced Attention Models (10 models)
Modern variants with specialized attention mechanisms.

| Item | Model Name | Source | Target | Risk | LOC |
|------|-----------|--------|--------|------|-----|
| 2.6 | Crossformer | Time-Series-Lib/Crossformer.py | crossformer_adapter.py | HIGH | 260 |
| 2.7 | ETSformer | Time-Series-Lib/ETSformer.py | etsformer_adapter.py | HIGH | 280 |
| 2.8 | iTransformer | Time-Series-Lib/iTransformer.py | itransformer_adapter.py | MEDIUM | 210 |
| 2.9 | PatchTST | Time-Series-Lib/PatchTST.py | patchtst_adapter.py | MEDIUM | 230 |
| 2.10 | Pyraformer | Time-Series-Lib/Pyraformer.py | pyraformer_adapter.py | HIGH | 250 |
| 2.11 | Reformer | Time-Series-Lib/Reformer.py | reformer_adapter.py | MEDIUM | 200 |
| 2.12 | TMixer | Time-Series-Lib/TimeMixer.py | timemixer_adapter.py | MEDIUM | 190 |
| 2.13 | TSMixer | Time-Series-Lib/TSMixer.py | tsmixer_adapter.py | MEDIUM | 180 |
| 2.14 | TiDE | Time-Series-Lib/TiDE.py | tide_adapter.py | MEDIUM | 170 |
| 2.15 | TemporalFusionTransformer | Time-Series-Lib/TemporalFusionTransformer.py | tft_adapter.py | HIGH | 300 |

**Subtotal Tier 2:** 10 models, ~2,400 LOC

### Tier 3: Specialized & Modern Models (10 models)
Recent architectures with unique approaches (Mamba, normalized transformers, MoE, etc).

| Item | Model Name | Source | Target | Risk | LOC |
|------|-----------|--------|--------|------|-----|
| 2.16 | Mamba | Time-Series-Lib/Mamba.py | mamba_adapter.py | HIGH | 280 |
| 2.17 | MambaSimple | Time-Series-Lib/MambaSimple.py | mamba_simple_adapter.py | MEDIUM | 160 |
| 2.18 | Nonstationary Transformer | Time-Series-Lib/Nonstationary_Transformer.py | nonstat_trans_adapter.py | HIGH | 240 |
| 2.19 | Koopa | Time-Series-Lib/Koopa.py | koopa_adapter.py | HIGH | 250 |
| 2.20 | MSGNet | Time-Series-Lib/MSGNet.py | msgnet_adapter.py | HIGH | 270 |
| 2.21 | TimeMoE | Time-Series-Lib/TimeMoE.py | timemoe_adapter.py | HIGH | 290 |
| 2.22 | SCINet | Time-Series-Lib/SCINet.py | scinet_adapter.py | MEDIUM | 200 |
| 2.23 | SegRNN | Time-Series-Lib/SegRNN.py | segrnn_adapter.py | MEDIUM | 180 |
| 2.24 | LightTS | Time-Series-Lib/LightTS.py | lightts_adapter.py | MEDIUM | 170 |
| 2.25 | FreTS | Time-Series-Lib/FreTS.py | frets_adapter.py | MEDIUM | 210 |

**Subtotal Tier 3:** 10 models, ~2,340 LOC

**Category 2 Summary:**
- Total models: 25
- Tier 1 (foundational): 5 models, ~950 LOC
- Tier 2 (advanced attention): 10 models, ~2,400 LOC  
- Tier 3 (specialized): 10 models, ~2,340 LOC
- **Total Category 2: ~5,690 LOC**
- Average risk: MEDIUM-HIGH
- Expected timeline: 10-12 days
- Depends on: Category 1 completion

---

## CATEGORY 3: TASK-SPECIFIC HANDLERS (PHASE 3)

Abstract experiment handlers adapted from Time-Series-Library for common tasks:

#### Item 3.1: BaseForecastingTask
- **Source:** Time-Series-Lib/exp/exp_long_term_forecasting.py + exp_short_term_forecasting.py
- **Target Path:** `liulian/timeseries_models/exp_handlers/base_forecasting.py`
- **Purpose:** Unified training/evaluation loop for forecasting tasks
- **Risk Level:** MEDIUM
- **Estimated LOC:** 300

#### Item 3.2: ClassificationTaskHandler
- **Source:** Time-Series-Lib/exp/exp_classification.py
- **Target Path:** `liulian/timeseries_models/exp_handlers/classification_handler.py`
- **Risk Level:** MEDIUM
- **Estimated LOC:** 280

#### Item 3.3: AnomalyDetectionHandler
- **Source:** Time-Series-Lib/exp/exp_anomaly_detection.py
- **Target Path:** `liulian/timeseries_models/exp_handlers/anomaly_detection_handler.py`
- **Risk Level:** MEDIUM
- **Estimated LOC:** 260

#### Item 3.4: ImputationHandler
- **Source:** Time-Series-Lib/exp/exp_imputation.py
- **Target Path:** `liulian/timeseries_models/exp_handlers/imputation_handler.py`
- **Risk Level:** MEDIUM
- **Estimated LOC:** 240

#### Item 3.5: ZeroShotForecastingHandler
- **Source:** Time-Series-Lib/exp/exp_zero_shot_forecasting.py
- **Target Path:** `liulian/timeseries_models/exp_handlers/zero_shot_handler.py`
- **Risk Level:** HIGH (requires special handling)
- **Estimated LOC:** 280

#### Item 3.6: Unified Data Loader Adapter
- **Source:** Time-Series-Lib/data_provider/
- **Target Path:** `liulian/timeseries_models/data_adapters.py`
- **Purpose:** Bridge between Time-Series-Library dataloaders and liulian's adapter system
- **Risk Level:** MEDIUM
- **Estimated LOC:** 350

**Category 3 Summary:**
- Total items: 6
- Total estimated LOC: 1,710
- Average risk: MEDIUM
- Expected timeline: 4-5 days
- Depends on: Category 1 & 2 completion

---

## DEPENDENCY ANALYSIS

### New Required Dependencies
```yaml
required:
  - torch >= 2.0          # Deep learning framework
  - numpy >= 1.21        # Numerical computing
  - pandas >= 1.3        # Data handling

optional:
  - scipy >= 1.7         # Signal processing (for wavelet layers)
  - scikit-learn >= 0.24 # Metrics and utilities
  - tensorboard >= 2.8   # Visualization
  - wandb >= 0.13        # Experiment tracking (already in liulian)
```

### Version Conflicts
- **torch:** Time-LLM requires torch>=1.9, Time-Series-Library requires torch>=1.10
  → Resolve to: **torch>=2.0** (supports all features)
- **numpy:** Both require numpy>=1.19
  → Use: **numpy>=1.21** (for performance)

---

## TIMELINE & EXECUTION PLAN

```
Phase 1: CORE LAYERS & UTILITIES (Days 1-4)
  ├─ Category 1.1-1.5: Basic Layers (Day 1)
  ├─ Category 1.6-1.10: Encoder-Decoder (Day 2) 
  ├─ Category 1.11-1.15: Conv & Utils (Days 3-4)
  └─ Integration Testing & Documentation (Day 4.5)

Phase 2: CORE MODELS (Days 5-16)
  ├─ Tier 1: Foundational (Day 5)
  ├─ Tier 2: Advanced Attention (Days 6-11)
  ├─ Tier 3: Specialized Models (Days 12-16)
  └─ Cross-Model Testing (Day 16.5)

Phase 3: TASK HANDLERS (Days 17-20)
  ├─ Forecasting, Classification, Anomaly Detection (Days 17-18)
  ├─ Imputation, Zero-Shot, Data Adapters (Days 19-20)
  └─ End-to-End Integration Testing (Day 20.5)

Buffer & Documentation: Days 21-22
```

---

## APPROVAL CHECKPOINT

**To proceed with atomic execution, please confirm:**

1. ✅ **Category 1 (Layers & Utils):** Proceed with 15 core components?
2. ✅ **Category 2 (Models):** Adapt all 25 models in 3 tiers?
3. ✅ **Category 3 (Task Handlers):** Create 6 task-specific handlers?
4. ✅ **Dependencies:** Update torch to >=2.0, add scipy (optional)?
5. ✅ **Timeline:** Allocate ~3 weeks for comprehensive adaptation?

---
