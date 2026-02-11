# Revised Adaptation Mapping Plan

**Run ID:** adapt_20260209_154421  
**Target Project:** liulian-python  
**Source Projects:** Time-LLM, Time-Series-Library  
**Date:** 2026-02-09  
**Strategy:** Minimal Wrapper Approach with proper architectural layers

---

## ARCHITECTURE ALIGNMENT

Based on liulian's architecture principles:

```
liulian/models/           ← ExecutableModel abstract interface (DO NOT modify)
  base.py

liulian/adapters/         ← Adapter layer: each adapter wraps ONE library/domain
  timeseries/             ← NEW: Time-series models wrapper
    __init__.py
    timellm.py            ← Adapter for Time-LLM (≤200 LOC)
    informer.py           ← Adapter for Informer (≤200 LOC)
    autoformer.py         ← Adapter for Autoformer (≤200 LOC)
    [etc. 25 model adapters total]

plugins/timeseries_models/   ← Domain-specific plugin layer
  __init__.py
  base_ts_adapter.py        ← ALREADY EXISTS: Base class
  
  layers/                   ← Shared neural network components
    __init__.py
    embed.py                ← Embedding module (shared, reusable)
    autocorrelation.py      ← AutoCorrelation layer
    fourier_correlation.py  ← Fourier-based correlation
    [etc. reusable layers]
  
  utils/                    ← Time-series utilities
    __init__.py
    metrics.py              ← Forecasting metrics
    losses.py               ← Custom losses
    timefeatures.py         ← Feature engineering
    masking.py              ← Attention masking
  
  models/                   ← Actual model implementations (torch.nn.Module)
    __init__.py
    timellm/
      config.py
      model.py              ← TimeLLM torch.nn.Module
    informer/
      config.py
      model.py              ← Informer torch.nn.Module
    [etc. for all 25 models]
  
  exp_handlers/             ← Task-specific experiment runners
    __init__.py
    base_forecasting.py
    classification.py
    anomaly_detection.py
    [etc. task handlers]
```

---

## THREE-LAYER ADAPTATION STRATEGY

### Layer 1: Core Infrastructure (plugins/timeseries_models/)
Reusable layers, utilities, and base implementations. Can be used by multiple models.

### Layer 2: Model Implementations (plugins/timeseries_models/models/)
Individual PyTorch model implementations (torch.nn.Module subclasses). Each model in its own directory with its config.

### Layer 3: Adapter Wrappers (liulian/adapters/timeseries/)
Thin ExecutableModel adapters (≤200 LOC each). Each wraps one model and exposes it via the liulian interface. Delegates actual forward logic to Layer 1/2.

---

## DETAILED BREAKDOWN

### CATEGORY 1: CORE INFRASTRUCTURE (PHASE 1)

**Target:** `plugins/timeseries_models/`

#### 1.1 Shared Layers (15 components)

| # | Layer | Source | Path | LOC | Risk |
|---|-------|--------|------|-----|------|
| 1.1a | Embedding | Time-Series-Lib/layers/Embed.py | layers/embed.py | 250 | LOW |
| 1.1b | AutoCorrelation | Time-Series-Lib/layers/AutoCorrelation.py | layers/autocorrelation.py | 180 | LOW |
| 1.1c | FourierCorrelation | Time-Series-Lib/layers/FourierCorrelation.py | layers/fourier_correlation.py | 150 | MED |
| 1.1d | MultiWaveletCorrelation | Time-Series-Lib/layers/MultiWaveletCorrelation.py | layers/multiwavelet_correlation.py | 200 | MED |
| 1.1e | StandardNorm | Time-Series-Lib/layers/StandardNorm.py | layers/standard_norm.py | 120 | LOW |
| 1.1f | TransformerEncDec | Time-Series-Lib/layers/Transformer_EncDec.py | layers/transformer_encdec.py | 200 | MED |
| 1.1g | AutoformerEncDec | Time-Series-Lib/layers/Autoformer_EncDec.py | layers/autoformer_encdec.py | 220 | MED |
| 1.1h | CrossformerEncDec | Time-Series-Lib/layers/Crossformer_EncDec.py | layers/crossformer_encdec.py | 250 | HIGH |
| 1.1i | ETSformerEncDec | Time-Series-Lib/layers/ETSformer_EncDec.py | layers/etsformer_encdec.py | 280 | HIGH |
| 1.1j | PyraformerEncDec | Time-Series-Lib/layers/Pyraformer_EncDec.py | layers/pyraformer_encdec.py | 260 | HIGH |
| 1.1k | ConvBlocks | Time-Series-Lib/layers/Conv_Blocks.py | layers/conv_blocks.py | 180 | LOW |
| 1.1l | DWTDecomposition | Time-Series-Lib/layers/DWT_Decomposition.py | layers/dwt_decomposition.py | 160 | MED |
| 1.1m | TimeFilterLayers | Time-Series-Lib/layers/TimeFilter_layers.py | layers/time_filter_layers.py | 200 | MED |
| 1.1n | MSGBlock | Time-Series-Lib/layers/MSGBlock.py | layers/msg_block.py | 170 | LOW |
| 1.1o | SelfAttention | Time-Series-Lib/layers/SelfAttention_Family.py | layers/self_attention_family.py | 240 | MED |

**Subtotal:** 15 layers, ~3,050 LOC, avg risk MEDIUM-LOW

#### 1.2 Utility Modules (4 components)

| # | Module | Path | LOC | Risk |
|---|--------|------|-----|------|
| 1.2a | Metrics | utils/metrics.py | 200 | LOW |
| 1.2b | Losses | utils/losses.py | 150 | LOW |
| 1.2c | TimeFeatures | utils/timefeatures.py | 180 | LOW |
| 1.2d | Masking & Preprocessing | utils/masking.py | 140 | LOW |

**Subtotal:** 4 modules, ~670 LOC, risk LOW

#### 1.3 Data Utilities (2 components)

| # | Component | Source | Path | LOC | Risk |
|---|-----------|--------|------|-----|------|
| 1.3a | DataFactory | Time-Series-Lib/data_provider/ | utils/data_factory.py | 250 | MED |
| 1.3b | DataLoader | Time-Series-Lib/data_provider/ | utils/data_loader.py | 200 | MED |

**Subtotal:** 2 components, ~450 LOC, risk MEDIUM

**Category 1 Summary:**
- **Total:** 21 infrastructure components
- **Total LOC:** ~4,170
- **Timeline:** 4-5 days
- **Pre-requisite for:** All models in Category 2

---

### CATEGORY 2: MODEL IMPLEMENTATIONS (PHASE 2)

**Target:** `plugins/timeseries_models/models/`

Each model gets:
- `models/<model_name>/config.py` - Configuration dataclass
- `models/<model_name>/model.py` - torch.nn.Module implementation
- Total per model: ~300-400 LOC (core implementation)

#### 2.1 Tier 1: Foundational Models (5 models)

| # | Model | Source | Path | Impl LOC | Risk |
|---|-------|--------|------|----------|------|
| 2.1a | TimeLLM | Time-LLM/models/TimeLLM.py | models/timellm/ | 350 | MED |
| 2.1b | Informer | Time-Series-Lib/models/Informer.py | models/informer/ | 320 | MED |
| 2.1c | Autoformer | Time-Series-Lib/models/Autoformer.py | models/autoformer/ | 340 | MED |
| 2.1d | DLinear | Time-LLM/models/DLinear.py | models/dlinear/ | 180 | LOW |
| 2.1e | FEDformer | Time-Series-Lib/models/FEDformer.py | models/fedformer/ | 380 | HIGH |

**Subtotal:** ~1,570 LOC

#### 2.2 Tier 2: Advanced Attention Models (10 models)

| # | Model | Path | LOC | Risk |
|---|-------|------|-----|------|
| 2.2a | Crossformer | models/crossformer/ | 420 | HIGH |
| 2.2b | ETSformer | models/etsformer/ | 450 | HIGH |
| 2.2c | iTransformer | models/itransformer/ | 340 | MED |
| 2.2d | PatchTST | models/patchtst/ | 380 | MED |
| 2.2e | Pyraformer | models/pyraformer/ | 400 | HIGH |
| 2.2f | Reformer | models/reformer/ | 360 | MED |
| 2.2g | TimeMixer | models/timemixer/ | 320 | MED |
| 2.2h | TSMixer | models/tsmixer/ | 300 | MED |
| 2.2i | TiDE | models/tide/ | 280 | MED |
| 2.2j | TemporalFusionTransformer | models/tft/ | 480 | HIGH |

**Subtotal:** ~3,730 LOC

#### 2.3 Tier 3: Specialized & Modern Models (10 models)

| # | Model | Path | LOC | Risk |
|---|-------|------|-----|------|
| 2.3a | Mamba | models/mamba/ | 420 | HIGH |
| 2.3b | MambaSimple | models/mamba_simple/ | 280 | MED |
| 2.3c | NonstationaryTransformer | models/nonstationary_transformer/ | 380 | HIGH |
| 2.3d | Koopa | models/koopa/ | 400 | HIGH |
| 2.3e | MSGNet | models/msgnet/ | 420 | HIGH |
| 2.3f | TimeMoE | models/timemoe/ | 440 | HIGH |
| 2.3g | SCINet | models/scinet/ | 360 | MED |
| 2.3h | SegRNN | models/segrnn/ | 340 | MED |
| 2.3i | LightTS | models/lightts/ | 300 | MED |
| 2.3j | FreTS | models/frets/ | 360 | MED |

**Subtotal:** ~3,700 LOC

**Category 2 Summary:**
- **Total:** 25 model implementations
- **Total LOC:** ~9,000 (model implementations + configs)
- **Adapters:** 25 thin adapters in `liulian/adapters/timeseries/` (~25 × 150 LOC = 3,750 LOC)
- **Total Category 2:** ~12,750 LOC
- **Timeline:** 10-12 days
- **Depends on:** Category 1 completion

---

### CATEGORY 3: TASK-SPECIFIC HANDLERS (PHASE 3)

**Target:** `plugins/timeseries_models/exp_handlers/`

These are experiment orchestrators for specific tasks.

| # | Handler | Purpose | Path | LOC | Risk |
|---|---------|---------|------|-----|------|
| 3.1 | BaseForecastingExperimental | Long & short-term forecasting | exp_handlers/base_forecasting.py | 350 | MED |
| 3.2 | ClassificationExperimental | Time-series classification | exp_handlers/classification.py | 300 | MED |
| 3.3 | AnomalyDetectionExperimental | Anomaly detection | exp_handlers/anomaly_detection.py | 280 | MED |
| 3.4 | ImputationExperimental | Missing value imputation | exp_handlers/imputation.py | 260 | MED |
| 3.5 | ZeroShotExperimental | Zero-shot forecasting | exp_handlers/zero_shot.py | 300 | HIGH |
| 3.6 | UnifiedDataAdapter | Bridge data loaders to liulian adapters | exp_handlers/unified_data_adapter.py | 350 | MED |

**Category 3 Summary:**
- **Total:** 6 task handlers
- **Total LOC:** ~1,840
- **Timeline:** 4-5 days
- **Depends on:** Category 1 & 2 completion

---

## DEPENDENCY UPDATES

### Add to pyproject.toml

```toml
[project.optional-dependencies]
timeseries = [
    "torch>=2.0",           # Deep learning
    "numpy>=1.21",          # Numerical computing
    "scipy>=1.7",           # Signal processing (wavelets, CWT)
    "scikit-learn>=0.24",   # Metrics
]

[build-system]
requires = ["setuptools>=68", "wheel", "torch>=2.0"]
```

---

## EXECUTION SCHEDULE

```
PHASE 1: Core Infrastructure (Days 1-5)
├─ Step 1.1: Layers 1.1a-1.1c (Day 1, ~580 LOC)
├─ Step 1.2: Layers 1.1d-1.1h (Day 2, ~1,130 LOC, integration test)
├─ Step 1.3: Layers 1.1i-1.1o (Day 3, ~1,340 LOC, integration test)
├─ Step 1.4: Utility modules 1.2a-1.2d (Day 4, ~670 LOC)
├─ Step 1.5: Data utilities 1.3a-1.3b (Day 4, ~450 LOC)
└─ Step 1.6: Documentation & Layer verification (Day 5)

PHASE 2: Model Implementations (Days 6-17)
├─ Step 2.1: Foundational models 2.1a-2.1e (Days 6-7, ~1,570 LOC model + 750 LOC adapters)
├─ Step 2.2: Advanced attention models 2.2a-2.2j (Days 8-12, ~3,730 LOC model + 1,500 LOC adapters)
├─ Step 2.3: Specialized models 2.3a-2.3j (Days 13-17, ~3,700 LOC model + 1,500 LOC adapters)
└─ Step 2.4: Cross-model integration testing (Day 17)

PHASE 3: Task Handlers (Days 18-21)
├─ Step 3.1: Core task handlers 3.1-3.3 (Days 18-19, ~930 LOC)
├─ Step 3.2: Specialized handlers 3.4-3.5 (Days 19-20, ~560 LOC)
├─ Step 3.3: Data adapter & integration (Day 20, ~350 LOC)
└─ Step 3.4: End-to-end testing (Day 21)

Buffer & Docs: Days 22-23
```

**Total Estimated:**
- **LOC:** ~27,860 lines of adapted code
- **Timeline:** 23 days (3+ weeks)
- **Test Coverage:** >80% (auto-test after each atomic step)
- **Feature Branch:** `feat/adapt-timeseries-20260209`

---

## APPROVAL CHECKPOINT

**REVISED PLAN CONFIRMATION:**

1. **Category 1 (Infrastructure):** 21 components, ~4,170 LOC in `plugins/timeseries_models/`?  
   ☐ Approve ☐ Modify

2. **Category 2 (Models):** 25 models, ~12,750 LOC split between `plugins/timeseries_models/models/` and `liulian/adapters/timeseries/`?  
   ☐ Approve ☐ Reduce scope ☐ Tiers 1+2 only

3. **Category 3 (Task Handlers):** 6 handlers, ~1,840 LOC in `plugins/timeseries_models/exp_handlers/`?  
   ☐ Approve ☐ Defer to Phase 4 ☐ Reduce scope

4. **Dependencies:** Update pyproject.toml with torch>=2.0, scipy, scikit-learn?  
   ☐ Approve ☐ Modify

5. **Timeline:** Allocate ~3 weeks with daily auto-testing?  
   ☐ Approve ☐ Accelerate ☐ Phase approach

---
