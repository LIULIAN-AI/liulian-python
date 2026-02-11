# Final Adaptation Plan - APPROVED ARCHITECTURE

**Run ID:** adapt_20260209_154421  
**Target Project:** liulian-python  
**Source Projects:** Time-LLM, Time-Series-Library  
**Architecture:** Centralized Core + Plugin Models  
**Status:** Ready for Phase 4 Execution

---

## FINAL STRUCTURE

```
liulian/
├── layers/                          ← NEW: Reusable neural components
│   └── timeseries/
│       ├── __init__.py
│       ├─── embed.py                (250 LOC)
│       ├── autocorrelation.py       (180 LOC)
│       ├── fourier_correlation.py   (150 LOC)
│       ├── multiwavelet_correlation.py (200 LOC)
│       ├── ...[15 total layers]
│       └── self_attention_family.py (240 LOC)
│
├── utils/                           ← Timeseries utilities
│   └── timeseries/
│       ├── __init__.py
│       ├── metrics.py               (200 LOC)
│       ├── losses.py                (150 LOC)
│       ├── timefeatures.py          (180 LOC)
│       ├── masking.py               (140 LOC)
│       ├── data_factory.py          (250 LOC)
│       └── data_loader.py           (200 LOC)
│
├── models/
│   ├── base.py                      (DO NOT MODIFY - ExecutableModel interface)
│   └── timeseries/                  ← Base class for TS models (optional, keeps clean)
│       └── base.py                  (extends ExecutableModel from plugins version)
│
├── adapters/
│   └── timeseries/                  ← THIN ADAPTERS: ExecutableModel wrappers
│       ├── __init__.py
│       ├── timellm.py               (~150 LOC)
│       ├── informer.py              (~150 LOC)
│       ├── autoformer.py            (~150 LOC)
│       ├── dlinear.py               (~150 LOC)
│       ├── fedformer.py             (~150 LOC)
│       ├── crossformer.py           (~150 LOC)
│       ├── etsformer.py             (~150 LOC)
│       ├── itransformer.py          (~150 LOC)
│       ├── patchtst.py              (~150 LOC)
│       ├── pyraformer.py            (~150 LOC)
│       ├── reformer.py              (~150 LOC)
│       ├── timemixer.py             (~150 LOC)
│       ├── tsmixer.py               (~150 LOC)
│       ├── tide.py                  (~150 LOC)
│       ├── tft.py                   (~150 LOC)
│       ├── mamba.py                 (~150 LOC)
│       ├── mamba_simple.py          (~150 LOC)
│       ├── nonstat_transformer.py   (~150 LOC)
│       ├── koopa.py                 (~150 LOC)
│       ├── msgnet.py                (~150 LOC)
│       ├── timemoe.py               (~150 LOC)
│       ├── scinet.py                (~150 LOC)
│       ├── segrnn.py                (~150 LOC)
│       ├── lightts.py               (~150 LOC)
│       └── frets.py                 (~150 LOC)
│
└── tasks/                           ← Could extend with TS-specific tasks if needed

plugins/
└── timeseries_models/               ← Domain-specific plugin
    ├── __init__.py
    ├── base_ts_adapter.py           (KEEP: Base class)
    │
    ├── models/                      ← FULL IMPLEMENTATIONS: torch.nn.Module
    │   ├── __init__.py
    │   ├── timellm/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── model.py             (~350 LOC)
    │   ├── informer/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── model.py             (~320 LOC)
    │   ├── autoformer/              ... [25 models total, avg ~350 LOC each]
    │   └── [22 more models...]
    │
    └── exp_handlers/                ← Task-specific runners
        ├── __init__.py
        ├── base_forecasting.py      (350 LOC)
        ├── classification.py        (300 LOC)
        ├── anomaly_detection.py     (280 LOC)
        ├── imputation.py            (260 LOC)
        ├── zero_shot.py             (300 LOC)
        └── unified_data_adapter.py  (350 LOC)
```

---

## BREAKDOWN BY COMPONENT

### Category 1: Core Layers & Utilities (~4,900 LOC)
- **liulian/layers/timeseries/**: 15 shared neural network layer implementations
- **liulian/utils/timeseries/**: Metrics, losses, features, masking, data utilities  
- **Total:** 21 reusable components
- **Timeline:** 4-5 days
- **Risk:** LOW to MEDIUM

### Category 2: Model Adapters + Implementations (~15,500 LOC)
- **plugins/timeseries_models/models/**: 25 torch.nn.Module implementations (~9,000 LOC)
  - Tier 1: 5 foundational models (TimeLLM, Informer, Autoformer, DLinear, FEDformer)
  - Tier 2: 10 advanced attention models (Crossformer, ETSformer, iTransformer, etc.)
  - Tier 3: 10 specialized/modern models (Mamba, Koopa, TimeMoE, etc.)
- **liulian/adapters/timeseries/**: 25 ExecutableModel wrapper adapters (~3,750 LOC, ≤150 LOC each)
- **Total:** 25 models wrapped
- **Timeline:** 10-12 days
- **Risk:** MEDIUM to HIGH

### Category 3: Task Handlers (~1,840 LOC)
- **plugins/timeseries_models/exp_handlers/**: 6 specialized experiment handlers
  - Base forecasting, classification, anomaly detection, imputation, zero-shot, data adapter
- **Timeline:** 4-5 days
- **Risk:** MEDIUM

---

## TOTAL SCOPE

| Category | Components | LOC | Days | Files |
|----------|------------|-----|------|-------|
| 1: Infrastructure | 21 | 4,900 | 4-5 | 21 new |
| 2: Models | 25 models + 25 adapters | 12,750 | 10-12 | 50 new |
| 3: Task Handlers | 6 | 1,840 | 4-5 | 6 new |
| **TOTAL** | **52** | **19,490** | **23** | **77 new files** |

---

## EXECUTION PHASES

### Phase 4A: Create Feature Branch & Core Layers
**Days 1-5**

Steps:
1. Create feature branch: `git checkout -b feat/adapt-timeseries-20260209`
2. Create directory structure (empty __init__.py files)
3. Adapt layers one-by-one (Steps 1.1a → 1.1o)
4. Adapt utilities (Steps 1.2a → 1.3b)
5. Integration testing
6. Commit: `feat(adapt): Add timeseries core layers and utilities`

### Phase 4B: Model Implementations & Adapters  
**Days 6-17**

Steps organized by Tier:
- **Days 6-7:** Tier 1 foundational models (5 models + 5 adapters with tests)
- **Days 8-12:** Tier 2 advanced attention models (10 models + 10 adapters with tests)  
- **Days 13-17:** Tier 3 specialized models (10 models + 10 adapters with tests)
- **Day 17:** Cross-model verification

Each model gets:
- Implementation file in `plugins/timeseries_models/models/<model>/`
- Config + model.py
- Adapter in `liulian/adapters/timeseries/<model>.py`
- Unit test in `tests/adapters/test_timeseries_<model>.py`
- Integration test for end-to-end forward pass

### Phase 4C: Task Handlers & Integration
**Days 18-21**

Steps:
- Days 18-19: Base forecasting + classification + anomaly handlers
- Days 19-20: Imputation + zero-shot + data adapter
- Day 20: End-to-end task integration tests
- Day 21: Documentation + example notebooks

### Phase 5: Finalization
**Days 22-23**

- Comprehensive integration testing
- Documentation updates
- Example notebooks & tutorials
- Performance benchmarking (optional)
- Merge to main

---

## DEPENDENCIES TO ADD

```toml
# pyproject.toml additions
[project.optional-dependencies]
timeseries = [
    "torch>=2.0",
    "numpy>=1.21", 
    "scipy>=1.7",
    "scikit-learn>=0.24",
]
```

---

## TESTING STRATEGY

**Auto-testing enabled** (default: run tests after each atomic step)

1. **Unit tests** - Each layer, model, adapter gets isolated tests
2. **Integration tests** - Test forward pass with realistic batch sizes
3. **Adapter tests** - Verify ExecutableModel interface compliance
4. **End-to-end tests** - Train/eval cycle with real data

**Coverage target:** >80%

---

## GIT STRATEGY

**Feature branch workflow:**
```bash
feat/adapt-timeseries-20260209
├── commit: "feat(adapt): Add 15 timeseries layers"
├── commit: "feat(adapt): Add timeseries utilities & metrics"
├── commit: "feat(adapt): Add 5 tier-1 models + adapters"
├── commit: "feat(adapt): Add 10 tier-2 models + adapters"
├── commit: "feat(adapt): Add 10 tier-3 models + adapters"
├── commit: "feat(adapt): Add 6 task handlers"
└── commit: "docs(timeseries): Add examples & benchmarks"
```

All commits include test artifacts and can be independently reviewed.

---

## READY FOR PHASE 4: INCREMENTAL EXECUTION

**Confirm approval to proceed:**
- ✅ Architecture correct (centralized core + plugin models)
- ✅ All 25 models approved
- ✅ 21 infrastructure components approved
- ✅ 6 task handlers approved  
- ✅ Testing & feature branch workflow understood

**Type "PROCEED" to start Phase 4 Execution**
