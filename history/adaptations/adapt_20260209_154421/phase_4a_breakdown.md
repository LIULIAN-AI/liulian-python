# Phase 4A: Atomic Execution Steps Breakdown

**Phase:** 4A - Core Infrastructure  
**Components:** 21 (15 layers + 4 utilities + 2 data adapters)  
**Total LOC:** ~4,900  
**Timeline:** 4-5 days with auto-testing  
**Testing:** Enabled - run tests after each step

---

## ATOMIC STEP SEQUENCE

Total steps: 21 + 6 integration steps = 27 atomic operations

### BATCH 1: Project Setup & Directory Structure (Step 0)

**Step 0.1: Create Feature Branch**
```bash
git checkout -b feat/adapt-timeseries-20260209
```
- Creates isolated branch for all adaptations
- All commits branch-local until final merge
- Easy rollback if needed

**Step 0.2: Create Directory Structure**
```bash
# Create required directories
mkdir -p liulian/layers/timeseries
mkdir -p liulian/utils/timeseries
mkdir -p plugins/timeseries_models/exp_handlers
mkdir -p liulian/adapters/timeseries

# Create __init__.py files (empty, to be filled)
touch liulian/layers/__init__.py
touch liulian/layers/timeseries/__init__.py
# ... [repeat for each new directory]
```
- **Files created:** 8
- **LOC:** ~50
- **Risk:** LOW
- **Tests:** None (file structure only)

**Commit:** `feat(adapt): Initialize timeseries module structure`

---

### BATCH 2: Core Neural Network Layers (Steps 1-15)

#### Step 1.1: Embedding Layer
- **Source:** `refer_projects/TimeSeries_20260209_153934/layers/Embed.py`
- **Target:** `liulian/layers/timeseries/embed.py`
- **Changes:**
  - Consolidate 4 embedding types into unified module
  - Rename internal classes (no conflicts)
  - Keep torch.nn.Module interface compatible
- **Files:** 1 new (embed.py)
- **LOC:** 250
- **Risk:** LOW
- **Tests:** 
  - `test_embedding_patch_projection()`
  - `test_embedding_temporal_position()`
  - `test_embedding_value_embedding()`
  - `test_embedding_unified_interface()`
- **Auto-test:** pytest tests/layers/test_timeseries_embed.py
- **Estimated time:** 30 min

**Commit:** `feat(adapt): Add timeseries embedding layer`

---

#### Step 1.2: AutoCorrelation Layer
- **Source:** `refer_projects/TimeSeries_20260209_153934/layers/AutoCorrelation.py`
- **Target:** `liulian/layers/timeseries/autocorrelation.py`
- **Key features:**
  - FFT-based correlation computation
  - Sparse autocorrelation with top-k selection
  - Masking support for attention
- **Files:** 1 new
- **LOC:** 180
- **Risk:** LOW
- **Tests:**
  - `test_autocorrelation_computation()`
  - `test_autocorrelation_sparsity()`
  - `test_autocorrelation_masking()`
  - `test_autocorrelation_batch_shapes()`
- **Dependencies:** torch, torch.fft (requires torch>=1.9)
- **Estimated time:** 25 min

**Commit:** `feat(adapt): Add autocorrelation layer`

---

#### Step 1.3: FourierCorrelation Layer (NEW)
- **Source:** `refer_projects/TimeSeries_20260209_153934/layers/FourierCorrelation.py`
- **Target:** `liulian/layers/timeseries/fourier_correlation.py`
- **Purpose:** Frequency-domain correlation via CWT
- **Files:** 1 new
- **LOC:** 150
- **Risk:** MEDIUM (new algorithm, needs validation)
- **Tests:**
  - `test_fourier_correlation_shape()`
  - `test_fourier_correlation_frequency_domain()`
  - `test_fourier_correlation_smoothness()`
  - `test_fourier_correlation_vs_autocorr()`
- **Dependencies:** torch.fft, scipy.signal (optional)
- **Estimated time:** 30 min

**Commit:** `feat(adapt): Add fourier correlation layer`

---

#### Step 1.4: MultiWaveletCorrelation Layer (NEW)
- **Source:** `refer_projects/TimeSeries_20260209_153934/layers/MultiWaveletCorrelation.py`
- **Target:** `liulian/layers/timeseries/multiwavelet_correlation.py`
- **Purpose:** CWT + FWT hybrid wavelet correlation
- **Files:** 1 new
- **LOC:** 200
- **Risk:** MEDIUM (wavelet transforms, scipy integration)
- **Tests:**
  - `test_multiwavelet_shape()`
  - `test_multiwavelet_scales()`
  - `test_multiwavelet_decomposition()`
  - `test_multiwavelet_reconstruction()`
- **Dependencies:** scipy.signal.morlet, torch
- **Estimated time:** 35 min

**Commit:** `feat(adapt): Add multiwavelet correlation layer`

---

#### Step 1.5: StandardNorm Layer
- **Source:** `refer_projects/TimeSeries_20260209_153934/layers/StandardNorm.py`
- **Target:** `liulian/layers/timeseries/standard_norm.py`
- **Purpose:** Stateful normalization for time-series (Instance/Channel/Batch modes)
- **Files:** 1 new
- **LOC:** 120
- **Risk:** LOW
- **Tests:**
  - `test_standardnorm_instance_mode()`
  - `test_standardnorm_channel_mode()`
  - `test_standardnorm_batch_mode()`
  - `test_standardnorm_statefulness()`
- **Estimated time:** 20 min

**Commit:** `feat(adapt): Add standardnorm layer`

**[Batch 2 Checkpoint]** — After Step 1.5
- **Status:** 5 reusable layers implemented (~900 LOC)
- **Integration test:** Verify all 5 layers chain together without shape mismatches
- **Command:** `pytest tests/layers/test_timeseries_*.py -v`

---

#### Steps 1.6-1.10: Encoder-Decoder Modules (5 layers)

Each following the same pattern:

| Step | Layer | LOC | Notes |
|------|-------|-----|-------|
| 1.6 | TransformerEncDec | 200 | Standard Transformer encoder-decoder |
| 1.7 | AutoformerEncDec | 220 | Decomposition-based Transformer |
| 1.8 | CrossformerEncDec | 250 | Dimension-exchange attention |
| 1.9 | ETSformerEncDec | 280 | Exponential smoothing layers |
| 1.10 | PyraformerEncDec | 260 | Pyramid attention structure |

**Batch 2A Checkpoint** (after 1.10):
- 10 layers total (~1,630 LOC)
- Integration test encoder-decoder chain
- Test shape transformations: (B, L, D) → (B, P, D)

---

#### Steps 1.11-1.15: Remaining Core Layers (5 layers)

| Step | Layer | LOC | Type | Risk | Notes |
|------|-------|-----|------|------|-------|
| 1.11 | ConvBlocks | 180 | Convolutional | LOW | 1x1, 3x3 conv combos |
| 1.12 | DWTDecomposition | 160 | Signal Proc | MED | Discrete wavelet transforms |
| 1.13 | TimeFilterLayers | 200 | Filtering | MED | Butterworth, Kalman filters |
| 1.14 | MSGBlock | 170 | Aggregation | LOW | Multi-scale graph block |
| 1.15 | SelfAttentionFamily | 240 | Attention | MED | Dot-product, linear, sparse variants |

**Batch 2B Checkpoint** (after 1.15):  
- All 15 layers complete (~3,050 LOC)
- Full integration test: Stack random models using these layers
- Performance benchmark: Layer forward pass timing

**Cumulative:** 15 layers, 3 hours total, ready for model implementations

---

### BATCH 3: Utility Modules (Steps 2.1-2.6)

#### Step 2.1: Metrics Module
- **Target:** `liulian/utils/timeseries/metrics.py`
- **Functions:** MAE, MSE, RMSE, MAPE, SMAPE, NMAE, NMSE
- **Variants:** Aggregate, per-horizon, weighted
- **LOC:** 200
- **Tests:** 8 test functions (one per metric)
- **Estimated time:** 25 min

**Commit:** `feat(adapt): Add timeseries metrics utilities`

---

#### Step 2.2: Losses Module
- **Target:** `liulian/utils/timeseries/losses.py`
- **Functions:** MSE, MAE, Huber, Quantile, MAPE losses
- **Variants:** Weighted, masked, horizon-aware
- **LOC:** 150
- **Tests:** 6 test functions
- **Estimated time:** 20 min

**Commit:** `feat(adapt): Add timeseries loss functions`

---

#### Step 2.3: TimeFeatures Module
- **Target:** `liulian/utils/timeseries/timefeatures.py`
- **Features:** Month, dayofweek, hour, day, minute, second
- **Variants:** Multiple calendar systems support
- **LOC:** 180
- **Tests:** 5 test functions (different temporal features)
- **Estimated time:** 20 min

**Commit:** `feat(adapt): Add timefeature engineering utilities`

---

#### Step 2.4: Masking Module
- **Target:** `liulian/utils/timeseries/masking.py`
- **Functions:** Attention masks, causal masks, padding masks
- **Utilities:** Mask generation, verification
- **LOC:** 140
- **Tests:** 4 test functions
- **Estimated time:** 15 min

**Commit:** `feat(adapt): Add masking utilities`

---

#### Step 2.5: DataFactory Module
- **Target:** `liulian/utils/timeseries/data_factory.py`
- **Purpose:** Unified dataset loading (ETTh, ETTm, ECL, Weather, Traffic, etc.)
- **LOC:** 250
- **Supports:** 10+ standard time-series datasets
- **Tests:** 5 test functions (load different datasets)
- **Estimated time:** 30 min

**Commit:** `feat(adapt): Add timeseries data factory`

---

#### Step 2.6: DataLoader Module
- **Target:** `liulian/utils/timeseries/data_loader.py`
- **Purpose:** Create torch Dataset/DataLoader from raw arrays
- **Features:** Normalization, windowing, scaling, train/val/test splits
- **LOC:** 200
- **Tests:** 6 test functions
- **Estimated time:** 25 min

**Commit:** `feat(adapt): Add timeseries dataloader utilities`

**Batch 3 Checkpoint** (after 2.6):
- All 6 utility modules complete (~1,120 LOC)
- Joint integration test: Load dataset → create batches → compute metrics
- Verify: Metrics on sample forecasts

---

### BATCH 4: Integration Testing & Documentation (Steps 3.1-3.6)

#### Step 3.1: Layer Integration Test
- **Test file:** `tests/integration/test_timeseries_layers_integration.py`
- **Validates:**
  - All 15 layers importable
  - Layer stacking without shape errors
  - Gradient flow through layer chains
  - Memory efficiency
- **LOC:** 150
- **Time:** 30 min

---

#### Step 3.2: Utilities Integration Test
- **Test file:** `tests/integration/test_timeseries_utils_integration.py`
- **Validates:**
  - Metrics computation correctness
  - Loss backward passes
  - Data loading end-to-end
  - Feature engineering correctness
- **LOC:** 200
- **Time:** 30 min

---

#### Step 3.3: Performance Benchmarking
- **Script:** `benchmarks/timeseries_layers_perf.py`
- **Measures:**
  - Layer forward/backward timing
  - Memory usage (peak & avg)
  - Throughput (samples/sec)
- **LOC:** 100
- **Time:** 20 min

---

#### Step 3.4: Module Documentation
- **Files:** Docstrings in each module
- **Format:** Sphinx-compatible RST in docstrings
- **LOC:** 400 (docstrings only)
- **Time:** 40 min

---

#### Step 3.5: Example Notebook
- **File:** `examples/timeseries_layers_demo.ipynb`
- **Content:**
  - Load a dataset
  - Create embeddings
  - Apply correlation layers
  - Visualize outputs
- **Cells:** 8-10
- **Time:** 30 min

---

#### Step 3.6: Update `__init__.py` Files
- **Files:** 4 (`liulian/layers/timeseries/__init__.py`, etc.)
- **Content:** Export all public classes/functions
- **LOC:** 100
- **Time:** 15 min

**Batch 4 Checkpoint:**
- All documentation complete
- Example notebook runs without errors
- 100% imports verified

---

## PHASE 4A EXECUTION TIMELINE

```
Day 1 (4 hours):
├─ 00:30 - Step 0.1-0.2: Setup & structure
├─ 00:30 - Step 1.1: Embedding layer
├─ 00:25 - Step 1.2: AutoCorrelation
├─ 00:30 - Step 1.3: FourierCorrelation
├─ 00:35 - Step 1.4: MultiWaveletCorrelation
├─ 00:20 - Step 1.5: StandardNorm
└─ 00:45 - Batch 2 integration test & checkpoint

Day 2 (4 hours):
├─ 00:30 - Steps 1.6-1.8: TransformerEncDec, AutoformerEncDec, CrossformerEncDec
├─ 00:30 - Steps 1.9-1.10: ETSformerEncDec, PyraformerEncDec
├─ 00:45 - Batch 2A integration test & checkpoint
├─ 01:30 - Steps 1.11-1.15: ConvBlocks, DWT, TimeFilter, MSG, SelfAttention
└─ 00:45 - Batch 2B integration test & checkpoint

Day 3 (3.5 hours):
├─ 00:25 - Step 2.1: Metrics
├─ 00:20 - Step 2.2: Losses
├─ 00:20 - Step 2.3: TimeFeatures
├─ 00:15 - Step 2.4: Masking
├─ 00:30 - Step 2.5: DataFactory
├─ 00:25 - Step 2.6: DataLoader
└─ 00:45 - Batch 3 integration test

Day 4 (3 hours):
├─ 00:30 - Step 3.1: Layer integration test
├─ 00:30 - Step 3.2: Utilities integration test  
├─ 00:20 - Step 3.3: Performance benchmark
├─ 00:40 - Step 3.4: Documentation
├─ 00:30 - Step 3.5: Example notebook
├─ 00:15 - Step 3.6: Init files export
└─ 00:15 - Final verification & commit

Day 5 (buffer):
├─ Bug fixes from Day 1-4
├─ Additional testing
├─ Code review & cleanup
└─ Ready for Phase 4B
```

**Total Phase 4A:** ~14.5 hours of execution ≈ 2 full days with breaks

---

## ATOMIC STEP CONFIRMATION

**Each step follows this pattern:**

1. **Generate** code from reference project + adaptation rules
2. **Present** change with diff preview (first 30 lines)
3. **Approve** - User confirms ✅ (or edits)
4. **Apply** - Write file(s) to disk
5. **Test** - Run pytest on that component
6. **Commit** - git add + git commit with message
7. **Report** - Show metrics & progress

---

## READY TO EXECUTE PHASE 4A?

**Confirm to start:**
- [ ] Create feature branch & directory structure
- [ ] Execute Step 1.1 (Embedding layer) first
- [ ] Proceed through sequence with auto-testing

**Type "EXECUTE PHASE 4A" to begin**
