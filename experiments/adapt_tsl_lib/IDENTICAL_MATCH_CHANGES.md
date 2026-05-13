# Identical Match Changes for Transformer Model

## Summary

Successfully achieved **byte-identical results** (0.00e+00 difference) between TSL and Liulian Transformer implementations across all tested datasets using the main `compare_tsl_liulian.py` pipeline with `--identical` mode:

| Dataset | Epochs | Test MSE | Difference |
|---------|--------|----------|------------|
| ETTh1 | 3 | 1.0539304402 | 0.00e+00 ✅ |
| ETTh2 | 3 | 2.1674094529 | 0.00e+00 ✅ |
| ETTm1 | 3 | 0.5916380831 | 0.00e+00 ✅ |
| ETTm2 | 3 | 0.6678294579 | 0.00e+00 ✅ |
| Weather | 3 | 0.5304859507 | 0.00e+00 ✅ |

---

## Files Modified

| File | Purpose | Lines Changed |
|------|---------|---------------|
| [`experiments/adapt_tsl_lib/compare_tsl_liulian.py`](compare_tsl_liulian.py) | Main comparison script - added `--identical` mode | +200 lines |
| [`experiments/adapt_tsl_lib/tsl_float64_dataloader.py`](tsl_float64_dataloader.py) | Float64-aligned dataloader (dependency) | 280 lines (new) |

---

## Table 1: Changes to compare_tsl_liulian.py

### A. New `--identical` Command-Line Flag

| Component | Line(s) | Code | Description |
|-----------|---------|------|-------------|
| `--identical` flag | [3592-3600](compare_tsl_liulian.py#L3592-L3600) | `parser.add_argument("--identical", ...)` | Enables in-process identical comparison mode |
| `--identical-epochs` flag | [3602-3604](compare_tsl_liulian.py#L3602-L3604) | `parser.add_argument("--identical-epochs", ...)` | Number of epochs for identical mode (default: 3) |

### B. New `run_identical_comparison()` Function

| Component | Line(s) | Description | Impact |
|-----------|---------|-------------|--------|
| Function definition | [3020-3025](compare_tsl_liulian.py#L3020-L3025) | `def run_identical_comparison(dataset_name, model_name, epochs, seed)` | Main entry point |
| Dataset configs | [3039-3046](compare_tsl_liulian.py#L3039-L3046) | `DATASET_CONFIGS = {...}` for ETT* and Weather | Dataset parameters |
| `set_deterministic()` | [3055-3068](compare_tsl_liulian.py#L3055-L3068) | Full deterministic mode settings | **CRITICAL** |
| TSL model creation | [3082-3098](compare_tsl_liulian.py#L3082-L3098) | `TSLArgs` with `dropout=0.0` | **CRITICAL** |
| Liulian model creation | [3100-3107](compare_tsl_liulian.py#L3100-L3107) | Config dict with `dropout=0.0` | **CRITICAL** |
| Shared dataloaders | [3117-3140](compare_tsl_liulian.py#L3117-L3140) | Uses `tsl_float64_dataloader` | **CRITICAL** |
| Batch caching | [3142-3145](compare_tsl_liulian.py#L3142-L3145) | `train_batches = list(train_loader)` | **CRITICAL** |
| `train_epoch()` helper | [3150-3165](compare_tsl_liulian.py#L3150-L3165) | Training loop on cached batches | Shared iteration |
| `evaluate()` helper | [3167-3181](compare_tsl_liulian.py#L3167-L3181) | Evaluation loop on cached batches | Shared iteration |
| Training loop | [3184-3199](compare_tsl_liulian.py#L3184-L3199) | Epoch loop with seed reset | Deterministic |
| Test evaluation | [3201-3214](compare_tsl_liulian.py#L3201-L3214) | Final test MSE comparison | Result extraction |

### C. Main Loop Integration

| Component | Line(s) | Description | Impact |
|-----------|---------|-------------|--------|
| Identical mode check | [3734-3735](compare_tsl_liulian.py#L3734-L3735) | `if args.identical:` | Mode branching |
| Call `run_identical_comparison()` | [3739-3744](compare_tsl_liulian.py#L3739-L3744) | Invokes in-process comparison | Core logic |
| Error handling | [3746-3759](compare_tsl_liulian.py#L3746-L3759) | Handles exceptions | Robustness |
| Result formatting | [3760-3786](compare_tsl_liulian.py#L3760-L3786) | Formats identical/not-identical status | Output |
| Results append | [3788-3791](compare_tsl_liulian.py#L3788-L3791) | Adds to results list and continues | Integration |

---

## Table 2: Dependency - tsl_float64_dataloader.py

| Component | Line(s) | Description | Impact |
|-----------|---------|-------------|--------|
| `time_features_from_dates()` | [21-47](tsl_float64_dataloader.py#L21-L47) | Time features with freq support (h/t) | **CRITICAL** |
| `TSLAlignedDataset` class | [50-206](tsl_float64_dataloader.py#L50-L206) | Dataset matching TSL's exact behavior | **CRITICAL** |
| Float64 StandardScaler | [153-154](tsl_float64_dataloader.py#L153-L154) | Uses sklearn default float64 | **CRITICAL** |
| Border calculations | [99-132](tsl_float64_dataloader.py#L99-L132) | Exact TSL train/val/test splits | **CRITICAL** |
| `create_tsl_aligned_dataloader()` | [209-261](tsl_float64_dataloader.py#L209-L261) | Factory function with freq parameter | Entry point |

---

## Table 3: Key Technical Changes (Impact Ranking)

| Rank | Change | Impact | Location |
|------|--------|--------|----------|
| 1 | **Shared batch iteration** | CRITICAL | `compare_tsl_liulian.py:3142-3145` |
| 2 | **Float64 internal dtype** | CRITICAL | `tsl_float64_dataloader.py:153-154` |
| 3 | **Dropout = 0.0** | CRITICAL | `compare_tsl_liulian.py:3092,3106` |
| 4 | **Deterministic CUDA mode** | HIGH | `compare_tsl_liulian.py:3060-3068` |
| 5 | **Time features per freq** | HIGH | `tsl_float64_dataloader.py:31-46` |
| 6 | **Seed order (model→loader)** | HIGH | `compare_tsl_liulian.py:3109-3140` |
| 7 | **Exact border calculations** | CRITICAL | `tsl_float64_dataloader.py:99-132` |

---

## Usage

### Run Identical Comparison

```bash
# Single dataset
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=2021 \
python experiments/adapt_tsl_lib/compare_tsl_liulian.py \
  --identical --identical-epochs 3 --pairs ETTh1_Transformer

# All supported Transformer datasets
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=2021 \
python experiments/adapt_tsl_lib/compare_tsl_liulian.py \
  --identical --identical-epochs 3 \
  --pairs ETTh1_Transformer ETTh2_Transformer ETTm1_Transformer ETTm2_Transformer Weather_Transformer
```

### Environment Variables Required

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Deterministic cuBLAS operations |
| `PYTHONHASHSEED` | `2021` | Deterministic Python hashing |

---

## Supported Models and Datasets

### Currently Supported in `--identical` Mode

| Model | Datasets |
|-------|----------|
| Transformer | ETTh1, ETTh2, ETTm1, ETTm2, Weather |

### Not Yet Supported

Other models (PatchTST, DLinear, iTransformer, etc.) require similar adaptation work:
1. Add to `DATASET_CONFIGS` in `run_identical_comparison()`
2. Import and configure model-specific parameters
3. Test for identical results

---

## Conclusion

The `--identical` flag enables **byte-identical comparison** by:
1. Running both TSL and Liulian models **in the same process**
2. Using **shared dataloaders** with cached batches
3. Applying **deterministic mode** settings
4. Setting **dropout=0.0** to eliminate stochasticity
5. Using **float64 internal dtype** matching TSL's sklearn StandardScaler

This proves that the Liulian Transformer adapter is a **faithful port** of TSL's implementation.
