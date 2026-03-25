# Demo: PatchTST on Swiss River 1990

This demo is a minimal, runnable baseline for LIULIAN forecasting.

## Goal

Run PatchTST on `swiss-river-1990` using the unified pipeline and verify that metrics + artifacts are generated.

## Option A: Run the Python Demo Script

```bash
python examples/forecasting_patchtst_swiss1990_pipeline.py --quick-test
```

Full run:

```bash
python examples/forecasting_patchtst_swiss1990_pipeline.py
```

## Option B: Run Directly from CLI

```bash
liulian run experiments/swiss_river/patchtst_config.yaml --quick_test
```

## What This Uses

- Config preset: `experiments/swiss_river/patchtst_config.yaml`
- Runtime entrypoint: `liulian.pipeline.run_experiment`
- Config merger: `liulian.config.load_config`

## Expected Outputs

At run completion, check the generated `artifacts` directory (path printed in terminal). You should find:

- `results.json`
- `predictions.npz`
- `figures/` plots (if visualization is enabled)

## Validation Checklist

- Process exits with code `0`
- Metrics are printed in terminal
- `results.json` exists and includes `metrics` + `timing`

## Adapt This Demo

Try these incremental changes:

1. Change `pred_len` from `7` to `14`
2. Change `batch_size` from `32` to `16`
3. Toggle `quick_test` off for full training
4. Swap model config to another YAML under `experiments/swiss_river/`
