# Forecasting Pipeline Guide

This guide shows how to run forecasting experiments using the unified pipeline in `liulian/pipeline.py`.

## Why This Pipeline

The forecasting pipeline standardizes the full workflow:

1. Seed and reproducibility setup
2. Dataset construction
3. Model construction
4. Data loader creation
5. Experiment run (train/eval)
6. Artifacts and `results.json` export

Use it when you want consistent behavior across datasets and models.

## Pipeline API Map

Core APIs:

- `liulian.config.load_config`: merge `DEFAULT_CONFIG < YAML < CLI`
- `liulian.pipeline.build_dataset`: create dataset from config
- `liulian.pipeline.build_model`: create model from config
- `liulian.pipeline.build_loaders`: create train/val/test loaders
- `liulian.pipeline.build_experiment`: wire Task + Data + Model + Runtime
- `liulian.pipeline.run_experiment`: run full end-to-end pipeline

Most users should call only `run_experiment(config)`.

## Quick Start (CLI)

Run PatchTST on Swiss River 1990:

```bash
liulian run experiments/swiss_river/patchtst_config.yaml
```

Quick test mode (small, fast smoke run):

```bash
liulian run experiments/swiss_river/patchtst_config.yaml --quick_test
```

Override key hyperparameters from CLI:

```bash
liulian run experiments/swiss_river/patchtst_config.yaml \
  --pred_len 14 \
  --train_epochs 20 \
  --batch_size 16
```

## Quick Start (Python)

```python
from liulian.config import load_config
from liulian.pipeline import run_experiment

cfg = load_config(
    'experiments/swiss_river/patchtst_config.yaml',
    cli_overrides={
        'quick_test': True,
        'hpo': False,
    },
)
summary = run_experiment(cfg)
print(summary.get('artifacts_dir'))
```

## Configuration Flow

The config merge order is:

1. `liulian.config.DEFAULT_CONFIG`
2. YAML config file (for example `experiments/swiss_river/patchtst_config.yaml`)
3. CLI or programmatic overrides

Practical advice:

- Keep experiment presets in YAML files under `experiments/`
- Use CLI overrides for one-off comparisons
- Use `quick_test: true` for sanity checks before long runs

## Artifacts and Results

After each run, the pipeline writes artifacts to an experiment directory (under `artifacts/`), typically including:

- `results.json`: structured summary for auditing and comparison
- `predictions.npz`: raw predictions and ground truth arrays
- `figures/`: generated forecast visualizations (if enabled)

For `results.json` fields, see [Results JSON](results_json.md).

## Common Troubleshooting

### OOM or very slow training

- Reduce `batch_size`
- Reduce model width (`d_model`, `d_ff`, `n_heads`)
- Start with `--quick_test` to validate pipeline integrity first

### Dataset not found

- Confirm dataset key in config `data:` is valid
- The pipeline attempts auto-download for supported datasets

### Unexpected split behavior

- Confirm `split_mode` and `train_split`
- Check whether your model expects `multi_channel` or `per_entity`

## Demo Gallery

For ready-to-run examples, see [Demo Gallery](demo_gallery.md).

## External Design References

These resources were used as style references for clear forecasting quickstarts:

- [PyTorch Forecasting tutorial](https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html)
- [GluonTS quick start](https://ts.gluon.ai/stable/tutorials/forecasting/quick_start_tutorial.html)
- [NeuralForecast quick start](https://nixtlaverse.nixtla.io/neuralforecast/docs/getting-started/quickstart.html)
