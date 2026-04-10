# Entity Identifier Forecasting Design (Embedding First, Then Onehot)

**Date:** 2026-04-10  
**Status:** Approved for implementation

## 1. Problem Statement

Implement end-to-end entity-identifier support and experiment orchestration for the approved forecasting scope:

- **Models (in order):** `lstm`, `transformer`, `dlinear`, `patchtst`, `itransformer`, `timellm`, `gpt4ts`
- **Datasets (in order):** `swiss-river-1990`, `swiss-river-2010`, `swiss-river-zurich`, `traffic`, `electricity`, `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`, `exchange_rate`
- **Identifier settings:** `none`, `embedding` (first), then `onehot`

Comparison experiments must be runnable from a new entrypoint family in `experiments/entity_identifier/`, support local smoke/dev execution, and support full-matrix cluster runs via Slurm.

## 2. Confirmed Constraints

1. Swiss River comparison mode is **`per_entity` only**.
2. Build **full matrix configs** for all listed models/datasets/identifier settings.
3. Run **local smoke subset** only; full runs are for cluster.
4. For TimeLLM/GPT4TS local smoke, fallback to config/registry validation is allowed if weights are unavailable, with clear instructions for real runs.
5. Slurm launcher style should align with `jobs/run_jobs_ray_tune.py`.
6. If a model already has correct identifier behavior, keep it and avoid unnecessary rewrites.

## 3. Design Overview

### 3.1 Model-layer strategy (hybrid native + fallback)

Use existing correct pathways where available, and add targeted native hooks where missing or required:

- **`patchtst`**: keep native `identifier_mode=embedding` + `id_integration=add_after_patch` for multi-channel embedding.
- **`transformer`, `dlinear`, `itransformer`, `timellm`**: retain current `EntityAwareMixin`/wrapper path as compatibility baseline; add focused native hooks only where needed to match section-4.1 behavior without breaking existing configs.
- **`lstm`, `gpt4ts`**: add missing adapter/entity integration support so they can participate in the same identifier experiment matrix.
- **Onehot phase**: rely on transparent data-layer feature concatenation (`identifier_mode=onehot`, `id_integration=concat_to_x`) and correct `enc_in` propagation.

### 3.2 Dataset-layer strategy

- Preserve Swiss River `per_entity` identity behavior.
- Ensure requested datasets have consistent identifier metadata propagation for experiment execution:
  - `num_embeddings` resolution from dataset station/sensor IDs
  - correct `identifier_mode` + `id_integration` behavior in dataset construction path
  - no regressions for `identifier_mode=none`

### 3.3 Experiment orchestration strategy

Create `experiments/entity_identifier/` with:

1. **Config generation**
   - full matrix generation over models × datasets × identifier settings
   - deterministic naming convention and output layout
2. **Runner entrypoint**
   - wraps existing `experiments/run.py`/pipeline behavior
   - supports:
     - dry-run (print commands/configs only)
     - local smoke subset execution
     - full matrix execution flags (for cluster)
3. **Comparison aggregation**
   - collect metrics per run into a unified results table (`none` vs `embedding` vs `onehot`)
   - summarize by model, dataset, and identifier mode

### 3.4 Cluster execution strategy

Add a Slurm submit helper in `experiments/entity_identifier/` patterned after `jobs/run_jobs_ray_tune.py`:

- sbatch script generation
- CPU/GPU profile switching
- configurable job resources
- per-job output/error paths
- matrix submission mode for full runs

## 4. Detailed Component Design

### 4.1 Code surfaces to change

- **Models/adapters**
  - `liulian/models/torch/lstm.py`
  - `liulian/models/torch/transformer.py`
  - `liulian/models/torch/dlinear.py`
  - `liulian/models/torch/patchtst.py`
  - `liulian/models/torch/itransformer.py`
  - `liulian/models/torch/timellm.py`
  - `liulian/models/torch/gpt4ts.py`
  - shared wrapper/helper files only where required
- **Dataset/pipeline glue**
  - dataset constructors and identifier propagation paths
  - runtime model build path if required for consistent matrix behavior
- **Experiment package**
  - new folder: `experiments/entity_identifier/`
  - generator, runner, compare, and slurm submit scripts

### 4.2 Experiment outputs

- machine-readable run summary (CSV/JSON)
- compact markdown/table summary for quick inspection
- logs grouped by model/dataset/identifier mode

### 4.3 TimeLLM/GPT4TS fallback behavior

When local prerequisites are missing:

- do not hard-fail full matrix generation
- mark runs as `skipped_missing_weights` in output summaries
- print explicit commands to download/cache required models
- include documented rerun instructions

## 5. Error Handling and Safety

1. Fail fast on invalid identifier mode/model combinations.
2. Keep backward compatibility for existing non-entity experiments.
3. Avoid silent behavior changes in default paths.
4. Preserve `identifier_mode=none` parity.

## 6. Validation Plan

1. **Dry checks**
   - matrix generation dry-run
   - runner command rendering
2. **Local smoke/dev**
   - representative subset across requested models/datasets/modes
   - include at least one Swiss River run and one multi-channel benchmark run
3. **Repo test pass**
   - run existing relevant tests plus targeted new tests for new logic
4. **Cluster readiness**
   - validate sbatch script generation and submission command paths

## 7. Out of Scope

- Full long-horizon training of entire matrix on local machine.
- Unrequested identifier modes (`numeric_id`, `sinusoidal`, `coordinates`, `descriptors`) in this implementation wave.
- Unrequested models/datasets beyond the confirmed scope.

## 8. Implementation Order

1. Embedding support across target models and dataset wiring.
2. Onehot support and transparent-path validation.
3. New `experiments/entity_identifier/` generator/runner/compare scripts.
4. Slurm submission utility aligned to existing job style.
5. Dry/smoke/dev validation and final detailed report.
