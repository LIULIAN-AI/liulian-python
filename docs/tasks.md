# Task Ledger

Living record of **completed / in-progress / future** work on the ML core
(`liulian-python`). Newest first inside each section. Every entry names its
artifacts (commits, files, run-tags) so it can be audited later. Agent
session task IDs (`#N`) are kept for traceability.

> Federation-level (platform / UI / ops) work is tracked in the
> `liulian-docs` hub repo — see `docs/strategy/MOVED.md`.
>
> **Convention:** whenever a work item finishes or a new one is decided,
> update this file in the same commit/PR.

## In progress

- **swiss3dt dropout-tuned baselines** (#30) — swiss-river-{1990, 2010,
  zurich} × lstm × {none, embedding}; dropout tuned `{0, 0.1}`
  (reference LSTM has none); per-trial Ray seeding active. Jobs
  5874880-82, run-tags `swiss3dt-*-20260612`. Remaining identifier modes
  (onehot / coordinates / sinusoidal / random) await scheme review.

## Future

- **Nowcast decoding scheme** (#31) — add a per-step decoding head to
  `lstm.py` for `task='nowcast'` (benchmark `LstmModel` style: head on
  every LSTM output, full-sequence masked loss); audit the other adapters
  (DMS models may need a `pred_len=seq_len` mode); optional per-step
  auxiliary loss for the forecast setting. Details:
  `docs/research/entity-id-deep/lstm.md` §9 + inline `TODO(nowcast)` in
  `liulian/models/torch/lstm.py`. **Do BEFORE any nowcast experiment.**
- **multi_channel coordinates injection** (#28) — pipeline never injects
  `config['coordinates']` / `config['station_ids']` for
  `ChannelTransparentWrapper`, so coordinates are offered only for
  swiss × lstm (per_entity). Wire the injection, re-offer the
  patchtst/dlinear pairs, rerun the invalidated swiss-river-1990 cells.
  Also needs a NaN-masking story before any 2010/zurich multi_channel run.
- **Invalidate pre-2026-06-11 swiss coordinates results in research docs**
  (#29) — those cells ran zero-vector identifiers (see Completed
  2026-06-11); mark them invalid in the 2026-05-15 slide / analysis /
  heatmap figures and replace with swiss3 reruns.
- **Case 2: outer-sweep param-analysis entry point** (#26) — sweep a
  transparent dim (e.g. `sinusoidal_dim`) while FIXING inner HPO to the
  swept value; the only way to study these dims on per_entity swiss-lstm
  (their features are frozen into the data loaders per run). Separate
  entry point, not the matrix runner.
- **sin/random identifier backfill for traffic + electricity** (#17) —
  deferred cells of the 2026-05 baseline matrix.
- **`hpo_local_mode` cleanup** — current Ray removed `local_mode`; the
  config knob is dead and ~56 e2e tests fail on it
  (`ray_optimizer.py:586`). Migrate the tests, drop the knob.
- **trainer.py predict-aggregate bug** — `TypeError` at
  `trainer.py:623` (`sum(all_entity_ids, [])` with tuples); pre-existing,
  one e2e test fails. Fix is a one-liner once approved.
- **results.json `model.*` shows pre-HPO defaults** — cosmetic: the model
  section reflects the initial config, while tuned values are in
  `hpo.best_hparams` (and `best_hparams.yaml`). Align when convenient.
- **Repo hygiene** — stray `ls` file at repo root (committed by mistake);
  root-level demo screenshots; decide what stays.

## Completed

### 2026-06-12

- **results.json `hpo.best_hparams` populated** — producer alias +
  builder fallback + regression tests (`tests/runtime/test_results_json.py`);
  the three completed swiss3 none runs backfilled in place from
  `best_hparams.yaml` (commit `b3acc80`).
- **Per-trial Ray seeding** — `_trainable` seeds
  `seed + trial_index`; trials are reproducible and differ only through
  sampled hyperparameters (commit `edec5d0`). Neither reference project
  (TSL, swiss benchmark) seeds trials; neither does CV — both select on a
  single fixed chronological hold-out, as we do.
- **Dropout for swiss lstm: tuned, not assumed** — reference `LstmModel`
  has no dropout; ours applied 0.1 silently. Now HPO chooses `{0, 0.1}`;
  config default 0.0.
- **LSTM decoding-scheme comment corrected** (commit `128fe03`) — ours is
  direct multi-step (TSL contract; = benchmark `ExtrapoLstmModel('limo')`);
  the benchmark's plain `LstmModel` decodes per time step and is NOT
  truncated; produce-then-truncate is TSL's exp loop.
- **Branch reorg** — `feat/platform-upgrade-2026-05` merged to `main`
  (merge `a3dc2f9`) and deleted (local + remote); new branches:
  `feat/update-2026-06` (platform), `exp/entity-identifier-2026-06`
  (current working branch).

### 2026-06-11

- **swiss3 none baselines on UBELIX (gratis)** — verified genuine
  (50 trials, 30 epochs, batch 32 fixed, no caps). Test denorm RMSE:
  1990 = 1.669 °C, 2010 = 1.646 °C, zurich = 1.562 °C. Run-tags
  `swiss3-*-20260611` (kept for old-vs-new dropout comparison).
- **Coordinates identifier fixed** (commit `421f1e7`) — was silently
  fake: with `graph_mode='none'` the topology never loaded and
  `make_entity_features` zero-filled → every pre-fix swiss coordinates
  cell measured "2 constant zero channels". Now: topology loads for
  coordinates mode, coords min-max normalized per dataset, missing
  coords RAISE (no silent zeros).
- **Per-station NaN drop** (commit `421f1e7`) — swiss-river-2010/zurich
  CSVs carry NaNs (stations join/leave the network); a single NaN sample
  drove the per_entity loss to NaN from epoch 1. Dropped per station;
  segment breaks keep windows from spanning gaps; no-op for 1990.
- **swiss-river-2010 / zurich registered** in the matrix; per-experiment
  launcher `experiments/entity_identifier/run_job.py`; swiss lstm
  `batch_size` 8 → 32 (fixed, not HPO-tuned) (commit `e851ca6`).
- **HPO search spaces externalized** to `liulian/optim/search_spaces.yaml`
  (mode-aware identifier params; front-end editable). Fixed the dead-knob
  bug where lstm always tuned `embedding_size` even for `none`; per-trial
  `ChannelTransparentWrapper` rebuild makes sinusoidal/random dims real
  knobs in multi_channel (commit `ccba987`).

### 2026-05 (baseline matrix — summary)

- Entity-identifier baseline matrix on UBELIX: {swiss-river-1990,
  traffic, electricity} × {lstm, patchtst, dlinear} × identifier modes;
  UBELIX tier setup + cost tracking (paygo 10 CHF cap,
  `jobs/ubelix_cost_tracker.py`); advisor progress slide and pred-true
  aggregation analysis (`docs/research/2026-05-15-*`).
  Caveat discovered 2026-06-11: the swiss *coordinates* cells of this
  matrix are invalid (zero-vector identifiers, see above).
