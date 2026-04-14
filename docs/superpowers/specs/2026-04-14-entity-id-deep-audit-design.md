# Entity-Identifier Deep Audit + Per-Model Research — Design Spec

_Date: 2026-04-14_  
_Status: Approved (user confirmed 2026-04-14)_  
_Type: Research / documentation (no code changes to `liulian/**`)_

## 1. Purpose

1. **Audit** every substantive claim in
   `docs/research/2026-04-10-entity-id-forecasting-report-refined.md`
   against the current repository state.
2. **Deepen** the per-model analysis from architectural prose
   (the refined report's current depth) down to
   *this-repo file + function + line-anchored* design recommendations,
   cross-referenced with each model's paper and official implementation.

Out of scope: changing any code under `liulian/**`, `plugins/**`, or
`experiments/**`. This project writes documentation only.

## 2. Inputs

| Input | Path |
|---|---|
| Prior refined report | `docs/research/2026-04-10-entity-id-forecasting-report-refined.md` |
| User-facing entity doc | `docs/entity_identifiers.md` |
| Runtime pipeline | `liulian/pipeline.py` |
| Data factory | `liulian/data/data_factory.py` |
| Data download surface | `liulian/data/download.py` |
| Core dataset class | `liulian/data/ts/timeseriesdataset.py` |
| CSV / PEMS loaders | `liulian/data/csv_dataset.py`, `liulian/data/pems_dataset.py` |
| Swiss loader | `liulian/data/swiss_river.py` |
| Runtime trainer | `liulian/runtime/trainer.py` |
| Model adapters | `liulian/models/torch/*.py` |
| Benchmark tool | `tools/run_benchmark.py` |
| Entity mixin | `liulian/models/torch/entity_mixin.py` |

## 3. Deliverables

```
docs/research/
├── 2026-04-10-entity-id-forecasting-report-refined.md   # edited: dated correction notes inline
├── 2026-04-14-audit-correction-log.md                    # Phase 1 output
└── entity-id-deep/
    ├── README.md                                          # index + cross-cutting summary
    ├── _template.md                                       # per-model schema
    ├── dlinear.md
    ├── transformer.md
    ├── informer.md
    ├── autoformer.md
    ├── fedformer.md
    ├── itransformer.md
    ├── patchtst.md
    ├── timesnet.md
    ├── timemixer.md
    ├── timexer.md
    ├── mamba.md
    ├── lstm.md
    ├── etsformer.md
    ├── lightts.md
    ├── reformer.md
    ├── gpt4ts.md
    ├── nonstationary_transformer.md
    ├── timellm.md
    ├── timemoe.md
    ├── swiss_lstm_family.md
    └── swiss_transformer_family.md
```

## 4. Per-model file schema

Each of the 21 per-model files must contain these sections in this order:

1. **Identity & provenance** — canonical name, paper URL, official repo URL
   with commit/tag where possible, this-repo adapter path.
2. **Architecture primer** — mechanism summary, forward-pass shape flow,
   one small ASCII diagram.
3. **This-repo audit** — what `liulian/models/torch/<model>.py` currently does
   for entity IDs (wrapper vs native); caveats surfaced by reading the code.
4. **Upstream reference** — file:line (or function) in the official repo where
   entity injection would hook; note what shapes/tensors are available there.
5. **Proposed ID injection design** — specific injection point(s) with
   architecture-grounded rationale; list alternatives considered and why rejected.
6. **Concrete code change sketch** — in this repo's existing adapter, where
   (file + function) the change goes; skeleton pseudocode or micro-diff only
   (no final implementation code).
7. **Feasibility & risks** — supports-after-revision verdict, parity tests
   required, known failure modes (channel-order, task-name, etc.).
8. **Citations & uncertainty** — paper link, repo link, this-repo line anchors,
   any external ablation evidence (STID, AGCRN, etc.), plus a
   "what I'm uncertain about" block.

## 5. Phases

### Phase 1 — AUDIT (gate here)

Claim-by-claim verification of the refined report using the current repo:

- A1 Dataset matrix (report §1.1) — runtime / factory / downloader / experiments columns.
- A2 Plumbing path (report §3) — `make_entity_features` gating, collate emit,
  trainer forwarding, wrapper paths, patchtst `add_after_patch`.
- A3 Model surfaces (report §2) — runtime keys, benchmark keys, alias exports.
- A4 Swiss multi-channel identifier-args omission claim (§3.2).
- A5 TimeMoE `task_name='zero_shot_forecast'` caveat (§2.3).
- A6 Cross-check vs `docs/entity_identifiers.md` — flag any inconsistency.

**Phase 1 output:** `docs/research/2026-04-14-audit-correction-log.md`.
Stop and wait for approval before Phase 2.

### Phase 2 — PER-MODEL DEEP DIVES (post-gate)

21 model files, batched 5 at a time:

- Batch B1: DLinear, Transformer, Informer, Autoformer, FEDformer
- Batch B2: iTransformer, PatchTST, TimesNet, TimeMixer, TimeXer
- Batch B3: Mamba, LSTM, ETSformer, LightTS, Reformer
- Batch B4: GPT4TS, NST, TimeLLM, TimeMoE, Swiss-LSTM-family, Swiss-Transformer-family

For each model: one `Read` of this-repo adapter + up to two `WebFetch`
calls (paper + official repo) + write of the per-model file + append index
row to `README.md`.

### Phase 3 — ANNOTATE + SUMMARIZE

- Edit prior refined report in place, adding dated corrections where Phase 1
  found issues (preserve originals).
- Write `entity-id-deep/README.md`: 21-row cross-cutting matrix (injection
  class, invasiveness, parity tests needed, paper evidence strength,
  wrapper-vs-native recommendation) + lessons section.

### Phase 4 — VERIFICATION

Self-check table mapping each item in the user's original ask to the
deliverable section that fulfills it; no-miss coverage tables for datasets
and models, mirroring the existing report's §8 format.

## 6. Ground rules

- No citation without a link or `file:line` anchor.
- On failed WebFetch: retry once, then write `[fetch-failed]` and continue;
  never fabricate code from external repos.
- Each per-model file must end with an explicit "uncertainties" block.
- No edits to `liulian/**`, `plugins/**`, `experiments/**` during this project.

## 7. Gating

- **G3 (audit-first gate)**: stop after Phase 1 for user review.
- After Phase 1 approval, run Phases 2 → 3 → 4 to completion without
  intermediate gates.

## 8. Estimated effort

| Phase | Reads | WebFetches | Writes | Time |
|---|---|---|---|---|
| 1 Audit | ~15 | 0 | 1 | 30–45 min |
| 2 Deep dives | ~21 | ~40 | 21 | 2–4 hours |
| 3 Annotate | 2 | 0 | 1 | ~15 min |
| 4 Verify | — | 0 | — | ~10 min |

## 9. Out of scope

- Running experiments or benchmarks.
- Editing model adapters, wrappers, or trainer.
- Publishing to mkdocs site (we write files; docs build is separate).
- Proposing changes to datasets beyond documenting current entity-ID wiring state.
