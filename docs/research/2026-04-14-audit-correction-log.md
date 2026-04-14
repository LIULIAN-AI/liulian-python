# Audit Correction Log — Entity-Identifier Forecasting Report

_Date: 2026-04-14_
_Audited document: `docs/research/2026-04-10-entity-id-forecasting-report-refined.md`_
_Cross-checked document: `docs/entity_identifiers.md`_
_Auditor method: read each cited file, verify line anchors, check claim content._

## 0. Verdict legend

| Verdict | Meaning |
|---|---|
| **CONFIRMED** | Claim is true; line anchor matches or is within ±2 lines |
| **CONFIRMED-ADJ** | Claim is true, but line anchor needs a minor adjustment |
| **STALE** | Claim was likely true at write-time but no longer matches the code |
| **WRONG** | Claim contradicts the current code |
| **UNVERIFIED** | Cannot be mechanically verified from code alone |

## 1. Dataset matrix claims (report §1 + code anchors §0)

| # | Claim (abbrev.) | Anchor given | Verdict | Evidence / correction |
|---|---|---|---|---|
| 1.1 | Runtime `pipeline.build_dataset` wires CSV/PEMS/Swiss datasets | `pipeline.py:135-271` | **CONFIRMED-ADJ** | `_CSV_DATASET_MAP` starts at L135. `build_dataset` body is L155–272 (not 135–271). Anchor is slightly misleading: L135–152 are the path maps, L155–272 is the function. No content error. |
| 1.2 | `data_factory` registry covers ETT / weather / electricity / traffic / exchange_rate / illness / solar / m4 / custom / PEMS{03,04,07,08} | `data_factory.py:32-54` | **CONFIRMED** | Exact match at L32–54 of `data_factory.py` (verified; `solar`, `m4`, `custom` present as registry-only). |
| 1.3 | Downloader auto-supports traffic/electricity/exchange_rate/weather/illness/ETT*/PEMS*; Swiss is manual | `download.py:61-80`, `208-271` | **CONFIRMED-ADJ** | `_DATASET_FILES` at L61–76 (not 61–80). `_MANUAL_DATASETS` set at L79. `ensure_dataset` body at L208–271 ✓. Content correct. |
| 1.4 | `solar/m4/custom` are registry-only, not runtime-ready via `pipeline.build_dataset` | Implied by §1.1 | **CONFIRMED** | `pipeline._CSV_DATASET_MAP` (L135-145) does **not** contain `solar`, `m4`, `custom`. `pipeline.build_dataset` raises `ValueError` for these unless handled by the `swiss-river` branch. |
| 1.5 | `swiss-river-*` runs via `SwissRiverDataset`, not via `data_factory` | §1.1 | **CONFIRMED** | `pipeline.py:176-201` (`data_name.startswith('swiss-river')` branch) imports `SwissRiverDataset`. No entry in `data_factory.DATASET_REGISTRY`. |
| 1.6 | Plugin adapters (`SwissRiverDatasetAdapter`, `TrafficDatasetAdapter`) are synthetic stubs | `plugins/hydrology/…:7-9`, `plugins/traffic/…:17-19` | **UNVERIFIED** | Not re-verified this pass — deferred to Phase 2; file paths exist per `plugins/` tree shown in project root listing. If incorrect, correction will be logged in Phase 2 cross-check. |
| 1.7 | Weather channels are heterogeneous (table §1.2) | Claim-only | **CONFIRMED** | Matches content of `manifests` / csv column semantics; no code contradiction. |

## 2. Plumbing path claims (report §3)

| # | Claim (abbrev.) | Anchor given | Verdict | Evidence / correction |
|---|---|---|---|---|
| 2.1 | `make_entity_features` returns `None` when `mode='none'` or `'embedding'`, otherwise builds a tiled vector | `timeseriesdataset.py:78-116` | **CONFIRMED** | L87–88 returns `None` for `none`/`embedding`; modes handled at L93–116; tiling `seq_len × D` at L118–119. |
| 2.2 | Entity features only appended when `self.station_name is not None` | `timeseriesdataset.py:1103-1126` | **CONFIRMED** | Exact: `if self.identifier_mode != 'none' and self.station_name is not None:` at L1103. Full block L1103–1126 ✓. |
| 2.3 | Collate returns 6-tuple with `entity_id_strs, entity_idx` when IDs are present | `timeseriesdataset.py:1211-1224` | **CONFIRMED** | Exact match: L1211 `if entity_id_strs is not None:` → L1217–1224 returns the 6-tuple. |
| 2.4 | Trainer forwards `fwd_kwargs['entity_ids']=batch_entity_idx` in embedding mode | `trainer.py:474-479` | **CONFIRMED** | Exact match at L474–479: `if self.use_entity_embedding and batch_entity_idx is not None: fwd_kwargs['entity_ids']=batch_entity_idx`. |
| 2.5 | `pipeline.build_model` wraps with `EntityWrapper` / `ChannelEntityWrapper` for embedding mode, skipping PatchTST+`add_after_patch` | `pipeline.py:441-477` | **CONFIRMED** | L441–444 condition on `identifier_mode=='embedding'` and `id_integration != 'add_after_patch'`. L454–467 `ChannelEntityWrapper` when `split_mode=='multi_channel'`. L468–482 `EntityWrapper` otherwise. |
| 2.6 | PatchTST has native `add_after_patch` path | `patchtst.py:73-89`, `142-153` | **CONFIRMED** | Gate at L73–80 (with `split_mode!='multi_channel'` ValueError); embedding creation L86–88; `_inject_entity_after_patch` at L142–153; uses `torch.arange(n_vars)` channel-order assumption at L151. |
| 2.7 | Channel wrapper uses fixed station buffer `[0..N-1]`, ignores per-sample IDs | Implied | **UNVERIFIED** | Requires separate read of `liulian/models/torch/entity_mixin.py`. Deferred — Phase 2 will verify when writing per-model files for each model using this wrapper. |

## 3. Model surface claims (report §2)

| # | Claim (abbrev.) | Anchor given | Verdict | Evidence / correction |
|---|---|---|---|---|
| 3.1 | Runtime keys: `dlinear, transformer, informer, autoformer, fedformer, itransformer, patchtst, timesnet, timemixer, timexer, mamba, lstm, etsformer, lightts, reformer, gpt4ts, nonstationary_transformer, timellm, timemoe` | Dynamic import in `pipeline.build_model` | **CONFIRMED** | `pipeline.py:427-432` uses `importlib.import_module(f'liulian.models.torch.{model_name}')`. All 19 module names exist as files in `liulian/models/torch/`. |
| 3.2 | Benchmark tool adapter map (`tools/run_benchmark.py`) | `run_benchmark.py:183-217` | **CONFIRMED-ADJ** | Core `adapter_map` dict at L183–198 (14 entries). Optional TimeLLM/TimeMoE appended at L201–212. Anchor 183–217 is slightly wide, but content matches. |
| 3.3 | `LSTMAdapter` exported from `swiss_lstm.py` (not vanilla `lstm` module) | `__init__.py:68-80`, `87-115` | **CONFIRMED** | `from .swiss_lstm import LSTMAdapter, …` at L68–75. Listed in `__all__` at L104. Vanilla `lstm.py` module is imported dynamically by runtime only; there is **no** `VanillaLSTMAdapter` class. |
| 3.4 | Runtime-only models (etsformer / lightts / reformer / gpt4ts / nonstationary_transformer) not in benchmark adapter map | Derived | **CONFIRMED** | Not present in the `run_benchmark.py` adapter_map (verified). |
| 3.5 | `mamba` runtime key resolved via shim re-exporting `mamba_model.Model` | Not specified | **UNVERIFIED** | Needs `liulian/models/torch/mamba.py` file read — deferred to Phase 2 B3 batch. |

## 4. Swiss multi-channel identifier-args omission (report §3.2)

| # | Claim | Anchor given | Verdict | Evidence / correction |
|---|---|---|---|---|
| 4.1 | Per-entity split passes `station_ids, identifier_mode, id_integration, coordinates, station_name` | `swiss_river.py:443-467` | **CONFIRMED** | All five args present in `TimeSeriesDataset(...)` constructor call at L443–466: `station_ids=self.station_ids` (L461), `identifier_mode=self.identifier_mode` (L462), `id_integration=self.id_integration` (L463), `coordinates=...` (L464), `station_name=station` (L465). |
| 4.2 | Multi-channel split (`_build_mc_split`) omits identifier args | `swiss_river.py:492-509` | **CONFIRMED** | `TimeSeriesDataset(...)` at L492–509 passes only data-shape/gap/noise args. No `station_ids`, no `identifier_mode`, no `id_integration`, no `station_name`. → `make_entity_features` path cannot activate for Swiss multi-channel regardless of config, because the child `TimeSeriesDataset` does not know it is entity-aware. |

This is a bona-fide bug with user-facing consequence: a user who sets `identifier_mode='onehot'` (or any transparent mode) together with `split_mode='multi_channel'` for Swiss data will see the config silently have **no effect**. Only the `embedding` mode (which uses the model-side `ChannelEntityWrapper`) still works.

## 5. TimeMoE task-name caveat (report §2.3)

| # | Claim | Anchor given | Verdict | Evidence / correction |
|---|---|---|---|---|
| 5.1 | Adapter defaults `task_name='zero_shot_forecast'` | `timemoe.py:53-56, 73-80` | **CONFIRMED-ADJ** | Default set at L74 inside `TimeMoEAdapter.__init__`'s `default_config` dict. Model's forward-pass check at L53: `if self.task_name == 'zero_shot_forecast':` → else returns `None` (L56). Anchor `53-56` is the check (correct); `73-80` is the adapter init (correct). Content is fine; just note the **default** line is L74 specifically. |
| 5.2 | If task is mismatched, model forward may be inactive for standard long-term task names | L53-56 | **CONFIRMED** | `forward(...)` at L52–56 explicitly returns `None` unless `task_name=='zero_shot_forecast'`. No other task handled. → Any standard TSL task name (`long_term_forecast` etc.) will silently yield `None` predictions. |

## 6. CSV / PEMS set `station_ids` but not `station_name`

| # | Claim | Anchor given | Verdict | Evidence / correction |
|---|---|---|---|---|
| 6.1 | CSV loader sets `kwargs['station_ids'] = feature_cols` when >1 feature but never `station_name` | `csv_dataset.py:289-307` | **CONFIRMED** | L289–291: `if 'station_ids' not in kwargs: if len(feature_cols) > 1: kwargs['station_ids'] = feature_cols`. No `station_name` assignment anywhere in the vicinity. |
| 6.2 | PEMS loader sets `kwargs['station_ids'] = feature_cols` but never `station_name` | `pems_dataset.py:140-157` | **CONFIRMED** | L140–143: `if 'station_ids' not in kwargs: kwargs['station_ids'] = feature_cols`. No `station_name` assignment. |

Combined with finding 4.2: transparent ID modes (`onehot`, `numeric_id`, `sinusoidal`, `coordinates`, `descriptors`, `embedding_idx`) are effectively **no-ops** on standard multi-channel CSV (`traffic`, `electricity`, `weather`, `exchange_rate`, `illness`, ETT*) and PEMS datasets under current code, because neither path sets `station_name`, and `make_entity_features` at `timeseriesdataset.py:1103` is gated on `self.station_name is not None`. Only **`embedding`** mode continues to work for these datasets because it is handled by the model-side `ChannelEntityWrapper` which uses a separate code path (`pipeline.py:454-467`).

## 7. Cross-check: `docs/entity_identifiers.md` (user-facing) vs refined report + code

| # | Location in user-facing doc | Claim | Verdict | Evidence / correction |
|---|---|---|---|---|
| 7.1 | L117 | "All 11 Time-Series-Library model adapters support entity identifiers through the `EntityAwareMixin`" | **CONFIRMED** | Exactly 11 TSL adapters are directly imported at `__init__.py:45-55`: DLinear, PatchTST, iTransformer, Informer, Autoformer, TimesNet, FEDformer, Transformer, TimeMixer, TimeXer, Mamba. |
| 7.2 | L117 | "through the `EntityAwareMixin`" | **WRONG / MISLEADING** | The mixin alone does not make these adapters entity-aware. The actual wrapping happens in `pipeline.build_model` via `EntityWrapper` / `ChannelEntityWrapper` (`pipeline.py:441-477`). PatchTST has a **native** path via `_inject_entity_after_patch`, not via the mixin. User-facing doc overstates the mixin's role. |
| 7.3 | L132–137 | Example config: `'entity_id_col': 0, # Column index in x_mark for entity IDs` | **STALE / CONFUSING** | Current plumbing does not consume `entity_id_col` from config. `TimeSeriesDataset` collate computes `entity_idx` from `entity_id_strs` via the dataset-level `station_ids` mapping (`timeseriesdataset.py:1213-1216`). The `entity_id_col` example key in the doc has no runtime consumer. |
| 7.4 | L186 | "ablation E.2.2 tests all 15 models × 10 entity datasets × 5 modes" | **UNVERIFIED** | Benchmark adapter_map has 14 entries (+2 optional = 16 max). "15" doesn't match an exact current count. Either `E.2.2` is an outdated plan number, or it is a forward-looking spec; either way not derivable from code. |
| 7.5 | L188–197 | Referenced CLI tools (`tools/generate_configs.py`, `tools/run_benchmark.py`, `tools/aggregate_results.py`) | **CONFIRMED** | All three files exist under `tools/`. |
| 7.6 | L102–110 | Mode table lists 7 modes | **PARTIAL** | 7 modes listed: none, embedding, onehot, numeric_id, sinusoidal, coordinates, descriptors. Missing from user-facing doc: `embedding_idx` (defined in `timeseriesdataset.py:78, 93-94`) and `feature_concat` (Swiss adapters). Refined report §3.3 correctly includes these two. |
| 7.7 | L16–65 | Dataset suitability tables | **CONFIRMED** | Matches refined report §1.1 content. |
| 7.8 | (Absent) | No warning about Swiss multi-channel silent no-op | **MISSING** | The finding in 4.2 is not documented in the user-facing doc. A user picking Swiss + multi_channel + any transparent mode will see no effect without explanation. |
| 7.9 | (Absent) | No warning about transparent modes being no-ops on standard CSV/PEMS | **MISSING** | The finding from 6.1 + 6.2 + 4.2 combined: transparent modes do nothing on traffic/electricity/weather/ETT*/PEMS* because `station_name` is never set. Not documented. |

## 8. Line-anchor quality summary

Overall, the refined report's line anchors are tight:

| Anchor quality | Count | Examples |
|---|---|---|
| Exact | 9 | §1.1 `data_factory:32-54`, §3 `timeseriesdataset:1211-1224`, `trainer:474-479`, `patchtst:142-153`, `swiss_river:443-467`, `swiss_river:492-509`, `timeseriesdataset:1103-1126`, `patchtst:73-89`, `pipeline:441-477` |
| ±2 lines | 2 | `download:61-80` (actual 61–76 + 79), `run_benchmark:183-217` (actual 183–198 core + 201–212 optional) |
| Misleading (function-wide vs feature) | 1 | `pipeline:135-271` (the contents of interest are 135–152 map + 155–272 function) |
| Unverified in this audit | 3 | Plugin adapters, `channel_entity_wrapper` buffer, `mamba` shim |

## 9. Summary of proposed corrections

To be merged into `2026-04-10-entity-id-forecasting-report-refined.md` as dated inline notes in Phase 3:

1. **§1.1 anchor clarification** — change `pipeline.py:135-271` to `pipeline.py:135-152 (maps) + 155-272 (build_dataset)`.
2. **§0 anchor #3 clarification** — change `download.py:61-80` to `download.py:61-76 (_DATASET_FILES) + 79 (_MANUAL_DATASETS)`.
3. **§2.2 anchor clarification** — `run_benchmark.py:183-198 (core map) + 201-212 (optional)` instead of `183-217`.
4. **§2.3 anchor precision** — note that TimeMoE default comes from `default_config = {'task_name': 'zero_shot_forecast'}` at `timemoe.py:74`, checked against `self.task_name` at L53, returns `None` otherwise at L56.

To be added to `docs/entity_identifiers.md` in Phase 3:

A. **Warning block on silent no-ops** — add a "⚠️ Known limitations" section covering:
   - Transparent ID modes (onehot / numeric_id / sinusoidal / coordinates / descriptors / embedding_idx) currently have no effect on `traffic`, `electricity`, `weather`, `exchange_rate`, `illness`, `ETT*`, `PEMS0{3,4,7,8}` because CSV/PEMS loaders do not set `station_name` (see `csv_dataset.py:289-291`, `pems_dataset.py:140-143`, `timeseriesdataset.py:1103`).
   - The Swiss multi-channel split (`swiss_river.py:472-511`) does not propagate `identifier_mode` / `id_integration` / `station_ids` / `station_name` into its internal `TimeSeriesDataset`, so the same no-op risk applies.
   - Embedding mode is unaffected — it flows through `ChannelEntityWrapper` / `EntityWrapper` / `PatchTST.add_after_patch`.
B. **Correct the `EntityAwareMixin` attribution** at L117 — mechanism is `EntityWrapper` / `ChannelEntityWrapper` wrapping in `pipeline.build_model`, not the mixin.
C. **Remove or clarify `entity_id_col`** at L132–137 — no runtime consumer; `entity_idx` is computed from `station_ids` + `entity_id_strs` in the collate.
D. **Document the two extra modes** `embedding_idx` and `feature_concat` in the mode table.

## 10. Gate — stop here (Phase 1 G3)

Phase 1 complete. **Awaiting user approval before starting Phase 2 (per-model deep dives)**.

Open items deferred to Phase 2:
- 1.6 Plugin adapters stubs (will check `plugins/hydrology/…` and `plugins/traffic/…`).
- 2.7 `ChannelEntityWrapper` / `EntityWrapper` fixed-buffer behavior (will read `entity_mixin.py` when writing transformer / dlinear / autoformer files).
- 3.5 `mamba.py` shim (will read when writing `mamba.md`).

No substantive claim in the refined report was found to be **WRONG** on the code side. The user-facing `docs/entity_identifiers.md` has several inaccuracies (7.2, 7.3) and gaps (7.8, 7.9) that Phase 3 will fix.
