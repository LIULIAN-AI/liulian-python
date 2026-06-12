# Entity Identifier Forecasting Research Report (Refined, All-Surface Audit)

_Date: 2026-04-10 (refined pass)_

This refined report extends and corrects the prior report by auditing **all possible dataset/model surfaces in this repository**, not only the active runtime subset.

---

## 0. Scope expansion and correction summary

| Area | Prior report | Refined report |
|---|---|---|
| Dataset surfaces | Runtime `pipeline.build_dataset` + `data_factory` | Runtime + `data_factory` + downloader surface + experiment usage + plugin adapters |
| Model surfaces | Runtime module models + Swiss adapters | Runtime modules + benchmark adapter map + export aliases + config generator naming |
| Naming reconciliation | Partial | Full lower-case runtime key ↔ CamelCase benchmark key map |
| Gap log | Limited | Explicit discrepancy table with omission / semantics / naming mismatches |
| Completion verification | Requirement-level | Requirement-level + no-miss dataset/model coverage tables |

### Core evidence anchors used in this refinement

1. Runtime dataset/model wiring: `liulian/pipeline.py:135-271`, `liulian/pipeline.py:342-487`  
2. Data-factory-only datasets: `liulian/data/data_factory.py:32-54`  
3. Downloader-supported dataset list/manual Swiss list: `liulian/data/download.py:61-80`, `liulian/data/download.py:208-271`  
4. Plugin adapters are synthetic stubs: `plugins/hydrology/swissriver_adapter.py:7-9`, `plugins/traffic/adapter.py:17-19`  
5. Entity feature creation requires `station_name`: `liulian/data/ts/timeseriesdataset.py:1103-1126`  
6. CSV/PEMS paths set `station_ids` but not `station_name`: `liulian/data/csv_dataset.py:289-307`, `liulian/data/pems_dataset.py:140-157`  
7. Swiss per-entity path sets `station_name`; Swiss multi-channel path omits identifier args: `liulian/data/swiss_river.py:443-467`, `liulian/data/swiss_river.py:492-509`  
8. Collate emits `entity_idx`; trainer forwards `entity_ids`: `liulian/data/ts/timeseriesdataset.py:1211-1224`, `liulian/runtime/trainer.py:474-479`  
9. Wrapper paths and PatchTST exception: `liulian/pipeline.py:441-477`, `liulian/models/torch/patchtst.py:73-89`, `liulian/models/torch/patchtst.py:142-153`  
10. Benchmark adapter surface and naming: `tools/run_benchmark.py:183-217`  
11. Export aliasing (notably `LSTMAdapter` from Swiss file): `liulian/models/torch/__init__.py:68-80`, `liulian/models/torch/__init__.py:87-115`  
12. Identifier mode definitions in data layer: `liulian/data/ts/timeseriesdataset.py:78-116`  
13. Swiss `feature_concat` mode in adapters: `liulian/models/torch/swiss_lstm.py:307-330`, `liulian/models/torch/swiss_transformer.py:18-23`, `liulian/models/torch/swiss_transformer.py:87-96`

---

## 1. Full dataset inventory (all repository surfaces)

### 1.1 Canonical dataset matrix

| Dataset key / class | Runtime `build_dataset` | `data_factory` | Downloader | Experiments YAML | Entity suitability | Current wiring verdict |
|---|---|---|---|---|---|---|
| `traffic` | Yes | Yes | Auto | Yes | **Strong** (homogeneous sensors) | Embedding path works; transparent modes in multi-channel are at risk of no-op unless `station_name` is set |
| `electricity` | Yes | Yes | Auto | Yes | **Strong** (homogeneous clients/meters) | Same as traffic |
| `PEMS03` | Yes | Yes | Auto | Yes | **Strong** (sensor entities) | Embedding path works; PEMS loader currently uses first feature channel only |
| `PEMS04` | Yes | Yes | Auto | No | **Strong** | Same as PEMS03 |
| `PEMS07` | Yes | Yes | Auto | No | **Strong** | Same as PEMS03 |
| `PEMS08` | Yes | Yes | Auto | No | **Strong** | Same as PEMS03 |
| `swiss-river-1990` | Yes | No | Manual | Yes | **Strong** (station entities) | Per-entity split supports full ID modes; multi-channel split does not pass identifier args at data layer |
| `swiss-river-2010` | Yes | No | Manual | No | **Strong** | Same as swiss-river-1990 |
| `swiss-river-zurich` | Yes | No | Manual | No | **Strong** | Same as swiss-river-1990 |
| `exchange_rate` | Yes | Yes | Auto | Yes | **Conditional** (few channels, weakly homogeneous) | Technically feasible; gains may be limited/unstable |
| `ETTh1` | Yes | Yes | Auto | Yes | **Not recommended** | Channels are heterogeneous physical variables, not entities |
| `ETTh2` | Yes | Yes | Auto | Yes | **Not recommended** | Same as ETTh1 |
| `ETTm1` | Yes | Yes | Auto | Yes | **Not recommended** | Same as ETTh1 |
| `ETTm2` | Yes | Yes | Auto | Yes | **Not recommended** | Same as ETTh1 |
| `weather` | Yes | Yes | Auto | Yes | **Not recommended** | Meteorological channels are heterogeneous variable dimensions |
| `illness` | Yes | Yes | Auto | Yes | **Not recommended** | Epidemiology indicators are heterogeneous statistics |
| `solar` | No | Yes | No | No | **Strong** if configured as plant-wise homogeneous channels | Not runnable through main runtime path until `pipeline.build_dataset` adds key |
| `m4` | No | Yes | No | No | **No** (mostly univariate competition series) | Registry-only; no strong entity-ID case |
| `custom` | No | Yes | No | No | **Depends on schema** | Registry-only; suitability is user-schema dependent |
| `SwissRiverDatasetAdapter` (plugin) | No | No | N/A | No | **Conceptually strong** | Plugin adapter currently synthetic/stub, not a full real-data pipeline |
| `TrafficDatasetAdapter` (plugin) | No | No | N/A | No | **Conceptually strong** | Plugin adapter currently synthetic/stub |

### 1.2 Weather variable-level audit (why IDs are semantically weak here)

| Variable cluster | Example columns | Interchangeable entity channels? | Verdict |
|---|---|---|---|
| Thermodynamic | `T (degC)`, `Tpot (K)`, `Tdew (degC)` | No | Feature semantics differ |
| Moisture/pressure | `p (mbar)`, `rh (%)`, `VPmax`, `H2OC` | No | Units and meanings differ |
| Wind | `wv (m/s)`, `max. wv (m/s)`, `wd (deg)` | No | Distinct physics |
| Rain/radiation | `rain (mm)`, `SWDR`, `PAR` | No | Distinct processes |

**Conclusion:** Weather channels should remain typed features, not entity IDs.

---

## 2. Full forecasting model inventory + naming reconciliation

### 2.1 Runtime module models (dynamic import surface)

`pipeline.build_model` dynamically imports `liulian.models.torch.<model>`, so runtime keys are module names:  
`dlinear, transformer, informer, autoformer, fedformer, itransformer, patchtst, timesnet, timemixer, timexer, mamba, lstm, etsformer, lightts, reformer, gpt4ts, nonstationary_transformer, timellm, timemoe`  
(with `mamba` resolved by shim `mamba.py` re-exporting `mamba_model.Model`).

### 2.2 Benchmark/config surfaces and alias reconciliation

| Canonical architecture | Runtime key | Benchmark key (`run_benchmark.py`) | Export/alias notes |
|---|---|---|---|
| DLinear | `dlinear` | `DLinear` | Direct map |
| Transformer | `transformer` | `Transformer` | Direct map |
| Informer | `informer` | `Informer` | Direct map |
| Autoformer | `autoformer` | `Autoformer` | Direct map |
| FEDformer | `fedformer` | `FEDformer` | Direct map |
| iTransformer | `itransformer` | `iTransformer` | Direct map |
| PatchTST | `patchtst` | `PatchTST` | Direct map |
| TimesNet | `timesnet` | `TimesNet` | Direct map |
| TimeMixer | `timemixer` | `TimeMixer` | Direct map |
| TimeXer | `timexer` | `TimeXer` | Direct map |
| Mamba | `mamba` | `Mamba` | Runtime via `mamba.py` shim |
| TimeLLM | `timellm` | `TimeLLM` (conditional add) | Optional dependency path |
| TimeMoE | `timemoe` | `TimeMoE` (conditional add) | Optional dependency path; task mode caveat (below) |
| Vanilla LSTM module | `lstm` | _(none)_ | Runtime-only module |
| ETSformer | `etsformer` | _(none)_ | Runtime-only module |
| LightTS | `lightts` | _(none)_ | Runtime-only module |
| Reformer | `reformer` | _(none)_ | Runtime-only module |
| GPT4TS | `gpt4ts` | _(none)_ | Runtime-only module |
| Nonstationary Transformer | `nonstationary_transformer` | _(none)_ | Runtime-only module |
| Swiss LSTM family | _(not runtime key)_ | `LSTMAdapter`, `ExtrapoLSTMAdapter` | `LSTMAdapter` exported from `swiss_lstm.py` |
| Swiss Transformer family | _(not runtime key)_ | `TransformerEncoderAdapter` | Alias exports include `SwissTransformerAdapter`, `SwissTransformerEmbeddingAdapter` |

### 2.3 Important practical caveat

`TimeMoE` adapter defaults to `task_name='zero_shot_forecast'`; if config does not align, model forward may be inactive for standard long-term task names (`liulian/models/torch/timemoe.py:53-56`, `liulian/models/torch/timemoe.py:73-80`).

---

## 3. Corrected entity plumbing path (data -> trainer -> model)

| Stage | What happens | Key risk |
|---|---|---|
| Data feature build | `make_entity_features` only executes when `station_name` is set | In many multi-channel CSV/PEMS paths, `station_name` is unset, so transparent modes can no-op |
| Split/entity IDs | `seg_entity_ids` only set when `station_name` exists | Batch-level `entity_idx` missing for many multi-channel datasets |
| Collate | Returns `(batch_x, batch_y, batch_x_mark, batch_y_mark, entity_id_strs, entity_idx)` when IDs exist | Missing IDs reduce per-sample embedding path availability |
| Trainer | For embedding mode, forwards `entity_ids=batch_entity_idx` to model | Works only when collate emits `entity_idx` |
| Model wrapping | `pipeline.build_model` wraps with `EntityWrapper` / `ChannelEntityWrapper` for embedding mode (except PatchTST add-after-patch) | Generic wrappers may be less architecture-optimal than native injection |
| PatchTST native path | `add_after_patch` adds station embeddings in patch-token space | Assumes channel order via `arange(n_vars)` |
| Channel wrapper path | Uses fixed station index buffer `[0..N-1]` and ignores per-sample IDs | Channel permutation assumptions must stay consistent |

### 3.1 Where it currently works best

1. **Embedding mode + multi-channel homogeneous datasets** (traffic/electricity/PEMS) via `ChannelEntityWrapper` or PatchTST post-patch embeddings.  
2. **Swiss per-entity mode** where `station_name` is explicitly set and entity strings/indices are emitted in collate.

### 3.2 Where behavior is fragile

1. Transparent modes on many multi-channel CSV/PEMS paths.  
2. Swiss multi-channel split currently does not pass identifier args into the constructed internal `TimeSeriesDataset` object (`liulian/data/swiss_river.py:492-509`).

### 3.3 Entity identifier forms currently considered in the codebase

| Identifier form | Implementation surface | Notes |
|---|---|---|
| `embedding` | `TimeSeriesDataset` + wrappers/adapters | Learned `nn.Embedding`; primary path for homogeneous multi-entity datasets |
| `onehot` | `make_entity_features()` | Transparent feature concat mode |
| `numeric_id` | `make_entity_features()` | Transparent scalar ID feature |
| `sinusoidal` | `make_entity_features()` | Transparent positional-style encoding |
| `coordinates` | `make_entity_features()` | Transparent geo feature (mainly Swiss/topology-aware datasets) |
| `descriptors` | `make_entity_features()` | Transparent user-provided descriptor vectors |
| `embedding_idx` | `make_entity_features()` | Explicit index feature variant (rarely used in configs) |
| `feature_concat` | Swiss adapters (`swiss_lstm.py`, `swiss_transformer.py`) | Separate `entity_features` tensor path in adapter forward |
| `entity_id_strs` / `entity_idx` batch metadata | DataLoader collate + trainer | String IDs for inverse-scaling/reporting; integer IDs for model embedding lookup |

---

## 4. Per-model feasibility and architecture-specific entity-ID design (all forecasting models)

### 4.1 Feasibility matrix

| Model (canonical) | Current support route | Support after revision? | Architecture-specific ID design | Why this design |
|---|---|---|---|---|
| dlinear | Adapter + mixin / runtime wrapper | **Yes** | Station-conditioned bias/affine on trend+season outputs (keep wrapper baseline) | Linear decomposition benefits from low-cost per-entity offsets |
| transformer | Adapter + mixin | **Yes** | Additive entity embedding at token embedding or dedicated entity token | Attention layers naturally absorb identity context |
| informer | Adapter + mixin | **Yes** | Same as Transformer before ProbSparse attention | Preserves sparse complexity while adding identity |
| autoformer | Adapter + mixin | **Yes** | Inject IDs before decomposition/autocorrelation blocks | Trend/season parts can be entity-conditioned |
| fedformer | Adapter + mixin | **Yes** | Entity-conditioned spectral gate/bias in Fourier/Wavelet components | IDs modulate frequency-domain signatures |
| itransformer | Adapter + mixin | **Yes** | Variate-token identity embeddings | Variates are attention tokens; identity is token metadata |
| patchtst | Adapter + native add-after-patch path | **Yes** | Prefer post-patch token-space station embedding (`add_after_patch`) | Patch tokens are the model’s native representation |
| timesnet | Adapter + mixin | **Yes** | Entity conditioning before FFT period blocks; optional period-weight gating | Period salience often differs by entity |
| timemixer | Adapter + mixin | **Yes** | Entity-conditioned multi-scale mixing gates | Entity dynamics differ across scales |
| timexer | Adapter + mixin | **Yes** | Entity context on endogenous branch + cross-attn bias | Endogenous/exogenous fusion should be entity-aware |
| mamba | Adapter + mixin | **Yes** | Input embedding + optional FiLM modulation of SSM blocks | SSM transition behavior can be entity-dependent |
| lstm (vanilla module) | Runtime wrapper only | **Yes** | Native `nn.Embedding` concat path (or wrapper fallback) | LSTM handles straightforward concatenative conditioning |
| etsformer | Runtime wrapper only | **Yes** | Inject IDs into level/growth/season components | ETS terms are entity-specific by construction |
| lightts | Runtime wrapper only | **Yes** | Entity-conditioned channel projection in IEBlock | Channel MLP mixing benefits from entity priors |
| reformer | Runtime wrapper only | **Yes** | Inject IDs before LSH projections / bucket assignment | Identity can influence locality/hash grouping |
| gpt4ts | Runtime wrapper only | **Yes** | Entity token/prefix before GPT2 patch stream | LLM token pipelines benefit from explicit identity tokens |
| nonstationary_transformer | Runtime wrapper only | **Yes** | Feed IDs into tau/delta projector and embedding stream | Entity-level nonstationarity differs |
| timellm | Adapter + mixin | **Yes** | Numeric/entity embedding side channel + prompt tag template | Prompt-only identity is brittle; numeric conditioning is stable |
| timemoe | Adapter + mixin (task caveat) | **Yes** | Entity-conditioned router prior/prefix expert token | MoE routing is the natural locus for specialization |
| LSTMAdapter (Swiss family) | Native Swiss adapter logic | **Yes** | Keep native embedding/transparent/feature-concat modes | Already mature and flexible |
| ExtrapoLSTMAdapter | Native Swiss adapter logic | **Yes** | Add ID-aware future-step embeddings in extrapolation heads | Horizon-specific behavior can be entity-specific |
| TransformerEncoderAdapter (Swiss family) | Native Swiss adapter logic | **Yes** | Keep embedding + feature-concat + transparent paths; strengthen multi-channel handling | Encoder-only path is already entity-capable |

### 4.2 Step-by-step revision plan (model-by-model)

| Model | Step-by-step plan |
|---|---|
| dlinear | 1) Add optional entity affine layer after trend+season sum. 2) Keep wrapper fallback for compatibility. 3) Add homogeneous-dataset ablations. |
| transformer | 1) Add `entity_injection` mode (`add_to_embed`/`entity_token`). 2) Keep no-ID parity tests. 3) Benchmark traffic/electricity. |
| informer | 1) Mirror transformer injection API. 2) Validate memory/runtime under long sequences. 3) Run entity ablations. |
| autoformer | 1) Inject identity before decomposition/autocorrelation. 2) Add decomposition-consistency tests. 3) Benchmark traffic + swiss-river. |
| fedformer | 1) Add optional entity spectral gate. 2) Add Fourier/Wavelet shape checks. 3) Compare with wrapper-only baseline. |
| itransformer | 1) Add explicit variate identity embeddings. 2) Validate channel-order assumptions. 3) Benchmark on traffic/PEMS. |
| patchtst | 1) Keep `add_after_patch` as primary. 2) Add explicit channel-order tests/permutation checks. 3) Keep wrapper path as fallback. |
| timesnet | 1) Inject IDs before FFT/top-k period selection. 2) Optionally gate period weights by ID. 3) Validate period stability. |
| timemixer | 1) Add entity-conditioned scale-mixing gates. 2) Validate multi-scale output consistency. 3) Measure latency impact. |
| timexer | 1) Inject identity into endogenous embedding stream. 2) Optionally bias exogenous cross-attention. 3) Validate exogenous integrity. |
| mamba | 1) Add native entity embedding/FILM option. 2) Keep wrapper mode. 3) Validate long-sequence stability and throughput. |
| lstm (module) | 1) Add adapter with `EntityAwareMixin` or native embedding path. 2) Add no-ID parity + entity smoke tests. 3) Register in benchmark map. |
| etsformer | 1) Add native entity hooks for level/growth/season blocks. 2) Add adapter-level entity support. 3) Benchmark vs wrapper-only. |
| lightts | 1) Add entity-aware channel-projection bias/gate. 2) Add adapter for benchmark path. 3) Validate speed/accuracy tradeoff. |
| reformer | 1) Add ID injection before LSH stack. 2) Add adapter and long-sequence correctness tests. 3) Compare to transformer injection. |
| gpt4ts | 1) Add adapter exposing runtime config parity. 2) Add entity-token prefix strategy. 3) Benchmark against wrapper-only conditioning. |
| nonstationary_transformer | 1) Add adapter. 2) Feed IDs into `tau_learner`/`delta_learner`. 3) Validate calibration by entity. |
| timellm | 1) Keep numeric conditioning as baseline. 2) Add optional prompt entity tags. 3) Ablate numeric-only vs numeric+prompt. |
| timemoe | 1) Align task-name semantics with runtime benchmark tasks. 2) Add entity-aware router bias/prefix path. 3) Add curated benchmark configs. |
| Swiss LSTM family | 1) Preserve rich mode matrix. 2) Improve naming clarity vs vanilla `lstm` module. 3) Add cross-dataset smoke tests. |
| Swiss Transformer family | 1) Preserve mode matrix. 2) Add stronger multi-channel handling/tests. 3) Harmonize docs with core entity guide. |

---

## 5. Discrepancy log vs prior report

| Type | Prior state | Refined correction | Impact |
|---|---|---|---|
| Omission | Plugin adapters not explicitly represented | Added `SwissRiverDatasetAdapter` and `TrafficDatasetAdapter` as plugin-surface datasets; marked as synthetic stubs | Prevents over-claiming plugin readiness |
| Omission | Downloader support surface not explicit | Added downloader matrix (`_DATASET_FILES` and manual Swiss set) | Clarifies what “supported” means operationally |
| Semantics mismatch | “Supported dataset” could be read as runtime-ready for all registry keys | Split runtime-ready (`pipeline`) vs registry-only (`data_factory`) keys (`solar/m4/custom`) | Avoids false assumption of runtime coverage |
| Naming mismatch | Lowercase runtime names and CamelCase benchmark names mixed | Added canonical name map and per-surface table | Prevents model coverage drift |
| Surface mismatch | Runtime module-only models not separated from benchmark adapter path | Explicitly flagged `lstm/etsformer/lightts/reformer/gpt4ts/nonstationary_transformer` as runtime-only in benchmark tooling | Prevents accidental omission in benchmarks |
| Alias ambiguity | `LSTMAdapter` could be misread as vanilla `lstm.py` adapter | Clarified `LSTMAdapter` export comes from `swiss_lstm.py` | Prevents wrong adapter assumptions |
| Execution caveat | TimeMoE practical run condition underemphasized | Added task-name caveat (`zero_shot_forecast`) | Prevents silent non-execution in standard flows |
| Data plumbing risk | Transparent mode no-op risk noted but not fully tied to split constructors | Added explicit constructor-level evidence (station_name gating vs CSV/PEMS/swiss-mc paths) | Stronger root-cause clarity |

---

## 6. Critical self-critique and uncertainty register

| Topic | Risk / uncertainty | Mitigation |
|---|---|---|
| Causal claim (“IDs help”) | This report is architectural and literature-backed; it is not a completed ablation run | Execute controlled ablations by dataset family and report significance |
| `custom` / `solar` | Suitability depends on schema semantics | Add schema validator: homogeneous-entity vs heterogeneous-feature classification |
| TimeMoE conditioning | Practical integration depends on generation/task API details | Implement and test explicit runtime adapter contract for forecasting tasks |
| Channel-order assumptions | `ChannelEntityWrapper` and PatchTST add-after-patch rely on stable channel indexing | Add permutation tests + explicit channel index mapping |
| Swiss multi-channel ID path | Data-layer identifier args omitted in Swiss MC split constructor | Add explicit pass-through for `identifier_mode/id_integration/station_ids` in MC builder |

---

## 7. Prioritized implementation roadmap

| Priority | Work package | Why now |
|---|---|---|
| P0 | Unify model/data registries across runtime, benchmark tool, and config generator | Eliminates naming/coverage drift |
| P0 | Fix transparent-ID plumbing for multi-channel CSV/PEMS/Swiss-MC paths | Removes silent no-op behavior |
| P0 | Clarify/standardize adapter naming (`VanillaLSTMAdapter` vs Swiss aliases) | Avoids wrong-model execution in experiments |
| P1 | Add benchmark adapters for runtime-only models (`etsformer`, `lightts`, `reformer`, `gpt4ts`, `nonstationary_transformer`, optional `lstm`) | Expands comparable model coverage |
| P1 | Promote architecture-native ID hooks where wrappers are currently the only path | Improves model-specific inductive bias |
| P2 | Extend LLM/MoE ID conditioning (TimeLLM prompt tags + TimeMoE router priors) | Higher upside, higher complexity |

---

## 8. Final completion verification

### 8.1 Requirement-to-deliverable mapping

| Requested item | Delivered |
|---|---|
| Read code/docs/files and list supported datasets/models | Sections 1 and 2 |
| Identify suitable datasets for entity identifiers | Sections 1.1 and 1.2 |
| Identify identifier forms currently considered | Section 3.3 (plus code anchors) |
| Identify current model support and mechanisms | Sections 2 and 4 |
| For each forecasting model: feasibility after revision + reasons + design | Section 4.1 |
| Model-by-model detailed step plan | Section 4.2 |
| Correct/refine prior report with explicit discrepancy log | Section 5 |
| Final detailed verification | This section (8) |

### 8.2 No-miss dataset coverage table

| Dataset surface item | Covered in this report |
|---|---|
| `traffic`, `electricity`, `exchange_rate`, `weather`, `illness` | Section 1.1 |
| `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2` | Section 1.1 |
| `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08` | Section 1.1 |
| `swiss-river-1990`, `swiss-river-2010`, `swiss-river-zurich` | Section 1.1 |
| `solar`, `m4`, `custom` | Section 1.1 |
| Plugin datasets (`SwissRiverDatasetAdapter`, `TrafficDatasetAdapter`) | Section 1.1 |

### 8.3 No-miss forecasting model coverage table

| Forecasting model surface item | Covered in this report |
|---|---|
| Runtime keys: `dlinear`, `transformer`, `informer`, `autoformer`, `fedformer`, `itransformer`, `patchtst`, `timesnet`, `timemixer`, `timexer`, `mamba`, `lstm`, `etsformer`, `lightts`, `reformer`, `gpt4ts`, `nonstationary_transformer`, `timellm`, `timemoe` | Sections 2.1 and 4 |
| Benchmark/Swiss adapter keys: `DLinear`, `Transformer`, `Informer`, `Autoformer`, `FEDformer`, `iTransformer`, `PatchTST`, `TimesNet`, `TimeMixer`, `TimeXer`, `Mamba`, `LSTMAdapter`, `ExtrapoLSTMAdapter`, `TransformerEncoderAdapter` | Sections 2.2 and 4 |
| Alias exports: `SwissLSTMAdapter`, `SwissExtrapoLSTMAdapter`, `SwissLSTMEmbeddingAdapter`, `SwissTransformerAdapter`, `SwissTransformerEmbeddingAdapter` | Sections 2.2 and 4 |

### 8.4 Citation completeness check

| Claim class | Code evidence | External evidence |
|---|---|---|
| Dataset support surfaces | Yes | N/A |
| Model support surfaces | Yes | N/A |
| Entity plumbing behavior | Yes | N/A |
| Model architecture rationale | Yes (code) | Yes (paper links below) |
| “Entity identity helps homogeneous MTS” rationale | Yes (repo modes) | Yes (STID, AGCRN + model papers) |

---

## 9. Online references used in this refined pass

### 9.1 Model papers

| Model | Paper/reference |
|---|---|
| DLinear | https://arxiv.org/pdf/2205.13504.pdf |
| Transformer | https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf |
| Informer | https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132 |
| Autoformer | https://openreview.net/pdf?id=I55UqU-M11y |
| FEDformer | https://proceedings.mlr.press/v162/zhou22g.html |
| iTransformer | https://arxiv.org/abs/2310.06625 |
| PatchTST | https://arxiv.org/abs/2211.14730 |
| TimesNet | https://openreview.net/pdf?id=ju_Uqw384Oq |
| TimeMixer | https://openreview.net/pdf?id=7oLshfEIC2 |
| TimeXer | https://arxiv.org/abs/2402.19072 |
| Mamba | https://arxiv.org/abs/2312.00752 |
| TimeLLM | https://arxiv.org/abs/2310.01728 |
| TimeMoE | https://arxiv.org/abs/2409.16040 |
| ETSformer | https://arxiv.org/abs/2202.01381 |
| LightTS | https://arxiv.org/abs/2207.01186 |
| Reformer | https://openreview.net/forum?id=rkgNKkHtvB |
| GPT4TS (One Fits All) | https://arxiv.org/abs/2302.11939 |
| Nonstationary Transformer | https://openreview.net/pdf?id=ucNDIDRNjjv |

---

## 10. Deep-dive corrections and refinements (2026-04-16)

_This section was added after completing per-model deep-dive audits in `docs/research/entity-id-deep/`. Each correction references the deep-dive file where the finding was made._

### 10.1 Corrections to Section 4.1 feasibility matrix

| Model | Prior claim (Section 4.1) | Corrected finding | Deep-dive file |
|---|---|---|---|
| TimeMoE | "Yes — entity-conditioned router prior/prefix expert token" | **No** for zero-shot mode. No training = no learnable parameters. Entity injection requires fine-tuning support (not implemented). Recommend excluding from entity-ID ablation. | `timemoe.md` |
| LSTM (vanilla) | "Add adapter with EntityAwareMixin or native embedding path" | Adapter (`lstm.py:96`) **lacks EntityAwareMixin** — must be retrofitted as prerequisite. Hidden-init (H2) is the LSTM-native primary design, not concat. | `lstm.md` |
| ETSformer | "Add native entity hooks for level/growth/season blocks" | Adapter (`etsformer.py:66`) **lacks EntityAwareMixin** — must be retrofitted. Additionally, `level_bias` (H3) is the architecturally natural injection for ETS decomposition — adds per-station offset to the exponential-smoothing level. | `etsformer.md` |
| LightTS | "Add entity-aware channel-projection bias/gate" | **No adapter class exists at all** — only `Model`. Must create `LightTSAdapter(EntityAwareMixin, TorchModelAdapter)` first. Primary design is `post_output_affine` (H3), not channel-projection bias — LightTS is too simple for mid-model injection. | `lightts.md` |
| Reformer | "Add ID injection before LSH stack" | **No adapter class exists** — must create `ReformerAdapter`. LSH bucketing partially cancels additive bias in dot-product → entity signal mainly flows through residual, not hash selection. | `reformer.md` |
| GPT4TS | "Add adapter exposing runtime config parity" | **No adapter class exists** — must create `GPT4TSAdapter`. Entity embedding enters frozen GPT-2 via `inputs_embeds`; LN (only trainable component) mediates entity conditioning. | `gpt4ts.md` |
| NST | "Add adapter; feed IDs into tau/delta projectors" | **No adapter class exists** — must create `NonstationaryTransformerAdapter`. Primary design is standard H2 `add_to_embed` (same as Transformer); tau/delta conditioning (H3) is secondary/deferred. | `nonstationary_transformer.md` |
| Mamba | "Add native entity embedding/FILM option" | Primary design is H2 `add_to_embed` pre-SSM — entity bias becomes input-dependent SSM parameterization (Δ, B, C). FiLM is over-engineering for the single-block TSL port. | `mamba.md` |
| TimeLLM | "Numeric/entity embedding side channel + prompt tag template" | H4 `entity_in_prompt` is the **architecturally unique** primary design — inject station name/description into the natural-language prompt. No other model can leverage text-based entity injection. H2 `add_to_patch_embed` is secondary. | `timellm.md` |
| TimeXer | "Entity context on endogenous branch + cross-attn bias" | H2′ `entity_global_token` — replace shared `glb_token` with per-entity learned token. **Architecturally most novel** in the suite: directly edits the existing entity-like slot that conditions cross-attention. | `timexer.md` |
| iTransformer | "Variate-token identity embeddings" | **Architecturally the best fit** — `(B, N, d_model)` shape already has variates as tokens. Entity bias directly conditions cross-variate attention. Prior description was correct in spirit but understated the architectural alignment. | `itransformer.md` |

### 10.2 Corrections to Section 2.2 adapter status

| Model | Prior status | Corrected status |
|---|---|---|
| `lstm` (vanilla) | "Runtime-only module" | `LSTMAdapter` exists at `lstm.py:96` but **inherits only `TorchModelAdapter`** — no EntityAwareMixin |
| `etsformer` | "Runtime-only module" | `ETSformerAdapter` exists at `etsformer.py:66` but **inherits only `TorchModelAdapter`** — no EntityAwareMixin, minimal `_build_model` style |
| `lightts` | "Runtime-only module" | **No adapter class at all** — only `Model` class. Dynamic import via `pipeline.build_model:442` |
| `reformer` | "Runtime-only module" | **No adapter class at all** — only `Model` class |
| `gpt4ts` | "Runtime-only module" | **No adapter class at all** — only `Model` class |
| `nonstationary_transformer` | "Runtime-only module" | **No adapter class at all** — only `Model` class |

### 10.3 New architectural taxonomy

The deep-dive audit revealed a natural taxonomy for entity injection designs:

| Category | Models | Primary injection pattern |
|---|---|---|
| **Standard H2 `add_to_embed`** | Transformer, Informer, Autoformer, FEDformer, NST, TimesNet, Mamba, Reformer | Additive `nn.Embedding(N, d_model)` after DataEmbedding, before encoder |
| **Channel-independent post-patch** | PatchTST (native), GPT4TS, TimeMixer, TimeLLM (secondary) | Additive in `(B*C, num_patches, d_model)` layout — restores identity lost by CI |
| **Architecture-specific slot** | TimeXer (`entity_global_token`), iTransformer (variate-as-token) | Leverages model's unique structural feature |
| **Text-based prompt** | TimeLLM (primary) | Entity name/description in natural-language prompt — unique to LLM-based models |
| **Hidden-state init** | LSTM (vanilla) | Per-entity `(h_0, c_0)` — classical RNN identity conditioning |
| **ETS-level bias** | ETSformer | Per-station offset on exponential-smoothing level — domain-aligned |
| **Output affine** | DLinear (primary), LightTS (primary) | Per-station `(N, pred_len)` bias on final output — minimal-parameter fallback |
| **Pre-existing custom system** | Swiss-LSTM, Swiss-Transformer | Independent entity system with embedding/feature_concat/transparent modes |
| **Not applicable** | TimeMoE | Zero-shot only — no learnable parameters |

### 10.4 Models requiring prerequisite work before entity injection

| Model | Prerequisite | Effort estimate |
|---|---|---|
| LSTM (vanilla) | Add `EntityAwareMixin` to `LSTMAdapter` inheritance | Small (copy MambaAdapter pattern) |
| ETSformer | Add `EntityAwareMixin` to `ETSformerAdapter` + expand from `_build_model` to `__init__` | Small-medium |
| LightTS | Create `LightTSAdapter(EntityAwareMixin, TorchModelAdapter)` from scratch | Medium |
| Reformer | Create `ReformerAdapter(EntityAwareMixin, TorchModelAdapter)` | Medium |
| GPT4TS | Create `GPT4TSAdapter(EntityAwareMixin, TorchModelAdapter)` | Medium |
| NST | Create `NonstationaryTransformerAdapter(EntityAwareMixin, TorchModelAdapter)` | Medium |

---

### 9.2 Entity-identity-specific evidence

| Topic | Reference |
|---|---|
| Spatial/temporal identity as a key forecasting factor | STID: https://arxiv.org/abs/2208.05233 |
| Node-specific embeddings improve traffic forecasting | AGCRN: https://arxiv.org/abs/2007.02842 |

### 9.3 Dataset-context references

| Topic | Reference |
|---|---|
| TSL benchmark ecosystem | https://github.com/thuml/Time-Series-Library |
| Traffic/Electricity benchmark provenance | https://arxiv.org/abs/1703.07015 |
| PEMS source | http://pems.dot.ca.gov |
| Weather source context | https://www.bgc-jena.mpg.de/wetter/ |
