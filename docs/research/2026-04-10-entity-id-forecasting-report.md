# Entity Identifier Research Report for Forecasting in LIULIAN

_Date: 2026-04-10_

## 0. Scope and approved process

| Item | Decision |
|---|---|
| Model scope | Active LIULIAN forecasting models only (core `liulian/models/torch/*` + in-project Swiss adapters), excluding `refer_projects/` adaptation work |
| Dataset focus | All currently supported datasets; traffic is the primary entity case; weather must be audited variable-by-variable |
| Deliverable | This report file (`docs/research/2026-04-10-entity-id-forecasting-report.md`) with markdown tables |
| Citation rule | Per-model online citations included |

| Planned step (approved) | Execution status | Main evidence |
|---|---|---|
| Code/docs inventory | Done | `liulian/pipeline.py`, `liulian/data/*`, `docs/*`, `experiments/*` |
| Dataset suitability audit | Done | dataset schema inspection (`traffic/electricity/weather/ETT/PEMS/swiss_river`) + docs |
| Entity plumbing audit | Done | `timeseriesdataset.py`, `entity_mixin.py`, `runtime/trainer.py`, `patchtst.py` |
| Model-by-model feasibility/design | Done | all active forecasting models listed below |
| Online deep references | Done | paper/repo links in Section 7 |
| Completion verification | Done | Section 6 checklist |

---

## 1. Current project inventory (datasets, models, entity support)

### 1.1 Supported datasets (current codebase)

#### A) Directly supported by `liulian.pipeline.build_dataset` (main runtime path)

| Dataset key | Loader | Entity semantics | Suitable for entity IDs? | Entity identifiers currently available/possible |
|---|---|---|---|---|
| `traffic` | `CustomCSVDataset` via `_CSV_DATASET_MAP` | 862 loop sensors (same variable type) | **Yes (strong)** | sensor column names (`0..860`, `OT`) as entity IDs; integer channel index |
| `electricity` | `CustomCSVDataset` | 321 clients/meters (homogeneous) | **Yes (strong)** | client column names (`0..319`, `OT`) |
| `PEMS03/04/07/08` | `PEMSDataset` | traffic sensors (homogeneous entities) | **Yes (strong)** | `sensor_0...sensor_N` |
| `swiss-river-1990/2010/zurich` | `SwissRiverDataset` | river stations (entity-centric) | **Yes (strong)** | station IDs from `*_wt`, entity index, optional coordinates/topology |
| `exchange_rate` | `CustomCSVDataset` | 8 currencies (small, mildly heterogeneous macro context) | **Conditional** | currency/channel column names |
| `ETTh1/ETTh2/ETTm1/ETTm2` | `ETTHourDataset` / `ETTMinuteDataset` | transformer load/temperature variables (feature types, not entities) | **No (recommended)** | technically possible to force IDs, but semantically weak |
| `weather` | `CustomCSVDataset` | meteorological variable channels at one station (feature types, not entities) | **No (recommended)** | no station identity per channel; channels are variable dimensions |
| `illness` | `CustomCSVDataset` | epidemiological summary indicators (feature types) | **No (recommended)** | channels are statistics/age slices, not interchangeable entities |

#### B) In `data_factory` registry (not wired through `pipeline.build_dataset`)

| Dataset key | Registered path | Entity suitability |
|---|---|---|
| `solar` | `CustomCSVDataset` in `data_factory.py` | **Yes (strong)** if used as PV-plant channels (137 plants) |
| `m4` | `M4Dataset` in `data_factory.py` | **No** (mostly univariate per series, no within-sample multi-entity channels) |
| `custom` | `CustomCSVDataset` in `data_factory.py` | **Depends on schema** |

### 1.2 Weather dataset variable-by-variable entity suitability audit

`weather.csv` has 21 non-date channels, including pressure/temperature/humidity/wind/rain/radiation features (`p (mbar)`, `T (degC)`, `rh (%)`, `wv (m/s)`, `rain (mm)`, `PAR`, etc.). These are heterogeneous physical variables.

| Variable cluster | Example columns | Interchangeable entities? | Conclusion |
|---|---|---|---|
| Thermodynamics | `T (degC)`, `Tpot (K)`, `Tdew (degC)`, `Tlog (degC)` | No | Feature channels, not entities |
| Moisture/pressure | `p (mbar)`, `rh (%)`, `VPmax`, `VPact`, `VPdef`, `sh`, `H2OC` | No | Different units and semantics |
| Wind | `wv (m/s)`, `max. wv (m/s)`, `wd (deg)` | No | Distinct meteorological measurements |
| Rain/radiation | `rain (mm)`, `raining (s)`, `SWDR`, `PAR`, `max. PAR` | No | Distinct environmental drivers |

**Result:** weather channels should remain feature dimensions; introducing per-channel entity IDs is not semantically aligned.

### 1.3 Forecasting models currently present

#### Core runtime module models (pipeline dynamic import path)

| Model key (config) | Module | In maintained experiment YAMLs |
|---|---|---|
| `dlinear` | `liulian.models.torch.dlinear` | Yes |
| `transformer` | `...transformer` | Yes |
| `informer` | `...informer` | Yes |
| `autoformer` | `...autoformer` | Yes |
| `fedformer` | `...fedformer` | Yes |
| `itransformer` | `...itransformer` | Yes |
| `patchtst` | `...patchtst` | Yes |
| `timesnet` | `...timesnet` | Yes |
| `timemixer` | `...timemixer` | Yes |
| `timexer` | `...timexer` | Yes |
| `mamba` | `...mamba` (shim to `mamba_model`) | Yes |
| `lstm` | `...lstm` | Yes |
| `etsformer` | `...etsformer` | Yes |
| `lightts` | `...lightts` | Yes |
| `reformer` | `...reformer` | Yes |
| `gpt4ts` | `...gpt4ts` | Yes |
| `nonstationary_transformer` | `...nonstationary_transformer` | Yes |
| `timellm` | `...timellm` | Yes |
| `timemoe` | `...timemoe` | Not in maintained YAMLs (module exists) |

#### Additional in-project forecasting adapters (benchmark/custom path)

| Adapter/class | File | Notes |
|---|---|---|
| `LSTMAdapter` | `liulian/models/torch/swiss_lstm.py` | Rich built-in entity-mode handling |
| `ExtrapoLSTMAdapter` | `.../swiss_lstm.py` | Extrapolation variant with future-step embeddings |
| `TransformerEncoderAdapter` | `liulian/models/torch/swiss_transformer.py` | Encoder-only transformer with entity modes |

### 1.4 Which entity identifiers are already considered in this project

| Identifier concept | Where created | Where consumed |
|---|---|---|
| String entity IDs (`entity_id_strs`) | `TimeSeriesSplit`/collate (`timeseriesdataset.py`) | inverse-transform and reporting paths |
| Integer entity index (`entity_idx`) | DataLoader collate in `TimeSeriesDataset.get_data_loaders` | `ForecastTrainer` forwards as `entity_ids` to model |
| Embedding IDs from marks (`entity_id_col`) | `x_mark_enc[..., entity_id_col]` | `EntityWrapper` / Swiss adapters embedding mode |
| Channel/station IDs in multi-channel | station order in `station_ids` | `ChannelEntityWrapper` or PatchTST internal embedding |
| Geographic coordinates | Swiss topology (`TopologySpec.coordinates`) | `coordinates` identifier mode |
| One-hot/numeric/sinusoidal/descriptors features | `make_entity_features()` | transparent model input augmentation |

### 1.5 Which models currently support entity IDs and how

| Support mechanism | Models | Current behavior |
|---|---|---|
| **EntityAwareMixin + wrappers** | dlinear, transformer, informer, autoformer, fedformer, itransformer, patchtst, timesnet, timemixer, timexer, mamba, timellm, timemoe | `identifier_mode='embedding'` wraps with `EntityWrapper` (per-entity) or `ChannelEntityWrapper` (multi-channel); transparent modes rely on data-layer feature concat |
| **PatchTST native patch-space injection** | patchtst | `id_integration='add_after_patch'` adds channel embeddings in `d_model` space post-patching |
| **Pipeline-level generic wrapping (module path)** | lstm, etsformer, lightts, reformer, gpt4ts, nonstationary_transformer (and any dynamic module) | `pipeline.build_model` wraps model with `EntityWrapper`/`ChannelEntityWrapper` when embedding mode is enabled |
| **Swiss custom adapters built-in modes** | LSTMAdapter, ExtrapoLSTMAdapter, TransformerEncoderAdapter | embedding + transparent + feature-concat style modes implemented directly in adapter/model pair |

---

## 2. Model-by-model feasibility and architecture-specific entity-ID design

### 2.1 Feasibility + proposed design matrix (all forecasting models)

| Model | Current support now | Can support after revision? | Recommended entity-ID design for this architecture | Why this design fits |
|---|---|---|---|---|
| dlinear | Yes (EntityAwareMixin/pipeline wrapper) | **Yes (already)** | Keep channel embedding (concat+proj) baseline; optional per-channel affine bias on trend/season heads | Linear decomposition benefits from low-overhead station-specific bias |
| transformer | Yes | **Yes (already)** | Add entity embedding to token/value embedding (or prepend entity token) before encoder/decoder attention | Standard attention naturally absorbs additive identity context |
| informer | Yes | **Yes (already)** | Same as Transformer; inject before ProbSparse attention | Preserves sparse-attention complexity while encoding station identity |
| autoformer | Yes | **Yes (already)** | Inject entity embedding before decomposition/autocorrelation blocks | Entity effects can shift trend/seasonal components |
| fedformer | Yes | **Yes (already)** | Entity-conditioned frequency-domain bias/gate in Fourier/Wavelet blocks | FEDformer models spectral structure; station IDs should modulate spectral amplitudes |
| itransformer | Yes | **Yes (already)** | Variate-token identity embedding (entity IDs as variate tokens) | iTransformer tokenizes variates; entity identity is native token metadata |
| patchtst | Yes + special native path | **Yes (already best path exists)** | Prefer `add_after_patch` (patch-space station embeddings); keep wrapper path as fallback | Patch tokens are model-native units; patch-space injection is better-aligned than raw concat |
| timesnet | Yes | **Yes (already)** | Inject IDs before 2D reshaping/period block; optional period-weight gating by entity | Period discovery and 2D conv can be entity-conditioned |
| timemixer | Yes | **Yes (already)** | Entity-conditioned multi-scale mixing gates | Multi-scale trend/season mixing can vary by station dynamics |
| timexer | Yes | **Yes (already)** | Entity embedding for endogenous stream + exogenous cross-attn bias | TimeXer separates endogenous/exogenous; entity ID should anchor endogenous branch |
| mamba | Yes | **Yes (already)** | Add entity embedding at input; optional FiLM-style modulation of SSM block | SSM state evolution can be station-conditioned with minimal overhead |
| timellm | Yes | **Yes (already)** | Keep numeric side-channel embedding; optionally add entity tags in prompt template | Prompt-only identity is brittle; numeric embedding keeps deterministic station signal |
| timemoe | Yes | **Yes (already)** | Entity-conditioned router prior / soft prompt before generation | Expert routing is the right place for station specialization |
| lstm (vanilla module) | Wrapper support via pipeline | **Yes** | Add explicit station embedding concat in model (or rely on wrapper) | LSTM input concatenation is simple and effective for station-specific offsets |
| etsformer | Wrapper support via pipeline | **Yes** | Inject IDs into level/growth/season initialization and decoder bias | ETS decomposition terms are station-specific by nature |
| lightts | Wrapper support via pipeline | **Yes** | Entity bias/gating in IEBlock channel projection | LightTS is MLP channel mixing; channel identity should influence mixing weights |
| reformer | Wrapper support via pipeline | **Yes** | Add IDs before LSH attention hashing/projection | Station identity should influence token grouping/buckets |
| gpt4ts | Wrapper support via pipeline | **Yes** | Append entity token embedding to patch sequence before frozen GPT2 | GPT-style token processing benefits from explicit identity token |
| nonstationary_transformer | Wrapper support via pipeline | **Yes** | Feed entity embedding into tau/delta projector and embedding stream | Station-level nonstationarity should alter de-stationary factors |
| LSTMAdapter (Swiss) | Yes (native rich modes) | **Yes (already)** | Keep existing mode support; harden generic multi-dataset path | Already mature; value is standardization and tests |
| ExtrapoLSTMAdapter | Yes (native) | **Yes (already)** | Keep existing + entity-aware future-step embeddings | Extrapolation head should include station context in future token embeddings |
| TransformerEncoderAdapter (Swiss) | Yes (native rich modes) | **Yes (already)** | Keep existing; add stronger multi-channel ID plumbing/tests | Encoder-only architecture already supports embedding/transparent modes |

### 2.2 Model-by-model step-by-step revision plan

> The table below is intentionally explicit and model-specific to avoid omissions.

| Model | Step-by-step plan |
|---|---|
| dlinear | 1) Add optional `entity_affine` flag in `dlinear.py` Model. 2) Apply station-conditioned affine after seasonal/trend sum. 3) Keep fallback to current wrapper path. 4) Add unit tests for shape + per-entity divergence. 5) Add traffic/electricity ablation configs. |
| transformer | 1) Add `entity_injection` config (`add_to_embed` / `entity_token`). 2) Inject in `enc_embedding` and decoder input path. 3) Ensure compatibility with `features=M/MS`. 4) Add regression tests (no-ID parity). 5) Benchmark on traffic + swiss-river. |
| informer | 1) Mirror transformer injection API. 2) Inject pre-ProbSparse attention. 3) Confirm no complexity blowup. 4) Add tests for long-sequence memory stability. 5) Run traffic/electricity ablation. |
| autoformer | 1) Add entity injection before decomposition/autocorrelation layers. 2) Keep decoder warmup behavior unchanged. 3) Add forecast-shape + decomposition-consistency tests. 4) Add traffic + swiss-river configs. 5) Compare against baseline wrapper. |
| fedformer | 1) Add station embedding-controlled spectral gate in Fourier/Wavelet blocks. 2) Guard with config flag. 3) Add spectral-shape tests. 4) Validate training stability. 5) Run entity ablation on traffic/electricity. |
| itransformer | 1) Add explicit variate/entity token embedding table keyed by channel index. 2) Sum into variate tokens before attention. 3) Validate with multi-channel datasets. 4) Add tests for channel-order invariance assumptions. 5) Benchmark traffic/PEMS. |
| patchtst | 1) Keep `add_after_patch` as default entity path. 2) Add optional dynamic `entity_ids` mapping for non-canonical channel order. 3) Add tests for channel permutation behavior. 4) Keep wrapper path documented as legacy. 5) Benchmark traffic/PEMS/swiss-river-mc. |
| timesnet | 1) Inject IDs before FFT period discovery and 2D reshape. 2) Optionally gate period weights by entity embedding. 3) Add tests around top-k period stability. 4) Add configs for traffic/electricity. 5) Compare against wrapper-only baseline. |
| timemixer | 1) Add entity-conditioned gates in season/trend mixing layers. 2) Keep backward-compatible default off. 3) Add unit tests for multi-scale output shape and determinism. 4) Add traffic/PEMS configs. 5) Evaluate latency impact. |
| timexer | 1) Add endogenous entity embeddings to EnEmbedding branch. 2) Optionally bias cross-attention using entity context. 3) Add exogenous-variable integrity tests. 4) Add targeted configs for datasets with exogenous marks. 5) Evaluate with traffic/electricity. |
| mamba | 1) Add input-side station embedding addition/concat path. 2) Optional FiLM modulation in SSM block parameters. 3) Add tests for long-sequence throughput and stability. 4) Add traffic/electricity configs. 5) Compare with wrapper-only mode. |
| timellm | 1) Keep numeric embedding injection path as primary. 2) Add optional entity text tags to prompt content builder. 3) Add cache-safe prompt templating tests. 4) Add ablation: numeric-only vs numeric+prompt-tag. 5) Benchmark on swiss-river and traffic. |
| timemoe | 1) Add entity-conditioned router bias (or prefix token). 2) Keep generation API unchanged. 3) Add tests for expert routing distribution by entity. 4) Add lightweight config template (currently missing in maintained experiments). 5) Benchmark zero-shot vs tuned. |
| lstm (vanilla module) | 1) Add optional in-model `nn.Embedding` path keyed by `entity_ids`. 2) Preserve wrapper compatibility. 3) Add unit tests for both paths. 4) Add traffic/swiss-river configs. 5) Document when to prefer native vs wrapper path. |
| etsformer | 1) Add entity embeddings to level/growth/season inputs. 2) Preserve encoder/decoder layer count constraints. 3) Add decomposition consistency tests. 4) Add configs on traffic/electricity. 5) Compare against wrapper baseline. |
| lightts | 1) Add station-conditioned bias/gate in IEBlock channel projection. 2) Keep low-parameter design. 3) Add tests for channel-projection correctness. 4) Add traffic/electricity configs. 5) Benchmark speed/accuracy tradeoff. |
| reformer | 1) Inject entity embedding before ReformerLayer input. 2) Verify bucket/hash behavior remains stable. 3) Add long-sequence correctness tests. 4) Add traffic/electricity configs. 5) Compare with Transformer injection behavior. |
| gpt4ts | 1) Add entity token embedding into patch token stream before GPT2 backbone. 2) Keep frozen-backbone policy (LN/PE only trainable) unless explicitly overridden. 3) Add tests for patch/entity token indexing. 4) Add traffic/electricity configs. 5) Benchmark against wrapper-only mode. |
| nonstationary_transformer | 1) Add entity features to `tau_learner`/`delta_learner` inputs. 2) Optionally inject into `DataEmbedding`. 3) Add tests for tau/delta numerical range by entity. 4) Add traffic/swiss-river configs. 5) Compare stability and calibration. |
| LSTMAdapter (Swiss) | 1) Keep existing identifier-mode matrix. 2) Add pipeline naming disambiguation docs (`lstm` vs Swiss adapter export). 3) Add cross-dataset tests (traffic + swiss-river). 4) Add CI smoke benchmark. 5) Consolidate docs with core entity guide. |
| ExtrapoLSTMAdapter | 1) Preserve extrapolation heads (`limo`/`fembed`). 2) Add entity-aware future-step embedding ablation. 3) Add synthetic tests for horizon scaling. 4) Add configs for swiss-river long horizon. 5) Document recommended mode by task. |
| TransformerEncoderAdapter (Swiss) | 1) Keep embedding/transparent/feature-concat modes. 2) Add explicit multi-channel entity index support path. 3) Add tests for mask embedding + entity interaction. 4) Add traffic and swiss-river configs. 5) Document mode-selection guidance. |

---

## 3. Prioritized implementation roadmap

| Priority | Work package | Why first |
|---|---|---|
| **P0** | Fix entity data plumbing consistency (transparent modes in multi-channel paths; channel-order assumptions; config name alignment between pipeline vs benchmark scripts) | Prevent silent no-op behavior and evaluation ambiguity |
| **P0** | Standardize model naming and registration (`pipeline` model keys vs `tools/run_benchmark.py` adapter map) | Avoid mismatched coverage and accidental model omissions |
| **P1** | Keep/expand PatchTST `add_after_patch` + iTransformer variate-token entity path | Highest expected gain on traffic/electricity/PEMS |
| **P1** | Add architecture-native entity hooks for module-only models (ETSformer/LightTS/Reformer/GPT4TS/NST/LSTM) | Converts generic wrapper support into model-aware identity handling |
| **P2** | Advanced LLM/MoE identity conditioning (TimeLLM prompt tags, TimeMoE routing priors) | Higher complexity, likely dataset-dependent gains |

---

## 4. Critical self-critique (what could break / be misleading)

| Finding | Evidence | Risk | Recommended fix |
|---|---|---|---|
| Transparent modes can no-op in many multi-channel datasets | `TimeSeriesDataset._precompute_tensors` only applies `make_entity_features` when `station_name is not None`; CSV/PEMS multi-channel paths do not set `station_name` | You may think onehot/numeric/coordinates are active when they are not | Add explicit multi-channel entity-feature generation path keyed by channel index |
| Two model orchestration paths are inconsistent | `pipeline.build_model` uses lowercase dynamic module import; `tools/run_benchmark.py` uses adapter map with CamelCase names | Coverage drift between “official experiments” and benchmark tooling | Unify registry and naming in one source of truth |
| `liulian.models.torch.LSTMAdapter` export points to Swiss adapter, not `lstm.py` adapter | `liulian/models/torch/__init__.py` imports `LSTMAdapter` from `swiss_lstm.py` | Potential confusion in code/tests and benchmark usage | Explicitly rename exports (`SwissLSTMAdapter`, `VanillaLSTMAdapter`) or document strongly |
| `timemoe` exists but lacks maintained experiment YAMLs | Model module exists; no standard configs in `experiments/*` | It can be forgotten in practical evaluations | Add curated configs and CI smoke run |
| Weather entity-ID attempts are semantically invalid | Variable audit + dataset docs | Potentially negative/unstable results from wrong inductive bias | Keep weather as heterogeneous feature-channel benchmark without entity IDs |

---

## 5. Final answers to your two core research questions

### Q1) “What is currently supported? Which datasets are suitable? What IDs and model support already exist?”

**Answer:**  
Supported datasets and models are fully enumerated in Sections 1.1 and 1.3.  
Strong entity-ID datasets in this project are **traffic, electricity, PEMS03/04/07/08, swiss-river variants** (and `solar` if run via data_factory path).  
Current identifier forms already considered include embedding indices, onehot/numeric/sinusoidal/coordinates/descriptors, plus per-sample `entity_idx` and per-channel station IDs.  
Current support mechanisms are already implemented via **EntityAwareMixin wrappers**, **PatchTST internal patch-space ID injection**, **pipeline-level generic wrappers**, and **Swiss custom adapters**.

### Q2) “For each forecasting model, can it support entity IDs after revision? If so, what design and why?”

**Answer:**  
For all forecasting models audited here, support is technically feasible; most already support entity IDs in baseline form.  
Model-specific architecture-aware designs and explicit step-by-step revision plans are provided in Sections 2.1 and 2.2.

---

## 6. Detailed completion verification against requested scope

| Requested item | Delivered? | Where |
|---|---|---|
| Read project code/docs deeply; list supported datasets and models | **Yes** | Sections 1.1, 1.3 |
| Identify which datasets are suitable for entity IDs | **Yes** | Section 1.1 suitability column + weather audit (1.2) |
| Identify which entity IDs are considered now | **Yes** | Section 1.4 |
| Identify which models currently support entity IDs and how | **Yes** | Section 1.5 |
| List all forecasting models and evaluate support-after-revision feasibility | **Yes** | Section 2.1 |
| Per-model detailed design, reasons, and architecture-aware plan | **Yes** | Sections 2.1 and 2.2 |
| Detailed process plan first + end-to-end verification | **Yes** | Section 0 + this Section 6 |
| Detailed report in markdown tables | **Yes** | Entire report |

---

## 7. Online references (per model)

| Model | Paper / primary reference | Implementation reference |
|---|---|---|
| dlinear | https://arxiv.org/pdf/2205.13504.pdf | https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py |
| transformer | https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf | https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py |
| informer | https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132 | https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py |
| autoformer | https://openreview.net/pdf?id=I55UqU-M11y | https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py |
| fedformer | https://proceedings.mlr.press/v162/zhou22g.html | https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py |
| itransformer | https://arxiv.org/abs/2310.06625 | https://github.com/thuml/iTransformer |
| patchtst | https://arxiv.org/pdf/2211.14730.pdf | https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py |
| timesnet | https://openreview.net/pdf?id=ju_Uqw384Oq | https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py |
| timemixer | https://openreview.net/pdf?id=7oLshfEIC2 | https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py |
| timexer | https://arxiv.org/abs/2402.19072 | https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py |
| mamba | https://arxiv.org/abs/2312.00752 | https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py |
| timellm | https://arxiv.org/abs/2310.01728 | https://github.com/KimMeen/Time-LLM |
| timemoe | https://arxiv.org/abs/2409.16040 | https://huggingface.co/Maple728/TimeMoE-50M |
| lstm / LSTMAdapter / ExtrapoLSTMAdapter | https://www.bioinf.jku.at/publications/older/2604.pdf | https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py |
| etsformer | https://arxiv.org/abs/2202.01381 | https://github.com/salesforce/ETSformer |
| lightts | https://arxiv.org/abs/2207.01186 | https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py |
| reformer | https://openreview.net/forum?id=rkgNKkHtvB | https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py |
| gpt4ts (One Fits All) | https://arxiv.org/abs/2302.11939 | https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All |
| nonstationary_transformer | https://openreview.net/pdf?id=ucNDIDRNjjv | https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py |
| transformerencoderadapter (Swiss) | https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf | In-project implementation: `liulian/models/torch/swiss_transformer.py` |

### Dataset/source references

| Topic | Reference |
|---|---|
| Time-Series-Library benchmark ecosystem | https://github.com/thuml/Time-Series-Library |
| Traffic/Electricity benchmark provenance (LSTNet) | https://arxiv.org/abs/1703.07015 |
| PEMS source context | http://pems.dot.ca.gov |
| Weather station source context | https://www.bgc-jena.mpg.de/wetter/ |
| LIULIAN entity-ID docs | `docs/entity_identifiers.md` |

