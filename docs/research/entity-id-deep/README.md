# Entity-Identifier Deep Dives — Cross-Cutting Summary

_Generated 2026-04-16. Per-model files follow the 8-section schema in `_template.md`._

## Coverage matrix

| Model | Deep-dive file | Has EntityAwareMixin? | Has adapter? | Primary injection design | `id_integration` key | Category |
|---|---|---|---|---|---|---|
| DLinear | `dlinear.md` | Yes | Yes | Per-station affine on trend+season output | `entity_affine` | Output affine |
| Transformer | `transformer.md` | Yes | Yes | Additive embedding post-DataEmbedding (enc+dec) | `add_to_embed` | Standard H2 |
| Informer | `informer.md` | Yes | Yes | Additive embedding post-DataEmbedding, pre-ProbSparse | `add_to_embed` | Standard H2 |
| Autoformer | `autoformer.md` | Yes | Yes | Additive embedding on both enc_out and dec_out | `add_to_embed_both` | Standard H2 |
| FEDformer | `fedformer.md` | Yes | Yes | Additive embedding post-DataEmbedding | `add_to_embed` | Standard H2 |
| iTransformer | `itransformer.md` | Yes | Yes | Per-variate additive in `(B, N, d_model)` | `add_to_embed` | Architecture-specific |
| PatchTST | `patchtst.md` | Yes (native) | Yes | Additive after patch embedding (existing) | `add_after_patch` | CI post-patch |
| TimesNet | `timesnet.md` | Yes | Yes | Additive after predict_linear, pre-TimesBlock | `add_to_embed` | Standard H2 |
| TimeMixer | `timemixer.md` | Yes | Yes | Additive post-CI-embed, shared across scales | `add_to_embed` | CI post-patch |
| TimeXer | `timexer.md` | Yes | Yes | Replace `glb_token` with per-entity token | `entity_global_token` | Architecture-specific |
| Mamba | `mamba.md` | Yes | Yes | Additive post-DataEmbedding, pre-SSM | `add_to_embed` | Standard H2 |
| LSTM (vanilla) | `lstm.md` | **No** | Partial | Per-entity learned (h_0, c_0) | `entity_hidden_init` | Hidden-state init |
| ETSformer | `etsformer.md` | **No** | Partial | Additive on level + post-embed | `add_to_embed` / `level_bias` | ETS-level bias |
| LightTS | `lightts.md` | **No** | **None** | Per-station output affine | `post_output_affine` | Output affine |
| Reformer | `reformer.md` | **No** | **None** | Additive post-DataEmbedding, pre-LSH | `add_to_embed` | Standard H2 |
| GPT4TS | `gpt4ts.md` | **No** | **None** | Additive after in_layer, pre-frozen-GPT2 | `add_to_patch_embed` | CI post-patch |
| NST | `nonstationary_transformer.md` | **No** | **None** | Additive post-DataEmbedding (enc+dec) | `add_to_embed` | Standard H2 |
| TimeLLM | `timellm.md` | Yes | Yes | Entity name in text prompt (primary) | `entity_in_prompt` | Text-based prompt |
| TimeMoE | `timemoe.md` | Yes | Yes | **N/A** — zero-shot only | N/A | Not applicable |
| Swiss-LSTM | `swiss_lstm.md` | Custom | Yes | `nn.Embedding` concat + feature_concat + transparent | (custom system) | Pre-existing custom |
| Swiss-Transformer | `swiss_transformer.md` | Custom | Yes | `nn.Embedding` concat + feature_concat + transparent | (custom system) | Pre-existing custom |

## Architectural taxonomy

### 1. Standard H2 `add_to_embed` (8 models)
Transformer, Informer, Autoformer, FEDformer, TimesNet, Mamba, Reformer, NST

Pattern: `enc_out = enc_out + entity_embed(entity_ids).unsqueeze(1)` at `(B, L, d_model)`.

Rationale: universal; entity bias enters attention/encoder as a uniform per-sample shift. Time-constant signal lives at DC — harmless to FFT-based period selection (TimesNet) and Fourier decomposition (FEDformer). Autoformer's auto-correlation routes it to trend automatically.

### 2. Channel-independent post-patch (4 models)
PatchTST, GPT4TS, TimeMixer, TimeLLM (secondary)

Pattern: `enc_out = enc_out + entity_embed(ids).unsqueeze(1)` at `(B*C, num_patches, d_model)`.

Rationale: CI folds channels into batch; identity is lost. Per-channel entity embedding **restores** what CI throws away. STID (arxiv:2208.05233) is the primary ablation evidence.

### 3. Architecture-specific slot (2 models)
- **TimeXer**: replace `glb_token` with per-entity token → conditions cross-attention query
- **iTransformer**: variate-as-token layout → entity bias directly in token space

These leverage each model's unique architectural feature rather than a generic embedding.

### 4. Text-based prompt (1 model)
TimeLLM: inject station name/description into the natural-language prompt. Unique to LLM-based models. Zero parameter overhead.

### 5. Hidden-state init (1 model)
LSTM: per-entity `(h_0, c_0)` — classical RNN identity conditioning.

### 6. ETS-level bias (1 model)
ETSformer: per-station offset on the exponential-smoothing level — domain-aligned with classical ETS.

### 7. Output affine (2 models)
DLinear, LightTS: per-station `(N, pred_len)` bias/scale on final output. Minimal-parameter fallback.

### 8. Pre-existing custom system (2 model families)
Swiss-LSTM, Swiss-Transformer: independent entity system with embedding, feature_concat, and all transparent modes. Most complete entity support in the codebase.

### 9. Not applicable (1 model)
TimeMoE: zero-shot only — no training, no learnable parameters.

## Prerequisite work summary

Six models need adapter creation or mixin retrofit before entity injection can be implemented:

| Model | Work needed | Pattern to follow |
|---|---|---|
| LSTM (vanilla) | Add `EntityAwareMixin` to existing adapter | `MambaAdapter` (mamba_model.py:80) |
| ETSformer | Expand adapter + add mixin | `MambaAdapter` |
| LightTS | Create adapter from scratch | `MambaAdapter` |
| Reformer | Create adapter from scratch | `TransformerAdapter` (transformer.py) |
| GPT4TS | Create adapter from scratch | `MambaAdapter` |
| NST | Create adapter from scratch | `TransformerAdapter` |

## Parameter overhead summary

| Model | Entity embedding size | Example (862 stations) |
|---|---|---|
| Most models | `N × d_model` | 862 × 512 = 441K |
| DLinear | `N × (pred_len + pred_len)` | 862 × 192 = 166K |
| LightTS | `N × pred_len` | 862 × 96 = 83K |
| LSTM | `2 × N × n_layers × d_model` | 2 × 862 × 2 × 64 = 221K |
| TimeXer | `N × d_model` (replaces shared glb_token) | 862 × 512 = 441K |
| TimeLLM (H4) | 0 (text prompt) | 0 |
| TimeMoE | N/A | N/A |

All overheads are negligible relative to model size (typically <1%).

## Parity testing strategy

All entity injection designs support a **zero-init parity test**: initialize `entity_embed.weight` to zeros → output must be bit-exact with the non-entity baseline. This is the standard validation approach across all deep-dive files.

## Key cross-cutting risks

1. **Channel-order assumption**: `ChannelEntityWrapper` and PatchTST's `_inject_entity_after_patch` use `arange(n_vars)` as entity IDs. Safe under stable channel order; breaks under permutation. Fix: plumb per-sample `entity_ids` from data layer.

2. **Transparent-mode silent no-op**: `make_entity_features` requires `station_name` to be set. CSV/PEMS loaders set `station_ids` but not `station_name` → transparent modes silently produce no entity features.

3. **Swiss multi-channel split**: does not pass `identifier_mode`/`id_integration`/`station_ids` to the internal `TimeSeriesDataset` constructor.

4. **Frozen-backbone models** (GPT4TS, TimeLLM, TimeMoE): entity signal enters via `inputs_embeds` but only affects trainable components (LN for GPT4TS, reprogramming for TimeLLM). TimeMoE has no trainable components at all.
