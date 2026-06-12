# Nonstationary Transformer (NST) — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | Non-stationary Transformer |
| Paper URL | https://openreview.net/pdf?id=ucNDIDRNjjv |
| Year / venue | NeurIPS 2022 — Liu et al., *Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting* |
| Official repo | https://github.com/thuml/Nonstationary_Transformers |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py |
| This-repo adapter | `liulian/models/torch/nonstationary_transformer.py` |
| Runtime key | `nonstationary_transformer` |
| Benchmark key | `Nonstationary_Transformer` |

## 2. Architecture primer

NST is a standard encoder-decoder Transformer with one addition: **De-stationary Attention**. Two MLP-based `Projector` modules learn `tau` (scaling) and `delta` (shift) factors from the raw input's mean/std statistics. These are passed into DSAttention to modulate attention scores, restoring non-stationary information lost by RevIN normalization.

```
x_enc (B, L, C)
     │
     ├── RevIN normalize                                              # L168-173
     │
     ├── tau_learner(x_raw, std_enc) → tau (B, 1)                    # L176-177
     ├── delta_learner(x_raw, mean_enc) → delta (B, seq_len)         # L178
     │
     ▼
 enc_embedding: DataEmbedding(C → d_model)                           # L77-80, L186
     │
     ▼
 Encoder(enc_out, tau=tau, delta=delta):                              # L83-102, L187
    DSAttention: attn_score = (Q·K^T + delta) / tau
     │
     ▼
 enc_out (B, L, d_model)
     │
     ▼
 dec_embedding: DataEmbedding(dec_in → d_model)                      # L106-109, L192
     │
     ▼
 Decoder(dec_out, enc_out, tau=tau, delta=delta):                     # L110-138, L193
    self-attn (DSAttention) + cross-attn (DSAttention)
     │
     ▼
 projection: Linear(d_model → c_out)                                 # L137
     │
     ▼
 de-normalize                                                        # L194
```

Complexity: `O(L² · d_model)` (same as vanilla Transformer). DSAttention adds per-step shift and per-sample scale to the attention score computation but doesn't change asymptotic complexity.

## 3. This-repo audit

- `Model` (`nonstationary_transformer.py:63-277`) — TSL port.
- `Projector` (`nonstationary_transformer.py:28-60`) — MLP for tau/delta.
- **No Adapter class** — accessed via dynamic import. Same gap as LightTS/Reformer/GPT4TS.
- **Audit findings:**
  - **Critical gap**: no adapter → no entity plumbing.
  - Architecture is structurally identical to vanilla Transformer (with DSAttention replacing standard attention). Entity injection design should mirror `transformer.md`'s H2 `add_to_embed`, with the added consideration that the `Projector` modules see raw statistics — entity identity could also flow through tau/delta.
  - `p_hidden_dims` and `p_hidden_layers` config parameters for the Projector are required (no defaults in Model — must be set by adapter).

## 4. Upstream reference

Official NST repo matches TSL port. Candidate hooks:

| Hook | Location (this repo) | Tensors in scope | NST-specific note |
|---|---|---|---|
| H1 pre-embed | before L186 | `x_enc (B, L, C)` | Gets embedded; attenuated. |
| H2 post-embed (enc) | after L186 | `(B, L, d_model)` | **Natural**: pre-encoder. Same as Transformer. |
| H2′ post-embed (dec) | after L192 | `(B, label_len+pred_len, d_model)` | Decoder injection for cross-attention. |
| H3 Projector entity-conditioning | tau/delta learners | `(B, 1)` / `(B, L)` | **NST-specific**: make tau/delta entity-conditioned. |
| H4 post-encoder | after L187 | `(B, L, d_model)` | Skips encoder attention. |
| H5 post-decoder | after L193 | `(B, label_len+pred_len, d_model)` | DLinear-style. |

## 5. Proposed ID injection design

**Prerequisite: create `NonstationaryTransformerAdapter(EntityAwareMixin, TorchModelAdapter)`.**

**Primary: H2+H2′ `add_to_embed_both` — per-entity additive embedding on both `enc_out` and `dec_out` after their respective `DataEmbedding`s, before encoder/decoder.**

Rationale:

1. Identical to Transformer/Informer/Autoformer primary design. NST's encoder-decoder architecture is standard Transformer with DSAttention — same entity injection logic applies.
2. Both encoder and decoder use DSAttention. Injecting entity bias into both ensures the attention's entity conditioning is complete.
3. `tau` and `delta` from the Projector operate on attention *scores* (scalars), while entity embedding operates on *representations* (`d_model`-dim). They are complementary, not redundant.

**Secondary: H3 `entity_conditioned_projector` — augment the Projector's input with entity embedding to make tau/delta per-entity.**

- **NST-specific architectural fit**: the Projector's purpose is to capture per-sample non-stationarity. Different stations have different non-stationarity patterns — conditioning the Projector on station ID is semantically aligned.
- Implementation: expand Projector's `2 * enc_in` input to `2 * enc_in + entity_dim`, concatenating the entity embedding.
- More invasive than H2; defer to H2 for initial implementation.

**Tertiary: H5 `post_output_affine`.**

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-embed | Same as Transformer: absorbed by DataEmbedding. |
| H4 post-encoder only | Skips DSAttention in encoder; breaks the main NST innovation. |
| Modifying DSAttention directly | `tau` and `delta` are scalars/per-step; adding entity conditioning there is less natural than H2 or H3. |

## 6. Concrete code change sketch

File: `liulian/models/torch/nonstationary_transformer.py`
Functions: `Model.__init__` (L69-162), `Model.forecast` (L164-195), new adapter.

```python
from liulian.models.torch.entity_mixin import EntityAwareMixin


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        ...
        self._use_entity_embed = (
            getattr(configs, 'identifier_mode', 'none') == 'embedding'
            and getattr(configs, 'id_integration', '') == 'add_to_embed'
        )
        if self._use_entity_embed:
            num_stations = getattr(configs, 'num_stations',
                                    getattr(configs, 'enc_in', 1))
            self.entity_embed = nn.Embedding(num_stations, configs.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        ...
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if self._use_entity_embed and entity_ids is not None:
            e = self.entity_embed(entity_ids).unsqueeze(1)
            enc_out = enc_out + e
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)
        ...
        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        if self._use_entity_embed and entity_ids is not None:
            dec_out = dec_out + e
        dec_out = self.decoder(dec_out, enc_out, ...)
        ...


class NonstationaryTransformerAdapter(EntityAwareMixin, TorchModelAdapter):
    def __init__(self, config):
        default_config = {
            'task_name': 'long_term_forecast', 'd_model': 512,
            'n_heads': 8, 'e_layers': 2, 'd_layers': 1, 'd_ff': 2048,
            'dropout': 0.1, 'activation': 'gelu', 'factor': 1,
            'embed': 'timeF', 'freq': 'h',
            'p_hidden_dims': [256, 256], 'p_hidden_layers': 2,
        }
        default_config.update(config)
        if 'c_out' not in default_config:
            default_config['c_out'] = default_config['enc_in']
        if 'dec_in' not in default_config:
            default_config['dec_in'] = default_config['enc_in']
        model_cfg = self._entity_model_config(default_config)
        model = Model(self._dict_to_namespace(model_cfg))
        super().__init__(model, default_config)
        self._init_entity_support(default_config)
```

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Requires adapter creation**. Then standard Transformer-style injection. |
| Parameter overhead | `num_stations × d_model` — identical to Transformer. |
| Parity test | Zero-init ⇒ bit-exact. |
| tau/delta interaction | Entity embedding and tau/delta are orthogonal: entity modifies *representations* (d_model vectors); tau/delta modifies *attention scores* (scalars). Both can coexist without interference. |
| Projector conditioning (H3) | More complex — requires widening Projector's input dim. Deferred to future ablation. |
| Non-forecast tasks | Imputation/anomaly/classification paths (L197-262) can be patched analogously but are out of scope. |

## 8. Citations & uncertainty

- Paper: https://openreview.net/pdf?id=ucNDIDRNjjv
- Official repo: https://github.com/thuml/Nonstationary_Transformers
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py
- This repo: `liulian/models/torch/nonstationary_transformer.py:63-277` (Model), `:28-60` (Projector). **No adapter class.**
- Structurally closest to `transformer.md` — same enc-dec architecture; DSAttention is the only architectural difference.

**Uncertainties:**
- Whether entity-conditioned Projector (H3) captures something beyond H2. Argument for: stations in different climatic regions have different non-stationarity profiles (e.g., seasonal vs trend-dominated). Argument against: the Projector's `enc_in`-dimensional input already captures per-sample statistics, which implicitly vary by station. Empirical question.
- The `p_hidden_dims`/`p_hidden_layers` config parameters have no defaults in Model — must be provided by the adapter. This is a minor implementation detail but could cause runtime errors if missed.
