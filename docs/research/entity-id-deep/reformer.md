# Reformer — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | Reformer |
| Paper URL | https://openreview.net/forum?id=rkgNKkHtvB |
| Year / venue | ICLR 2020 — Kitaev et al., *Reformer: The Efficient Transformer* |
| Official repo | https://github.com/google/trax/tree/master/trax/models/reformer |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py |
| This-repo adapter | `liulian/models/torch/reformer.py` |
| Runtime key | `reformer` |
| Benchmark key | `Reformer` |

## 2. Architecture primer

Reformer replaces full `O(L²)` attention with **LSH (locality-sensitive hashing) attention** at `O(L log L)`. Encoder-only design in the TSL port; for forecasting, the decoder placeholder is concatenated to the encoder input.

```
x_enc (B, L, C)                           x_dec (B, label_len+pred_len, C)  (placeholder)
     │                                              │
     │ (short_forecast only) per-sample RevIN       │
     │ normalize                                    │
     ▼                                              │
  concat on L axis:                                 │
  [x_enc, x_dec[:, -pred_len:, :]] ←────────────────┘                       # L76 / L96
     │
     ▼
 DataEmbedding (C → d_model) + time-feature + position                      # L40-43, L82
     │
     ▼
 Encoder (e_layers × EncoderLayer(ReformerLayer(LSH attn), d_model, d_ff)): # L46-61, L83
    • LSH bucket size 4, n_hashes 4 (defaults in L28)
    • chunked attention within hash buckets → O(L log L)
     │
     ▼
 projection: Linear(d_model → c_out)                                        # L70-72, L84
     │
     ▼
 (B, L+pred_len, c_out) → slice last pred_len
```

Complexity: `O((L+pred_len) · log(L+pred_len) · d_model)`. **Distinctive**: encoder operates on the concatenated `[enc; dec]` sequence, so the decoder slot is itself attended to during encoding.

## 3. This-repo audit

- `Model` (`reformer.py:22-148`) is a TSL port.
- **No Adapter class at all** — same as LightTS. Accessed via `importlib.import_module('liulian.models.torch.reformer')` from `pipeline.build_model:442`.
- **Audit findings:**
  - **Critical gap**: no adapter → no `EntityAwareMixin`. Same situation as LightTS.
  - Reformer has both `long_forecast` and `short_forecast` branches (L74-107). Short variant applies RevIN normalization before concat-with-dec; long does not. Entity injection sites must be chosen with awareness of this split.
  - `label_len` is NOT used in `long_forecast` (only `-pred_len:` slice of `x_dec` is concatenated) — standard TSL forecasting idiom.
  - Search spaces present in `liulian/optim/search_spaces.py:828-833`.

## 4. Upstream reference

Official trax Reformer is for NLP; TSL port adapts the LSH attention. Candidate hooks:

| Hook | Location (this repo) | Tensors in scope | Reformer-specific note |
|---|---|---|---|
| H1 pre-concat | before L76/L96 | `x_enc (B, L, C)` + `x_dec (B, ..., C)` | Must inject into both halves consistently. |
| H2 post-embed | after L82/L102 | `(B, L+pred_len, d_model)` | **Natural**: pre-encoder. Covers both enc and dec positions. |
| H3 per-layer in encoder | inside EncoderLayer | `(B, L+pred_len, d_model)` | Over-parameterized; LSH bucketing already mixes across positions. |
| H4 post-encoder | after L83/L103 | `(B, L+pred_len, d_model)` | Skips LSH attention. |
| H5 post-projection | after L84/L104 | `(B, L+pred_len, c_out)` | DLinear-style. |

## 5. Proposed ID injection design

**Prerequisite: create `ReformerAdapter(EntityAwareMixin, TorchModelAdapter)`.**

**Primary: H2 `add_to_embed` — per-entity additive embedding on `enc_out` after `DataEmbedding`, before LSH encoder. Applied in both `long_forecast` (L82) and `short_forecast` (L102) branches.**

Rationale:

1. Consistent with Transformer/Informer/Mamba primary design: bias the `d_model` representation at the pre-encoder stage. Aligns Reformer with the rest of the suite.
2. LSH bucketing groups tokens by query-key similarity. A per-entity additive bias in `d_model` space shifts the hash distribution uniformly per-sample → **same bucketing pattern for a given station**, but across-station the bucketing can differ. This is a subtle but potentially beneficial interaction: stations with similar patterns should end up in similar buckets.
3. In `short_forecast` (L87-107), normalization happens BEFORE concat-with-dec and BEFORE embedding. Inject after L102 (post-embed) to keep the entity bias invariant to normalization — same rule as Informer's short_forecast.

**Secondary: H5 `post_projection_affine` — `num_stations × c_out` bias on the post-projection tensor.**

- DLinear-style fallback; cheap; skips LSH.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-concat | Must handle the `x_dec` placeholder (often just zeros) + RevIN — attenuated and complicates both forecast variants. |
| H3 per-layer | LSH's bucketing already distributes position info globally per-layer; residual flow carries identity adequately. |
| Per-bucket entity hint | Would require touching `ReformerLayer`'s LSH logic — out-of-scope, violates the "wrap, don't modify vendor code" principle. |

## 6. Concrete code change sketch

File: `liulian/models/torch/reformer.py`
Functions: `Model.__init__` (L28-72), `Model.long_forecast` (L74-85), `Model.short_forecast` (L87-107), `Model.forward` (L132-148), new `ReformerAdapter`.

```python
from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.entity_mixin import EntityAwareMixin


class Model(nn.Module):
    def __init__(self, configs, bucket_size=4, n_hashes=4):
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

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if self._use_entity_embed and entity_ids is not None:
            enc_out = enc_out + self.entity_embed(entity_ids).unsqueeze(1)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        ...
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if self._use_entity_embed and entity_ids is not None:
            enc_out = enc_out + self.entity_embed(entity_ids).unsqueeze(1)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out * std_enc + mean_enc


class ReformerAdapter(EntityAwareMixin, TorchModelAdapter):
    def __init__(self, config):
        default_config = {'task_name': 'long_term_forecast', 'd_model': 128,
                          'n_heads': 8, 'e_layers': 2, 'd_ff': 256,
                          'dropout': 0.1, 'activation': 'gelu',
                          'embed': 'timeF', 'freq': 'h'}
        default_config.update(config)
        if 'c_out' not in default_config:
            default_config['c_out'] = default_config['enc_in']
        model_cfg = self._entity_model_config(default_config)
        model = Model(self._dict_to_namespace(model_cfg))
        super().__init__(model, default_config)
        self._init_entity_support(default_config)
```

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Requires adapter creation**. Then standard Transformer-style injection. |
| Parameter overhead | `num_stations × d_model` — tiny. |
| Parity test | Zero-init `entity_embed` ⇒ bit-exact baseline for both forecast variants. |
| LSH-attention interaction | Time-constant additive bias shifts all `L+pred_len` token representations by the same vector per sample → LSH hash is a function of `(q, k)` direction; a common additive bias mostly cancels in dot-product. **Net effect on bucketing: small.** Entity discrimination should still come through in residual + post-attention pathways. |
| Long-vs-short forecast parity | Two injection sites (L82 and L102) both need entity injection — easy to miss one in a patch. Suggest a helper method `_inject_entity(enc_out, entity_ids)`. |
| Task coverage | Imputation/anomaly/classification paths present (L109-130); patch analogously if expanding scope. |

## 8. Citations & uncertainty

- Paper: https://openreview.net/forum?id=rkgNKkHtvB
- Official repo: https://github.com/google/trax/tree/master/trax/models/reformer
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py
- This repo: `liulian/models/torch/reformer.py:22-148` (Model — no Adapter class). Dynamic import via `pipeline.build_model:442`.
- LSH attention internals: `liulian/models/torch/layers/attention.py::ReformerLayer` (not inspected in depth for this doc).

**Uncertainties:**
- Whether LSH bucketing actually benefits from per-sample entity shifts. The hash function's sign-of-random-projection design partially cancels common additive bias; the entity signal may only manifest in post-attention residual. Ablating H2 vs H5 would quantify how much the LSH block uses the entity.
- Bucket/hashes defaults (`bucket_size=4`, `n_hashes=4`) are TSL-chosen; shorter sequences may see LSH degenerate to full attention, making Reformer == Transformer. Entity design identical in that regime.
