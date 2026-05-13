# Transformer — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | Vanilla Transformer (for time series) |
| Paper URL | https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf |
| Year / venue | NeurIPS 2017 — Vaswani et al., *Attention Is All You Need* |
| Official repo | https://github.com/tensorflow/tensor2tensor (original); time-series port at https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py |
| This-repo adapter | `liulian/models/torch/transformer.py` |
| Runtime key | `transformer` |
| Benchmark key | `Transformer` |

## 2. Architecture primer

Vanilla encoder-decoder Transformer applied to MTS. Tokens are **time steps**, not channels; each time step gets projected from `enc_in` channels to `d_model` via `DataEmbedding`, which adds value-embedding + positional-embedding + (optional) time-feature embedding.

```
x_enc (B, L, C)   x_mark_enc (B, L, T)
      │                 │
      ▼                 ▼
   DataEmbedding ──► enc_tokens (B, L, d_model)        # L40-46
      │
      ▼
   Encoder (full self-attn, e_layers)                    # L48-69
      │
      ▼
   enc_out (B, L, d_model)
                   │
   x_dec (B, label_len+pred_len, C)                  x_mark_dec
      │                                                 │
      ▼                                                 ▼
   DataEmbedding ──► dec_tokens (B, L_d, d_model)     # L75-81
      │
      ▼
   Decoder (causal self-attn + cross-attn, d_layers)     # L82-114
      │
      ▼
   dec_out (B, L_d, c_out)  →  slice last pred_len      # L171
```

Complexity is `O(L^2)`. Attention is over the temporal axis; channels are mixed only in the linear projection of `DataEmbedding`. This is relevant because a **time-uniform** entity signal will be attended-to in every head, while a **channel-specific** signal has to travel through the embedding projection first.

## 3. This-repo audit

- `Model` (`transformer.py:29-181`) is a verbatim TSL port. **No native entity hook.**
- `TransformerAdapter` (`transformer.py:184-230`) inherits `(EntityAwareMixin, TorchModelAdapter)` — delegates entity setup to the mixin (`entity_mixin.py:316-346`). Entity integration is therefore:
  1. **`identifier_mode != 'embedding'`** (e.g. `onehot`, `numeric_id`, `sinusoidal`, `coordinates`, `descriptors`): the **dataset** appends extra columns to `x_enc`, widening `enc_in`. Mixin widens `model_cfg.enc_in` via `_entity_model_config` (`entity_mixin.py` `_entity_model_config`) so `DataEmbedding` simply consumes the wider input. Clean, decoupled.
  2. **`identifier_mode == 'embedding'`**: model is wrapped by `EntityWrapper` (per-entity split) or `ChannelEntityWrapper` (multi-channel split) in `pipeline.build_model` (`pipeline.py:441-482`). Wrapper prepends `embedding_size` learned channels to `x_enc` *before* it reaches `DataEmbedding`, then projects back to the original `enc_in` before calling `self._model.forward`.
- **Audit findings (unchanged):**
  - `ChannelEntityWrapper` uses fixed `arange(N)` buffer (`entity_mixin.py:200-202`) and **ignores per-sample `entity_ids`** (L247-248) — same risk as DLinear.
  - Transparent modes depend on `station_name` being set by the data layer; CSV/PEMS loaders don't do this → silent no-op on `traffic`/`electricity`/`PEMS*`.

## 4. Upstream reference

Official TSL `models/Transformer.py` mirrors exactly the layout in this repo. Hook candidates:

| Hook | Location (this repo) | Tensors in scope |
|---|---|---|
| H1 pre-embed | before `enc_embedding` (L127) | `x_enc (B, L, C)` — raw input |
| H2 post-embed, pre-encoder | between L127 and L128 | `enc_out (B, L, d_model)` |
| H3 inside encoder (per-layer) | within `EncoderLayer.forward` | `(B, L, d_model)` |
| H4 post-encoder, pre-decoder | after L128 | `enc_out (B, L, d_model)` |
| H5 dedicated entity token | prepend to enc_tokens at L127 | token seq of length `L+1` |
| H6 output-head bias | after L135 / before L171 slice | `(B, L_d, c_out)` |

## 5. Proposed ID injection design

**Primary: H2 `add_to_embed` — add a broadcasted per-entity embedding to `enc_out` (and `dec_out`) at the `d_model` level.**

Rationale:

1. The only truly "entity-level" signal is a single vector per entity. At `d_model` granularity (after `DataEmbedding`), a learned `nn.Embedding(num_stations, d_model)` vector broadcast across time steps mirrors how *segment embeddings* (BERT) and *speaker embeddings* (ASR) are added in token space. Architecturally faithful: additive in the same space as value + positional + time-feature embeddings already live.
2. Preserves `O(L^2)` complexity; adds `num_stations × d_model` parameters (~440K for traffic, negligible).
3. Uses the per-sample `entity_ids` properly, side-stepping `ChannelEntityWrapper`'s fixed-buffer caveat.

**Secondary: H5 `entity_token` — prepend a learned entity token to the encoder sequence (length goes `L → L+1`).**

- Stronger (attention can dynamically route entity info), but changes attention shape and breaks all pre-computed positional embeddings if not handled carefully.
- Recommend as a follow-up ablation, not the default.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 concat extra channels (current wrapper behaviour) | Entity info is absorbed into `DataEmbedding`'s channel projection — it *works* (and is the fallback) but loses identity interpretability and inflates `enc_in`. |
| H3 per-layer injection | Over-parameterized; no evidence Transformer needs entity signal refreshed layer-by-layer when attention can carry it. |
| H6 output bias only | Too late — attention cannot see entity identity. Equivalent to a per-entity output-head shift. |

## 6. Concrete code change sketch

File: `liulian/models/torch/transformer.py`
Functions: `Model.__init__` (L35-124), `Model.forecast` (L126-135), `Model.forward` (L165-181)

```python
class Model(nn.Module):
    def __init__(self, configs):
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
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if self._use_entity_embed and entity_ids is not None:
            e = self.entity_embed(entity_ids)                 # (B, d_model)
            enc_out = enc_out + e.unsqueeze(1)                 # broadcast over L
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        ...
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        if self._use_entity_embed and entity_ids is not None:
            dec_out = dec_out + e.unsqueeze(1)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, entity_ids=None):
        ...
```

Also: `pipeline.build_model` must **not** wrap with `ChannelEntityWrapper` when `id_integration == 'add_to_embed'`, and `TorchModelAdapter._prepare_model_inputs` must thread `entity_ids` into the forward kwargs (already plumbed from `trainer.py:474-479` via `fwd_kwargs['entity_ids']`).

For **per-entity split** (one model per station), use `num_stations=1` and a constant id — equivalent to no-op, so fall back to the existing `EntityWrapper`. Multi-channel-split is where this shines.

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes.** Most natural of all TSL transformer variants (dense `d_model` space). |
| Parameter overhead | `num_stations × d_model` — e.g. traffic (862 × 512 = 441K) on a ~10M model. |
| Parity test | With zero-init entity embedding, outputs must match pre-change baseline (addition of zero is identity). |
| Channel-order risk | Uses per-sample `entity_ids`, so no arange(N) channel-order assumption. Safer than H1 wrapper. |
| Interaction with `DataEmbedding` scaling | `DataEmbedding` output is post-LayerNorm-less sum of three embeddings. Adding a fourth is fine; may require careful init (N(0, 0.02)) to avoid dominating. |
| Risk with `imputation`/`anomaly_detection`/`classification` heads | These tasks don't use the decoder; adding entity embed to `enc_out` still works but the forward path must be patched for each task. Keep injection inside `forecast` for now; explicitly skip for non-forecast tasks. |

## 8. Citations & uncertainty

- Paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py
- This repo: `liulian/models/torch/transformer.py:29-181` (Model), `:184-230` (Adapter); wrapper applied via `liulian/models/torch/entity_mixin.py:52-257`.
- Evidence for additive segment/speaker embeddings: BERT https://arxiv.org/abs/1810.04805 (segment type emb), SpeakerNet https://arxiv.org/abs/2010.12653 (speaker emb).

**Uncertainties:**
- Relative strength of `add_to_embed` vs `entity_token` is empirical. STID-style evidence favours additive for homogeneous MTS; entity-token wins when per-entity dynamics are heterogeneous.
- Per-entity split with a single-station model is degenerate; entity embedding collapses to a constant bias. Recommend falling back to `EntityWrapper` there.
