# Informer — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | Informer |
| Paper URL | https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132 |
| Year / venue | AAAI 2021 (Best Paper) — Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting* |
| Official repo | https://github.com/zhouhaoyi/Informer2020 |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py |
| This-repo adapter | `liulian/models/torch/informer.py` |
| Runtime key | `informer` |
| Benchmark key | `Informer` |

## 2. Architecture primer

Informer is the Transformer plus three modifications: **ProbSparse attention** (O(L log L) sparse scoring), **self-attention distillation** (ConvLayer reduces `L → L/2` between encoder layers), and a **generative-style decoder** that predicts `pred_len` tokens in one forward pass (no autoregression). Token axis and channel handling are identical to Transformer: tokens are time steps, channels fuse inside `DataEmbedding`.

```
x_enc (B, L, C)                x_mark_enc
      │                            │
      ▼                            ▼
   DataEmbedding ──► enc_tokens (B, L, d_model)          # L43-49
      │
      ▼
   Encoder with ProbSparse + ConvLayer distil
     L → L/2 → L/4 ...                                    # L59-83
      │
      ▼
   enc_out (B, L', d_model)

   x_dec (B, label_len+pred_len, C) with pred_len zeros
      │
      ▼
   DataEmbedding → dec_tokens
      │
      ▼
   Decoder (masked ProbSparse self-attn, cross-attn)      # L85-117
      │
      ▼
   dec_out (B, label_len+pred_len, c_out)  → last pred_len # L204
```

Distillation means `enc_out` length is not `L` but rather `L/(2^(e_layers-1))` — relevant only if we pick an H3-style hook that lives inside the encoder stack.

## 3. This-repo audit

- `Model` (`informer.py:29-217`) is a verbatim TSL port. **No native entity hook.**
- `InformerAdapter` (`informer.py:220-305`) inherits `(EntityAwareMixin, TorchModelAdapter)`. Same plumbing as Transformer: `_entity_model_config` widens `enc_in` for transparent modes; `EntityWrapper`/`ChannelEntityWrapper` wraps for `embedding` mode (set in `pipeline.build_model`).
- `_prepare_model_inputs` (`informer.py:275-305`) constructs `x_dec` from `x_enc[-label_len:]` + zeros. **No entity-aware code here** — entity handling is left to mixin/wrapper.
- **Audit findings (same as Transformer):** `ChannelEntityWrapper` uses fixed-buffer `arange(N)`; transparent modes depend on `station_name` which CSV/PEMS never set.

## 4. Upstream reference

Official Informer2020 structure mirrors TSL port exactly. Candidate hooks:

| Hook | Location (this repo) | Tensors in scope |
|---|---|---|
| H1 pre-embed | before `enc_embedding` (L134) | `x_enc (B, L, C)` |
| H2 post-embed | after L134 (long_forecast) / L154 (short_forecast) | `enc_out (B, L, d_model)` |
| H3 between encoder ConvLayers | inside `Encoder.forward` | shapes shrink by 2× each |
| H4 post-encoder | after L136 | `enc_out (B, L', d_model)` |
| H5 dedicated entity token | prepend to enc_tokens at L134 | adds `+1` to L |
| H6 output-head bias | after L138 | `(B, L_d, c_out)` |

## 5. Proposed ID injection design

**Primary: H2 `add_to_embed` — per-entity additive embedding at the post-`DataEmbedding`, pre-encoder stage.**

Rationale:

1. Same logic as Transformer: `d_model` space is the right abstraction level for entity identity. Adding a broadcast `nn.Embedding(num_stations, d_model)` vector to `enc_out` (and to `dec_out`) is the smallest, most faithful modification.
2. **Crucially: inject *before* the distillation ConvLayers.** ConvLayers in `layers/transformer_blocks.py` are temporally local — they preserve channel identity per token but pool adjacent time steps. Because entity identity is time-uniform, adding it pre-distill means every surviving distilled token still carries it.
3. Preserves ProbSparse attention semantics — the sparse selector scores tokens by query-key magnitude; an entity-biased key is exactly what we want when multiple stations share a batch.

**Secondary: H1 `add_to_x` — add a per-entity signal to the raw input (pre-embed).**

- Cheaper (no new parameters beyond a `num_stations × C` table), but the resulting signal has to survive the embedding's conv/linear projection. Works but attenuated. Fine as a cheaper fallback knob.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H3 between distillation layers | Would require `e_layers` separate `nn.Embedding`s and has no architectural motivation. ProbSparse already attends broadly enough. |
| H4 post-encoder | Entity signal never reaches encoder self-attention — wastes the main representational channel. |
| H5 entity token | Changes `L` mid-distil (ConvLayer's kernel geometry breaks when `L` is odd). |
| H6 output-bias | Attention cannot see entity — same weakness as in Transformer. |

## 6. Concrete code change sketch

File: `liulian/models/torch/informer.py`
Functions: `Model.__init__` (L35-127), `Model.long_forecast` (L129-140), `Model.short_forecast` (L142-161), `Model.forward` (L201-217)

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

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        ...
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        if self._use_entity_embed and entity_ids is not None:
            e = self.entity_embed(entity_ids)                 # (B, d_model)
            enc_out = enc_out + e.unsqueeze(1)                 # broadcast over L
            dec_out = dec_out + e.unsqueeze(1)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out
```

`short_forecast` (`L142-161`) must apply the same injection **after normalization** (so the entity signal isn't scaled by per-sample std). Place injection at the same location as `long_forecast` — it's post-embedding.

Pipeline change: same as Transformer — skip `ChannelEntityWrapper` wrap when `id_integration == 'add_to_embed'`; thread `entity_ids` via `fwd_kwargs` (already done in `trainer.py:474-479`).

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes**, clean. |
| Parameter overhead | `num_stations × d_model` — identical to Transformer. |
| Parity test | Zero-init entity_embed; verify bit-exact output pre/post. |
| Distillation interaction | ConvLayer operates along time; entity signal is broadcast along time → survives distillation. |
| Short-forecast normalization | Must inject *after* the internal `mean_enc / std_enc` normalization (L144-149) — otherwise the entity signal gets normalized away. |
| Task coverage | Only patch `long_forecast` / `short_forecast`. Leave `imputation`/`anomaly_detection`/`classification` unpatched (entity info typically irrelevant for their current use in this repo). |

## 8. Citations & uncertainty

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
- Official repo: https://github.com/zhouhaoyi/Informer2020
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py
- This repo: `liulian/models/torch/informer.py:29-217` (Model), `:220-305` (Adapter); wrapper in `liulian/models/torch/entity_mixin.py:52-257`.

**Uncertainties:**
- Whether ProbSparse attention materially benefits from entity conditioning on homogeneous MTS (traffic/electricity) is unknown. STID-style additive embedding on vanilla Transformer is our closest analogue.
- Effect of entity embedding magnitude on ProbSparse query/key scoring — large entity norms could dominate sparsity selection. Recommend init std `0.02` and monitor.
