# Autoformer — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | Autoformer |
| Paper URL | https://openreview.net/pdf?id=I55UqU-M11y |
| Year / venue | NeurIPS 2021 — Wu et al., *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting* |
| Official repo | https://github.com/thuml/Autoformer |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py |
| This-repo adapter | `liulian/models/torch/autoformer.py` |
| Runtime key | `autoformer` |
| Benchmark key | `Autoformer` |

## 2. Architecture primer

Autoformer replaces dot-product self-attention with **AutoCorrelation** (FFT-based period discovery) and threads **progressive series decomposition** through every encoder/decoder layer: each block splits its input into seasonal + trend, sends seasonal through AutoCorrelation, and accumulates trend into a running prediction.

```
x_enc (B, L, C)           x_mark_enc
      │                       │
      ▼                       ▼
DataEmbedding_wo_pos   ──► enc_tokens (B, L, d_model)   # L53-59; note: NO positional
      │
      ▼
Encoder: each layer = AutoCorrelationLayer + decomp                # L61-83
      │
      ▼
enc_out (B, L, d_model)

decomp(x_enc) → (seasonal_init, trend_init)          # L148
trend_init_dec = concat(trend_init[-label_len:], mean_repeat_pred) # L150
seasonal_init_dec = concat(seasonal_init[-label_len:], zeros)      # L151-153
      │
      ▼
DataEmbedding_wo_pos(seasonal_init_dec) → dec_tokens       # L161
      │
      ▼
Decoder: each layer refines seasonal (AutoCorrelation) and
         accumulates trend_init through LayerNorm-less sum          # L96-130
      │
      ▼
dec_out = trend_part + seasonal_part                                # L166
      → slice last pred_len
```

Complexity: `O(L log L)`. Two distinctive properties:

1. **Position-free embedding (`DataEmbedding_wo_pos`)** — value + optional time-feature embeddings, no sinusoidal position. Relevant: if we add per-entity embedding at H2, we're effectively restoring some per-position structure — but it's *entity*-structure, not *time*-structure, so it's compatible.
2. **Trend channel flows through decoder as raw (B, L, c_out)** — not `d_model`. It bypasses attention entirely and is only summed at the end. Any injection into the trend path must be in `c_out` dimensions.

## 3. This-repo audit

- `Model` (`autoformer.py:34-222`) is a verbatim TSL port. **No native entity hook.**
- `AutoformerAdapter` (`autoformer.py:225-309`) inherits `(EntityAwareMixin, TorchModelAdapter)`. Same plumbing as Transformer/Informer.
- `_prepare_model_inputs` (`autoformer.py:280-309`) constructs decoder input similarly to Informer (label_len + pred_len zeros). No entity handling.
- **Audit findings (same as Transformer/Informer):** `ChannelEntityWrapper` fixed-buffer limitation; transparent modes are silent no-ops on CSV/PEMS.

## 4. Upstream reference

Official Autoformer repo matches TSL port. Hook candidates specific to Autoformer's architecture:

| Hook | Location (this repo) | Tensors in scope | Autoformer-specific note |
|---|---|---|---|
| H1 pre-decomp | before `self.decomp(x_enc)` (L148) | `x_enc (B, L, C)` | Same as DLinear's H1 — decomposition averages can smooth signal. |
| H2 post-embed (seasonal path) | after L155 / L161 | `(B, L, d_model)` | Affects seasonal branch only; trend carries unchanged. |
| H2′ post-embed both enc & dec | L155 + L161 | `(B, L, d_model)` and `(B, L_d, d_model)` | Keeps symmetry. |
| H3 trend-path injection | L150 / inside decoder accumulation | `trend (B, L_d, c_out)` | Trend is in `c_out` — cheaper, bias-like. |
| H4 output-head | after L166 | `(B, L_d, c_out)` | Same as DLinear's post-output affine. |

## 5. Proposed ID injection design

**Primary: H2′ `add_to_embed_both` — add the same per-entity embedding to both encoder and decoder token streams after `DataEmbedding_wo_pos`.**

Rationale:

1. Autoformer's AutoCorrelation picks up *periodicity*, not identity. Periodicity is a time-series property; identity is an external feature. Adding entity embedding at the `d_model` level means every AutoCorrelation head sees it as a per-sample bias that survives the seasonal→trend decomposition inside each layer (the decomp module is a moving-average low-pass; a time-uniform entity embedding stays entirely in the trend branch at each layer's decomp).
2. **This is an architecturally interesting property**: because Autoformer decomposes at every layer, an additive entity embedding is *automatically routed* to the trend component — which is precisely the path that (a) bypasses attention, (b) accumulates into the final prediction. Entity identity naturally acts as a per-station trend offset, which matches the intuition that stations differ most in their baseline level.
3. Preserves `O(L log L)`; adds `num_stations × d_model` parameters.

**Secondary: H3 `trend_bias` — add a learned `(num_stations, c_out)` vector to `trend_init` (L150) before it enters the decoder.**

- Cheapest: `num_stations × c_out` params. Interpretable: a per-station trend offset.
- Caveat: acts only on trend, not seasonality. If the gain is purely level-shift this is enough; if periodicity is also station-specific, H2′ subsumes it.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-decomp | Moving-average decomp (kernel=25) smooths the embedding — signal attenuated. |
| H4 output-head affine | Same as DLinear H4; loses the inductive-bias alignment that Autoformer's decomp provides. Still valid as a **baseline**. |
| H2 seasonal-only (dec only) | Asymmetric — encoder ignores entity. Harms cross-attention cheaply. |

## 6. Concrete code change sketch

File: `liulian/models/torch/autoformer.py`
Functions: `Model.__init__` (L41-140), `Model.forecast` (L142-167), `Model.forward` (L206-222)

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
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]],
                            device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if self._use_entity_embed and entity_ids is not None:
            e = self.entity_embed(entity_ids)           # (B, d_model)
            enc_out = enc_out + e.unsqueeze(1)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_len = seasonal_init.shape[1]
        if x_mark_dec is not None and x_mark_dec.shape[1] > dec_len:
            x_mark_dec = x_mark_dec[:, :dec_len, :]
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        if self._use_entity_embed and entity_ids is not None:
            dec_out = dec_out + e.unsqueeze(1)

        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        return trend_part + seasonal_part
```

Pipeline: skip `ChannelEntityWrapper` wrap when `id_integration == 'add_to_embed'`; propagate `entity_ids` (already plumbed via `trainer.py:474-479`).

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes**, and particularly well-aligned with Autoformer's trend/seasonal architecture. |
| Parameter overhead | `num_stations × d_model` — ~441K for traffic. |
| Parity test | Zero-init entity_embed ⇒ bit-exact match with pre-change baseline. |
| Trend/seasonal routing | Inside each layer's `decomp`, time-uniform entity signal (broadcast along L) is exactly the DC component — routed fully to the trend branch. Good inductive fit. |
| AutoCorrelation interaction | AutoCorrelation uses FFT on `(B, L, d_model)` per-head. A constant-along-L entity vector is zero-frequency after FFT on the L-axis → not selected as a period. **No spurious period artefacts** — verified on paper, should confirm with a synthetic test. |
| `_wo_pos` embedding note | Autoformer intentionally drops positional embedding. Adding entity embedding does not re-introduce *positional* structure — it adds *identity* structure. Compatible. |

## 8. Citations & uncertainty

- Paper: https://openreview.net/pdf?id=I55UqU-M11y
- Official repo: https://github.com/thuml/Autoformer
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py
- This repo: `liulian/models/torch/autoformer.py:34-222` (Model), `:225-309` (Adapter); wrapper in `liulian/models/torch/entity_mixin.py:52-257`.
- Supporting evidence for trend-as-entity-offset intuition: STID https://arxiv.org/abs/2208.05233 (shows most gain is from level/trend-level per-node embedding, not periodicity-level).

**Uncertainties:**
- Whether H3 `trend_bias` alone is sufficient (cheaper, interpretable) vs the full H2′. Claim: H3 ≥ H4 (DLinear affine) because it shifts trend *before* the decoder integrates it into seasonal; needs empirical check.
- Whether `decomp`'s moving-average kernel size (25 default) interacts with entity embedding in any way — on paper no, but worth logging during training.
