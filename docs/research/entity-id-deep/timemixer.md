# TimeMixer — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | TimeMixer |
| Paper URL | https://openreview.net/pdf?id=7oLshfEIC2 |
| Year / venue | ICLR 2024 — Wang et al., *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting* |
| Official repo | https://github.com/kwuking/TimeMixer |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py |
| This-repo adapter | `liulian/models/torch/timemixer.py` |
| Runtime key | `timemixer` |
| Benchmark key | `TimeMixer` |

## 2. Architecture primer

TimeMixer's three ideas: (a) **multi-scale downsampling** (pyramid of `1×`, `1/W×`, `1/W²×`, … via avg/max/conv pooling), (b) **past decomposable mixing** — at each scale decompose into seasonal + trend, then bottom-up-mix seasonal and top-down-mix trend across scales, (c) **channel independence (default=1)** — same `1→d_model` embedding applied independently per channel, channels folded into batch dimension as `B*N`.

```
x_enc (B, L, C)
     │
     ├── per-scale RevIN: Normalize[i] for i in [0 .. down_sampling_layers]
     │
     ▼
 Multi-scale downsample: x_enc → [x0 (L), x1 (L/W), x2 (L/W²), ...]    # L314-359
     │
     │  (channel_independence=1)
     ▼
 Reshape each scale: (B, T, C) → (B*C, T, 1)                           # L370-373
     │
     ▼
 DataEmbedding_wo_pos(1 → d_model) on each scale                       # L210-216 (CI branch)
     │
     ▼
 pdm_blocks (e_layers):
    at each scale: decomp (moving_avg or DFT_series_decomp)            # L134-139
       → season_list, trend_list
       → MultiScaleSeasonMixing (bottom-up)                            # L148
       → MultiScaleTrendMixing (top-down)                              # L149
       → out = season + trend; out_cross_layer MLP; add residual
     │
     ▼
 For each scale i:
    predict_layers[i]: Linear(seq_len/W^i → pred_len)
    projection_layer: Linear(d_model → 1)                               # L255-260 (CI branch)
     │
     ▼
 Sum across scales → reshape (B, C, pred_len) → (B, pred_len, C) → de-norm
```

Complexity mostly linear in L, repeated across scales. **Critical property for entity ID:** in the channel-independence branch, each channel becomes an independent sample (`B*N` batch). No cross-channel interaction — same identifier-blindness issue as PatchTST.

## 3. This-repo audit

- `Model` (`timemixer.py:187-556`) is a verbatim TSL port. **No native entity hook.**
- `TimeMixerAdapter` (`timemixer.py:559-606`) inherits `(EntityAwareMixin, TorchModelAdapter)`. Default `channel_independence=True` (L590).
- **Audit findings (relevant):**
  - In channel-independence mode, reshape to `(B*N, T, 1)` before embedding folds channel → batch. `EntityWrapper` (per-entity-split variant) — when used — would have to understand this batch expansion; `ChannelEntityWrapper` wraps **before** this reshape happens.
  - Per-scale `Normalize` layers (L228-239) apply *per-channel* standardization — this is already an implicit per-variate adjustment, but does not encode identity (only first/second moment).
  - Transparent modes are still silent no-ops on CSV/PEMS due to `station_name` not being set.

## 4. Upstream reference

Official TimeMixer matches TSL port. Candidate hooks (CI branch):

| Hook | Location (this repo) | Tensors in scope | TimeMixer-specific note |
|---|---|---|---|
| H1 pre-downsample | before L362 | `(B, L, C)` | Gets downsampled across scales; signal spreads to every scale automatically. |
| H2 post-CI reshape | after L371 (`(B*N, T, 1)`) | `(B*N, T, 1)` | One-dim channel — entity bias here acts *per-channel-per-sample*. |
| H3 post-embed | after L391 (CI) | `(B*N, T, d_model)` | **Natural** — pre-PDM. |
| H4 per-PDM-block | inside pdm_blocks | same shape | Over-parameterized. |
| H5 post-PDM, pre-predict | after L398 | per-scale list of `(B*N, T_i, d_model)` | Per-scale entity bias. |
| H6 post-projection | after L413 (CI) | `(B, N, pred_len)` | Like DLinear H4. |

## 5. Proposed ID injection design

**Primary: H3 `add_to_embed_ci` — per-entity additive embedding added to every scale's `enc_out` in the `(B*N, T_i, d_model)` layout, before pdm_blocks.**

Rationale:

1. Channel-independence mode makes each `(B*N, T, 1)` batch entry a single-channel slice of one original channel. For multi-channel split, each of the `N` channels *is* a station — so the entity id for entry `b*N + n` is station `n` (when batch order is preserved).
2. Adding `nn.Embedding(num_stations, d_model)` keyed by the channel index (or by per-sample `entity_ids`) at this point restores identity that channel-independence throws away. This is exactly STID's motivation for per-node embedding (STID is conceptually closest to post-CI per-channel embedding).
3. Must be applied to **every scale** simultaneously. The scales share a single `enc_embedding`, so the entity embedding can be shared too — one `nn.Embedding` used at every scale. This keeps parameter count unchanged.

**Secondary: H6 `post_output_affine` — per-entity `(num_stations × pred_len)` bias/scale on the final `(B, N, pred_len)` output.**

- DLinear-style fallback; cheap; entity signal skips the mixing.
- Good baseline; PDM cannot leverage identity this way.

**Tertiary: H2 `concat_after_ci_reshape` — concatenate a one-hot-like per-entity 1-D feature onto the `1`-channel axis before embedding.**

- Equivalent to transparent `onehot` mode; no new design needed if `station_name` plumbing is fixed upstream.
- Redundant with H3 if H3 is implemented.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-downsample | Entity signal propagates by construction to every scale — but is averaged by `avg`/`conv` pooling. OK in principle but less direct than H3. |
| H4 per-PDM-block | PDM blocks use residual connections → identity propagates; extra per-block embedding is wasteful. |
| H5 per-scale pre-predict | Per-scale separate embeddings would explode parameters with no clear motivation (all scales see the same station). |

## 6. Concrete code change sketch

File: `liulian/models/torch/timemixer.py`
Functions: `Model.__init__` (L193-292), `Model.forecast` (L361-403), `Model.forward` (L539-556)

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
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        ...
        # after the CI reshape loop, x_list is a list of (B*N, T_i, 1) per scale
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        if self._use_entity_embed and self.channel_independence:
            # CI reshaped to (B*N, T, d_model): entity index = channel index n
            # entity_ids is (B,) — broadcast per-channel
            if entity_ids is None:
                ids = torch.arange(self.enc_in, device=enc_out_list[0].device)
                ids = ids.unsqueeze(0).expand(B, -1).reshape(-1)  # (B*N,)
            else:
                ids = entity_ids  # must already be (B*N,) or (N,) broadcast-compatible
            emb = self.entity_embed(ids)                    # (B*N, d_model)
            enc_out_list = [enc_out + emb.unsqueeze(1) for enc_out in enc_out_list]

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        ...
```

Pipeline: skip `ChannelEntityWrapper` wrap when `id_integration='add_to_embed'`; thread `entity_ids`.

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes**, with moderate care around CI reshape and per-scale broadcast. |
| Parameter overhead | `num_stations × d_model` — default `d_model=32` so ~28K params for 862 stations. Tiny. |
| Parity test | Zero-init ⇒ bit-exact. |
| Channel-independence compatibility | Entity injection *restores* per-channel identity that CI throws away — this is exactly where identity injection matters most. Strongest qualitative case in the suite. |
| Multi-scale sharing | Shared `entity_embed` across scales: correct (one station has one identity). Do NOT allocate per-scale embeddings — would overfit. |
| Non-CI mode interaction | When `channel_independence=0`, embedding's `configs.enc_in` input covers all channels; adding `entity_embed` is then per-sample (like Transformer). Separate code path. |
| Normalize layers | RevIN-style per-sample normalization is applied BEFORE downsample/embedding; entity embedding added post-embedding is not affected. |

## 8. Citations & uncertainty

- Paper: https://openreview.net/pdf?id=7oLshfEIC2
- Official repo: https://github.com/kwuking/TimeMixer
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py
- This repo: `liulian/models/torch/timemixer.py:187-556` (Model), `:559-606` (Adapter); wrapper via `liulian/models/torch/entity_mixin.py:52-257`.
- STID https://arxiv.org/abs/2208.05233 — closest ablation evidence for CI + node embedding.

**Uncertainties:**
- Whether the `pdm_blocks` residual connections alone are enough to carry entity signal across scales, or whether per-scale re-injection helps. Start with shared embedding (one injection pre-PDM), ablate per-scale later.
- In multivariate (non-CI) mode, entity injection is less obviously useful because attention-like interactions are absent and cross_layer MLP mixes channels already. Default to CI mode for traffic/electricity.
