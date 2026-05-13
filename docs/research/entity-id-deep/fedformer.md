# FEDformer — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | FEDformer |
| Paper URL | https://proceedings.mlr.press/v162/zhou22g.html |
| Year / venue | ICML 2022 — Zhou et al., *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting* |
| Official repo | https://github.com/MAZiqing/FEDformer |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py |
| This-repo adapter | `liulian/models/torch/fedformer.py` |
| Runtime key | `fedformer` |
| Benchmark key | `FEDformer` |

## 2. Architecture primer

FEDformer = Autoformer-style decomposed decoder + **frequency-domain attention** (FourierBlock / FourierCrossAttention). A random (or low-frequency) subset of Fourier modes is selected; attention is computed on those modes only, giving `O(L)` complexity.

```
x_enc (B, L, C)              x_mark_enc
      │                          │
      ▼                          ▼
DataEmbedding ──► enc_tokens (B, L, d_model)        # L56-62 (uses full DataEmbedding, not _wo_pos)
      │
      ▼
Encoder: AutoCorrelationLayer(FourierBlock) + decomp  # L99-116
      │
      ▼
enc_out (B, L, d_model)

decomp(x_enc) → seasonal_init, trend_init            # L157
trend_init_dec = concat(trend_init[-label_len:], mean_repeat_pred)
seasonal_init_dec = F.pad(seasonal_init[-label_len:], (0,0,0,pred_len))  # L159-162
      │
      ▼
DataEmbedding(seasonal_init_dec, x_mark_dec)          # L168
      │
      ▼
Decoder: AutoCorrelationLayer(FourierBlock) [self-att]
         + AutoCorrelationLayer(FourierCrossAttention) + decomp   # L118-142
      │
      ▼
trend_part + seasonal_part → slice last pred_len       # L173
```

Key differences vs Autoformer: (a) uses `DataEmbedding` *with* positional encoding (L56), (b) FourierBlock replaces AutoCorrelation — attention lives in frequency domain.

## 3. This-repo audit

- `Model` (`fedformer.py:37-220`) is a verbatim TSL port with the "Fourier" version hard-wired (the paper also has a "Wavelet" version that is absent in TSL). **No native entity hook.**
- `FEDformerAdapter` (`fedformer.py:223-270`) inherits `(EntityAwareMixin, TorchModelAdapter)`. Same plumbing as Autoformer.
- **Audit findings:**
  - `ChannelEntityWrapper` fixed-buffer and transparent-mode silent-no-op caveats apply identically.
  - FourierBlock operates on complex modes of the input along the **time axis**; a constant-along-time entity embedding has all its energy at mode 0 (DC) — frequency-domain attention will typically *not* select mode 0 if `mode_select='random'` or `'lowest'` with `modes=32` (it *may* include mode 0 depending on selection). This affects how entity signal survives attention.

## 4. Upstream reference

Official `FEDformer/models/FEDformer.py` mirrors TSL port. Candidate injection sites:

| Hook | Location (this repo) | Tensors in scope | FEDformer-specific note |
|---|---|---|---|
| H1 pre-decomp | before L157 | `x_enc (B, L, C)` | Smoothed by moving-avg decomp. |
| H2 post-embed (enc + dec) | after L163 + L168 | `(B, L, d_model)` | Pre-Fourier; entity signal at DC mode of each channel. |
| H3 post-encoder, pre-decoder cross | after L169 | `enc_out (B, L, d_model)` | Injects only into cross-attention K/V. |
| H4 trend-path bias | on `trend_init` (L158) | `(B, L_d, c_out)` | Bypasses Fourier entirely — only in trend. |
| H5 output-head affine | after L173 | `(B, L_d, c_out)` | Like DLinear H4. |

## 5. Proposed ID injection design

**Primary: H2 `add_to_embed` — per-entity additive embedding at post-`DataEmbedding`, on both encoder and decoder sides.**

Rationale:

1. Same `d_model`-level additive argument as Autoformer. Because decomp is applied inside each layer, time-constant entity embedding is routed to the trend branch automatically — propagates through the whole decoder into the final trend.
2. **Fourier-domain consideration:** A time-constant entity embedding (broadcast over L) is concentrated at mode 0 in the FFT. Whether Fourier attention attends to mode 0 depends on `mode_select_method`. This repo's default is `'random'` with `modes=32` (L43, L94); over `L/2` possible modes, mode 0 is included in roughly `32 / (L/2)` random draws. **Key observation:** even if the entity signal is *not* attended within FourierBlock, it survives the residual connection inside the block (AutoCorrelationLayer applies attention + residual). So identity is preserved regardless.
3. Preserves `O(L)` complexity; adds `num_stations × d_model` parameters.

**Secondary: H4 `trend_bias` — per-entity bias added directly to `trend_init` (L158).**

- Extremely cheap (`num_stations × c_out`). Targets the bypass path that the paper emphasises (decomposed trend is a core inductive bias).
- Competitive with H2 when the per-station difference is primarily baseline/slope rather than periodic.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-decomp | Decomp moving-avg smooths entity signal; wasted parameters. |
| H3 enc→dec cross only | Decoder self-attn doesn't see entity — asymmetric; weaker than H2. |
| H5 post-output affine | Fine as a cheap baseline, but doesn't leverage decomposition + Fourier interaction. |

## 6. Concrete code change sketch

File: `liulian/models/torch/fedformer.py`
Functions: `Model.__init__` (L43-153), `Model.forecast` (L155-174), `Model.forward` (L204-220)

```python
class Model(nn.Module):
    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
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
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :],
                              (0, 0, 0, self.pred_len))

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_len = seasonal_init.shape[1]
        if x_mark_dec is not None and x_mark_dec.shape[1] > dec_len:
            x_mark_dec = x_mark_dec[:, :dec_len, :]
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)

        if self._use_entity_embed and entity_ids is not None:
            e = self.entity_embed(entity_ids)            # (B, d_model)
            enc_out = enc_out + e.unsqueeze(1)
            dec_out = dec_out + e.unsqueeze(1)

        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        return trend_part + seasonal_part
```

Pipeline: skip `ChannelEntityWrapper` wrap when `id_integration == 'add_to_embed'`; thread `entity_ids` (already plumbed from `trainer.py:474-479`).

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes.** |
| Parameter overhead | `num_stations × d_model` — identical to Autoformer/Transformer. |
| Parity test | Zero-init entity_embed ⇒ baseline-identical output. |
| Fourier-mode interaction | Time-constant entity signal lives at FFT mode 0; may or may not be among the `modes` randomly selected. Residual connection guarantees survival regardless. Consider `mode_select_method='lowest'` to force mode 0 inclusion. |
| Decomposition routing | Same as Autoformer — time-uniform signal goes entirely to the trend branch inside each in-layer decomp. Compatible. |
| Position encoding | FEDformer uses `DataEmbedding` (with pos), unlike Autoformer's `_wo_pos`. Adding entity embedding adds a 4th term; `DataEmbedding` already sums value+pos+time so one more term is consistent. |

## 8. Citations & uncertainty

- Paper: https://proceedings.mlr.press/v162/zhou22g.html
- Official repo: https://github.com/MAZiqing/FEDformer
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py
- This repo: `liulian/models/torch/fedformer.py:37-220` (Model), `:223-270` (Adapter); wrapper in `liulian/models/torch/entity_mixin.py:52-257`.
- Related: STID https://arxiv.org/abs/2208.05233 (additive per-node identity); Autoformer (preceding decomp architecture).

**Uncertainties:**
- Sensitivity of gains to `mode_select_method`. Random-mode selection may stochastically drop the entity DC mode. Proposed mitigation: switch to `'lowest'` when `id_integration='add_to_embed'` is active, or enlarge `modes` default.
- Whether the **Wavelet** version of FEDformer (not ported here) would need a different injection point. Out of scope for this repo until ported.
