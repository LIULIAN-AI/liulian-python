# TimesNet — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | TimesNet |
| Paper URL | https://openreview.net/pdf?id=ju_Uqw384Oq |
| Year / venue | ICLR 2023 — Wu et al., *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis* |
| Official repo | https://github.com/thuml/TimesNet |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py |
| This-repo adapter | `liulian/models/torch/timesnet.py` |
| Runtime key | `timesnet` |
| Benchmark key | `TimesNet` |

## 2. Architecture primer

TimesNet converts the 1D input time series into **2D grids** along `top_k` discovered FFT periods, then applies an **Inception-style 2D convolution** on each grid. Results are aggregated by period-weighted softmax.

```
x_enc (B, L, C)
     │
     ├── per-sample RevIN normalize                       # L151-154
     │
     ▼
 DataEmbedding (C → d_model)                              # L156 → (B, L, d_model)
     │
     ▼
 predict_linear: Linear(seq_len → seq_len+pred_len)  (on L axis)  # L157
     │
     ▼
 Stack of TimesBlocks (e_layers):                        # L158-159
    For each: FFT_for_Period → top_k periods
              reshape (B, T, d_model) → (B, d_model, cycles, period)
              Inception_Block_V1 2D conv → reshape back
              softmax-weighted sum over top_k paths + residual
     │
     ▼
 projection: Linear(d_model → c_out)                      # L160
     │
     ▼
 dec_out (B, L+pred_len, c_out) → de-norm → slice last pred_len  # L161-168, L236
```

Complexity depends on `top_k × cycles × period`; in practice comparable to `O(L · d_model · k)` plus 2D conv. Notably, **channels mix at `DataEmbedding`**; inside `TimesBlock`, the 2D convolution convolves over `(cycles, period)` — the channel axis `N` in `(B, T, N)` is the `d_model` axis (not raw channels; those were already absorbed).

## 3. This-repo audit

- `Model` (`timesnet.py:110-246`) is a verbatim TSL port. **No native entity hook.**
- `TimesNetAdapter` (`timesnet.py:249-295`) inherits `(EntityAwareMixin, TorchModelAdapter)`. Standard plumbing.
- Adapter sets `pred_len=0` for non-forecast tasks (L289-290) — important because `TimesBlock` reshapes on `seq_len + pred_len`.
- **Audit findings:**
  - `ChannelEntityWrapper` fixed-buffer and transparent-mode silent-no-op caveats apply.
  - Per-variate structure is collapsed into `d_model` after embedding — injecting entity signal at `d_model` level means all `top_k` period-grids inherit it.

## 4. Upstream reference

Official TimesNet repo matches TSL port. Candidate hooks:

| Hook | Location (this repo) | Tensors in scope | TimesNet-specific note |
|---|---|---|---|
| H1 pre-embed | before L156 | `x_enc (B, L, C)` | Gets absorbed by embedding; attenuated. |
| H2 post-embed, pre-predict_linear | between L156 and L157 | `(B, L, d_model)` | **Most natural.** Entity signal added before FFT-period detection. |
| H3 post-predict_linear, pre-TimesBlock | between L157 and L158 | `(B, L+pred_len, d_model)` | Entity bias is now distributed across the future-prediction slots too. |
| H4 inside TimesBlock (pre-2D conv) | L96 | `(B, d_model, cycles, period)` | Per-layer injection; over-parameterized. |
| H5 post-projection output bias | after L160 | `(B, L+pred_len, c_out)` | Like DLinear H4. |

## 5. Proposed ID injection design

**Primary: H3 `add_to_embed_extended` — add per-entity embedding to `enc_out` after `predict_linear`, broadcast over the full `L+pred_len` axis.**

Rationale:

1. `predict_linear` (L157) is a Linear applied on the time axis — it expands `seq_len → seq_len + pred_len`. Injecting *after* this layer means the entity signal also lives in the "future" half of the tensor, where the predictions are ultimately read out.
2. FFT in `FFT_for_Period` (L25) detects periods from the tensor at `(B, T, N)`; a time-constant entity embedding lives at DC — does not spuriously produce new periods (same zero-frequency argument as Autoformer/FEDformer).
3. Each TimesBlock's 2D convolution will pool the entity signal (constant across `cycles` and `period` axes after reshape) into a uniform per-channel bias — a clean "per-station offset" interpretation.

**Secondary: H2 `add_to_embed` — same idea, but before `predict_linear`.**

- Slight risk that `predict_linear` (L157) reshapes time axis nontrivially. Since it's a Linear on length, a time-constant bias remains time-constant. So H2 is roughly equivalent to H3; H3 is preferred for clarity.

**Tertiary: H5 `post_output_affine` — `num_stations × c_out` bias added to final output.**

- DLinear-style fallback. Cheap; skips TimesBlock entirely for identity.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-embed | Absorbed by `DataEmbedding`; weaker. |
| H4 per-TimesBlock | No clear architectural motivation; residual connection would already propagate entity signal. |

## 6. Concrete code change sketch

File: `liulian/models/torch/timesnet.py`
Functions: `Model.__init__` (L116-148), `Model.forecast` (L150-168), `Model.forward` (L230-246)

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
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)                  # (B, L, d_model)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L+pred_len, d_model)

        if self._use_entity_embed and entity_ids is not None:
            e = self.entity_embed(entity_ids)                             # (B, d_model)
            enc_out = enc_out + e.unsqueeze(1)                            # broadcast over time

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)
        ...
        return dec_out
```

Pipeline: skip `ChannelEntityWrapper` wrap when `id_integration='add_to_embed'`; thread `entity_ids` through trainer (already done).

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes**, natural fit. |
| Parameter overhead | `num_stations × d_model` — TimesNet uses smaller `d_model=64` default, so overhead is tiny (e.g. 862 × 64 = 55K params). |
| Parity test | Zero-init `entity_embed` ⇒ bit-exact baseline. |
| FFT-period interaction | Time-constant entity signal lives at DC bin (zero-frequency); excluded from period selection by L38 (`frequency_list[0] = 0`). **No spurious-period risk.** |
| 2D conv interaction | Inside TimesBlock, entity bias is time-constant → maps to a constant across `(cycles, period)` axes after reshape → acts as a per-d_model bias inside Inception. Clean. |
| Task coverage | Patch in forecast paths; imputation/anomaly_detection/classification can be patched analogously but are out of scope for the current forecasting focus. |

## 8. Citations & uncertainty

- Paper: https://openreview.net/pdf?id=ju_Uqw384Oq
- Official repo: https://github.com/thuml/TimesNet
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
- This repo: `liulian/models/torch/timesnet.py:110-246` (Model), `:249-295` (Adapter); wrapper via `liulian/models/torch/entity_mixin.py:52-257`.
- STID https://arxiv.org/abs/2208.05233 remains the strongest cited ablation.

**Uncertainties:**
- TimesBlock's 2D convolution has local receptive fields; the entity signal lives globally (constant across grid). Whether the convolution can "ignore" this constant in favour of variation is not certain — likely the BN inside Inception handles it, but worth monitoring.
- Softmax period-weighted aggregation may amplify/dampen the entity signal differently at each `top_k` path. Test with fixed `top_k=1` to isolate.
