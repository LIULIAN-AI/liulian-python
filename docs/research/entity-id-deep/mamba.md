# Mamba — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | Mamba |
| Paper URL | https://arxiv.org/abs/2312.00752 |
| Year / venue | COLM 2024 — Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* |
| Official repo | https://github.com/state-spaces/mamba |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py |
| This-repo adapter | `liulian/models/torch/mamba_model.py` (re-export shim `liulian/models/torch/mamba.py`) |
| Runtime key | `mamba` |
| Benchmark key | `Mamba` |

## 2. Architecture primer

Mamba replaces attention with a **selective state space model (SSM)**. The key idea: make the SSM parameters `(Δ, B, C)` input-dependent, so the model can selectively propagate or forget information. Linear complexity in `L`.

```
x_enc (B, L, C)
     │
     ├── per-sample RevIN normalize                    # L59-64
     │
     ▼
 DataEmbedding (C → d_model) + time-feature + position  # L41-47, L66
     │
     ▼
 MambaBlock (external `mamba_ssm` package)               # L49-54, L67
    • selective conv1d over L (kernel `d_conv`)
    • selective SSM state `d_state = d_ff`
    • expansion factor `expand`
    • input-dependent Δ, B, C generation
     │
     ▼
 out_layer: Linear(d_model → c_out)                      # L56, L68
     │
     ▼
 x_out (B, L, c_out) * std_enc + mean_enc → slice last pred_len
```

Complexity: `O(L · d_model · d_state)` (linear in L). Single-block design (e_layers ignored in TSL port — only one `MambaBlock`). Forecast-only in this port.

## 3. This-repo audit

- `Model` (`mamba_model.py:25-77`) is a TSL port wrapping the external `mamba_ssm.Mamba` block. **No native entity hook.**
- `MambaAdapter` (`mamba_model.py:80-118`) inherits `(EntityAwareMixin, TorchModelAdapter)`. Standard plumbing.
- **Audit findings:**
  - External-dep gate: `mamba_ssm` must be installed. Testing entity injection requires the package — mock-test via `channel_independent` split on small synthetic data.
  - `ChannelEntityWrapper` fixed-buffer `arange(n_vars)` caveat applies.
  - Transparent modes (`onehot` etc.) inflate `enc_in`; MambaBlock processes it unchanged.
  - `label_len` / decoder inputs are unused (encoder-only).

## 4. Upstream reference

Official Mamba (vision/NLP) is identical. For time series, TSL simplification: single MambaBlock, RevIN wrap. Candidate hooks:

| Hook | Location (this repo) | Tensors in scope | Mamba-specific note |
|---|---|---|---|
| H1 pre-embed | before L66 | `x_enc (B, L, C)` | Gets absorbed by DataEmbedding's linear. |
| H2 post-embed | after L66 | `(B, L, d_model)` | **Natural**: pre-SSM. |
| H3 post-SSM | after L67 | `(B, L, d_model)` | Post-selective-scan; entity bias no longer modulates SSM state. |
| H4 post-projection | after L68 | `(B, L, c_out)` | DLinear-style output-only. |

## 5. Proposed ID injection design

**Primary: H2 `add_to_embed` — per-entity additive embedding added to `enc_out` after `DataEmbedding`, before MambaBlock.**

Rationale:

1. The selective SSM's `Δ(x), B(x), C(x)` are input-dependent — adding a station-specific bias before the block means each station gets its own **selection pattern**. This is the Mamba-native analogue of "conditional attention" in Transformer.
2. A time-constant bias to `(B, L, d_model)` adds a DC component that the selective conv1d naturally carries through (equivalent to a learnable per-station offset in every channel of the SSM input).
3. Post-block injection (H3) bypasses the state-space recurrence entirely — defeating the purpose.

**Secondary: H4 `post_output_affine` — `num_stations × c_out` bias on `x_out` before de-normalization.**

- DLinear-style fallback; cheap; skips SSM.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-embed | Absorbed by DataEmbedding's linear; attenuated. |
| H3 post-SSM | Misses Mamba's main gain — the selective scan is where station-specific state should form. |
| Per-entity `Δ`/`B`/`C` parameters | Would require editing `mamba_ssm` internals — breaks the external-dep abstraction. |

## 6. Concrete code change sketch

File: `liulian/models/torch/mamba_model.py`
Functions: `Model.__init__` (L31-56), `Model.forecast` (L58-71), `Model.forward` (L73-77)

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

    def forecast(self, x_enc, x_mark_enc, entity_ids=None):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)                         # (B, L, d_model)
        if self._use_entity_embed and entity_ids is not None:
            e = self.entity_embed(entity_ids)                         # (B, d_model)
            x = x + e.unsqueeze(1)                                    # broadcast over L
        x = self.mamba(x)
        x_out = self.out_layer(x)
        return x_out * std_enc + mean_enc
```

Pipeline: skip `ChannelEntityWrapper` wrap when `id_integration='add_to_embed'`; thread `entity_ids`.

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes**, natural fit post-DataEmbedding. |
| Parameter overhead | `num_stations × d_model` — tiny (e.g. 862 × 128 = 110K). |
| Parity test | Zero-init `entity_embed` ⇒ bit-exact baseline. |
| External-dep | Requires `mamba-ssm` — tests must be skipped gracefully without it (existing adapter already guards import at L33). |
| Selective-scan interaction | Entity bias enters `Δ(x), B(x), C(x)` computation (input-dependent) → each station's input-projection pattern subtly shifts. This is the intended architectural gain. **Risk:** if `Δ` becomes saturated for a specific station due to large bias, gradients through SSM may destabilize. Mitigation: init `entity_embed` small (`N(0, 0.02)`). |
| Forecast-only port | Hooks in `forecast` only — no imputation/classification paths to update. |

## 8. Citations & uncertainty

- Paper: https://arxiv.org/abs/2312.00752
- Official repo: https://github.com/state-spaces/mamba
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py
- This repo: `liulian/models/torch/mamba_model.py:25-77` (Model), `:80-118` (Adapter); wrapper via `liulian/models/torch/entity_mixin.py:52-257`.
- Related: S-Mamba for time series (arxiv:2403.11144) uses Mamba on variate tokens (iTransformer-style) — its variate-as-token layout would make H2 a per-variate bias, matching our iTransformer design.

**Uncertainties:**
- Whether input-dependent SSM parameterization dilutes a simple additive bias more than attention does. Mamba's conv1d/gate mechanism could suppress a time-constant signal. Ablation idea: compare H2 to H4 (output-affine) to isolate whether the SSM actually uses the entity bias or just passes it through.
- Per-variate vs per-sample entity id: Mamba here is not channel-independent; `enc_in` covers all channels. Per-sample `entity_ids` (which station produces this batch) is the natural semantic.
