# GPT4TS — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | GPT4TS (One Fits All) |
| Paper URL | https://arxiv.org/abs/2302.11939 |
| Year / venue | NeurIPS 2023 — Zhou et al., *One Fits All: Power General Time Series Analysis by Pretrained LM* |
| Official repo | https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/GPT4TS.py |
| This-repo adapter | `liulian/models/torch/gpt4ts.py` |
| Runtime key | `gpt4ts` |
| Benchmark key | `GPT4TS` |

## 2. Architecture primer

GPT4TS fine-tunes **only LayerNorm and positional embedding** of a frozen GPT-2 backbone. Channel-independent: each channel is processed as a separate sample.

```
x_enc (B, L, C)
     │
     ├── _patchify:                                                  # L113-129
     │   RevIN normalize → permute (B, C, L) → unfold
     │   → (B*C, num_patches, patch_size)
     │
     ▼
 in_layer: Linear(patch_size → gpt2_dim=768)                        # L96, L134
     │
     ▼
 Frozen GPT-2 backbone (gpt_layers):                                # L59-74, L135
    • only ln_1, ln_2 (per-layer LayerNorm) + ln_f (final) + wpe are trainable
    • all attention + MLP weights frozen
     │
     ▼
 last_hidden_state (B*C, num_patches, gpt2_dim)
     │
     ▼
 reshape → out_layer: Linear(gpt2_dim * num_patches → pred_len)      # L99-101, L138-139
     │
     ▼
 reshape (B, C, pred_len) → permute (B, pred_len, C) → de-normalize  # L140-143
```

Complexity: GPT-2's `O(num_patches² · gpt2_dim)` attention, repeated per channel. **Channel-independent** like PatchTST — each of the `C` channels is an independent sample in the `B*C` batch.

## 3. This-repo audit

- `Model` (`gpt4ts.py:23-196`) — TSL-inspired port.
- **No Adapter class** — accessed via `importlib.import_module('liulian.models.torch.gpt4ts')` at `pipeline.build_model:442`.
- **Audit findings:**
  - **Critical gap**: no adapter → no `EntityAwareMixin` → entity plumbing does not work. Same situation as LightTS/Reformer.
  - Channel-independence via `_patchify` (L113-129): `(B, L, C) → (B*C, num_patches, patch_size)`. Same implications as PatchTST: entity signal must be per-channel within the `B*C` batch.
  - Frozen GPT-2 backbone — injecting entity embeddings **before** GPT-2's frozen layers means the entity signal modulates LayerNorm activations (since only LN is trainable). This is actually well-suited: LN's gain/bias, if conditioned on entity, learns per-station scale/shift at every layer.
  - `_patchify` returns `(means, stdev)` for de-normalization — entity bias should be injected after patching but before GPT-2, so de-norm is unaffected.

## 4. Upstream reference

Official repo matches TSL. Candidate hooks:

| Hook | Location (this repo) | Tensors in scope | GPT4TS-specific note |
|---|---|---|---|
| H1 pre-patchify | before L132 | `x_enc (B, L, C)` | Gets patchified + normalized; attenuated. |
| H2 post-in_layer | after L134 | `(B*C, num_patches, gpt2_dim)` | **Natural**: pre-GPT-2. Entity bias enters frozen attention + trainable LN. |
| H3 post-GPT-2 | after L135 | `(B*C, num_patches, gpt2_dim)` | Post-backbone; entity never reaches attention/LN. |
| H4 post-out_layer | after L139 | `(B*C, pred_len)` | DLinear-style. |

## 5. Proposed ID injection design

**Prerequisite: create `GPT4TSAdapter(EntityAwareMixin, TorchModelAdapter)`.**

**Primary: H2 `add_to_patch_embed` — per-entity additive `nn.Embedding(num_stations, gpt2_dim)` on `(B*C, num_patches, gpt2_dim)` after `in_layer`, before GPT-2.**

Rationale:

1. Channel-independent processing makes each channel a separate sample. Entity identity is lost unless explicitly injected — same STID motivation as PatchTST.
2. GPT-2's frozen attention weights compute fixed attention patterns. The entity bias shifts the input representation into a different region of GPT-2's latent space — equivalent to selecting a different "semantic context" for each station. The trainable LayerNorm then adapts the conditioning for each station.
3. This is architecturally analogous to PatchTST's `add_after_patch` — both inject identity into `(B*C, num_tokens, d_model)` layout. GPT4TS's "tokens" are patches, same as PatchTST.

**Secondary: H4 `post_output_affine` — `(num_stations, pred_len)` bias on `(B*C, pred_len)` output.**

- Cheap; skips GPT-2 backbone.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-patchify | RevIN normalization absorbs additive bias; patchify breaks it into uncontrolled chunks. |
| H3 post-GPT-2 | Entity signal never reaches attention or LayerNorm — defeats the purpose of using a pretrained backbone for contextual conditioning. |
| Modifying GPT-2's positional embedding per-entity | Invasive; `wpe` is shared across all channels in `B*C`; would require per-sample position bias — technically possible but breaks frozen-weight design. |

## 6. Concrete code change sketch

File: `liulian/models/torch/gpt4ts.py`
Functions: `Model.__init__` (L35-111), `Model.forecast` (L131-144), new `GPT4TSAdapter`.

```python
from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.entity_mixin import EntityAwareMixin


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        ...
        self._use_entity_embed = (
            getattr(configs, 'identifier_mode', 'none') == 'embedding'
            and getattr(configs, 'id_integration', '') == 'add_to_patch_embed'
        )
        if self._use_entity_embed:
            num_stations = getattr(configs, 'num_stations',
                                    getattr(configs, 'enc_in', 1))
            self.entity_embed = nn.Embedding(num_stations, self.gpt2_dim)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        x, means, stdev, B, C = self._patchify(x_enc)
        x = self.in_layer(x)                              # (B*C, num_patches, gpt2_dim)
        if self._use_entity_embed:
            if entity_ids is None:
                ids = torch.arange(C, device=x.device).repeat(B)
            else:
                ids = entity_ids                           # must be (B*C,)
            x = x + self.entity_embed(ids).unsqueeze(1)   # broadcast over patches
        outputs = self.gpt2(inputs_embeds=x).last_hidden_state
        ...


class GPT4TSAdapter(EntityAwareMixin, TorchModelAdapter):
    def __init__(self, config):
        default_config = {'task_name': 'long_term_forecast', 'd_model': 768,
                          'd_ff': 768, 'patch_len': 16, 'gpt_layers': 6,
                          'dropout': 0.1}
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
| Supports after revision? | **Requires adapter creation**. Then natural PatchTST-style injection. |
| Parameter overhead | `num_stations × gpt2_dim` — e.g. 862 × 768 = 662K. Modest but notable (GPT-2 has ~125M params, most frozen; trainable params are few). |
| Parity test | Zero-init `entity_embed` ⇒ bit-exact baseline. |
| Frozen-backbone interaction | Entity bias enters via `inputs_embeds`; frozen attention processes it as shifted input. LN (trainable) adapts — per-station information flows through LN gain/bias. Natural and intended. |
| Channel-independence | Same `arange(C)` pattern as PatchTST. Per-sample `entity_ids` as `(B*C,)` when available. |
| External-dep (transformers) | GPT-2 download required; existing code handles local-first with fallback (L58-74). |

## 8. Citations & uncertainty

- Paper: https://arxiv.org/abs/2302.11939
- Official repo: https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/GPT4TS.py
- This repo: `liulian/models/torch/gpt4ts.py:23-196` (Model — **no adapter class**). Dynamic import.
- Related: PatchTST (`add_after_patch`) — identical channel-independent patching layout; GPT4TS simply replaces the Transformer encoder with a frozen GPT-2.

**Uncertainties:**
- Whether frozen GPT-2 layers can meaningfully condition on entity identity through LN alone. The entity bias shifts the distribution of activations; LN then re-centers/re-scales. If LN capacity is insufficient, the entity signal may be partially washed out. Ablation: compare trainable-LN-only baseline to trainable-LN + entity embedding.
- `gpt2_dim=768` is large for entity embedding — may overparameterize for small station counts. Consider a smaller entity embedding projected up: `nn.Embedding(N, 64) → Linear(64, 768)`.
