# PatchTST — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | PatchTST |
| Paper URL | https://arxiv.org/abs/2211.14730 |
| Year / venue | ICLR 2023 — Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* |
| Official repo | https://github.com/yuqinie98/PatchTST |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py |
| This-repo adapter | `liulian/models/torch/patchtst.py` |
| Runtime key | `patchtst` |
| Benchmark key | `PatchTST` |

## 2. Architecture primer

PatchTST has two key ideas: **patch** the input into non-overlapping windows (tokens of fixed length, like ViT), and treat each **channel independently** (CI — "channel independence"). Every channel is processed by the same Transformer stack; tokens are `(patch_num, d_model)` per channel.

```
x_enc (B, L, C)
     │
     ├── normalize per-sample (RevIN)                    # L157-160
     │
     ▼
 permute → (B, C, L)                                     # L163
     │
     ▼
 PatchEmbedding: unfold into patches, Linear(patch_len → d_model)
                                  → (B*C, patch_num, d_model)  # L165
     │
     │ ← entity injection (native hook present)           # L166 (_inject_entity_after_patch)
     ▼
 Encoder (full self-attention over patch_num)             # L91-114
     │
     ▼
 (B*C, patch_num, d_model) → reshape (B, C, d_model, patch_num) # L172-176
     │
     ▼
 FlattenHead: Linear(d_model * patch_num → pred_len) per channel # L122-127
     │
     ▼
 dec_out (B, C, pred_len) → permute → (B, pred_len, C) → de-normalize
```

Complexity `O(patch_num^2 · d_model)` per channel; channels processed as an expanded batch dimension (`B*C`). This is fundamentally channel-independent — attention *never* sees other channels.

## 3. This-repo audit

- `Model` (`patchtst.py:55-302`) is a TSL port with **native entity support added** — first such case in the suite.
- **Native hook:** `_inject_entity_after_patch` (`patchtst.py:142-153`). When `identifier_mode=='embedding'` and `id_integration=='add_after_patch'`:
  - `self.entity_embedding = nn.Embedding(num_stations, d_model)` at L86-88
  - L150-152: `ids = torch.arange(n_vars).repeat(batch_size)` — **same fixed-arange channel-order assumption as `ChannelEntityWrapper`**.
  - L153: `enc_out + emb.unsqueeze(1)` adds entity embedding broadcast over patch_num.
  - Guarded for `split_mode=='multi_channel'` only (L77-80 raises if not).
- Applied in all four task paths: forecast (L166), imputation (L203), anomaly_detection (L235), classification (L267).
- Adapter `PatchTSTAdapter` (`patchtst.py:305-369`) inherits `(EntityAwareMixin, TorchModelAdapter)`. `_entity_model_config` widens `enc_in` for transparent modes.
- **Audit findings:**
  - Same `arange(n_vars)` channel-order assumption as `ChannelEntityWrapper`. Persists today because per-sample `entity_ids` from `trainer.py:474-479` are *not* consumed here — L151 uses `torch.arange` instead. If/when real variate-indexed ids become available, swap in.
  - `pipeline.build_model` has a dedicated exception at L419-425 that skips the `ChannelEntityWrapper` wrap when `id_integration=='add_after_patch'` — consistent with the native hook.

## 4. Upstream reference

Official PatchTST repo has a more elaborate backbone with "channel shared" and "channel independent" flags; TSL's port simplifies to CI-only. Injection candidates:

| Hook | Location (this repo) | Tensors in scope | PatchTST-specific note |
|---|---|---|---|
| H1 pre-patch | before L163/L165 | `x_enc (B, L, C)` | CI flattens channel → batch; signal must be per-channel-per-step. |
| H2 **post-patch** | after L165 | `(B*C, patch_num, d_model)` | **Current native hook.** Natural site — tokens are already per-channel. |
| H3 per-encoder-layer | inside Encoder | `(B*C, patch_num, d_model)` | Over-parameterized. |
| H4 post-encoder | after L170 | same shape | Misses self-attention interaction. |
| H5 post-head | after L179-180 | `(B, pred_len, C)` | Like DLinear H4 affine. |

## 5. Proposed ID injection design

**Primary: keep and refine the existing H2 `add_after_patch`.**

It is already correct architecturally. Refinements required:

1. **Replace fixed `arange(n_vars)` with per-sample `entity_ids`** when available. Current L151 hard-codes channel-index as entity-id; breaks on any channel-permuted batch. Migration path:
   ```python
   if entity_ids is None:
       ids = torch.arange(n_vars, device=enc_out.device).repeat(batch_size)
   else:
       # entity_ids shape (B, n_vars) or (B,)
       ids = _broadcast_entity_ids(entity_ids, batch_size, n_vars).reshape(-1)
   emb = self.entity_embedding(ids)
   ```
2. **Plumb `entity_ids` through `forward` / `forecast`** — currently not threaded, because trainer's `fwd_kwargs['entity_ids']` is accepted by `EntityWrapper`/`ChannelEntityWrapper`, not by the inner `Model`. Accept `entity_ids` in `forecast` / `forward` signature, pass to `_inject_entity_after_patch`.

Rationale for keeping H2:

- Patch tokens are **already per-channel** (each slot in `B*C` dim is one variate's patches). A learned `nn.Embedding(num_stations, d_model)` has perfect shape alignment.
- The authors' channel-independence is a double-edged sword: no cross-channel attention means the *only* way for the model to discriminate stations is via identity injection.
- STID-style per-node embedding was originally motivated exactly by PatchTST-like CI models that lose identity without it.

**Secondary: H5 `post_head_affine` — add `num_stations × pred_len` bias/scale to the per-channel forecast.**

- Equivalent to DLinear H4.
- Cheaper, but entity signal never reaches attention.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-patch | RevIN normalization in front of it kills a pure additive bias; attenuated. |
| H3 per-layer | No gain over single H2 injection + residual connections. |
| Dedicated entity-patch token | Changes patch_num dimension per channel; head is sized `d_model * patch_num`, so head output length would shift — invasive. |

## 6. Concrete code change sketch

File: `liulian/models/torch/patchtst.py`
Functions: `Model._inject_entity_after_patch` (L142-153), `Model.forecast` (L155-185), `Model.forward` (L286-302)

```python
def _inject_entity_after_patch(self, enc_out, n_vars, entity_ids=None):
    if not self._use_add_after_patch:
        return enc_out
    batch_size = enc_out.shape[0] // n_vars
    if entity_ids is None:
        ids = torch.arange(n_vars, device=enc_out.device).repeat(batch_size)
    else:
        # Accept (B,) scalar-per-sample or (B, n_vars) per-variate ids.
        if entity_ids.ndim == 1:
            # per-sample scalar → replicate for each of n_vars tokens
            ids = entity_ids.unsqueeze(1).repeat(1, n_vars).reshape(-1)
        else:
            ids = entity_ids.reshape(-1)
    emb = self.entity_embedding(ids)
    return enc_out + emb.unsqueeze(1)

def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
    ...
    enc_out, n_vars = self.patch_embedding(x_enc)
    enc_out = self._inject_entity_after_patch(enc_out, n_vars, entity_ids=entity_ids)
    ...

def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, entity_ids=None):
    ...
```

Also update `_prepare_model_inputs` to pass `entity_ids` if present in `inputs`.

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes — already partially supported.** Refinement focuses on per-sample `entity_ids` plumbing. |
| Parameter overhead | `num_stations × d_model` — already allocated (L88). |
| Parity test | With zero-init `entity_embedding` or `id_integration != 'add_after_patch'`, outputs bit-exact vs baseline. |
| Channel-order risk | **Currently present** (L151 uses `arange`). Fix: accept per-sample `entity_ids`. Fallback to arange is safe when ids absent. |
| CI compatibility | CI-preserving by construction — every variate is processed independently, only entity embedding couples identity to token. |
| Per-entity split | Explicitly disallowed (L77-80 raises). Valid: per-entity split already has one model per station so identity is baked into the weights. |

## 8. Citations & uncertainty

- Paper: https://arxiv.org/abs/2211.14730
- Official repo: https://github.com/yuqinie98/PatchTST
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py
- This repo: `liulian/models/torch/patchtst.py:55-302` (Model), `:305-369` (Adapter); native hook at `:142-153`.
- Directly related ablation evidence: STID https://arxiv.org/abs/2208.05233 (per-node embedding is primary source of gain for CI models on homogeneous MTS).

**Uncertainties:**
- Whether entity embedding should be broadcast over `patch_num` or vary per-patch (e.g., `nn.Embedding(num_stations, d_model)` vs `nn.Embedding(num_stations, d_model * patch_num)`). Current broadcast is the standard. Per-patch variation would explode parameters without clear motivation.
- Whether migrating from `arange` to per-sample `entity_ids` will produce measurable change — on multi-channel split with stable channel order, **they are equivalent**. The change is about *safety under permutation*, not accuracy.
