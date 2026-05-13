# LightTS — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | LightTS |
| Paper URL | https://arxiv.org/abs/2207.01186 |
| Year / venue | arXiv 2022 — Zhang et al., *Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures* |
| Official repo | N/A (TSL-only in practice) |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py |
| This-repo adapter | `liulian/models/torch/lightts.py` |
| Runtime key | `lightts` |
| Benchmark key | `LightTS` |

## 2. Architecture primer

LightTS is a **pure-MLP** model. Core ideas:

1. **Continuous sampling**: reshape `(B, L, N) → (B, num_chunks, chunk_size, N)` → chunks preserve temporal adjacency.
2. **Interval sampling**: reshape `(B, L, N) → (B, chunk_size, num_chunks, N)` → chunks interleave time steps.
3. Each sampling stream feeds through an **IEBlock** (interval/entity block): spatial MLP + channel-projection MLP (identity-initialized via `torch.nn.init.eye_` on L37).
4. Auto-regressive skip: `Linear(seq_len → pred_len)` highway.

```
x_enc (B, L, N)
     │
     ├── pad to multiple of chunk_size                                # L118-121
     │
     ├── AR highway: Linear(seq_len → pred_len) per channel           # L123-124
     │
     ├── Continuous sampling  (B*N, chunk_size, num_chunks)
     │      → IEBlock(layer_1)                                        # L127-131
     │      → Linear(num_chunks → 1) → (B*N, d_model/4)
     │
     ├── Interval sampling     (B*N, chunk_size, num_chunks)
     │      → IEBlock(layer_2)                                        # L134-138
     │      → Linear(num_chunks → 1) → (B*N, d_model/4)
     │
     ▼
 concat → reshape (B, d_model/2, N)                                   # L140-142
     │
     ▼
 IEBlock(layer_3, out_dim=pred_len, num_node=enc_in)                  # L106-111, L144
     │
     ▼
 out = out + highway → (B, pred_len, N)                               # L145
```

**Critical property**: IEBlock's `channel_proj = Linear(num_node, num_node)` initialized to identity (L36-37). This is the *only* direct cross-channel interaction in LightTS — and it is already "per-node" in structure. Entity injection has a natural home here.

## 3. This-repo audit

- `Model` (`lightts.py:49-176`) is a TSL port.
- **No Adapter class at all** — there is only `IEBlock` and `Model`. The model is constructed directly by `pipeline.build_model` via `importlib.import_module('liulian.models.torch.lightts')` at `pipeline.py:442`.
- **Audit findings:**
  - **Most severe gap in B3**: no adapter → no `EntityAwareMixin` → no `_entity_model_config` widening → transparent modes cannot work, embedding modes cannot work. `lightts` entity support is **fundamentally not wired**.
  - `IEBlock` is described in its docstring as "**I**nterval/**E**ntity Block" — the name suggests original authors intended an entity role. Currently `channel_proj` is just a per-node linear with identity init, not an actual entity embedding.
  - Chunk-size padding (L118-121) re-writes `seq_len` in-place → interacts with any entity bias that needs to be broadcast over `L`.
  - Search spaces present in `liulian/optim/search_spaces.py:828-833` → model is actually used in runs despite missing adapter.

## 4. Upstream reference

No well-known official repo; TSL port is canonical in this codebase. Candidate hooks:

| Hook | Location (this repo) | Tensors in scope | LightTS-specific note |
|---|---|---|---|
| H1 pre-pad | before L118 | `x_enc (B, L, N)` | Bias added before AR + MLP chain. |
| H2 per-channel bias in IEBlock.channel_proj | inside IEBlock | `(num_node, num_node)` | **IEBlock's "E"-for-Entity hint**: replace identity-init `channel_proj` with per-entity shift. |
| H3 post-layer_3 | after L144 | `(B, pred_len, N)` | Post-model output bias. |
| H4 AR highway per-entity | modifies L123-124 | `(B, seq_len, N) → (B, pred_len, N)` | Per-station AR offset. |

## 5. Proposed ID injection design

**Prerequisite: create `LightTSAdapter(EntityAwareMixin, TorchModelAdapter)`.** The pipeline's dynamic-import path needs either a proper adapter class, or the pipeline must be taught to wrap `Model` directly. Creating an adapter is cleaner and aligns with every other model.

**Primary: H3 `post_output_affine` — per-station `(num_stations × pred_len)` additive bias, broadcast across channels.**

Rationale:

1. LightTS is extremely lightweight — its entire pipeline is MLPs with identity-initialized cross-channel proj. Adding a per-station pre-attention embedding is over-engineering for a model this simple.
2. Output-affine is the DLinear-style minimal intervention: cheap, architecturally compatible, and matches the model's "Less Is More" philosophy.
3. For multi-channel split (where each of `N` channels is a station), the output-affine table can be `(num_stations, pred_len)` with per-channel lookup.

**Secondary: H2 `ieblock_entity_proj` — replace `IEBlock.channel_proj`'s identity-init `Linear(num_node, num_node)` with a per-entity `nn.Embedding(num_stations, num_node)` additive bias on its output, so station identity modulates the cross-node mixing.**

- Architecturally suggestive — the IEBlock docstring names "Entity" — but intrusive; touches the most-shared building block.
- Would over-parameterize if applied at all three IEBlocks; restrict to `layer_3` (final per-channel aggregation) where `num_node == enc_in == num_stations`.

**Tertiary: H1 `add_to_input` — additive `nn.Embedding(num_stations, ?)` broadcast over `(L, N)` before padding.**

- Interacts with chunk-padding (zeros appended to the `L` axis); awkward.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H4 AR-highway-only | Only modifies the linear skip; bypasses both sampling streams — too partial. |
| Per-IEBlock entity injection | Three IEBlocks with different `num_node` values would need three separate embedding tables — parameter inflation without clear benefit. |

## 6. Concrete code change sketch

File: `liulian/models/torch/lightts.py`
Functions: `Model.__init__` (L55-87), `Model.encoder` (L115-146), new `LightTSAdapter`.

```python
from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.entity_mixin import EntityAwareMixin


class Model(nn.Module):
    def __init__(self, configs, chunk_size=24):
        super().__init__()
        ...
        self._use_post_affine = (
            getattr(configs, 'identifier_mode', 'none') == 'embedding'
            and getattr(configs, 'id_integration', '') == 'post_output_affine'
        )
        if self._use_post_affine:
            num_stations = getattr(configs, 'num_stations',
                                    getattr(configs, 'enc_in', 1))
            self.entity_out_bias = nn.Embedding(num_stations, self.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        out = self.encoder(x_enc)
        if self._use_post_affine and entity_ids is not None:
            bias = self.entity_out_bias(entity_ids).unsqueeze(-1)   # (B, pred_len, 1)
            out = out + bias                                         # broadcast across N
        return out


class LightTSAdapter(EntityAwareMixin, TorchModelAdapter):
    def __init__(self, config):
        default_config = {'task_name': 'long_term_forecast', 'd_model': 128, 'dropout': 0.1}
        default_config.update(config)
        if 'c_out' not in default_config:
            default_config['c_out'] = default_config['enc_in']
        model_cfg = self._entity_model_config(default_config)
        model = Model(self._dict_to_namespace(model_cfg))
        super().__init__(model, default_config)
        self._init_entity_support(default_config)
```

Update `liulian/models/torch/__init__.py` to export `LightTSAdapter`; update `pipeline.build_model` to use it (or keep dynamic-import and just add the class — it'll be picked up either way).

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Requires adapter creation**. Then straightforward. |
| Parameter overhead | H3: `num_stations × pred_len` — for 862 × 96 = 83K. Tiny. |
| Parity test | Zero-init `entity_out_bias` ⇒ bit-exact. |
| Dynamic-import path | `importlib.import_module` already handles arbitrary class names — adding `LightTSAdapter` should slot in; check pipeline's lookup key convention (typically `{CamelCaseKey}Adapter`). |
| Chunk-padding interaction | H3 applied post-encoder on final shape `(B, pred_len, N)` — no interaction with chunk/pad logic. Clean. |
| H2 IEBlock-level interaction | If chosen, the identity-init `channel_proj` preserves baseline parity at zero-init for any entity delta. Higher engineering risk: multiple IEBlocks use the class. |

## 8. Citations & uncertainty

- Paper: https://arxiv.org/abs/2207.01186
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py
- This repo: `liulian/models/torch/lightts.py:49-176` (Model), `:19-46` (IEBlock — **"I/E" = Interval/Entity per its docstring**). No Adapter class yet. Dynamic import via `pipeline.build_model:442`.
- Related: DLinear (paper arxiv:2205.13504) — LightTS's closest MLP cousin. DLinear has native `entity_affine` support; LightTS should mirror it.

**Uncertainties:**
- Whether an output-affine bias alone closes most of the identity gap for LightTS. Given the model's MLP simplicity, there's little "capacity to be entity-aware" in the middle — arguably most of the gap *is* at the output.
- Whether the IEBlock's "E-for-Entity" original intent in the authors' framing implies a deeper design (explicit entity embedding inside `channel_proj`). Revisiting the paper would be useful; no TSL doc captures it.
