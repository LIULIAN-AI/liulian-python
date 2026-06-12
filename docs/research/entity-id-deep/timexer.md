# TimeXer — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | TimeXer |
| Paper URL | https://arxiv.org/abs/2402.19072 |
| Year / venue | NeurIPS 2024 — Wang et al., *TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables* |
| Official repo | https://github.com/thuml/TimeXer |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py |
| This-repo adapter | `liulian/models/torch/timexer.py` |
| Runtime key | `timexer` |
| Benchmark key | `TimeXer` |

## 2. Architecture primer

TimeXer's unique move is splitting the input into **endogenous** (target) and **exogenous** (covariate) streams, embedding them differently, then fusing via cross-attention:

- **Endogenous** path — patch + value embedding + positional embedding + a learnable **global token** appended after patches (`EnEmbedding`, L43-64). Shape `(B*n_vars, patch_num+1, d_model)`.
- **Exogenous** path — inverted embedding (variate-as-token, like iTransformer): `DataEmbedding_inverted(seq_len → d_model)` per variate. Shape `(B, n_vars_ex, d_model)`.
- **Encoder** — each layer runs self-attention on endogenous patches, then cross-attention from the **last (global) token only** against the exogenous token sequence. Output fuses back: `[patches, updated_global_token]`.

```
x_enc (B, L, C) where last channel is endogenous (features='MS')
     │
     ├── per-sample RevIN normalize
     │
     ▼
 Endogenous (last channel):
    EnEmbedding: patch (patch_len) + linear + positional + glb_token
        → (B*n_vars_en, patch_num+1, d_model)      # L172-177, L54-64

 Exogenous (other channels):
    DataEmbedding_inverted(L → d_model) per variate
        → (B, n_vars_ex, d_model)                   # L178-184

     │
     ▼
 TimeXerEncoder (e_layers):
    self_attn on endogenous patches                 # L121
    cross_attn: global_token queries exogenous tokens   # L126-136
    fuse: concat(patches, updated_global_token)     # L147
     │
     ▼
 FlattenHead: reshape → Linear(head_nf → pred_len)   # L218-224
     │
     ▼
 dec_out (B, pred_len, 1 or C) → de-norm
```

Complexity: `O((patch_num+1)^2 · d_model)` endogenous self-attn + `O(1 · n_vars_ex · d_model)` cross-attn (only the global token queries). Forecast-only (no imputation/classification heads present in TSL port).

## 3. This-repo audit

- `Model` (`timexer.py:155-304`) is a verbatim TSL port. **No native entity hook.**
- `TimeXerAdapter` (`timexer.py:307-360`) inherits `(EntityAwareMixin, TorchModelAdapter)`. Default `features='MS'` (L331). Forecast-only.
- `glb_token` at L50 is `(1, n_vars, 1, d_model)` — a **per-variate** learned token, already entity-like in a weak sense (per-variate slot). However, since `features='MS'`, `n_vars = 1` and `glb_token` reduces to a single shared vector.
- **Audit findings:**
  - In `features='MS'`, there is only one endogenous variate (the target). Entity identity then refers to *which station* provides that target — which is exactly what the Swiss-river per-entity split handles naturally.
  - In `features='M'` (multivariate, L297-299), endogenous has all channels (`forecast_multi` path, L260-290). Then `n_vars == C` and `glb_token` is per-variate.
  - Same transparent-mode no-op caveat as others.

## 4. Upstream reference

Official TimeXer matches TSL port. Candidate hooks (forecast path):

| Hook | Location | Tensors in scope | TimeXer-specific note |
|---|---|---|---|
| H1 pre-embed | before L237 | `x_enc (B, L, C)` | Gets split and separately embedded. |
| H2 post-`EnEmbedding` (on endogenous patches) | after L237 | `(B*n_vars, patch_num+1, d_model)` | **Natural**: endogenous stream. |
| H2′ **global-token bias** | on `glb_token` directly | `(1, n_vars, 1, d_model)` | Replace/augment with a per-entity token. |
| H3 post-`ex_embedding` | after L240 | `(B, n_vars_ex, d_model)` | Per-variate bias on exogenous (iTransformer-style). |
| H4 post-encoder | after L242 | `(B*n_vars, patch_num+1, d_model)` | Skips encoder. |
| H5 post-FlattenHead | after L248 | `(B, n_vars, pred_len)` | DLinear-style. |

## 5. Proposed ID injection design

**Primary: H2′ `entity_global_token` — replace the shared `glb_token` with a per-entity learned token `nn.Embedding(num_stations, d_model)`.**

Rationale:

1. TimeXer's `glb_token` is already architected as "the slot that summarises the endogenous stream and routes exogenous information back via cross-attention". Making this token **per-entity** is the most surgically targeted entity injection possible — it directly conditions how the cross-attention reads exogenous variables for each station.
2. For `features='MS'` (the common target-prediction setting), `n_vars_en == 1`; a per-entity global token replaces the single shared vector with `entity_ids`-selected one: `glb = entity_embed(entity_ids).reshape(B, 1, 1, d_model)`.
3. For `features='M'`, replace across all `n_vars`: `glb[:, v]` becomes `entity_embed(station_of_variate[v])`.
4. **Architecturally most novel** of all forecasting models in the suite — leverages TimeXer's explicit global-token design instead of tacking a generic embedding onto `enc_out`.

**Secondary: H2 `add_to_en_embed` — additive per-entity embedding on `en_embed` after `EnEmbedding` (before encoder).**

- Similar to Transformer/Informer's `add_to_embed`. Broadcasts entity across patches (including the global token).
- Simpler; less targeted than H2′.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-embed | Gets absorbed by endogenous + exogenous paths separately; signal fragmented. |
| H3 post-exogenous-embed | TimeXer's whole point is that exogenous informs the endogenous target. Entity identity is about *the target station*, not about exogenous variate characteristics. |
| H4/H5 output-side | Misses the encoder's cross-attention — wastes TimeXer's main gain. |

## 6. Concrete code change sketch

File: `liulian/models/torch/timexer.py`
Functions: `EnEmbedding.__init__` (L46-52), `EnEmbedding.forward` (L54-64), `Model.__init__` (L161-224), `Model.forecast` (L226-258), `Model.forecast_multi` (L260-290), `Model.forward` (L292-304)

```python
class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout, num_stations=None):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        if num_stations is not None:
            self.entity_glb = nn.Embedding(num_stations, d_model)
            self.glb_token = None  # sentinel: use entity_glb
        else:
            self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
            self.entity_glb = None
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, entity_ids=None):
        n_vars = x.shape[1]
        if self.entity_glb is not None and entity_ids is not None:
            # (B,) → (B, 1, 1, d_model) → expand to (B, n_vars, 1, d_model)
            glb = self.entity_glb(entity_ids).view(-1, 1, 1, self.entity_glb.embedding_dim)
            glb = glb.expand(-1, n_vars, -1, -1)
        else:
            glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars
```

`Model.__init__`: accept `num_stations` when `id_integration=='entity_global_token'`, pass to `EnEmbedding`. `forecast`/`forecast_multi`/`forward` accept and propagate `entity_ids`.

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes**, and architecturally the most elegant — directly edits the existing entity-like slot. |
| Parameter overhead | `num_stations × d_model` (replaces `1 × 1 × 1 × d_model` or `1 × n_vars × 1 × d_model`). Net change negligible. |
| Parity test | With `num_stations=1`, entity_glb output must match the original `glb_token` behaviour bit-for-bit (modulo init). For strict parity, keep the old path and expose the new via `id_integration`. |
| Cross-attention interaction | The per-entity global token becomes the **query** in cross-attention — exogenous variables are now read *conditional on the station*. This is the intended architectural gain. |
| features='M' branch | Complexity: each of `n_vars` variates may belong to a different station. For multi-channel split with stations-as-channels, the natural mapping is `entity_glb(arange(n_vars))` — same as PatchTST's `_inject_entity_after_patch`. |
| features='MS' branch | Cleanest case: single endogenous channel, one station per batch. Per-sample `entity_ids` directly selects the global token. |

## 8. Citations & uncertainty

- Paper: https://arxiv.org/abs/2402.19072
- Official repo: https://github.com/thuml/TimeXer
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py
- This repo: `liulian/models/torch/timexer.py:155-304` (Model), `:307-360` (Adapter); wrapper via `liulian/models/torch/entity_mixin.py:52-257`.
- Design heritage for per-entity query: BERT [CLS] token (per-sample), Perceiver latent queries, STID node embedding.

**Uncertainties:**
- Whether replacing (rather than adding to) `glb_token` causes training instability at init. Mitigation: init `entity_glb.weight` with the same `randn`-like distribution as the original `glb_token`.
- Whether the gain is concentrated in `features='MS'` (target station prediction with exogenous context) rather than `features='M'`. The paper emphasises MS — identity injection should help most there.
