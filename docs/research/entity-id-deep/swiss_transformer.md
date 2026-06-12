# Swiss-Transformer Family — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | Swiss-Transformer family (TransformerEncoderAdapter, SwissTransformerEmbeddingAdapter) |
| Paper URL | N/A — custom liulian models, originally from swiss-river-network-benchmark |
| Year / venue | N/A |
| Official repo | N/A |
| Canonical TSL impl | N/A |
| This-repo adapter | `liulian/models/torch/swiss_transformer.py` |
| Runtime keys | `transformer_encoder`, `swiss_transformer_embedding` |
| Benchmark keys | `TransformerEncoder`, `SwissTransformerEmbedding` |

## 2. Architecture primer

Three model variants with shared `_TransformerBaseAdapter` logic:

### SwissTransformerEmbeddingModel (primary, with optional entity embedding)
```
e (B, L) int IDs → nn.Embedding(num_embeddings, embedding_size) → emb
x (B, L, input_size)
     │
     ├── cat[emb, x] (if embedding)                                   # L238-240
     │
     ▼
 input_proj: Linear(input_size + emb_dim → d_model)                   # L160-161, L241
     │
     ▼
 Positional encoding: sinusoidal / learnable / RoPE                    # L186-199, L256-257
     │
     ▼
 (optional) mask_embedding for missing values                          # L205-206, L260-261
     │
     ▼
 nn.TransformerEncoder (or RoFormerModel for RoPE)                     # L192-199, L271-285
    causal mask (use_current_x=True) or full mask (extrapolation)
     │
     ▼
 ReLU → Linear(d_model → c_out) → (B, L, c_out)                      # L202, L287
     │
     ├── (extrapolation) slice last future_steps                       # L220, L288-289
     │
     ▼
 output (B, L or future_steps, c_out)
```

### SwissTransformerModel (no embedding — thin wrapper)
```
Same as above with num_embeddings=0, e=None.                          # L293-306
```

### TransformerEntityFeatureModel (pre-computed entity features)
```
x (B, L, input_size) + entity_features (B, L, entity_dim)
     → cat → Linear(input_size + entity_dim → d_model)               # L342, L370-371
     → pos_encoding → causal mask → TransformerEncoder → Linear      # L372-378
```

## 3. This-repo audit

- `SwissTransformerEmbeddingModel` (L96-290): full-featured encoder-only transformer.
- `SwissTransformerModel` (L293-306): no-embedding wrapper.
- `TransformerEntityFeatureModel` (L309-378): pre-computed feature concat variant.
- `_TransformerBaseAdapter(TorchModelAdapter)` (L386-448): shared forward logic.
- `TransformerEncoderAdapter(_TransformerBaseAdapter)` (L451-556): general-purpose.
- `SwissTransformerEmbeddingAdapter` (L566-572): backward-compatible alias.
- **Key observation: does NOT inherit `EntityAwareMixin`**. Same custom entity system as Swiss-LSTM:
  - `_TRANSPARENT_MODES`, `_EMBEDDING_MODE`, `_FEATURE_CONCAT_MODE` (L77-88).
  - Dispatch in `_forward_torch_model` (L400-448).
  - ID extraction from `x_mark_enc[:, :, entity_id_col]` — per-timestep.
- **Audit findings:**
  - **Second-most complete entity support** after Swiss-LSTM. Covers embedding, feature_concat, transparent modes.
  - `SwissTransformerEmbeddingModel` concatenates entity embedding to input *before* `input_proj` (L238-241). This is different from TSL-family models that add entity embedding to the `d_model` space. Here, entity embedding enters at the raw-feature level and gets linearly projected alongside input features.
  - Supports causal masking, RoPE (via HuggingFace `RoFormerModel`), mask_embedding for missing values, and extrapolation via future-step embeddings. Most feature-rich Transformer in the codebase.
  - `input_proj` dimension is `input_size + emb_dim → d_model` — the entity embedding dimension is separate from `d_model` and doesn't need to match.
  - Extrapolation mode (`use_current_x=False`): future-step embeddings are learnable position-like vectors concatenated after the history. If entity embedding is active, entity embedding is also applied to future positions (L246-249).

## 4. Upstream reference

No upstream — custom implementation. Entity injection **already built-in**.

| Hook | Location | Tensors | Note |
|---|---|---|---|
| H1 concat-to-input (existing) | `SwissTransformerEmbeddingModel.forward` L238-241 | `(B, L, input_size+emb_dim)` | **Already implemented.** |
| H2 feature-concat (existing) | `TransformerEntityFeatureModel.forward` L370-371 | `(B, L, input_size+entity_dim)` | **Already implemented.** |
| H3 add-to-d_model | after input_proj, L241 | `(B, L, d_model)` | Not implemented — would add entity bias in d_model space (like TSL-family). |
| H4 post-encoder | after transformer, L287 | `(B, L, d_model)` | Not implemented. |

## 5. Proposed ID injection design

**Status: already fully supported.** Same completeness as Swiss-LSTM.

**Existing designs (no changes needed):**

1. **`embedding` mode**: `SwissTransformerEmbeddingModel` concatenates `nn.Embedding` output with `x` before `input_proj`. IDs per-timestep from `x_mark_enc`. *(H1)*
2. **`feature_concat` mode**: `TransformerEntityFeatureModel` concatenates pre-computed features. *(H2)*
3. **Transparent modes**: data layer pre-concatenates; adapter passes through.

**Potential improvements:**

1. **H3 `add_to_d_model`**: TSL-family style additive entity embedding in `d_model` space post-`input_proj`. Architecturally different from concat-to-input (H1): concat uses the entity signal as raw input; additive in `d_model` space treats it as a learned bias. Whether one is better is empirical.
2. **Unify with `EntityAwareMixin`**: same consideration as Swiss-LSTM. Low priority.
3. **RoPE + entity**: when using `RoFormerModel`, entity embedding enters via `inputs_embeds` — compatible but untested.

**Design comparison — concat-to-input vs add-to-d_model:**

| Aspect | H1 concat-to-input (current) | H3 add-to-d_model (TSL-style) |
|---|---|---|
| Entity embedding dim | Independent (`embedding_size`) | Must match `d_model` |
| Interaction with input_proj | Joint projection; learned weight matrix mixes entity with features | Separate; entity bypasses `input_proj` |
| Parameter count | `input_proj` wider: `(input_size + emb) × d_model` | Extra `nn.Embedding(N, d_model)` |
| Architectural precedent | STID-style; concat-to-input common in graph/spatial models | Transformer entity injection; standard in TSL-family |

## 6. Concrete code change sketch

No changes required for current functionality. For H3 (future enhancement):

```python
class SwissTransformerEmbeddingModel(nn.Module):
    def __init__(self, ..., id_integration='concat_to_input'):
        ...
        self._id_integration = id_integration
        if id_integration == 'add_to_d_model' and num_embeddings > 0:
            self.entity_embed_additive = nn.Embedding(num_embeddings, d_model)
            self.input_proj = nn.Linear(input_size, d_model)  # no emb_dim
        else:
            # existing concat-to-input path
            ...

    def forward(self, e, x, ...):
        if self._id_integration == 'add_to_d_model':
            x = self.input_proj(x)
            if self.entity_embed_additive is not None and e is not None:
                x = x + self.entity_embed_additive(e)
        else:
            # existing concat path
            if self.embedding is not None and e is not None:
                emb = self.embedding(e)
                x = torch.cat([emb, x], dim=-1)
            x = self.input_proj(x)
        ...
```

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Entity support status | **Already fully implemented** — embedding, feature_concat, all transparent modes. |
| Compatibility with EntityAwareMixin | Uses its own parallel system. Same as Swiss-LSTM. |
| RoPE compatibility | Entity embedding enters before RoFormerModel as part of `inputs_embeds` — compatible. |
| Extrapolation entity handling | Entity embedding applied to both history and future positions (L246-249) — correct design. |
| Missing-value masking | `mask_embedding` at L260-261 is applied post-entity-injection — no interaction. |
| Per-timestep IDs | Like Swiss-LSTM, uses `(B, L)` IDs from `x_mark_enc` — more general than TSL-family `(B,)`. |

## 8. Citations & uncertainty

- This repo: `liulian/models/torch/swiss_transformer.py:96-290` (SwissTransformerEmbeddingModel), `:293-306` (SwissTransformerModel), `:309-378` (TransformerEntityFeatureModel), `:386-448` (_TransformerBaseAdapter), `:451-556` (TransformerEncoderAdapter), `:566-572` (SwissTransformerEmbeddingAdapter).
- Originally adapted from swiss-river-network-benchmark `model.py`.
- Positional encoding: Vaswani et al. 2017 (sinusoidal), Su et al. 2021 (RoPE).

**Uncertainties:**
- Whether concat-to-input (H1) or add-to-d_model (H3) produces better entity injection. The concat approach learns a joint entity-feature projection, which may capture entity-feature interactions that the additive approach misses. On the other hand, the additive approach keeps entity and feature representations separable, which may regularize better.
- Whether the RoPE variant's entity injection behaves identically to the sinusoidal/learnable variants. The RoFormerModel handles positional information internally via rotary embeddings — entity signal enters via `inputs_embeds` and should be orthogonal, but untested.
