# Swiss-LSTM Family — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | Swiss-LSTM family (LSTMAdapter, ExtrapoLSTMAdapter, SwissLSTMEmbeddingAdapter) |
| Paper URL | N/A — custom liulian models, originally from swiss-river-network-benchmark |
| Year / venue | N/A |
| Official repo | N/A |
| Canonical TSL impl | N/A |
| This-repo adapter | `liulian/models/torch/swiss_lstm.py` |
| Runtime keys | `lstm` (via `LSTMAdapter`), `extrapo_lstm`, `swiss_lstm_embedding` |
| Benchmark keys | `LSTM`, `ExtrapoLSTM`, `SwissLSTMEmbedding` |

## 2. Architecture primer

Four model variants, sharing `_LSTMBaseAdapter` logic:

### SwissLstmModel (basic)
```
x_enc (B, L, input_size) → LSTM(n_layers) → ReLU → Linear(hidden → c_out) → (B, L, c_out)
```

### ExtrapoLstmModelLIMO (extrapolation, last-input-multiple-output)
```
x_enc (B, L, input_size)
     │
     ├── slice: x[:, :-future_steps]                                  # L128
     │
     ▼
 LSTM(n_layers) → last hidden → ReLU → Linear(hidden → future_steps*c_out)  # L129-131
     │
     ▼
 reshape (B, future_steps, c_out)
```

### ExtrapoLstmModelFEmbed (extrapolation, future-step embedding)
```
x_hist → input_proj → cat[x_hist, future_step_embedding] → LSTM → Linear → (B, future_steps, c_out)
```

### LstmEmbeddingModel (entity embedding)
```
e (B, L) int IDs → nn.Embedding(num_embeddings, embedding_size) → emb (B, L, embedding_size)
x (B, L, input_size) → cat[emb, x] → LSTM(input_size + embedding_size) → Linear → (B, L, c_out)
```

### LstmEntityFeatureModel (pre-computed entity features)
```
x (B, L, input_size) + entity_features (B, L, entity_dim) → cat → LSTM(input_size + entity_dim) → Linear
```

## 3. This-repo audit

- `SwissLstmModel` (L49-83), `ExtrapoLstmModelLIMO` (L86-131), `ExtrapoLstmModelFEmbed` (L134-186), `LstmEmbeddingModel` (L189-235), `LstmEntityFeatureModel` (L238-285).
- `_LSTMBaseAdapter(TorchModelAdapter)` (L317-396): shared forward logic; dispatches by `_entity_mode`.
- `LSTMAdapter(_LSTMBaseAdapter)` (L399-480): general-purpose; selects model variant based on `identifier_mode`.
- `ExtrapoLSTMAdapter(_LSTMBaseAdapter)` (L483-524): LIMO/FEmbed.
- `SwissLSTMEmbeddingAdapter` (L536-542): backward-compatible alias forcing `identifier_mode='embedding'`.
- **Key observation: does NOT inherit `EntityAwareMixin`**. Instead, implements its own entity handling:
  - `_TRANSPARENT_MODES` (L294-303): `onehot`, `coordinates`, `sinusoidal`, `random`, `descriptors`, `numeric_id` — features already in `x_enc`.
  - `_EMBEDDING_MODE` (L306): integer IDs from `x_mark_enc[:, :, entity_id_col]`.
  - `_FEATURE_CONCAT_MODE` (L309): separate `entity_features` tensor.
- **Audit findings:**
  - **Most complete entity support in the codebase** — covers embedding, feature_concat, and all transparent modes. Predates and parallels `EntityAwareMixin`.
  - Does NOT use `EntityWrapper`/`ChannelEntityWrapper` — bypasses the wrapper system entirely. Entity handling is direct in `_forward_torch_model` (L342-396).
  - ID extraction from `x_mark_enc[:, :, entity_id_col]` at L371 — per-timestep IDs (shape `(B, L)`), not per-sample. This supports scenarios where entity ID changes within a window (e.g., multi-entity sliding windows).
  - No `_entity_model_config` call → transparent-mode `enc_in` widening must be handled externally (by the data layer or pipeline config).

## 4. Upstream reference

No upstream — custom liulian implementation. Entity injection is **already built-in** via `LstmEmbeddingModel`.

| Hook | Location | Tensors | Note |
|---|---|---|---|
| H1 embedding-concat (existing) | `LstmEmbeddingModel.forward` L231-234 | `(B, L, embedding_size)` cat to `x` | **Already implemented.** |
| H2 feature-concat (existing) | `LstmEntityFeatureModel.forward` L282-284 | `(B, L, entity_dim)` cat to `x` | **Already implemented.** |
| H3 hidden-init | LSTM `(h_0, c_0)` | `(n_layers, B, hidden)` | Not implemented — would need new model variant. |
| H4 post-output | after `self.linear` | `(B, L, c_out)` | Not implemented. |

## 5. Proposed ID injection design

**Status: already fully supported.** The Swiss-LSTM family has the most complete entity injection of any model family in the codebase.

**Existing designs (no changes needed):**

1. **`embedding` mode**: `LstmEmbeddingModel` concatenates `nn.Embedding(num_embeddings, embedding_size)` output with `x_enc` before LSTM input. IDs extracted per-timestep from `x_mark_enc`. *(H1)*
2. **`feature_concat` mode**: `LstmEntityFeatureModel` concatenates pre-computed entity features with `x_enc`. *(H2)*
3. **Transparent modes**: data layer pre-concatenates entity features into `x_enc`; adapter passes through unchanged.

**Potential improvements:**

1. **H3 `entity_hidden_init`**: add per-entity learned `(h_0, c_0)` initial states. Same design as proposed for vanilla LSTM (`lstm.md` H2). Would create a new model variant `LstmEntityHiddenModel`.
2. **Unify with `EntityAwareMixin`**: refactor `_LSTMBaseAdapter` to optionally inherit `EntityAwareMixin` for consistency with TSL-family adapters. Low priority — the custom system works well and is more feature-complete than the mixin.
3. **ExtrapoLSTM entity support**: `ExtrapoLSTMAdapter` defaults `identifier_mode='none'` (L523). The LIMO/FEmbed models don't have entity variants. Adding entity support to extrapolation models would require new model classes.

## 6. Concrete code change sketch

No changes required for current functionality. For H3 (future enhancement):

```python
class LstmEntityHiddenModel(nn.Module):
    def __init__(self, input_size, num_embeddings, hidden_size, num_layers, c_out=1, dropout=0.0):
        super().__init__()
        self.entity_h0 = nn.Embedding(num_embeddings, num_layers * hidden_size)
        self.entity_c0 = nn.Embedding(num_embeddings, num_layers * hidden_size)
        self._n_layers = num_layers
        self._hidden = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, c_out))

    def forward(self, e, x):
        # e: (B, L) — use first timestep's entity ID for initial state
        entity_id = e[:, 0]
        h0 = self.entity_h0(entity_id).view(-1, self._n_layers, self._hidden).transpose(0, 1).contiguous()
        c0 = self.entity_c0(entity_id).view(-1, self._n_layers, self._hidden).transpose(0, 1).contiguous()
        out, _ = self.lstm(x, (h0, c0))
        return self.linear(out)
```

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Entity support status | **Already fully implemented** — embedding, feature_concat, all transparent modes. |
| Compatibility with EntityAwareMixin | Uses its own parallel system. Not a bug — the Swiss-LSTM system predates the mixin and is more feature-complete. |
| ExtrapoLSTM gap | Extrapolation models lack entity variants. Medium-priority gap if extrapolation + entity is needed. |
| Per-timestep vs per-sample IDs | Swiss-LSTM uses per-timestep IDs `(B, L)` from `x_mark_enc` — more general than the per-sample `(B,)` IDs used by TSL-family models. |
| Data-layer dependency | `embedding` mode requires `x_mark_enc` to contain integer entity IDs at `entity_id_col`. Data layer must populate this correctly — see `timeseriesdataset.make_entity_features`. |

## 8. Citations & uncertainty

- This repo: `liulian/models/torch/swiss_lstm.py:49-83` (SwissLstmModel), `:86-131` (ExtrapoLstmModelLIMO), `:134-186` (ExtrapoLstmModelFEmbed), `:189-235` (LstmEmbeddingModel), `:238-285` (LstmEntityFeatureModel), `:317-396` (_LSTMBaseAdapter), `:399-480` (LSTMAdapter), `:483-524` (ExtrapoLSTMAdapter), `:536-542` (SwissLSTMEmbeddingAdapter).
- Originally adapted from swiss-river-network-benchmark `model.py`.

**Uncertainties:**
- Whether the concat-to-input approach (current) is strictly better or worse than hidden-state initialization (H3). For short sequences, H3 may help more; for long sequences, the concat approach provides continuous entity conditioning at every timestep.
- Whether unifying with `EntityAwareMixin` is worth the refactoring cost. The current custom system is battle-tested; the mixin is designed for TSL-family adapters. Forcing unification may introduce regressions.
