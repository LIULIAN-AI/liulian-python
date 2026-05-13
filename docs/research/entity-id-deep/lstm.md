# LSTM (vanilla) — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | LSTM (vanilla baseline) |
| Paper URL | https://www.bioinf.jku.at/publications/older/2604.pdf (Hochreiter & Schmidhuber 1997) |
| Year / venue | Neural Computation 1997 |
| Official repo | N/A (built-in `torch.nn.LSTM`) |
| Canonical TSL impl | N/A — this is a liulian baseline, not TSL-ported |
| This-repo adapter | `liulian/models/torch/lstm.py` |
| Runtime key | `lstm` (vanilla; distinct from swiss_lstm family) |
| Benchmark key | `LSTM` |

## 2. Architecture primer

Minimal LSTM encoder → linear projection baseline. Ignores time marks, decoder inputs, and all structure beyond `x_enc`.

```
x_enc (B, L, enc_in)
     │
     ▼
 nn.LSTM(input_size=enc_in, hidden_size=d_model, n_layers)    # L54-60
     │
     ▼
 last_hidden (B, d_model)                                     # L86 (take last step)
     │
     ▼
 dropout → Linear(d_model → pred_len*c_out) → reshape         # L61, L89-91
     │
     ▼
 (B, pred_len, c_out)
```

Complexity: `O(L · d_model · (enc_in + d_model))` linear in L. **No normalization** (unlike most TSL models). **No time-feature embedding**.

## 3. This-repo audit

- `Model` (`lstm.py:21-93`): minimal baseline.
- `LSTMAdapter` (`lstm.py:96-122`): **does NOT inherit `EntityAwareMixin`** — only `TorchModelAdapter`.
- **Audit findings:**
  - **Critical gap**: without `EntityAwareMixin`, none of the entity plumbing works for this adapter. `pipeline.build_model` would not invoke `_entity_model_config` nor `_init_entity_support`, and `ChannelEntityWrapper` / `EntityWrapper` would not wrap this model when requested.
  - `c_out` defaults to `enc_in` (L117-118) — same TSL convention.
  - Note: there is a separate Swiss-LSTM family (`liulian/models/torch/swiss_lstm.py`) covered in B4.

## 4. Upstream reference

No TSL reference. Candidate hooks (would require adapter retrofit first):

| Hook | Location (this repo) | Tensors in scope | LSTM-specific note |
|---|---|---|---|
| H1 pre-LSTM concat | before L83 | `x_enc (B, L, enc_in)` | Concatenate one-hot/id channel; breaks shape contract — requires adapter-level enc_in widening. |
| H2 initial hidden state bias | `hidden_init` | `(n_layers, B, d_model)` | Per-entity learned initial hidden state — classical RNN identity injection. |
| H3 additive to LSTM output | on `lstm_out` | `(B, L, d_model)` | Post-LSTM — simple additive. |
| H4 pre-projection additive | on `last_hidden` | `(B, d_model)` | Per-sample offset before linear projection. |
| H5 post-projection | on `out` | `(B, pred_len, c_out)` | DLinear-style. |

## 5. Proposed ID injection design

**Prerequisite: retrofit `LSTMAdapter` to inherit `(EntityAwareMixin, TorchModelAdapter)`** — mirror the structure of `MambaAdapter` (`mamba_model.py:80-118`).

**Primary: H2 `entity_hidden_init` — per-entity learned initial hidden/cell states `(h_0, c_0)` passed into `self.lstm(x_enc, (h_0, c_0))`.**

Rationale:

1. LSTM's state space is fundamentally about *what the network remembers*. A per-entity initial state means the network starts out "remembering" the identity — classical RNN pattern (e.g., NMT decoder conditioned on encoder state).
2. Two embeddings: `nn.Embedding(num_stations, n_layers*d_model)` each for `h_0` and `c_0` — reshape to `(n_layers, B, d_model)`. Roughly `2 × num_stations × n_layers × d_model` parameters.
3. Architecturally the most LSTM-native injection point. H3/H4 (additive to output) treat the model as a black-box feature extractor and ignore the RNN's recurrent structure.

**Secondary: H4 `add_to_last_hidden` — per-entity additive `nn.Embedding(num_stations, d_model)` on `last_hidden` before projection.**

- Simpler; zero interaction with the RNN recurrence. Equivalent to learning a station-specific shift on the final feature vector.

**Tertiary: H5 `post_output_affine` — `(num_stations × pred_len × c_out)` bias on final output.**

- DLinear-style; tiniest parameter overhead; identity skips the entire LSTM.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-concat | Requires inflating `enc_in` — breaks adapter config; equivalent to transparent `onehot` mode, which the framework already supports (after fixing the `station_name` plumbing). |
| H3 per-step additive | Time-constant bias added at each LSTM step is absorbed by the LSTM's input gate — indistinguishable from H4 in expected behaviour. |

## 6. Concrete code change sketch

File: `liulian/models/torch/lstm.py`
Functions: `Model.__init__` (L41-62), `Model.forward` (L64-93), `LSTMAdapter` (L96-122)

```python
from liulian.models.torch.entity_mixin import EntityAwareMixin   # new import


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        ...
        self._use_entity_hidden = (
            getattr(configs, 'identifier_mode', 'none') == 'embedding'
            and getattr(configs, 'id_integration', '') == 'entity_hidden_init'
        )
        if self._use_entity_hidden:
            num_stations = getattr(configs, 'num_stations',
                                    getattr(configs, 'enc_in', 1))
            self.entity_h0 = nn.Embedding(num_stations, n_layers * d_model)
            self.entity_c0 = nn.Embedding(num_stations, n_layers * d_model)
            self._n_layers = n_layers
            self._d_model = d_model

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, entity_ids=None):
        if self._use_entity_hidden and entity_ids is not None:
            B = x_enc.shape[0]
            h0 = self.entity_h0(entity_ids).view(B, self._n_layers, self._d_model).transpose(0, 1).contiguous()
            c0 = self.entity_c0(entity_ids).view(B, self._n_layers, self._d_model).transpose(0, 1).contiguous()
            lstm_out, _ = self.lstm(x_enc, (h0, c0))
        else:
            lstm_out, _ = self.lstm(x_enc)
        ...


class LSTMAdapter(EntityAwareMixin, TorchModelAdapter):    # changed inheritance
    def __init__(self, config):
        default_config = {...}
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
| Supports after revision? | **Requires adapter retrofit** (mixin) + model edits. Then yes. |
| Parameter overhead | `2 × num_stations × n_layers × d_model` — for 862 stations, 2 layers, d_model=64: ~221K. Tiny. |
| Parity test | Zero-init embeddings ⇒ default zero initial state ⇒ bit-exact baseline. |
| Data flow | Entity `(h_0, c_0)` is per-sample; per-channel-identity is not modelled because LSTM here is not channel-independent (all `enc_in` channels fed jointly). Natural for per-entity split. |
| Transparent-mode compatibility | Once mixin is added, `onehot`/`numeric_id` work via `enc_in` widening — standard plumbing. |
| Swiss-LSTM overlap | The Swiss-LSTM family already has richer entity support via its own embedding adapter — this vanilla LSTM is a baseline for comparison. |

## 8. Citations & uncertainty

- Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
- This repo: `liulian/models/torch/lstm.py:21-93` (Model), `:96-122` (Adapter — **lacks `EntityAwareMixin`**); Swiss-LSTM family at `liulian/models/torch/swiss_lstm.py` (separate, covered in B4).
- Related: AGCRN (arxiv:2007.02842) — per-entity RNN parameter adaptation; this is a fuller-blown version of entity-conditional RNN than the simple hidden-init hook.

**Uncertainties:**
- Whether a learned `(h_0, c_0)` actually helps beyond the RNN's own capacity — for long sequences, the initial state gets overwritten quickly. Short-horizon tasks benefit more.
- In multi-channel split, LSTM processes all `enc_in` channels jointly → per-sample entity id is ill-defined (*whose* station is this sample about?). Only unambiguous in per-entity split. Document this constraint.
