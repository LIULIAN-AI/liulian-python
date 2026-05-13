# iTransformer — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | iTransformer |
| Paper URL | https://arxiv.org/abs/2310.06625 |
| Year / venue | ICLR 2024 — Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting* |
| Official repo | https://github.com/thuml/iTransformer |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py |
| This-repo adapter | `liulian/models/torch/itransformer.py` |
| Runtime key | `itransformer` |
| Benchmark key | `iTransformer` |

## 2. Architecture primer

iTransformer's single idea: **invert the roles of time and channel axes**. Each *variate* (channel) becomes a token; the full sequence of length `L` is first projected to a single `d_model` vector per variate via `DataEmbedding_inverted(L → d_model)`. Self-attention therefore operates **across variates**, capturing inter-variate dependencies directly. A final `Linear(d_model → pred_len)` projects each variate's representation back to the forecast horizon.

```
x_enc (B, L, N)      x_mark_enc (B, L, T)
      │
      ├── normalize per-sample (mean/std on L)            # L84-87 (RevIN-style)
      │
      ▼
 DataEmbedding_inverted : concat time-feats → Linear(L → d_model) per variate
                                      → enc_out (B, N, d_model)   # L92
      │
      ▼
  Encoder: self-attention across N variates                # L43-64
      │
      ▼
  enc_out (B, N, d_model)
      │
      ▼
  projection: Linear(d_model → pred_len) per variate       # L70, L95
      │
      ▼
  permute → (B, pred_len, N)  →  de-normalize              # L95-98
```

Complexity: `O(N^2 · d_model)` — scales with variates squared but not time. Attention sees **each variate as a token**; tokens are therefore already "channel slots" — this has direct implications for entity injection.

## 3. This-repo audit

- `Model` (`itransformer.py:24-169`) is a verbatim TSL port. **No native entity hook.**
- `iTransformerAdapter` (`itransformer.py:172-231`) inherits `(EntityAwareMixin, TorchModelAdapter)`. Same plumbing as Transformer family.
- **Key observation:** in iTransformer, a variate == a token. The `(B, N, d_model)` encoder input is exactly the shape where "add per-entity bias" is trivial: each variate is one row. If each channel is one station (multi-channel split), `N == num_stations`.
- Audit findings (same caveats):
  - `ChannelEntityWrapper` fixed-buffer limitation applies — but here it is actually aligned with the architecture (a fixed `arange(N)` is the correct per-variate indexing).
  - Transparent modes (`onehot`, `numeric_id`, …) expand `C` via data-layer concat, which inflates `N`. The resulting extra "entity-ID variates" get attention mass from every real variate — coarse but functional. Gating still needs `station_name` to be set by the data layer (CSV/PEMS miss this).

## 4. Upstream reference

Official iTransformer repo matches TSL port. Candidate hooks (note the inverted axis semantics):

| Hook | Location (this repo) | Tensors in scope | Inverted-axis note |
|---|---|---|---|
| H1 pre-embed | before L92 | `x_enc (B, L, N)` | Can add per-variate bias directly on `N` axis. |
| H2 post-embed | after L92 | `enc_out (B, N, d_model)` | **Natural**: `N` axis is the token axis. |
| H3 inside encoder (per-layer) | Encoder layers | `(B, N, d_model)` | Over-parameterized. |
| H4 post-encoder | after L93 | `(B, N, d_model)` | Skips attention over entity info. |
| H5 post-projection, pre-permute | after L95 before L97-98 | `(B, N, pred_len)` | Entity-specific output shift. |

## 5. Proposed ID injection design

**Primary: H2 `add_to_embed` — per-entity additive embedding in the `N`-token space, directly after `DataEmbedding_inverted`.**

Rationale:

1. In iTransformer, `enc_out` is already indexed by variate (row = channel). Adding `nn.Embedding(num_stations, d_model)(entity_ids)` exactly matches the shape — one bias vector per variate per sample. This is the *cleanest* possible injection across all forecasting models in this suite.
2. Self-attention then operates on **entity-conditioned tokens**: the attention score between variate *i* and variate *j* becomes a function of both their historical value pattern *and* their identity. This directly supports the authors' core claim that iTransformer learns cross-variate dependencies — with entity injection, these dependencies become entity-aware.
3. Zero complexity impact; `num_stations × d_model` parameters (e.g., traffic: 862 × 512 = 441K, ~0.5% of a ~10M model).

**Secondary: H5 `per_variate_bias` — add `(num_stations, pred_len)` bias to the projection output.**

- Cheaper (`num_stations × pred_len`), matches DLinear H4 philosophy.
- Doesn't participate in cross-variate attention → weaker.

**Split-mode compatibility:**

- **Multi-channel split** (default for traffic/electricity): `N == num_stations`. Per-sample `entity_ids` is a 1-D tensor of length `N` = `arange(N)` in channel order → same as `ChannelEntityWrapper`. H2 works.
- **Per-entity split** (Swiss-river): `N == features_per_station`. One model per station; `entity_ids` is a scalar per batch replicated to all `N` variates. H2 in this case adds the same vector to all `N` tokens → entity becomes a *global* bias across variates. Still valid, slightly redundant with H1's identical-across-`L` behaviour in the other Transformer models.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-embed | `DataEmbedding_inverted` is `Linear(L+T → d_model)`; per-variate entity signal added to raw `x_enc` gets averaged across `L` by the linear. Less efficient than direct H2. |
| H3 per-layer | No motivation — attention already sees it in H2. |
| H4 post-encoder | Entity never reaches attention; wastes the main benefit. |

## 6. Concrete code change sketch

File: `liulian/models/torch/itransformer.py`
Functions: `Model.__init__` (L29-80), `Model.forecast` (L82-99), `Model.forward` (L153-169)

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
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)    # (B, N, d_model)
        if self._use_entity_embed:
            if entity_ids is None:
                ids = torch.arange(N, device=enc_out.device)
                emb = self.entity_embed(ids)               # (N, d_model)
                enc_out = enc_out + emb.unsqueeze(0)       # broadcast batch
            else:
                # entity_ids shape: (B,) or (B, N)
                if entity_ids.ndim == 1:
                    ids = torch.arange(N, device=enc_out.device)
                    emb = self.entity_embed(ids)
                    enc_out = enc_out + emb.unsqueeze(0)
                else:
                    enc_out = enc_out + self.entity_embed(entity_ids)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out
```

Same pipeline/wrapper bypass as other models when `id_integration='add_to_embed'`.

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes — architecturally the best fit in the suite.** The `(B, N, d_model)` shape is already the "one-token-per-variate" layout. |
| Parameter overhead | `num_stations × d_model` — tiny. |
| Parity test | Zero-init entity_embed ⇒ bit-exact baseline match. |
| Multi-channel vs per-entity split | Multi-channel: ideal. Per-entity: degenerate (entity is a global bias). Consider wrapper fallback for per-entity split. |
| Channel-order risk | In multi-channel split, per-sample `entity_ids` is `arange(N)` (equivalent to `ChannelEntityWrapper`). Safer: plumb real per-variate `entity_ids` through the data layer once CSV/PEMS start populating `station_ids`-aligned tensors. |
| Normalization interaction | Injection is **after** `DataEmbedding_inverted`, which already absorbs the RevIN-style normalization via its linear on L. Entity bias is independent of normalization — good. |

## 8. Citations & uncertainty

- Paper: https://arxiv.org/abs/2310.06625
- Official repo: https://github.com/thuml/iTransformer
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py
- This repo: `liulian/models/torch/itransformer.py:24-169` (Model), `:172-231` (Adapter); wrapper in `liulian/models/torch/entity_mixin.py:52-257`.
- Closely related: STID https://arxiv.org/abs/2208.05233 — its "spatial identity embedding" is a direct analogue, though STID uses MLP rather than attention.

**Uncertainties:**
- iTransformer's own paper does not discuss per-entity embedding; it just treats variates as tokens. Whether the additional entity embedding produces gain on top of the already channel-aware attention is an open empirical question — possibly the smallest absolute gain of all transformer variants (because iTransformer is already partially entity-aware by construction).
- In per-entity split mode, entity embedding is redundant — consider defaulting to `identifier_mode='none'` for this case in sweeps.
