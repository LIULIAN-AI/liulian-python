# DLinear — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | DLinear (Decomposition-Linear) |
| Paper URL | https://arxiv.org/abs/2205.13504 |
| Year / venue | AAAI 2023 — Zeng et al., *Are Transformers Effective for Time Series Forecasting?* |
| Official repo | https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py |
| This-repo adapter | `liulian/models/torch/dlinear.py` |
| Runtime key | `dlinear` |
| Benchmark key | `DLinear` |

## 2. Architecture primer

DLinear is the simplest model in the suite and the most informative baseline: a linear layer on top of a trend/seasonal decomposition.

```
x_enc (B, L, C)
   │
   ▼
series_decomp(x, moving_avg=25)        # L16 of liulian decomposition
   ├── trend_init (B, L, C)
   └── seasonal_init (B, L, C)
         │
         │ permute to (B, C, L)
         ▼
  Linear_Trend:    nn.Linear(L, pred_len)     # shared across C by default
  Linear_Seasonal: nn.Linear(L, pred_len)
         │
         │ sum + permute back to (B, pred_len, C)
         ▼
  dec_out (B, pred_len, C)
```

Two modes exist (`individual=False` default, or `True`). Individual mode allocates a separate linear per channel — an important observation because per-channel parameters are the crudest possible form of entity identity: each variate already has a dedicated weight matrix. Current TSL-aligned default in this repo is `individual=False`, i.e. channel-shared.

## 3. This-repo audit

- `Model` subclass at `dlinear.py:21-145` is a verbatim TSL port. No entity awareness inside the `Model`.
- `DLinearAdapter` at `dlinear.py:147-208` inherits `(EntityAwareMixin, TorchModelAdapter)`. Entity support is delegated entirely to the mixin's `_init_entity_support` at `entity_mixin.py:316-346`, which wraps `self._model` into `EntityWrapper` (per-entity split) or `ChannelEntityWrapper` (multi-channel split) when `identifier_mode == 'embedding'`.
- **Audit finding (2.7):** `ChannelEntityWrapper` registers a **fixed** station index buffer `[0..N-1]` at `entity_mixin.py:200-202` and its `forward` accepts `entity_ids` but ignores them (L247-248 comment: *"accepted but ignored — station indices are pre-registered internally"*). This is fine **only** when channel order is stable across batches; any channel permutation silently breaks the identity-to-embedding binding.
- Entity plumbing for transparent modes (`onehot`, `numeric_id`, `sinusoidal`, `coordinates`, `descriptors`) depends on the data layer setting `station_name` — which CSV/PEMS loaders never do (audit 6.1, 6.2). For DLinear on `traffic`/`electricity`/etc., transparent modes are therefore **silent no-ops**.

## 4. Upstream reference

Official TSL `models/DLinear.py`:

```
encoder(x):
    seasonal_init, trend_init = decomposition(x)        # <-- PRE-DECOMP HOOK
    seasonal_output = Linear_Seasonal(seasonal_init)
    trend_output    = Linear_Trend(trend_init)           # <-- POST-LINEAR HOOK
    return (seasonal_output + trend_output).permute(0,2,1)
```

Candidate injection points in upstream (and mirrored in `dlinear.py:80-104`):

| Hook | Where | Tensors in scope |
|---|---|---|
| H1 pre-decomp | before `decomposition(x)` | `x ∈ (B, L, C)` — raw input |
| H2 post-decomp | after `permute` of seasonal/trend | `(B, C, L)` each |
| H3 post-linear (bias) | after `Linear_*` produces `(B, C, pred_len)` | final trend/seasonal outputs |
| H4 post-sum affine | final `(B, pred_len, C)` | final prediction |

## 5. Proposed ID injection design

**Primary: H4 post-sum affine — per-station learnable bias + scale.**

Rationale:

1. DLinear's core inductive bias is *shared linear projection across channels*. Adding per-entity **bias + scale** on the output preserves the channel-shared linear while letting each station shift/rescale — the cheapest, most architecturally-faithful form of identity.
2. This is functionally equivalent to the `individual=True` mode *reduced to an affine post-layer* — you keep the cheap shared weights but buy back entity-specific output statistics. Known from STID (Shao et al., 2022) to be the dominant channel of gain on homogeneous-entity MTS.
3. Does not alter DLinear's O(L) complexity; adds `2 * num_stations * pred_len` parameters, which is O(N × H) — tiny.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-decomp concat | Expands `C` by `emb_size`, breaking the "same-C-in-same-C-out" assumption of downstream Linear_*. Requires projection back, which is what `EntityWrapper` does today — it works, but gains are attenuated because decomposition averaging smooths the embedding signal. |
| H2 post-decomp addition | Adds entity signal to both trend *and* seasonal; no clear rationale for asymmetric coupling, and you pay parameters twice. |
| H3 post-linear bias per entity (trend only) | Cleaner than H4 at first glance, but trend-only injection leaves seasonal component unconditioned — asymmetric without justification. |

Keep current **`EntityWrapper` / `ChannelEntityWrapper` (H1-style)** as a **fallback baseline** — do not remove it. Expose H4 as a new `id_integration='post_output_affine'` mode.

## 6. Concrete code change sketch

File: `liulian/models/torch/dlinear.py`
Function: `Model.encoder` (L80-104) and `Model.__init__` (L26-78)

```python
class Model(nn.Module):
    def __init__(self, configs, individual=None):
        ...
        # New: entity post-output affine (only active when configured)
        self._use_entity_post_affine = (
            getattr(configs, 'identifier_mode', 'none') == 'embedding'
            and getattr(configs, 'id_integration', '') == 'post_output_affine'
        )
        if self._use_entity_post_affine:
            num_stations = getattr(configs, 'enc_in', 1)  # channel == station in MC
            self.entity_scale = nn.Parameter(torch.ones(num_stations, self.pred_len))
            self.entity_bias  = nn.Parameter(torch.zeros(num_stations, self.pred_len))

    def encoder(self, x):
        ...
        out = (seasonal_output + trend_output).permute(0, 2, 1)  # (B, pred_len, C)
        if self._use_entity_post_affine:
            out = out * self.entity_scale.T + self.entity_bias.T
        return out
```

`pipeline.build_model` wrapping branch (`pipeline.py:441-477`) must be extended: when `id_integration == 'post_output_affine'`, do **not** wrap with `ChannelEntityWrapper` (the model handles identity internally).

Parity test required: with `identifier_mode='none'`, `entity_scale=1` and `entity_bias=0` must be unused; compare output bit-exact to pre-change baseline on 1 synthetic batch.

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes**, straightforward. |
| Parameter overhead | `O(N × H)` — e.g. traffic N=862, H=96 → 165K extra params; negligible vs 165K model-wide. |
| Parity test | Zero-init entity_bias, ones-init entity_scale → identical output to baseline, modulo the redundant mul/add; drop behind a flag for `identifier_mode='none'`. |
| Channel-order risk | Inherits from `ChannelEntityWrapper` — same arange(N) assumption. Add a permutation test. |
| Per-entity split mode | In per-entity split, `C = features_per_station`, so `enc_in ≠ num_stations` — the design must select between channel-wise affine (MC) and sample-wise affine (PE). Recommend: in PE split, fall back to the existing `EntityWrapper` (already works). |

## 8. Citations & uncertainty

- Paper: https://arxiv.org/abs/2205.13504
- Official: https://github.com/cure-lab/LTSF-Linear
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py
- This repo: `liulian/models/torch/dlinear.py:21-145` (Model), `:147-208` (Adapter); wrapper applied via `liulian/models/torch/entity_mixin.py:316-346`.
- Ablation evidence for post-output per-entity affine on homogeneous MTS: STID https://arxiv.org/abs/2208.05233 (spatial-temporal identity + explicit node embeddings as the primary source of gain).

**Uncertainties:**
- Whether post-output affine beats pre-input wrapper on `traffic`/`electricity` is ultimately empirical. STID evidence is indirect (STID injects earlier and uses a non-linear tail).
- Per-entity split with DLinear is low-signal anyway (one model sees one station's few channels); entity ID may over-fit. Test with a small `embedding_size=4`.
