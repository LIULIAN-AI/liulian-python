# ETSformer — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | ETSformer |
| Paper URL | https://arxiv.org/abs/2202.01381 |
| Year / venue | arXiv 2022 — Woo et al., *ETSformer: Exponential Smoothing Transformers for Time-series Forecasting* |
| Official repo | https://github.com/salesforce/ETSformer |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py |
| This-repo adapter | `liulian/models/torch/etsformer.py` |
| Runtime key | `etsformer` |
| Benchmark key | `ETSformer` |

## 2. Architecture primer

ETSformer decomposes input into three components — **level**, **growth**, **seasonality** — inspired by classical exponential smoothing (ETS). Each encoder layer extracts growth + season with attention-like blocks; the decoder re-assembles them and adds the last level value.

```
x_enc (B, L, C)
     │
     ├── (training) Transform.transform(x_enc)  (σ=0.2 augmentation)   # L51-52
     │
     ▼
 DataEmbedding (C → d_model)                                            # L25, L53
     │
     ▼
 Encoder (e_layers, each an EncoderLayer(d_model, n_heads, enc_in, seq_len, pred_len, top_k)):
    returns (level, growths, seasons)                                  # L28-37, L54
     │
     ▼
 Decoder (d_layers == e_layers, each DecoderLayer(d_model, n_heads, c_out, pred_len)):
    returns (growth, season)                                           # L39-46, L56
     │
     ▼
 preds = level[:, -1:] + growth + season                               # L57
     │
     ▼
 slice last pred_len                                                   # L63
```

Complexity mostly `O(L · d_model²)` with `top_k`-based frequency filtering in the seasonal head. **Strict constraint**: `e_layers == d_layers` (L22).

## 3. This-repo audit

- `Model` (`etsformer.py:8-64`) is a TSL port.
- `ETSformerAdapter` (`etsformer.py:66-68`) inherits **only `TorchModelAdapter`** — **NOT `EntityAwareMixin`**.
- **Audit findings:**
  - **Critical gap**: adapter does not inherit the mixin, so no entity plumbing is hooked up. Pipeline's `ChannelEntityWrapper` wrap would succeed mechanically but `_init_entity_support` / `_entity_model_config` are never called.
  - Adapter is a minimal `_build_model` style (no `__init__` override). Retrofit needed.
  - Forecast path only; imputation/anomaly/classification assertions present at L17 but not wired through forward.
  - Adapter not exported from `liulian/models/torch/__init__.py` — accessed via `importlib.import_module` path in `pipeline.build_model` (L442).

## 4. Upstream reference

Official ETSformer (salesforce) matches TSL port in structure; TSL simplifies some activation and growth-head details. Candidate hooks:

| Hook | Location (this repo) | Tensors in scope | ETSformer-specific note |
|---|---|---|---|
| H1 pre-embed | before L53 | `x_enc (B, L, C)` | Also fed to encoder as raw input (L54 passes `x_enc` alongside `res`). |
| H2 post-embed | after L53 | `(B, L, d_model)` | **Natural**: pre-encoder. |
| H3 bias on `level` | after L54 | `(B, L, c_out)` level component | Per-station shift on the level — **matches classical ETS intuition**: the level is the per-station mean. |
| H4 bias on preds | after L57 | `(B, pred_len, c_out)` | Output-only. |

## 5. Proposed ID injection design

**Prerequisite: retrofit `ETSformerAdapter` to inherit `(EntityAwareMixin, TorchModelAdapter)`** and provide `__init__` calling `_entity_model_config` + `_init_entity_support`.

**Primary: H2 `add_to_embed` — per-entity additive embedding on `res` (post-`DataEmbedding`).**

Rationale:

1. Standard TSL-transformer pattern; aligns ETSformer with the rest of the suite.
2. Entity bias enters both level-extraction and growth/season extraction paths inside each encoder layer (since both operate on `res`).
3. The `Transform.transform` augmentation (L52) is applied to `x_enc` *before* embedding — entity injection happens after augmentation, so the identity signal is clean.

**Secondary: H3 `level_bias` — `nn.Embedding(num_stations, c_out)` added to `level[:, -1:]` before composing `preds` at L57.**

- **ETSformer-specific architectural fit**: the `level` is conceptually the running mean/baseline of the series. A per-station level bias is the textbook way to inject identity in an exponential-smoothing context.
- Cheap: `num_stations × c_out` parameters. Skips encoder/decoder for identity.

**Tertiary: H4 `post_output_affine` — `(num_stations × pred_len × c_out)` bias on `preds`.**

- DLinear-style fallback.

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-embed | Gets absorbed by DataEmbedding + also consumed by encoder's `x_enc` branch (L54) → signal split/attenuated. |
| Per-encoder-layer injection | Encoder layers already have residual flow of `res`; extra injections redundant. |

## 6. Concrete code change sketch

File: `liulian/models/torch/etsformer.py`
Functions: `Model.__init__` (L12-47), `Model.forecast` (L49-58), `ETSformerAdapter` (L66-68)

```python
from liulian.models.torch.entity_mixin import EntityAwareMixin


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        ...
        self._use_entity_embed = (
            getattr(configs, 'identifier_mode', 'none') == 'embedding'
            and getattr(configs, 'id_integration', '') == 'add_to_embed'
        )
        self._use_level_bias = (
            getattr(configs, 'id_integration', '') == 'level_bias'
        )
        if self._use_entity_embed:
            num_stations = getattr(configs, 'num_stations',
                                    getattr(configs, 'enc_in', 1))
            self.entity_embed = nn.Embedding(num_stations, configs.d_model)
        if self._use_level_bias:
            num_stations = getattr(configs, 'num_stations',
                                    getattr(configs, 'enc_in', 1))
            self.entity_level = nn.Embedding(num_stations, configs.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc, x_mark_enc)
        if self._use_entity_embed and entity_ids is not None:
            res = res + self.entity_embed(entity_ids).unsqueeze(1)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        last_level = level[:, -1:]
        if self._use_level_bias and entity_ids is not None:
            last_level = last_level + self.entity_level(entity_ids).unsqueeze(1)
        return last_level + growth + season


class ETSformerAdapter(EntityAwareMixin, TorchModelAdapter):
    def __init__(self, config):
        model_cfg = self._entity_model_config(config)
        model = Model(self._dict_to_namespace(model_cfg))
        super().__init__(model, config)
        self._init_entity_support(config)
```

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Requires adapter retrofit** + model edits. Then yes; the level-bias path is an unusually clean architectural fit. |
| Parameter overhead | H2: `num_stations × d_model`. H3: `num_stations × c_out` — trivial. |
| Parity test | Zero-init both embeddings ⇒ bit-exact baseline. |
| Transform augmentation | Augmentation applied before embedding; entity bias is clean post-augmentation. |
| `e_layers == d_layers` constraint | Unaffected by entity injection. |
| ETS semantics | **H3 `level_bias` is the most classical-ETS-aligned injection in the suite** — level is literally the per-series baseline in exponential smoothing. Strong motivation. |

## 8. Citations & uncertainty

- Paper: https://arxiv.org/abs/2202.01381
- Official repo: https://github.com/salesforce/ETSformer
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py
- This repo: `liulian/models/torch/etsformer.py:8-64` (Model), `:66-68` (Adapter — **lacks `EntityAwareMixin`**); dynamic import via `pipeline.build_model:442`.

**Uncertainties:**
- Whether H3 `level_bias` on its own matches or beats H2 `add_to_embed` empirically. H3 is theoretically aligned with ETS semantics but bypasses the encoder — in practice, a model with enough data may prefer H2's richer injection.
- TSL port's simplification vs official Salesforce implementation — if there are architectural gaps (e.g., DampedTrend decomposition) that differ, the `level_bias` interpretation might not map cleanly; check `etsformer_blocks.py` details before finalizing H3.
