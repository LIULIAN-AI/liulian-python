# TimeMoE — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | TimeMoE |
| Paper URL | https://arxiv.org/abs/2409.16040 |
| Year / venue | arXiv 2024 — Shi et al., *TimeMoE: Billion-Scale Time Series Foundation Models with Mixture of Experts* |
| Official repo | https://github.com/Time-MoE/Time-MoE |
| Canonical TSL impl | https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMoE.py |
| This-repo adapter | `liulian/models/torch/timemoe.py` |
| Runtime key | `timemoe` |
| Benchmark key | `TimeMoE` |

## 2. Architecture primer

TimeMoE is a **pretrained foundation model** (Maple728/TimeMoE-50M) used for **zero-shot forecasting only**. No training, no fine-tuning. Channel-independent via reshape.

```
x_enc (B, L, C)
     │
     ├── RevIN normalize                                              # L38-41
     │
     ▼
 reshape (B*C, L) — each channel is a univariate sample              # L44
     │
     ▼
 model.generate(x, max_new_tokens=pred_len)                          # L45
    • Pretrained autoregressive MoE model
    • No trainable parameters at inference
     │
     ▼
 output (B*C, L+pred_len)
     │
     ▼
 reshape (B, pred_len, C) → de-normalize                             # L46-49
```

Complexity: MoE model inference. **Key constraint**: `task_name='zero_shot_forecast'` only (L53). The model is frozen — no gradient flows. This fundamentally limits entity injection to **inference-time** techniques.

## 3. This-repo audit

- `Model` (`timemoe.py:19-56`): thin wrapper around HuggingFace `AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M')`.
- `TimeMoEAdapter` (`timemoe.py:59-101`): **inherits `(EntityAwareMixin, TorchModelAdapter)`**.
- **Audit findings:**
  - Adapter has `EntityAwareMixin` — entity plumbing technically works.
  - **Fundamental limitation**: zero-shot only. No training → `nn.Embedding` cannot be learned. Entity embedding in the traditional sense is **not applicable**.
  - `model.generate()` at L45 is the only forward path — the API accepts raw token sequences, not `inputs_embeds`. Injecting learned embeddings into the generation pipeline requires accessing model internals.
  - `_entity_model_config` widening `enc_in` for transparent modes would only affect the reshape at L44 — the pretrained model processes univariate sequences regardless.

## 4. Upstream reference

Official TimeMoE is a foundation model with autoregressive generation. No training hooks exist. Candidate "hooks" are limited:

| Hook | Location (this repo) | Tensors in scope | TimeMoE-specific note |
|---|---|---|---|
| H1 pre-generate input bias | before L45 | `(B*C, L)` raw values | Can add a per-station offset to the raw time series before generation. |
| H2 post-generate output bias | after L46 | `(B*C, pred_len)` | Per-station output shift. |

## 5. Proposed ID injection design

**TimeMoE is fundamentally incompatible with learned entity embeddings.** Zero-shot = no training = no learnable parameters = no `nn.Embedding`.

**Only viable approach: H2 `post_generate_affine` — pre-computed (not learned) per-station bias/scale on the output.**

Rationale:

1. If per-station historical statistics are available (mean, std), a simple bias correction can be applied post-generation. This is analogous to "calibration" rather than "entity injection".
2. This does not require any trainable parameters — it's a fixed post-processing step.
3. **This is NOT entity injection in the research sense** — it's output calibration. Document this distinction clearly.

**Alternative: fine-tune TimeMoE with entity conditioning.**

- Would require changing `task_name` from `zero_shot_forecast` to a trainable mode.
- At that point, TimeMoE becomes a fine-tuned foundation model, not a zero-shot model.
- Entity injection (H2-style on `inputs_embeds` if the API supports it) would then follow GPT4TS/TimeLLM patterns.
- Out of scope for the current zero-shot-only design.

**Recommendation: exclude TimeMoE from the entity-identifier ablation study.** Document as "not applicable — zero-shot only" in the cross-cutting matrix.

## 6. Concrete code change sketch

No meaningful code change for zero-shot mode. If future fine-tuning support is added:

```python
class Model(nn.Module):
    def finetune_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        # Would require: model.forward(inputs_embeds=...) instead of model.generate()
        # + unfreezing some layers
        # + entity_embed = nn.Embedding(num_stations, model.config.hidden_size)
        raise NotImplementedError("TimeMoE fine-tuning with entity injection not yet supported")
```

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports entity injection? | **No** in zero-shot mode. Would require fine-tuning support (not currently implemented). |
| Zero-shot constraint | Fundamental — no training = no learnable entity embeddings. |
| Adapter has EntityAwareMixin | Yes, but functionally unused for zero-shot. Transparent modes (widening enc_in) may work for data-layer concat but the pretrained model processes univariate — extra channels just get separate generation calls. |
| Recommendation | **Exclude from entity-ID ablation.** Mark as "N/A — zero-shot only" in the cross-cutting matrix. |

## 8. Citations & uncertainty

- Paper: https://arxiv.org/abs/2409.16040
- Official repo: https://github.com/Time-MoE/Time-MoE
- TSL port: https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMoE.py
- This repo: `liulian/models/torch/timemoe.py:19-56` (Model), `:59-101` (Adapter — has `EntityAwareMixin` but functionally unused for zero-shot).

**Uncertainties:**
- Whether fine-tuning TimeMoE (unfreezing top layers + adding entity embedding) would yield competitive results vs training-from-scratch models. Foundation model fine-tuning for entity-specific TS forecasting is an open research area.
- Whether the `AutoModelForCausalLM` API supports `inputs_embeds` for generation — if so, entity embeddings could be prepended as "prompt tokens" at inference time (in-context conditioning). Would need investigation.
