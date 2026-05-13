# TimeLLM — Entity-Identifier Deep Dive

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | Time-LLM |
| Paper URL | https://arxiv.org/abs/2310.01728 |
| Year / venue | ICLR 2024 — Jin et al., *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models* |
| Official repo | https://github.com/KimMeen/Time-LLM |
| Canonical TSL impl | N/A (custom implementation) |
| This-repo adapter | `liulian/models/torch/timellm.py` |
| Runtime key | `timellm` |
| Benchmark key | `TimeLLM` |

## 2. Architecture primer

Time-LLM "reprograms" a frozen LLM (LLAMA / GPT-2 / BERT / TinyLLaMA / Qwen) by cross-attending time-series patch embeddings against LLM vocabulary embeddings, then passing the result through the frozen LLM. Channel-independent.

```
x_enc (B, L, N)
     │
     ├── Normalize(enc_in, affine=False)                              # L362, L377
     │
     ├── permute → (B, N, L) → reshape (B*N, L, 1)                  # L379-380
     │
     ├── compute stats (min, max, median, lags, trend) per channel    # L382-386
     │
     ├── generate text prompt per channel (B*N prompts)               # L388-406
     │   "Dataset description: ... min value ... max value ..."
     │
     ▼
 Tokenize prompts → LLM vocabulary embedding                         # L413-423
     │
     ├── source_embeddings = mapping_layer(word_embeddings)            # L426-428
     │                         (vocab_size → num_tokens=1000)
     │
     ▼
 TimeLLMPatchEmbedding(x_enc) → enc_out (B*N, num_patches, d_model)  # L329-335, L433-435
     │
     ▼
 ReprogrammingLayer:                                                   # L342-344, L436-438
    cross-attn: Q=enc_out, K=V=source_embeddings
    → (B*N, num_patches, d_llm)
     │
     ▼
 concat [prompt_embeddings, reprogrammed_patches]                      # L439
     │
     ▼
 Frozen LLM backbone (llm_layers) → last_hidden_state                 # L443
     │
     ▼
 slice d_ff channels → reshape → FlattenHead → (B, N, pred_len)       # L445-457
     │
     ▼
 permute → denormalize → (B, pred_len, N)                             # L457-459
```

Complexity: LLM's O((prompt_len + num_patches)² · d_llm) per channel. **Channel-independent** via `(B*N, ...)` reshape.

## 3. This-repo audit

- `Model` (`timellm.py:87-470`): custom port. Multiple LLM backends.
- `ReprogrammingLayer` (`timellm.py:40-84`): cross-attention reprogramming.
- `TimeLLMAdapter` (`timellm.py:473-541`): **inherits `(EntityAwareMixin, TorchModelAdapter)`**. Standard plumbing.
- **Audit findings:**
  - Adapter has `EntityAwareMixin` — entity plumbing works. `_init_entity_support` called at L520.
  - Channel-independent: `(B, L, N) → (B*N, L, 1)` at L380. Same per-channel entity injection opportunity as PatchTST/GPT4TS.
  - Text prompt (L388-406) constructs per-channel statistics. **Unique opportunity**: inject entity name/description directly into the text prompt — "Station: Thur at Andelfingen, elevation 420m" — leveraging the LLM's language understanding. This is the only model in the suite where *text-based* entity injection is architecturally natural.
  - `_entity_model_config` would widen `enc_in` for transparent modes, but since `enc_in` affects `Normalize`, `FlattenHead`, and `TimeLLMPatchEmbedding`, this works structurally.

## 4. Upstream reference

Official Time-LLM repo uses LLAMA primarily; this port adds GPT-2/BERT/TinyLLaMA/Qwen. Candidate hooks:

| Hook | Location (this repo) | Tensors in scope | TimeLLM-specific note |
|---|---|---|---|
| H1 pre-patch | before L433 | `(B*N, L, 1)` | One-dim per channel; little room for additive. |
| H2 post-patch | after L435 | `(B*N, num_patches, d_model)` | **Natural**: PatchTST-like. |
| H3 post-reprogramming | after L438 | `(B*N, num_patches, d_llm)` | Entity enters frozen LLM directly. |
| H4 text-prompt injection | in prompt generation L388-406 | string | **Unique**: entity name/description in natural-language prompt. |
| H5 post-LLM | after L443 | `(B*N, prompt_len+num_patches, d_llm)` | Post-backbone. |
| H6 post-FlattenHead | after L456 | `(B, N, pred_len)` | DLinear-style. |

## 5. Proposed ID injection design

**Primary: H4 `entity_in_prompt` — inject entity name/description into the text prompt per channel.**

Rationale:

1. **Architecturally unique to TimeLLM**: no other model in the suite can leverage text-based entity descriptions. The prompt template (L394-403) already describes per-channel statistics in natural language. Adding "Station: {station_name}, Region: {region}" is trivial and directly leverages the LLM's pretrained knowledge about geographic entities, infrastructure types, etc.
2. Zero parameter overhead — the entity signal enters through the LLM's existing vocabulary embedding, not through a new `nn.Embedding`.
3. Requires `station_name` or `station_description` to be available in config/batch metadata. This aligns with the existing `entity_descriptors` mode in the data layer.

**Secondary: H2 `add_to_patch_embed` — per-entity additive `nn.Embedding(num_stations, d_model)` on `(B*N, num_patches, d_model)` after `TimeLLMPatchEmbedding`, before `ReprogrammingLayer`.**

- Standard PatchTST/GPT4TS approach. Works without text metadata.
- Entity bias enters the reprogramming cross-attention as queries — the model "asks different questions of the LLM vocabulary" for different stations.

**Tertiary: H6 `post_output_affine` — output bias.**

Alternatives considered and rejected:

| Alternative | Why rejected |
|---|---|
| H1 pre-patch | Single-channel `(B*N, L, 1)` — too narrow for meaningful embedding. |
| H3 post-reprogramming | Reprogramming already mapped to d_llm; adding entity bias at this large dimension is wasteful. |
| H5 post-LLM | Entity never reaches LLM or reprogramming. |

## 6. Concrete code change sketch

File: `liulian/models/torch/timellm.py`
Functions: `Model.__init__` (L88-363), `Model.forecast` (L375-461)

### H4 text-prompt injection:
```python
def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None,
             station_names=None):
    ...
    prompt = []
    for b in range(x_enc.shape[0]):
        station_desc = ''
        if station_names is not None:
            idx = b // N if entity_ids is None else entity_ids[b].item()
            station_desc = f'Station: {station_names[idx]}. '
        prompt_ = (
            f'<|start_prompt|>Dataset description: {self.description} '
            f'{station_desc}'
            f'Task description: forecast the next {self.pred_len} steps ...'
        )
        prompt.append(prompt_)
    ...
```

### H2 add_to_patch_embed:
```python
class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        ...
        self._use_entity_embed = (
            getattr(configs, 'identifier_mode', 'none') == 'embedding'
            and getattr(configs, 'id_integration', '') == 'add_to_patch_embed'
        )
        if self._use_entity_embed:
            num_stations = getattr(configs, 'num_stations',
                                    getattr(configs, 'enc_in', 1))
            self.entity_embed = nn.Embedding(num_stations, configs.d_model)

    def forecast(self, ...):
        ...
        enc_out, n_vars = self.patch_embedding(x_enc)
        if self._use_entity_embed:
            ids = torch.arange(n_vars, device=enc_out.device).repeat(B)
            enc_out = enc_out + self.entity_embed(ids).unsqueeze(1)
        enc_out = self.reprogramming_layer(enc_out, ...)
        ...
```

## 7. Feasibility & risks

| Item | Verdict |
|---|---|
| Supports after revision? | **Yes — adapter already has EntityAwareMixin.** H2 is straightforward. H4 (text prompt) is unique but requires metadata plumbing. |
| Parameter overhead | H4: zero. H2: `num_stations × d_model` (d_model=32 default, so ~28K for 862 stations). |
| Parity test | H2: zero-init ⇒ bit-exact. H4: empty station_desc string ⇒ identical prompt. |
| LLM backbone variety | H2 works regardless of which LLM backbone is selected (LLAMA/GPT2/BERT/TinyLLaMA/Qwen). H4 text injection too. |
| Prompt-based entity injection | H4 is the most novel injection design in the entire suite. Risk: LLM may not meaningfully leverage station names without fine-tuning. Mitigation: combine H4 + H2 for both text and embedding signals. |
| Channel-independence | Same `arange(n_vars)` per-channel pattern as PatchTST/GPT4TS. |

## 8. Citations & uncertainty

- Paper: https://arxiv.org/abs/2310.01728
- Official repo: https://github.com/KimMeen/Time-LLM
- This repo: `liulian/models/torch/timellm.py:87-470` (Model), `:40-84` (ReprogrammingLayer), `:473-541` (Adapter — **has `EntityAwareMixin`**).
- Related: GPT4TS (`gpt4ts.md`) — simpler LLM-based model without reprogramming; shares channel-independent structure.

**Uncertainties:**
- Whether the frozen LLM can actually leverage station names in the prompt (H4). The LLM has been pretrained on text but not fine-tuned for geographic/domain entities — station names might be out-of-vocabulary or semantically meaningless. Descriptive text ("alpine river station, elevation 1200m") may work better than proper names.
- Whether `d_model=32` (TimeLLM's patch embedding dim, not `d_llm`) provides enough capacity for entity embedding. The reprogramming layer projects `d_model → d_llm`, so the entity signal gets projected up — should be fine if `d_model` is large enough to represent station differences.
