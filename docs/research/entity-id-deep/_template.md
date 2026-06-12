# `<MODEL>` — Entity-Identifier Deep Dive

_Per-model schema template (2026-04-14). Every file under `entity-id-deep/` must contain these sections in this order._

## 1. Identity & provenance

| Field | Value |
|---|---|
| Canonical name | |
| Paper URL | |
| Year / venue | |
| Official repo (or canonical impl) | |
| This-repo adapter | `liulian/models/torch/<file>.py` |
| This-repo runtime key | |
| This-repo benchmark key | |

## 2. Architecture primer

_Purpose: 200-300 words on what the model does and how tensors flow. One ASCII block._

```
Shape flow:
  x_enc (B, L, C) → ...
```

## 3. This-repo audit

What the current `liulian/models/torch/<file>.py` does for entity IDs:
- Native native hooks present?  (cite lines)
- Wrapper route via `pipeline.build_model` (`EntityWrapper` / `ChannelEntityWrapper` / `add_after_patch`)?
- Any gotchas observed while reading the file.

## 4. Upstream reference

Where in the official implementation would an entity-ID embedding naturally hook in?
Cite `file:line` (or function) and what tensors are in scope there.

## 5. Proposed ID injection design

One primary injection point with rationale grounded in the architecture.
Alternatives considered + reason rejected.

## 6. Concrete code change sketch

In **this repo's** existing adapter:
- File: `liulian/models/torch/<file>.py`
- Function: …
- Approximate location: …
- Pseudocode / micro-diff (not final code).

## 7. Feasibility & risks

- Supports-after-revision verdict.
- Parity tests required.
- Known failure modes.

## 8. Citations & uncertainty

- Paper link
- Repo link
- This-repo line anchors
- External ablation evidence (STID, AGCRN, etc.) where relevant
- **Uncertainties:** what I'm not sure about and why.
