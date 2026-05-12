# ADR 0002 — Custom agent orchestrator, not LangGraph

- **Status**: accepted
- **Decision date**: 2026-05-12
- **Decision owner**: Linlin Jia
- **Supersedes**: iteration 1's "use LangGraph for agent orchestration"

## Context

The LIULIAN agent layer (`liulian-agent` service) needs:
- Multi-provider LLM routing (DeepSeek / GLM / Gemini / Claude / Ollama).
- Typed tool surface (Pydantic) for `data` / `model` / `bi` personas.
- Region-aware fallback (e.g. Gemini blocked in CN).
- Per-turn cost tracking + audit logging.
- SSE streaming to the web frontend.

Two paths considered:

1. **LangGraph** (LangChain ecosystem): mature graph orchestrator;
   broad community; many transitive deps.
2. **Custom**: hand-write ~300 LOC orchestrator using `pydantic`,
   `httpx`, `asyncio` directly. Model after `neobanker-agent`.

## Decision

Build custom, modeled on `neobanker-agent`. Reuse 70%+ of neobanker's
`agent/` package (loop, planner, intent, state, provider_registry,
provider_policy, conversation_cache, reliability) and `llm/` package
(gateway, harness, config, providers). Rename `agent` → `liulian_agent`,
swap bank tools for forecasting tools, repoint prompts.

## Rationale

- **Dependency mass**: LangGraph pulls LangChain core + 30+
  transitive packages with sharp release cadences.
- **Abstraction tax**: for our 5-tool surface, LangGraph's primitives
  are overkill; debugging a misbehaving graph requires learning their
  abstractions.
- **Pinning risk**: both LangGraph and LangChain refactor fast; our
  long-tail maintenance cost grows.
- **Production validation**: `neobanker-agent` is already running in
  production with the exact patterns we need (multi-provider routing,
  region fallback, SSE streaming). Reusing it is verified-good code.
- **Operator continuity**: same SSE event names, same provider config,
  same `/health` endpoint as neobanker. The operator runs both with
  the same mental model.

## Alternatives considered

- **LangGraph**: as above; rejected for dep mass + abstraction tax.
- **CrewAI / AutoGen multi-agent frameworks**: overkill for our
  3-persona need; the personas are tool-scoped, not autonomous-team.
- **DSPy**: focuses on optimization-time prompting, not runtime
  orchestration; would solve a different problem.

## Consequences

- (+) ~300 LOC orchestrator owned end-to-end; debuggable on one screen.
- (+) Verified-good code path inherited from neobanker.
- (+) Verbatim SSE event shape compatible with neobanker frontend
  hooks (which we fork for `liulian-web`).
- (−) No community plugins ecosystem; we add our own.

## LLM provider matrix

| Provider | Module | Default for | Status |
|---|---|---|---|
| DeepSeek V4 Flash | new `DeepSeekProvider` (OpenAI-compat; reuse `GLMProvider` base) | default for all calls | sprint Day 5 |
| GLM 4.6 | reuse from neobanker-agent | Chinese tasks | sprint Day 5 |
| Gemini 3.1 Pro / Flash-Lite | reuse from neobanker-agent | long context (>200k), multimodal | sprint Day 5 |
| Claude (Sonnet / Opus) | reuse from neobanker-agent | high-quality reasoning | M2 |
| Ollama (qwen2.5-7b local) | reuse from neobanker-agent | offline / sovereign | sprint Day 5 |
| Mock | reuse from neobanker-agent | tests | sprint Day 5 |

## Cross-references

- `PLATFORM_BLUEPRINT.md` §7 (agent layer)
- `PLATFORM_DESIGN.md` §7 (agent conversation flows)
- `NEOBANKER_REUSE_MAP.md` §2 (file-by-file reuse plan)
