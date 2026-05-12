# ADR 0006 — Fork-and-adapt from neo-banker repos

- **Status**: accepted (iteration 3 decision)
- **Decision date**: 2026-05-12

## Context

LIULIAN's frontend, backend, agent, crawler, and deploy CLI are
needed *now* for the M1 sprint. The user's neo-banker organization
already has all of these in production-shaped form:

- `neobanker-agent` (Python · FastAPI · multi-provider LLM gateway)
- `neobanker-crawler` (Python · FastAPI · scheduled ingest)
- `neobanker-frontend-MVP-V3` (Next.js · Ant Design · ECharts · canvas
  orchestrator + report builder + chat sidebar)
- `neobanker-backend-MVP-V2` (Java · Spring Boot)
- `neoctl` (Python · click · paramiko · deploy CLI)
- `neobanker-dev-env` (Codespaces dev container)

The user explicitly said *"大部分的配置和agent相关的东西都可以复用优化自
neobanker的几个库, 包括前端后端agent和爬虫"*. The question: green-field
or fork?

## Decision

Each new LIULIAN repo (except `liulian-python` and `liulian-mobile`)
starts as a **fork** of the corresponding neobanker repo, with renames
+ domain swap on day one. Green-field where no source exists.

| LIULIAN repo | Source |
|---|---|
| `liulian-agent` | `neo-banker/neobanker-agent` |
| `liulian-ingest` | `neo-banker/neobanker-crawler` |
| `liulian-web` | `neo-banker/neobanker-frontend-MVP-V3` |
| `liulian-ops` | `neo-banker/neoctl` |
| `liulian-dev-env` | `neo-banker/neobanker-dev-env` |
| `liulian-api` | green-field (Java backend not portable; *patterns* borrowed) |
| `liulian-mobile` | green-field (Expo template) |
| `liulian-design-system` | seeded from `liulian-python/.worktrees/gui-demo/styles/main.css` |

Detailed file-by-file map: `NEOBANKER_REUSE_MAP.md`.

## Rationale

- **Production-shaped**: these repos are running in production for
  neobanker; we inherit verified-good code paths.
- **Time saved**: estimated 2 to 4 weeks across the platform versus
  green-field. (Range, not point estimate; see audit note below.)
- **Operator continuity**: same SSE event names, same `/health`
  endpoints, same provider config, same deploy verbs (`neoctl deploy`
  → `liulianctl deploy`). The operator runs both with one mental model.
- **Bilingual docs already exist**: every neobanker doc has an `.en.md`
  mirror; we adopt the same discipline (with `.zh.md` per
  `claude-config:bilingual-docs` convention; see §11.2 of reuse map).

## Audit caveats

Per `AUDIT_REPORT_2026-05-12.md §B.3`, reuse fractions are *ranges
TBD after Sprint Day 1 fork spike*, not measured. The platform's
success does NOT depend on hitting any particular fraction; only on
shipping M1 end-to-end.

## The sacred separation (NON-NEGOTIABLE)

- **Code, architecture, plumbing, vocabulary, deploy patterns**:
  reuse / adapt / fork freely.
- **Visual design, brand voice, UI composition, iconography, typography,
  micro-interactions**: original. Reference for direction; never copy.

Operationalised in `conventions/UI_AUDIT_CHECKLIST.md`: the
**neobanker-copy test** is one of the four PR-merge gates.

## Alternatives considered

- **Green-field everything**: rejected on time grounds; would push M1
  past the ARTORG-AIHN deadline.
- **Vendor library reuse only** (e.g. LangChain, Refine.dev): rejected;
  these introduce abstraction tax and visual lookalike risk
  (ADR 0002 + 0007).

## Consequences

- (+) Sprint Day 1 reframes from "scaffold" to "fork + measure".
- (+) Operator muscle memory preserved.
- (−) Domain swap is real work, not free; bank vocab must be exorcised
  from prompts, configs, demos, fixtures.
- (−) Visual originality discipline must be enforced on every PR or
  the brand collapses into a neobanker reskin.

## Cross-references

- `NEOBANKER_REUSE_MAP.md` (the full plan)
- `PLATFORM_BLUEPRINT.md` §4 (multi-repo split)
- `conventions/UI_AUDIT_CHECKLIST.md` §A (sacred rule enforcement)
