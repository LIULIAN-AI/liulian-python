---
title: LIULIAN — Concrete Reuse Map from neo-banker
status: living document; the centre of iteration-3 planning
owner: Linlin Jia (jajupmochi)
created: 2026-05-12
last_revised: 2026-05-12
inputs:
  - neo-banker/neobanker-agent       (private, cloned to /tmp/neobanker-refs/)
  - neo-banker/neobanker-crawler     (private, cloned)
  - neo-banker/neobanker-frontend-MVP-V3 (private, cloned)
  - neo-banker/neoctl                (private, cloned)
  - neo-banker/neobanker-dev-env     (private, cloned)
  - jajupmochi/claude-config         (public, cloned)
> **Language:** English | [中文](NEOBANKER_REUSE_MAP.zh.md) *(zh stub pending)*
---

# LIULIAN — Concrete Reuse Map from neo-banker

> Every LIULIAN repo starts as a **fork** of a neo-banker repo where one
> exists. This doc lists, file by file, what we copy, what we keep, what
> we adapt, and what we delete. It is the source of truth for iteration
> 3 of the platform plan; the blueprint and sprint docs both depend on
> it.

## 0.1 Sacred rule — code reuse vs. visual originality (NON-NEGOTIABLE)

The single most important rule in this entire doc:

> **Code, architecture, plumbing, vocabulary, deploy patterns** —
> reuse / adapt / fork freely from neobanker and the 12 reference
> platforms.
>
> **Visual design, brand voice, UI composition, iconography,
> typography choices, micro-interactions** — **original**. Reference
> for direction; never copy. LIULIAN must have *its own* aesthetic.

Operationalised: the *reuse fractions* quoted later in this doc apply
to **code lines and file structure**. They do **not** apply to design.
Even when we reuse `CanvasOrchestrator.tsx` (90% of its logic), every
visible pixel rendered by it is restyled per LIULIAN's tokens, every
piece of copy is rewritten, every shape and rhythm is decided by *our*
brand — anchored in the `feat/gui-demo` editorial-Swiss canon
(§14.5).

Concrete tests we apply on every PR (lifted into `docs/strategy/conventions/UI_AUDIT_CHECKLIST.md`):

- *AI-slop test*: would a viewer say "AI made this" without doubt?
  → reject.
- *Category-reflex test*: could someone guess our domain from palette
  alone ("hydrology → blue", "finance → navy+gold")? → reject if yes.
- *Neobanker-copy test*: does any screenshot look like a neobanker
  screen with text swapped? → reject; restyle until the lineage is
  invisible.
- *gui-demo cross-check*: is the warm-bone canvas + UniBe red as
  the sole spot color + editorial typography still load-bearing?
  → if not, restate why.

This rule overrides any reuse-percentage incentive in this document.
Reuse code, originate design.

## 0. Why fork-and-adapt rather than green-field

The five repos under `neo-banker/*` we have access to (agent, crawler,
frontend, neoctl, dev-env) are *production-shaped*, *well-tested*, and
*bilingually documented* solutions to the same architecture LIULIAN
needs: a Python core + a FastAPI agent service with multi-provider LLM
gateway + a Next.js frontend with AI chat + a deploy CLI + a unified
Codespaces dev env. Re-deriving them from scratch costs weeks for zero
incremental insight. Forking + adapting costs days for the same
production-readiness *and* keeps the operator's muscle memory aligned
(same CLI verbs, same SSE event names, same .env vocabulary).

The user's words: *"大部分的配置和agent相关的东西都可以复用优化自
neobanker的几个库，包括前端后端agent和爬虫"*. This document operationalises
that.

## 1. Repo-to-repo fork map

| LIULIAN repo (new) | Forked from | Branch model | Cross-link to LIULIAN-specific change set |
|---|---|---|---|
| **`liulian-agent`** | `neo-banker/neobanker-agent` | fork → rename → squash → push to `jajupmochi/liulian-agent`; keep history visible | §2 below |
| **`liulian-ingest`** | `neo-banker/neobanker-crawler` | fork → rename → push | §3 |
| **`liulian-web`** | `neo-banker/neobanker-frontend-MVP-V3` | fork → rename → push; redesign brand layer only | §4 |
| **`liulian-ops`** | `neo-banker/neoctl` | fork → rename → push | §5 |
| **`liulian-dev-env`** | `neo-banker/neobanker-dev-env` | fork → rename → push | §6 |
| **`liulian-api`** | (none — green-field) | new repo; *borrows patterns* from neobanker-backend-MVP-V2 (Spring Boot → FastAPI translation) | §7 |
| **`liulian-mobile`** | (none — Expo template) | new repo; *borrows* design tokens from `liulian-web` and SSE event shapes from `liulian-agent` | §8 |
| **`liulian-python`** | THIS repo | existing — only minor cleanup happens here (package boundary, gui-demo stays as orphan branch) | §9 |
| **`liulian-design-system`** | (none) | new repo; *seeded* from `apps/web/styles/main.css` of the `feat/gui-demo` worktree of this repo | §10 |

Plus an organisation-wide config repo:

| Config artefact | Sourced from | LIULIAN use |
|---|---|---|
| `.claude/` per-repo | `jajupmochi/claude-config` via `/init-claude-config` | scaffold each new repo with the curated 9 rules + relevant skills + hooks + recommendations |
| Documentation rules | `claude-config/rules/bilingual-docs/RULE.md` | adopted as `docs/strategy/conventions/DOCUMENTATION_RULES.md` (see §11) |

---

## 2. `liulian-agent` ← `neobanker-agent`

### 2.1 What the source repo gives us (already production-grade)

`neobanker-agent` is a FastAPI service on port 8000 with `/health` and
SSE-streaming `/agent/chat`. Architecture (mirror, then explain):

```
┌──────────────────────────────────────────────────────────────┐
│ main.py — FastAPI bootstrap + lifespan + routers             │
├──────────────────────────────────────────────────────────────┤
│ agent/                                                       │
│   loop.py        — main exec loop (intent → planner → tool)  │
│   intent.py      — LLM intent classifier (call #1)           │
│   intent_shortcuts.py — early-exit short paths               │
│   planner.py     — tool orchestration (LLM call #N)          │
│   state.py       — session state + ToolCallRecord            │
│   provider_registry.py — thread-safe Provider registry       │
│   provider_policy.py   — region routing (CN-block Gemini)    │
│   conversation_cache.py — cross-turn memory + entity         │
│   catalog_cache.py — pre-loaded data catalog                 │
│   context_memory.py — frontend context injection             │
│   demo_scenarios.py — canned-demo matching                   │
│   reliability.py — 5-tier source reliability scoring         │
│   suggestions.py — next-step recommendation generator        │
│   error_log.py   — structured error logging                  │
├──────────────────────────────────────────────────────────────┤
│ llm/                                                         │
│   config.py      — config + dotenv parsing (ProviderSpec)    │
│   harness.py     — LLM call wrapper                          │
│   gateway.py     — multi-region provider chain               │
│   providers/                                                 │
│     claude.py · gemini.py · glm.py · ollama.py · mock.py     │
├──────────────────────────────────────────────────────────────┤
│ tools/                                                       │
│   db_reader · calculator · web_search · bank_matcher         │
├──────────────────────────────────────────────────────────────┤
│ routers/                                                     │
│   chat.py        — SSE /agent/chat                           │
│   settings.py    — /agent/settings/*                         │
├──────────────────────────────────────────────────────────────┤
│ datasource/                                                  │
│   client.py      — DataSourceClient (CSV / backend / auto)   │
├──────────────────────────────────────────────────────────────┤
│ prompts/                                                     │
│   intent / planner / summary / suggestions templates         │
├──────────────────────────────────────────────────────────────┤
│ fixtures/    — demo scenarios + bank-data JSON               │
│ neobanker-agent.service — systemd unit                       │
│ mkdocs.yml + docs/ — bilingual docs (10× .md + .en.md pairs) │
└──────────────────────────────────────────────────────────────┘
```

**SSE event stream**: `thinking → trace → intent → tool_call →
tool_result → response → suggestions → done`. Frontend hooks into this
directly with EventSource.

**Provider chain pattern**: `GatewayDecision` holds region + chain +
used/skipped/errors; if Gemini is in a CN-blocked region we skip it
silently and try GLM/Claude/Ollama in order. The same pattern works
unchanged for LIULIAN.

### 2.2 File-by-file plan

**Keep verbatim or with rename-only**:

| File / dir | Status | Notes |
|---|---|---|
| `main.py` | rename refs only (`agent` → `liulian_agent`) | FastAPI entrypoint, lifespan |
| `agent/loop.py` | keep | core loop generalises to forecasting |
| `agent/intent.py` | keep | classifier; we add new intents (see §2.4) |
| `agent/planner.py` | keep | tool orchestration |
| `agent/state.py` | keep | session state |
| `agent/provider_registry.py` | keep verbatim | provider registry — generic |
| `agent/provider_policy.py` | keep verbatim | region routing — generic |
| `agent/conversation_cache.py` | keep | memory layer |
| `agent/catalog_cache.py` | adapt | catalog = manifest list, not bank list |
| `agent/context_memory.py` | keep verbatim | generic |
| `agent/reliability.py` | keep | 5-tier scheme reused as-is for forecast confidence |
| `agent/suggestions.py` | keep | generic |
| `agent/error_log.py` | keep | generic |
| `agent/intent_shortcuts.py` | adapt | shortcuts become forecasting-specific (e.g. "show me Bern station" → direct query_forecasts call) |
| `llm/config.py` | keep verbatim | ProviderSpec dataclass |
| `llm/harness.py` | keep verbatim | LLM call wrapper |
| `llm/gateway.py` | keep verbatim | gateway + region routing |
| `llm/providers/*` | keep verbatim | claude.py · gemini.py · glm.py · ollama.py · mock.py — all 5 reused |
| `routers/chat.py` | keep | SSE chat router; same shape |
| `routers/settings.py` | adapt | settings UI knobs differ |
| `datasource/client.py` | adapt | swap "bank data" for "forecasting catalog" — interface unchanged |
| `prompts/intent.py` (etc.) | rewrite | LIULIAN's intents differ from neobanker's |
| `neobanker-agent.service` | rename to `liulian-agent.service` | systemd unit |
| `mkdocs.yml` | keep | docs build |
| `docs/*` | rewrite content, keep file structure | bilingual docs survive |

**Tools (`tools/`) — total rewrite, same registry pattern**:

| neobanker tool | LIULIAN tool (replaces) | Pydantic input |
|---|---|---|
| `db_reader` | `query_forecasts` | `dataset_id`, `station_ids`, `window`, `models` |
| `calculator` | `compute_metric` | `metric: 'mae'|'rmse'|'crps'|'coverage90'`, `run_ids` |
| `web_search` | keep (`web_search`) | as-is; useful for "what's the weather forecast for X tomorrow" |
| `bank_matcher` | `station_matcher` | fuzzy-match user-typed station names to manifest IDs |
| — (new) | `recommend_model` | `dataset_id`, returns ranked-list |
| — (new) | `propose_hpo_space` | `model_id`, returns search space JSON |
| — (new) | `diagnose_failed_run` | `run_id`, returns reason + remediation |
| — (new) | `add_panel` | `report_id`, `panel_spec` |
| — (new) | `create_alert_rule` | `expr`, `channel` |

All tools share the `ToolRegistry` pattern from `tools/registry.py` —
copied verbatim.

### 2.3 What we delete

- `fixtures/demo_bank_data.json` + `demo_scenarios.json` — replaced by
  `fixtures/demo_swissriver_scenarios.json`.
- bank-domain prompts in `prompts/`.
- `tools/bank_matcher.py` — replaced by `station_matcher`.

### 2.4 New intents (LIULIAN-specific)

Append to `agent/intent.py`'s intent vocabulary:

- `forecast_query` — "what's the prediction for Bern tomorrow"
- `forecast_compare` — "compare TimesNet vs Chronos on this dataset"
- `alert_setup` — "alert me when Q95 > 850"
- `model_recommendation` — "which model fits this data"
- `run_diagnosis` — "why did run X fail"
- Keep generic: `conversation`, `data_lookup`, `comparison`

### 2.5 LLM provider repointing

`neobanker-agent` ships with **GLM / Gemini / Claude / Ollama / Mock**
provider modules already; we *add* DeepSeek (which is OpenAI-API-compatible
so we may not need a new module — can use `GLMProvider`'s base-URL knob
or write a 30-line `DeepSeekProvider`). Total work: ≤ 1h.

### 2.6 Estimated reuse fraction

Lines copied verbatim: ~70%.
Lines adapted (rename, prompt rewrite, tool swap): ~25%.
New code: ~5%.

---

## 3. `liulian-ingest` ← `neobanker-crawler`

### 3.1 Source structure (lean)

```
neobanker-crawler/
├── src/crawler/         (Python package)
├── config/
│   ├── sources.yaml         (declarative source list)
│   └── data_dictionary.yaml (target schema)
├── tools/
│   └── scan_java_columns.py
├── docs/specs/
├── tests/
├── neobanker-crawler.service (systemd unit)
└── pyproject.toml          (uv + ruff + mypy strict)
```

### 3.2 File-by-file plan

- `src/crawler/` → rename package to `liulian_ingest/`; preserve every
  utility (HTTP retries, idempotency markers, manifest writer).
- `config/sources.yaml` → **rewrite** with our actual sources:
  - `swisstopo-bafu` — Swiss hydrology
  - `meteoswiss-precip` — Swiss precipitation
  - `swissgrid` — Swiss energy
  - `physionet-mit-bih` — ECG demo
  - `caltrans-pems` — US traffic
  - `electricity-uci` — UCI electricity load
- `config/data_dictionary.yaml` → **rewrite** with LIULIAN's manifest
  schema (mirror `liulian-python/manifests/_schema.yaml`).
- `tools/scan_java_columns.py` → drop; unrelated.
- `docs/specs/` → keep structure, rewrite to LIULIAN's data contracts.
- `neobanker-crawler.service` → `liulian-ingest.service`.
- `pyproject.toml` → rename project, keep all dev tools (uv / ruff /
  mypy strict / pytest).

### 3.3 New: manifest auto-PR

Crawler writes a parquet to MinIO + emits a *suggested manifest* (YAML)
to `liulian-python/manifests/{vertical}/`. The PR is auto-opened by a
GitHub Actions job in `liulian-ingest` against `liulian-python`'s
`main`. The Pydantic-typed schema is shared via a tiny `liulian-schemas`
sub-package published on PyPI.

### 3.4 Estimated reuse fraction

Lines copied verbatim: ~80%.
Lines adapted (sources.yaml, dictionary): ~15%.
New code (manifest auto-PR): ~5%.

---

## 4. `liulian-web` ← `neobanker-frontend-MVP-V3`

### 4.1 Source stack (rich!)

```json
{
  "next": "^14.2.0",
  "@clerk/nextjs": "^5.7.5",
  "antd": "^5.26.4",
  "@ant-design/x": "^1.2.0",          // AI-chat UI primitives
  "antd-style": "^3.7.1",
  "echarts": "^5.6.0",
  "echarts-for-react": "^3.0.2",
  "@amcharts/amcharts5": "^5.14.3",   // geographic maps
  "@amcharts/amcharts5-geodata": "^5.1.5",
  "react-mosaic-component": "^7.0.0-beta0",   // tile-able dashboards!
  "react-resizable-panels": "^4.10.0",        // split panes
  "contentlayer": "^0.3.4",           // MDX content
  "framer-motion": "^12.38.0",
  "lucide-react": "^1.11.0",
  "react-markdown": "^10.1.0",
  "next-themes": "^0.4.6",
  "next-intl": (next-intl)            // confirmed in user memory
}
```

Plus storybook, vitest, playwright, e2e.

### 4.2 Decision: hybrid antd + shadcn

`neobanker-frontend-MVP-V3` is **all-antd**. Our LIULIAN brand is
editorial-Swiss with UniBe red on warm paper — antd's enterprise blue
defaults fight us. But `@ant-design/x` is best-in-class for AI chat
UI, and `react-mosaic-component` is best-in-class for tile-able
dashboards. So:

| Layer | Choice | Reason |
|---|---|---|
| Marketing site (`/`) + Studio (`/studio`) | **shadcn/ui + Tailwind** | Editorial brand; pixel-level control |
| Forecast canvas (`/forecast`) frame | **shadcn/ui** | Brand surface |
| Forecast canvas **inside-panels** | **react-mosaic-component** (from neobanker stack) | Tile + drag + resize is the BI canvas idiom |
| Charts | **echarts-for-react** | Same as neobanker |
| Map | **MapLibre GL** (LIULIAN, not amcharts5) | Swisstopo open tiles + WebGL; we add @amcharts5-geodata as fallback |
| Chat sidebar | **@ant-design/x** | World-class AI chat primitives; antd themed to brand colours via `antd-style` |
| Auth | **@clerk/nextjs** | Same as neobanker |
| i18n | **next-intl** | Same as neobanker; we adopt the bilingual rule for messages too |
| MDX content (docs + blog) | **contentlayer** | Same as neobanker |
| Animation | **framer-motion** | Same |
| Tests | **vitest + playwright** | Same |
| Storybook | **storybook** | Same |
| Theme switcher | **next-themes** | Same |
| Icons | **lucide-react** | Same — phosphor was also considered; lucide wins because neobanker uses it |
| Resizable panels | **react-resizable-panels** | For station-list ↔ canvas splits |
| Markdown | **react-markdown** | For agent's response rendering |

This hybrid keeps neobanker's mature plumbing while letting our brand
breathe through the editorial Swiss design layer.

### 4.3 File-by-file plan

- `app/` → keep App Router structure; the route groups change:
  - neobanker has `/dashboard`, `/products`, `/companies`, `/news`, …
  - LIULIAN has `/forecast`, `/studio`, `/agents`, `/admin`, …
  - rewrite the route segments; the layout chrome (`app/layout.tsx`,
    `app/(marketing)/layout.tsx`) is *adapted* (replace antd default
    chrome with shadcn + brand tokens).
- `components/` → keep `components/chat/*` (chat sidebar), `components/charts/*`,
  `components/mosaic/*`; **delete** `components/banking/*`, etc.
- `config/` → rewrite (new env vars: `LIULIAN_API_URL`, `LIULIAN_AGENT_URL`).
- `content/` → rewrite (replace bank news posts with LIULIAN release notes + docs).
- `contentlayer.config.js` → keep with new content types
  (BlogPost → ReleaseNote, Product → VerticalCard).
- `contexts/` → keep `ThemeContext`, `AuthContext`; rewrite domain
  contexts.
- `deploy.sh`, `dev.sh`, `dev-docker.sh`, `ecosystem.config.js` → keep
  with hostname renames.
- `e2e/` → keep playwright config; rewrite test scenarios.
- `hooks/` → keep generic hooks; rewrite domain hooks.
- `i18n.ts`, `middleware.ts`, `messages/{en,zh}.json` → keep; rewrite
  string tables.
- `next.config.js`, `postcss.config.js`, `tailwind.config.ts` → keep
  with brand token overrides from `liulian-design-system`.
- `lib/` → keep generic libs (api client, telemetry); rewrite domain libs.
- `playwright.config.ts`, `vitest.config.ts` → keep.

### 4.4 Brand override layer

Even though we keep antd for the chat + tables, we override **every**
ant token via `antd-style`:

```ts
// liulian-web/lib/antd-theme.ts
import { theme } from 'antd';
import { tokens } from '@liulian/design-tokens';

export const liulianAntdTheme = {
  algorithm: theme.defaultAlgorithm,
  token: {
    colorPrimary: tokens.colors.unibeRed,        // #E20613
    colorBgBase: tokens.colors.canvasWarm,
    colorTextBase: tokens.colors.inkCharcoal,
    fontFamily: tokens.fonts.body.join(', '),    // Switzer, …
    fontFamilyCode: tokens.fonts.mono.join(', '),
    borderRadius: 10,
    borderRadiusLG: 14,
    wireframe: false,
  },
  components: {
    Button: { borderRadius: 6, fontWeight: 500 },
    Card: { borderRadiusLG: 10 },
    // …
  },
};
```

Apply globally via `<ConfigProvider theme={liulianAntdTheme}>` in
`app/layout.tsx`.

### 4.5 Visual substrate

The `feat/gui-demo` branch of `liulian-python` is the **canonical visual
direction** (editorial Swiss; UniBe red `#E20613`; Fraunces + Switzer +
JetBrains Mono; 4-tab IA Data/Train/Inference/Insight; warm-bone canvas
with low-opacity red radial spots; 12-col × 8-row bento grid). All
visual decisions in `liulian-web` cite the `gui-demo`'s
`styles/main.css` and `docs/design-report.md`.

### 4.6 Estimated reuse fraction

Lines copied verbatim: ~50% (chrome + chat + charts + tooling).
Lines adapted (route swap, antd theme override, content rewrite): ~40%.
New code (BI canvas with mosaic + map + agent integration): ~10%.

---

## 5. `liulian-ops` ← `neoctl`

### 5.1 Source structure (already minimal + production-grade)

```
neoctl/
├── neoctl/
│   ├── cli.py          # click entrypoint
│   ├── config.py       # config loader
│   ├── deploy/         # subcommands (api / frontend / agent / all)
│   ├── detect.py       # env/health detector
│   ├── doctor.py       # diagnostics
│   ├── llm_setup.py    # Ollama + vLLM bootstrapping on GPU host
│   ├── ssh.py          # paramiko wrapper
│   └── tunnel.py       # autossh forward/reverse tunnels
├── docs/
│   ├── architecture.md
│   ├── commands-reference.md
│   ├── deployment-guide.md
│   ├── manual-deployment.md   # we already studied this
│   ├── troubleshooting.md
│   └── index.md
├── pyproject.toml      # click + paramiko + pyyaml + rich + httpx
└── tests/
```

### 5.2 File-by-file plan

- `neoctl/cli.py` → keep verbatim shape, rename `neoctl` → `liulianctl`,
  swap service names.
- `neoctl/config.py` → keep verbatim; reload config from
  `~/.liulian/config.yaml`.
- `neoctl/deploy/` → keep verbatim shape, swap services
  (`backend` → `api`, `frontend` → `web`, `agent` → keep, add `ingest`,
  add `mobile` via EAS Build invocation).
- `neoctl/llm_setup.py` → keep verbatim — exact same Ollama + vLLM
  bootstrap pattern (port 37434 for Ollama, 38000 for vLLM).
- `neoctl/ssh.py` + `tunnel.py` → keep verbatim. The autossh tunnels are
  identical for LIULIAN's GPU host.
- `neoctl/detect.py` + `doctor.py` → keep verbatim; very generic.
- `docs/manual-deployment.md` → keep structure, swap service names.

### 5.3 What we add

- `liulianctl/deploy/mobile.py` → wrap `eas build --profile preview
  --platform all` + `eas update` for OTA.
- `liulianctl/deploy/web.py` → wrap `vercel deploy --prod`.
- `liulianctl/manifest_sync.py` → run `liulian-ingest` once, then PR
  the manifest into `liulian-python`.

### 5.4 Estimated reuse fraction

Lines copied verbatim: ~85%.
Adapted: ~10%.
New: ~5%.

---

## 6. `liulian-dev-env` ← `neobanker-dev-env`

### 6.1 Source (per repo description)

> "Unified Codespaces dev environment for neobanker frontend/backend/agent"

A devcontainer / Codespaces config that boots all repos with one click.

### 6.2 Adaptation

- Add `liulian-python`, `liulian-api`, `liulian-agent`, `liulian-ingest`,
  `liulian-web`, `liulian-mobile` to the workspace.
- Pre-install: `uv`, `pnpm`, `docker compose`, `terraform`, `helm`,
  `kubectl`, `eas-cli`, `gh`.
- Pre-seed `.env` files with placeholder API keys for DeepSeek / GLM /
  Gemini / Clerk / Sentry.
- One-click `make dev` → `docker compose up` of postgres-timescaledb,
  redis, minio, prometheus, grafana, loki, tempo.

### 6.3 Estimated reuse fraction

Lines copied verbatim: ~70%.
Adapted: ~30%.

---

## 7. `liulian-api` (green-field, but borrowing patterns)

`neobanker-backend-MVP-V2` is Java/Spring Boot — not directly portable
to our Python stack, but the **API shapes** (controllers, DTOs,
pagination, error envelopes) translate cleanly to FastAPI. We
specifically borrow:

- The pagination contract: `{items, total, page, page_size}`.
- The error envelope: `{code, message, details}` (RFC-7807-ish).
- The audit-log row shape.
- The Clerk-JWT verification middleware (Spring Filter → FastAPI
  middleware translation).
- The CORS allowlist pattern (`AGENT_CORS_ALLOWED_ORIGINS` env var —
  same name reused).

Tracker + experiment-runner-around-`liulian-python` is wholly new
(neobanker has no analogue). Schema in PLATFORM_BLUEPRINT §5.3.

---

## 8. `liulian-mobile` (green-field on Expo template)

No neobanker mobile to fork. We start from `pnpm create expo-app --template tabs-typescript`.

We borrow from `liulian-web`:
- Design tokens (`@liulian/design-tokens`).
- API types (codegen'd from OpenAPI of `liulian-api`).
- The SSE-consumer pattern for the chat sidebar — neobanker-frontend's
  EventSource hook ports directly to React Native's `EventSource`
  polyfill.

---

## 9. `liulian-python` (this repo) — minimal change

Per the user's instruction *"liulian-python 应该在一个专门的 branch 上
做"*, all platform work happens on `feat/platform-upgrade-2026-05`. The
only changes to this repo:

1. Strategy docs in `docs/strategy/` (this set).
2. Documentation conventions in `docs/strategy/conventions/`.
3. ADRs in `docs/strategy/adr/`.
4. *No* changes to `liulian/` core (the Python package stays
   API-compatible). The package boundary cleanups proposed in iteration
   1 are deferred — `liulian-api` consumes `liulian` as a dependency
   via PyPI version pin; it doesn't import internals.
5. `experiments/` and `manifests/` untouched (the research moat).
6. `README.md` updated last (Day 6 of sprint) to point at the new
   sibling repos.

The `feat/gui-demo` orphan branch is preserved as-is; `liulian-web`
imports its design substrate via copy, not via git.

---

## 10. `liulian-design-system` (new, seeded from gui-demo)

Source: `liulian-python/.worktrees/gui-demo/styles/main.css` + the
brand decisions documented in `liulian-python/.worktrees/gui-demo/docs/design-report.md`.

Output of the new repo:
- `tokens.json` — single source of truth.
- `tokens.css` — CSS custom properties.
- `tailwind.preset.js` — Tailwind preset.
- `tokens.ts` — TS const for type-safe access in `liulian-web`.
- `tokens.rn.ts` — RN StyleSheet exports for `liulian-mobile`.
- `antd-theme.ts` — antd ConfigProvider theme block for chat sidebar.
- `figma.fig` — companion Figma library (linked from README).

Published as `@liulian/design-tokens` on the GitHub Packages npm
registry (private at first; public when M2 ships).

---

## 11. Documentation rules (adopt from `jajupmochi/claude-config`)

The user runs his own curated `claude-config` repo with **9 workflow
rules**. LIULIAN adopts a focused subset, codified as
`docs/strategy/conventions/DOCUMENTATION_RULES.md`. The most
load-bearing for this work:

### 11.1 `bilingual-docs` (claude-config rule, opted in for LIULIAN)

- Every repo-level human-facing doc ships as **`NAME.md`** (English
  canonical) + **`NAME.zh.md`** (Chinese mirror).
- Switcher header on top of each file:
  - `NAME.md`: `> **Language:** English | [中文](NAME.zh.md)`
  - `NAME.zh.md`: `> **语言：** [English](NAME.md) | 中文`
- Same headings, same TOC anchors, same code blocks; only prose
  translated.
- **Don't translate**: code, identifiers, filenames, JSON / YAML keys,
  URLs, hierarchy IDs, status markers.
- **Exceptions**: `CLAUDE.md`, `CLAUDE.local.md`, `SKILL.md`,
  `RULE.md`, hook READMEs, internal-only research notes.

### 11.2 Bilingual-canon orientation

Note neobanker's docs use the *opposite* convention: Chinese-canonical,
`.en.md` mirror. We follow **claude-config** (English-canonical,
`.zh.md` mirror) because (a) it's the user's curated standard, and (b)
English-canonical is more discoverable to international contributors.
When forking neobanker docs, swap the suffix: their `architecture.md`
(zh) becomes our `ARCHITECTURE.zh.md`, and `architecture.en.md`
becomes our `ARCHITECTURE.md`.

### 11.3 Four-layer doc structure

Each repo ships docs in four layers, aligned with the user's
"四层文档" intent (corresponding to claude-config's 9-rule structure
and the ADR-style separation):

```
docs/
├── strategy/                  ← L1–L2: long-term vision + architecture (this set)
│   ├── PLATFORM_BLUEPRINT.md  + .zh.md
│   ├── PLATFORM_DESIGN.md     + .zh.md
│   ├── ONE_WEEK_SPRINT.md     + .zh.md
│   ├── REFERENCE_DESIGNS.md
│   ├── NEOBANKER_REUSE_MAP.md + .zh.md
│   ├── adr/                   ← per-decision records
│   │   ├── 0001-multi-repo-split.md
│   │   ├── 0002-custom-agent-not-langgraph.md
│   │   ├── 0003-timescaledb-not-tdengine-now.md
│   │   ├── 0004-unibe-red-as-anchor.md
│   │   ├── 0005-tracker-three-entity-unified-table.md
│   │   └── 0006-fork-and-adapt-from-neobanker.md
│   └── conventions/           ← L0: doc/code rules
│       ├── DOCUMENTATION_RULES.md   (bilingual + 4-layer + headers)
│       ├── COMMIT_RULES.md
│       └── BRANCHING_RULES.md
├── L3-design/                 ← surface design (brand, IA, BI canvas, agent flows)
│   └── PLATFORM_DESIGN.md  (lives in strategy/ for now; promoted later)
├── L4-runbooks/               ← operator-facing
│   ├── deployment-guide.md    + .zh.md
│   ├── manual-deployment.md   + .zh.md  (mirrored from neoctl/docs)
│   ├── troubleshooting.md     + .zh.md
│   └── environment-setup.md   + .zh.md
└── api-reference.md           + .zh.md  (auto-generated from OpenAPI)
```

The four layers are: **L1 Vision** (why) → **L2 Architecture** (how
at system) → **L3 Design** (how at surface) → **L4 Implementation
+ runbooks** (what / when). L0 conventions sit underneath everything
(rules of engagement).

### 11.4 Other claude-config rules we adopt

| claude-config rule | Adopted? | Where applied |
|---|---|---|
| `chinese-output` | yes (personal) | end-of-turn final reply to user is Chinese |
| `pre-edit-confirmation` | yes (universal) | list + plan + go pattern before edits |
| `phased-planning` | yes (universal) | multi-file work phased |
| `plugin-preflight` | yes (universal) | verify before invoke |
| `output-brevity` | yes | no trailing recap |
| `tool-proactivity` | yes | installed skills fire when matched |
| `no-reread-files` | yes | trust session memory |
| `ui-iteration-loop` | yes (ui-project) | 8-iteration autonomous UI redesign with chrome-devtools screenshots |
| `bilingual-docs` | yes (opt-in) | this whole doc applies it |

These are imported into each new LIULIAN repo via
`/init-claude-config` (the claude-config setup skill).

---

## 12. CI/CD reused from neobanker

### 12.1 Per-repo GitHub Actions pattern

Each Python repo gets the same `.github/workflows/ci.yml`:

```yaml
# pattern lifted from neobanker-crawler/.github/workflows/ (mirrored)
name: ci
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy .
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: timescale/timescaledb-ha:pg16
        env: {POSTGRES_PASSWORD: ci, POSTGRES_DB: liulian_ci}
        ports: [5432:5432]
        options: --health-cmd pg_isready
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen --all-extras
      - run: uv run pytest -m "not slow and not download" --cov
```

JS repos use the analogous pattern (pnpm + biome/eslint + tsc + vitest +
playwright smoke).

### 12.2 Reusable workflows in `liulian-ops/.github/workflows/`

Hoist common steps to `liulian-ops`:

- `python-lint.yml`, `python-mypy.yml`, `python-pytest.yml`,
  `python-image.yml`
- `js-lint.yml`, `js-tsc.yml`, `js-vitest.yml`, `js-playwright.yml`,
  `js-image.yml`
- `release-tag.yml`, `release-notes.yml`

Per-repo workflow becomes a 6-line caller:

```yaml
jobs:
  ci:
    uses: jajupmochi/liulian-ops/.github/workflows/python-lint.yml@v1
```

### 12.3 Deploy workflow

Per repo: `release.yml` builds an image, pushes to GHCR, then triggers
`liulianctl deploy <service>` via SSH (paramiko in `liulian-ops`). This
matches neobanker's pattern (`neoctl deploy all` invoked from a CI
runner with SSH credentials).

---

## 13. What this reuse map enables

- **Sprint Day 1** is now "fork 5 repos and rename", not "scaffold 5
  greenfield repos". Estimated time saving: 2 days across the 7-day
  sprint.
- **Sprint Day 5** (agent) becomes "swap tools + prompts, repoint
  providers", not "design an agent from scratch".
- **Sprint Day 6** (deploy) leverages `neoctl` verbatim — manual
  deployment guide already exists.
- **Documentation cadence** inherits claude-config's discipline
  (bilingual, 4-layer, ADRs).

The first concrete commit of iteration 3 is this very document. The
second is the ADR (`0006-fork-and-adapt-from-neobanker.md`). The third
is the revised `PLATFORM_BLUEPRINT.md` (with this doc cross-linked).

---

## 14. Deeper reuse: backend, canvas orchestrator, modern framework

### 14.1 Backend (`neobanker-backend-MVP-V2` Spring Boot 3.1) — pattern translation

The neobanker backend is Java/Spring Boot. We translate, not fork, but
the *shape* is what we reuse:

| Spring Boot 3.1 pattern (source) | FastAPI / SQLModel translation (LIULIAN target) |
|---|---|
| Maven + `pom.xml` | `pyproject.toml` (uv) + `requirements.lock` |
| `spring-boot-starter-actuator` (`/actuator/health`) | FastAPI `/healthz` + `/readyz` (already in plan) |
| `spring-boot-starter-data-jpa` + `@Entity` | SQLModel + Alembic |
| `spring-boot-starter-data-elasticsearch` | Postgres trigram / pg_search initially; Elasticsearch only at M3 if needed |
| `spring-boot-starter-data-redis` | redis-py + arq workers (already in plan) |
| `spring-boot-starter-mail` | `aiosmtplib` for alert email delivery |
| `io.minio:minio` | `minio-py` (already in plan) |
| `mapstruct` (DTO ↔ Entity mappers) | Pydantic v2 `model_validate` / `model_dump` (no separate mapper layer) |
| `lombok` (getters/setters) | dataclasses + Pydantic (no boilerplate) |
| 44+ `@RestController`s | FastAPI `APIRouter`s, one per resource |
| `@Service` + `@Repository` layers | `services/*.py` + `repositories/*.py` (kept symmetrical to make hires legible) |
| `application.yml` profiles | `liulian_api/config.py` per-environment (`dev`/`staging`/`prod`) via env vars; we *avoid* spring-style profile bloat |
| `import_data/` CSV → MySQL via Python | identical: `liulian-ingest` writes manifests + parquet → Postgres-TimescaleDB |
| `restart_neobanker.sh` | `liulianctl restart api` (already in plan) |
| 30 DB tables, mostly banking domain | replace with our 12-table tracker + forecast schema (PLATFORM_BLUEPRINT §5.3) |

**What we explicitly do NOT borrow**: the Java-specific verbosity (DTO
↔ entity mappers, Lombok ceremony, profile bloat). FastAPI + Pydantic v2
collapse three Java layers into one.

**API surface translation example** (Spring Boot → FastAPI):

```java
// SOURCE: src/main/java/com/neobanker/neobank/controller/CompanyController.java
@RestController
@RequestMapping("/api/companies")
public class CompanyController {
    @GetMapping("/{id}") public CompanyDto getById(@PathVariable Long id) { ... }
    @GetMapping       public Page<CompanyDto> list(Pageable p, @ModelAttribute Filter f) { ... }
    @PostMapping      public CompanyDto create(@RequestBody @Valid CompanyCreate body) { ... }
    @PutMapping("/{id}")    public CompanyDto update(...) { ... }
    @DeleteMapping("/{id}") public void delete(...) { ... }
}
```

→

```python
# TARGET: liulian_api/routers/experiments.py
router = APIRouter(prefix="/experiments", tags=["experiments"])

@router.get("/{id}", response_model=ExperimentRead)
async def get(id: UUID, svc: ExperimentService = Depends()): ...

@router.get("", response_model=Page[ExperimentRead])
async def list(filt: ExperimentFilter = Depends(), p: Pagination = Depends(), ...): ...

@router.post("", response_model=ExperimentRead, status_code=201)
async def create(body: ExperimentCreate, svc: ExperimentService = Depends()): ...

@router.put("/{id}", response_model=ExperimentRead)
async def update(id: UUID, body: ExperimentUpdate, svc: ExperimentService = Depends()): ...

@router.delete("/{id}", status_code=204)
async def delete(id: UUID, svc: ExperimentService = Depends()): ...
```

Pagination contract preserved verbatim across the translation: `{items,
total, page, page_size}`. Error envelope preserved: `{code, message,
details}`. Audit-log row shape preserved. This makes cross-platform
developer onboarding zero-friction.

### 14.2 Frontend assistant/canvas-orchestrator — the BIG reuse

`neobanker-frontend-MVP-V3/components/assistant/` contains a complete
**AI-driven canvas + report builder**. Files:

```
components/assistant/
├── CanvasOrchestrator.tsx       ← orchestrates panels on canvas
├── DynamicCanvas.tsx            ← canvas-level layout engine
├── CanvasToolbar.tsx            ← canvas top bar (add/save/share)
├── ReportBuilder.tsx            ← drag-and-drop report builder
├── WidgetConfigPanel.tsx        ← per-widget config sidebar
├── widgets/                     ← widget library (panels)
├── SessionSidebar.tsx           ← left rail with sessions/chats
├── SmartSwitcherDock.tsx        ← multi-view dock
├── StatusBar.tsx                ← bottom system status
├── HeaderBar.tsx                ← canvas header
├── BankSelector.tsx             ← entity picker  ← rename to StationSelector
├── DegradationBanner.tsx        ← service-degraded banner
├── ToastContainer.tsx           ← toasts
├── data/                        ← canvas data layer
├── fluent/                      ← micro UI elements
├── hooks/                       ← canvas state hooks
├── utils/                       ← helpers
└── types.ts                     ← TS types
```

This is **directly reusable** for LIULIAN's `/forecast` canvas. Rename
+ rewire:

| Source file | LIULIAN file | Action |
|---|---|---|
| `CanvasOrchestrator.tsx` | `app/forecast/components/CanvasOrchestrator.tsx` | rename refs only |
| `DynamicCanvas.tsx` | same name | keep verbatim |
| `CanvasToolbar.tsx` | same | minor brand adjustments |
| `ReportBuilder.tsx` | same | keep verbatim — this IS our report builder |
| `WidgetConfigPanel.tsx` | same | keep verbatim |
| `widgets/` directory | rewrite all individual widgets | bank widgets → forecast widgets (per BI 8 panels in PLATFORM_BLUEPRINT §8) |
| `SessionSidebar.tsx` | same | keep — chat sessions |
| `SmartSwitcherDock.tsx` | same | useful for multi-canvas |
| `StatusBar.tsx` | same | keep |
| `HeaderBar.tsx` | same | brand adjustments |
| `BankSelector.tsx` | `StationSelector.tsx` | rewrite per-entity logic |
| `DegradationBanner.tsx` | same | when agent is offline |
| `data/` | rewrite | API endpoints differ |
| `hooks/` | adapt | useCanvas / useWidget / useAgent — generic mostly |
| `types.ts` | adapt | rename Bank → Station + add Forecast/Run types |

**This single reuse saves 2–3 weeks of bespoke React work** for a
production-grade canvas with drag/drop/save/share/widget-config.

### 14.3 Modern professional dashboard framework — Refine.dev for /studio

User feedback: *"或许可使用更现代专业的产品模式代码和框架"*. After
2026-survey of the dashboard framework landscape:

- **Refine.dev** — headless framework, multi-UI (Mantine / Ant Design /
  MUI / Chakra / shadcn), data-provider abstraction (REST / GraphQL /
  Supabase / Hasura / Strapi), auth (Auth0 / Keycloak / custom JWT),
  built-in `useTable` / `useForm` / `useSelect` with sort+filter+page.
- **shadcn-admin** — popular OSS template; basic CRUD scaffolding.
- **Tremor** — analytics-specialised; KPI cards + chart primitives.
- **Mantine** — 100+ components, strong for B2B; not a framework but a
  component library.

**Decision** (encoded as ADR 0007 below): **Refine.dev for `/studio`
CRUD layers, hand-rolled (using `liulian-design-system` + `react-mosaic-component`
+ `@ant-design/x` + ECharts + MapLibre) for `/forecast` BI canvas,
shadcn-admin patterns for `/admin`, framer-motion + Magic UI / Aceternity
for `/(marketing)`.**

Layer map:

| Surface | Framework | Why |
|---|---|---|
| `/(marketing)` landing | hand-rolled + framer-motion + Magic UI / Aceternity | brand is editorial Swiss; we control every pixel |
| `/studio/experiments` | **Refine.dev** + shadcn UI | CRUD over experiments, runs, models, datasets — generic, framework saves 2 weeks |
| `/studio/models` | **Refine.dev** + shadcn UI | model registry CRUD |
| `/studio/datasets` | **Refine.dev** + shadcn UI | dataset CRUD + manifest editor |
| `/studio/users` (M3+) | **Refine.dev** + Clerk integration | user / tenant admin |
| `/forecast` BI canvas | **hand-rolled** + neobanker `assistant/Canvas*` + `react-mosaic-component` + ECharts + MapLibre + `@ant-design/x` chat sidebar | the differentiated surface |
| `/forecast/r/{slug}` shared report | hand-rolled + ECharts + Plotly fan-charts | read-only public reports |
| `/agents/{persona}` chat-only | **`@ant-design/x`** | AI chat is its specialty |
| `/admin` ops console | **shadcn-admin** patterns | per-tenant ops, jobs, alerts |
| `/docs` MDX site | **contentlayer** (from neobanker) + Tailwind typography | technical + bilingual |

Refine.dev's *headless* mode is critical: we can plug `liulian-api`
straight in as a custom data provider, no GraphQL/Supabase
intermediary.

### 14.4 12 reference-platform designs we explicitly reuse

The original `REFERENCE_DESIGNS.md` listed 12 platforms. Concrete
reuse pulled forward into this map:

| Reference | Specific design reused | Where applied in LIULIAN |
|---|---|---|
| Time-Series-Library (THU-ML) | leaderboard table layout | `/docs/benchmarks/leaderboard.md` (auto-gen nightly via `bench.yml` workflow) |
| Chronos / Chronos-2 | "Pipeline.from_pretrained(...)" API shape | `liulian.adapters.chronos.adapter` mirrors signature |
| GluonTS | probabilistic confidence-band default (Q05/Q95) | `<ForecastChart />` default rendering |
| pytorch-forecasting | TFT attention interpretation panel | extra widget in `/forecast` canvas (M2+) |
| TSL (torch-spatiotemporal) | engines / data / model separation | already mirrored in `liulian/runtime/` |
| tslearn | k-Shape clustering → cohort widget | new widget: "Station Cohorts" |
| ClearML | three-layer story shape (infra / dev / serve) | adapted: research core / platform / vertical |
| MLflow | tracking REST endpoint shape | `liulian-api` exposes MLflow-compatible REST shim (PLATFORM_BLUEPRINT §11.4) |
| W&B | "compare runs" multi-select table view | `/studio/experiments/compare?ids=...` page |
| Ultralytics HUB | "train → export → deploy" 3-click flow | `/studio` → `/forecast` flow with one-click model promote |
| FineBI | "management cockpit" KPI + map + trend layout | the 8-panel SwissRiver canvas |
| Power BI custom visual | solid obs + dashed forecast + shaded CI | `<ForecastChart />` exact rendering |
| Tremor | KPI cards / sparkbars / micro-metrics | `/forecast` KPI strip |
| TDengine | hypertable-style perf, stream processing | TimescaleDB now (PLATFORM_BLUEPRINT §5.1); migrate target |
| HydroForecast | hero landing imagery + scientific copy register | `/(marketing)` hero |
| k-dense.ai | three-verb hero rhythm + real scientific visuals | `/(marketing)` hero copy |
| Datadog | severity ribbon for alerts | `/forecast` alert-timeline panel |

The cumulative effect: every panel of LIULIAN has a *named ancestor* in
a respected product. None of our visual language is "AI-default generic".

### 14.5 gui-demo as the visual canon — explicit combination plan

The `feat/gui-demo` orphan branch (commit `ce13ff0`) contains 6
iterations of brand work culminating in the **editorial Swiss + UniBe
red** identity. It is the canonical visual reference. The new
`liulian-web` repo's design phase is:

1. **Import** `gui-demo/styles/main.css` → `liulian-design-system/tokens.css`
   + auto-generate Tailwind preset.
2. **Recreate** the 4-tab IA (Data/Train/Inference/Insight) as the
   `/studio` left rail (each tab maps to a Refine.dev resource).
3. **Translate** gui-demo's bento grid (12-col × 8-row) into
   `react-mosaic-component` default layout for `/forecast`.
4. **Mirror** the 13-rule minimalist-ui audit table in
   `docs/strategy/conventions/UI_AUDIT_CHECKLIST.md` — every PR ticks
   each row.
5. **Preserve** the hand-rolled SVG charts from gui-demo as fallback
   renderers (when ECharts is unavailable, e.g. PDF export at server).

The gui-demo's `docs/design-report.md` is the **brand bible** — copied
verbatim to `docs/strategy/conventions/BRAND_VOICE.md` with bilingual
mirror.

### 14.6 ADRs reflecting iteration 3

These join the seeded list (PLATFORM_BLUEPRINT §18) — see
`docs/strategy/adr/`:

- `0006-fork-and-adapt-from-neobanker.md` — each new LIULIAN repo
  starts as a fork.
- `0007-refine-dot-dev-for-studio-crud.md` — Refine.dev for `/studio`
  layers; hand-rolled BI canvas for `/forecast`.
- `0008-frontend-canvas-orchestrator-reuse.md` — adopt
  `neobanker-frontend/components/assistant/Canvas*` files; rename
  Bank → Station.
- `0009-spring-boot-pattern-translation.md` — Spring Boot 3.1
  controller / DTO / service / repository pattern translates to FastAPI
  router / Pydantic / service / repository; no Mapstruct, no Lombok
  ceremony.
- `0010-hybrid-shadcn-antd-refine-stack.md` — record the framework
  hybrid (shadcn primary, antd ConfigProvider-themed for chat +
  high-density tables, Refine.dev for studio CRUD).

### 14.7 Estimated total reuse — revised with deeper inspection

| Repo | Verbatim | Adapted | New | Confidence |
|---|---|---|---|---|
| `liulian-agent` | 70% | 25% | 5% | high — code reviewed |
| `liulian-ingest` | 80% | 15% | 5% | high — small surface |
| `liulian-web` | **~55%** | ~30% | ~15% | medium — depends on Refine.dev fit |
| `liulian-ops` | 85% | 10% | 5% | high — code reviewed |
| `liulian-dev-env` | 70% | 30% | 0% | medium |
| `liulian-api` | 0% verbatim (Java→Python) | 50% pattern translation | 50% new | medium |
| `liulian-mobile` | 0% (greenfield) | 40% pattern from web | 60% new | high |
| `liulian-design-system` | 80% (from gui-demo) | 15% | 5% | high |

Weighted by repo size, the platform-wide reuse fraction is ~55–60% —
a **~3 month** acceleration on a from-scratch build, conservatively
estimated.

---

*Cross-links: see [PLATFORM_BLUEPRINT.md](PLATFORM_BLUEPRINT.md) §4 for
the high-level multi-repo decision, [ONE_WEEK_SPRINT.md](ONE_WEEK_SPRINT.md)
§2 for daily tasks now reframed as fork+adapt, and `adr/0006-0010` for
decision records of iteration 3.*
