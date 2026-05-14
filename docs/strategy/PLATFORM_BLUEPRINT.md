---
title: LIULIAN Platform Blueprint — Master
status: living document (revised iteration 2)
owner: Linlin Jia (jajupmochi)
created: 2026-05-12
last_revised: 2026-05-12 (multi-repo split, custom agent, UniBe red palette)
branch_of_record: feat/platform-upgrade-2026-05  (merge back to main after sprint)
four-layer companion docs (cross-link, don't duplicate):
  L1  Vision         — this doc, §1–§3
  L2  Architecture   — this doc, §4–§9
  L3  Design         — PLATFORM_DESIGN.md      (brand, BI canvas, agent flows)
  L4  Implementation — ONE_WEEK_SPRINT.md      (7-day execution)
  References         — REFERENCE_DESIGNS.md    (12-platform audit, refresh every 6 months)
bilingual:
  - PLATFORM_BLUEPRINT.zh.md   (Chinese summary, shorter)
audience: future contributors, recruiters, prospective investors
length: long; treat sections as independent reference, not linear reading
---

# LIULIAN Platform Blueprint (Master)

> Single source of truth for *what LIULIAN is becoming*. The four-layer
> design follows ADR-style separation: L1 Vision (why) → L2 Architecture
> (how-at-system-level) → L3 Design (how-at-surface-level) → L4
> Implementation (what-this-week). When in doubt, the deeper layer wins
> over the shallower one *only with an ADR explaining the override*.

## 0. What changed in iteration 2 (2026-05-12)

Compared to the first blueprint draft, the following decisions are
**revised**; older revisions live in git history:

1. **Multi-repo, not monorepo.** Modelled on liulian (`backend` +
   `frontend` + `agent` as three repos, orchestrated by `neoctl` CLI).
   We split the same way: `liulian-python` (core, this repo) +
   `liulian-api` + `liulian-web` + `liulian-mobile` + `liulian-agent` +
   `liulian-ingest` + `liulian-design-system` + `liulian-ops`. See §4.
2. **Custom agent, not LangGraph.** Built around `liulian-agent` repo
   using GLM-4.6 + DeepSeek-V4 + Gemini-3.1 via a thin provider-agnostic
   adapter (LiteLLM optional). Tool surface is Pydantic-typed; orchestration
   is a hand-written state graph in ~300 lines. See §7.
3. **Tracker covers tasks + experiments + agents** (three entity types,
   one tracker), not just experiments. See §8.
4. **Brand palette anchors on UniBe red `#E20613`** (matches the
   `feat/gui-demo` design already committed). Editorial Swiss aesthetic
   on a warm-bone canvas; *not* dark-mode SaaS default. See §11 and the
   companion `PLATFORM_DESIGN.md`.
5. **Branch workflow:** all platform work happens on
   `feat/platform-upgrade-2026-05` (this branch); merge back to `main`
   once L4 (sprint) completes. Any concurrent main work merged in
   pre-merge.
6. **gui-demo as design substrate:** the `feat/gui-demo` worktree's
   visual system (Fraunces · Switzer · JetBrains Mono · UniBe red ·
   warm paper · 4-tab IA Data/Train/Inference/Insight) is the canonical
   starting point for `liulian-web`. See REFERENCE_DESIGNS §F1.

---

## 1. Positioning (L1)

**One-liner**: *Open-source production stack for spatio-temporal AI —
research-grade model zoo · production-grade BI · agentic workflows ·
sovereign deployment.*

**The crisper four-word version we test**: *Liquid Intelligence for Time.*

**Why these words**:

- *Liquid* — the original LIULIAN brand etymology + the literal Swiss
  river flagship.
- *Intelligence* — not "AI" (saturated) or "analytics" (commodified).
- *for Time* — claims temporal as our axis; "for" reads as a service, not
  a buzzword.

### Anti-positioning

| Confused with | Why we're different |
|---|---|
| Time-Series-Library (THU-ML) | They are a model zoo. We are the *task runtime + BI + agent layer on top of* a curated model zoo (theirs + ours). |
| ClearML / W&B | Language-agnostic experiment trackers. We are an *opinionated TS/ST product* with first-class spatial data, BI canvas, and domain plugins. |
| Power BI / FineBI | General BI on relational data. We are TS/ST-native BI with forecasting overlays, prediction intervals, online retraining. |
| HydroForecast | Closed-source SaaS for one vertical. We are open-source core + vertical plugins (hydrology, traffic, energy, healthcare). |
| Ultralytics HUB | Vision-only. We are temporal-first. |
| liulian | Financial vertical, Java backend, business analyst audience. We are scientific vertical, Python backend, research-engineer audience. Borrow their *operational pattern* (multi-repo, neoctl-style CLI), not their domain. |

## 2. Audiences and personas (L1)

| Persona | Why they show up | What they convert on |
|---|---|---|
| **Open-source contributor** (PhD/researcher) | New SOTA model published, wants to benchmark | "Drop your model into `adapters/`, run our 5-dataset matrix in 1 command" |
| **Domain analyst** (hydrologist, energy trader, clinical researcher) | Has a SwissRiver-like dataset, needs forecasts + dashboards | BI canvas with their data, prediction intervals, exportable report |
| **ML platform engineer** (hiring manager evaluating you) | Lands on the repo from your CV | README architecture, Helm chart, GitHub Actions green, Grafana screenshots, `liulian-ops` CLI demo |
| **VC analyst / scout** | Lands from a referral | One-page product video, "deployed at X stations across CH", live SwissRiver demo URL |
| **Healthcare RSE recruiter** (ARTORG-AIHN) | Sees the platform from a job application | Mobile app, FastAPI swagger, ECG demo, privacy posture |

## 3. Success metrics for L1 (what makes M6 fundable)

- ≥ 1 paying pilot (CHF 1k–5k / mo) by **2026-09**.
- ≥ 25 monthly active users (each ran ≥1 experiment in last 28 d) by **2026-09**.
- ≥ 100 GitHub stars from non-friends by **2026-09**.
- ≥ 1 advisor on cap table (Mougiakakou / Fischer / Riesen / ETH-CH ML founder).
- Pitch deck + 18-month financial model + first LOI by **2026-11**.

---

## 4. Multi-repo architecture (L2)

The single-repo monorepo proposed in iteration 1 is replaced by a
**federation of focused repos** orchestrated by an operations CLI. Same
pattern as liulian (`backend` + `frontend` + `agent` repos, deployed
via `neoctl`).

### 4.1 Repo map

**7-repo federation** (revised after Sprint Day 1 evening user feedback;
see ADR 0001 §Mid-sprint mergers / un-mergers):

| Repo | Stack | Purpose | Branch in flight |
|---|---|---|---|
| **`liulian-python`** (this repo) | Python 3.10+ · numpy · pyyaml · torch (extra) · ray (extra) | **Research core only**: tasks · data · models · adapters · runtime · optim · viz · plugins. *No web, no FastAPI, no UI deps in core.* Stays `pip install liulian`-clean. | `feat/platform-upgrade-2026-05` (carries `docs/strategy/` only) |
| **`liulian-api`** | Python · FastAPI · Pydantic v2 · SQLModel · plain Postgres (TimescaleDB after M1) · Redis · arq | HTTP gateway over `liulian-python`. Owns OpenAPI contract. **Day-1 live: 6 endpoints (healthz, readyz, models, experiments x4, forecasts x2), 6/6 tests pass.** | `main` |
| **`liulian-agent`** | Python · custom orchestrator · DeepSeek/GLM/Gemini/Claude/Ollama providers · pgvector | Stand-alone LLM agent service (FastAPI on port 8001 with `/health`, mirroring liulian). 3 personas (data · model · BI). **Day-5 live: 9 tools registered, 9/9 tests pass.** | `main` |
| **`liulian-ingest`** | Python · async httpx · playwright · pydantic-extra-types | Runtime crawler service: scheduled fetchers for swisstopo BAFU / SwissGrid / PhysioNet etc; writes parquet to MinIO + auto-PRs manifest to `liulian-python`. **Kept separate** from ops (runtime service ≠ infra tooling). | `main` |
| **`liulian-web`** | TypeScript · Next.js 14 (App Router) · ECharts · Antd (chat sidebar) · Tailwind · framer-motion · contentlayer · MapLibre (planned) | BI canvas + marketing site + studio. Derives visual system from `feat/gui-demo`. **Day-3 live: `app/forecast/` page with ForecastChart + StationList + KpiStrip components; standalone landing HTML.** | `main` |
| **`liulian-mobile`** | TypeScript · Expo SDK 51 · React Native · Expo Router · Victory Native XL | Mobile companion. iOS + Android via single codebase. | placeholder; Day 4 |
| **`liulian-design-system`** | TypeScript + CSS + JSON | `@liulian/design-tokens` npm package. Source: `src/tokens.json`. Emits CSS / ESM / CJS / TS types / RN StyleSheet / Tailwind preset / antd ConfigProvider. **Kept separate** from web (multiple future consumers: mobile / marketing slides / Figma library / email templates). **Day-1 live: 65 tokens × 7 outputs.** | `main` |
| **`liulian-ops`** | Python CLI (`liulianctl`, fork of neoctl) + Helm + Terraform + reusable GH Actions + **`devcontainer/`** subfolder (was `liulian-dev-env`, merged in) | Operations + IaC + local dev environment. Owns `infra/{helm,terraform,grafana}/`, `.github/workflows/*` reusable workflows, and the Codespaces devcontainer config. | `main` |

### 4.2 Why multi-repo (audit-honest justification)

The audit (`AUDIT_REPORT_2026-05-12.md §B.1`) flagged the original
justification as preference-presented-as-evidence. Honest version:

**Defensible reasons** (survive research-critic):

| Reason | Why it holds |
|---|---|
| Operator muscle memory | User already runs liulian with multi-repo + `neoctl`; reusing that mental model is genuinely lower cognitive cost than learning Turborepo. |
| Smaller per-repo surface for OSS contributors | Model contributors to `liulian-python` shouldn't clone Helm charts; frontend contributors shouldn't need Python. Per-repo onboarding is genuinely easier. |
| Divergent release cadences | Mobile (EAS Build / store review) and web (Vercel push) and Python core (PyPI tag) ship on different rhythms. Multi-repo makes the cadences honest. |
| Cross-repo type sharing via published OpenAPI artefact | Versioned schema, codegen consumed by web + SDK + mobile. Cleaner than shared source files in a monorepo for our case (same idea as protobuf). |

**Claims we drop** (would not survive audit):

- *"CI is faster"* — unmeasured. Turborepo with caching would likely
  match per-repo CI for JS apps.
- *"Saves 2 days in the sprint"* — a priori guess.

**Reversibility clause**:
If cross-repo coordination becomes painful by M3 (e.g. opening 4 PRs to
ship one feature), we can collapse the three JS repos (`liulian-web` +
`liulian-mobile` + `liulian-design-system`) into a Turborepo monorepo
while keeping the five Python repos separate. One-week migration, not
one-month. Tracked in `adr/0001-multi-repo-split.md`.

### 4.3 What stays on this repo (`liulian-python`)

**ONLY**:

```
liulian-python/
├── liulian/                 # the Python package — pip-installable
│   ├── tasks/  data/  models/  adapters/  runtime/  optim/
│   └── viz/  plugins/  utils/  cli.py
├── manifests/               # canonical dataset manifests (data contracts)
├── experiments/             # research scripts — the moat
├── tests/                   # unit/integration tests
├── docs/                    # MkDocs technical site + strategy + adapter guide
│   └── strategy/            # this doc + L3 + L4 + references + zh + ADRs
├── healthcare-demo/         # ECG demo for ARTORG-AIHN application
├── refer_projects/          # external repos pinned for reference; gitignored mostly
├── jobs/                    # UBELIX SLURM scripts
├── .worktrees/gui-demo/     # the kept demo (orphan branch)
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

Everything else (`apps/api`, `apps/web`, `apps/mobile`, `infra/`,
`deployments/`) **migrates to its own repo** in the new architecture.

### 4.4 Cross-repo contracts

The contract surface between repos is intentionally narrow:

1. **`liulian-python` → `liulian-api`** — Python import (`pip install -e
   git+https://github.com/liulian-ai/liulian-python@vX.Y.Z`). API depends
   on tagged versions, not main.
2. **`liulian-api` → `liulian-web` / `liulian-mobile` / `liulian-sdk`** —
   OpenAPI schema published per release. Web + SDK codegen consumes a
   pinned schema URL.
3. **`liulian-api` ↔ `liulian-agent`** — HTTP between two FastAPI
   services. Agent has its own DB rows in shared Postgres (audit-only)
   but does not import API's models.
4. **`liulian-ingest` → MinIO + Postgres** — writes blobs + manifest rows;
   never invokes API; never imported by API.
5. **`liulian-design-system` → `liulian-web` / `liulian-mobile`** —
   npm package. Versioned. Breaking changes are major-version bumps.
6. **`liulian-ops` → everything** — orchestrates via SSH + kubectl +
   GitHub API. No code dependency on the others.

The contract diagram lives in `PLATFORM_DESIGN.md §1`.

### 4.5 Branch workflow on `liulian-python` (this repo)

1. All platform-related changes on `feat/platform-upgrade-2026-05`.
2. Periodically `git fetch origin main && git merge --no-ff main` to
   pull in concurrent main work (especially ARTORG-AIHN healthcare-demo
   scaffolding).
3. The `feat/gui-demo` orphan branch is *kept*; the new `liulian-web`
   repo imports its design tokens + screen archetypes but is not
   downstream of the branch.
4. Merge `feat/platform-upgrade-2026-05` → `main` at sprint end (Day 7),
   tagging `v0.6.0-portfolio`.

---

## 5. Backend (`liulian-api`)

(Stack details unchanged from iteration 1; recapitulated here briefly.)

### Stack

| Concern | Choice | Rationale |
|---|---|---|
| Framework | FastAPI 0.110+ | OpenAPI auto-gen → SDKs for free; async; Python culture continuity |
| Validation | Pydantic v2 | Settled standard; type-safe DTOs reused by SDK |
| ORM | SQLModel (Pydantic + SQLAlchemy) | One model class for DB + API |
| DB | **PostgreSQL + TimescaleDB extension** | See §5.1 below — load-bearing decision |
| Object storage | MinIO (dev) → S3 (prod) | Boring, portable |
| Cache / queue | Redis + arq | arq is async-native, fits FastAPI; we document Celery as the alternative for users who need it |
| Auth | Clerk (demo) → OAuth2 + per-tenant RBAC (v2) | Clerk = ship in hours; OAuth2 = audit-ready when first paying customer arrives |
| Background ML | Ray Serve for online inference; arq for batch | Ray is already pinned for HPO; one infra fewer |
| Observability | OpenTelemetry → Prometheus + Grafana + Tempo + Loki | Standard. Sentry on top |

### 5.1 Why TimescaleDB (operational, not slogan)

LIULIAN's `run_metric`, `forecast`, and `alert` tables are append-only
time-keyed streams that will reach millions of rows within M2. Two
candidate paths:

1. **Plain Postgres** + manual partitioning (`pg_partman` + `pg_cron`).
   Works. Requires us to write and maintain the partition schedule.
2. **TimescaleDB extension** (Postgres-compatible, Apache-2.0).
   Hypertables auto-partition. `CREATE EXTENSION timescaledb` is the
   only adoption cost; SQLModel works unchanged.

Choosing TimescaleDB for these three tables eliminates partition-management
code at zero query-language cost. That is the primary justification.

Secondary, narrative-only benefits:

- A TS product running on a TS-native primitive has internal
  consistency. Some engineer-reviewers in hiring loops appreciate the
  fit; others don't notice. Treat this as small upside, not the
  reason.
- Customers running plain Postgres can adopt TimescaleDB the same way
  we did. Same migration story for both sides.
- Eject path: if scale exceeds what TimescaleDB handles (rare on our
  trajectory), TDengine is the documented migration target
  (REFERENCE_DESIGNS §D1).

Sprint-time pragmatism: Day 1 of `liulian-api` uses **plain Postgres**;
TimescaleDB extension is enabled in a follow-up commit only after the
M1 demo is shipping. This removes one risk vector from the sprint
without losing the option.

### 5.2 API surface (v1)

(Identical to iteration 1; for brevity, see git blame on the previous
revision of this section. Key endpoints:)

```
/healthz /readyz
/experiments  (CRUD + run + abort)
/models       (list + capabilities + predict + predict-batch)
/datasets     (upload + preview + manifest)
/forecasts    (list + diff + intervals)
/alerts       (rules + history)
/agents/{name}/invoke + /runs
/reports
/tasks        ← NEW: see §8
```

The OpenAPI schema is the source of truth; SDK + web + mobile all consume
it via codegen.

### 5.3 Storage schema (key tables — revised)

The tracker tables now cover **three entity types** (§8). Schema diff
from iteration 1:

```sql
-- New: tasks are scheduled work items (cron, on-demand) wrapping experiments
CREATE TABLE task (
  id UUID PRIMARY KEY,
  tenant_id UUID NOT NULL,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,                     -- 'train' | 'predict' | 'retrain' | 'ingest' | 'alert-sweep'
  schedule TEXT,                          -- cron expression; null = on-demand
  config_yaml JSONB NOT NULL,
  status TEXT NOT NULL,                   -- 'pending' | 'queued' | 'running' | 'completed' | 'failed'
  last_run_id UUID,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Renamed from `run`: each task / experiment / agent invocation produces a run
CREATE TABLE run (
  id UUID PRIMARY KEY,
  parent_kind TEXT NOT NULL,              -- 'task' | 'experiment' | 'agent'
  parent_id UUID NOT NULL,
  status TEXT NOT NULL,
  started_at TIMESTAMPTZ,
  ended_at TIMESTAMPTZ,
  duration_ms INT GENERATED ALWAYS AS (EXTRACT(EPOCH FROM (ended_at-started_at))*1000) STORED,
  metrics_summary JSONB,
  metadata JSONB
);
CREATE INDEX ON run (parent_kind, parent_id, started_at DESC);

-- Hypertable: detailed metrics over time
CREATE TABLE run_metric (
  run_id UUID NOT NULL,
  step INT NOT NULL,
  name TEXT NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  recorded_at TIMESTAMPTZ DEFAULT now()
);
SELECT create_hypertable('run_metric', 'recorded_at');

-- Agent-specific run details
CREATE TABLE agent_run_step (
  run_id UUID REFERENCES run,
  step INT,
  role TEXT,                              -- 'planner' | 'tool' | 'reflection' | 'final'
  content TEXT,                           -- LLM message
  tool_name TEXT,
  tool_args JSONB,
  tool_result JSONB,
  tokens_in INT,
  tokens_out INT,
  cost_usd DOUBLE PRECISION
);

-- Existing experiment, artifact, forecast, alert tables unchanged from iter 1.
```

The `run` table is the *centre of the universe* — tasks, experiments,
and agents all reference it. The web `/studio` view renders the unified
"my recent activity" list from this one table.

---

## 6. Frontend (`liulian-web`) — see PLATFORM_DESIGN.md for depth

Stack summary; full brand + IA + panel-by-panel design is in L3 doc
**`PLATFORM_DESIGN.md`**.

| Concern | Choice | Rationale |
|---|---|---|
| Framework | Next.js 14 (App Router, RSC) | SSR for share-by-link pages; same stack as liulian frontend |
| Type-safe RPC | tRPC + openapi-typescript | tRPC inside the web app; OpenAPI codegen for FastAPI calls |
| Styling | Tailwind + CSS variables | Token-based |
| UI primitives | shadcn/ui (Radix under the hood) | Accessibility, copy-not-install |
| KPI / dashboard primitives | **Tremor** | Production polish for sparkbars, KPI cards |
| Time-series charts | **ECharts** | Big-data + brush + geo |
| Map | **MapLibre GL** + swisstopo tiles | Open-source, WebGL |
| Probabilistic plots | Plotly (rare) | Fan-chart cases ECharts is awkward |
| State | TanStack Query + Zustand | Server + UI split |
| Forms | react-hook-form + zod | zod mirrors Pydantic |
| Auth | Clerk Next.js SDK | Drop-in |
| i18n | next-intl (en · zh-CN · de-CH) | en first |
| Tests | Vitest + Playwright | |

**Visual substrate**: `feat/gui-demo` branch of *this* repo. Its design
tokens (UniBe red `#E20613` + Fraunces + Switzer + JetBrains Mono +
4-tab IA Data/Train/Inference/Insight) are the canonical starting point.
See `PLATFORM_DESIGN.md §2` for the full brand spec.

---

## 7. Mobile (`liulian-mobile`)

### 7.1 Mobile stack — full rationale

Mobile is a **separate repo** because:
- Release cadence: Expo OTA + EAS Build is unaligned with web's Vercel push.
- Bundle size: pulling the whole monorepo for a 50MB Expo project is wasteful.
- Toolchain: Expo's `eas` CLI is the operator UX; lives where the app lives.

**Stack** (with rationale per choice):

| Layer | Choice | Why this over alternatives |
|---|---|---|
| Framework | **React Native 0.74** + **Expo SDK 51** | RN = single JS/TS codebase compiles to native iOS + Android; Expo = managed workflow (no Xcode dance for most cases) + OTA updates + EAS Build for `.ipa`/`.apk` |
| Routing | **Expo Router** | File-system routing analogous to Next.js App Router; same mental model as `liulian-web` |
| Charts | **Victory Native XL** | Declarative API; works with Reanimated v3; alternative `react-native-skia` is faster but lower-level. XL = the new actively-maintained line |
| Animation | **React Native Reanimated 3** | UI-thread animations; no JS bridge stutter |
| Forms | **react-hook-form + zod** | Same as web — type schemas shared via `liulian-design-system` |
| Storage | **AsyncStorage** + **Expo SecureStore** | Plain + sensitive (auth tokens) |
| HTTP | **fetch** + tRPC client | tRPC client over HTTP; codegen types from `liulian-api` OpenAPI |
| Push | **Expo Notifications** | Cross-platform without Firebase setup hassles |
| File pick | **expo-document-picker** | ECG / CSV upload for ARTORG-AIHN healthcare branch |
| Auth | **Clerk Expo SDK** | Mirrors web |
| i18n | **expo-localization** + ICU MessageFormat | Same key set as `liulian-web` |
| Tests | **Jest + Detox** | unit + E2E |

**Why Expo (managed) over bare React Native**:

- *Single codebase* iOS + Android — costed-out alternative would be two
  swift/kotlin apps = 3× the work.
- *OTA updates* — push a bugfix without app-store review.
- *EAS Build* — builds happen in the cloud; no local Xcode required for
  Linux dev. Means we can build `.ipa` from a Linux laptop with one
  command.
- *Expo Go QR* — share the app with a recruiter / pilot user without
  TestFlight enrollment friction. Critical for the sprint.
- **Cost**: slightly larger bundle (~6 MB extra), fewer native modules
  out of the box, but you can "eject" to bare RN per-screen if needed.
  Net: never a problem at our scale.

**Distribution path**:

- *Sprint (demo)*: Expo Go QR — anyone with the Expo Go app scans, runs.
- *Internal (week 2+)*: EAS Build preview → downloadable `.apk` URL +
  TestFlight invite-only.
- *Production (M3+)*: EAS Submit → Play Store + App Store. Out of sprint
  scope.

### 7.2 Mobile UX (see PLATFORM_DESIGN.md §5 for full screens)

Three tabs: **Home** (alerts + recent runs) · **Forecast** (single-station
viewer with same fan chart logic as web) · **Alerts** (severity ribbon
+ acknowledge action). Plus a quick-action sheet for "run zero-shot
forecast".

---

## 8. Tracker — three entity types in one (`liulian-api` services/tracker.py)

The tracker covers **tasks** + **experiments** + **agents** under one
`run` table (see §5.3). UI surfaces a unified activity feed at
`/studio/activity` and per-entity drill-downs.

### 8.1 Entity model

```
        ┌──────────────┐
        │   Tenant     │
        └──────┬───────┘
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
┌──────┐  ┌──────────┐  ┌────────┐
│ Task │  │Experiment│  │ Agent  │       ← three "parents"
└──┬───┘  └────┬─────┘  └───┬────┘
   └───────────┴────────────┘
               │
               ▼
            ┌─────┐
            │ Run │  ← one row per execution; the activity-feed row
            └──┬──┘
               │
        ┌──────┴──────────┐
        ▼                 ▼
   ┌──────────┐      ┌────────────┐
   │ Metric   │      │ Artifact   │
   │ (hyper.) │      │  (forecast │
   └──────────┘      │   chart,   │
                     │   weights) │
                     └────────────┘
```

### 8.2 Why one table over three

If we built `experiment_run` + `task_run` + `agent_run` as separate
tables:

- Three separate "recent activity" queries → UI complexity multiplied.
- Cost / duration / quality cross-cuts impossible: "what's our most
  expensive run yesterday across all entity types?".
- Three places to instrument.

One `run` table + `parent_kind` discriminator is the canonical pattern
(see Airflow's `TaskInstance`, Temporal's `WorkflowExecution`).

### 8.3 What we do that MLflow doesn't

- Agents are first-class runs with token / cost telemetry per step.
- Forecast objects with intervals are first-class artefacts (MLflow
  stores them as opaque blobs).
- TimescaleDB-backed metrics → rolling-MAE-over-time queries native.
- Direct link from a run to the BI panel that *uses* its model.

### 8.4 MLflow-compatible REST shim

`liulian-api` exposes a subset of MLflow's tracking REST API
(`/api/2.0/mlflow/runs/create`, `/log-metric`, etc.) that translates to
the same `run` table. Users with `mlflow.set_tracking_uri(...)` clients
can keep using them — we wedge into the MLflow user base without forking
their UI.

---

## 9. Agent — custom-built (`liulian-agent`)

### 9.1 Why custom, not LangGraph

LangGraph (the iteration-1 choice) is an excellent library; however,
for our needs it adds:

- *Dependency mass* — LangChain core deps + LangGraph deps = 30+
  transitive packages, many with sharp release cadences.
- *Abstraction tax* — to debug a misbehaving graph you have to learn
  their abstractions; for the 5-tool surface we have, this is overkill.
- *Pinning risk* — both libraries refactor fast; our long-tail
  maintenance cost grows.

A custom orchestrator at ~300 LOC, using **only** `pydantic`, `httpx`,
and `asyncio`, is the right size. We model liulian's `~/liulian/agent`
which is a FastAPI service that exposes `/health` + agent endpoints.

### 9.2 Provider layer — the three you have tokens for

| Provider | Default-for | Cost (May 2026, per 1M tokens) | Special features |
|---|---|---|---|
| **DeepSeek V4** Flash | default for all calls | $0.14 in / $0.28 out · 1M context | Cheapest production-grade; strong reasoning |
| **DeepSeek V4 Pro** | hard analytical questions | $0.435 in / $0.87 out (promo) | Reasoning-heavy |
| **GLM-4.6** (智谱) | Chinese-language tasks | ~$0.20 in / $0.60 out (typical) | Best zh tokenisation; CN-data residency |
| **Gemini 3.1 Pro** | long context (>200k) + multimodal | $2 in / $12 out (≤200k) | 1M+ context; vision for sensor imagery |
| **Gemini 3.1 Flash-Lite** | tool-call routing | $0.10 in / $0.40 out | Cheap router |
| **Ollama + qwen2.5-7b** (local) | sovereign / offline | $0 + hardware | Privacy-mandatory deployments |

Provider abstraction: a small `Provider` interface (~80 LOC):

```python
class Provider(Protocol):
    async def complete(
        self, messages: list[Message], tools: list[Tool], **opts
    ) -> Completion: ...
```

Implementations: `DeepSeekProvider`, `GLMProvider`, `GeminiProvider`,
`OllamaProvider`. Each handles its provider's quirks (Gemini uses
`/v1beta/models/.../generateContent`; DeepSeek and GLM are OpenAI-
compatible).

We optionally compose via **LiteLLM** as a *fallback* path (one binary
with all providers); the bespoke layer is the default because it lets us
log token cost per provider into our `agent_run_step` table cleanly.

### 9.3 Orchestrator — hand-written, ~300 LOC

A deterministic state machine (`PLAN → CALL_TOOL → REFLECT → ...`) with
explicit retries, max-step caps, cost ceilings, and prompt caching. Each
state writes one row to `agent_run_step` (see §5.3). The full file is
designed to fit on a screen and a half — easy to reason about, easy to
modify.

### 9.4 Three agent personas

| Agent | Surface (tools, Pydantic-typed) | Triggered from |
|---|---|---|
| **data** | `list_files(s3_prefix)`, `summarise_csv(uri)`, `propose_manifest(uri)`, `validate_manifest(yaml)`, `detect_topology(manifest_id)`, `detect_seasonality(series_id)` | Studio / Data tab |
| **model** | `list_models()`, `recommend_model(dataset_id)`, `propose_hpo_space(model)`, `read_run_logs(run_id)`, `diagnose_failed_run(run_id)`, `compare_runs(ids)` | Studio / Models tab |
| **bi**    | `query_forecasts(filter)`, `add_panel(report_id, spec)`, `set_filter(report_id, filter)`, `create_alert_rule(spec)`, `export_report(id)` | Forecast canvas (chat sidebar) |

Each tool is a Pydantic-input function in `liulian-agent/tools/{data,model,bi}.py`.
The orchestrator JSON-schemas them and feeds the JSON-schema into the LLM
provider's function-calling slot (works on DeepSeek, GLM, Gemini, OpenAI,
and Anthropic equally).

### 9.5 Safety + privacy in the agent

- **Default provider = local Ollama** when `LIULIAN_OFFLINE=1`.
- **Never send raw signal arrays to a cloud LLM** — tools that need data
  return *summary statistics* to the LLM, not the data itself.
- **PII redaction** — pre-LLM scrubbing of tenant-identifying fields
  (see `liulian_agent/redact.py`).
- **Cost ceiling per run** — `max_usd_per_run` (default $0.50) enforced
  by the orchestrator; over-budget runs short-circuit with a
  `cost-exceeded` reason.
- **Audit log** — every step persists; tenant-admin can replay.

### 9.6 Reference patterns we use

- liulian `~/liulian/agent` deploys as a Python+uv FastAPI service on
  port 8000 with `/health` — we mirror this **exactly** so `liulianctl
  deploy agent` matches the muscle memory.
- `dsa` / `TradingAgents-CN` / `Vibe-Trading` (CN financial-agent OSS) —
  patterns for *registry of skills*, *DAG of tool calls*. We borrow the
  registry-of-skills idea (one decorator per tool, auto-registered) and
  ignore the multi-agent swarm complexity (not needed at our scale).

---

## 10. Data ingest (`liulian-ingest`)

The "crawler" equivalent to liulian's data layer. Scheduled fetchers
write to MinIO + manifests. Stack:

- Python 3.10+ · `httpx` async · `playwright-python` (when JS rendering
  is required) · `pydantic` for output schemas · `apscheduler` (cron-in-
  process) or external cron via `liulian-ops`.
- Source adapters: `swisstopo-bafu` (Swiss hydrology), `swissgrid`
  (energy), `meteoswiss` (weather), `physionet` (healthcare), `noaa` (US
  weather), `caltrans-pems` (US traffic).
- Output: parquet to MinIO under `s3://liulian-raw/{source}/{date}/...`
  and manifest YAML to `manifests/` automatically PR-ed to
  `liulian-python`.
- Idempotent: re-running fills in only missing windows.
- Deployed as a cron-driven Kubernetes Job (production) or systemd timer
  (single-VM staging).

---

## 11. Brand and visual identity (L3 — full detail in PLATFORM_DESIGN.md)

### 11.1 Anchor — UniBe red

The brand is anchored on **University of Bern red**: `#E20613` (the only
hex you must memorise). The `feat/gui-demo` design proved this works
beautifully on a warm-bone canvas with editorial typography.

### 11.2 Full token set (excerpt — full set in `liulian-design-system`)

```css
/* Canvas & ink */
--canvas-warm    : #FBFBFA;            /* warm paper, never #fff */
--surface-pure   : #FFFFFF;            /* card surface */
--ink-charcoal   : #131313;            /* body, never #000 */
--ink-muted      : #5C6066;            /* secondary text */
--ink-faint      : #8E9296;            /* tertiary */
--hairline       : #EAEAEA;            /* 1px borders */

/* UniBe red — the brand */
--unibe-red      : #E20613;            /* the only spot color */
--unibe-red-tint : #FDEBEC;            /* pale pastel for pill backgrounds */
--unibe-red-deep : #B00510;            /* deep variant for hover/focus */

/* Status pastels (rare) */
--pastel-green   : #EDF3EC;  /* on text: #346538 — OK / healthy */
--pastel-blue    : #E1F3FE;  /* on text: #1F6C9F — informational */
--pastel-yellow  : #FBF3DB;  /* on text: #956400 — warning */

/* Optional secondary accents (very sparingly) */
--unibe-ocean    : ~#0066B3  /* from unibeCols R package; used in chart secondary series only */
--unibe-green    : ~#509A39  /* same; e.g. ground-truth vs predicted contrast */
--unibe-apricot  : ~#E6863A  /* same; tertiary chart series */
```

`--unibe-red` is the brand anchor and is used **rarely and
deliberately**: the wordmark, the active station marker, the predicted-
forecast line, the elevated-alert pill, the threshold-crossing marker.
Not for buttons, not for backgrounds, not for hover states.

### 11.3 Typography

- **Display**: `Fraunces` (variable, op-size 9–144, SOFT/WONK axes
  available). Tightened tracking on display sizes. Optional WONK 1 on a
  single accent character (e.g. red italic *U* in LIULIAN) — proven in
  gui-demo.
- **Body / UI**: `Switzer` (Fontshare). Geometric Swiss grotesque.
- **Mono**: `JetBrains Mono`. `font-variant-numeric: tabular-nums` on
  any numeric column.
- **Banned**: Inter, Roboto, Open Sans (per minimalist-ui skill).

### 11.4 What we *avoid* (anti-references from gui-demo iteration 1)

- AI-default dark theme + cyan + violet + amber + grid lines.
- Glow filters (`feGaussianBlur` drop shadows on active cards).
- Topographic-contour repeating gradients.
- Generic SaaS hero copy ("Where intelligence converges...").
- Heavy drop shadows (`shadow-md`/`-lg`/`-xl`).
- Side-stripe accents.
- Gradient text.
- `rounded-full` on cards.
- `#000` / `#fff`.

Full design depth: **PLATFORM_DESIGN.md**.

---

## 12. CI/CD and Ops (`liulian-ops`)

### 12.1 Per-repo CI

Each of the seven repos has its own `.github/workflows/ci.yml`. Standard
pattern (Python repos):

```yaml
# in liulian-python, liulian-api, liulian-agent, liulian-ingest
name: ci
on: [push, pull_request]
jobs:
  lint:
    uses: liulian-ai/liulian-ops/.github/workflows/python-lint.yml@v1
  type:
    uses: liulian-ai/liulian-ops/.github/workflows/python-mypy.yml@v1
  test:
    uses: liulian-ai/liulian-ops/.github/workflows/python-pytest.yml@v1
  build-image:
    uses: liulian-ai/liulian-ops/.github/workflows/python-image.yml@v1
```

(JS repos use a parallel set: `js-lint.yml`, `js-tsc.yml`, `js-vitest.yml`,
`js-playwright.yml`, `js-image.yml`.)

Reusable workflows live in **`liulian-ops/.github/workflows/`** — one
place to update CI behaviour across all repos.

### 12.2 The `liulianctl` CLI

Mirrored from `neoctl`. Implemented as a Python `click` CLI inside
`liulian-ops`. Commands:

```bash
liulianctl bootstrap                      # provision a fresh env (clone all repos, .env, etc.)
liulianctl deploy api                     # rolling deploy of liulian-api
liulianctl deploy agent                   # rolling deploy of liulian-agent
liulianctl deploy web                     # vercel deploy of liulian-web
liulianctl deploy mobile                  # eas build + publish for liulian-mobile
liulianctl deploy all                     # all of the above, ordered
liulianctl logs <service> [--follow]      # SSH + kubectl logs
liulianctl restart <service>              # rolling restart
liulianctl manifest sync                  # liulian-ingest → liulian-python manifests/ PR
liulianctl tunnel ollama                  # autossh forward-tunnel to GPU LLM (per liulian doc)
```

Under the hood: SSH + `kubectl` + `helm upgrade` + GitHub API + Vercel
API + EAS CLI. We document the *manual* fallback for each command —
mirrored verbatim from liulian's `manual-deployment.md`.

### 12.3 Where infra/deployment files live (revised)

| Concern | Lives in | Why |
|---|---|---|
| Dockerfile (per service) | each service's repo | image rebuilds when service rebuilds |
| docker-compose.dev.yml (multi-service local dev) | `liulian-ops/compose/` | crosses repos by definition |
| Helm chart (multi-service deploy unit) | `liulian-ops/helm/liulian-platform/` | one chart, sub-charts per service |
| Terraform modules | `liulian-ops/terraform/{aws-eks,hetzner-k3s}/` | one place |
| Grafana dashboards (JSON) | `liulian-ops/grafana/dashboards/` | shipped with deploy |
| Reusable GH Actions workflows | `liulian-ops/.github/workflows/*.yml` | called by per-service repos |
| Per-environment values | `liulian-ops/helm/values/{dev,staging,prod}.yaml` | one place |
| Cluster bootstrap docs | `liulian-ops/docs/` | runbooks |
| Per-service Helm values overrides | each service's `helm-values.yaml` | service-local context |

This makes `liulian-ops` a single, focused repo for everything that
*crosses* the federation. Each service stays simple.

### 12.4 Environments

| Env | Stack | Audience | Cost |
|---|---|---|---|
| `dev` | docker-compose on laptop | engineer | $0 |
| `demo` | Railway (api) + Vercel (web) + Expo Go (mobile) + TimescaleDB cloud free | recruiter / VC | $5/mo |
| `staging` | Hetzner CX31 VPS + k3s | internal | $20/mo |
| `prod` | EKS (AWS) or AKS — customer-paid | customer | usage |

### 12.5 Observability (unchanged from iter 1)

OTel → Prometheus + Grafana + Tempo + Loki. Pre-built dashboards under
`liulian-ops/grafana/dashboards/`:

1. API health
2. Inference perf per model
3. Training jobs queue
4. Forecast quality (rolling MAE / CRPS)
5. Agent cost (tokens × $ per provider per tenant)

---

## 13. Privacy and security

Unchanged from iteration 1 (§13 below).

- **Data residency**: single-tenant on customer-owned infra by default.
- **PII**: zero; demos use public PhysioNet MIT-BIH and swisstopo open
  hydrology.
- **Local-LLM path**: documented (Ollama + qwen / DeepSeek-r1-local).
  Provider abstraction makes the swap one env-var.
- **Secrets**: 1Password Connect / AWS SSM; never `.env` committed.
- **Audit**: every write API call emits an audit-log row + `agent_run_step`
  for agent calls; retained 365 days.
- **Compliance posture**: not certified; documented path to GDPR / Swiss
  FADP / HIPAA.
- **SBOM**: `cyclonedx-bom` produced in CI.

---

## 14. Documentation (L3-ish; full in PLATFORM_DESIGN.md §7)

Tiered:

1. **Marketing landing** (`apps/docs-site/`) — single-page, embedded
   demo video, three CTAs.
2. **MkDocs Material technical site** under `/docs/` of each Python
   repo — architecture, adapter contract, manifest spec, plugin
   authoring, deployment runbooks.
3. **OpenAPI / Swagger** at `/api/docs` on `liulian-api`.
4. **Examples gallery** under `experiments/` and `examples/` in
   `liulian-python`.
5. **Strategy docs** under `docs/strategy/` of `liulian-python` — this
   doc + L3 + L4 + ADRs + references + zh mirrors.

Each guide opens with a working snippet; concepts second.

**Bilingual rule**: per user CLAUDE.md, in-repo files default to English
(code-language). However, *strategy docs* (consumed by stakeholders as
well as engineers) ship in both English (canonical) and Chinese (mirror):

```
docs/strategy/
├── PLATFORM_BLUEPRINT.md        ← canonical EN
├── PLATFORM_BLUEPRINT.zh.md     ← Chinese mirror
├── PLATFORM_DESIGN.md           ← EN
├── PLATFORM_DESIGN.zh.md        ← zh (sprint scope: write later if time)
├── ONE_WEEK_SPRINT.md           ← EN
├── ONE_WEEK_SPRINT.zh.md        ← zh
├── REFERENCE_DESIGNS.md         ← EN only (research notes)
└── adr/                         ← per-decision; EN by default
    ├── 0001-multi-repo-split.md
    ├── 0002-custom-agent-not-langgraph.md
    ├── 0003-timescaledb-not-tdengine-now.md
    ├── 0004-unibe-red-as-anchor.md
    └── 0005-tracker-three-entity-unified-table.md
```

---

## 15. Roadmap (three-track timeline, sprint-extended)

The original M1–M6 table assumed full-time effort. Linlin's actual
constraint is ~20 hours/week alongside the postdoc role. Research
deadlines (ICPR camera-ready, SNSF / ERC cycles) will collide.
Therefore three timelines are tracked in parallel; M1 is fixed for
all three because the ARTORG application is an external deadline.

| Milestone | Aggressive (zero slip) | Realistic (1 slip) | Conservative (multiple slips) | Defining deliverable |
|---|---|---|---|---|
| **M1: Portfolio-ready** | 2026-05-19 | 2026-05-19 | 2026-05-19 | Live demo URLs + 7 repos initialised + ARTORG submission |
| **M2: BI flagship** | 2026-06-30 | 2026-08-01 | 2026-09 | 8-panel SwissRiver canvas + Chronos zero-shot + agent v1 |
| **M3: Multi-tenant cloud** | 2026-07 | 2026-09 | 2026-10 | Helm + Terraform on EKS, Clerk auth, status page |
| **M4: Vertical pilots** | 2026-08 | 2026-10 | 2026-12 | Energy demand + healthcare ECG case studies |
| **M5: Agent autopilot** | 2026-09 | 2026-12 | 2027-Q1 | Nightly anomaly sweep, retrain decisions, drift detection |
| **M6: Pre-seed** | 2026-11 | 2027-Q1 | 2027-Q2 | Pitch deck + financial model + first LOI |

Plan-of-record: **Realistic**. We communicate Realistic timing to
external stakeholders; we work toward Aggressive; we hold
Conservative as the floor.

Each milestone has one **visible artefact** (URL, deck, chart) legible
without reading code. The artefact, not the date, is the deliverable.

---

## 16. Long-term extensibility

Adding a new vertical is configuration, not code:

1. **Dataset** — drop a manifest under `manifests/{vertical}/` in
   `liulian-python` (or auto-generated by `liulian-ingest`).
2. **Plugin** — optionally add `plugins/{vertical}/` for custom
   adapters, feature engineering, domain ontology.
3. **BI** — register a `vertical.json` describing default panels, map
   projection, station-icon set, thresholds. Loaded by `liulian-web` at
   tenant init.
4. **Agent** — domain-specific tools as Pydantic-typed callables;
   registry pattern picks them up automatically.
5. **Deployment** — Helm values overlay (`values/{vertical}.yaml`)
   controls flags, default models, alert channels.

A vertical that requires touching `liulian-python` goes through a
core-extension ADR.

---

## 17. Funding readiness — what investors need to see

(Unchanged from iter 1.)

Concrete artefacts staged by milestone:

- **M2**: 2-minute demo video; 3+ non-friend GitHub stars; first blog
  post by a non-team user.
- **M3**: a paying pilot (CHF 1k–5k / mo). Targets in order: swisstopo /
  WSL / Eawag for water; Inselspital / ARTORG-AIHN for healthcare;
  Axpo / BKW for energy.
- **M4**: ≥ 25 monthly active users.
- **M5**: technical advisor on cap table (Mougiakakou / Fischer / a
  CH-based fintech-ML founder).
- **M6**: 12-slide deck + 18-month financial model + first LOI.

Narrative: **vertical TS/ST AI for regulated sectors** — angle is
*spatio-temporal native* + *sovereign data residency*.

---

## 18. ADRs (Architecture Decision Records)

The blueprint is opinionated; the ADRs are *what we'd change our mind
about*. Each ADR is one file in `docs/strategy/adr/` with:

- Title
- Status: proposed / accepted / superseded
- Context
- Decision
- Consequences
- Date

Seeded ADRs (written as part of iteration 2):

1. `0001-multi-repo-split.md` — adopt federated repos over monorepo.
2. `0002-custom-agent-not-langgraph.md` — build a 300-LOC agent over
   importing LangGraph.
3. `0003-timescaledb-not-tdengine-now.md` — postgres-flavoured TS over
   AGPL TS-native; revisit at M3.
4. `0004-unibe-red-as-anchor.md` — UniBe red `#E20613` is the spot
   colour; one accent only.
5. `0005-tracker-three-entity-unified-table.md` — one `run` table with
   `parent_kind` discriminator, not three tables.

When a future iteration changes a decision, append a new ADR with
`status: supersedes #N`.

---

## 19. Decisions still open

Resolved in iteration 2 (no longer open):

- ~~Streamlit vs Next.js for the sprint~~ → Next.js from Day 3.
- ~~Helm + Terraform in sprint or M2?~~ → scaffolded in sprint Day 7,
  polished in M2.
- ~~Tracker: build vs MLflow?~~ → custom impl + MLflow-compatible REST.

Still open:

- BentoML vs Ray Serve vs plain FastAPI for model serving — plain
  FastAPI through M2; revisit at M3 when first paying customer's
  perf requirements appear.
- pgvector vs Qdrant for the agent's long-term memory — pgvector for M2
  simplicity (one DB); revisit at M5 when ≥ 100k embeddings.
- iOS-only vs Android-also screenshots for ARTORG application — pending
  Mac availability; sprint will ship Android-first with Expo Go.

---

*See `PLATFORM_DESIGN.md` for the brand, BI, agent, and mobile design
depth. See `ONE_WEEK_SPRINT.md` for the 7-day execution. See
`REFERENCE_DESIGNS.md` for the 12-platform audit notes. See
`docs/strategy/adr/` for individual decision records.*
