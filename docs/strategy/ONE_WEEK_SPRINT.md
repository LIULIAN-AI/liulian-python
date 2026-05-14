---
title: LIULIAN One-Week Sprint Plan
status: time-boxed (2026-05-13 → 2026-05-19)
owner: Linlin Jia (jajupmochi)
depends_on: PLATFORM_BLUEPRINT.md
companion: REFERENCE_DESIGNS.md
goal: end Day 7 with a demo-able platform that satisfies both an ARTORG-AIHN application AND ML-platform-engineer-tier signals
length: long; daily sections are self-contained
---

# LIULIAN — One-Week Sprint (2026-05-13 → 2026-05-19)

> Bootstrap the [PLATFORM_BLUEPRINT.md](PLATFORM_BLUEPRINT.md) into a live,
> portfolio-strong, recruiter-clickable demo in 7 days of focused work.

## 0. Why this sprint exists

Two simultaneous deliverables converge on Day 7:

1. **ARTORG-AIHN application** (deadline 2026-05-19) — needs a mobile app
   + backend + healthcare-flavoured demo + privacy posture.
2. **General ML/Platform Engineer pitch** — needs K8s/Helm scaffold,
   FastAPI gateway, Grafana dashboard, BI canvas, agent integration.

The two sets overlap by ~80%. The plan below builds the shared 80% and
splits only on Day 6 (deploy + docs) to make sure both audiences are
served.

## 1. Pre-sprint pre-flight (do *before* Day 1)

- [ ] Verify Python 3.10 + `uv` installed; `uv pip install -e ".[dev,torch]"`
- [ ] Verify Node 20 + `pnpm` 9 installed
- [ ] Verify Docker 24+ with compose v2
- [ ] Verify `gh` CLI authed; `vercel` CLI installed; `railway` CLI installed
- [ ] Decide deployment target — Railway (recommended) or Fly.io for the
      backend; Vercel for the web; Expo Go for mobile
- [ ] Reserve domain `liulian.app` (or `liulian.dev`); turn on DNSSEC
- [ ] Create empty `feat/one-week-sprint` branch from `main`

## 2. Daily plan

Each day has a single **flagship deliverable** (visible to a recruiter) and a
**scaffold deliverable** (infrastructure not yet visible but required for the
next day). End-of-day verification is a curl / browser / phone action,
not "tests pass".

---

### Day 1 — Tue 2026-05-13 — *fork-and-measure spike + scaffold + identity*

> **Audit-driven revision** (per `AUDIT_REPORT_2026-05-12.md` and the
> iteration-3 multi-repo decision in `ADR 0001`): the iteration-1 plan
> here was based on a monorepo; we now fork from liulian-ai per
> `LIULIAN_REUSE_MAP.md §1`. Plain Postgres is the Day-1 default;
> TimescaleDB extension is enabled only after M1 demo is shipping
> (`ADR 0003`).

**Flagship**: a screenshotable marketing landing page at `localhost:3000`
with hero, three editorial bands, and a single text-arrow CTA — built
with LIULIAN brand tokens.
**Scaffold**: 8 repos initialised and forked per
`LIULIAN_REUSE_MAP.md §1`; plain Postgres up via docker-compose;
FastAPI `/healthz` returns 200; `@liulian/design-tokens` published to
private npm and consumed by web.

#### Tasks (priority order)

**0. Fork-and-measure spike (FIRST, 90 min cap)** — replaces the
"guess" rows in `LIULIAN_REUSE_MAP.md §14.7` with measured numbers.

   For each forkable repo, in order: agent → crawler → neoctl →
   frontend → dev-env:
   - `gh repo fork liulian-ai/liulian-<x> --clone --org=jajupmochi`
   - rename remotely (`gh repo rename liulian-<x>`) and locally
   - strip bank-domain code: delete `bank_*` modules, bank prompts,
     bank fixtures, bank scenario JSON
   - `cloc .` → count LOC remaining
   - update the `Estimated reuse fraction` section of the reuse map
     with the measured number (replaces the TBD range)
   - commit rename + strip in one PR per repo

   If the spike runs long, ship what's measured and finish on Day 2
   morning. Do not let it block the rest of Day 1.

1. **Workspace scaffolding (in `liulian-python` only)** (60 min)
   - Move existing `liulian/` package contents stay as-is (no
     refactor to `packages/liulian-core/` — the iter-3 multi-repo
     decision means the package itself stays at `liulian/`; only
     metadata + boundary changes).
   - Run `pytest` to confirm zero regressions on `liulian-python`.
   - This repo only carries `docs/strategy/` + minor metadata changes
     during the sprint; everything else moves to its own repo.

2. **Design tokens** (1.5h) — `packages/design-tokens/`:
   - One source file `tokens.json` with the OKLCH palette + spacing + radii +
     typography stack from PLATFORM_BLUEPRINT.md §6.
   - Build script generates: `tokens.css` (CSS vars), `tailwind.preset.js`,
     `tokens.ts` (TS const), `tokens.rn.ts` (React Native StyleSheet).
   - Import the preset in `apps/web/tailwind.config.ts`.

3. **FastAPI bootstrap** (1.5h) — `apps/api/`:
   - `main.py` with `/healthz` and `/readyz`.
   - `pyproject.toml` declares dependency on `liulian-core` via path.
   - `uvicorn liulian_api.main:app --reload` → `curl localhost:8000/healthz`
     → `{"status":"ok","liulian_core":"0.0.1"}`.

4. **Next.js landing** (2h) — `apps/web/(marketing)/page.tsx`:
   - Fraunces + Switzer fonts loaded via `next/font/google`.
   - Hero: "Liquid Intelligence for spatio-temporal forecasting" (Fraunces
     56px, tracking -0.025em).
   - Three feature blocks (model zoo · BI canvas · agent layer) — Tremor cards.
   - Single CTA: "Open SwissRiver demo →" (links to `/forecast`; routes
     to placeholder this day).
   - No animation yet (Day 5 polish).

5. **Mobile bootstrap** (1h):
   - `pnpm create expo-app apps/mobile --template tabs-typescript`.
   - Replace the default screens with Home / Forecast / Alerts stubs.
   - Import design tokens; confirm Fraunces and Switzer render via
     `expo-font`.

6. **Docs-site bootstrap** (0.5h):
   - `mkdocs new apps/docs-site` (or convert existing `docs/`).
   - Use Material theme; brand the header with token CSS vars.

7. **Commit** (single commit per task; the day ends with ~6 commits).

#### Verification

- `curl localhost:8000/healthz` → 200.
- `pnpm --filter web dev` → landing renders with brand fonts.
- `pnpm --filter mobile start` → Expo Go QR code shows Home tab.
- `mkdocs serve` in `apps/docs-site/` → branded docs.
- Existing tests still pass: `pytest -m "not slow and not download"`.

---

### Day 2 — Wed 2026-05-14 — *backend depth*

**Flagship**: working Swagger UI at `/api/docs` with /experiments,
/forecasts, /models — recruiter can hit "Try it out" on `/models` and
get a list.
**Scaffold**: SQLModel schema + Alembic migrations; SDK package emits
typed methods from OpenAPI.

#### Tasks

1. **DB + ORM** (2h):
   - `docker-compose up -d postgres redis minio` — local services.
   - Use **TimescaleDB-flavoured Postgres image** (`timescale/timescaledb-ha:pg16`).
   - SQLModel tables per §5 of the blueprint (experiment / run / run_metric /
     artifact / forecast / model_card / dataset / alert_rule / alert).
   - Alembic init + first migration.

2. **Service layer + endpoints** (3h):
   - `services/experiments.py` — create / list / get / run; wraps
     `liulian.runtime.experiment.Experiment` via the in-process Python API.
   - `services/models.py` — registry that scans `liulian.models.torch.*`
     for adapters; returns capability vectors per model.
   - `services/forecasts.py` — produce / list / get / diff.
   - `services/datasets.py` — upload manifest + parquet; preview.
   - Endpoints wire to services (FastAPI dependency injection).

3. **OpenAPI codegen** (1h):
   - Dump OpenAPI to `apps/api/openapi.json` via FastAPI's CLI export.
   - In `apps/web/`, run `openapi-typescript ../api/openapi.json -o src/api/types.ts`.
   - Set up a `pnpm gen:api` script that re-runs codegen.

4. **Python SDK skeleton** (1h) — `packages/liulian-sdk/`:
   - Generated client from `openapi-python-client`.
   - Convenience wrappers in `liulian_sdk/__init__.py`:
     `Client(base_url).experiments.create_from_yaml(path)`.

5. **Auth stub** (0.5h):
   - Add `Authorization: Bearer <demo-token>` requirement on write endpoints.
   - Hardcode the demo token in env var for the sprint; defer Clerk to M3.

6. **Tests** (1h):
   - One happy-path pytest per endpoint using `httpx.AsyncClient`.
   - Fixture spins up an SQLite-backed app for fast CI.

#### Verification

- Swagger UI at `localhost:8000/docs` — `GET /models` returns 30+ entries.
- `python -c "from liulian_sdk import Client; print(Client('http://localhost:8000').models.list())"` → list.

---

### Day 3 — Thu 2026-05-15 — *BI canvas v0 + first real chart*

**Flagship**: `/forecast` page shows a real ECharts time-series rendered
from a real SwissRiver forecast loaded from the backend.
**Scaffold**: tRPC layer; ECharts wrapper component; map placeholder; data
fetching pattern.

This is the **make-or-break day** — by Thu evening you must have a
chart, not a wireframe.

#### Tasks

1. **Seed real data** (1h):
   - Wire the existing `experiments/swiss_river/` runner to produce one
     forecast against an existing checkpoint.
   - `make seed` → API has 1 experiment, 1 run, 5 forecasts for 5
     stations.

2. **tRPC layer** (1h) — `apps/web/src/server/trpc/`:
   - Routers: `experiments`, `forecasts`, `models`, `agents`.
   - Each router proxies to the FastAPI client (server-side fetch).

3. **Canvas shell** (1h):
   - Route `/forecast` with the four-quadrant layout from PLATFORM_BLUEPRINT
     §6 (stations sidebar · map · timeseries · distribution).
   - shadcn/ui `<Card>` containers with 1px hairline border.
   - "Add panel" button stubbed.

4. **Canonical time-series chart** (3h):
   - `<ForecastChart />` component wrapping ECharts.
   - Props: `observation: TimeSeries`, `forecast: Forecast`, `intervals:
     Quantile[]`.
   - Layers: observation (solid `--ink-charcoal`), forecast mean (dashed
     `--river-blue`), Q05–Q95 fan (semi-transparent `--river-blue` α=0.18),
     alert markers (rust dots).
   - Brush-zoom enabled (`dataZoom: [{ type: 'inside' }, { type: 'slider' }]`).
   - Theme aligned to OKLCH tokens.

5. **Map placeholder with real stations** (1h):
   - MapLibre GL with swisstopo lightbase tile.
   - 28 stations as circle markers, click → cross-filter the time-series.
   - No topology overlay yet (Day 5).

6. **Loading + error states** (0.5h):
   - shadcn `<Skeleton>`s; error toast on fetch fail.

#### Verification

- Open `localhost:3000/forecast` → see a chart with real fan from the
  forecast database; click a map marker → chart updates.

---

### Day 4 — Fri 2026-05-16 — *mobile + BI deepening*

**Flagship-mobile**: Expo Go QR runs the app on iOS Simulator + Android
emulator + physical phone; the Forecast tab shows the same chart logic as
web.
**Flagship-web**: add panels — distribution (residual histogram),
multi-model overlay, KPI strip.
**Scaffold**: shared chart logic in `packages/liulian-charts/` consumed by
both web and mobile.

#### Tasks (mobile half ~ 4h, web half ~ 4h)

1. **`packages/liulian-charts/`** (1.5h):
   - Pure functions: `computeQuantileFan`, `alignSeries`, `crpsCoverage`.
   - Web and mobile both depend on it — keep visual rendering separate.

2. **Mobile screens** (2.5h):
   - Home: greeting + alert summary + "Run quick forecast" CTA.
   - Forecast: station picker (segmented control) + Victory Native chart
     (same fan logic).
   - Alerts: list with severity ribbon.
   - Backend URL via `EXPO_PUBLIC_API_URL`.
   - Test on iOS Simulator (if Mac) + Android emulator (via Android Studio).

3. **Web — distribution panel** (1.5h):
   - `<ResidualHistogram />` — ECharts custom-series + density curve.

4. **Web — multi-model overlay** (1h):
   - Top-bar model multiselect; chart fans both / three models with
     hue offsets.

5. **Web — KPI strip** (1h):
   - Tremor `<Metric />` cards: MAE / RMSE / CRPS / Coverage@90 per
     selected model.

#### Verification

- Scan Expo Go QR on your phone → app runs, forecast chart loads.
- Web `/forecast` has four working panels.

---

### Day 5 — Sat 2026-05-17 — *agents + ML integrations + map topology*

**Flagship**: BI agent in a chat sidebar that can say "Show me stations
where Q95 exceeded threshold last week" and the canvas reacts. Plus
Chronos-2 zero-shot forecast as a button.
**Scaffold**: LangGraph node graph; LiteLLM proxy; topology overlay on the
map.

#### Tasks

1. **Chronos-2 adapter** (1.5h):
   - `liulian/adapters/chronos/adapter.py` wrapping `Chronos2Pipeline`.
   - Capabilities `["zero_shot", "probabilistic", "univariate", "multivariate"]`.
   - `GET /models` now lists Chronos; `POST /models/chronos-2/predict`
     works without training.
   - Add a UI button "Zero-shot forecast (Chronos-2)" on the canvas — one
     click swaps the chart's forecast with Chronos's output. Demo-gold.

2. **TSL adapter promotion** (1h):
   - Promote `experiments/adapt_tsl_lib/` to `packages/liulian-core/adapters/tsl/`.
   - Expose at least one TSL model (e.g. `STConvNet`) via `/models`.

3. **Map topology overlay** (1h):
   - Manifest entry for SwissRiver river-network edges (upstream IDs).
   - SVG overlay on MapLibre with curve edges sized by mean discharge,
     colored by current residual.

4. **Agent layer v0** (3h):
   - `packages/liulian-agents/` LangGraph definitions:
     - **BI agent**: tools `query_forecasts`, `add_panel`, `set_filter`.
     - **Data agent**: tools `summarize_dataset`, `propose_manifest`.
     - **Model agent**: tools `recommend_model`, `compare_runs`.
   - LiteLLM proxy in `apps/api/` so the agent can call Anthropic /
     OpenAI / local Ollama via one interface.
   - FastAPI endpoint `POST /agents/{name}/invoke` streams the agent's
     response.
   - Web sidebar `<ChatPanel />` in `/forecast` — uses Vercel AI SDK
     `useChat()` with the stream.

5. **One canned interaction** (0.5h):
   - Prompt: "Add an alert for stations where Q95 > 850 m³/s next week"
     → agent calls `create_alert_rule` → BI updates with new ribbon row.

#### Verification

- Chat sidebar → "show me Bern station" → time-series filters to Bern.
- Click "Zero-shot Chronos-2" → chart fan updates within ~2 seconds.
- Map shows topology graph; clicking edge shows lead-lag scatter.

---

### Day 6 — Sun 2026-05-18 — *cloud deploy + observability + docs*

**Flagship**: three deployment URLs printed in README, all returning 200;
Grafana dashboard accessible at `localhost:3001`; demo video uploaded.
**Scaffold**: GitHub Actions CI green; Helm chart compiles; Terraform
modules linted.

#### Tasks

1. **Deploy backend** (1.5h):
   - `apps/api/Dockerfile` (multi-stage, distroless final).
   - `railway init` → connect repo → Postgres-TimescaleDB add-on → deploy.
   - Verify `https://liulian-api.up.railway.app/healthz` → 200.
   - Set `EXPO_PUBLIC_API_URL` + `NEXT_PUBLIC_API_URL` to this URL.

2. **Deploy web** (1h):
   - `vercel link apps/web` → `vercel deploy --prod`.
   - Verify `https://liulian-web.vercel.app/forecast` renders.

3. **Mobile distribution** (1h):
   - `eas build --profile preview --platform all`.
   - Upload `.apk` to a downloadable URL; print Expo Go QR in README.
   - **Do not** submit to app stores.

4. **Observability** (1.5h):
   - Local: `docker-compose -f infra/compose/observability.yml up`
     (prom + grafana + loki + tempo).
   - Pre-build Grafana dashboards (JSON) per blueprint §12; ship in
     `infra/grafana/dashboards/`.

5. **CI** (1h):
   - `.github/workflows/ci.yml` per blueprint §12: ruff + mypy + pytest
     + biome + tsc + vitest + playwright smoke.
   - Make CI green. If a job fails, fix root cause — *never* skip with
     `if: false`.

6. **Helm + Terraform scaffold** (1h):
   - `infra/helm/liulian-platform/` chart compiles (`helm template`
     succeeds, no values needed).
   - `infra/terraform/aws-eks/` + `infra/terraform/hetzner-k3s/` —
     `terraform init && terraform validate` passes for both.

7. **Docs** (1.5h):
   - Rewrite `README.md` from PLATFORM_BLUEPRINT.md §1, §3, §6, §15.
   - Architecture diagram (Mermaid).
   - Screenshots (web + mobile).
   - "Try it now" with the live URLs and Expo QR.
   - Privacy + Security section verbatim from the strategy doc.
   - `docs/ARTORG-PORTFOLIO.md` (the 250-word block for the application
     PDF).
   - `docs/DEMO.md` (end-to-end demo script).
   - `docs/strategy/PLATFORM_BLUEPRINT.md` and `ONE_WEEK_SPRINT.md` —
     already exist; just link them.

8. **Demo video** (1h):
   - 90 seconds: open landing → click "Demo" → forecast canvas appears →
     click a station → chat sidebar → "zero-shot forecast" → mobile QR
     scan → same chart on phone.
   - Upload to Loom; embed in README + landing.

#### Verification

- All three deploy URLs return 200.
- `gh workflow run ci.yml --ref feat/one-week-sprint` → green within 10
  min.
- `helm template infra/helm/liulian-platform` → produces YAML.
- README renders cleanly on github.com.

---

### Day 7 — Mon 2026-05-19 — *polish + submit + tag*

**Flagship**: ARTORG application PDF submitted with portfolio section
citing live URLs. v0.6.0 tag pushed.
**Scaffold**: end-to-end smoke test green on the live URLs; "what's not
yet" honest-mature section in README.

#### Tasks

1. **End-to-end smoke** (1.5h):
   - Fresh browser → web URL → forecast canvas works.
   - Fresh phone → Expo Go QR → app runs.
   - Run an `experiments/swiss_river/` job through the live API:
     `liulian-sdk` from a fresh `uv venv` → succeeds.

2. **"What's not yet done (on purpose)" section in README** (0.5h):
   - Per-tenant auth (production design described).
   - On-device inference (ONNX runtime; next iteration).
   - Federated learning (research direction; relevant to ARTORG).
   - Formal compliance (HIPAA / GDPR / FADP — would be done with
     clinical partners).

3. **ARTORG application package** (1h):
   - Copy `docs/ARTORG-PORTFOLIO.md` block into the application's
     portfolio PDF.
   - Verify live URLs match what the doc claims.
   - Submit per the JD requirements.

4. **Tag + release** (0.5h):
   - `git tag v0.6.0-portfolio` → push.
   - GitHub release notes referencing the demo URLs.

5. **Hand-off note for week 2** (0.5h):
   - Top of `docs/strategy/PLATFORM_BLUEPRINT.md` §15 lists M2 — copy
     the M2 deliverables into a `docs/strategy/M2_PLAN.md` to seed next
     week.

6. **Buffer** (2h): the previous five days will overrun. This is the
   buffer.

#### Verification

- Application submitted; confirmation email saved.
- Tag `v0.6.0-portfolio` visible on github.com.
- All ten checklist items in PLATFORM_BLUEPRINT §9 (verification) check
  green.

## 3. Cut-list — what to drop if the sprint runs hot

In strict order — first cut first:

1. **Helm + Terraform scaffold** (Day 6 last hour) — defer to M2. The
   stubs are valuable but not visible to demo viewers.
2. **Multi-model overlay on Day 4** — keep single-model fan; add overlay in M2.
3. **Topology overlay on Day 5** — single-station markers are enough for
   demo.
4. **Mobile native build (.apk)** — Expo Go QR alone is enough.
5. **TSL adapter promotion** — defer to M2; Chronos-2 alone gives the
   "external model" story.
6. **Agent BI tool surface** — keep agent as Q&A only; defer write tools
   (add_panel, set_filter) to M2.

**Never cut**: the FastAPI + Swagger UI, the canvas with the fan chart,
one Expo Go-running mobile screen, the README. Those are the irreducible
demo.

## 4. Risks and pre-mitigations

| Risk | Pre-mitigation |
|---|---|
| Existing tests break when renaming `liulian/` → `packages/liulian-core/` | Day 1 *first* task; do nothing else until `pytest` green |
| Chronos-2 weights are large (~700MB)/slow to install | Pin Chronos-Bolt (smaller) for the sprint; document Chronos-2 as opt-in |
| Railway Postgres-TimescaleDB free tier limits | Fall back to `pg_partman` + `pg_cron` on plain Postgres; defer Timescale to staging |
| MapLibre + swisstopo tile attribution | Read swisstopo OpenData license; bake attribution into the map footer |
| Agent calling third-party LLM with sensitive data | Default LiteLLM provider = local Ollama / qwen; cloud LLM behind a per-tenant feature flag |
| Mac-less applicant can't test iOS | Use Expo Go on a borrowed iPhone for one screenshot; main demo is Android emulator + Expo Go |
| Day 6 deployment cascade-fails | Reserve Day 7 morning as backup deploy window; keep Day 6 push to be a tested commit |

## 5. End-of-sprint verification checklist (replicates PLATFORM §17, scoped to M1)

- [ ] GitHub repo public; README renders cleanly
- [ ] `https://liulian-api.up.railway.app/healthz` → 200
- [ ] `https://liulian-web.vercel.app/forecast` → renders forecast chart
- [ ] Expo Go QR works on at least one physical device
- [ ] `gh workflow view ci.yml` shows green latest run
- [ ] `helm template infra/helm/liulian-platform` produces valid YAML
- [ ] `terraform validate` passes for both modules
- [ ] `docs/ARTORG-PORTFOLIO.md` text copied into application PDF
- [ ] 90-second demo video embedded in README
- [ ] Privacy + Security section in README
- [ ] No PHI / no real patient data in repo
- [ ] `v0.6.0-portfolio` tag pushed
- [ ] `docs/strategy/M2_PLAN.md` created with M2 deliverables

## 6. After Sprint — interview / VC-call ready talking points

**Architecture rationale**:
- *"Why FastAPI?"* — OpenAPI auto-gen → SDK + web codegen → end-to-end
  type safety. Async-native; same Python culture as the research core.
- *"Why TimescaleDB?"* — we are a TS product; we eat our own dogfood.
  Hypertables make run-metric and forecast-history queries fast at any
  scale without manual sharding.
- *"Why ECharts not Plotly?"* — order-of-magnitude better at large
  datasets in the browser; theming engine lets us speak the brand.
- *"Why Expo not native?"* — single codebase iOS+Android, OTA, no Xcode.
  At our scale the cost is invisible; if we ship to enterprise we can
  fall back to bare RN for one screen.
- *"Why LangGraph for agents?"* — deterministic state graph; explicit
  tool surface; testable.
- *"Why Helm + Terraform now?"* — the day after a paying customer signs,
  we deploy to their AWS/Azure account in 1 day. The scaffold makes that
  promise real.

**Healthcare angle (ARTORG)**:
- ECG demo flavour for an AIHN application; same platform, plug-in
  manifest swap.
- Privacy posture: PhysioNet open data only; no third-party LLM by
  default; documented on-device inference path.
- Federated learning sketched in the M5 roadmap.

**Spatio-temporal moat (VC)**:
- 30+ models in the zoo, 17 benchmark datasets in the matrix — published
  in a leaderboard auto-updated nightly.
- TSL integration → first-class spatial graphs.
- Chronos-2 zero-shot lowers the cold-start cost for a new vertical.

---

*Cross-reference: [PLATFORM_BLUEPRINT.md](PLATFORM_BLUEPRINT.md) §3
(architecture), §6 (frontend), §8 (BI), §12 (deploy).*
