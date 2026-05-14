---
title: LIULIAN — Reference Designs (what we learned, what we steal, what we avoid)
status: living document
owner: Linlin Jia (jajupmochi)
created: 2026-05-12
companion_docs:
  - PLATFORM_BLUEPRINT.md
  - ONE_WEEK_SPRINT.md
goal: Capture the design intelligence we gathered from 12 reference platforms,
      so the LIULIAN team can revisit decisions without re-doing the research.
update_rule: re-audit every six months; bump `last_audited` per entry.
---

# LIULIAN Reference Designs

Twelve platforms, each chosen for one specific reason we wanted to learn
from. The entries follow a common template:

> **One-liner** — **What they do well** — **What we steal** — **What we
> deliberately avoid** — **Last audited**

The point is *not* a competitor matrix. It's a memory bank — we revisit
when blueprint sections feel weak.

---

## A. Time-series / ML cores (the engine references)

### A1. Time-Series-Library (THU-ML)

- **One-liner**: 40+ deep TS model implementations under one repo (Tsinghua ML).
- **Does well**:
  - Curated *and updated* model zoo — TimesNet, iTransformer, foundation
    models (TimesFM, Moirai), TimeMoE, Chronos2, Mamba.
  - Bilingual README; leaderboard table; Docker quick-start.
  - Bash scripts per task × dataset → predictable benchmark.
- **Steal**:
  - The benchmark leaderboard as a *first-class artefact* on our docs
    site, auto-generated nightly.
  - The flat `models/` + `exp/` separation between architectures and
    experiment harness.
  - Bilingual (en/zh) docs from day one.
- **Avoid**:
  - Notebook-feel — they have no API/UI/agent layer; we are the layer.
  - "Limited bandwidth for new features" disclaimer — we will commit to a
    monthly release cadence.
- **Last audited**: 2026-05-12

### A2. Chronos / Chronos-2 (Amazon Science)

- **One-liner**: Pretrained zero-shot TS transformers; Apache-2.0.
- **Does well**:
  - Univariate + multivariate + covariate-informed zero-shot prediction.
  - Three product lines (Chronos / Chronos-Bolt / Chronos-2) staged by
    speed-vs-size; clean upgrade story.
  - HuggingFace + SageMaker quickstarts; Apache-2.0 lets us redistribute.
- **Steal**:
  - Adopt Chronos-Bolt and Chronos-2 as the *built-in zero-shot path*.
    One-click "skip training, just forecast" is a demo asset.
  - The clean API: `Pipeline.from_pretrained(...)` → `predict_df(...)`.
  - Quantile predictions out-of-the-box.
- **Avoid**:
  - Their README design — it's HuggingFace-default. Our brand is
    editorial Swiss; we present these models inside our visual system.
- **Last audited**: 2026-05-12

### A3. GluonTS (AWS Labs)

- **One-liner**: Probabilistic TS modeling on PyTorch/MXNet, JMLR paper, mature.
- **Does well**:
  - Probabilistic forecasts as first-class — confidence bands in the
    default visualization.
  - `PandasDataset` abstraction is clean.
  - SageMaker / production story.
- **Steal**:
  - Probabilistic output as a *first-class API field*, not an add-on.
  - Re-use their plotting conventions (50%/90% bands) for our charts.
- **Avoid**:
  - MXNet legacy weight; we are PyTorch-only.
- **Last audited**: 2026-05-12

### A4. pytorch-forecasting

- **One-liner**: TimeSeriesDataSet + Lightning + interpretability.
- **Does well**:
  - `TimeSeriesDataSet` handles missing values, variable horizons, group
    structures — the messy real-world stuff.
  - Built-in interpretation (TFT attention plots) → directly powers a BI
    panel.
  - Optuna integration baked in.
- **Steal**:
  - The "interpretation panel" pattern — attention weights or
    feature-importance plots inside the BI canvas.
  - Patterns from `TimeSeriesDataSet` to inform our manifest spec's
    handling of grouped/multi-entity data.
- **Avoid**:
  - Tying everything to Lightning — we already have our own state
    machine in `runtime/`.
- **Last audited**: 2026-05-12

### A5. TSL (torch-spatiotemporal)

- **One-liner**: Spatio-temporal GNNs on top of PyTorch + PyG.
- **Does well**:
  - Clean factoring of `models/` × `data/` × `engines/` — engines == inference loops.
  - Pre-configured datasets and notebooks; PyG-friendly.
  - Gentle docs ("A Gentle Introduction" notebook).
- **Steal**:
  - The engines abstraction informs our `runtime/` — we keep ours but
    align terminology.
  - Promote our `experiments/adapt_tsl_lib/` study to a first-class
    adapter under `packages/liulian-core/adapters/tsl/`.
- **Avoid**:
  - PyG hard dependency in core — optional extra only.
- **Last audited**: 2026-05-12

### A6. tslearn

- **One-liner**: scikit-learn-style classical TS (DTW, k-Shape, SAX).
- **Does well**:
  - Familiar sklearn API for non-DL practitioners.
  - Clustering algorithms (k-Shape, KernelKMeans) ideal for *cohort* analysis.
- **Steal**:
  - Wrap tslearn cluster outputs as a *Cohort* panel on the BI canvas —
    group similar stations / patients / sensors automatically.
- **Avoid**:
  - Building a competitor classical lib; thin wrapper only.
- **Last audited**: 2026-05-12

---

## B. ML platforms (the platform references)

### B1. ClearML

- **One-liner**: AI infrastructure platform — control plane, dev center, GenAI engine.
- **Does well**:
  - Three-layer architecture story (infra · dev · serve) cleanly explains
    enterprise reach.
  - K8s-native; on-prem / cloud / hybrid story.
  - 2,100+ users; research labs, finance, defense, semiconductors.
- **Steal**:
  - The *three-layer story* shape, but our three layers are
    **research core · platform · vertical**, not infra/dev/serve.
  - Auto-logging philosophy ("track everything without instrumentation").
- **Avoid**:
  - Their UI density and learning curve — repeatedly flagged as steep.
  - Their visual identity (blue + white SaaS-default).
- **Last audited**: 2026-05-12

### B2. MLflow

- **One-liner**: Open-source experiment tracking + model registry.
- **Does well**:
  - Language-agnostic; Python / R / Java / REST.
  - Auto-logging for TF / PyTorch / sklearn.
  - Drop-in; runs anywhere.
- **Steal**:
  - Build *MLflow-compatible REST endpoints* on our tracker so existing
    MLflow clients work transparently — wedge into the user base.
  - The model registry concept (Stage = `Staging` / `Production` /
    `Archived`).
- **Avoid**:
  - Their default UI; ship our own.
  - SQLAlchemy core dep at scale; we use SQLModel + Pydantic.
- **Last audited**: 2026-05-12

### B3. Weights & Biases

- **One-liner**: SaaS experiment tracking with the most polished UI in the space.
- **Does well**:
  - Sweep + table view UX is best-in-class.
  - Auto-records code version, hyperparams, system metrics, predictions.
  - Heavy emphasis on "compare runs" workflows.
- **Steal**:
  - The *compare runs* page pattern — multi-select runs, side-by-side
    metrics, divergence highlights.
- **Avoid**:
  - SaaS-only positioning; we are open-core with self-host as first
    option.
  - Pricing model (per-seat $50–200) — we charge by deployment, not seat.
- **Last audited**: 2026-05-12

### B4. Ultralytics HUB

- **One-liner**: End-to-end CV platform; train → export → deploy.
- **Does well**:
  - 17 export formats; 43 deployment regions; scale-to-zero edges.
  - Project / model / dataset hierarchy with team collaboration.
  - Free + Pro ($29/mo) + Enterprise tiering.
- **Steal**:
  - Export-format diversity story for our model registry (ONNX,
    TorchScript, TFLite for the on-device-inference path).
  - The "train → export → deploy in three clicks" UX as a north star for
    our `/studio` → `/forecast` flow.
- **Avoid**:
  - YOLO-centric branding; we are temporal-first.
- **Last audited**: 2026-05-12

### B5. ClearML / Comet / Neptune common patterns

- **Steal**: tags on runs; pinned runs; "report" as a saved view.
- **Avoid**: cluttered nav with infinite tabs.

---

## C. BI references (the UX & dashboard references)

### C1. FineBI (FanRuan)

- **One-liner**: Zero-SQL enterprise BI; 36,000+ customers; China-dominant.
- **Does well**:
  - "Zero SQL" Excel-style modeling for non-technical users.
  - "Management cockpit" templates — operational dashboards with KPI
    + map + trend in one screen.
  - Industry-vertical playbooks (manufacturing, healthcare, finance).
- **Steal**:
  - The cockpit layout pattern (map+KPI+trend) is the right SwissRiver
    shape — see PLATFORM_BLUEPRINT §8.
  - Vertical-playbook docs: "How to set up LIULIAN for X industry in 30
    min".
- **Avoid**:
  - The default rainbow palette and over-decorated icons.
  - Heavy dependency on Chinese-only docs/support — we lead with en, add
    zh.
- **Last audited**: 2026-05-12

### C2. Power BI custom visual (lucazav/power-bi-time-series-custom-visual)

- **One-liner**: R-Plotly custom visual rendering forecast + intervals.
- **Does well**:
  - Demonstrates the canonical *forecast-with-intervals* visual shape
    that LIULIAN inherits.
  - Multi-model overlay via "Model ID" / "Model Description" fields.
- **Steal**:
  - The visual encoding: solid observation + dashed forecast + shaded
    intervals + sparse markers — directly our `<ForecastChart />`.
- **Avoid**:
  - R runtime dependency; we render in-browser with ECharts.
  - `.pbiviz` packaging is Power-BI-only; we'd never lock into MS BI.
- **Last audited**: 2026-05-12

### C3. Tremor (tremor.so)

- **One-liner**: 35+ Tailwind+Radix dashboard components built on Recharts.
- **Does well**:
  - Ship beautiful KPI cards, sparkbars, micro-metrics in minutes.
  - Tailwind-native, copy-not-install.
  - Open-source.
- **Steal**:
  - Use Tremor for KPI strip + Bar Lists + Tracker components.
  - Tremor's spacing and typography defaults are very close to our brand.
- **Avoid**:
  - Tremor's *charts* themselves (Recharts) are weaker than ECharts for
    large datasets and complex overlays. We use Tremor cards + ECharts
    charts.
- **Last audited**: 2026-05-12

---

## D. Time-series infrastructure (the storage / engine references)

### D1. TDengine (taosdata)

- **One-liner**: High-perf TSDB for IoT/IIoT; cloud-native; AGPL-3.0.
- **Does well**:
  - Native distributed system with sharding, RAFT, K8s-ready.
  - "Cloud Native" *and* embeddable; X64 + ARM64.
  - Stream processing + caching built in.
  - 24.9k GitHub stars; mature ecosystem.
- **Steal**:
  - When we outgrow TimescaleDB, TDengine is the migration target —
    document this on the storage page.
  - Their *stream processing* idea — push the rolling-MAE computation
    into the DB rather than the API layer.
- **Avoid**:
  - AGPL-3.0 forces our deployments to be open-source. TimescaleDB
    (Postgres extension, Apache-2.0) is friendlier for early commercial.
  - Adopting another DB in the sprint week; defer to M3+.
- **Last audited**: 2026-05-12

---

## E. Domain SaaS (the vertical SaaS references)

### E1. HydroForecast (Upstream Tech)

- **One-liner**: Probabilistic streamflow forecasts as a closed SaaS.
- **Does well**:
  - Tight verticalisation: water utilities, hydropower, energy traders,
    mining, government.
  - Hourly-to-annual horizons; multiple delivery (dashboard + API).
  - Earned-trust marketing: extreme-weather case studies, clean-energy
    angle.
- **Steal**:
  - Their hero shot pattern — landscape + map overlay; we adopt a
    swisstopo-tiled CH map with our station nodes for our landing.
  - The "probabilistic, grounded in science" copy is the right
    register for our brand voice.
- **Avoid**:
  - Closed-source / enterprise-pricing-only — we keep open-core.
  - Single-vertical positioning — we are multi-vertical with hydrology
    flagship.
- **Last audited**: 2026-05-12

### E2. k-dense.ai

- **One-liner**: AI research automation across 250+ scientific databases.
- **Does well**:
  - Dark navy + charcoal palette + scientific visualizations (PCA,
    quantum chemistry, genomics) — proves "scientific-credibility" UI
    can be premium.
  - Hero: "Research. Analyze. Synthesize. / From Question to Insight."
  - Institutional partnership signals (MIT, Harvard, Stanford).
- **Steal**:
  - Three-verb hero rhythm — for us: "Observe. Forecast. Decide."
  - Showing *real* scientific visualizations as proof, not stock imagery.
  - "Publish-ready outputs" framing for our report builder.
- **Avoid**:
  - Dark mode as default — our editorial-Swiss brand is warm-bone light.
  - Generic "AI agent" badge fatigue.
- **Last audited**: 2026-05-12

### E3. Datadog (informal reference for alerting)

- **Steal**: severity ribbon UX, SLO-style ribbons.
- **Avoid**: their visual density.

---

## F. Frontend / monorepo (the engineering references)

### F1. liulian (linlin's prior fintech project)

- **One-liner**: Next.js 14 + Ant Design + Spring Boot + MySQL/ES/Redis
  fintech platform.
- **Does well** (from the project's own design notes):
  - Domain-rich (44 controllers, 70 entities) yet legible.
  - i18n out of the box (en / zh-CN / zh-HK).
  - Multi-chart library blend (ECharts + Recharts + amcharts5) chosen per
    use case.
- **Steal**:
  - Stack confidence: Next.js 14 App Router is well-known to the author.
  - Multi-chart strategy: use the *right tool per panel*.
  - i18n early.
- **Differs**:
  - We use shadcn/ui (not Ant Design) because shadcn integrates
    Tailwind better and ships with our minimalist visual identity. Ant
    Design's defaults are too enterprise-default-blue.
  - Backend is Python (FastAPI) not Java (Spring Boot) — Python is the
    research-core language; matching cuts impedance.
- **Last audited**: 2026-05-12 (memory file dated 2025; verify on
  next visit)

### F2. T3 stack / next-forge / Saasfly / turborepo-shadcn

- **One-liner**: Production-grade Next.js + tRPC + shadcn + Turborepo monorepo templates.
- **Steal**:
  - Workspace structure from `create-t3-turbo`: `tRPC routers + DB
    schema + auth` shared between web and mobile in one repo.
  - `next-forge` patterns for SaaS infra (Clerk + Stripe + Posthog).
- **Avoid**:
  - Direct fork — we tailor structure to having Python core + JS apps.
- **Last audited**: 2026-05-12

---

## G. Open questions revisited periodically

These were not fully answered in 2026-05-12 audit; revisit later:

- **GluonTS in 2026** — is the project still active? If MXNet is fully
  abandoned, do we still need a `adapters/gluonts/` layer?
- **ClearML pricing tiers** — opaque on website; would self-host pay-as-go
  fit our cost model?
- **Ultralytics partnership model** — could we plug into their export
  pipeline for the CV-on-temporal-imagery use case (e.g. radar
  precipitation)?
- **FineBI export formats** — could we ingest a `.cpt` template as a way
  to onboard FineBI users to LIULIAN?

---

## H. Periodic re-audit checklist

Every six months, walk through each entry and:

- [ ] Verify the One-liner is still accurate.
- [ ] Update Last audited date.
- [ ] Note any divergence between what we built and what we said we'd
      steal — explain why.
- [ ] Drop or add references based on the market.
- [ ] Cross-link relevant blueprint sections.

Next re-audit due: **2026-11-12**.

---

*Cross-reference: [PLATFORM_BLUEPRINT.md](PLATFORM_BLUEPRINT.md) §1–6
for how these references shape the platform direction; [ONE_WEEK_SPRINT.md](ONE_WEEK_SPRINT.md)
for which references' patterns we apply in the immediate sprint.*
