# LIULIAN GUI Demo — Design Report

**Author:** generated under `feat/gui-demo` orphan branch
**Date:** 2026-05-05
**Audience:** investors, external collaborators, internal review

---

## 1. Brief

Build a **20-second product demo** of the LIULIAN spatiotemporal research platform that can be dropped into a slide deck as a video or GIF. Frontend-only, mock data is acceptable, but the demo must be:

1. *Read-at-a-glance* by a non-engineer audience (investor / partner).
2. *Accurate to the actual platform* (a research tool from UniBE × ETH for Swiss-river forecasting).
3. *Visually distinctive* — must not look like a "generic AI dashboard mockup".

This report captures the research, design choices, two iterations, and final state.

---

## 2. Context audit

Before writing any markup, the author re-read four sources of truth:

- `README.md` — the architecture diagram and the four-layer model (Data / Task / Model / Runtime).
- `CLAUDE.md` — the project's conventions and the phrase that mattered most: *"task-driven framework with strict layer boundary"*.
- `docs/research/2026-04-16-spatio-temporal-intelligence-research-proposal.md` — what the platform is positioned as scientifically.
- `experiments/swiss_river/` — the dataset that already drives the test runs.

Findings used in the demo:

- Real models in tree: `swiss_lstm.py`, `swiss_transformer.py`, plus DLinear / Mamba / ETSformer / TSMixer adapters. The HPO leaderboard reflects this exact list.
- Real manifest filename: `swiss-river-1990.yaml`. Reproduced verbatim in the data-agent overlay.
- Real entity-aware mode names from `entity_mixin.py`: `hash`, `embed`, `onehot`, `none`. Reproduced in the HPO config strings.
- Real station window: 1989 → 2026, ~2 100 stations. Rounded to "2 143" for the demo.

Doing this audit first is what gives the demo its specificity. None of the strings on screen are filler.

---

## 3. Narrative

The demo is, in cinematic terms, a **single take from substrate to alert**, in four beats:

1. **Brand** (0 – 3 s). One word, one accent character. The accent is the red italic `U` — visually carrying the "liquid / unique" identity, and previewing the only spot colour the rest of the page will use.
2. **Data substrate** (3 – 5 s). The Swiss map draws itself, sensors light up, telemetry rows scroll. This is the layer most investors recognise — *"OK, you have the data."*
3. **Compute** (5 – 8 s). Loss curves draw, HPO leaderboard ranks. *"OK, you train models."*
4. **Inference + insight** (8 – 17 s). The forecast chart and the reasoned brief are the two most expensive cards in the layout because they are the most expensive *claims*: the platform doesn't just produce numbers, it produces a sentence a hydrologist can act on. The threshold-crossing marker is the climax.

A single sentence summarises the demo: *"From manifest, to forecast, to a hydrologist's risk brief — in twenty seconds."* The closing badge says exactly that.

---

## 4. Iteration log

### Iteration 1 — Editorial dark theme

Initial direction. Deep ink background `#04060d`, cyan `#4ffbe9` as primary accent, gradient mesh radials in cyan + violet + amber, SVG `feGaussianBlur` drop-shadow filters on the active elements, topographic-contour repeating-linear-gradient on the body.

**What worked**

- Sense of depth from the radial mesh + grain.
- Loss curves and forecast chart popped against the dark canvas.
- Data agent map felt cinematic.

**What did not work** (and why iteration 2 happened)

- *AI-generic*. Almost every auto-generated AI dashboard mockup on the public internet uses the same combination: deep ink + cyan + purple + amber + grid lines. A user / investor seeing the demo would silently classify it as "yet another AI page". This is a perception risk worth paying real cost to avoid.
- *No brand grounding*. There was no reason for the cyan; it was just "AI-default". A spatiotemporal research platform from UniBE has an actual brand colour available — UniBE red.
- *Minimalist-ui audit*. Running the design through the bundled `minimalist-ui` skill flagged: glow filters, primary-coloured large surfaces, gradient meshes, and the use of clichéd phrasing in headlines.

Captured at `screenshots/iter01_*.png` for reference.

### Iteration 2 — UniBE red on warm paper

Repaint, not refactor. The component structure, the timeline, and the data are unchanged; only the styling and copy.

**Palette**

- Canvas: `#FBFBFA` warm paper + 6 % SVG turbulence grain + two extremely faint red radial spots (`opacity 0.018 – 0.025`) at the diagonals, for depth without flatness.
- Surfaces: pure white (`#FFFFFF`) with a 1 px `#EAEAEA` hairline. No shadows on cards in the resting state.
- Text: `#131313` primary, `#5C6066` secondary, `#8E9296` faint.
- Accent: UniBE red `#E20613`. Used for the wordmark `U`, the active stations, the train loss line, the predicted forecast line, the CI band, the top HPO row, the `ELEVATED` pill, the badge dot, and nothing else.
- Pastels for status pills: pale-green `#EDF3EC` (live, healthy), pale-red `#FDEBEC` (elevated risk, predicted-vs-observed CI), pale-blue `#E1F3FE` (informational), pale-yellow `#FBF3DB` (warnings).

**Typography**

- *Display:* Fraunces (variable, op-sized at 144) for the wordmark; Instrument Serif fallback. The variation `WONK 1` lets the italic accent character carry small typographic flourishes that distinguish it from the upright letters.
- *Body:* Switzer (Fontshare). No Inter, no Roboto, no Open Sans — those would mark the page as auto-generated.
- *Mono:* JetBrains Mono. Tabular figures (`font-variant-numeric: tabular-nums`) on numeric columns to keep KPI / score columns aligned.

**Copy rewrite**

Cliché-heavy phrases were replaced with plain ones:

| Before | After |
|---|---|
| "Where space and time converge in intelligence." | "Modeling rivers, sensors, and signals — together, in time." |
| "INITIALISING SUBSTRATE" | "Loading substrate" |
| "PRODUCTION READY · END-TO-END ORCHESTRATION" | "End-to-End — Manifest · Train · Reason · Forecast — in 24 hours" |
| "Spatiotemporal Intelligence Platform" | "Spatiotemporal Research Platform" |
| "Zürich · ETHZ Lab" | "Bern · Hydrology Lab" |

**Motion**

Removed the SVG `flow-overlay` (dashed lines floating between empty cells made it look like noise rather than data flow). All animations now use only `transform` and `opacity`. The most theatrical moment — the threshold crossing marker pulsing three times at t = 17 s — is also the only moment the badge is allowed to appear.

Captured at `screenshots/iter02_*.png`.

---

## 5. Component-by-component notes

### Hero

```
┌───────────  University of Bern · Spatiotemporal Research  ─────────────┐
│                                                                         │
│           ─── SPATIOTEMPORAL RESEARCH PLATFORM ───                       │
│                                                                         │
│                       L I U L I A N                                      │
│                          ^red italic U                                   │
│                                                                         │
│          Modeling rivers, sensors, and signals — together, in time.      │
│                                                                         │
│             ┌─ LOADING SUBSTRATE ─ 2 143 SENSORS · 12 MODELS · 4 AGENTS ┐│
│             ████████████████████████████████████████████████████████████││
│                                                                         │
│                    Data Agent · Model Atelier · Inference · Insight       │
└─────────────────────────────────────────────────────────────────────────┘
                                          (rotated marks on the side rails)
```

The hero is intentionally bibliographic — it reads like the front page of a scientific report, not a SaaS landing page. This sets the tone for the rest of the dashboard, which uses the same restraint.

### Bento grid

A 12-column / 8-row CSS Grid. Asymmetric on purpose — the data agent and insight cards are the largest because they are the most semantically loaded for an investor. Training and HPO are mid-sized; data stream and KPIs are the smallest. The forecast chart spans seven columns at the bottom because it's the climax.

### Charts

All three charts are hand-rolled SVG (`scripts/charts.js`), not Chart.js or D3. Reasons:

- The animation is bespoke (lines drawing themselves via `stroke-dasharray` + `stroke-dashoffset` transitions, the CI band fading in 200 ms after the predicted line starts to draw).
- Total third-party JS on the page is zero; the entire payload is < 50 KB.
- Editing colours / legends takes one CSS variable change; with a chart library it would take working through the library's theming layer.

---

## 6. Audit checklist (post-iteration 2)

Run against the `minimalist-ui` skill's negative-constraint list:

| Rule                                                          | Status   |
|---------------------------------------------------------------|----------|
| No `Inter`/`Roboto`/`Open Sans`                                | passed   |
| No generic thin-line icon libraries                            | passed (no icon library in use)|
| No heavy drop shadows                                          | passed (cards have no shadow at rest)|
| No primary-coloured large backgrounds                          | passed (only the 28-px brand mark)|
| No gradients / neon / 3D glassmorphism                         | passed (only one 6 px topbar `backdrop-filter` blur)|
| No `rounded-full` for cards or primary buttons                 | passed (cards are 10 px, pills are explicit pills) |
| No emojis                                                      | passed   |
| No "John Doe" / "Acme Corp" / "Lorem Ipsum"                    | passed (real station codes from the manifest)|
| No AI cliché copy                                              | passed (full rewrite)|
| Text colours never absolute black                              | passed (`#131313` for body)|
| `1 px solid #EAEAEA` divider rule                              | passed (every hairline uses `--hairline: #EAEAEA`)|
| Animations on `transform` + `opacity` only                     | passed (no `top`/`left`/`width`/`height` animation)|
| Type contrast (display serif × geometric sans × mono)          | passed (Fraunces / Switzer / JetBrains Mono)|

All thirteen audited rules pass.

---

## 7. Risks / known limitations

- **Fontshare CDN dependency.** Switzer is loaded from `api.fontshare.com`. If the CDN is unreachable during recording, the body sans falls back to Helvetica Neue, which is acceptable but loses some character. For production recording, mirror the font locally.
- **Demo time fixed at 20 s.** The narrative is dense — investors who pause-and-read will appreciate it, investors who watch it once may need a second viewing. This is the right trade-off for a slide insert; for a longer pitch, the same components can be re-staged across 60 s.
- **No real backend.** Stream rows recycle a fixed 15-row buffer, KPI numbers are static, the forecast crossing point is computed once at page load. A reviewer who inspects the network tab will see no XHR / WebSocket. This is intentional — the demo's job is the visual argument, not a backend test.

---

## 8. What ships in this branch

- `index.html` — single-page demo entry.
- `styles/main.css` — full design system + components, ~700 lines, no preprocessor.
- `scripts/mock-data.js` — illustrative datasets.
- `scripts/charts.js` — SVG renderers (Swiss map, loss, forecast, sparkline).
- `scripts/scene-controller.js` — 20-second timeline orchestrator with `?freeze`, `?speed`, `?loop` URL params for screenshot / recording.
- `screenshots/iter01_*.png` — iteration 1 (rejected dark theme), kept for the iteration log.
- `screenshots/iter02_*.png` — iteration 2 (final).
- `README.md` — operator-facing documentation.
- `docs/design-report.md` — this file.

Everything else from the parent repo is intentionally absent: this is an orphan branch.
