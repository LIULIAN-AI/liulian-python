---
title: LIULIAN Platform Design (L3) — brand, BI canvas, agent flows, mobile UX
status: living document
owner: Linlin Jia (jajupmochi)
created: 2026-05-12
last_revised: 2026-05-12
companion_docs:
  - PLATFORM_BLUEPRINT.md      (L1–L2 vision + architecture)
  - ONE_WEEK_SPRINT.md         (L4 implementation)
  - LIULIAN_REUSE_MAP.md     (concrete fork-and-adapt plan)
  - conventions/UI_AUDIT_CHECKLIST.md  (every PR runs through this)
source_canon:
  - .worktrees/gui-demo/styles/main.css       (the visual canon: tokens, layout)
  - .worktrees/gui-demo/docs/design-report.md (the brand bible)
  - github:CTU-Bern/unibeCols                 (UniBe corporate palette)
> **Language:** English | [中文](PLATFORM_DESIGN.zh.md) *(zh stub pending)*
---

# LIULIAN Platform Design (L3)

> This document is the *visible surface* of LIULIAN: brand voice, color,
> typography, BI canvas panel-by-panel, agent conversation flows, mobile
> screens. Everything a designer or a frontend engineer needs to ship a
> screen without asking. The blueprint (L1–L2) answers *why*; this doc
> answers *what does it look like*.

## 0. Visual originality contract (the single most important rule)

LIULIAN reuses code, architecture, and operational patterns from
liulian and the 12 reference platforms. **Every visible pixel is
original.** No screenshot of any LIULIAN surface should be mistakable
for:

- a liulian screen with text swapped
- a Refine.dev template with logo swapped
- a generic shadcn-admin starter
- any 2026 SaaS-default admin panel
- a Vercel / Linear / Stripe surface with our copy on it

Every PR runs the 4 tests in `conventions/UI_AUDIT_CHECKLIST.md` §I:
AI-slop, category-reflex, liulian-copy, gui-demo cross-check. Any
failure blocks merge.

---

## 1. Brand voice (L3.0)

### 1.1 The four-word position

**Liquid Intelligence for Time.**

Used as the wordmark caption on `/(marketing)`. The `U` in
*Liquid* and *LIULIAN* shares a stylistic accent: Fraunces variable
font, WONK 1, weight 600, italic, in `--unibe-red`. That single
character carries the brand's signature.

### 1.2 The 25-word elevator

*Open-source production stack for spatio-temporal AI. Research-grade
model zoo, production-grade BI canvas, agentic workflows, sovereign
deployment. Designed in Bern.*

### 1.3 Voice attributes

| Attribute | Yes | No |
|---|---|---|
| Register | scientific, editorial | corporate, breezy, hype-y |
| Pace | considered, sparse | rapid, energetic |
| Verbs | concrete (observe, manifest, train, forecast, alert) | abstract (revolutionize, transform, unlock) |
| Pronouns | "we" rare; mostly impersonal scientific voice | "your team", "you" excessive |
| Numbers | exact, with units | "10x" / "100x" empty multipliers |
| Adjectives | restrained, technical | superlatives (best, smartest, fastest) |
| Banned words | seamless, unleash, elevate, next-gen, game-changer, delve, harness, leverage (as verb), revolutionize, transform (in marketing context) | — |

### 1.4 Anti-references (we are not these)

- *Tableau / Power BI generic dashboard*: too template-y, too blue,
  too "enterprise"
- *Crypto / Web3 neon dashboards*: garish, distracting motion
- *Generic AI tool with cyan + violet gradients*: AI-slop default
- *Vercel marketing*: brilliant but already imitated 1000 times
- *Linear product UI*: our /studio aspires *toward* its discipline but
  uses different typography and palette to avoid look-alike

### 1.5 Reference platforms we admire (citing what we steal)

- *Monocle magazine*: typography hierarchy, paper feel, photo
  restraint, sentence rhythm
- *Bloomberg Terminal lineage*: information density done with
  microtypography and whitespace, never visual clutter
- *Linear*: command palette as primary nav, restrained chrome
- *Müller-Brockmann / Swiss typographic tradition*: grid systems,
  asymmetric balance, restrained color
- *Edward Tufte (data viz)*: ink-data ratio discipline, sparkline
  inspiration
- *gui-demo iteration 2 (this repo)*: the canonical realised LIULIAN
  brand

---

## 2. Design tokens (L3.1)

The single source of truth lives in `liulian-design-system/tokens.json`
(new repo) and is built into per-platform exports:

- `tokens.css` (CSS custom properties)
- `tailwind.preset.js` (Tailwind preset)
- `tokens.ts` (TypeScript const for `liulian-web`)
- `tokens.rn.ts` (React Native StyleSheet for `liulian-mobile`)
- `antd-theme.ts` (antd ConfigProvider for chat sidebar + tables)

### 2.1 Color (OKLCH; hex provided for reference)

```css
/* — Canvas + Ink — */
--canvas-warm     : oklch(98.5% 0.005 90);     /* #FBFBFA */
--surface-pure    : oklch(99.5% 0.002 90);     /* #FFFFFF (functional only) */
--ink-charcoal    : oklch(22.5% 0.01 250);     /* #131313 */
--ink-muted       : oklch(50% 0.02 250);       /* #5C6066 */
--ink-faint       : oklch(63% 0.015 250);      /* #8E9296 */
--hairline        : oklch(92% 0.005 250);      /* #EAEAEA */

/* — UniBe red (the brand) — */
--unibe-red       : oklch(56.5% 0.20 25.5);    /* #E20613 */
--unibe-red-tint  : oklch(94% 0.04 25.5);      /* #FDEBEC */
--unibe-red-deep  : oklch(46% 0.20 25.5);      /* #B00510 (hover/focus) */

/* — Status pastels (rare; reserved for severity ribbons + pills) — */
--pastel-green    : oklch(95% 0.03 145);       /* #EDF3EC */
--pastel-blue     : oklch(94% 0.04 240);       /* #E1F3FE */
--pastel-yellow   : oklch(95% 0.05 90);        /* #FBF3DB */

/* — Optional UniBe secondary (chart series only, sparingly) — */
--unibe-ocean     : oklch(50% 0.13 230);       /* ~#0066B3 */
--unibe-green     : oklch(60% 0.13 145);       /* ~#509A39 */
--unibe-apricot   : oklch(67% 0.13 65);        /* ~#E6863A */
```

**Color strategy**: *Committed* (per `impeccable` taxonomy). UniBe red
carries identity at ~5% of total inked area; warm-bone neutrals carry
~92%; pastels and chart secondaries together ~3%.

**Rules**:

1. **UniBe red `#E20613` is the sole spot color**, used only on:
   wordmark `U`, active station marker, predicted-forecast line, CI band
   color, severity-elevated pill, threshold-crossing marker, leading dot
   on selected row. Maximum 2 visible red elements per viewport on most
   pages; 3 only on the alert canvas.
2. **Never use `#000` or `#fff`**. The browser default `<body>` background
   resets to `--canvas-warm`.
3. **Status pastels** appear only inside pills or severity ribbons,
   never as large area fills.
4. **Secondary UniBe colors** (ocean, green, apricot) appear ONLY as
   chart series colors when more than one is needed; never as button
   colors or surface backgrounds.

### 2.2 Typography

```css
--font-display    : 'Fraunces', 'Instrument Serif', 'Charter', serif;
--font-body       : 'Switzer', 'Inter Tight', system-ui, sans-serif;
--font-mono       : 'JetBrains Mono', 'IBM Plex Mono', monospace;
```

**Variable axes** on Fraunces:

- `opsz` 9..144 (optical size)
- `wght` 300..900
- `SOFT` 0..100
- `WONK` 0..1 (used at WONK=1 for accent italic `U`)

**Scale** (1.25 ratio, geometric):

| Step | Use | Size | Family / weight | Tracking |
|---|---|---|---|---|
| **Display 1** | hero wordmark | 96–144px | Fraunces / 600 italic on accent char | -0.035em |
| **Display 2** | section bands | 56px | Fraunces / 500 roman | -0.025em |
| **H1** | page title | 40px | Fraunces / 500 | -0.02em |
| **H2** | section heading | 28px | Fraunces / 500 | -0.015em |
| **H3** | sub-heading | 20px | Switzer / 600 | -0.005em |
| **H4** | small heading | 16px | Switzer / 600 | -0.005em |
| **Body L** | long-form prose | 17px / 1.6 | Switzer / 400 | -0.005em |
| **Body M** | UI text | 15px / 1.55 | Switzer / 400 | 0 |
| **Body S** | meta / footer | 13px / 1.5 | Switzer / 400 | 0 |
| **Caption** | chart annotation | 11px / 1.4 | Switzer / 500 uppercase | 0.04em |
| **Mono** | numbers, code, time | 13/15px | JetBrains Mono / 400, tabular-nums | 0 |
| **Mono Caption** | running header | 11px | JetBrains Mono / 500 uppercase | 0.04em |

**Rules**:

1. Body line length cap 72ch; never wider.
2. Hierarchy ratio ≥ 1.25 between adjacent steps.
3. `font-variant-numeric: tabular-nums` on every numeric column or
   metric (KPI, table cells, run metrics).
4. `font-feature-settings: 'ss01', 'ss02'` on Switzer where it helps
   character disambiguation (g/q tail).
5. Inter / Roboto / Open Sans are BANNED.

### 2.3 Spacing + radii + borders

```css
--space-1  : 4px;
--space-2  : 8px;
--space-3  : 12px;
--space-4  : 16px;
--space-5  : 24px;
--space-6  : 32px;
--space-7  : 48px;
--space-8  : 64px;
--space-9  : 96px;
--space-10 : 128px;

--radius-sm  : 4px;
--radius-md  : 10px;   /* card default */
--radius-lg  : 14px;
--radius-xl  : 20px;
--radius-pill : 9999px;

--border-hairline : 1px solid var(--hairline);
```

**Rules**:

1. Section padding between major bands: 96px or 128px (`--space-9 / -10`).
2. Card resting state: 1px hairline border, 10px radius, no drop shadow.
3. Hover state on cards: hairline color shifts to `--ink-faint` (subtle);
   never adds drop shadow at rest.
4. Spacing varies for rhythm; same padding everywhere is monotony.
5. No `rounded-full` on cards or primary buttons.

### 2.4 Motion

| Use | Curve | Duration |
|---|---|---|
| Scroll entry (`opacity` + `translateY(12px)`) | `cubic-bezier(0.16, 1, 0.3, 1)` (ease-out-quart) | 600ms |
| Hover lift (`opacity` shift on border) | `cubic-bezier(0.33, 1, 0.68, 1)` | 200ms |
| Page transition | none (instant; SSR) | 0ms |
| Threshold-crossing `流` Easter egg | ease-out-quart | 300ms once |
| Skeleton shimmer | linear | 1500ms loop |

**Rules**:

1. Animate `transform` + `opacity` only. Never `top` / `left` / `width` /
   `height`.
2. No bounce, no elastic, no spring physics.
3. `prefers-reduced-motion`: respect via media query; all motion → instant.

### 2.5 Iconography

- Library: **Phosphor Icons (Regular weight 1.5px stroke)** for system
  icons. Lucide as fallback (used by liulian; consistent with code
  reuse).
- Custom icons (hand-drawn-feel): for any *domain-specific* glyph
  (river-network node, threshold marker, etc.), draw bespoke at 24px
  grid with 1.5px stroke matching Phosphor.
- Icon-in-cell BANNED in tables: text + status typography only.

---

## 3. Information architecture

### 3.1 `liulian-web` route map

```
/                        ← marketing landing (single-page editorial)
/(marketing)/about       ← longer-form positioning
/(marketing)/research    ← published papers + benchmark leaderboard
/(marketing)/blog        ← contentlayer-driven posts
/(marketing)/pricing     ← M3+ only

/forecast                ← BI canvas (THE killer demo surface)
/forecast/r/:slug        ← shared report (read-only public link)

/studio                  ← workspace shell with 4-tab IA
/studio/data             ← datasets + manifests
/studio/train            ← experiments + runs + HPO sweeps
/studio/inference        ← forecasts + scenarios
/studio/insight          ← reports + scheduled summaries

/agents/data             ← data agent chat
/agents/model            ← model agent chat
/agents/bi               ← BI agent chat (also reachable inline on /forecast)

/admin                   ← tenant + user + audit (M3+)

/docs                    ← MkDocs-rendered technical docs (via contentlayer)
/api/*                   ← FastAPI proxy (server-side)
```

The 4-tab IA inside `/studio` matches the gui-demo's
**Data / Train / Inference / Insight** — the canonical LIULIAN
mental model.

### 3.2 Page chrome

Every page (except `/(marketing)`) carries the **scientific running
header** at the top:

```
LIULIAN · Studio · 2026-05-12 14:38 UTC · swiss-river-1990 / lstm / entity=none / seed=42
```

- JetBrains Mono, 11px, uppercase letter-spacing 0.04em
- Color: `--ink-faint` (#8E9296)
- Sticky, 32px height
- The trailing run-coordinates segment is dynamic (depends on the
  active context)

Below the header: 56px navigation strip (sidebar collapsed) + the
main content. No breadcrumbs (the running header replaces them).

---

## 4. The BI canvas — `/forecast` (the killer demo)

This is THE page that determines whether a viewer says "I want to meet
whoever built this". Every detail matters.

### 4.1 Top-level layout

12-column × 8-row CSS Grid on desktop ≥ 1440px (matches gui-demo's
bento grid). Single column scroll on mobile.

```
┌─────────────────────────────────────────────────────────────────────┐
│ Running header                                                       │
├─[ 56px nav ]──[ canvas ]────────────────────────────────────────────┤
│              ┌────────────────────────────────────────────────────┐  │
│              │ Map (col 1–7, row 1–4)                              │  │
│              │ — MapLibre + swisstopo + topology overlay           │  │
│              └────────────────────────────────────────────────────┘  │
│              ┌──────────────────┐ ┌────────────────────────────────┐ │
│              │ Stations list    │ │ Multi-model fan chart          │ │
│              │ (col 8–10, r1–4) │ │ (col 8–12, r1–4 OR col 1–7 r5) │ │
│              │ — sticky filter  │ │ — Q05/Q95 fan + alert markers  │ │
│              └──────────────────┘ └────────────────────────────────┘ │
│              ┌──────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────┐  │
│              │ KPI MAE  │ │ KPI RMSE │ │ KPI CRPS    │ │ Coverage │  │
│              │ (1, 5)   │ │ (2, 5)   │ │ (3, 5)      │ │ @90 (4,5)│  │
│              └──────────┘ └──────────┘ └─────────────┘ └──────────┘  │
│              ┌────────────────────────┐ ┌──────────────────────────┐ │
│              │ Correlation matrix     │ │ Alert severity ribbon    │ │
│              │ (col 1–6, row 6–8)     │ │ (col 7–12, row 6–8)      │ │
│              └────────────────────────┘ └──────────────────────────┘ │
│              [ Add panel + ]                                          │
│                                                                       │
└───────────────────────────────────────────────────────[ chat sidebar ]┘
                                              ┌─ 360px right rail ────┐
                                              │ BI agent chat         │
                                              │ — @ant-design/x       │
                                              │ — themed brand tokens │
                                              └───────────────────────┘
```

Tile drag/resize via `react-mosaic-component`; the orchestration shell
(`CanvasOrchestrator` + `ReportBuilder` + per-widget config panel) is
adapted from a private fintech codebase the author worked on
previously — see `adr/0008-canvas-orchestrator-reuse.md` for the
attribution and adaptation map. Layouts persist per user; share URL
freezes them read-only.

### 4.2 The eight canonical panels

#### Panel 1: Interactive river-network map

- **Renderer**: MapLibre GL with swisstopo `pixelkarte-grau` tiles
  (open license, attribution baked into footer).
- **Topology overlay**: SVG layer above raster; edges = upstream →
  downstream relationships from manifest; edge thickness ∝ mean
  discharge; edge color ∝ current forecast residual relative to actual.
- **Station nodes**: concentric rings.
  - Outer ring: forecast uncertainty band width (Q95 − Q05), drawn in
    `--unibe-red-tint`.
  - Inner dot: current observation, `--ink-charcoal` for normal,
    `--unibe-red` for active station.
- **Interactions**:
  - click → cross-filter the entire canvas to this station
  - right-click → "Open station profile" peek pane
  - long-press → "Run scenario forecast"
- **Edge case**: more than 200 edges → switch from SVG overlay to
  canvas; topology becomes a heatmap of node density.
- **Empty state**: shows the swisstopo basemap centered on
  Bern with a single Fraunces caption: *"No stations match the active
  filter. Press ⌘K to widen."*

#### Panel 2: Forecast time-series with prediction intervals (canonical)

- **Renderer**: ECharts custom-themed.
- **Layers** (back to front):
  1. CI band fill: `--unibe-red-tint` at α 0.18
  2. Forecast mean: `--unibe-red` dashed 1.5px
  3. Observation: `--ink-charcoal` solid 1.5px
  4. Threshold markers: 4px dots `--unibe-red` with 200ms ripple on
     first appearance
  5. Annotations (user-drawn): `--ink-muted` 1px
- **Axes**: Y in physical units with `font-variant-numeric: tabular-nums`;
  X in localized timestamps (en-CH default).
- **Brush**: `dataZoom: [{ type: 'inside' }, { type: 'slider' }]` —
  the slider sits below the chart in `--ink-faint`.
- **Easter egg**: when any threshold marker appears for the first time
  in a session, render the `流` glyph at the marker's position for
  300ms (opacity 0 → 0.45 → 0). Distinctive, domain-true, never
  repeated on the same marker.
- **Tooltip**: timestamp (mono) + values stacked vertically with units
  (e.g. `408.2 m³/s`). No icons.

#### Panel 3: Multi-model overlay (compare A vs B vs Chronos)

- Pick up to 3 models via dropdown multi-select (`cmdk` palette also
  works).
- Fan colors:
  - Model 1: `--unibe-red` family
  - Model 2: `--unibe-ocean` family
  - Model 3: `--unibe-green` family
- KPI strip beside the chart shows per-model MAE / RMSE / CRPS /
  Coverage@90.
- Toggle: "Diff mode" subtracts ground truth from each fan; toggle:
  "Pairwise" computes Model A − Model B residual and renders as a
  single fan around zero.

#### Panel 4: Cross-station correlation matrix

- Reordered by topology (upstream → downstream), so the river-shape is
  visually emergent (not alphabetical).
- Each cell: a heatmap value (Spearman ρ over the selected window).
- Cell color scale: warm-bone at 0, deepens to `--unibe-red` at +1,
  `--unibe-ocean` at -1.
- Click a cell → pop a scatter plot of the two stations' residuals
  with a fitted line.
- Reorder toggle: topology / alphabetical / hierarchical clustering.

#### Panel 5: Anomaly / alert severity ribbon

- Datadog-lineage layout, restyled.
- X axis: time (matches canvas window).
- Y axis: one row per active alert rule.
- Each ribbon segment colored by severity:
  - Watch: `--pastel-yellow` background, `--ink-muted` text
  - Elevated: `--unibe-red-tint` background, `--unibe-red-deep` text
  - Critical: `--unibe-red` background, `--canvas-warm` text
- Click a segment → opens the originating forecast context in the
  time-series panel.
- SOP shortcuts on segment right-click: "Notify on-call",
  "Acknowledge", "Mark false-positive".

#### Panel 6: Scenario / counterfactual

- A small form with: precipitation delta (+/− %), temperature delta,
  upstream flow delta, time range.
- On apply, the BI agent invokes the counterfactual prediction tool and
  renders a *third fan* (in dashed grey) on the time-series panel.
- The form preserves last-used values per session.

#### Panel 7: Station profile (modal-less peek pane)

- Slides in from the right rail (replaces chat sidebar temporarily)
  when a station is right-clicked.
- Tabs: Hydrograph (long history) · Seasonality (STL decomposition) ·
  Latest forecast · Alert history · Lead-lag neighbors.
- Close button (⌘W or Escape); does not block canvas interaction.

#### Panel 8: Report builder

- Drag and drop layout via `react-grid-layout` (or the existing
  `react-mosaic-component`).
- Save → POST `/reports` → returns `slug` for `/forecast/r/{slug}`
  shareable URL.
- "Make public" toggle exposes read-only view.
- Export PDF (server-side Puppeteer) preserves typography.

### 4.3 Chat sidebar (right rail, 360px)

Connects to `liulian-agent` via SSE (`/agents/bi/invoke`). Tool calls
render inline as scientific notebook-style "method calls":

```
> show me Bern station Q95 last week
  [bi-agent] query_forecasts(
    station_id="aare-bern-2.1",
    metric="q95",
    window="-7d..now"
  ) -> 17 forecasts found

  [bi-agent] add_panel(report_id=..., panel_spec=...)
  -> panel added to canvas

  Result: Bern's Q95 stayed within historical norm except 2026-05-09
  when precipitation upstream lifted it 12% (visible in the new
  panel on your canvas, top right).
```

- Method calls use JetBrains Mono.
- Plain-text responses use Switzer.
- Citations appear as Chicago-style superscript footnotes when the
  agent references manifest data, papers, or other forecasts.
- No avatar bubbles. The presentation is a notebook, not a chat.

### 4.4 Empty / loading / error states

- **Loading**: warm-bone shimmer skeleton matching the eventual layout
  (Linear-lineage). Skeleton shimmer is `--canvas-warm` to `--surface-pure`
  at 1.5s linear loop.
- **Empty**: a single Fraunces italic line in `--ink-muted`, generously
  white-spaced. No illustrations.
- **Error**: a hairline-bordered card with one line of plain-language
  error, one `cmdk`-keyed retry action, and a structured error code in
  JetBrains Mono. No icons.

### 4.5 Keyboard discipline

- `⌘K` / `Ctrl+K`: command palette (primary nav)
- `j` / `k`: row navigation in tables and stations list
- `gg` / `G`: top / bottom
- `o`: open peek for the focused row
- `e`: edit the focused row's metadata
- `?`: open the keyboard reference (Fraunces caption inside a peek)
- `/`: focus the search field
- `Esc`: close any peek / dialog
- `b`: toggle BI agent sidebar
- `m`: toggle map / table view of the stations panel

---

## 5. `/studio` — Linear-meets-Bloomberg editorial Swiss

The studio is the research-engineer's home. It is dense, keyboard-first,
and reads as a *scientific instrument*. Not an admin panel.

### 5.1 Sidebar

- 56px collapsed default; expands on hover to 220px.
- Items: Data · Train · Inference · Insight · Agents · Admin · Docs.
- Active item: a `--unibe-red` 6×6 dot to the left of the label
  (replaces side-stripe).
- Footer of sidebar: user avatar (initials in Fraunces) + tenant name +
  `⌘K` hint.

### 5.2 List page archetype

Every list page (experiments, runs, models, datasets) shares an
**asymmetric two-column layout**:

```
┌────────────────────────────────┬──────────────────────────────┐
│ Table (60% width)              │ Explainer (40% width)         │
│  - sticky toolbar              │  - Fraunces italic Q&A:       │
│  - status as typography        │     "What is a run?"          │
│  - mono numbers                │     "A single training trial. │
│  - keyboard nav (j/k/o/e)      │      Created by an experiment.│
│  - row right-click → peek      │      Compare them in the      │
│                                 │      table to the left."     │
│                                 │  - inline glossary chip       │
│                                 │  - one anchor chart           │
└────────────────────────────────┴──────────────────────────────┘
```

The 60% table is the workhorse; the 40% explainer is the editorial
twist that signals "this is a scientific tool". Different list pages
get different anchor charts:

- `/studio/data`: dataset size histogram
- `/studio/train`: experiment-count-per-week sparkbar
- `/studio/inference`: forecast-count-per-day sparkbar
- `/studio/insight`: report-views-per-week sparkbar

### 5.3 Detail page archetype

A single full-width page split into three vertical bands:

```
┌─────────────────────────────────────────────┐
│ Title band                                   │
│  - Fraunces 40px run name                    │
│  - JetBrains Mono 13px run coordinates       │
│  - Status (typographic) + actions            │
├─────────────────────────────────────────────┤
│ Metrics band                                 │
│  - 4 KPI cards (Tremor primitives, themed)   │
│  - inline sparkline per metric               │
├─────────────────────────────────────────────┤
│ Logs / params / forecasts band               │
│  - tabbed: Logs · Params · Forecasts · Audit │
│  - keyboard nav between tabs (h/l)           │
└─────────────────────────────────────────────┘
```

Tab content uses the canvas time-series chart (panel 2) when a
forecast is selected, ensuring brand consistency across surfaces.

### 5.4 Command palette (⌘K)

- Trigger: `⌘K` / `Ctrl+K` opens a 600×400 modal anchored top-center.
- Search input: 24px Switzer, no border (only a hairline beneath).
- Results: grouped by section (Data, Train, Inference, Insight,
  Agents, Settings, Help).
- Each result row: 14px Switzer label + 11px JetBrains Mono right-aligned
  keyboard shortcut.
- Selection: 6×6 `--unibe-red` dot at left of the active row.
- ⌘K palette is the **primary navigation** for `/studio`. Sidebar is
  a fallback / discovery aid.

### 5.5 Forms

- Single-column, max-width 480px, generous vertical rhythm.
- Labels above inputs (Caption style, uppercase, letter-spaced).
- Inputs: 1px hairline, no fill, 4px radius. Focus ring: 1px
  `--unibe-red-deep`, no glow.
- Inline validation: errors below the input in `--unibe-red-deep`,
  Body S size.
- Primary submit button: full-width on mobile, auto-width on desktop;
  `--ink-charcoal` fill, `--canvas-warm` text, 6px radius.
- Cancel: text link, `--ink-muted`.

---

## 6. `/(marketing)` — the landing surface

The landing page is the only place where LIULIAN allows itself
*editorial flourish*. Magazine-like, restrained, scientific.

### 6.1 Hero

- Full-bleed background: swisstopo satellite tile of the Aare basin at
  Brienzersee, overlaid with `--canvas-warm` at 88% opacity. (A real
  Swiss-river photo, not stock.)
- Centered display text on the warm-bone overlay:

  > **Liquid Intelligence for Time.**

  Fraunces 96–144px responsive, weight 500 roman, with the `U` in
  *Liquid* and the `U` in any visible occurrence of *LIULIAN* set in
  italic WONK 1, weight 600, color `--unibe-red`.

- Sub-line (one sentence, exactly 17 words):

  > *Open-source production stack for spatio-temporal AI: a research-grade
  > model zoo wrapped in production-grade BI.*

- Live counter strip below: `2,143 sensors live · 12 models in benchmark
  · 4 agents ready`. JetBrains Mono 13px uppercase, `--ink-faint`. The
  numbers tick in real time via SSE from a public endpoint.

### 6.2 Three bands

Below the hero, three full-width bands divided by **vertical rules**
(1px hairlines), not cards. Each band:

- One Fraunces 28px section heading
- One Switzer 17px paragraph (max 65ch)
- One real chart from the live platform (not stock; via embedded
  EChart pointed at `/api/public/...`)
- One anchor sentence in italic at the band's foot

Bands:

1. **Manifest** — *"Every dataset starts with a contract."* — chart
   is the swiss-river-1990 manifest YAML rendered with syntax highlight
2. **Train** — *"Thirty models, one runtime."* — chart is the
   benchmark leaderboard sparkline (real, updated nightly)
3. **Forecast** — *"From prediction to a hydrologist's risk brief, in
   twenty seconds."* — chart is the canonical forecast fan (live data)

### 6.3 Inline demo video

Single 1440×900 screen capture of the BI canvas, auto-play muted loop,
60 seconds, on viewport entry. Border: 1px hairline, 10px radius.

### 6.4 CTA

A single sentence at the foot of the page:

> *Open SwissRiver demo →*

No button. No badges. No "Schedule a demo" form. The arrow is the
button. Hovering shifts the arrow 4px right at 200ms ease-out-quart.

### 6.5 Footer

- Three columns: Product (links) · Research (publications) · About
  (team, license, contact).
- Bottom: copyright line + UniBe attribution + license badge.

---

## 7. Agent conversation flows

Three agent personas; same engine, different tool surfaces. SSE event
shapes (inherited from `liulian-agent`):

```
event: thinking      data: { message: "Resolving station-id…" }
event: trace         data: { step: "intent", detail: "forecast_query" }
event: intent        data: { intent: "forecast_query", entities: [...] }
event: tool_call     data: { tool: "query_forecasts", input: { … } }
event: tool_result   data: { tool: "query_forecasts", output: { … } }
event: response      data: { text: "…", references: [...] }
event: suggestions   data: ["next q1", "next q2", "next q3"]
event: done          data: null
```

### 7.1 Data agent (`/agents/data`)

- **Persona**: a meticulous data steward.
- **Tools**: `list_files`, `summarise_csv`, `propose_manifest`,
  `validate_manifest`, `detect_topology`, `detect_seasonality`.
- **Opening prompt** (auto-shown on empty session):

  > *"I help shape datasets into manifests. Drop a CSV path or a
  > MinIO URI; I'll suggest a manifest. You always approve before I
  > write."*

- **Surface**: chat sidebar (right rail) with a left pane that previews
  the manifest YAML being proposed (live).

### 7.2 Model agent (`/agents/model`)

- **Persona**: a quietly opinionated TS modeler.
- **Tools**: `list_models`, `recommend_model`, `propose_hpo_space`,
  `read_run_logs`, `diagnose_failed_run`, `compare_runs`.
- **Opening prompt**:

  > *"Given a dataset and a horizon, I recommend a model + an HPO space.
  > I read your run logs when training fails. I never train without
  > your go."*

### 7.3 BI agent (`/agents/bi`, primary on `/forecast`)

- **Persona**: a senior analyst who answers in evidence.
- **Tools**: `query_forecasts`, `add_panel`, `set_filter`,
  `create_alert_rule`, `export_report`.
- **Opening prompt** (on `/forecast` empty canvas):

  > *"Ask in plain English. Examples: 'Show stations where Q95 went
  > above 850 last week.' 'Compare TimesNet and Chronos for Bern.'
  > 'Alert me if any upstream station crosses elevated.'"*

### 7.4 Cost + safety UX

- **Per-message cost indicator** in the bottom right of the chat
  message: `$0.0023 · 1,840 tokens in / 412 tokens out · DeepSeek V4`.
  JetBrains Mono 10px, `--ink-faint`.
- **Cost ceiling reached banner**: `--pastel-yellow` background, one
  line Fraunces italic: *"This run exceeded the per-conversation
  budget. Tell me to raise it or restart."*
- **Provider degradation banner** (e.g. Gemini blocked in CN):
  `--pastel-blue` background, one line: *"Routing to GLM for this
  conversation (Gemini unavailable in your region)."* Same SSE event
  shape as liulian's `DegradationBanner.tsx`.

---

## 8. Mobile UX (`liulian-mobile`)

### 8.1 Tab structure

Three tabs (Expo Router file-system):

```
app/
├── (tabs)/
│   ├── _layout.tsx        ← tab bar config
│   ├── index.tsx          ← Home: alerts + recent runs
│   ├── forecast.tsx       ← Forecast: single-station viewer
│   └── alerts.tsx         ← Alerts: severity ribbon (collapsed)
├── station/[id].tsx       ← Station profile (deep-link)
├── chat.tsx               ← Quick BI agent question (modal-less)
└── _layout.tsx            ← root layout with brand fonts
```

### 8.2 Brand tokens on mobile

Tokens import from `@liulian/design-tokens` as RN StyleSheet exports.
Fraunces + Switzer + JetBrains Mono loaded via `expo-font` from the
embedded `.ttf` files (not Google Fonts at runtime; offline-safe).

### 8.3 Home screen

```
┌─────────────────────────────┐
│ Running header (mono 10px)  │
├─────────────────────────────┤
│ Good morning, Linlin         │ ← Fraunces 28px
│ 2 alerts · 4 forecasts ready │ ← Switzer 15px
│                              │
│ [ Run a quick forecast ]     │ ← single CTA, charcoal fill
│                              │
│ Recent alerts                │ ← Caption uppercase
│ ┌─────────────────────────┐  │
│ │ Bern · Q95 elevated     │  │ ← row with red dot
│ │ 2 hours ago             │  │
│ └─────────────────────────┘  │
│ ┌─────────────────────────┐  │
│ │ Thun · Watch state      │  │
│ │ Yesterday               │  │
│ └─────────────────────────┘  │
└─────────────────────────────┘
```

### 8.4 Forecast tab

Single station picker (segmented control across the top) + Victory
Native XL fan chart (matches web's panel 2 brand exactly). Brush works
with two-finger gesture; tap to annotate.

### 8.5 Alerts tab

Severity ribbon adapted to vertical layout. Each row collapses /
expands on tap to reveal the forecast context.

### 8.6 Quick forecast modal (sheet)

Reachable from the Home CTA. Pick station + horizon (presets 24h /
48h / 7d). On submit, runs Chronos-2 zero-shot (`/models/chronos-2/predict`)
and displays the fan chart inline. The "zero-shot demo" is the
audience-recognized win.

### 8.7 Push notifications

Triggered by alert rules. Body uses Switzer (system font fallback on
OS render). Tapping opens the relevant `station/[id]` deep link.

---

## 9. Documentation surface (`/docs`)

### 9.1 Long-form editorial

`/docs` uses contentlayer + MDX. Body uses **Fraunces** (yes, serif
body) for prose passages — the docs read like a scientific paper, not
a Stripe API reference. Code blocks use **JetBrains Mono**. Captions
under figures use **Switzer**.

### 9.2 Two-column with margin notes

Tufte-lineage layout:

```
┌────────────────────────────────────┬─────────────┐
│ Prose column (max 65ch)             │ Margin      │
│  - Fraunces 17/1.65 body            │  - Switzer  │
│  - JetBrains Mono code blocks       │    13px     │
│  - inline citations                 │    italic   │
│  - figures full-width as needed     │  - footnote │
│                                     │    refs     │
│                                     │  - "see also│
│                                     │    " links  │
└────────────────────────────────────┴─────────────┘
```

Code block syntax highlighting uses a custom theme matching brand
tokens (no Solarized / Monokai etc.).

### 9.3 Inline charts

Whenever a doc page would show a `print(metric)` output, an actual
ECharts visualization is rendered inline (data piped from `/api/public/...`).
The docs are dogfood for the platform.

---

## 10. Accessibility

- AA contrast: body text on `--canvas-warm` is 14.5:1 (well above 4.5:1
  minimum).
- All actions keyboard-reachable; focus rings 1px `--unibe-red-deep`.
- `prefers-reduced-motion` collapses all motion to instant.
- All charts ship with a hidden `<table>` of the underlying data for
  screen readers.
- Color is never the only signal of state (status uses typography too).
- Localization-ready throughout via `next-intl`; bilingual docs per
  `claude-config:bilingual-docs` rule (en + zh canonical).

---

## 11. Brand voice in the wild — copy library

Snippets we reuse across surfaces. Maintain in `liulian-design-system/copy.json`.

| Key | English | Chinese (canonical for marketing) |
|---|---|---|
| hero.title | Liquid Intelligence for Time. | 流动的智能，关于时间。 |
| hero.subtitle | Open-source production stack for spatio-temporal AI: a research-grade model zoo wrapped in production-grade BI. | 时空 AI 的开源生产栈：研究级模型库，工业级商务智能。 |
| hero.cta | Open SwissRiver demo → | 打开瑞士河流演示 → |
| counter.suffix | sensors live · models in benchmark · agents ready | 个传感器在线 · 个模型在基准 · 个智能体就绪 |
| band.manifest.title | Every dataset starts with a contract. | 每份数据集都从一份契约开始。 |
| band.train.title | Thirty models, one runtime. | 三十个模型，一个运行时。 |
| band.forecast.title | From prediction to a hydrologist's risk brief, in twenty seconds. | 从预测到水文学家的风险简报，二十秒。 |
| agent.bi.opening | Ask in plain English. | 直接用中文问。 |
| empty.no_results | No stations match the active filter. Press ⌘K to widen. | 当前筛选没有匹配站点。按 ⌘K 放宽条件。 |

Future strings are PRed to the same copy library, never inlined.

---

## 12. Open design questions (track and resolve)

- **/agents/bi modality**: chat-only sidebar vs. inline-edit-the-canvas
  ("agent moves panels for you")? Pilot inline-edit at M2; fall back to
  chat-only if it confuses analysts.
- **Mobile rounded corners**: 10px (match web) or 16px (iOS native
  preference)? Test on physical iPhone in sprint Day 5; decide then.
- **Marketing animations**: how restrained? Currently zero scroll-jacking;
  may add subtle parallax on hero. Pilot at M2 critique.
- **Dark mode**: rejected as default per gui-demo principles; the
  warm-bone canvas is the *primary* identity. Optional inverted theme
  at M3 if customer demands. Color tokens are already OKLCH so the
  inversion is mechanical when needed.

---

## 13. The PR-merge UI audit

Every frontend PR runs `conventions/UI_AUDIT_CHECKLIST.md` (46 items
across 11 sections). Reviewer pastes the audit block into the PR body;
unchecked rows block merge.

The non-negotiable rows:

- Visual originality (no liulian/Refine/template lookalike)
- Brand canon (UniBe red ≤ 2 visible per viewport on most pages)
- AI-slop test pass
- Category-reflex test pass
- gui-demo cross-check pass

---

*See `PLATFORM_BLUEPRINT.md` for the architectural why behind these
surface decisions. See `LIULIAN_REUSE_MAP.md §14.3` for the
"Linear-meets-Bloomberg editorial Swiss" framing source. See
`conventions/UI_AUDIT_CHECKLIST.md` for the per-PR gates.*
