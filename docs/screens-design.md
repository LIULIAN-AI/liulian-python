# LIULIAN GUI Demo — Multi-Screen Design Specification

**Author:** under `feat/gui-demo`
**Date:** 2026-05-05
**Revision:** 3 (replaces the iter01 dark dashboard and the iter02 single-page bento)

---

## 0. Why this revision exists

The previous bento dashboard, even with the UniBE red repaint, still felt **AI-generic**. The reasons are now explicit:

- A 7-card bento grid that displays *everything at once* is the visual signature of AI-generated marketing pages. Real research and data tools rarely look like this.
- "Agent" eyebrows on every card lean into the 2024–2026 AI hype vocabulary.
- A single dashboard cannot make any one of the four product surfaces feel deep — every card has 1/7 of the screen.
- Decorative elements (corner brackets, rotated side marks, status pills with extreme letter-spacing) all read as design templates rather than working software.

This revision restructures the demo into a **four-screen product application**, each screen modelled after a real industry tool. The 20-second story becomes a **product tour**: tab → reveal → tab → reveal. Tab transitions are the moments of delight.

---

## 1. Positioning and references

The four screens map directly to four research/ML tools an investor or partner will already recognise:

| Screen           | Reference product(s)              | What it has to feel like |
|------------------|-----------------------------------|------------------------|
| **Data**         | Snowflake catalog, Hex Magic data view, Linear data | A data engineer's workbench: schema tree, table preview, validation column |
| **Train**        | Weights & Biases, MLflow tracking | A training console: live loss chart on the left, runs table on the right |
| **Inference**    | Replicate, Hugging Face Inference Endpoints, Modal | An API playground: endpoint card, input form, output panel, code snippet |
| **Insight**      | Hex Magic, Julius AI, Mode Helix, Cursor chat | A chat with the dataset: messages, embedded charts, source citations |

The brand surface (top nav + LIULIAN wordmark + status bar) is the only thing that stays constant across screens. Everything else is purpose-built for the task.

---

## 2. Visual identity refresh

| Layer | Iter 2 (replaced) | Iter 3 (this spec) |
|---|---|---|
| Layout primitive | 7-card bento grid | One screen at a time, tab-switched |
| Typography hero | Fraunces wordmark with red `U` | **Kept** — appears only in the 2 s opener |
| Typography UI | Switzer + JetBrains Mono | **Same body fonts**, used more functionally; less display serif on the working screens |
| Status pills | "STREAMING", "RISK ELEVATED" all-caps | Lowercase status badges with small dots — closer to Linear / Notion |
| Eyebrows on cards | Every card has a mono eyebrow | Removed. Section titles are plain. |
| Corner brackets | On cards | Removed. |
| Rotated side marks | On hero | Removed. |
| Connector flow lines | On dashboard | Removed (already removed in iter 2 for the same reason). |
| Decorative ornament | Significant | Practically zero. The product *is* the ornament. |

**Words removed from the page entirely:** "Agent" (as a label suffix), "PRODUCTION READY", "ORCHESTRATION", "INTELLIGENCE PLATFORM", "SUBSTRATE", "STREAMING", "ELEVATED" (only as a capital pill — the word is fine in body copy).

**Words added:** "Run", "Endpoint", "Latency", "Throughput", "Schema", "Source", "Send", "Run inference", "Step", "Epoch". These are the actual nouns of working ML tools.

---

## 3. Per-screen specifications

Common chrome (every screen):

```
┌────────────────────────────────────────────────────────────────────┐
│ [LIULIAN]    Data  ·  Train  ·  Inference  ·  Insight    ⏤  status │   ← top nav
├────────────────────────────────────────────────────────────────────┤
│ {screen body}                                                       │
└────────────────────────────────────────────────────────────────────┘
```

The active tab is underlined in UniBE red, 2 px, with a 200 ms slide animation when the active tab changes.

### 3.1 Data (`#screen-data`)

```
┌──────────┬──────────────────────────────────────┬───────────────────┐
│ Sidebar  │  Main                                │ Right             │
│ 220 px   │  fluid                               │ 280 px            │
├──────────┼──────────────────────────────────────┼───────────────────┤
│ Datasets │  ▸ swiss-river-1990 / discharge.parq │  Schema           │
│  ●sR-90  │                                       │  ✓ 10/10 fields  │
│   sR-00  │  manifest.yaml ─────────────────────  │                   │
│   mch-rad│  1  name: swiss-river-1990            │  Integrity        │
│          │  2  schema: hydro-v2                  │  sha256 ✓ verified│
│ Columns  │  3  freq: 10min                       │                   │
│  ts      │  4  span: [1990-01-01, 2026-05-05]    │  Coverage         │
│  station │  5  stations: 2143                    │  2138 / 2143      │
│  value   │                                       │                   │
│  qc      │  Preview ─────────────────────────── │  Stations         │
│  basin   │  ts            station  value  unit  │  ●●●●●●           │
│  lat     │  14:38:00 RHE-BS  1342.7  m³/s      │  (mini map)       │
│  lon     │  14:37:50 AAR-BE   548.2  m³/s      │                   │
│          │  14:37:40 RHE-DI   412.6  m³/s      │  Last sync        │
│          │  ...                                  │  47 min ago       │
│          │  (rows tick in)                       │                   │
└──────────┴──────────────────────────────────────┴───────────────────┘
```

**Animation choreography (4 s window):**
- 0.0 s: sidebar dataset list cascades in (60 ms stagger).
- 0.4 s: manifest.yaml types itself in (line-by-line, 80 ms per line).
- 1.6 s: data table header appears, then rows tick in at 200 ms intervals.
- 2.6 s: right column's schema/integrity blocks appear; mini map stations dot in.
- 3.6 s: hold.

### 3.2 Train (`#screen-train`)

```
┌────────────────────────────────────┬───────────────────────────────┐
│ Main 60 %                          │ Runs sidebar 40 %             │
├────────────────────────────────────┼───────────────────────────────┤
│ ▸ transformer-entity-aware-v3      │  ⌕ filter…                    │
│   epoch 56/80 · running            │ ─────────────────────────────│
│                                    │  Run        Model       MSE   │
│  ┌────────── loss curve ─────────┐ │  ●t-002    Transformer  .0418│
│  │                                │ │  ✓t-001    DLinear+E    .0432│
│  │   train  ▬ red                 │ │  ✓t-000    Mamba        .0451│
│  │   val    ▬▬▬ charcoal dashed   │ │  ✓t-aaf    LSTM         .0476│
│  │                                │ │  ✓t-93c    ETSformer    .0492│
│  └────────────────────────────────┘ │  ✓t-77b    TSMixer      .0514│
│                                    │                               │
│  val_mse  0.0418  ▼-18.6%          │                               │
│  val_mae  0.1293  ▼-14.2%          │                               │
│  GPU mem  9.8/24G                  │                               │
│  step/s   142                       │                               │
│                                    │                               │
│  config.yaml ─────────────────────  │                               │
│  1  model: transformer              │                               │
│  2  dim: 128                        │                               │
│  3  layers: 4                       │                               │
│  4  entity_mode: hash               │                               │
│  5  lr: 3e-4                        │                               │
└────────────────────────────────────┴───────────────────────────────┘
```

**Animation (3.5 s window):**
- 0.0 s: header row appears with model name + epoch counter (ticking).
- 0.3 s: loss chart axis appears; train line draws over 1.6 s; val line draws after 0.6 s.
- 1.4 s: metric cards count up (`val_mse` from 0.78 → 0.0418 with fade-in deltas).
- 2.0 s: runs table cascades in row-by-row (140 ms stagger). Top row is the active run, in pale red.
- 3.0 s: config.yaml types itself (5 lines, 80 ms each). Hold.

### 3.3 Inference (`#screen-inference`)

```
┌──────────────────────────────────────────────────────────────────┐
│  Endpoint                                                        │
│  ┌──[POST]── api.liulian.ch/v1/forecast ─────────── 2.3 ms p50 ─┐│
│  │ Forecast discharge at any Swiss station, 1–72 h horizon     ││
│  └─────────────────────────────────────────────────────────────┘│
├───────────────────────────┬──────────────────────────────────────┤
│ Input                     │ Output                              │
│  station  [Rhein-Basel ▾] │  ┌─── forecast (24 h) ────────────┐ │
│  horizon  [────●───] 24 h │  │   observed ▬                    │ │
│  ci       [0.95         ] │  │   predicted ▬ red              │ │
│  return   [json ▾]        │  │   95 % CI ░░░ red pale         │ │
│                           │  │                                  │ │
│  [Run inference  →]       │  │  threshold ┄ 2 400 m³/s        │ │
│                           │  │  marker  ● red at T+18 h        │ │
│                           │  └──────────────────────────────────┘ │
│                           │  latency  8 ms       throughput  1.2K req/s│
├───────────────────────────┴──────────────────────────────────────┤
│  python ─────────────────────────────────────────────────────────│
│  1  resp = liulian.forecast(                                     │
│  2     station="RHE-BS", horizon=24, ci=0.95)                    │
│  3  print(resp.crossing)  # 2026-05-08T14:00Z, p=0.87            │
└──────────────────────────────────────────────────────────────────┘
```

**Animation (3.5 s window):**
- 0.0 s: endpoint card slides in.
- 0.4 s: input fields appear in cascade (station / horizon / ci / return type).
- 1.2 s: "Run inference" button is "clicked" (subtle scale + flash); latency badge appears with a count-up.
- 1.6 s: output chart materialises (CI band fades in first, then observed line draws, then predicted line draws on top).
- 2.6 s: code snippet types itself.

### 3.4 Insight (`#screen-insight`)

```
┌──────────┬─────────────────────────────────────────┬─────────────┐
│ Sessions │ Conversation                            │ Context     │
│ 220 px   │ fluid                                   │ 240 px      │
├──────────┼─────────────────────────────────────────┼─────────────┤
│ ●Today   │                                          │ Dataset     │
│  Flood   │   ┌─ 14:38 · you ─────────────────────┐ │ swiss-river │
│  risk    │   │ Is there flood risk in the next   │ │ -1990       │
│          │   │ 7 days?                           │ │             │
│ Past     │   └────────────────────────────────────┘ │ Model       │
│  Aare    │                                          │ transformer │
│  anomaly │   ┌─ 14:38 · Insight ─────────────────┐ │ -entity     │
│  Compare │   │ Yes — Rhein-Basel will exceed     │ │ -aware-v3   │
│  2025/26 │   │ 2 400 m³/s on May 8, ~14:00 UTC,  │ │             │
│          │   │ with 87 % confidence (elevated).  │ │ Sources     │
│ + new    │   │                                   │ │ ‣ stations  │
│          │   │ ┌─── mini forecast chart ──────┐  │ │ ‣ manifest  │
│          │   │ │                              │  │ │ ‣ run t-002 │
│          │   │ └──────────────────────────────┘  │ │ ‣ baseline  │
│          │   │                                   │ │             │
│          │   │ The crossing happens 18 h from    │ │ Reasoning   │
│          │   │ now. Confidence is tight (±60     │ │ trail       │
│          │   │ m³/s) — upstream Aare is well-    │ │ data fetch  │
│          │   │ observed.                         │ │ forecast    │
│          │   └─ sources: ··· ─────────────────── │ │ reasoning   │
│          │                                          │ │ ↳ done     │
│          │   [Ask the agent...]              [Send]│ │             │
└──────────┴─────────────────────────────────────────┴─────────────┘
```

**Animation (5–6 s window — the longest, because this is the climax):**
- 0.0 s: session list and context column appear with a subtle slide.
- 0.3 s: user message bubble appears (right-aligned, gray).
- 0.5 s: agent header appears with a typing indicator (3 dots).
- 0.9 s: typing indicator replaced by the streamed answer — words appear in chunks (~80 ms per chunk), with `Rhein-Basel`, `2 400 m³/s`, `May 8 ~14:00 UTC`, `87 %`, `elevated` rendered as subtle red emphasis.
- 2.6 s: mini chart slides in inside the bubble (height ~120 px).
- 3.4 s: second paragraph types in.
- 4.4 s: source chips appear at the bubble footer.
- 5.0 s: reasoning trail in the right sidebar marks each step done in sequence.

---

## 4. Tab navigation

Top nav layout:

```
[ □  LIULIAN ]    Data  ·  Train  ·  Inference  ·  Insight       LIVE   14:38 UTC   Bern · Hydro Lab
                  ───
```

- The dot below the active tab is a 2 px UniBE-red bar that slides between positions on tab change (`transform` only, ~280 ms).
- Hover state on inactive tabs: text colour shifts from `#5C6066` to `#131313` over 120 ms; no underline appears.
- Tabs are real `<button>` elements with `aria-current` for the active one.

## 5. Twenty-second timeline (revised)

| t       | Event                                                                              |
|---------|------------------------------------------------------------------------------------|
| 0.0 s   | Hero overlay paints. Wordmark `LIULIAN` reveals letter-by-letter; the `U` is the red italic accent.  |
| 1.2 s   | Tagline "Modeling rivers, sensors, and signals — together, in time." fades in.     |
| 1.8 s   | Loader bar fills.                                                                  |
| 2.4 s   | Hero shrinks toward top-left and fades out. Top nav becomes visible.               |
| 2.5 s   | **Data** tab activates. Datasets list cascades in.                                 |
| 2.9 s   | manifest.yaml types in.                                                             |
| 3.5 s   | Data table rows tick in.                                                            |
| 4.6 s   | Right column (schema / integrity / mini map) reveals.                              |
| 6.0 s   | **Tab transition: Data → Train.** Underline slides; main content cross-fades.      |
| 6.3 s   | Loss chart axis + train line drawing.                                              |
| 7.4 s   | Val line drawing.                                                                   |
| 7.6 s   | Metric cards count up.                                                             |
| 8.2 s   | Runs table cascades in.                                                            |
| 9.4 s   | config.yaml types in.                                                              |
| 10.5 s  | **Tab transition: Train → Inference.**                                             |
| 10.7 s  | Endpoint card visible. Input fields appear.                                        |
| 11.7 s  | Run-inference button "clicks" (scale dip).                                         |
| 12.0 s  | Latency badge counts up (0 → 8 ms).                                                |
| 12.3 s  | Output chart materialises (CI → observed → predicted).                             |
| 13.0 s  | Code snippet types in.                                                             |
| 13.8 s  | **Tab transition: Inference → Insight.**                                           |
| 14.0 s  | Sessions list appears.                                                             |
| 14.2 s  | User message appears.                                                              |
| 14.5 s  | Agent typing indicator.                                                            |
| 14.9 s  | Streamed reply begins.                                                             |
| 16.6 s  | Mini chart appears inside agent bubble.                                            |
| 17.4 s  | Second paragraph + sources.                                                        |
| 18.4 s  | Reasoning trail in right sidebar marks each step.                                  |
| 19.4 s  | Hero stamp: a single line "From manifest to forecast — in 24 hours." fades in at the bottom centre. |
| 20.0 s  | Hold. `?loop=1` reloads.                                                            |

The five tab activations (`Data`, `Train`, `Inference`, `Insight`, finale stamp) are the "story beats". An investor watching once still gets the message just from the underline jumping across the tab strip.

---

## 6. Component inventory

To be implemented in the new `styles/main.css`:

| Component | Used on | Notes |
|---|---|---|
| `.nav-tabs` | every screen | Buttons + sliding underline |
| `.dataset-list` | data sidebar | Tree-like, dot for selected |
| `.code-block` | data, train, inference | Mono pre with line numbers, key colouring |
| `.data-table` | data | Sticky header, hover rows, mono numerics |
| `.runs-table` | train | Compact, status-dot column, active row red |
| `.metric-card` | train | Number + delta + label, no decoration |
| `.endpoint-card` | inference | Method badge + URL + p50 latency |
| `.field` | inference | Label + control (select / slider / input) |
| `.btn.primary` | inference | Red bg, white text, 4 px radius, no shadow |
| `.chat-message` | insight | User vs agent variant, bubble + meta |
| `.mini-chart` | insight | Inline 320×120 forecast chart |
| `.session-list` | insight | Vertical list with timestamps |
| `.context-list` | insight | Right column key-value list with chips |
| `.status-bar` | every screen | Bottom bar with run id + clock + alerts |

## 7. Mock data

Per-screen content lives in `scripts/mock-data.js`:

```js
MOCK.datasets   = [{name, version, status, count}, ...]
MOCK.columns    = [{name, type, nullable, sample}, ...]
MOCK.dataRows   = [...]                        // data table preview
MOCK.manifest   = "name: swiss-river-1990\n..."
MOCK.runs       = [{run_id, model, status, val_mse, ...}, ...]
MOCK.config     = "model: transformer\n..."
MOCK.endpoints  = [...]
MOCK.codeSnippet = "resp = liulian.forecast(...)"
MOCK.chat       = [{role, segments: [...]}, ...]
MOCK.contextRefs = [...]
```

Existing items (sensor stations, loss curve generator, forecast generator, KPI list) stay; new items are appended.

## 8. AI-genericness checklist (revised)

Specific patterns that the new design **must not contain**, learned from why iter 2 still felt AI-generic:

- ☑ No bento grid showing every module at once.
- ☑ No "Agent" label suffix on visible UI titles. (The product can have an `Insight` view; the visible label says `Insight`, not `Insight Agent`.)
- ☑ No corner brackets, rotated side marks, or "scan-line" effects.
- ☑ No "STREAMING" / "ELEVATED" / "LIVE" all-caps pills with extreme letter-spacing.
- ☑ No mono-eyebrow above every section heading.
- ☑ No display-serif title above every working pane (only the hero uses Fraunces; everything else uses the body sans).
- ☑ No "·" decorative separators in body copy (only in the top nav, where they read as breadcrumbs).
- ☑ No `radial-gradient` background on every working screen — the screens are flat white, only the hero uses warm radials.
- ☑ Real input affordances: hover states, focus rings, enabled/disabled buttons.
- ☑ Number columns aligned with `font-variant-numeric: tabular-nums`.
- ☑ Status dots render as `<svg>` filled circles, not Unicode `●`.

A second audit pass at the end of iter 3 will tick this list against the live UI.

---

## 9. Implementation order

1. **Skeleton & nav** — replace `index.html` with the four-screen scaffold and the tab nav. Each screen gets an empty container with the right inner grid.
2. **Tokens trim** — strip the `main.css` of the bento-era components; keep `:root` tokens but remove unused selectors.
3. **Components** — implement `.code-block`, `.data-table`, `.runs-table`, `.chat-message`, `.field`, `.btn.primary` once, used across screens.
4. **Per-screen wiring** — fill each `<section>`'s body using existing chart functions where possible, plus the new components.
5. **Scene controller v3** — switch from "reveal cards in cascade" to "switch tabs at fixed times". Add a sliding-underline animation.
6. **Mini-chart inside the chat bubble** — extracted from `Charts.drawForecast`, smaller, fits in a 320×120 frame.
7. **Three rounds of browser screenshot review**, each pass tightening one of: spacing, type-size hierarchy, mock data realism.
8. **Final audit** against §8.
9. **Update README + design report**, regenerate screenshots, push.

This document is the contract. The code is being written from this spec, not the other way around.
