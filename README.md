# LIULIAN — GUI Demo

A self-contained, frontend-only product demo of the **LIULIAN Spatiotemporal Research Platform** for the University of Bern × ETH hydrology workflow. The page is designed to be recorded once as a 20-second video or animated GIF and dropped into a slide for an investor / stakeholder presentation.

```
.worktrees/gui-demo/
├── index.html              page structure
├── styles/main.css         design system + components
├── scripts/
│   ├── mock-data.js        illustrative datasets (no real telemetry)
│   ├── charts.js           SVG renderers (map / loss / forecast / spark)
│   └── scene-controller.js 20-second timeline orchestrator
├── screenshots/            captured stage references
└── README.md               this file
```

The branch (`feat/gui-demo`) is intentionally an **orphan branch** — only the demo code lives here, with no inheritance from the main research codebase, so it can be served, embedded, or shipped without the wider Python project.

---

## Why this exists

LIULIAN is a research framework. Investors and external collaborators do not benchmark against `pytest -v`; they look at a screen and ask **"what does this team actually ship?"** The demo answers that question in twenty seconds:

1. We have a real **data substrate** (Swiss-River 1990 — 2026, 2 143 stations).
2. We have a real **model atelier** (LSTM, Transformer, DLinear, Mamba, ETSformer, TSMixer — entity-aware variants).
3. We have a real **inference engine** (24-hour discharge forecast with 95 % CI).
4. We have a real **insight layer** (natural-language risk briefs grounded in the forecast).

Each of those four claims maps to one card on the dashboard.

---

## Design language

| Aspect | Choice | Rationale |
|---|---|---|
| Canvas | Warm paper `#FBFBFA` + faint grain | Editorial document feel, not a "neon SaaS" |
| Accent | UniBE red `#E20613` (used sparingly) | Brand-correct, semantically loaded — predictions, alerts, active stations |
| Display type | `Fraunces` variable serif (italic for the accent character) | Editorial, distinctive, plays well with the red |
| Body type | `Switzer` | Geometric-humanist, none of the Inter/Roboto generic look |
| Mono | `JetBrains Mono` | Standard for technical labels and tabular numerals |
| Card chrome | `1px solid #EAEAEA`, 10 px radius, no shadow | Per `minimalist-ui` audit rules — restraint over ornament |
| Motion | Pure `transform` + `opacity` only, < 1 s curves | Quiet sophistication, never spectacle |
| Status pills | Desaturated pastels (`#EDF3EC`, `#FDEBEC`, `#E1F3FE`) | Color-by-meaning, not by decoration |

What we explicitly **avoided** (to remove the AI-generic look):

- No cyan / purple gradient meshes
- No neon `box-shadow` or SVG `feGaussianBlur` glow
- No glassmorphism beyond a `backdrop-filter` on the topbar
- No `Inter` / `Roboto` / `Open Sans`
- No emoji icons
- No copy clichés ("Elevate", "Seamless", "Unleash", "Next-Gen")
- No purple-on-white gradients

These constraints come from the bundled `minimalist-ui` skill, which was used as the audit checklist during iteration 2.

---

## The 20-second timeline

Wall-clock seconds, all triggered by `scripts/scene-controller.js`:

| t        | Event                                                                              |
|----------|------------------------------------------------------------------------------------|
| 0.0 s    | Hero overlay paints. Wordmark `LIULIAN` reveals letter-by-letter; the `U` is the red italic accent. |
| 1.4 s    | Loader bar fills under "2 143 sensors · 12 models · 4 agents".                     |
| 2.6 s    | Hero starts to fade out (opacity + 1.5 % scale).                                   |
| 3.0 s    | **Data Agent** card reveals. Swiss river map draws — country outline, then 4 river paths, then 15 station markers cascading. Two stations glow red (active). |
| 3.25 s   | **Live Discharge** stream begins, one row every 340 ms. `RHE-DI` and `THU-AN` flagged `RISE`. |
| 3.5 s    | **Ingestion KPIs** cards mount with red sparklines.                                |
| 5.0 s    | **Training** card reveals. Train (red solid) and val (charcoal dashed) loss curves draw across 80 epochs. Live meta numbers tick: epoch 0/80 → 80/80, GPU 78 %, val MSE 0.0418. |
| 5.4 s    | **HPO leaderboard** cascades in: Transformer, DLinear+E, Mamba, LSTM, ETSformer, TSMixer. Top row pinned in red. |
| 8.4 s    | **Insight Agent** card reveals.                                                    |
| 8.7 s    | **Forecast** card reveals — observed (charcoal) + predicted (red) lines draw, 95 % CI band materialises behind. |
| 9.0 s    | **Insight typing** begins: a multi-segment natural-language brief streams in at ~80 char/s, with `RHEIN-BASEL`, `2 400 m³/s`, `MAY 8 14:00 UTC`, `87 %` highlighted in red and `ELEVATED` rendered as a red pill. |
| 9.0–14.5 s | The pipeline trail (`Data fetched → Forecast computed → Reasoning → Alert dispatch`) progresses one step at a time. |
| 14.0 s   | Insight typing completes; cursor disappears.                                       |
| 15.7 s   | **End-to-End** stamp slides in at the bottom centre — a black-on-white pill with a red signal dot. |
| 17.0 s   | Forecast threshold-crossing marker pulses to draw the eye.                         |
| 18.0–20 s | Hold on the full dashboard.                                                        |

`?loop=1` reloads at 20 s for unattended recording. `?freeze=hero|data|train|insight|finale` pauses at a stage for screenshotting. `?speed=2` doubles the timeline.

---

## Module reference

### Data Agent (`.card.data-agent`)
A bare Switzerland silhouette with four river paths and fifteen sensor stations. Two stations (Rhein-Basel and Aare-Bern) are coloured red and pulse — these are the basins under active forecasting load. The bottom-left chip surfaces the manifest filename, the SHA-256 verification result, and the cache hit-state — three claims that map directly to LIULIAN's data-contract architecture in the main repo.

### Live Discharge (`.card.data-stream`)
A scrolling table of `(timestamp, station code, m³/s value, status tag)` rows. Rows tagged `RISE` are flagged in red; the rest are pale green. The panel updates continuously to convey "this is a live feed" without being noisy.

### Ingestion KPIs (`.card.data-kpi`)
Four KPI cards: stations online, records today, manifest hash status, p99 latency. Each carries a 16-point sparkline. Numbers use Fraunces with optical-size 60 for editorial presence.

### Training (`.card.training`)
The classic train/val loss-vs-epoch view, but rendered Swiss-style — solid red train, dashed charcoal val, no neon. Below the curves: live epoch / loss / learning-rate / GPU utilisation.

### HPO Leaderboard (`.card.hpo`)
Six rows from Ray Tune ASHA. The top row is pinned with the UniBE red treatment (red border, red text, pale red surface). Each row exposes the configuration string in monospace so an evaluator can read the actual hyperparameters.

### Insight Agent (`.card.insight`)
Two parts. The italic Fraunces brief is a typed-out natural-language summary that calls out the basin, the threshold, the timestamp, and the confidence. Below sit four KPI tiles, including an `ELEVATED` alert tile in red. A four-step pipeline trail at the bottom shows the progression `Data fetched → Forecast computed → Reasoning → Alert dispatch`.

### Forecast (`.card.forecast`)
A 6-day horizon for Rhein-Basel discharge: charcoal observed series, red predicted continuation, faint red 95 % CI band, dashed `NOW` line, deep-red flood threshold at 2 400 m³/s, and a red marker on the predicted threshold-crossing point at T+18 h. The marker pulses three times at t = 17 s for emphasis.

---

## Running it locally

```bash
# from the worktree root
python3 -m http.server 8765 --bind 127.0.0.1
# then open http://127.0.0.1:8765/index.html
```

The page is pure HTML / CSS / JS — no build step, no node_modules, no API. Mock data is hardcoded in `scripts/mock-data.js`.

### URL parameters

| Param | Effect |
|---|---|
| `?freeze=hero` | Stop after the hero stage (~1.5 s in). Useful for capturing the brand reveal. |
| `?freeze=data` | Stop after the data row reveals (~4.5 s in). |
| `?freeze=train` | Stop after the training row reveals (~7.5 s in). |
| `?freeze=insight` | Stop after the insight row reveals and typing has run (~13.5 s in). |
| `?freeze=finale` | Stop in the final hold (~19 s in) — the badge will be visible. |
| `?speed=2` | Double-time the whole timeline. |
| `?loop=1` | Auto-reload at t = 20 s — for unattended video capture. |

---

## Recording the demo

For a slide deck, the simplest workflow is screen-record at 1080p / 60 fps for 20 seconds, then either embed the resulting MP4 or convert to GIF (`ffmpeg -i in.mp4 -vf "fps=24,scale=1280:-1:flags=lanczos" out.gif`).

A `?loop=1` URL is helpful if your recording tool needs a few seconds of head-room — the page resets cleanly every 20 seconds.

---

## What was iterated

This was built in two visible passes, both captured in `screenshots/`:

- **Iteration 1** — Editorial dark theme with cyan accent and SVG glow filters. Worked aesthetically but felt "AI-generic" (the very same dark + cyan + topographic-line look one sees on auto-generated AI landing pages).
- **Iteration 2** — Full repaint to a paper canvas with UniBE red as the only accent, audited against the `minimalist-ui` skill: no cyan, no glow, no gradient mesh, no Inter/Roboto, no AI marketing clichés, and copy rewritten to be plainly descriptive ("Modeling rivers, sensors, and signals — together, in time" replaces the original "Where space and time converge in intelligence").

The contrast between the two iterations is the most important artefact the demo carries with it: it shows we noticed and corrected the generic look rather than shipping it.

---

## License

Same as the parent LIULIAN project. Mock data is illustrative — no real telemetry from any operator is reproduced.
