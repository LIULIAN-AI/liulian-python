# LIULIAN — GUI Demo

A self-contained, frontend-only product demo of the **LIULIAN spatiotemporal research platform** (UniBE × ETH hydrology workflow). The page is built as a **four-screen product tour** — Data → Train → Inference → Insight — and is meant to be screen-recorded once as a 20-second video or animated GIF and dropped into a presentation slide.

```
.worktrees/gui-demo/
├── index.html                  page structure (top nav + 4 screens)
├── styles/main.css             design system + components
├── scripts/
│   ├── mock-data.js            illustrative datasets (no real telemetry)
│   ├── charts.js               SVG renderers (map / loss / forecast / spark)
│   └── scene-controller.js     20-second timeline orchestrator
├── screenshots/
│   ├── final_01_hero.png       hero (t = 1.5 s)
│   ├── final_02_data.png       data screen
│   ├── final_03_train.png      train screen
│   ├── final_04_inference.png  inference screen
│   ├── final_05_insight.png    insight screen
│   ├── final_06_finale.png     insight + closing stamp
│   ├── iter01_*.png            rejected dark/cyan design (kept for log)
│   ├── iter02_*.png            rejected single-page bento (kept for log)
│   └── iter03_*.png            this revision in progress
├── docs/
│   ├── screens-design.md       multi-screen design specification
│   └── design-report.md        end-to-end design report
└── README.md                   this file
```

The branch (`feat/gui-demo`) is intentionally an **orphan branch** so the demo ships independently of the main research codebase.

---

## The four screens

The demo is structured exactly the way real research/data tools are structured: a top tab strip switches between purpose-built workspaces. Reference products in parentheses.

### Data — *Snowflake catalog × Hex Magic*

```
[ datasets sidebar ] [ manifest.yaml + data preview table ] [ schema · integrity · coverage + mini map ]
```

What the screen says: *"We have a real data substrate. The schema is verified, the manifest is hashed, 2 138 of 2 143 stations are streaming."*

### Train — *Weights & Biases*

```
[ run header · live loss curve · 4 metric cards · config.yaml ]   [ filterable runs table ]
```

What the screen says: *"We train models. The active Transformer-E run is at epoch 56/80, val MSE 0.0418 — 18.6 % below baseline. Five other architectures are ranked behind it in the runs table."*

### Inference — *Replicate / Hugging Face Inference Endpoints*

```
[ POST  api.liulian.ch/v1/forecast    p50 2.3 ms · p99 8 ms · 1 240 r/s ]
[ inputs ]  [ output forecast chart with 95 % CI band, threshold, marker ]
[ python code snippet ]
```

What the screen says: *"You can call this from Python. Latency is single-digit ms. The output already includes a threshold-crossing point and a probability."*

### Insight — *Hex Magic / Julius AI / Cursor chat*

The Insight tab is **scrollable** and contains three stacked sections:

1. **Conversation** (top, full viewport) — sessions sidebar, chat with user prompt → streamed agent response with embedded mini forecast chart + colour-coded source chips, context + reasoning trail right rail.
2. **River network · live** (mid, full-width panel) — a credible Switzerland map (country outline, six major lakes, seven rivers — Rhine, Rhône, Aare, Reuss, Limmat, Inn, Ticino — and 30 stations with realistic placement). The at-risk Rhein-Basel station is pulse-marked, with the upstream Aare contributors (`AAR-BE`, `AAR-BG`, `AAR-TH`) highlighted. The right rail of the panel shows an `Affected basin` callout (drainage area, current vs T+18 h, flood threshold, probability) and a legend.
3. **Upstream contributors** (bottom row) — five mini cards, one per major contributor station, each showing current m³/s, predicted at T+18 h, share of basin discharge, and a micro-bar visualising the share.

During the live demo the page auto-scrolls from the chat down to the river-network section at t = 17.4 s — that's the credibility moment.

What the screen says: *"Ask a question, get a sentence a hydrologist can act on. Then look at where the answer comes from on a real map of the basins."*

---

## The 20-second tour

| t | Beat |
|---|------|
| 0.0 s  | Hero overlay: wordmark `LIULIAN` reveals letter-by-letter; the `U` is red italic. Pretitle reads *"Liquid Intelligence · Unified Logic · Interactive Adaptive Networks"* — the canonical LIULIAN slogan. |
| 1.0 s  | Tagline: *"Modeling rivers, sensors, signals. Together, in time."* |
| 1.4 s  | Loader bar fills under "2 143 stations · 12 models". |
| 2.4 s  | Hero fades. Top tab strip becomes visible. |
| 2.5 s  | **Tab → Data.** Datasets list cascades; manifest types in; data table rows tick in. |
| 6.0 s  | **Tab → Train.** Loss chart axis appears, train line draws over 2 s, then val (dashed); metric cards count up; runs table cascades; config.yaml types in. |
| 10.0 s | **Tab → Inference.** Endpoint card visible; "Run inference" button flashes red; output chart materialises; latency badge counts 0 → 8 ms. |
| 13.8 s | **Tab → Insight.** User question appears; typing dots; agent response streams in with red emphasis on `Rhein-Basel`, `2 400 m³/s`, `May 8 ~14:00 UTC`, `87 %`, and `elevated`. |
| 16.6 s | Mini forecast chart slides into the agent bubble. |
| 17.4 s | Page auto-scrolls down to the **River network · live** section. The Switzerland map paints country outline, six major lakes, seven rivers; 30 stations cascade in; the at-risk Rhein-Basel marker pulses with a halo. The right-rail callout populates the basin numbers. |
| 18.4 s | Reasoning trail in the right rail marks each step done in sequence. |
| 18.8 s | A single line slides in at the bottom: *"From manifest to forecast, in 24 hours."* |
| 20.0 s | Hold. `?loop=1` reloads. |

The four tab activations are the story beats. An investor watching once still gets the message just from the underline jumping across the tab strip.

---

## Why this revision exists

This branch went through three iterations. All are kept in `screenshots/` for reference.

- **iter01** — dark canvas + cyan + violet + amber + gradient mesh + topographic-line backgrounds. Visually striking, but read as the same template every AI-generated mockup uses on the public internet. Rejected.
- **iter02** — light paper canvas, UniBE red, single-page bento dashboard with 7 cards. Less AI-generic, but the bento pattern itself is a tell — investors see "yet another AI dashboard" in the layout. Rejected.
- **iter03** *(this revision)* — split into four screens, each modelled after a real industry tool. Tab-switched. The product *is* the ornament; almost no decoration remains. The iter01 and iter02 screenshots stay in the repo as part of the audit log.

Specific patterns deliberately removed across the journey:

- ☑ No bento grid / single-page dashboard.
- ☑ No "Agent" suffix in visible UI titles.
- ☑ No corner brackets, rotated side marks, or scan-line effects.
- ☑ No all-caps "STREAMING" / "ELEVATED" / "LIVE" pills with extreme letter-spacing.
- ☑ No mono eyebrow above every section.
- ☑ No display-serif title above every working pane (Fraunces is used only for the hero wordmark and the brand name in the topbar).
- ☑ No `Inter` / `Roboto` / `Open Sans` / `Lucide`.
- ☑ No glow filters, no purple gradient, no neon.
- ☑ No emoji.
- ☑ No copy clichés ("Production Ready", "End-to-End Orchestration", "Spatiotemporal Intelligence", "Where space and time converge…").

Words actually used on the page: `Run`, `Endpoint`, `Latency`, `Throughput`, `Schema`, `Source`, `Send`, `Step`, `Epoch`, `val MSE`. These are the actual nouns of working ML tools.

---

## Running it locally

```bash
# from the worktree root
python3 -m http.server 8765 --bind 127.0.0.1
# then open http://127.0.0.1:8765/index.html
```

Pure HTML / CSS / JS. No build step, no node_modules, no API calls. Total payload is well under 100 KB plus the Google Fonts and Fontshare CDN requests.

### URL parameters

| Param                | Effect                                                              |
|----------------------|---------------------------------------------------------------------|
| `?freeze=hero`       | Freeze on the hero wordmark.                                        |
| `?freeze=data`       | Freeze on the Data screen.                                          |
| `?freeze=train`      | Freeze on the Train screen.                                         |
| `?freeze=inference`  | Freeze on the Inference screen.                                     |
| `?freeze=insight`    | Freeze on the Insight chat (with response complete).                |
| `?freeze=map`        | Freeze on the river-network map (Insight scrolled to map section).  |
| `?freeze=finale`     | Freeze on map + closing stamp visible.                              |
| `?speed=2`           | Run the 20-second timeline at 2× speed.                             |
| `?loop=1`            | Auto-reload at t = 20 s.                                            |

---

## Recording

For a slide deck, screen-record at 1920×1080 / 60 fps for 20 seconds, then either embed the resulting MP4 or convert to GIF:

```bash
ffmpeg -i in.mp4 -vf "fps=24,scale=1280:-1:flags=lanczos" out.gif
```

`?loop=1` is helpful if your recording tool needs head-room — the page resets cleanly every 20 seconds.

---

## License

Same as the parent LIULIAN project. All on-screen data is illustrative; no real operator telemetry is reproduced.
