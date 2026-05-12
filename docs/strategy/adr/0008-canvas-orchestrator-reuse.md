# ADR 0008 — Reuse `neobanker-frontend/components/assistant/Canvas*` as the BI canvas foundation

- **Status**: accepted
- **Decision date**: 2026-05-12

## Context

LIULIAN's `/forecast` BI canvas needs:

- Drag-and-resize tile layout (the report builder).
- A per-tile config sidebar.
- A canvas-level toolbar (save / share / export).
- A session sidebar (history of canvas states).
- Smart switcher / dock for multi-view layouts.
- Status bar.
- Error / degradation banners.

Building these from scratch would cost ~2-3 weeks of focused React
work. Meanwhile, `neobanker-frontend-MVP-V3/components/assistant/`
contains exactly these primitives in production form.

## Decision

Fork the `assistant/` directory verbatim, restyle pixel-by-pixel per
LIULIAN brand tokens, swap bank-domain widgets for forecasting
widgets. The file structure is preserved; the renderings are wholly
new.

### File-by-file map

| Source | LIULIAN target | Action |
|---|---|---|
| `CanvasOrchestrator.tsx` | `app/forecast/components/CanvasOrchestrator.tsx` | rename refs only |
| `DynamicCanvas.tsx` | same | keep verbatim (logic) |
| `CanvasToolbar.tsx` | same | minor brand adjustments |
| `ReportBuilder.tsx` | same | keep verbatim |
| `WidgetConfigPanel.tsx` | same | keep verbatim |
| `widgets/` | rewrite all individual widgets | bank widgets → forecasting widgets (the 8 panels from `PLATFORM_DESIGN.md §4`) |
| `SessionSidebar.tsx` | same | keep |
| `SmartSwitcherDock.tsx` | same | keep |
| `StatusBar.tsx` | same | keep |
| `HeaderBar.tsx` | same | brand adjustments |
| `BankSelector.tsx` | `StationSelector.tsx` | rewrite per-entity logic |
| `DegradationBanner.tsx` | same | keep |
| `ToastContainer.tsx` | same | keep |
| `data/` | rewrite | API endpoints differ |
| `hooks/` | adapt | useCanvas / useWidget / useAgent generic mostly |
| `types.ts` | adapt | Bank → Station + add Forecast / Run types |

## Rationale

- **Saves ~2-3 weeks** of bespoke React work.
- **Verified-good code path** running in neobanker production.
- **Brand-pure**: per `NEOBANKER_REUSE_MAP §0.1`, every rendered pixel
  is restyled to LIULIAN tokens; the reuse is structural, not visual.
- **Operator continuity**: same component family across products.

## Visual-originality discipline

Every fork-then-edit cycle ends with a screenshot diff vs the source
neobanker screen. If the LIULIAN screen is visually mistakable for the
neobanker screen with text swapped, restyle further. This is the
**neobanker-copy test** in `conventions/UI_AUDIT_CHECKLIST.md §I`.

## Alternatives considered

- **`react-mosaic-component` alone** (the underlying library
  neobanker uses): we still adopt it. The `assistant/` family adds
  composed UX (config panel, save state, toolbar) on top.
- **`react-grid-layout`**: dominant in npm; weaker for resizable
  multi-pane needs we have.
- **Build from scratch**: rejected (time).

## Consequences

- (+) BI canvas surface is ready to wire to real data on sprint Day 3.
- (+) The report builder is "free" (functionality-wise).
- (−) Bank-domain assumptions hide in some hooks; Day 3 spike will
  surface them.
- (−) Visual originality must be enforced per-pixel; one careless PR
  could regress to "neobanker lookalike".

## Cross-references

- `NEOBANKER_REUSE_MAP.md` §14.2 (the file map)
- `PLATFORM_DESIGN.md` §4 (the 8 BI panels)
- `conventions/UI_AUDIT_CHECKLIST.md` §I (4 PR tests)
