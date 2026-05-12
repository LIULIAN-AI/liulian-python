# ADR 0004 — UniBe red `#E20613` as the brand anchor

- **Status**: accepted (carries forward from gui-demo iteration 2)
- **Decision date**: 2026-05-05 (gui-demo iter 2); restated 2026-05-12.

## Context

LIULIAN needs a distinctive brand identity that:

- Doesn't fall into the AI-default cyan + violet + amber + black
  pattern.
- Doesn't fall into the category reflex for hydrology (glacier blue)
  or for ML platforms (navy + gold).
- Has real backing in the platform's origin (University of Bern PRG).
- Plays nicely with a warm-bone editorial Swiss palette.

## Decision

The brand is anchored on **University of Bern red**: `#E20613`. Used
*deliberately and rarely* as the sole spot color across all LIULIAN
surfaces.

## Rationale

- **Defies category reflex**: hydrology-default = glacier blue.
  UniBe red is unexpected and forces a second look.
- **Has real institutional grounding**: traces back to the
  University of Bern corporate identity (`unibeCols` R package by
  CTU-Bern).
- **Plays well with editorial Swiss palette**: high-chroma red on a
  warm-bone canvas reads scientific (Müller-Brockmann lineage), not
  political or alarmist (because the warm-bone reduces emotional
  temperature).
- **Tested in gui-demo iteration 2**: against iteration 1's dark + cyan
  AI-generic; iteration 2 with UniBe red on warm paper was the
  unanimous "yes" both internally and per `impeccable` skill audit.

## Usage rules (binding)

UniBe red appears on, and only on:

1. Wordmark `U` (the italic WONK 1 accent character).
2. Active station marker on map.
3. Predicted-forecast line in chart.
4. CI band color in forecast fan.
5. Severity-elevated pill.
6. Threshold-crossing marker.
7. Leading dot (6×6 px) on selected row in tables.
8. Focus ring for inputs.
9. Destructive-action confirmation button background.

NOT used on:

- Buttons (default `--ink-charcoal` fill).
- Hover states (just deepen the hairline border).
- Backgrounds (warm-bone or pure white only).
- Plain text (charcoal).
- Decoration / accents that aren't carrying state.

**Cap**: maximum 2 visible red elements per viewport on most pages;
3 only on the alert canvas.

## Variants

```css
--unibe-red       : #E20613   (primary)
--unibe-red-tint  : #FDEBEC   (pill backgrounds, CI band fills)
--unibe-red-deep  : #B00510   (hover, focus, destructive-confirm)
```

## Secondary palette (chart series ONLY, never UI)

For multi-model overlay or multi-series chart needs:

```css
--unibe-ocean     : ~#0066B3
--unibe-green     : ~#509A39
--unibe-apricot   : ~#E6863A
```

These are inherited from the official UniBe corporate palette
(`unibeCols`). They appear as chart series colors only; never as button
or surface backgrounds.

## Alternatives considered

- **Default navy + gold (finance ML reflex)**: too generic; identical
  to ten other platforms.
- **Glacier blue (hydrology reflex)**: too on-the-nose; reads as
  domain-cliché.
- **Black + cyan (AI-default)**: rejected in gui-demo iteration 1
  audit.
- **No spot color (pure neutrals)**: drains brand energy; rejected.

## Consequences

- (+) Distinctive at first glance; passes category-reflex test.
- (+) Institutional grounding adds narrative weight.
- (+) Coherent with editorial Swiss type + warm paper canvas.
- (−) Red is high-attention; the discipline of "2 elements per viewport
  max" must be enforced via UI audit checklist.

## Cross-references

- `.worktrees/gui-demo/docs/design-report.md` (origin story,
  iteration 1 vs 2)
- `PLATFORM_BLUEPRINT.md` §11.1 (brand anchor in architecture context)
- `PLATFORM_DESIGN.md` §2 (full token set)
- `conventions/UI_AUDIT_CHECKLIST.md` §B / §G (per-PR enforcement)
- `github.com/CTU-Bern/unibeCols` (source of secondary palette)
