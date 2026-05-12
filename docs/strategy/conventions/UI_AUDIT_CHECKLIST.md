---
title: LIULIAN UI Audit Checklist
status: convention (must pass on every PR touching frontend)
owner: Linlin Jia (jajupmochi)
created: 2026-05-12
sourced_from:
  - userSettings:impeccable           (shared design laws + absolute bans)
  - userSettings:minimalist-ui        (premium utilitarian minimalism protocol)
  - jajupmochi/claude-config          (workflow rules + bilingual rule)
  - liulian-python/.worktrees/gui-demo/docs/design-report.md  (the brand canon)
> **Language:** English | [中文](UI_AUDIT_CHECKLIST.zh.md) *(zh stub pending)*
---

# LIULIAN UI Audit Checklist

> Every PR that adds or modifies a visible surface (HTML/CSS/JSX) MUST
> pass every row below before merge. The reviewer ticks each box in
> the PR body. Any unchecked row blocks merge.

## A. Sacred rules (non-negotiable)

- [ ] **Code reuse vs visual originality kept clean.** No screenshot
      of the new surface should look like a neobanker / Refine.dev /
      shadcn-template screen with text swapped. (NEOBANKER_REUSE_MAP
      §0.1)
- [ ] **Bilingual rule honored** (claude-config bilingual-docs):
      every user-facing string is keyed via `next-intl`; en and zh
      both translated; switcher visible.

## B. Color

- [ ] OKLCH-defined tokens used (never raw hex outside
      `liulian-design-system`).
- [ ] No `#000000` / `#FFFFFF` in computed styles.
- [ ] Neutrals tinted toward brand hue (chroma 0.005–0.01).
- [ ] Color strategy declared on the page (Restrained / Committed /
      Full palette / Drenched).
- [ ] UniBe red `#E20613` used only on (a) brand accents, (b) status
      states (alert, elevated), (c) destructive confirmations. *Never*
      on backgrounds, buttons, hover states.

## C. Typography

- [ ] Body line length ≤ 75ch.
- [ ] Hierarchy ratio ≥ 1.25 between steps.
- [ ] Fonts: Fraunces (display) / Switzer (body) / JetBrains Mono
      (code, numbers). **NO** Inter / Roboto / Open Sans.
- [ ] Tabular numerals on numeric columns
      (`font-variant-numeric: tabular-nums`).
- [ ] Status uses *typography* (italic / weight) not chromatic pills,
      where possible.

## D. Layout

- [ ] Spacing varies for rhythm (not same padding everywhere).
- [ ] No nested cards.
- [ ] Containers used only when necessary.
- [ ] Generous section spacing (≥ 96px between major bands).
- [ ] Hairline borders (`1px solid var(--hairline)`); no resting drop
      shadows.

## E. Motion

- [ ] Animations on `transform` + `opacity` only (no layout-property
      animation).
- [ ] Ease-out exponential curves (quart / quint / expo). No bounce,
      no elastic.
- [ ] Scroll-entry uses `IntersectionObserver`, not scroll listeners.
- [ ] No autoplay video on hero unless muted + loop + ≤ 4 MB.

## F. Absolute bans (match-and-refuse; auto-block)

- [ ] No **side-stripe borders** > 1px as colored accent on cards,
      list items, callouts, alerts.
- [ ] No **gradient text** (`background-clip: text` + gradient bg).
- [ ] No **glassmorphism** as decoration (rare and purposeful only).
- [ ] No **hero-metric template** (big number + label + sparkline
      ring).
- [ ] No **identical card grids** (same-sized cards in a long list,
      icon + heading + text, repeated).
- [ ] No **modal as first thought** (exhaust inline / progressive
      alternatives first).
- [ ] No **`rounded-full`** on cards or primary buttons.
- [ ] No **emojis** in markup, code, alt text.
- [ ] No **AI-cliché copy** ("Elevate" / "Seamless" / "Unleash" /
      "Next-Gen" / "Game-changer" / "Delve").
- [ ] No **em dashes** (`—`) in user-facing copy. Use commas, colons,
      semicolons, periods, or parentheses.
- [ ] No **placeholder names** ("John Doe" / "Acme Corp" / "Lorem
      Ipsum"). Use real station codes from `manifests/`.

## G. Brand canon checks (LIULIAN-specific)

- [ ] Canvas color: `--canvas-warm` (`#FBFBFA`) or `--surface-pure`
      (`#FFFFFF`).
- [ ] Body text: `--ink-charcoal` (`#131313`); never absolute black.
- [ ] Card border: 1px solid `--hairline` (`#EAEAEA`), 10px radius.
- [ ] At most one UniBe red element visible per viewport on most
      pages. On the forecast canvas, the cap is two: active station
      marker + threshold-crossing marker.

## H. Distinctive moves (positive checks — at least one per page)

Pick at least one signature move on every meaningful page:

- [ ] Fraunces wonk-1 italic accent on a single character
- [ ] Scientific running header (date · run-coordinates)
- [ ] Scientific notation default for numbers outside `[0.01, 1000]`
- [ ] Bento-grid asymmetry (mixed tile sizes; no two cards identical
      and adjacent)
- [ ] Editorial two-column asymmetry (60% table / 40% explainer)
- [ ] Domain-true Easter egg (used at most once per page)

## I. The four tests (run by the reviewer at end of PR)

- [ ] **AI-slop test**: would a viewer say "AI made this" without
      doubt? If yes → reject.
- [ ] **Category-reflex test**: could the palette be guessed from
      "hydrology" / "TS forecasting" alone? If yes → rework color
      strategy.
- [ ] **Neobanker-copy test**: does the page look like a neobanker
      screen with text swapped? If yes → restyle.
- [ ] **gui-demo cross-check**: warm-bone canvas + UniBe red as sole
      spot + editorial typography still load-bearing? If not, justify.

## J. Accessibility (delegated to automated tools where possible)

- [ ] Lighthouse a11y score ≥ 95 on the page being changed
      (run `make audit-a11y`).
- [ ] All interactive elements keyboard-focusable.
- [ ] Color contrast AA on text (≥ 4.5:1 for body, ≥ 3:1 for large).

## K. Performance (delegated)

- [ ] LCP ≤ 2.5s on M1-MacBook over throttled 4G (run
      `make audit-perf`).
- [ ] No layout shift on font swap (FOUT prevented via
      `font-display: swap` + size-adjust matched).

---

## Reviewer note format

In the PR body, the reviewer pastes this block with each box checked
or with a short justification for any unchecked box:

```markdown
## UI audit (UI_AUDIT_CHECKLIST.md)

A. Sacred rules     ✓ originality kept (no neobanker lookalike); bilingual ✓
B. Color            ✓ Restrained; UniBe red on threshold marker only
C. Typography       ✓ Fraunces 32/24/16; tabular nums on metrics
D. Layout           ✓ varied spacing; hairline borders
E. Motion           ✓ opacity + transform only
F. Absolute bans    ✓
G. Brand canon      ✓
H. Distinctive      ✓ wonk-1 italic on hero `U` + bento asymmetry
I. Four tests       ✓ all pass
J. A11y             ✓ Lighthouse 97
K. Perf             ✓ LCP 1.8s

Reviewer: <name>
Date: <iso8601>
Branch: <branch>
```

If any row is `~` (partial) or `✗`, the PR cannot merge until
addressed.

---

*Source skills: `impeccable` (shared design laws), `minimalist-ui`
(premium utilitarian protocol), `claude-config:bilingual-docs`,
`gui-demo design-report.md` (brand canon).*
