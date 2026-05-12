# ADR 0007 — Reject Refine.dev as a UI source; hand-roll `/studio`

- **Status**: accepted (revised after user feedback 2026-05-12)
- **Decision date**: 2026-05-12 (revision of earlier draft adopting Refine.dev)

## Context

The original iteration of the platform plan proposed using Refine.dev
for `/studio` CRUD layers (experiments, models, datasets, users).
Refine.dev offers:

- Headless framework with multi-UI support (Mantine / Ant Design / MUI
  / Chakra / shadcn).
- Data-provider abstraction (REST / GraphQL / Supabase / Hasura).
- Built-in `useTable` / `useForm` / `useSelect` hooks with sort + filter
  + pagination + audit.
- ~2 weeks of CRUD scaffolding saved.

User feedback (2026-05-12): *"Refine.dev 设计太普通了, 没特色, 不抓眼球"*.
After inspection, agreed: Refine.dev's templates look like every other
2020s admin panel (same gray cards, same Inter font, same blue primary).
That violates the visual originality rule in
`NEOBANKER_REUSE_MAP.md §0.1`.

## Decision

**Reject Refine.dev as a UI source.**

`/studio` is hand-rolled with a **Linear-meets-Bloomberg-terminal
editorial Swiss** aesthetic, using:

- shadcn/ui primitives for accessible components.
- TanStack Query + TanStack Table for headless data + table primitives.
- `cmdk` library for the command palette (primary nav).
- `@liulian/design-tokens` brand tokens (Fraunces / Switzer / JetBrains
  Mono / UniBe red on warm bone canvas).

We may revisit Refine.dev's *data-provider abstraction* at M3 ONLY if
per-tenant RBAC + audit-log boilerplate becomes painful. Never for UI.

## Rationale

- **Identity is the brand**: LIULIAN's positioning is vertical
  scientific-AI for regulated sectors. A generic admin look-and-feel
  contradicts the positioning.
- **Refine.dev visual cost**: every screenshot looks like Refine.dev;
  brand differentiation is zero.
- **TanStack Query/Table covers the headless need**: ~200 LOC of CRUD
  hooks gives us the same `useList` / `useGetOne` / `useCreate` /
  `useUpdate` / `useDelete` ergonomics, on top of our own design.

## Time delta

- Hand-rolled `/studio`: ~2 to 3 weeks beyond Refine.dev templates.
- Pay-back: a brand-coherent surface that compounds across every
  future pitch, blog post, and demo.

## Counter-investments worth tracking at M2

The same 2 to 3 weeks could go to (per `AUDIT_REPORT_2026-05-12.md §B.4`):

- (a) Deeper Chronos vs Time-Series-Library benchmark.
- (b) A swisstopo / Eawag pilot conversation.
- (c) A public blog post on a SwissRiver dataset insight.

We pick the studio investment because brand consistency is the
*positioning* call; benchmarks and pilots are individually higher-
leverage but compound less.

## Distinctive moves applied in `/studio`

Per `NEOBANKER_REUSE_MAP.md §14.3` and `PLATFORM_DESIGN.md §5`:

1. Cmd+K palette as primary nav (Linear lineage; restyled editorially).
2. 56px collapsed sidebar (opposite of SaaS 240-280px dogma).
3. Dense monotype tables (JetBrains Mono numbers, j/k row nav).
4. Asymmetric two-column list pages (60% table + 40% Fraunces italic
   explainer).
5. Status as typography (italic "running", bold-roman "failed").
6. Scientific running header on every page.
7. Numbers default to scientific notation outside [0.01, 1000].
8. 6×6 px UniBe red leading dot on selected row (NOT a side stripe).
9. One Easter egg: `流` ripple on threshold crossing.

## Alternatives considered

- **Refine.dev with custom theme override**: still bears Refine.dev's
  structural fingerprint; brand differentiation marginal.
- **shadcn-admin starter**: marginally better than Refine.dev but still
  recognisable as a template.
- **Mantine Admin**: stronger components but still B2B-generic look.
- **Full custom on shadcn primitives**: chosen.

## Cross-references

- `NEOBANKER_REUSE_MAP.md` §14.3 (the full direction)
- `PLATFORM_DESIGN.md` §5 (`/studio` design)
- `AUDIT_REPORT_2026-05-12.md` §B.4 (the trade-off audit)
