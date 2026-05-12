# ADR 0010 — Hybrid shadcn + antd frontend stack

- **Status**: accepted
- **Decision date**: 2026-05-12

## Context

`liulian-web` derives from `neobanker-frontend-MVP-V3`, which is
**all-Ant-Design**. Ant Design 5 plus `@ant-design/x` (AI chat
primitives) plus `antd-style` (CSS-in-JS for antd) plus echarts-for-react.

The user's brand requires **editorial Swiss with warm-bone canvas +
UniBe red + Fraunces + Switzer**, which fights Ant Design's
enterprise-blue defaults.

But `@ant-design/x` is best-in-class for AI chat UI, and Ant Design's
`Table` component is best-in-class for high-density data.

## Decision

**Hybrid stack**. Use shadcn/ui primitives as the default; use Ant
Design ONLY for two surfaces:

1. **The chat sidebar** (on `/forecast`, `/agents/*`): `@ant-design/x`
   primitives, themed via `antd-style` + `ConfigProvider` to LIULIAN
   tokens.
2. **High-density data tables** in `/studio` where TanStack Table feels
   under-equipped (rare; we audit at M2).

## The brand override layer

```ts
// liulian-web/lib/antd-theme.ts
import { theme } from 'antd';
import { tokens } from '@liulian/design-tokens';

export const liulianAntdTheme = {
  algorithm: theme.defaultAlgorithm,
  token: {
    colorPrimary: tokens.colors.unibeRed,        // #E20613
    colorBgBase: tokens.colors.canvasWarm,       // #FBFBFA
    colorTextBase: tokens.colors.inkCharcoal,    // #131313
    fontFamily: tokens.fonts.body.join(', '),    // Switzer …
    fontFamilyCode: tokens.fonts.mono.join(', '),
    borderRadius: 10,
    borderRadiusLG: 14,
    wireframe: false,
  },
  components: {
    Button: { borderRadius: 6, fontWeight: 500 },
    Card: { borderRadiusLG: 10 },
    // …
  },
};
```

Applied via `<ConfigProvider theme={liulianAntdTheme}>` only on
component subtrees that need antd. Everything else uses shadcn +
Tailwind + brand tokens directly.

## Rationale

- **shadcn/ui for primary surfaces**: copy-not-install philosophy;
  pixel-level control matches editorial-Swiss brand discipline.
- **`@ant-design/x` for chat sidebar**: world-class AI chat
  primitives (typing indicators, message renderer, citation footnotes,
  tool-call inline cards). Writing this from scratch is weeks of
  work for marginal differentiation.
- **Brand override is mechanical**: ConfigProvider takes our tokens;
  the antd surface is recognisable as ours, not as antd-default.

## Alternatives considered

- **All-antd** (matching neobanker exactly): rejected; antd defaults
  fight editorial Swiss brand. Override layer would have to override
  too many tokens.
- **All-shadcn / hand-roll the chat**: ~2 weeks of work for the chat
  UI alone, with no incremental design benefit.
- **Mantine for everything**: more components than shadcn but still
  B2B-template-y; rejected on brand grounds.

## Consequences

- (+) Chat UI ready on Day 5 of sprint.
- (+) Brand-pure on primary surfaces (marketing, studio, forecast).
- (+) Code reuse from neobanker frontend possible for the chat layer.
- (−) Two component systems in the same app; theming discipline must
  hold via the antd override layer.
- (−) Bundle size larger than single-system; mitigated by tree-shaking
  + dynamic imports for the chat sidebar.

## Final stack tally

| Layer | Choice |
|---|---|
| App framework | Next.js 14 App Router |
| RPC | tRPC + openapi-typescript |
| Styling | Tailwind CSS + CSS variables + brand tokens |
| UI primitives | shadcn/ui (Radix under the hood) |
| Charts | echarts-for-react (custom theme matching brand) |
| Maps | MapLibre GL + swisstopo open tiles |
| Probabilistic | Plotly (rare; fan-charts where ECharts is awkward) |
| State (server) | TanStack Query |
| State (UI) | Zustand |
| Forms | react-hook-form + zod |
| Tables | TanStack Table (default); Ant Design Table (rare, high-density) |
| Chat sidebar | @ant-design/x via ConfigProvider override |
| Tile dashboards | react-mosaic-component (forked from neobanker) |
| Resizable panes | react-resizable-panels |
| MDX content | contentlayer |
| Animation | framer-motion |
| Icons | Phosphor Icons (Regular); Lucide fallback |
| Auth | Clerk Next.js |
| i18n | next-intl (en · zh-CN) |
| Tests | Vitest + Playwright |
| Storybook | yes (per neobanker pattern) |

## Cross-references

- `NEOBANKER_REUSE_MAP.md` §4 (the `liulian-web` fork plan)
- `PLATFORM_DESIGN.md` §2 (brand tokens)
- `PLATFORM_DESIGN.md` §4.3 (chat sidebar specification)
