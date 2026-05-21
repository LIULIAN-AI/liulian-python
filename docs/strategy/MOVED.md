# `docs/strategy/` has moved — index

`liulian-python` is the ML core. The federation-level docs that used to
live in `docs/strategy/` are now split across two destinations:

1. **`liulian-docs`** — the *federation hub* repo, the canonical home
   for platform-level docs (blueprint, ADRs, sprint history, agile
   templates, user stories, advisor briefings).
2. **Per-domain sibling repos** — UI / design / dev-env / ops docs
   live with the repo that owns them.

## Full relocation table

| What | New home |
|---|---|
| Platform blueprint (federation overview) | [`liulian-docs/docs/strategy/PLATFORM_BLUEPRINT.md`](../../../liulian-docs/docs/strategy/PLATFORM_BLUEPRINT.md) ([zh](../../../liulian-docs/docs/strategy/PLATFORM_BLUEPRINT.zh.md)) |
| Neobanker reuse map | [`liulian-docs/docs/strategy/NEOBANKER_REUSE_MAP.md`](../../../liulian-docs/docs/strategy/NEOBANKER_REUSE_MAP.md) ([zh](../../../liulian-docs/docs/strategy/NEOBANKER_REUSE_MAP.zh.md)) |
| One-week sprint plan | [`liulian-docs/docs/strategy/ONE_WEEK_SPRINT.md`](../../../liulian-docs/docs/strategy/ONE_WEEK_SPRINT.md) ([zh](../../../liulian-docs/docs/strategy/ONE_WEEK_SPRINT.zh.md)) |
| Audit report 2026-05-12 | [`liulian-docs/docs/strategy/AUDIT_REPORT_2026-05-12.md`](../../../liulian-docs/docs/strategy/AUDIT_REPORT_2026-05-12.md) ([zh](../../../liulian-docs/docs/strategy/AUDIT_REPORT_2026-05-12.zh.md)) |
| Advisor briefing 2026-05-19 | [`liulian-docs/docs/advisor-report-2026-05-19/`](../../../liulian-docs/docs/advisor-report-2026-05-19/) |
| All ADRs (0001–0011) | [`liulian-docs/docs/strategy/adr/`](../../../liulian-docs/docs/strategy/adr/) |
| Agile essentials & templates | [`liulian-docs/docs/agile/essentials.md`](../../../liulian-docs/docs/agile/essentials.md) |
| User stories (2026-05-21) | [`liulian-docs/docs/user-stories/`](../../../liulian-docs/docs/user-stories/) |
| BI canvas / panels spec | [`liulian-design-system/docs/PLATFORM_DESIGN.md`](../../../liulian-design-system/docs/PLATFORM_DESIGN.md) |
| 12-platform reference design memory bank | [`liulian-design-system/docs/REFERENCE_DESIGNS.md`](../../../liulian-design-system/docs/REFERENCE_DESIGNS.md) |
| External references index | [`liulian-design-system/docs/REFERENCES.md`](../../../liulian-design-system/docs/REFERENCES.md) |
| UI audit checklist | [`liulian-design-system/docs/conventions/UI_AUDIT_CHECKLIST.md`](../../../liulian-design-system/docs/conventions/UI_AUDIT_CHECKLIST.md) |
| Forecast redesign rounds R1–R5 | [`liulian-web/docs/redesign-2026-05-20/`](../../../liulian-web/docs/redesign-2026-05-20/) |
| Forecast dashboard rounds R6–R8 | [`liulian-web/docs/redesign-2026-05-21/`](../../../liulian-web/docs/redesign-2026-05-21/) |
| Forecast redesign spec (R5) | [`liulian-web/docs/specs/2026-05-20-forecast-redesign-design.md`](../../../liulian-web/docs/specs/2026-05-20-forecast-redesign-design.md) |
| Forecast dashboard spec (R8) | [`liulian-web/docs/specs/2026-05-21-forecast-dashboard-design.md`](../../../liulian-web/docs/specs/2026-05-21-forecast-dashboard-design.md) |
| Local 3-tier stack guide | [`liulian-dev-env/docs/LOCAL_STACK_2026-05-19.md`](../../../liulian-dev-env/docs/LOCAL_STACK_2026-05-19.md) |
| UBELIX cluster tiers reference | [`liulian-dev-env/docs/ubelix-cluster-tiers.md`](../../../liulian-dev-env/docs/ubelix-cluster-tiers.md) |
| UBELIX cost ledger | [`liulian-ops/docs/ubelix-cost-ledger.md`](../../../liulian-ops/docs/ubelix-cost-ledger.md) |

## What stays in `liulian-python`

Only docs about the ML library itself:

- `docs/index.md`, `docs/datasets.md`, `docs/manifest_spec.md`,
  `docs/adapter_guide.md`, `docs/forecasting_pipeline.md`,
  `docs/search_spaces.md`, `docs/entity_identifiers.md`,
  `docs/tsl_comparison.md`, `docs/training_comparison.md`,
  `docs/contributing.md`
- `docs/models/`, `docs/research/`
- `docs/superpowers/specs/2026-04-10-entity-identifier-forecasting-design.md`
- `docs/superpowers/specs/2026-04-14-entity-id-deep-audit-design.md`

## Why we restructured

The seed repo (`liulian-python`) accumulated docs that weren't about
Python at all — design rounds, ADRs touching all repos, advisor
briefings, sprint history. A reader who came to the seed repo expecting
"how do I write an adapter?" had to wade past all of that.

Splitting along the federation seam — federation hub + per-repo
domain docs — restores the principle that each repo's `docs/`
describes what that repo does.
