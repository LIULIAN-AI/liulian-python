# ADR 0001 — Multi-repo split (eight repos, federated)

- **Status**: accepted (iteration 2 of platform plan, 2026-05-12)
- **Decision date**: 2026-05-12
- **Decision owner**: Linlin Jia (jajupmochi)
- **Supersedes**: an earlier draft proposing a monorepo with workspace tooling.
- **Audit status**: PASS with caveats (see `AUDIT_REPORT_2026-05-12.md §B.1`).

## Context

LIULIAN is evolving from a single Python research package into a
production-grade platform with backend, frontend, mobile, agent, ingest,
design-system, and ops surfaces. The repo strategy must serve:

- A solo developer with AI-assisted velocity.
- Open-source contributors who land on a single surface.
- Genuinely divergent release cadences (mobile EAS-build, web Vercel,
  Python PyPI).
- A funding narrative that reads as "deployable platform", not
  "academic monorepo".

## Decision (revised 2026-05-12 evening after Sprint Day 1)

Adopt a **federated 7-repo** architecture. Initial draft proposed 8;
mid-sprint user feedback challenged the count, and after analysis we
landed at **7** by merging only what truly co-shares category, not
just by repo-count target:

1. `liulian-python` (this repo; research core, pip-installable)
2. `liulian-api` (FastAPI gateway, greenfield)
3. `liulian-agent` (FastAPI agent service, fork of neobanker-agent)
4. **`liulian-ingest`** (crawler / runtime data-plumbing service,
   fork of neobanker-crawler) — **separate from `liulian-ops`**
   because it is a *runtime service* with its own failure modes
   (external API limits, MinIO writes, scheduler state), not infra
   tooling.
5. `liulian-web` (Next.js BI canvas + studio + marketing,
   fork of neobanker-frontend-MVP-V3)
6. `liulian-mobile` (Expo React Native companion, greenfield)
7. **`liulian-design-system`** (`@liulian/design-tokens` npm package,
   greenfield seeded from gui-demo CSS) — **separate from
   `liulian-web`** because it has multiple future consumers (mobile,
   marketing slides, Figma library, email templates) and warrants
   standalone semver.
8. `liulian-ops` (deploy CLI `liulianctl` + Helm + Terraform +
   reusable GH Actions + **devcontainer** subfolder, fork of neoctl) —
   `liulian-dev-env` is **merged in** because devcontainer config is
   pure devops/setup and shares ops' lifecycle.

Cross-repo contracts are narrow and versioned (OpenAPI artefact from
`liulian-api`; `@liulian/design-tokens` npm package; HTTP between
services).

### Mid-sprint mergers / un-mergers (decision audit)

| Action | Verdict | Reasoning |
|---|---|---|
| Merge ingest → ops | reverted | runtime service ≠ infra tooling; different lifecycles, different failure modes |
| Merge design-system → web | reverted | multi-future-consumer config asset; warrants own semver |
| Merge dev-env → ops | kept | devcontainer + docker-compose are pure devops setup; same lifecycle as ops itself |

Lesson recorded: do not collapse repos purely to reduce count;
collapse only when the merged units genuinely co-share category (same
lifecycle, same failure modes, same operator).

## Defensible rationale

- **Operator muscle memory**: the user already runs neobanker with
  multi-repo + `neoctl`. Reusing that mental model is genuinely lower
  cognitive cost.
- **Smaller per-repo surface for OSS contributors**: clearer onboarding
  for each persona.
- **Divergent release cadences**: mobile (EAS / store review), web
  (Vercel push), Python core (PyPI tag), API (Docker image + Helm) ship
  on different rhythms; multi-repo makes this honest.
- **Versioned cross-repo type sharing via OpenAPI artefact**: cleaner
  than shared source files in a monorepo for our case (same idea as
  protobuf).

## Rationale we explicitly DO NOT claim

- "CI is faster than monorepo": unmeasured. Turborepo with caching
  would likely match per-repo CI for JS apps.
- "Saves N days in the sprint": a priori guess.

## Alternatives considered

- **Single repo + Turborepo**: viable; would simplify cross-repo
  refactors. Rejected for operator-muscle-memory + OSS-onboarding
  reasons.
- **Hybrid (Python multi-repo + JS Turborepo monorepo)**: a future
  reversal target if cross-repo coordination becomes painful.

## Consequences

- (+) Per-repo CI is small and fast.
- (+) OSS contributors clone one surface.
- (+) Release cadences are honest.
- (−) Cross-cutting features require coordinated PRs.
- (−) Per-repo `.github/workflows/` ceremony is more files (mitigated
  by reusable workflows in `liulian-ops`).

## Reversibility

**One-week migration cost** to collapse the three JS repos
(`liulian-web` + `liulian-mobile` + `liulian-design-system`) into a
Turborepo monorepo, keeping the five Python repos separate. Trigger:
opening 4 PRs to ship one feature becomes routine.

## Cross-references

- `PLATFORM_BLUEPRINT.md` §4 (architecture summary)
- `NEOBANKER_REUSE_MAP.md` §1 (repo-to-repo fork map)
- `AUDIT_REPORT_2026-05-12.md` §B.1 (audit findings folded into §4.2)
