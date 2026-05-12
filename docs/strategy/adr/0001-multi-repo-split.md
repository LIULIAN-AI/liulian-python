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

## Decision

Adopt a **federated multi-repo** architecture with eight repos:

1. `liulian-python` (this repo; research core, pip-installable)
2. `liulian-api` (FastAPI gateway, new)
3. `liulian-agent` (FastAPI agent service, fork of neobanker-agent)
4. `liulian-ingest` (crawler / ingest, fork of neobanker-crawler)
5. `liulian-web` (Next.js BI canvas + studio + marketing,
   fork of neobanker-frontend-MVP-V3)
6. `liulian-mobile` (Expo React Native companion, greenfield)
7. `liulian-design-system` (tokens npm package, seeded from gui-demo CSS)
8. `liulian-ops` (deploy CLI + Helm + Terraform + reusable GH Actions,
   fork of neoctl)

Cross-repo contracts are narrow and versioned (OpenAPI artefact from
`liulian-api`; `@liulian/design-tokens` npm package; HTTP between
services).

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
