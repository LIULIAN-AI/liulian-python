# ADR 0011 — `liulian-dev-env` is its own repo, separate from `liulian-ops`

- **Status**: accepted (revises ADR 0001 §Mid-sprint mergers)
- **Decision date**: 2026-05-13
- **Decision owner**: Linlin Jia
- **Audit context**: see `AUDIT_REPORT_2026-05-12.md` §B.1 for the
  "category-not-count" rule that this ADR operationalises.

## Context

Iteration 3 (ADR 0001 mid-sprint amendments) merged `liulian-dev-env`
into `liulian-ops/devcontainer/` to reduce repo count. User feedback
2026-05-13 reframed the question correctly: *dev-env* (one-click
developer environment) and *ops* (production deploy + IaC) are
fundamentally different categories with different audiences, lifecycles,
secret-handling rules, and failure costs. Repo count is not a sufficient
reason to merge them.

## Decision

Split them back apart:

1. **`liulian-dev-env`** carries the one-click developer environment
   (Docker workspace image, docker-compose, devcontainer.json for VS
   Code Remote / Codespaces, .gitpod.yml, Makefile orchestration,
   3-OS bootstrap scripts).
2. **`liulian-ops`** carries cross-service production tooling
   (`liulianctl` CLI, Helm umbrella chart, Terraform modules,
   reusable GitHub Actions workflows, Grafana dashboards, deploy
   runbooks).
3. **Per-repo CI** lives in each service repo's `.github/workflows/`
   and calls reusable workflows from `liulian-ops`.

## Decision rule (to apply consistently going forward)

| Question | Answer determines where it lives |
|---|---|
| Where does this run? | Developer's laptop → `liulian-dev-env`. Cloud/cluster → `liulian-ops`. |
| Who runs it? | Engineer at IDE → `dev-env`. CI/CD pipeline + on-call → `ops`. |
| What secrets does it carry? | `.env.example` placeholders → `dev-env`. Vault / SSM / SealedSecrets references → `ops`. |
| What's the failure cost? | "rebuild your laptop image" → `dev-env`. "user-facing outage" → `ops`. |
| What OS does it run on? | Linux + macOS + Windows WSL → `dev-env`. Linux containers / cluster only → `ops`. |
| What's its release cadence? | Frequent, per-developer-preference → `dev-env`. Tied to platform releases, audited → `ops`. |

If two surfaces share three or more answers from above, consider
co-locating them. Repo count is **not** an admissible reason.

## Mapping after this ADR (state on 2026-05-13)

### `liulian-dev-env` contains

- `Dockerfile` — workspace image (Ubuntu + node 20 + python 3.11 + uv
  + helm + kubectl + terraform + gh).
- `docker-compose.dev.yml` — workspace + postgres + redis + minio sidecars,
  with `--profile llm` (ollama) and `--profile obs` (prom + grafana).
- `Makefile` — `make dev / install / api / agent / web / status / shell / logs / destroy`.
- `.devcontainer/devcontainer.json` — VS Code Remote Containers / Codespaces.
- `.gitpod.yml` — Gitpod cloud IDE.
- `scripts/bootstrap-{linux,macos,windows,codespaces}.{sh,ps1}` — one-shot
  per-platform installers that clone the federation + start services.
- `.env.example` — LLM provider keys (DeepSeek / GLM / Gemini / Anthropic)
  + offline-mode flag.
- `obs/prometheus.yml` — local-dev scrape config.
- `README.md` (+ `.zh.md`) — onboarding.

### `liulian-ops` contains

- `neoctl/` — Python click CLI for production deploys (`liulianctl`).
- `helm/liulian-platform/` — umbrella Helm chart (to be filled in M2).
- `terraform/{aws-eks,hetzner-k3s}/` — cluster IaC.
- `grafana/dashboards/` — pre-built JSON.
- `.github/workflows/` — reusable workflows called by per-repo CI.
- `docs/runbooks/` — deploy + incident SOPs.

### Per service repo contains

- `.github/workflows/ci.yml` — lint / test / build, **calls** reusable
  workflows from `liulian-ops` via `uses: liulian-ai/liulian-ops/.github/workflows/X.yml@v1`.
- `Dockerfile` — how *this* service is packaged.

## Consequences

- (+) Each surface has a coherent identity again. `liulian-dev-env`
  becomes the new-developer welcome mat; `liulian-ops` becomes the
  ops-engineer reference.
- (+) Repo count: 7 functional repos (or 8 with `liulian-mobile`
  placeholder). Same as where iteration 3 settled.
- (+) Developer onboarding is sharper: one repo, one badge, three
  commands.
- (−) `liulian-ops`'s value is now narrower; if at M2 we find Helm + TF
  + CLI all live in one repo *and* it's small, we might still merge
  ops into another cross-cutting repo — that would be a new ADR.

## Action items (this commit)

- Repos under new GitHub org `liulian-ai` (transferred 2026-05-13).
- Repos made **public**.
- `liulian-ops/devcontainer/` contents moved back to `liulian-dev-env/`.
- New: `liulian-dev-env/.devcontainer/`, `/.gitpod.yml`, `scripts/bootstrap-*`.
- `liulian-ops` now contains only the deploy/IaC subset.
- `PLATFORM_BLUEPRINT.md §4.1` repo table updated to reflect 8 repos
  (the placeholder included) under `liulian-ai`.
- `NEOBANKER_REUSE_MAP.md §1` repo map updated.

## Cross-references

- `ADR 0001` (multi-repo split — superseded in part by this ADR)
- `AUDIT_REPORT_2026-05-12.md` §B.1 (category-not-count rule)
- `PLATFORM_BLUEPRINT.md` §4.1
- `liulian-dev-env/README.md` (the onboarding surface itself)
