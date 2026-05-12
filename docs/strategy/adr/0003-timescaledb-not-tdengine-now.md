# ADR 0003 — TimescaleDB for time-series storage; revisit at scale

- **Status**: accepted with day-1 caveat (plain Postgres first)
- **Decision date**: 2026-05-12
- **Audit status**: PASS after de-sloganing (see `AUDIT_REPORT_2026-05-12.md §B.2`)

## Context

LIULIAN's `run_metric`, `forecast`, and `alert` tables are append-only
time-keyed streams. By M2 these will hold millions of rows. We need:

- Time-bucket queries fast at scale (e.g. rolling-30-day MAE).
- Automatic partition management (don't write our own cron).
- Compatibility with FastAPI + SQLModel.
- Open license suitable for both self-host and commercial deployments.

## Decision

Use **TimescaleDB** (Postgres extension, Apache-2.0) for these three
tables. `CREATE EXTENSION timescaledb` is the only adoption cost;
SQLModel works unchanged.

**Sprint pragmatism**: Day 1 of `liulian-api` uses **plain Postgres**.
TimescaleDB extension is enabled in a follow-up commit only after the
M1 demo is shipping. This removes one risk vector from the sprint.

## Rationale (primary)

- **Partition management at zero cost**: hypertables auto-partition by
  time; we don't maintain `pg_partman` schedules.
- **Same query language**: still PostgreSQL; SQLModel + Alembic work
  unchanged.
- **Migration story for customers**: a customer on plain Postgres can
  adopt TimescaleDB via `CREATE EXTENSION timescaledb` with zero
  downtime. Same path we took.

## Rationale (secondary, narrative-only)

- A TS product running on a TS-native primitive has internal
  consistency. Some engineer-reviewers in hiring loops appreciate it;
  others don't notice. Small upside, not the deciding factor.

## Alternatives considered

- **Plain Postgres + pg_partman + pg_cron**: works through M4 at our
  scale. Simpler stack, more partition-management code to maintain.
  We adopt this for Sprint Day 1 only; switch to TimescaleDB before M2.
- **TDengine** (AGPL-3.0): higher TS-native perf; AGPL-3.0 forces
  open-source on all deployments. Reject now; documented as the
  M4+ migration target in `REFERENCE_DESIGNS.md §D1`.
- **InfluxDB**: separate query language; another DB to operate.
- **ClickHouse**: column store overkill for our row-shape workloads.

## Consequences

- (+) Hypertables eliminate partition cron jobs.
- (+) Same Postgres skill set; no new operations training.
- (+) Migration path for customers documented.
- (−) One additional extension to install in deployments (mitigated:
  Timescale ships TimescaleDB-HA Docker images).
- (−) Some advanced TimescaleDB features (continuous aggregates,
  retention policies) are pay-for in Timescale Cloud; we use the
  open-source self-host path.

## Migration triggers (revisit at)

- **M4 (2026-08)**: if we're at >100M rows in any hypertable and
  TimescaleDB self-host perf becomes a bottleneck → evaluate TDengine.
- **M5 (2026-09)**: if customers ask for sub-second aggregations over
  multi-billion-row windows → evaluate ClickHouse for the analytics
  layer only.

## Cross-references

- `PLATFORM_BLUEPRINT.md` §5.1 (TimescaleDB rationale)
- `PLATFORM_BLUEPRINT.md` §5.3 (storage schema with hypertable
  declaration)
- `REFERENCE_DESIGNS.md` §D1 (TDengine future-migration notes)
