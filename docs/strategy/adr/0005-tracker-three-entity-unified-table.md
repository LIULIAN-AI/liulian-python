# ADR 0005 — Tracker covers task + experiment + agent in a unified `run` table

- **Status**: accepted
- **Decision date**: 2026-05-12

## Context

LIULIAN executes three categories of work:

1. **Experiments** (research runs, HPO trials, benchmark sweeps).
2. **Tasks** (scheduled or on-demand: retrain, ingest, alert-sweep).
3. **Agents** (LLM invocations through `liulian-agent`).

Each produces metrics, artifacts, and an audit trail. The question:
one tracker schema or three?

## Decision

**One `run` table with a `parent_kind` discriminator** (`experiment` |
`task` | `agent`). Detailed metrics live in a `run_metric` hypertable
(TimescaleDB). Artifacts live in an `artifact` table. Agent-specific
step-level detail lives in `agent_run_step`.

## Rationale

- **Unified activity feed**: one query lists everything the user did
  in the last 24h; cross-cuts (cost / duration / quality) work across
  entity types.
- **One place to instrument**: middleware + observability hook the
  `run` table once; all three entity types benefit.
- **Industry pattern**: Airflow `TaskInstance`, Temporal
  `WorkflowExecution`, MLflow `Run` follow the same shape.

## Schema (abbreviated)

```sql
CREATE TABLE run (
  id UUID PRIMARY KEY,
  parent_kind TEXT NOT NULL,    -- 'task' | 'experiment' | 'agent'
  parent_id UUID NOT NULL,
  status TEXT NOT NULL,         -- 'pending' | 'running' | 'completed' | 'failed' | 'aborted'
  started_at TIMESTAMPTZ,
  ended_at TIMESTAMPTZ,
  duration_ms INT GENERATED ALWAYS AS
    (EXTRACT(EPOCH FROM (ended_at - started_at)) * 1000) STORED,
  metrics_summary JSONB,
  metadata JSONB
);
CREATE INDEX ON run (parent_kind, parent_id, started_at DESC);

CREATE TABLE run_metric (        -- hypertable
  run_id UUID NOT NULL,
  step INT NOT NULL,
  name TEXT NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  recorded_at TIMESTAMPTZ DEFAULT now()
);
SELECT create_hypertable('run_metric', 'recorded_at');

CREATE TABLE agent_run_step (    -- per-step LLM details for agent runs
  run_id UUID REFERENCES run,
  step INT,
  role TEXT,                     -- 'planner' | 'tool' | 'reflection' | 'final'
  content TEXT,
  tool_name TEXT,
  tool_args JSONB,
  tool_result JSONB,
  tokens_in INT,
  tokens_out INT,
  cost_usd DOUBLE PRECISION
);
```

## Alternatives considered

- **Three separate tables** (`experiment_run`, `task_run`,
  `agent_run`): cleaner separation; but every cross-cut query must
  `UNION ALL` three tables, and we'd need three "recent activity"
  endpoints. Rejected.
- **Inheritance via table-inheritance (Postgres feature)**: adds
  ORM-incompatibility complexity for marginal benefit.

## Consequences

- (+) Unified activity feed (`/studio/activity`).
- (+) Single set of indexes covers all three queries.
- (+) Easy MLflow-compatibility shim: MLflow's `runs/create` maps to
  any of the three by setting `parent_kind`.
- (−) The `parent_id` foreign key is polymorphic; we enforce it at
  application level (Pydantic validator) rather than DB constraint.
- (−) `metrics_summary` JSONB shape varies by `parent_kind`; documented
  per kind in `liulian-api/services/tracker.py` docstrings.

## Cross-references

- `PLATFORM_BLUEPRINT.md` §5.3 (storage schema in context)
- `PLATFORM_BLUEPRINT.md` §8 (the tracker design)
