---
name: new-adapter
description: Scaffold a new model adapter under liulian/adapters/<name>/ with the required _vendor.py + adapter.py + smoke-test layout, enforcing the strict adapter contract from docs/adapter_guide.md.
disable-model-invocation: true
---

Scaffold a new adapter for the library named in `$ARGUMENTS` (use snake_case; if omitted, ask the user).

## Before writing anything

1. Confirm the name with the user if ambiguous. Refuse names that collide with an existing directory under `liulian/adapters/`.
2. Read `liulian/adapters/dummy/` as the canonical reference implementation — mimic its shape and ABC method signatures.
3. Read `docs/adapter_guide.md` for the full contract.

## Hard contract (do NOT violate)

Adapters MAY:

- Import and wrap any external library (isolate imports in `_vendor.py`, raise `ImportError` with a `pip install -e '.[<extra>]'` hint if missing).
- Implement `configure(task, config)`, `forward(batch)`, `save(path)`, `load(path)`, `capabilities()`.
- Return predictions as `{"predictions": np.ndarray}`.

Adapters MUST NOT:

- Run training loops (that belongs to `liulian/runtime/`).
- Compute loss or metrics (owned by `liulian/tasks/`).
- Preprocess, slice, or scale data (owned by `liulian/data/`).
- Log anything (owned by `liulian/loggers/`).
- Branch on task type (e.g. `if task.name == "PredictionTask":` is forbidden).

## Files to create

```
liulian/adapters/<name>/
├── __init__.py         # re-export the adapter class
├── _vendor.py          # isolated 3rd-party imports with install-hint ImportError
└── adapter.py          # <=200 LOC, subclasses ExecutableModel

tests/adapters/test_<name>.py   # synthetic-data smoke test, <1s on CPU
```

Also update `liulian/adapters/__init__.py` to expose the new adapter class.

## After scaffolding

1. Run `pytest tests/adapters/test_<name>.py -v` to confirm the smoke test passes.
2. Invoke the `/verify` skill to check format/lint/mypy/fast-tests pass.
3. Remind the user to declare `capabilities()` honestly (`deterministic` / `probabilistic` / `uncertainty`) — the runtime dispatches on this.
