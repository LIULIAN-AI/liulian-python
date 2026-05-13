---
name: verify
description: Run the exact checks CI runs — ruff lint, ruff format check, mypy strict, and fast pytest (no slow/download markers). Use before marking work done or opening a PR.
---

Run the full verification gate for this repo, in order. Stop on first failure and report it plainly.

```bash
ruff check liulian/ tests/ plugins/
ruff format --check liulian/ tests/ plugins/
mypy liulian/
pytest -m "not slow and not download" -v
```

Notes for the agent:

- These are the same commands the GitHub Actions lint + test-core jobs run. If this passes locally, CI will pass (modulo `test_integration.py`, which CI runs only on Python 3.12 with `[torch-models]`).
- `mypy` runs in strict mode (`disallow_untyped_defs=true`) — missing type annotations on public APIs will fail.
- If `ruff format --check` fails, run `ruff format liulian/ tests/ plugins/` to fix, then re-run verify.
- Do NOT run `black` — this repo uses ruff-format despite the stale `[tool.black]` block in `pyproject.toml`.
- Pytest config lives in `pyproject.toml` (`testpaths = ["tests"]`); ad-hoc root-level scripts like `test_etsformer_*.py` and `_test_batch.py` are not part of the suite and should not be invoked.
