# LIULIAN Development Guide

**Liquid Intelligence and Unified Logic for Interactive Adaptive Networks**

A research-oriented Python library for spatiotemporal model experimentation — training, evaluation, and inference over time-series, graph, and spatiotemporal data.

## Build, Test, and Lint

### Installation

```bash
# Install with uv (recommended)
uv pip install -e ".[dev,logging]"

# Install everything (dev tools + torch + hpo + docs)
uv pip install -e ".[all]"
```

### Running Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=liulian --cov-report=term-missing

# Run a single test file
pytest tests/data/test_data.py -v

# Run a specific test function
pytest tests/adapters/test_dummy_adapter.py::test_dummy_forward -v

# Run tests with specific markers
pytest -m "not slow" -v              # Skip slow tests
pytest -m "not download" -v          # Skip tests requiring downloads
```

**Test requirements:** All tests must run in <30 seconds on CPU (no GPU required). Coverage target: ≥60%.

### Code Quality

```bash
# Format code (use ruff, not black)
ruff format liulian/ tests/ plugins/

# Lint and auto-fix
ruff check --fix liulian/ tests/ plugins/

# Type check
mypy liulian/

# Pre-commit hooks (runs ruff + pytest quick)
pre-commit run --all-files
```

**Style:** 88-char line length, single quotes (ruff-formatted), black-compatible import sorting.

### Documentation

```bash
# Build and serve docs locally
mkdocs serve

# Build static site
mkdocs build
```

Docs are deployed automatically to GitHub Pages via `.github/workflows/docs.yml` on push to main.

## Architecture Overview

LIULIAN uses a **task-driven experiment framework** with strict separation of concerns:

```
Task → Data → Model → Runtime → Experiment
```

### Core Layers

1. **Task Layer** (`liulian/tasks/`)
   - Defines **what** to predict and **how** to measure success
   - `PredictionTask` with `PredictionRegime` (input_length, forecast_horizon, output_type)
   - Tasks own metrics, loss functions, and batch preparation logic
   - **Models never implement task-specific logic**

2. **Data Layer** (`liulian/data/`)
   - `DataSplit` abstraction: `x`, `y`, `split_name` (train/val/test)
   - **Data contracts:** YAML manifests (`manifests/`) define field schemas, topology, and integrity hashing
   - Loaders: `CSVDataset`, `SwissRiverDataset`, `PEMSDataset`, `M4Dataset`
   - Scalers: `StandardScaler`, `MinMaxScaler` in `liulian/data/scalers.py`

3. **Model Layer** (`liulian/models/`)
   - **`ExecutableModel` ABC** is the only interface models must implement:
     - `configure(task, config)` — Initialize model with hyperparameters
     - `forward(batch)` — Run inference, return `{"predictions": array}`
     - `save(path)`, `load(path)` — Checkpoint management
     - `capabilities()` — Declare what the model supports (deterministic, probabilistic, uncertainty)

4. **Adapter Layer** (`liulian/adapters/`)
   - Wraps external models behind `ExecutableModel` interface
   - **One adapter per library** (e.g., `DummyModel` baseline)
   - See **Adapter Rules** below for strict contract

5. **Runtime Layer** (`liulian/runtime/`)
   - **`Experiment`** orchestrator: wires Task + Data + Model + Logger + Optimizer
   - **State machine:** `INIT → TRAIN → EVAL → INFER → COMPLETED`
   - `ForecastTrainer` handles PyTorch training loops (gradient descent, LR scheduling, early stopping, checkpointing)
   - `ExperimentSpec` defines the full experiment configuration

6. **Optimizer Layer** (`liulian/optim/`)
   - Hyperparameter optimization via Ray Tune (`ray_optimizer.py`)
   - Search spaces defined in `search_spaces.py`
   - Fallback to grid sweep if Ray unavailable

7. **Loggers** (`liulian/loggers/`)
   - `LocalLogger` — JSON-lines files
   - `WandbLogger` — Weights & Biases integration

### Plugin Architecture

Domain-specific code lives in `plugins/` directory:

```
plugins/
├── hydrology/      # SwissRiver dataset adapter
└── traffic/        # Traffic data adapters (stub)
```

Plugins extend `BaseDataset` or provide domain-specific `ExecutableModel` adapters. They should NOT leak into core `liulian/` modules.

## Adapter Rules (Critical Contract)

When creating a model adapter in `liulian/adapters/`, follow these **hard rules**:

### ✅ Allowed

- Wrap any external model library (PyTorch, sklearn, custom)
- Implement initialization, forward pass, save/load
- Return predictions in standard format: `{"predictions": np.ndarray}`
- Declare capabilities: `{"deterministic": bool, "probabilistic": bool, "uncertainty": bool}`

### ❌ Forbidden

- Training loops (runtime layer handles this)
- Loss computation (task layer owns this)
- Metric calculation (task layer owns this)
- Data preprocessing/slicing (data layer handles this)
- Logging (logger layer handles this)
- Task-specific logic (e.g., `if task.name == "PredictionTask"`)

### File Structure

```python
# adapters/mymodel/__init__.py
from .adapter import MyModelAdapter

# adapters/mymodel/_vendor.py (isolate 3rd-party imports)
try:
    from external_library import ExternalModel
except ImportError:
    raise ImportError("Install external_library: pip install external_library")

# adapters/mymodel/adapter.py
from ._vendor import ExternalModel
from liulian.models.base import ExecutableModel

class MyModelAdapter(ExecutableModel):
    def configure(self, task, config):
        self.model = ExternalModel(**config)
    
    def forward(self, batch):
        preds = self.model(batch["X"])
        return {"predictions": preds}
    
    def capabilities(self):
        return {"deterministic": True, "probabilistic": False, "uncertainty": False}
```

### Testing Requirements

Every adapter needs a **smoke test** in `tests/adapters/`:

- Uses synthetic data only (no real datasets)
- Runs on CPU in <1 second
- Validates output shape and keys

```python
def test_my_adapter_forward():
    model = MyModelAdapter()
    model.configure(task=None, config={"forecast_horizon": 12})
    batch = {"X": np.random.randn(4, 36, 3).astype(np.float32)}
    output = model.forward(batch)
    assert "predictions" in output
    assert output["predictions"].shape == (4, 12, 3)
```

**Target:** ≤200 LOC per adapter file. See `liulian/adapters/dummy/adapter.py` for reference implementation.

## Key Conventions

### Configuration Management

- **Optional dependencies:** Core requires only `numpy` + `pyyaml`. Everything else is extras (`[torch]`, `[hpo]`, `[logging]`, `[dev]`)
- **Dependency management:** Use `uv` (lock-file first, fast resolver). Lock file: `uv.lock`
- **Python version:** ≥3.10 (CI tests 3.10, 3.11, 3.12)

### Data Manifests

Datasets should have a YAML manifest in `manifests/`:

```yaml
name: my_dataset
version: 1.0
fields:
  - name: timestamp
    type: datetime
    role: temporal
  - name: value
    type: float32
    role: target
topology:
  type: sequence
  length: variable
integrity:
  hash: sha256:abc123...
```

See `docs/manifest_spec.md` for full specification.

### Experiment Structure

Experiments live in `experiments/` directory organized by dataset:

```
experiments/
├── swiss_river/
├── electricity/
├── traffic/
├── weather/
└── etth1/
```

Each experiment directory contains:
- Configuration YAML files
- Runner scripts
- Results subdirectories

### Import Style

```python
# Standard library
import os
from typing import Dict, List, Optional

# Third-party (numpy first, then alphabetical)
import numpy as np
import torch
import yaml

# Local imports (absolute from package root)
from liulian.data.base import DataSplit
from liulian.models.base import ExecutableModel
from liulian.tasks.base import PredictionTask
```

### Type Annotations

Required in all public APIs. Mypy runs with strict mode (`disallow_untyped_defs=true`).

```python
def forward(self, batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Docstring here."""
    ...
```

### Error Messages

Be explicit about missing dependencies:

```python
try:
    import wandb
except ImportError:
    raise ImportError(
        "WandB logging requires the 'logging' extra. "
        "Install with: uv pip install -e '.[logging]'"
    )
```

### CLI Entry Point

```bash
# CLI is available after installation
liulian --help
liulian run experiments/my_experiment/config.yaml
```

Implemented in `liulian/cli.py` with subcommands for `run`, `eval`, `infer`.

## Common Workflows

### Adding a New Model Adapter

1. Create `liulian/adapters/mymodel/` directory
2. Add `_vendor.py` for 3rd-party imports
3. Implement `adapter.py` inheriting from `ExecutableModel`
4. Write smoke test in `tests/adapters/test_mymodel.py`
5. Update `liulian/adapters/__init__.py` to expose it
6. Verify: `pytest tests/adapters/test_mymodel.py -v`

### Running an Experiment

```python
from liulian.runtime.experiment import Experiment
from liulian.runtime.spec import ExperimentSpec
from liulian.tasks.base import PredictionTask, PredictionRegime
from liulian.data.csv_dataset import CSVDataset
from liulian.adapters.dummy import DummyModel

# Define task
regime = PredictionRegime(input_length=96, forecast_horizon=24, output_type="deterministic")
task = PredictionTask(regime=regime)

# Load data
dataset = CSVDataset.from_manifest("manifests/my_data.yaml")

# Create model
model = DummyModel()

# Define experiment spec
spec = ExperimentSpec(
    name="my_experiment",
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
)

# Run experiment
experiment = Experiment(spec, task, dataset, model)
summary = experiment.run()
print(summary["test_metrics"])
```

### Adding a Domain Plugin

1. Create `plugins/mydomain/` directory
2. Add dataset adapter or model extending `BaseDataset` / `ExecutableModel`
3. Add manifest YAML if applicable
4. Write tests in `tests/` that mock external dependencies
5. Document in `docs/datasets.md` or `docs/models/`

## Pytest Markers

Use markers to categorize tests:

```python
@pytest.mark.slow
def test_large_training_run():
    """Tests that take >5 seconds."""
    ...

@pytest.mark.download
def test_download_pretrained_model():
    """Tests requiring internet connection."""
    ...

@pytest.mark.main_branch
def test_swiss_river_e2e():
    """Tests only run on main branch (full integration)."""
    ...
```

Run selectively:
```bash
pytest -m "not slow and not download" -v
```

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

- **ci.yml** — Lint (ruff) + test (pytest) on Python 3.10-3.12
- **docs.yml** — Auto-deploy MkDocs to GitHub Pages
- **publish.yml** — Publish package to PyPI

Tests run on every PR and push to main. Coverage report uploaded to PR comments.

## Documentation Structure

- `docs/architecture.md` — System design and design decisions
- `docs/adapter_guide.md` — Model adapter contract (required reading)
- `docs/manifest_spec.md` — Data contract specification
- `docs/cli.md` — Command-line usage
- `docs/datasets.md` — Built-in dataset catalog
- `docs/contributing.md` — Development guidelines

## Project Status

**Current:** MVP1 (v0.0.1, pre-alpha)
- ✅ Task-driven experiment paradigm
- ✅ Data contracts with YAML manifests
- ✅ ExecutableModel interface + DummyModel baseline
- ✅ State machine runtime + Experiment orchestrator
- ✅ Ray Tune optimizer with fallback
- ✅ Local + WandB logging
- ✅ 60%+ test coverage

**Planned (v1+):**
- FastAPI experiment server
- Online/streaming mode
- Human-in-the-loop (HITL) integration
- Probabilistic output support
- Extended adapter library (PyTorch, sklearn models)

---

For more details, see:
- [README.md](../README.md) — Quick start and overview
- [docs/contributing.md](../docs/contributing.md) — Development guidelines
- [docs/adapter_guide.md](../docs/adapter_guide.md) — Model adapter contract
