# Entity Identifier Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ordered entity-identifier forecasting support (`none`, `embedding`, `onehot`) for the approved models/datasets and ship a runnable local smoke workflow plus Slurm full-matrix execution.

**Architecture:** Keep existing correct identifier behavior, add missing integrations in a model-by-model sequence, and add a dedicated `experiments/entity_identifier/` orchestration package that reuses `experiments/run.py`. Embed-first implementation is completed and validated before onehot is enabled in the same matrix workflow. Cluster execution follows the style of `jobs/run_jobs_ray_tune.py` with parameterized sbatch generation.

**Tech Stack:** Python, PyTorch, pytest, YAML configs, liulian pipeline/runtime, Slurm (`sbatch`).

---

## File Structure (locked before implementation)

### Create

- `experiments/entity_identifier/__init__.py` — package marker.
- `experiments/entity_identifier/matrix.py` — ordered constants for models/datasets/modes.
- `experiments/entity_identifier/generate_configs.py` — full matrix YAML generation.
- `experiments/entity_identifier/run.py` — dry-run/smoke/full execution wrapper around `experiments/run.py`.
- `experiments/entity_identifier/compare.py` — aggregate `{none, embedding, onehot}` result tables.
- `experiments/entity_identifier/submit_slurm.py` — sbatch generation/submission.
- `tests/models/torch/test_lstm.py` — LSTM entity tests.
- `tests/models/torch/test_gpt4ts.py` — GPT4TS adapter/entity tests.
- `tests/runtime/test_entity_identifier_pipeline.py` — matrix/generator/runner/compare/submitter tests.
- `docs/research/2026-04-10-entity-identifier-implementation-report.md` — final detailed report.

### Modify

- `liulian/models/torch/entity_mixin.py`
- `liulian/pipeline.py`
- `liulian/models/torch/lstm.py`
- `liulian/models/torch/transformer.py`
- `liulian/models/torch/dlinear.py`
- `liulian/models/torch/patchtst.py`
- `liulian/models/torch/itransformer.py`
- `liulian/models/torch/timellm.py`
- `liulian/models/torch/gpt4ts.py`
- `liulian/models/torch/__init__.py`
- `liulian/data/swiss_river.py`
- `liulian/data/csv_dataset.py`
- `liulian/data/pems_dataset.py`
- `tests/models/torch/test_entity_integration.py`
- `tests/models/torch/test_transformer.py`
- `tests/models/torch/test_dlinear.py`
- `tests/models/torch/test_patchtst.py`
- `tests/models/torch/test_itransformer.py`
- `tests/models/torch/test_timellm.py`

---

### Task 1: Shared identifier plumbing hardening (embedding + onehot)

**Files:**
- Modify: `liulian/models/torch/entity_mixin.py`
- Modify: `liulian/pipeline.py`
- Test: `tests/models/torch/test_entity_integration.py`

- [ ] **Step 1: Write failing tests for multi-channel onehot path**

```python
def test_build_model_wraps_multichannel_onehot():
    from types import SimpleNamespace
    from liulian.pipeline import build_model
    from liulian.models.torch.entity_mixin import ChannelOneHotWrapper

    dataset = SimpleNamespace(station_ids=['s0', 's1', 's2'])
    cfg = {
        'model': 'dlinear',
        'task_name': 'long_term_forecast',
        'seq_len': 16,
        'pred_len': 4,
        'enc_in': 3,
        'dec_in': 3,
        'c_out': 3,
        'identifier_mode': 'onehot',
        'id_integration': 'concat_to_x',
        'split_mode': 'multi_channel',
        'num_embeddings': 3,
        'embedding_size': 4,
    }
    m = build_model(cfg, dataset=dataset)
    assert isinstance(m, ChannelOneHotWrapper)
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/models/torch/test_entity_integration.py -k multichannel_onehot -v`  
Expected: FAIL (`ChannelOneHotWrapper` missing / model not wrapped).

- [ ] **Step 3: Implement minimal shared fix**

```python
class ChannelOneHotWrapper(nn.Module):
    def __init__(self, inner_model: nn.Module, num_stations: int) -> None:
        super().__init__()
        self.inner = inner_model
        self.register_buffer('station_ids', torch.arange(num_stations, dtype=torch.long))
        self.enc_proj = nn.Linear(1 + num_stations, 1)
        self.dec_proj = nn.Linear(1 + num_stations, 1)

    def _augment(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        b, t, n = x.shape
        onehot = torch.nn.functional.one_hot(self.station_ids, num_classes=n).float()
        onehot = onehot.unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1)
        return proj(torch.cat([x.unsqueeze(-1), onehot], dim=-1)).squeeze(-1)
```

```python
if config.get('identifier_mode') == 'onehot' and config.get('split_mode') == 'multi_channel':
    from liulian.models.torch.entity_mixin import ChannelOneHotWrapper
    model = ChannelOneHotWrapper(model, num_stations=config['num_embeddings'])
```

- [ ] **Step 4: Run test and verify pass**

Run: `pytest tests/models/torch/test_entity_integration.py -k multichannel_onehot -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liulian/models/torch/entity_mixin.py liulian/pipeline.py tests/models/torch/test_entity_integration.py
git commit -m "feat: add robust onehot support for multichannel entity path"
```

---

### Task 2: Model 1/7 — LSTM entity integration

**Files:**
- Modify: `liulian/models/torch/lstm.py`
- Create: `tests/models/torch/test_lstm.py`

- [ ] **Step 1: Write failing LSTM entity tests**

```python
def test_lstm_adapter_embedding_mode_initializes():
    from liulian.models.torch.lstm import LSTMAdapter
    cfg = {'seq_len': 24, 'pred_len': 6, 'enc_in': 4, 'identifier_mode': 'embedding', 'num_embeddings': 10}
    m = LSTMAdapter(cfg)
    assert hasattr(m, '_model')


def test_lstm_adapter_onehot_forward_runs():
    import numpy as np
    from liulian.models.torch.lstm import LSTMAdapter
    cfg = {'seq_len': 24, 'pred_len': 6, 'enc_in': 7, 'c_out': 7, 'identifier_mode': 'onehot'}
    m = LSTMAdapter(cfg)
    out = m.run({'x_enc': np.random.randn(2, 24, 7).astype(np.float32)})
    assert out['predictions'].shape[1] == 6
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/models/torch/test_lstm.py -v`  
Expected: FAIL (current adapter not fully wired for entity mixin path).

- [ ] **Step 3: Implement LSTM adapter wiring**

```python
from liulian.models.torch.entity_mixin import EntityAwareMixin


class LSTMAdapter(EntityAwareMixin, TorchModelAdapter):
    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'd_model': 64,
            'e_layers': 2,
            'dropout': 0.1,
            'task_name': 'long_term_forecast',
            'c_out': config.get('c_out', config.get('enc_in', 1)),
        }
        default_config.update(config)
        model_cfg = self._entity_model_config(default_config)
        model = Model(self._dict_to_namespace(model_cfg))
        super().__init__(model, default_config)
        self._init_entity_support(default_config)
```

- [ ] **Step 4: Run test and verify pass**

Run: `pytest tests/models/torch/test_lstm.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liulian/models/torch/lstm.py tests/models/torch/test_lstm.py
git commit -m "feat: wire entity-aware support into vanilla lstm adapter"
```

---

### Task 3: Model 2/7 — Transformer native entity hook (keep fallback)

**Files:**
- Modify: `liulian/models/torch/transformer.py`
- Modify: `tests/models/torch/test_transformer.py`

- [ ] **Step 1: Write failing transformer entity-hook test**

```python
def test_transformer_native_entity_add_to_embed_runs():
    import numpy as np
    from liulian.models.torch.transformer import TransformerAdapter
    cfg = {
        'task_name': 'long_term_forecast',
        'seq_len': 32,
        'label_len': 16,
        'pred_len': 8,
        'enc_in': 4,
        'dec_in': 4,
        'c_out': 4,
        'entity_injection': 'add_to_embed',
        'identifier_mode': 'embedding',
        'num_embeddings': 10,
    }
    m = TransformerAdapter(cfg)
    x = np.random.randn(2, 32, 4).astype(np.float32)
    mark = np.zeros((2, 32, 4), dtype=np.float32)
    mark[:, :, 0] = 3
    out = m.run({'x_enc': x, 'x_mark_enc': mark, 'x_dec': np.zeros((2, 24, 4), dtype=np.float32), 'x_mark_dec': np.zeros((2, 24, 4), dtype=np.float32)})
    assert out['predictions'].shape == (2, 8, 4)
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/models/torch/test_transformer.py -k native_entity_add_to_embed -v`  
Expected: FAIL (`entity_injection` not handled natively).

- [ ] **Step 3: Implement native hook**

```python
self.entity_injection = getattr(configs, 'entity_injection', 'wrapper')
if self.entity_injection == 'add_to_embed':
    self.entity_embedding = nn.Embedding(getattr(configs, 'num_embeddings', 1), configs.d_model)

def _inject_entity(self, enc_out, x_mark_enc):
    if self.entity_injection != 'add_to_embed' or x_mark_enc is None:
        return enc_out
    entity_ids = x_mark_enc[:, 0, 0].long().clamp_min(0)
    return enc_out + self.entity_embedding(entity_ids).unsqueeze(1)
```

- [ ] **Step 4: Run test and verify pass**

Run: `pytest tests/models/torch/test_transformer.py -k native_entity_add_to_embed -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liulian/models/torch/transformer.py tests/models/torch/test_transformer.py
git commit -m "feat: add native transformer entity embedding injection mode"
```

---

### Task 4: Model 3/7 — DLinear native entity affine head (keep fallback)

**Files:**
- Modify: `liulian/models/torch/dlinear.py`
- Modify: `tests/models/torch/test_dlinear.py`

- [ ] **Step 1: Write failing dlinear entity-affine test**

```python
def test_dlinear_entity_affine_mode_runs():
    import numpy as np
    from liulian.models.torch.dlinear import DLinearAdapter
    cfg = {
        'task_name': 'long_term_forecast',
        'seq_len': 32,
        'pred_len': 8,
        'enc_in': 4,
        'dec_in': 4,
        'c_out': 4,
        'identifier_mode': 'embedding',
        'num_embeddings': 12,
        'entity_affine': True,
    }
    m = DLinearAdapter(cfg)
    x = np.random.randn(2, 32, 4).astype(np.float32)
    mark = np.zeros((2, 32, 4), dtype=np.float32)
    mark[:, :, 0] = 2
    out = m.run({'x_enc': x, 'x_mark_enc': mark})
    assert out['predictions'].shape == (2, 8, 4)
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/models/torch/test_dlinear.py -k entity_affine_mode_runs -v`  
Expected: FAIL (`entity_affine` not implemented).

- [ ] **Step 3: Implement native affine path**

```python
self.entity_affine = getattr(configs, 'entity_affine', False)
if self.entity_affine:
    n = getattr(configs, 'num_embeddings', 1)
    self.entity_scale = nn.Embedding(n, self.channels)
    self.entity_bias = nn.Embedding(n, self.channels)

def _apply_entity_affine(self, y, x_mark_enc):
    if not self.entity_affine or x_mark_enc is None:
        return y
    eid = x_mark_enc[:, 0, 0].long().clamp_min(0)
    scale = self.entity_scale(eid).unsqueeze(1)
    bias = self.entity_bias(eid).unsqueeze(1)
    return y * (1.0 + scale) + bias
```

- [ ] **Step 4: Run test and verify pass**

Run: `pytest tests/models/torch/test_dlinear.py -k entity_affine_mode_runs -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liulian/models/torch/dlinear.py tests/models/torch/test_dlinear.py
git commit -m "feat: add dlinear entity-conditioned affine output option"
```

---

### Task 5: Models 4/7 and 5/7 — PatchTST and iTransformer entity paths

**Files:**
- Modify: `liulian/models/torch/patchtst.py`
- Modify: `liulian/models/torch/itransformer.py`
- Modify: `tests/models/torch/test_patchtst.py`
- Modify: `tests/models/torch/test_itransformer.py`

- [ ] **Step 1: Write failing tests**

```python
def test_patchtst_add_after_patch_requires_multichannel():
    import pytest
    from liulian.models.torch.patchtst import Model
    class Cfg:
        task_name='long_term_forecast'; seq_len=32; pred_len=8; enc_in=4; d_model=32; factor=1; dropout=0.1; n_heads=4; d_ff=64; e_layers=1; activation='gelu'
        identifier_mode='embedding'; id_integration='add_after_patch'; split_mode='per_entity'
    with pytest.raises(ValueError):
        Model(Cfg())


def test_itransformer_variate_identity_embedding_runs():
    import numpy as np
    from liulian.models.torch.itransformer import iTransformerAdapter
    cfg = {'task_name': 'long_term_forecast', 'seq_len': 32, 'pred_len': 8, 'enc_in': 4, 'dec_in': 4, 'c_out': 4, 'use_variate_id_embedding': True}
    m = iTransformerAdapter(cfg)
    out = m.run({'x_enc': np.random.randn(2, 32, 4).astype(np.float32), 'x_mark_enc': np.zeros((2, 32, 4), dtype=np.float32)})
    assert out['predictions'].shape == (2, 8, 4)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/models/torch/test_patchtst.py tests/models/torch/test_itransformer.py -k "add_after_patch_requires_multichannel or variate_identity_embedding_runs" -v`  
Expected: FAIL on iTransformer native identity path test.

- [ ] **Step 3: Implement iTransformer native variate embedding and keep PatchTST behavior**

```python
self.use_variate_id_embedding = getattr(configs, 'use_variate_id_embedding', False)
if self.use_variate_id_embedding:
    self.variate_embedding = nn.Embedding(getattr(configs, 'enc_in', 1), configs.d_model)

def _inject_variate_identity(self, enc_out):
    if not self.use_variate_id_embedding:
        return enc_out
    n = enc_out.size(1)
    idx = torch.arange(n, device=enc_out.device)
    return enc_out + self.variate_embedding(idx).unsqueeze(0)
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/models/torch/test_patchtst.py tests/models/torch/test_itransformer.py -k "add_after_patch_requires_multichannel or variate_identity_embedding_runs" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liulian/models/torch/patchtst.py liulian/models/torch/itransformer.py tests/models/torch/test_patchtst.py tests/models/torch/test_itransformer.py
git commit -m "feat: preserve patchtst entity path and add itransformer variate identity embedding"
```

---

### Task 6: Models 6/7 and 7/7 — TimeLLM and GPT4TS integration

**Files:**
- Modify: `liulian/models/torch/timellm.py`
- Modify: `liulian/models/torch/gpt4ts.py`
- Modify: `liulian/models/torch/__init__.py`
- Modify: `tests/models/torch/test_timellm.py`
- Create: `tests/models/torch/test_gpt4ts.py`

- [ ] **Step 1: Write failing tests**

```python
def test_timellm_entity_prompt_prefix_builder():
    from liulian.models.torch.timellm import _build_entity_prompt_prefix
    assert _build_entity_prompt_prefix(7) == "Entity ID: 7. "


def test_gpt4ts_adapter_exposed_from_torch_init():
    import liulian.models.torch as mt
    assert hasattr(mt, 'GPT4TSAdapter')
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/models/torch/test_timellm.py tests/models/torch/test_gpt4ts.py -k "entity_prompt_prefix_builder or adapter_exposed_from_torch_init" -v`  
Expected: FAIL because helper and adapter export are missing.

- [ ] **Step 3: Implement helper + adapter + export**

```python
def _build_entity_prompt_prefix(entity_id: int) -> str:
    return f'Entity ID: {int(entity_id)}. '
```

```python
class GPT4TSAdapter(EntityAwareMixin, TorchModelAdapter):
    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': config.get('enc_in', 7),
            'dec_in': config.get('dec_in', config.get('enc_in', 7)),
            'c_out': config.get('c_out', config.get('enc_in', 7)),
            'd_model': 768,
            'd_ff': 768,
            'dropout': 0.1,
            'patch_len': 16,
        }
        default_config.update(config)
        model_cfg = self._entity_model_config(default_config)
        model = Model(self._dict_to_namespace(model_cfg))
        super().__init__(model, default_config)
        self._init_entity_support(default_config)
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/models/torch/test_timellm.py tests/models/torch/test_gpt4ts.py -k "entity_prompt_prefix_builder or adapter_exposed_from_torch_init" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liulian/models/torch/timellm.py liulian/models/torch/gpt4ts.py liulian/models/torch/__init__.py tests/models/torch/test_timellm.py tests/models/torch/test_gpt4ts.py
git commit -m "feat: add timellm prompt-entity helper and gpt4ts entity adapter"
```

---

### Task 7: Dataset wiring for Swiss/Traffic/Electricity/PEMS/Exchange

**Files:**
- Modify: `liulian/data/swiss_river.py`
- Modify: `liulian/data/csv_dataset.py`
- Modify: `liulian/data/pems_dataset.py`
- Modify: `tests/runtime/test_benchmark_pipeline.py`

- [ ] **Step 1: Write failing dataset-wiring tests**

```python
def test_swiss_multichannel_constructor_preserves_identifier_args():
    from liulian.data.swiss_river import SwissRiverDataset
    ds = SwissRiverDataset(data_name='swiss-river-1990', split_mode='multi_channel', identifier_mode='onehot', id_integration='concat_to_x')
    info = ds.get_split('train').info()
    assert info['identifier_mode'] == 'onehot'


def test_csv_and_pems_provide_station_ids_for_embedding_resolution():
    from liulian.data.csv_dataset import CSVDataset
    from liulian.data.pems_dataset import PEMSDataset
    assert hasattr(CSVDataset, '__init__')
    assert hasattr(PEMSDataset, '__init__')
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/runtime/test_benchmark_pipeline.py -k "swiss_multichannel_constructor_preserves_identifier_args or station_ids_for_embedding_resolution" -v`  
Expected: FAIL on Swiss multi-channel identifier propagation.

- [ ] **Step 3: Implement dataset fixes**

```python
mc_ds = TimeSeriesDataset(
    splits={split_name: df[['epoch_day'] + feature_cols + target_cols]},
    time_col='epoch_day',
    feature_cols=feature_cols,
    target_cols=target_cols,
    seq_len=self.seq_len,
    pred_len=self.pred_len,
    task=self.task,
    station_ids=self.station_ids,
    identifier_mode=self.identifier_mode,
    id_integration=self.id_integration,
)
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/runtime/test_benchmark_pipeline.py -k "swiss_multichannel_constructor_preserves_identifier_args or station_ids_for_embedding_resolution" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liulian/data/swiss_river.py liulian/data/csv_dataset.py liulian/data/pems_dataset.py tests/runtime/test_benchmark_pipeline.py
git commit -m "fix: propagate identifier args across swiss/csv/pems dataset paths"
```

---

### Task 8: Build `experiments/entity_identifier/` matrix generator and runner

**Files:**
- Create: `experiments/entity_identifier/__init__.py`
- Create: `experiments/entity_identifier/matrix.py`
- Create: `experiments/entity_identifier/generate_configs.py`
- Create: `experiments/entity_identifier/run.py`
- Create: `experiments/entity_identifier/compare.py`
- Create: `tests/runtime/test_entity_identifier_pipeline.py`

- [ ] **Step 1: Write failing matrix/runner tests**

```python
def test_matrix_order_and_size():
    from experiments.entity_identifier.matrix import MODELS, DATASETS, IDENTIFIER_MODES
    assert MODELS == ['lstm', 'transformer', 'dlinear', 'patchtst', 'itransformer', 'timellm', 'gpt4ts']
    assert DATASETS == ['swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich', 'traffic', 'electricity', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'exchange_rate']
    assert IDENTIFIER_MODES == ['none', 'embedding', 'onehot']


def test_generator_count(tmp_path):
    from experiments.entity_identifier.generate_configs import generate_all
    count = generate_all(tmp_path)
    assert count == 210
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/runtime/test_entity_identifier_pipeline.py -k "matrix_order_and_size or generator_count" -v`  
Expected: FAIL (modules not present).

- [ ] **Step 3: Implement matrix/generator/runner/compare**

```python
# matrix.py
MODELS = ['lstm', 'transformer', 'dlinear', 'patchtst', 'itransformer', 'timellm', 'gpt4ts']
DATASETS = ['swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich', 'traffic', 'electricity', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'exchange_rate']
IDENTIFIER_MODES = ['none', 'embedding', 'onehot']
```

```python
# generate_configs.py
def generate_all(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for model in MODELS:
        for dataset in DATASETS:
            for mode in IDENTIFIER_MODES:
                cfg = build_config(model=model, dataset=dataset, identifier_mode=mode)
                (output_dir / f'{dataset}__{model}__{mode}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
                count += 1
    return count
```

```python
# run.py local fallback for llm models
def missing_weights_for_model(model: str) -> bool:
    return model in {'timellm', 'gpt4ts'} and os.environ.get('LIULIAN_ALLOW_LLM_DOWNLOAD', '0') != '1'
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/runtime/test_entity_identifier_pipeline.py -k "matrix_order_and_size or generator_count" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/entity_identifier/__init__.py experiments/entity_identifier/matrix.py experiments/entity_identifier/generate_configs.py experiments/entity_identifier/run.py experiments/entity_identifier/compare.py tests/runtime/test_entity_identifier_pipeline.py
git commit -m "feat: add entity identifier experiment matrix generator and runner"
```

---

### Task 9: Add Slurm submitter aligned to `jobs/run_jobs_ray_tune.py`

**Files:**
- Create: `experiments/entity_identifier/submit_slurm.py`
- Modify: `tests/runtime/test_entity_identifier_pipeline.py`

- [ ] **Step 1: Write failing submitter test**

```python
def test_submitter_generates_valid_sbatch_text():
    from experiments.entity_identifier.submit_slurm import build_sbatch_script
    text = build_sbatch_script(job_name='eid-test', command='python experiments/entity_identifier/run.py --full', use_gpu=True)
    assert '#SBATCH --job-name=' in text
    assert '#SBATCH --output=' in text
    assert 'python experiments/entity_identifier/run.py --full' in text
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/runtime/test_entity_identifier_pipeline.py -k submitter_generates_valid_sbatch_text -v`  
Expected: FAIL (submitter missing).

- [ ] **Step 3: Implement submitter**

```python
def build_sbatch_script(job_name: str, command: str, use_gpu: bool) -> str:
    header = [
        '#!/bin/bash',
        f'#SBATCH --job-name="{job_name}"',
        '#SBATCH --output="outputs/%x.o%J"',
        '#SBATCH --error="errors/%x.e%J"',
        '#SBATCH --time=24:00:00',
    ]
    if use_gpu:
        header.extend(['#SBATCH --partition=gpu', '#SBATCH --gres=gpu:1'])
    else:
        header.append('#SBATCH --partition=cpu')
    return '\n'.join(header + ['', command, ''])
```

- [ ] **Step 4: Run test and verify pass**

Run: `pytest tests/runtime/test_entity_identifier_pipeline.py -k submitter_generates_valid_sbatch_text -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/entity_identifier/submit_slurm.py tests/runtime/test_entity_identifier_pipeline.py
git commit -m "feat: add slurm submitter for entity identifier experiment matrix"
```

---

### Task 10: Final verification and detailed report

**Files:**
- Create: `docs/research/2026-04-10-entity-identifier-implementation-report.md`
- Modify: `docs/research/2026-04-10-entity-id-forecasting-report-refined.md`

- [ ] **Step 1: Run dry + smoke + dev verification**

Run:

```bash
python experiments/entity_identifier/generate_configs.py --dry-run
python experiments/entity_identifier/run.py --dry-run
pytest tests/runtime/test_entity_identifier_pipeline.py -v
pytest tests/models/torch/test_entity_integration.py tests/models/torch/test_lstm.py tests/models/torch/test_gpt4ts.py -v
pytest -m "not slow and not download" -v
```

Expected: all listed checks pass.

- [ ] **Step 2: Write detailed report file**

```markdown
# Entity Identifier Implementation Report

## Completed Scope
- Models: lstm, transformer, dlinear, patchtst, itransformer, timellm, gpt4ts
- Datasets: swiss-river-1990, swiss-river-2010, swiss-river-zurich, traffic, electricity, PEMS03, PEMS04, PEMS07, PEMS08, exchange_rate
- Modes: none, embedding, onehot

## Verification Commands and Outcomes
| Command | Outcome |
|---|---|
| `python experiments/entity_identifier/generate_configs.py --dry-run` | PASS |
| `python experiments/entity_identifier/run.py --dry-run` | PASS |
| `pytest tests/runtime/test_entity_identifier_pipeline.py -v` | PASS |
| `pytest tests/models/torch/test_entity_integration.py tests/models/torch/test_lstm.py tests/models/torch/test_gpt4ts.py -v` | PASS |
| `pytest -m "not slow and not download" -v` | PASS |

## Cluster Usage
- Example submit command and generated sbatch snippet

## TimeLLM/GPT4TS Weight Instructions
- export LIULIAN_ALLOW_LLM_DOWNLOAD=1
- run model-specific smoke command
```

- [ ] **Step 3: Run formatting/lint on touched surfaces**

Run:

```bash
ruff format liulian/ tests/ experiments/entity_identifier/
ruff check --fix liulian/ tests/ experiments/entity_identifier/
```

Expected: no formatting/lint errors.

- [ ] **Step 4: Final commit**

```bash
git add liulian/ experiments/entity_identifier/ tests/ docs/research/2026-04-10-entity-identifier-implementation-report.md
git commit -m "feat: implement entity-identifier forecasting matrix with local and slurm workflows"
```

---

## Spec Coverage (self-check)

1. Embedding first, then onehot: **Tasks 1–10**.
2. Model order (`lstm`, `transformer`, `dlinear`, `patchtst`, `itransformer`, `timellm`, `gpt4ts`): **Tasks 2–6 in order**.
3. Dataset scope/order: **Task 8 matrix constants + Task 7 plumbing**.
4. New entrypoint under `experiments/entity_identifier/`: **Tasks 8–9**.
5. Comparison among `none`, `embedding`, `onehot`: **Task 8 compare**.
6. Dry tests + dev tests: **Task 10**.
7. Slurm integration reused/refined from existing style: **Task 9**.
8. Detailed final report file: **Task 10**.
