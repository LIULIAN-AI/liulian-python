"""End-to-end pipeline anchor tests for all model × dataset combinations.

Every model+dataset combination exercised through the full
``run_experiment`` pipeline (config → dataset → model → train → eval →
metrics → predictions) with hard-recorded numeric baselines.

Currently anchored
------------------
* **LSTM + Swiss River 1990** (per_entity mode) — 4 scenarios
  (single/tune × embedding/none)
* **DLinear + Swiss River 1990** (multi_channel mode) — 4 scenarios
  (single/tune × embedding/none)

Baselines live in :pymod:`tests.e2e.baselines` so they can be imported
by helper scripts and kept out of test logic.

Adding a new model
------------------
1. Record baselines: ``python _record_baselines.py <scenario>``
2. Add the dict to ``tests/e2e/baselines.py``
3. Add a ``Test<Model>Swiss1990`` class below with the appropriate
   ``SCENARIOS`` and ``BASELINES`` attributes.

These tests are gated behind ``@pytest.mark.main_branch`` so they only
run on PR/push to main (``pytest -m main_branch``).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.e2e.baselines import DLINEAR_SWISS1990, LSTM_SWISS1990, PATCHTST_SWISS1990

pytestmark = pytest.mark.main_branch

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), '..', '..', 'dataset', 'swiss_river'
)


# ── Shared helpers ──────────────────────────────────────────────────────


def _base_config(
    model: str = 'lstm',
    split_mode: str = 'per_entity',
    identifier_mode: str = 'none',
    hpo: bool = False,
    id_integration: str | None = None,
    **overrides: object,
) -> dict:
    """Generic e2e config builder for any model + Swiss River 1990.

    Uses the REAL Swiss River 1990 dataset but with tiny slices and a
    minimal model for speed (~1–2 s per scenario).  Every pipeline stage
    is exercised — config, dataset loading, scaling, model build, train
    loop, eval loop, metrics, predictions — nothing is skipped:

    - ``batch_size=16, max_train_iters=5`` — 80 train samples/epoch
    - ``max_eval_iters=5`` — 80 val/test samples evaluated
    - ``seq_len=10, pred_len=3`` — short windows
    - ``train_epochs=1, d_model=16, e_layers=1`` — minimal model
    - ``hpo_num_samples=2`` — minimal HPO trials
    - CPU-only — avoids GPU variance and setup overhead
    """
    from liulian.config import load_config

    if id_integration is None:
        id_integration = (
            'add_after_patch'
            if model == 'patchtst' and identifier_mode == 'embedding'
            else 'concat_to_x'
        )

    cfg = load_config()  # start from DEFAULT_CONFIG
    cfg.update(
        # Data — real swiss-river-1990, tiny slices via iter caps
        data='swiss-river-1990',
        seq_len=10,
        pred_len=3,
        split_mode=split_mode,
        scaler='minmax',
        train_split=0.8,
        # Task
        task='forecast',
        use_current_x=True,
        use_full_history=False,
        short_subsequence_method='drop',
        gap_mode='split',
        max_mask_consecutive=10,
        # Noise
        noise_type=None,
        # Historical targets
        include_historical_y='none',
        include_historical_predicted_y=False,
        # Entity
        identifier_mode=identifier_mode,
        id_integration=id_integration,
        embedding_size=4,
        # Graph
        graph_mode='none',
        # Model — minimal but complete (no layers/stages skipped)
        model=model,
        d_model=16,
        e_layers=1,
        enc_in=None,  # auto-detect
        individual=False,
        moving_avg=25,
        # Training — tiny slices, still exercises every stage
        batch_size=16,
        max_train_iters=5,  # 5 × 16 = 80 train samples
        max_eval_iters=5,  # 5 × 16 = 80 val/test samples
        train_epochs=1,
        learning_rate=0.001,
        loss='mse',
        metrics='rmse,mae,nse',
        patience=5,
        lradj='none',
        num_workers=0,
        show_progress=False,
        eval_denorm=True,
        # Logging
        wandb_project=None,
        dev_run=True,
        # HPO
        hpo=hpo,
        hpo_num_samples=2 if hpo else 0,
        hpo_local_mode=True,
        hpo_grace_period=1,
        hpo_reduction_factor=2,
        hpo_resources_cpu=1,
        hpo_resources_gpu=0,
        hpo_save_checkpoints=True,
        hpo_trim_checkpoints=False,
        # Viz
        auto_viz=False,
        # Seed
        seed=2026,
        # Quick — MUST be False so no shortcuts are applied
        quick_test=False,
    )
    # Per-scenario or per-model extras
    cfg.update(overrides)
    return cfg


# Tolerance for floating-point comparison
_ATOL = 1e-4  # absolute tolerance
_RTOL = 1e-3  # relative tolerance


def _run_and_collect(cfg: dict, tmp_path) -> dict:
    """Run the pipeline and return metrics + first few predictions.

    Uses *tmp_path* as CWD so artifacts land in a temp dir.
    Every expected output is **hard-asserted** here so that a missing
    field causes an immediate, loud failure — never a silent pass.
    """
    import os

    from liulian.pipeline import run_experiment

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        summary = run_experiment(cfg)
    finally:
        os.chdir(old_cwd)

    # ---- Status must be 'ok' ----
    assert summary['status'] == 'ok', (
        f'Experiment failed with status={summary["status"]}'
    )

    # ---- Artifacts directory must exist ----
    artifacts_dir = os.path.join(str(tmp_path), summary['artifacts_dir'])
    assert os.path.isdir(artifacts_dir), (
        f'Artifacts dir does not exist: {artifacts_dir}'
    )

    # ---- Metrics must be present ----
    assert 'metrics' in summary, 'Summary missing "metrics" key'
    metrics = summary['metrics']
    assert 'test' in metrics, 'Metrics missing "test" key'
    test_metrics = metrics['test']

    for required_key in ('mse', 'rmse', 'mae', 'nse'):
        assert required_key in test_metrics, f'test metrics missing "{required_key}" key'
        assert np.isfinite(test_metrics[required_key]), (
            f'test["{required_key}"] = {test_metrics[required_key]} is not finite'
        )

    result: dict = {
        'summary': summary,
        'test_mse': test_metrics['mse'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_nse': test_metrics['nse'],
    }

    # ---- Predictions must be present ----
    assert 'predictions' in summary, 'Summary missing "predictions" key'
    preds_dict = summary['predictions']
    assert preds_dict is not None, 'predictions dict is None'
    for pkey in ('preds', 'trues', 'times'):
        assert pkey in preds_dict, f'predictions missing "{pkey}" key'

    preds = preds_dict['preds']
    trues = preds_dict['trues']
    assert preds.shape == trues.shape, (
        f'preds shape {preds.shape} != trues shape {trues.shape}'
    )
    result['pred_shape'] = tuple(preds.shape)

    # First 5 predictions flattened (for reproducibility check)
    flat = preds.numpy().flatten()
    assert len(flat) >= 5, f'Too few predictions: {len(flat)}'
    result['pred_first5'] = flat[:5].tolist()

    return result


# ── Baseline assertion ──────────────────────────────────────────────────


def _assert_baseline(result: dict, bl: dict, scenario: str) -> None:
    """Assert all result values match the recorded baseline.

    Every comparison is a hard assertion — no silent passes.
    If baselines have not been recorded yet (``test_mse is None``), the
    test **fails** with an informative message showing the values to
    record.
    """
    if bl['test_mse'] is None:
        pytest.fail(
            f'Baselines not yet recorded for {scenario}! Record these values:\n'
            f"  'pred_shape': {result['pred_shape']},\n"
            f"  'test_mse': {result['test_mse']!r},\n"
            f"  'test_rmse': {result['test_rmse']!r},\n"
            f"  'test_mae': {result['test_mae']!r},\n"
            f"  'test_nse': {result['test_nse']!r},\n"
            f"  'pred_first5': {result['pred_first5']!r},\n"
        )

    # Shape
    assert result['pred_shape'] == bl['pred_shape'], (
        f'Shape mismatch: {result["pred_shape"]} vs {bl["pred_shape"]}'
    )
    assert len(result['pred_shape']) == 3, 'Preds should be (N, pred_len, C)'
    assert result['pred_shape'][1] == 3, 'pred_len should be 3'

    # Metrics
    np.testing.assert_allclose(
        result['test_mse'],
        bl['test_mse'],
        atol=_ATOL,
        rtol=_RTOL,
        err_msg='MSE mismatch',
    )
    np.testing.assert_allclose(
        result['test_rmse'],
        bl['test_rmse'],
        atol=_ATOL,
        rtol=_RTOL,
        err_msg='RMSE mismatch',
    )
    np.testing.assert_allclose(
        result['test_mae'],
        bl['test_mae'],
        atol=_ATOL,
        rtol=_RTOL,
        err_msg='MAE mismatch',
    )
    np.testing.assert_allclose(
        result['test_nse'],
        bl['test_nse'],
        atol=_ATOL,
        rtol=_RTOL,
        err_msg='NSE mismatch',
    )

    # First 5 predicted values
    np.testing.assert_allclose(
        result['pred_first5'],
        bl['pred_first5'],
        atol=_ATOL,
        rtol=_RTOL,
        err_msg='First 5 prediction values mismatch',
    )


# ═══════════════════════════════════════════════════════════════════════
# LSTM + Swiss River 1990 (per_entity mode)
# ═══════════════════════════════════════════════════════════════════════


class TestLstmSingleEmb:
    """LSTM single run with learnable entity embedding."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='lstm', split_mode='per_entity',
            identifier_mode='embedding', hpo=False,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, LSTM_SWISS1990['single_emb'], 'lstm_single_emb')


class TestLstmSingleNoEmb:
    """LSTM single run without entity embedding."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='lstm', split_mode='per_entity',
            identifier_mode='none', hpo=False,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, LSTM_SWISS1990['single_no_emb'], 'lstm_single_no_emb')


class TestLstmTuneEmb:
    """LSTM Ray Tune HPO with learnable entity embedding."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='lstm', split_mode='per_entity',
            identifier_mode='embedding', hpo=True,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, LSTM_SWISS1990['tune_emb'], 'lstm_tune_emb')


class TestLstmTuneNoEmb:
    """LSTM Ray Tune HPO without entity embedding."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='lstm', split_mode='per_entity',
            identifier_mode='none', hpo=True,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, LSTM_SWISS1990['tune_no_emb'], 'lstm_tune_no_emb')


# ═══════════════════════════════════════════════════════════════════════
# DLinear + Swiss River 1990 (multi_channel mode)
# ═════════════════════════════════════════════════════════════════════


class TestDLinearSingleNoEmb:
    """DLinear single run (multi_channel mode, no embedding)."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='dlinear', split_mode='multi_channel',
            identifier_mode='none', hpo=False,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, DLINEAR_SWISS1990['single_no_emb'], 'dlinear_single_no_emb')


class TestDLinearSingleEmb:
    """DLinear single run (multi_channel mode, with channel embedding)."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='dlinear', split_mode='multi_channel',
            identifier_mode='embedding', hpo=False,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, DLINEAR_SWISS1990['single_emb'], 'dlinear_single_emb')


class TestDLinearTuneNoEmb:
    """DLinear Ray Tune HPO (multi_channel mode, no embedding)."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='dlinear', split_mode='multi_channel',
            identifier_mode='none', hpo=True,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, DLINEAR_SWISS1990['tune_no_emb'], 'dlinear_tune_no_emb')


class TestDLinearTuneEmb:
    """DLinear Ray Tune HPO (multi_channel mode, with channel embedding)."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='dlinear', split_mode='multi_channel',
            identifier_mode='embedding', hpo=True,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, DLINEAR_SWISS1990['tune_emb'], 'dlinear_tune_emb')


# ═══════════════════════════════════════════════════════════════════════
# PatchTST + Swiss River 1990 (multi_channel mode)
# ═══════════════════════════════════════════════════════════════════════


class TestPatchTSTSingleNoEmb:
    """PatchTST single run (multi_channel mode, no embedding)."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='patchtst', split_mode='multi_channel',
            identifier_mode='none', hpo=False,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, PATCHTST_SWISS1990['single_no_emb'], 'patchtst_single_no_emb')


class TestPatchTSTSingleEmb:
    """PatchTST single run (multi_channel mode, add-after-patch embedding)."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='patchtst', split_mode='multi_channel',
            identifier_mode='embedding', hpo=False,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, PATCHTST_SWISS1990['single_emb'], 'patchtst_single_emb')


class TestPatchTSTTuneNoEmb:
    """PatchTST Ray Tune HPO (multi_channel mode, no embedding)."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='patchtst', split_mode='multi_channel',
            identifier_mode='none', hpo=True,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, PATCHTST_SWISS1990['tune_no_emb'], 'patchtst_tune_no_emb')


class TestPatchTSTTuneEmb:
    """PatchTST Ray Tune HPO (multi_channel mode, add-after-patch embedding)."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(
            model='patchtst', split_mode='multi_channel',
            identifier_mode='embedding', hpo=True,
        )
        result = _run_and_collect(cfg, tmp_path)
        _assert_baseline(result, PATCHTST_SWISS1990['tune_emb'], 'patchtst_tune_emb')

