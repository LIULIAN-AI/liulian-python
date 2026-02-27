"""End-to-end pipeline tests for the Swiss River 1990 dataset.

Four scenarios are tested using the *real* Swiss River 1990 data through
the full ``run_experiment`` pipeline (config → dataset → model → train →
eval → metrics → predictions):

1. **Single run with embedding** (``identifier_mode='embedding'``, ``hpo=False``)
2. **Single run without embedding** (``identifier_mode='none'``, ``hpo=False``)
3. **Ray Tune with embedding** (``identifier_mode='embedding'``, ``hpo=True``)
4. **Ray Tune without embedding** (``identifier_mode='none'``, ``hpo=True``)

Each test asserts:
- Output shapes (predictions, ground truth, metrics dict structure)
- Numerical metric values against hard-recorded baselines (within tolerance)
- First few predicted values match recorded baselines

These tests are gated behind ``@pytest.mark.main_branch`` so they only
run on PR/push to main (``pytest -m main_branch``).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.main_branch

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__), '..', '..', 'dataset', 'swiss_river'
)


# ── Shared helpers ──────────────────────────────────────────────────────


def _base_config(identifier_mode: str, hpo: bool) -> dict:
    """Create config for e2e tests using the full pipeline.

    Uses the REAL Swiss River 1990 dataset but with tiny slices and a
    minimal model for speed (~1 s per scenario).  Every pipeline stage
    is exercised — config, dataset loading, scaling, model build, train
    loop, eval loop, metrics, predictions — nothing is skipped:

    - ``batch_size=16, max_train_iters=5`` — 80 train samples/epoch
    - ``max_eval_iters=5`` — 80 val/test samples evaluated
    - ``seq_len=10, pred_len=3`` — short windows
    - ``train_epochs=1, d_model=16, e_layers=1`` — minimal LSTM
    - ``hpo_num_samples=2`` — minimal HPO trials
    - CPU-only — avoids GPU variance and setup overhead
    """
    from liulian.config import load_config

    cfg = load_config()  # start from DEFAULT_CONFIG
    cfg.update(
        # Data — real swiss-river-1990, tiny slices via iter caps
        data='swiss-river-1990',
        seq_len=10,
        pred_len=3,
        split_mode='ts',
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
        id_integration='concat_to_x',
        embedding_size=4,
        # Graph
        graph_mode='none',
        # Model — minimal but complete (no layers/stages skipped)
        model='lstm',
        d_model=16,
        e_layers=1,
        enc_in=None,  # auto-detect
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
    return cfg


# ── Hard-recorded baselines ─────────────────────────────────────────────
# Recorded with seed=2026, train_epochs=1, d_model=16, e_layers=1,
# batch_size=16, max_train_iters=5, max_eval_iters=5, seq_len=10,
# pred_len=3, embedding_size=4, CPU-only.

BASELINES: dict[str, dict] = {
    'single_emb': {
        'pred_shape': (80, 3, 1),
        'test_mse': 0.010576223023235798,
        'test_rmse': 0.10278971493244171,
        'test_mae': 0.09050349444150925,
        'test_nse': -22.188804817199706,
        'pred_first5': [
            0.24317395687103271,
            0.01247786357998848,
            0.07042207568883896,
            0.2431498020887375,
            0.012451037764549255,
        ],
    },
    'single_no_emb': {
        'pred_shape': (80, 3, 1),
        'test_mse': 0.010558596625924111,
        'test_rmse': 0.1026900440454483,
        'test_mae': 0.08943474292755127,
        'test_nse': -22.051172065734864,
        'pred_first5': [
            0.2431810051202774,
            0.014721062034368515,
            0.07161492109298706,
            0.24325557053089142,
            0.014784488826990128,
        ],
    },
    'tune_emb': {
        'pred_shape': (80, 3, 1),
        'test_mse': 0.31837276220321653,
        'test_rmse': 0.5638174653053284,
        'test_mae': 0.5606559753417969,
        'test_nse': -695.2455627441407,
        'pred_first5': [
            0.5937272310256958,
            0.7241061925888062,
            0.6904590129852295,
            0.5936862230300903,
            0.7240447998046875,
        ],
    },
    'tune_no_emb': {
        'pred_shape': (80, 3, 1),
        'test_mse': 0.23910034000873565,
        'test_rmse': 0.48849965929985045,
        'test_mae': 0.4870070040225983,
        'test_nse': -521.6271301269531,
        'pred_first5': [
            0.5881748199462891,
            0.5626541972160339,
            0.6376810669898987,
            0.5880754590034485,
            0.5625452399253845,
        ],
    },
}

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
    # artifacts_dir is relative to tmp_path (the CWD during run_experiment)
    artifacts_dir = os.path.join(str(tmp_path), summary['artifacts_dir'])
    assert os.path.isdir(artifacts_dir), (
        f'Artifacts dir does not exist: {artifacts_dir}'
    )

    # ---- Metrics must be present ----
    assert 'metrics' in summary, 'Summary missing "metrics" key'
    metrics = summary['metrics']
    assert 'final_test' in metrics, 'Metrics missing "final_test" key'
    final_test = metrics['final_test']

    for required_key in ('mse', 'rmse', 'mae', 'nse'):
        assert required_key in final_test, f'final_test missing "{required_key}" key'
        assert np.isfinite(final_test[required_key]), (
            f'final_test["{required_key}"] = {final_test[required_key]} is not finite'
        )

    result: dict = {
        'summary': summary,
        'test_mse': final_test['mse'],
        'test_rmse': final_test['rmse'],
        'test_mae': final_test['mae'],
        'test_nse': final_test['nse'],
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


# ── Baseline assertion (shared by all 4 tests) ─────────────────────────


def _assert_baseline(result: dict, bl: dict, scenario: str) -> None:
    """Assert all result values match the recorded baseline.

    Every comparison is a hard assertion — no silent passes.
    If baselines have not been recorded yet (all ``None``), the test
    **fails** with an informative message showing the values to record.
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


# ── Tests ───────────────────────────────────────────────────────────────


class TestSingleRunEmbedding:
    """Scenario 1: Single training run with learnable entity embedding."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(identifier_mode='embedding', hpo=False)
        result = _run_and_collect(cfg, tmp_path)
        bl = BASELINES['single_emb']
        _assert_baseline(result, bl, 'single_emb')


class TestSingleRunNoEmbedding:
    """Scenario 2: Single training run without entity embedding."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(identifier_mode='none', hpo=False)
        result = _run_and_collect(cfg, tmp_path)
        bl = BASELINES['single_no_emb']
        _assert_baseline(result, bl, 'single_no_emb')


class TestTuneEmbedding:
    """Scenario 3: Ray Tune HPO with learnable entity embedding."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(identifier_mode='embedding', hpo=True)
        result = _run_and_collect(cfg, tmp_path)
        bl = BASELINES['tune_emb']
        _assert_baseline(result, bl, 'tune_emb')


class TestTuneNoEmbedding:
    """Scenario 4: Ray Tune HPO without entity embedding."""

    def test_pipeline(self, tmp_path):
        cfg = _base_config(identifier_mode='none', hpo=True)
        result = _run_and_collect(cfg, tmp_path)
        bl = BASELINES['tune_no_emb']
        _assert_baseline(result, bl, 'tune_no_emb')
