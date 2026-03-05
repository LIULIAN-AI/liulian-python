"""One-shot script to record baseline values for e2e pipeline tests.

Run: .venv/bin/python _record_baselines.py
"""
import os
import sys
import tempfile

import numpy as np


def _base_config(identifier_mode, hpo, model='lstm', split_mode='per_entity'):
    from liulian.config import load_config

    cfg = load_config()
    cfg.update(
        data='swiss-river-1990',
        seq_len=10,
        pred_len=3,
        split_mode=split_mode,
        scaler='minmax',
        train_split=0.8,
        task='forecast',
        use_current_x=True,
        use_full_history=False,
        short_subsequence_method='drop',
        gap_mode='split',
        max_mask_consecutive=10,
        noise_type=None,
        include_historical_y='none',
        include_historical_predicted_y=False,
        identifier_mode=identifier_mode,
        id_integration='concat_to_x',
        embedding_size=4,
        graph_mode='none',
        model=model,
        d_model=16,
        e_layers=1,
        enc_in=None,
        batch_size=16,
        max_train_iters=5,
        max_eval_iters=5,
        train_epochs=1,
        learning_rate=0.001,
        loss='mse',
        metrics='rmse,mae,nse',
        patience=5,
        lradj='none',
        num_workers=0,
        show_progress=False,
        eval_denorm=True,
        wandb_project=None,
        dev_run=True,
        hpo=hpo,
        hpo_num_samples=2 if hpo else 0,
        hpo_local_mode=True,
        hpo_grace_period=1,
        hpo_reduction_factor=2,
        hpo_resources_cpu=1,
        hpo_resources_gpu=0,
        hpo_save_checkpoints=True,
        hpo_trim_checkpoints=False,
        auto_viz=False,
        seed=2026,
        quick_test=False,
    )
    return cfg


def run_scenario(name, cfg):
    from liulian.pipeline import run_experiment

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            summary = run_experiment(cfg)
        finally:
            os.chdir(old_cwd)

    ft = summary['metrics']['final_test']
    preds = summary['predictions']['preds']
    flat = preds.numpy().flatten()[:5].tolist()
    return {
        'pred_shape': tuple(preds.shape),
        'test_mse': ft['mse'],
        'test_rmse': ft['rmse'],
        'test_mae': ft['mae'],
        'test_nse': ft['nse'],
        'pred_first5': flat,
    }


SCENARIOS = {
    # LSTM scenarios
    'lstm_single_emb': lambda: _base_config('embedding', False),
    'lstm_single_no_emb': lambda: _base_config('none', False),
    'lstm_tune_emb': lambda: _base_config('embedding', True),
    'lstm_tune_no_emb': lambda: _base_config('none', True),
    # DLinear scenarios (no embedding)
    'dlinear_single_no_emb': lambda: _base_config('none', False, model='dlinear', split_mode='multi_channel'),
    'dlinear_tune_no_emb': lambda: _base_config('none', True, model='dlinear', split_mode='multi_channel'),
    # DLinear scenarios (with channel embedding)
    'dlinear_single_emb': lambda: _base_config('embedding', False, model='dlinear', split_mode='multi_channel'),
    'dlinear_tune_emb': lambda: _base_config('embedding', True, model='dlinear', split_mode='multi_channel'),
    # PatchTST scenarios (multi_channel mode)
    'patchtst_single_no_emb': lambda: _base_config('none', False, model='patchtst', split_mode='multi_channel'),
    'patchtst_single_emb': lambda: _base_config('embedding', False, model='patchtst', split_mode='multi_channel'),
    'patchtst_tune_no_emb': lambda: _base_config('none', True, model='patchtst', split_mode='multi_channel'),
    'patchtst_tune_emb': lambda: _base_config('embedding', True, model='patchtst', split_mode='multi_channel'),
    # PatchTST + patch-level entity embedding
    'patchtst_entity_single': lambda: _base_config('none', False, model='patchtst_entity', split_mode='multi_channel'),
    'patchtst_entity_tune': lambda: _base_config('none', True, model='patchtst_entity', split_mode='multi_channel'),
}


if __name__ == '__main__':
    # Run a specific scenario or all
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(SCENARIOS.keys())
    for name in targets:
        if name not in SCENARIOS:
            print(f'Unknown scenario: {name}')
            continue
        print(f'\n{"=" * 60}')
        print(f'Running: {name}')
        print(f'{"=" * 60}')
        cfg = SCENARIOS[name]()
        result = run_scenario(name, cfg)
        print(f'\nBASELINE for {name}:')
        for k, v in result.items():
            print(f'    {k!r}: {v!r},')
