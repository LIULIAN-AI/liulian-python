"""Regression tests for the results.json builder (see docs/results_json.md).

Pins the `hpo.best_hparams` contract: before 2026-06-12 the experiment
summary stored the tuned config under `best_config` while the builder read
`best_hparams`, so every results.json shipped an EMPTY best_hparams dict.
"""

from liulian.pipeline import build_results_dict


def _summary(hpo: dict) -> dict:
    return {
        'metrics': {
            'test': {'rmse': 0.1},
            'hpo': hpo,
        },
        'artifacts_dir': 'artifacts/x',
    }


def test_best_hparams_lands_in_results_json() -> None:
    """The tuned config must appear in both hpo sections (was: empty)."""
    tuned = {'d_model': 83, 'e_layers': 2, 'learning_rate': 0.0038}
    res = build_results_dict(
        config={'data': 'swiss-river-1990', 'model': 'lstm'},
        summary=_summary(
            {
                'best_config': tuned,
                'best_hparams': tuned,
                'best_value': 0.0115,
                'n_trials': 50,
            }
        ),
        elapsed=1.0,
    )
    assert res['hpo']['best_hparams'] == tuned
    assert res['metrics']['hpo']['best_hparams'] == tuned


def test_best_hparams_falls_back_to_legacy_best_config_key() -> None:
    """Summaries carrying only the legacy `best_config` key still populate
    the documented `best_hparams` field."""
    tuned = {'d_model': 41}
    res = build_results_dict(
        config={},
        summary=_summary({'best_config': tuned, 'best_value': 0.01, 'n_trials': 50}),
        elapsed=1.0,
    )
    assert res['hpo']['best_hparams'] == tuned
