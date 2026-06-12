"""Hard-recorded baselines for e2e pipeline anchor tests.

Each dict maps *scenario name* → expected metric values.
Recorded with:
    seed=2026, train_epochs=1, d_model=16, e_layers=1,
    batch_size=16, max_train_iters=5, max_eval_iters=5,
    seq_len=10, pred_len=3, embedding_size=4, CPU-only.

The tolerance used by ``_assert_baseline`` is ``atol=1e-4, rtol=1e-3``.

To re-record, run::

    .venv/bin/python _record_baselines.py <scenario_name>

then paste the output here.

Models covered
--------------
* LSTM + Swiss River 1990 (per_entity)
* DLinear + Swiss River 1990 (multi_channel)
* PatchTST + Swiss River 1990 (multi_channel)
* Informer + Swiss River 1990 (multi_channel)
* Autoformer + Swiss River 1990 (multi_channel)
* FEDformer + Swiss River 1990 (multi_channel)
* TimesNet + Swiss River 1990 (multi_channel)
* Transformer + Swiss River 1990 (multi_channel)
* iTransformer + Swiss River 1990 (multi_channel)
* TimeMixer + Swiss River 1990 (multi_channel)
* TimeXer + Swiss River 1990 (multi_channel)
* Mamba + Swiss River 1990 (multi_channel)
* Nonstationary Transformer + Swiss River 1990 (multi_channel)
* LightTS + Swiss River 1990 (multi_channel)
* Reformer + Swiss River 1990 (multi_channel)
* GPT4TS + Swiss River 1990 (multi_channel)
"""

from __future__ import annotations


def _placeholder(n: int = 4) -> dict[str, dict]:
    """Return *n* placeholder scenario dicts with ``test_mse=None``.

    When ``test_mse is None``, ``_assert_baseline`` fails with an
    informative message asking the user to record baselines.
    """
    keys = ['single_no_emb', 'single_emb', 'tune_no_emb', 'tune_emb']
    return {
        k: {
            'pred_shape': None,
            'test_mse': None,
            'test_rmse': None,
            'test_mae': None,
            'test_nse': None,
            'pred_first5': None,
        }
        for k in keys[:n]
    }


# ── LSTM + Swiss River 1990 + per_entity mode ──────────────────────────────

LSTM_SWISS1990: dict[str, dict] = {
    'single_emb': {
        'pred_shape': (80, 3, 1),
        'test_mse': 0.010576223023235798,
        'test_rmse': 0.10278971493244171,
        'test_mae': 0.09050349444150925,
        'test_nse': -22.188804817199706,
        'pred_first5': [
            8.430373191833496,
            3.3735146522521973,
            4.643651962280273,
            8.42984390258789,
            3.372926712036133,
        ],
    },
    'single_no_emb': {
        'pred_shape': (80, 3, 1),
        'test_mse': 0.010558596625924111,
        'test_rmse': 0.1026900440454483,
        'test_mae': 0.08943474292755127,
        'test_nse': -22.051172065734864,
        'pred_first5': [
            8.430527687072754,
            3.4226858615875244,
            4.669799327850342,
            8.432162284851074,
            3.4240758419036865,
        ],
    },
    'tune_emb': {
        'pred_shape': (80, 3, 1),
        'test_mse': 0.31837276220321653,
        'test_rmse': 0.5638174653053284,
        'test_mae': 0.5606559753417969,
        'test_nse': -695.2455627441407,
        'pred_first5': [
            16.114500045776367,
            18.972408294677734,
            18.234861373901367,
            16.113601684570312,
            18.97106170654297,
        ],
    },
    'tune_no_emb': {
        'pred_shape': (80, 3, 1),
        'test_mse': 0.23910034000873565,
        'test_rmse': 0.48849965929985045,
        'test_mae': 0.4870070040225983,
        'test_nse': -521.6271301269531,
        'pred_first5': [
            15.992792129516602,
            15.433380126953125,
            17.07796859741211,
            15.99061393737793,
            15.430991172790527,
        ],
    },
}

# ── DLinear + Swiss River 1990 + multi_channel mode ─────────────────────────

DLINEAR_SWISS1990: dict[str, dict] = {
    'single_no_emb': {
        'pred_shape': (80, 3, 28),
        'test_mse': 0.09372462928295136,
        'test_rmse': 0.3058091998100281,
        'test_mae': 0.26276912093162536,
        'test_nse': -9.494775199890137,
        'pred_first5': [
            15.136460304260254,
            15.160778045654297,
            14.184622764587402,
            14.622276306152344,
            16.056987762451172,
        ],
    },
    'single_emb': {
        'pred_shape': (80, 3, 28),
        'test_mse': 0.36586005687713624,
        'test_rmse': 0.6047032117843628,
        'test_mae': 0.4789117395877838,
        'test_nse': -40.19646301269531,
        'pred_first5': [
            -4.652027606964111,
            -6.958632469177246,
            5.898652076721191,
            12.108003616333008,
            -0.09789279103279114,
        ],
    },
    'tune_no_emb': {
        'pred_shape': (80, 3, 28),
        'test_mse': 0.014668050967156888,
        'test_rmse': 0.1194050133228302,
        'test_mae': 0.09226270616054535,
        'test_nse': -0.45391130447387695,
        'pred_first5': [
            7.797860145568848,
            6.961730003356934,
            6.833188056945801,
            6.49877405166626,
            8.218507766723633,
        ],
    },
    'tune_emb': {
        'pred_shape': (80, 3, 28),
        'test_mse': 0.1541128635406494,
        'test_rmse': 0.39256036281585693,
        'test_mae': 0.31192893385887144,
        'test_nse': -16.59686908721924,
        'pred_first5': [
            16.439678192138672,
            8.605259895324707,
            16.777732849121094,
            7.275578022003174,
            9.26685619354248,
        ],
    },
}


# ── PatchTST + Swiss River 1990 + multi_channel mode ──────────────────────

PATCHTST_SWISS1990: dict[str, dict] = {
    'single_no_emb': {
        'pred_shape': (80, 3, 28),
        'test_mse': 0.02560381405055523,
        'test_rmse': 0.15877650678157806,
        'test_mae': 0.14092261344194412,
        'test_nse': -1.6908583164215087,
        'pred_first5': [
            10.78228759765625,
            10.378812789916992,
            9.913472175598145,
            9.902558326721191,
            11.501137733459473,
        ],
    },
    'single_emb': {
        'pred_shape': (80, 3, 28),
        'test_mse': 0.032259497046470645,
        'test_rmse': 0.17842237651348114,
        'test_mae': 0.15585381984710694,
        'test_nse': -2.4686111211776733,
        'pred_first5': [
            11.668331146240234,
            9.960518836975098,
            10.728205680847168,
            10.555916786193848,
            11.630074501037598,
        ],
    },
    'tune_no_emb': {
        'pred_shape': (80, 3, 28),
        'test_mse': 0.03555614352226257,
        'test_rmse': 0.18685804009437562,
        'test_mae': 0.16769659519195557,
        'test_nse': -2.723482084274292,
        'pred_first5': [
            11.692076683044434,
            11.347626686096191,
            11.126836776733398,
            11.243354797363281,
            12.396095275878906,
        ],
    },
    'tune_emb': {
        'pred_shape': (80, 3, 28),
        'test_mse': 0.022381814196705818,
        'test_rmse': 0.14865126162767411,
        'test_mae': 0.13340612947940828,
        'test_nse': -1.4086602687835694,
        'pred_first5': [
            10.551055908203125,
            9.899022102355957,
            9.098849296569824,
            9.382139205932617,
            10.257912635803223,
        ],
    },
}


# ── Informer + Swiss River 1990 + multi_channel mode ──────────────────────

INFORMER_SWISS1990: dict[str, dict] = _placeholder()


# ── Autoformer + Swiss River 1990 + multi_channel mode ─────────────────────

AUTOFORMER_SWISS1990: dict[str, dict] = _placeholder()


# ── FEDformer + Swiss River 1990 + multi_channel mode ──────────────────────

FEDFORMER_SWISS1990: dict[str, dict] = _placeholder()


# ── TimesNet + Swiss River 1990 + multi_channel mode ───────────────────────

TIMESNET_SWISS1990: dict[str, dict] = _placeholder()


# ── Transformer + Swiss River 1990 + multi_channel mode ────────────────────

TRANSFORMER_SWISS1990: dict[str, dict] = _placeholder()


# ── iTransformer + Swiss River 1990 + multi_channel mode ───────────────────

ITRANSFORMER_SWISS1990: dict[str, dict] = _placeholder()


# ── TimeMixer + Swiss River 1990 + multi_channel mode ──────────────────────

TIMEMIXER_SWISS1990: dict[str, dict] = _placeholder()


# ── TimeXer + Swiss River 1990 + multi_channel mode ───────────────────────

TIMEXER_SWISS1990: dict[str, dict] = _placeholder()


# ── Mamba + Swiss River 1990 + multi_channel mode ─────────────────────────

MAMBA_SWISS1990: dict[str, dict] = _placeholder()


# ── Nonstationary Transformer + Swiss River 1990 + multi_channel mode ──────

NONSTATIONARY_TRANSFORMER_SWISS1990: dict[str, dict] = _placeholder()


# ── LightTS + Swiss River 1990 + multi_channel mode ───────────────────────

LIGHTTS_SWISS1990: dict[str, dict] = _placeholder()


# ── Reformer + Swiss River 1990 + multi_channel mode ──────────────────────

REFORMER_SWISS1990: dict[str, dict] = _placeholder()


# ── GPT4TS + Swiss River 1990 + multi_channel mode ────────────────────────

GPT4TS_SWISS1990: dict[str, dict] = _placeholder()
