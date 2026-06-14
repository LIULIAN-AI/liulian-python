"""Equivalence: model-layer transparent injection == data-layer bake-in.

`EntityTransparentWrapper` (id_injection='model') must feed the inner model
BITWISE-IDENTICAL inputs to the legacy data-layer bake-in
(`make_entity_features` concatenated at window-build time). This is the
contract that keeps wrapper-injected runs comparable with the data-injected
baselines (swiss3 2026-06-11, swiss3dt 2026-06-12).
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from liulian.data.ts.timeseriesdataset import make_entity_features
from liulian.models.torch.entity_mixin import EntityTransparentWrapper

IDS = ['alpha', 'beta', 'gamma', 'delta']
COORDS = {
    'alpha': (600000.0, 200000.0),
    'beta': (610000.0, 210000.0),
    'gamma': (620000.0, 190000.0),
    'delta': (605000.0, 205000.0),
}
MODES = ('onehot', 'sinusoidal', 'random', 'coordinates', 'numeric_id')


class _CaptureInner(nn.Module):
    """Stub inner model that records the x_enc it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.last_x: torch.Tensor | None = None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        self.last_x = x_enc
        return x_enc[..., :1]


def _wrapper(mode: str) -> EntityTransparentWrapper:
    return EntityTransparentWrapper(
        _CaptureInner(),
        mode,
        IDS,
        coordinates=COORDS if mode == 'coordinates' else None,
        sinusoidal_dim=8,
        random_dim=8,
        random_seed=7,
    )


@pytest.mark.parametrize('mode', MODES)
def test_feature_table_matches_data_layer_bakein(mode: str) -> None:
    """Table row i must equal the data-layer block for station i."""
    w = _wrapper(mode)
    for i, name in enumerate(IDS):
        baked = make_entity_features(
            name,
            IDS,
            mode,
            seq_len=5,
            coordinates=COORDS if mode == 'coordinates' else None,
            sinusoidal_dim=8,
            random_dim=8,
            random_seed=7,
        )
        assert baked is not None
        assert torch.equal(w.feature_table[i].unsqueeze(0).expand(5, -1), baked)


@pytest.mark.parametrize('mode', MODES)
def test_forward_concat_layout_matches_bakein(mode: str) -> None:
    """Wrapper concat = [base block | identifier block], same as bake-in."""
    w = _wrapper(mode)
    B, T, base = 3, 6, 2
    x = torch.randn(B, T, base)
    eidx = torch.tensor([2, 0, 3])
    w(x, entity_ids=eidx)
    out = w.inner.last_x
    assert out is not None
    assert out.shape == (B, T, base + w.feature_dim)
    assert torch.equal(out[..., :base], x)
    for b in range(B):
        assert torch.equal(out[b, :, base:], w.feature_table[eidx[b]].unsqueeze(0).expand(T, -1))


def test_requires_entity_ids() -> None:
    """No silent fallback: missing entity ids must raise, not zero-fill."""
    w = _wrapper('onehot')
    with pytest.raises(ValueError, match='entity_ids'):
        w(torch.randn(2, 4, 1))


_ZURICH = Path(__file__).resolve().parents[3] / 'dataset' / 'swiss_river' / 'zurich_train.csv'


@pytest.mark.skipif(not _ZURICH.exists(), reason='swiss CSVs not present (gitignored)')
@pytest.mark.parametrize('mode', ('onehot', 'sinusoidal', 'random', 'coordinates'))
def test_swiss_dataset_both_injection_paths_identical(mode: str) -> None:
    """End-to-end on real zurich data: data-baked sample == wrapper output."""
    from liulian.data.swiss_river import SwissRiverDataset

    kw = dict(
        data_name='swiss-river-zurich',
        identifier_mode=mode,
        split_mode='per_entity',
        graph_mode='none',
        sinusoidal_dim=8,
        random_identifier_dim=8,
        random_identifier_seed=7,
        max_samples=50,
    )
    ds_data = SwissRiverDataset(id_injection='data', **kw)
    ds_model = SwissRiverDataset(id_injection='model', **kw)
    tr_d = ds_data.get_split('train')
    tr_m = ds_model.get_split('train')
    assert len(tr_d) == len(tr_m)

    w = EntityTransparentWrapper(
        _CaptureInner(),
        mode,
        [str(s) for s in ds_model.station_ids],
        coordinates=(ds_model.topology.coordinates if ds_model.topology else None),
        sinusoidal_dim=8,
        random_dim=8,
        random_seed=7,
    )
    sid_to_idx = {str(s): i for i, s in enumerate(ds_model.station_ids)}
    for i in (0, len(tr_d) // 2, len(tr_d) - 1):
        baked_x = tr_d[i][0]
        item = tr_m[i]
        base_x, eid = item[0], item[4]
        w(base_x.unsqueeze(0), entity_ids=torch.tensor([sid_to_idx[str(eid)]]))
        assert w.inner.last_x is not None
        assert torch.equal(w.inner.last_x[0], baked_x)


# --------------------------------------------------------------------------- #
# ChannelTransparentWrapper (multi_channel) — algebraic _augment correctness   #
# --------------------------------------------------------------------------- #
from liulian.models.torch.entity_mixin import ChannelTransparentWrapper  # noqa: E402


class _IdentInner(nn.Module):
    """Returns x_enc unchanged so tests can read the augmented tensor."""

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return x_enc


def _materializing_augment(x, proj, feats):
    """The pre-2026-06-14 O(B·T·N²) reference implementation."""
    B, T, N = x.shape
    f = feats.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
    aug = torch.cat([x.unsqueeze(-1), f], dim=-1)
    return proj(aug).squeeze(-1)


@pytest.mark.parametrize('mode', ['onehot', 'sinusoidal', 'random', 'coordinates'])
def test_channel_augment_matches_materializing_reference(mode: str) -> None:
    """The algebraic _augment must equal the old materializing form."""
    torch.manual_seed(0)
    w = ChannelTransparentWrapper(
        _IdentInner(),
        mode,
        num_stations=len(IDS),
        coordinates=COORDS if mode == 'coordinates' else None,
        sinusoidal_dim=8,
        random_dim=8,
        random_seed=7,
        station_ids=IDS,
    )
    x = torch.randn(3, 6, len(IDS))
    got = w._augment(x, w.enc_proj)
    ref = _materializing_augment(x, w.enc_proj, w.channel_features)
    assert torch.allclose(got, ref, atol=1e-5)


def test_channel_onehot_does_not_materialize_n_squared() -> None:
    """onehot on a big channel count must not allocate an (N,N)-per-element
    tensor — the algebraic path keeps peak work O(B·T·N)."""
    N = 800  # ~ traffic scale; the old path would build (B,T,N,1+N)
    w = ChannelTransparentWrapper(
        _IdentInner(),
        'onehot',
        num_stations=N,
        station_ids=[str(i) for i in range(N)],
    )
    x = torch.randn(8, 16, N)
    out = w(x)  # would OOM-scale under the materializing path
    assert out.shape == (8, 16, N)
    assert torch.isfinite(out).all()
