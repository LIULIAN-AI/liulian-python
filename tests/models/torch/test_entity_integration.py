"""Tests for EntityAwareMixin + EntityWrapper on TSL adapters.

Verifies that all 11 TSL adapters work with:
- 'none' mode (baseline, no entity features)
- 'embedding' mode (entity IDs in x_mark_enc, EntityWrapper wraps the model)
- 'onehot' mode (transparent — entity features already in x_enc)
"""

from __future__ import annotations

import pytest
import torch

from liulian.models.torch.entity_mixin import EntityAwareMixin, EntityWrapper


# ---- Adapter factory --------------------------------------------------------

# (adapter_cls_path, adapter_cls_name, extra_config)
_TSL_ADAPTERS: list[tuple[str, str, dict]] = [
    ('liulian.models.torch.dlinear', 'DLinearAdapter', {}),
    ('liulian.models.torch.transformer', 'TransformerAdapter', {}),
    ('liulian.models.torch.informer', 'InformerAdapter', {}),
    ('liulian.models.torch.autoformer', 'AutoformerAdapter', {}),
    ('liulian.models.torch.fedformer', 'FEDformerAdapter', {}),
    ('liulian.models.torch.itransformer', 'iTransformerAdapter', {}),
    ('liulian.models.torch.patchtst', 'PatchTSTAdapter', {}),
    ('liulian.models.torch.timesnet', 'TimesNetAdapter', {}),
    (
        'liulian.models.torch.timemixer',
        'TimeMixerAdapter',
        {
            'down_sampling_layers': 1,
            'down_sampling_window': 2,
        },
    ),
    ('liulian.models.torch.timexer', 'TimeXerAdapter', {'c_out': 1}),
]

# Mamba needs mamba_ssm which may not be installed
_HAS_MAMBA = False
try:
    import importlib

    _mamba_mod = importlib.import_module('mamba_ssm')
    _HAS_MAMBA = True
except ImportError:
    pass

if _HAS_MAMBA:
    _TSL_ADAPTERS.append(('liulian.models.torch.mamba_model', 'MambaAdapter', {}))


def _get_adapter_cls(module_path: str, cls_name: str):
    """Import and return adapter class."""
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


_ADAPTER_IDS = [t[1] for t in _TSL_ADAPTERS]


@pytest.fixture(params=_TSL_ADAPTERS, ids=_ADAPTER_IDS)
def adapter_spec(request):
    """Yield (adapter_cls, extra_config) for each TSL adapter."""
    mod_path, cls_name, extra = request.param
    cls = _get_adapter_cls(mod_path, cls_name)
    return cls, extra


# ---- Helpers ----------------------------------------------------------------

_BASE_CFG = {
    'seq_len': 32,
    'pred_len': 8,
    'enc_in': 4,
    'label_len': 16,
}


def _forward_adapter(adapter, batch_size: int = 2) -> torch.Tensor:
    """Run adapter.forward() with a minimal batch dict."""
    cfg = adapter._config
    seq_len = cfg.get('seq_len', 32)
    pred_len = cfg.get('pred_len', 8)
    enc_in = cfg.get('enc_in', 4)
    label_len = cfg.get('label_len', 16)

    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.zeros(batch_size, seq_len, 4)
    x_dec = torch.zeros(batch_size, label_len + pred_len, enc_in)
    x_mark_dec = torch.zeros(batch_size, label_len + pred_len, 4)

    batch = {
        'x_enc': x_enc,
        'x_mark_enc': x_mark_enc,
        'x_dec': x_dec,
        'x_mark_dec': x_mark_dec,
    }
    out = adapter.forward(batch)
    return out['predictions']


def _forward_model_direct(model, cfg, adapter_cfg=None, batch_size: int = 2) -> torch.Tensor:
    """Call model(x_enc, x_mark_enc, x_dec, x_mark_dec) directly (trainer path).

    For EntityWrapper models, the wrapper augments x_enc/x_dec internally,
    so we supply tensors with the *original* enc_in dimensions.
    """
    # Determine device from model parameters
    device = next(model.parameters()).device

    seq_len = cfg.get('seq_len', 32)
    pred_len = cfg.get('pred_len', 8)
    enc_in = cfg.get('enc_in', 4)
    label_len = cfg.get('label_len', 16)

    # If the model is an EntityWrapper, we need to supply original-sized
    # tensors (the wrapper will augment them).  For x_dec, enc-dec models
    # need dec_in features; use enc_in as fallback.
    dec_in = cfg.get('dec_in', enc_in)

    x_enc = torch.randn(batch_size, seq_len, enc_in, device=device)
    x_mark_enc = torch.zeros(batch_size, seq_len, 4, device=device)
    # Put valid station IDs in col 0 for embedding mode
    x_mark_enc[:, :, 0] = 3
    x_dec = torch.zeros(batch_size, label_len + pred_len, dec_in, device=device)
    x_mark_dec = torch.zeros(batch_size, label_len + pred_len, 4, device=device)

    model.eval()
    with torch.no_grad():
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    if isinstance(out, tuple):
        out = out[0]
    return out


# ---- Tests ------------------------------------------------------------------


class TestEntityAwareMixin:
    """Tests for EntityAwareMixin static helpers."""

    def test_none_mode_returns_same_config(self):
        cfg = {'enc_in': 7, 'identifier_mode': 'none'}
        result = EntityAwareMixin._entity_model_config(cfg)
        assert result is cfg  # same dict, no copy
        assert result['enc_in'] == 7

    def test_embedding_mode_returns_same_config(self):
        """With projection-based EntityWrapper, enc_in is NOT adjusted."""
        cfg = {
            'enc_in': 7,
            'identifier_mode': 'embedding',
            'embedding_size': 16,
        }
        result = EntityAwareMixin._entity_model_config(cfg)
        assert result is cfg  # no adjustment — wrapper handles it
        assert result['enc_in'] == 7

    def test_transparent_mode_returns_same_config(self):
        for mode in (
            'onehot',
            'coordinates',
            'sinusoidal',
            'random',
            'descriptors',
            'numeric_id',
        ):
            cfg = {'enc_in': 7, 'identifier_mode': mode}
            result = EntityAwareMixin._entity_model_config(cfg)
            assert result is cfg
            assert result['enc_in'] == 7


class TestEntityWrapper:
    """Tests for EntityWrapper nn.Module."""

    def test_embedding_injection(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 3)  # enc_in=4

            def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
                return self.proj(x_enc)

        model = FakeModel()
        # Wrapper projects (4 + 10) → 4, then FakeModel projects 4 → 3
        wrapper = EntityWrapper(model, enc_in=4, num_embeddings=10, embedding_size=10, entity_id_col=0)

        B, T, C = 2, 8, 4
        x_enc = torch.randn(B, T, C)
        x_mark = torch.zeros(B, T, 2)
        x_mark[:, :, 0] = 3  # station ID = 3

        out = wrapper(x_enc, x_mark)
        assert out.shape == (B, T, 3)
        assert torch.isfinite(out).all(), 'EntityWrapper embedding produced non-finite'

    def test_no_mark_passthrough(self):
        """Without x_mark_enc, wrapper should pass x_enc unchanged."""

        class Identity(torch.nn.Module):
            def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
                return x_enc

        wrapper = EntityWrapper(Identity(), enc_in=3, num_embeddings=5, embedding_size=4)
        x = torch.randn(2, 8, 3)
        out = wrapper(x)
        assert torch.equal(out, x)


class TestTSLAdaptersModeNone:
    """All TSL adapters work in 'none' mode (default — no entity features)."""

    def test_forward_none_mode(self, adapter_spec):
        cls, extra = adapter_spec
        cfg = {**_BASE_CFG, **extra, 'identifier_mode': 'none'}
        adapter = cls(cfg)
        pred = _forward_adapter(adapter)
        assert pred.ndim == 3
        assert torch.isfinite(pred).all(), 'None-mode forward produced non-finite'

    def test_inherits_mixin(self, adapter_spec):
        cls, extra = adapter_spec
        assert issubclass(cls, EntityAwareMixin)


class TestTSLAdaptersEmbeddingMode:
    """All TSL adapters work in 'embedding' mode.

    The EntityWrapper should inject embeddings from x_mark_enc col 0 into
    x_enc, and the model should produce valid output.
    """

    def test_adapter_forward_embedding(self, adapter_spec):
        cls, extra = adapter_spec
        cfg = {
            **_BASE_CFG,
            **extra,
            'identifier_mode': 'embedding',
            'embedding_size': 8,
            'num_embeddings': 20,
            'entity_id_col': 0,
        }
        adapter = cls(cfg)

        # Model should be wrapped
        assert isinstance(adapter._model, EntityWrapper)

        # Forward pass via adapter (inference path)
        B, seq_len = 2, cfg['seq_len']
        x_enc = torch.randn(B, seq_len, cfg['enc_in'])
        x_mark = torch.zeros(B, seq_len, 4)
        x_mark[:, :, 0] = 5  # station ID = 5
        label_len = cfg.get('label_len', 16)
        pred_len = cfg['pred_len']
        x_dec = torch.zeros(B, label_len + pred_len, cfg['enc_in'])
        x_mark_dec = torch.zeros(B, label_len + pred_len, 4)

        batch = {
            'x_enc': x_enc,
            'x_mark_enc': x_mark,
            'x_dec': x_dec,
            'x_mark_dec': x_mark_dec,
        }
        out = adapter.forward(batch)
        assert 'predictions' in out
        assert out['predictions'].ndim == 3
        assert torch.isfinite(out['predictions']).all(), 'Embedding adapter forward produced non-finite'

    def test_model_direct_embedding(self, adapter_spec):
        """Test trainer path: model(x_enc, x_mark, dec_inp, dec_mark)."""
        cls, extra = adapter_spec
        cfg = {
            **_BASE_CFG,
            **extra,
            'identifier_mode': 'embedding',
            'embedding_size': 8,
            'num_embeddings': 20,
            'entity_id_col': 0,
        }
        adapter = cls(cfg)
        # Trainer calls model directly as nn.Module
        # Use the original cfg (not adjusted) — EntityWrapper augments
        pred = _forward_model_direct(adapter._model, cfg)
        assert pred.ndim == 3
        assert torch.isfinite(pred).all(), 'Direct embedding model produced non-finite'


class TestTSLAdaptersTransparentMode:
    """TSL adapters with transparent entity modes (entity features in x_enc)."""

    def test_onehot_mode(self, adapter_spec):
        cls, extra = adapter_spec
        n_stations = 5
        # Entity dim = n_stations (one-hot), so enc_in = 4 + 5 = 9
        cfg = {
            **_BASE_CFG,
            **extra,
            'enc_in': _BASE_CFG['enc_in'] + n_stations,
            'identifier_mode': 'onehot',
        }
        adapter = cls(cfg)

        # Model should NOT be wrapped for transparent modes
        assert not isinstance(adapter._model, EntityWrapper)
        assert adapter._entity_mode == 'onehot'

        pred = _forward_adapter(adapter)
        assert pred.ndim == 3
        assert torch.isfinite(pred).all(), 'Onehot-mode forward produced non-finite'
