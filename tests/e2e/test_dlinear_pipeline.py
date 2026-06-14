"""Unit tests for DLinear pipeline generalisation.

Validates that the pipeline generalisation changes work correctly:
1. Config: MODEL_DEFAULTS['dlinear'] is applied, split_mode='multi_channel' flows through
2. Pipeline: build_model dispatches dlinear correctly, auto enc_in for multi_channel mode
3. Entity: ChannelEntityWrapper augments per-channel embeddings
4. Target names: dataset.info() returns correct target_names per split mode

E2E anchor tests with hard baselines live in ``test_e2e_pipeline.py``.
"""

from __future__ import annotations

import os

import pytest


def _torch_available():
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _torch_available(), reason='torch not installed')


# ===========================================================================
# 1. Config tests
# ===========================================================================


class TestDLinearModelDefaults:
    """MODEL_DEFAULTS['dlinear'] entries are applied via apply_model_defaults."""

    def test_dlinear_in_model_defaults(self):
        from liulian.config import MODEL_DEFAULTS

        assert 'dlinear' in MODEL_DEFAULTS
        defaults = MODEL_DEFAULTS['dlinear']
        assert defaults['split_mode'] == 'multi_channel'
        assert defaults['identifier_mode'] == 'none'
        assert defaults['batch_size'] == 32

    def test_apply_model_defaults_sets_dlinear_keys(self):
        from liulian.config import DEFAULT_CONFIG, apply_model_defaults

        import copy

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg['model'] = 'dlinear'

        apply_model_defaults(cfg)

        # These should be overridden by MODEL_DEFAULTS
        assert cfg['split_mode'] == 'multi_channel'
        assert cfg['identifier_mode'] == 'none'
        assert cfg['batch_size'] == 32
        assert cfg['dropout'] == 0.0

    def test_user_override_takes_precedence(self):
        """User CLI overrides should not be overwritten by MODEL_DEFAULTS."""
        from liulian.config import load_config

        # Simulate: user passes --model dlinear --split_mode per_entity via CLI
        cfg = load_config(
            cli_overrides={
                'model': 'dlinear',
                'split_mode': 'per_entity',
                'identifier_mode': 'embedding',
                'batch_size': 64,
            }
        )

        # CLI overrides should survive over MODEL_DEFAULTS
        assert cfg['split_mode'] == 'per_entity'
        assert cfg['identifier_mode'] == 'embedding'
        assert cfg['batch_size'] == 64

    def test_quick_test_no_forced_embedding(self):
        """QUICK_TEST_OVERRIDES should not force identifier_mode."""
        from liulian.config import QUICK_TEST_OVERRIDES

        assert 'identifier_mode' not in QUICK_TEST_OVERRIDES

    def test_load_config_applies_model_defaults(self):
        """load_config should auto-apply MODEL_DEFAULTS for the model."""
        from liulian.config import load_config

        cfg = load_config(cli_overrides={'model': 'dlinear'})
        # DLinear model defaults should be applied
        assert cfg['split_mode'] == 'multi_channel'
        assert cfg['identifier_mode'] == 'none'
        assert cfg['batch_size'] == 32

    def test_load_config_from_dlinear_yaml(self):
        """The dlinear_config.yaml loads correctly."""
        from liulian.config import load_config

        yaml_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'experiments',
            'swiss_river',
            'dlinear_config.yaml',
        )
        if not os.path.exists(yaml_path):
            pytest.skip('dlinear_config.yaml not found')

        cfg = load_config(yaml_path)
        assert cfg['model'] == 'dlinear'
        assert cfg['split_mode'] == 'multi_channel'
        # identifier_mode is set by the YAML (may be 'none', 'embedding',
        # 'embedding_idx', etc.); just verify the key exists.
        assert 'identifier_mode' in cfg


# ===========================================================================
# 2. Pipeline unit tests
# ===========================================================================


class TestBuildModelDLinear:
    """build_model correctly instantiates DLinear without special-casing."""

    def test_dynamic_import_dlinear(self):
        import torch

        from liulian.pipeline import build_model

        cfg = {
            'model': 'dlinear',
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 0,
            'individual': False,
            'moving_avg': 25,
            'task_name': 'long_term_forecast',
            'identifier_mode': 'none',
            'split_mode': 'multi_channel',
        }

        model = build_model(cfg)
        assert model is not None

        # Forward pass
        x = torch.randn(2, 96, 7)
        xm = torch.zeros(2, 96, 4)
        xd = torch.zeros(2, 24, 7)
        xmd = torch.zeros(2, 24, 4)
        model.eval()
        with torch.no_grad():
            out = model(x, xm, xd, xmd)
        assert out.shape == (2, 24, 7)
        assert torch.isfinite(out).all()

    def test_unknown_model_raises(self):
        from liulian.pipeline import build_model

        with pytest.raises(ValueError, match='Unknown model'):
            build_model({'model': 'nonexistent_xyzzy', 'identifier_mode': 'none'})


class TestBuildModelWithChannelWrapper:
    """build_model wraps with ChannelEntityWrapper in multi_channel + embedding mode."""

    def test_channel_wrapper_applied(self):
        import torch
        from liulian.models.torch.entity_mixin import ChannelEntityWrapper
        from liulian.pipeline import build_model

        cfg = {
            'model': 'dlinear',
            'enc_in': 10,
            'dec_in': 10,
            'c_out': 10,
            'seq_len': 32,
            'pred_len': 8,
            'label_len': 0,
            'individual': False,
            'moving_avg': 25,
            'task_name': 'long_term_forecast',
            'identifier_mode': 'embedding',
            'split_mode': 'multi_channel',
            'num_embeddings': 10,
            'embedding_size': 4,
        }

        model = build_model(cfg)
        assert isinstance(model, ChannelEntityWrapper)

        # Forward pass
        x = torch.randn(2, 32, 10)
        xm = torch.zeros(2, 32, 4)
        xd = torch.zeros(2, 8, 10)
        xmd = torch.zeros(2, 8, 4)
        model.eval()
        with torch.no_grad():
            out = model(x, xm, xd, xmd)
        assert out.shape == (2, 8, 10)
        assert torch.isfinite(out).all()

    def test_per_entity_mode_uses_regular_wrapper(self):
        from liulian.models.torch.entity_mixin import EntityWrapper
        from liulian.pipeline import build_model

        cfg = {
            'model': 'dlinear',
            'enc_in': 3,
            'dec_in': 3,
            'c_out': 3,
            'seq_len': 32,
            'pred_len': 8,
            'label_len': 0,
            'individual': False,
            'moving_avg': 25,
            'task_name': 'long_term_forecast',
            'identifier_mode': 'embedding',
            'split_mode': 'per_entity',
            'num_embeddings': 20,
            'embedding_size': 4,
        }

        model = build_model(cfg)
        assert isinstance(model, EntityWrapper)


class TestAutoEncIn:
    """auto_detect_enc_in returns feature dimension from dataset."""

    def test_auto_detect(self):
        from unittest.mock import MagicMock

        from liulian.pipeline import auto_detect_enc_in

        split = MagicMock()
        split.feat_dim = 32
        dataset = MagicMock()
        dataset.get_split.return_value = split

        result = auto_detect_enc_in(dataset)
        assert result == 32


class TestModelAwareSearchSpace:
    """Default HPO search space adapts to model type."""

    def test_dlinear_search_space(self):
        """Swiss DLinear search space: learning_rate + moving_avg, with
        batch_size FIXED at 32 (user decision 2026-06-14, not tuned) and no
        LSTM/PatchTST-specific params."""
        from liulian.pipeline import build_optimizer

        cfg = {
            'model': 'dlinear',
            'hpo': True,
            'train_epochs': 5,
            'hpo_num_samples': 2,
            'hpo_grace_period': 1,
            'hpo_reduction_factor': 2,
            'hpo_resources_cpu': 1,
            'hpo_resources_gpu': 0,
            'hpo_local_mode': True,
            'hpo_save_checkpoints': False,
            'hpo_trim_checkpoints': False,
            'hpo_keep_best_n': 1,
            'hpo_trim_best_n': False,
            'hpo_trim_keep_best': False,
            'hpo_trim_keep_last': False,
            'hpo_resume': False,
            'hpo_scheduler': 'asha',
            'data': 'swiss-river-1990',
            'task': 'forecast',
            'split_mode': 'multi_channel',
            'identifier_mode': 'none',
        }

        optimizer = build_optimizer(cfg)
        assert optimizer is not None

        space = cfg['search_space']
        assert 'learning_rate' in space
        assert 'moving_avg' in space
        # batch_size FIXED at 32 for swiss (dlinear_swiss space) -> not tuned
        assert 'batch_size' not in space
        # Should NOT have LSTM/PatchTST-specific params
        assert 'embedding_size' not in space
        assert 'd_model' not in space

    def test_lstm_search_space_has_embedding(self):
        """LSTM with embedding mode should include embedding_size."""
        from liulian.pipeline import build_optimizer

        cfg = {
            'model': 'lstm',
            'hpo': True,
            'train_epochs': 5,
            'hpo_num_samples': 2,
            'hpo_grace_period': 1,
            'hpo_reduction_factor': 2,
            'hpo_resources_cpu': 1,
            'hpo_resources_gpu': 0,
            'hpo_local_mode': True,
            'hpo_save_checkpoints': False,
            'hpo_trim_checkpoints': False,
            'hpo_keep_best_n': 1,
            'hpo_trim_best_n': False,
            'hpo_trim_keep_best': False,
            'hpo_trim_keep_last': False,
            'hpo_resume': False,
            'hpo_scheduler': 'asha',
            'data': 'swiss-river-1990',
            'task': 'forecast',
            'split_mode': 'per_entity',
            'identifier_mode': 'embedding',
        }

        optimizer = build_optimizer(cfg)
        assert optimizer is not None

        space = cfg['search_space']
        assert 'learning_rate' in space
        assert 'd_model' in space
        assert 'e_layers' in space
        assert 'embedding_size' in space


# ===========================================================================
# 3. ChannelEntityWrapper unit tests
# ===========================================================================


class TestChannelEntityWrapper:
    """Unit tests for ChannelEntityWrapper (multi_channel mode entity embedding)."""

    def test_forward_shape(self):
        import torch
        from liulian.models.torch.entity_mixin import ChannelEntityWrapper

        class _Identity(torch.nn.Module):
            def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
                return x_enc

        wrapper = ChannelEntityWrapper(
            inner_model=_Identity(),
            num_stations=5,
            embedding_size=4,
        )

        x = torch.randn(2, 10, 5)
        out = wrapper(x)
        assert out.shape == (2, 10, 5)
        assert torch.isfinite(out).all()

    def test_forward_with_decoder(self):
        """x_dec should also be augmented."""
        import torch
        from liulian.models.torch.entity_mixin import ChannelEntityWrapper

        class _EncDec(torch.nn.Module):
            def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
                if x_dec is not None:
                    return x_dec[:, :, : x_enc.size(2)]
                return x_enc

        wrapper = ChannelEntityWrapper(
            inner_model=_EncDec(),
            num_stations=7,
            embedding_size=3,
        )

        x_enc = torch.randn(2, 32, 7)
        x_dec = torch.randn(2, 16, 7)
        out = wrapper(x_enc, x_dec=x_dec)
        assert out.shape == (2, 16, 7)
        assert torch.isfinite(out).all()

    def test_embedding_is_learnable(self):
        """Gradients should flow through station_embedding."""
        import torch
        from liulian.models.torch.entity_mixin import ChannelEntityWrapper

        class _Proj(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
                return self.fc(x_enc)

        wrapper = ChannelEntityWrapper(
            inner_model=_Proj(),
            num_stations=5,
            embedding_size=4,
        )

        x = torch.randn(2, 10, 5)
        out = wrapper(x)
        loss = out.sum()
        loss.backward()

        # station_embedding should have gradients
        assert wrapper.station_embedding.weight.grad is not None
        assert wrapper.station_embedding.weight.grad.abs().sum() > 0

    def test_augment_modifies_values(self):
        """Output should differ from identity (embedding has effect)."""
        import torch
        from liulian.models.torch.entity_mixin import ChannelEntityWrapper

        class _Identity(torch.nn.Module):
            def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
                return x_enc

        wrapper = ChannelEntityWrapper(
            inner_model=_Identity(),
            num_stations=3,
            embedding_size=8,
        )

        x = torch.randn(2, 10, 3)
        wrapper.eval()
        with torch.no_grad():
            out = wrapper(x)

        # The embedding + projection should modify the values
        assert not torch.allclose(out, x, atol=1e-5)


# ===========================================================================
# 4. Target names derivation
# ===========================================================================


DATASET_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'swiss_river')


class TestTargetNames:
    """Verify target_names in dataset.info() for different split modes."""

    @pytest.fixture(autouse=True)
    def _check_data(self):
        if not os.path.isdir(DATASET_ROOT):
            pytest.skip('Swiss River dataset not available')

    def test_multi_channel_mode_target_names(self):
        from liulian.data.swiss_river import SwissRiverDataset

        ds = SwissRiverDataset(
            data_name='swiss-river-1990',
            split_mode='multi_channel',
            seq_len=10,
            pred_len=3,
        )
        info = ds.info()
        target_names = info.get('target_names')
        assert target_names is not None
        assert len(target_names) > 1
        assert all('_wt' in name for name in target_names)

    def test_per_entity_mode_target_names(self):
        from liulian.data.swiss_river import SwissRiverDataset

        ds = SwissRiverDataset(
            data_name='swiss-river-1990',
            split_mode='per_entity',
            seq_len=10,
            pred_len=3,
        )
        info = ds.info()
        target_names = info.get('target_names')
        assert target_names is not None
        assert target_names == ['water_temperature']
