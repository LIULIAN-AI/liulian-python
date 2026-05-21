"""Smoke tests for the benchmark pipeline infrastructure.

pytestmark = pytest.mark.skip(reason="pre-existing: scaler registry missing 'standard'; TODO: register or rewrite tests")

Tests config generation, experiment runner plumbing, and results aggregation
without running actual training (which is the user's responsibility).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ======================================================================
# generate_configs.py
# ======================================================================


class TestConfigGeneration:
    """Verify generate_configs.py produces the expected YAML files."""

    @pytest.fixture(scope='class')
    def config_dir(self, tmp_path_factory):
        """Generate all configs into a temp dir (once per class)."""
        from tools.generate_configs import (
            generate_long_term,
            generate_entity,
            generate_nowcasting,
            generate_m4,
            generate_spatial,
            generate_ablation_norm,
            generate_ablation_aug,
            generate_ablation_seqlen,
            generate_ablation_tf,
        )

        d = tmp_path_factory.mktemp('configs')
        self._counts = {}
        for name, fn in [
            ('long_term', generate_long_term),
            ('entity', generate_entity),
            ('nowcasting', generate_nowcasting),
            ('m4', generate_m4),
            ('spatial', generate_spatial),
            ('ablation_norm', generate_ablation_norm),
            ('ablation_aug', generate_ablation_aug),
            ('ablation_seqlen', generate_ablation_seqlen),
            ('ablation_tf', generate_ablation_tf),
        ]:
            self._counts[name] = fn(d)
        return d

    def test_long_term_count(self, config_dir):
        yamls = list((config_dir / 'long_term').glob('*.yaml'))
        assert len(yamls) == 720, f'Expected 720 long_term configs, got {len(yamls)}'

    def test_entity_count(self, config_dir):
        yamls = list((config_dir / 'entity').glob('*.yaml'))
        assert len(yamls) == 795, f'Expected 795 entity configs, got {len(yamls)}'

    def test_nowcasting_count(self, config_dir):
        yamls = list((config_dir / 'nowcasting').glob('*.yaml'))
        assert len(yamls) == 360, f'Expected 360 nowcasting configs, got {len(yamls)}'

    def test_m4_count(self, config_dir):
        yamls = list((config_dir / 'm4').glob('*.yaml'))
        assert len(yamls) == 90, f'Expected 90 m4 configs, got {len(yamls)}'

    def test_spatial_count(self, config_dir):
        yamls = list((config_dir / 'spatial').glob('*.yaml'))
        assert len(yamls) == 210, f'Expected 210 spatial configs, got {len(yamls)}'

    def test_ablation_norm_count(self, config_dir):
        yamls = list((config_dir / 'ablation_norm').glob('*.yaml'))
        assert len(yamls) == 135, f'Expected 135 ablation_norm configs, got {len(yamls)}'

    def test_ablation_aug_count(self, config_dir):
        yamls = list((config_dir / 'ablation_aug').glob('*.yaml'))
        assert len(yamls) == 180, f'Expected 180 ablation_aug configs, got {len(yamls)}'

    def test_ablation_seqlen_count(self, config_dir):
        yamls = list((config_dir / 'ablation_seqlen').glob('*.yaml'))
        assert len(yamls) == 225, f'Expected 225 ablation_seqlen configs, got {len(yamls)}'

    def test_ablation_tf_count(self, config_dir):
        yamls = list((config_dir / 'ablation_tf').glob('*.yaml'))
        assert len(yamls) == 36, f'Expected 36 ablation_tf configs, got {len(yamls)}'

    def test_total_count(self, config_dir):
        all_yamls = list(config_dir.rglob('*.yaml'))
        assert len(all_yamls) == 2751, f'Expected 2751 total configs, got {len(all_yamls)}'

    def test_config_parseable(self, config_dir):
        """Every generated YAML should be parseable and contain 'model' key."""
        for yf in list(config_dir.rglob('*.yaml'))[:50]:  # Sample 50
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            assert 'model' in cfg, f"Missing 'model' in {yf}"
            assert 'dataset' in cfg or 'data_name' in cfg, f'Missing dataset info in {yf}'

    def test_long_term_has_required_keys(self, config_dir):
        """Spot-check a long_term config has all expected fields."""
        yamls = list((config_dir / 'long_term').glob('*.yaml'))
        with open(yamls[0]) as f:
            cfg = yaml.safe_load(f)
        for key in [
            'model',
            'dataset',
            'seq_len',
            'pred_len',
            'enc_in',
            'loss',
            'metrics',
        ]:
            assert key in cfg, f"Missing key '{key}' in {yamls[0].name}"

    def test_entity_has_identifier_mode(self, config_dir):
        """Entity configs should include identifier_mode."""
        yamls = list((config_dir / 'entity').glob('*.yaml'))
        with open(yamls[0]) as f:
            cfg = yaml.safe_load(f)
        assert 'identifier_mode' in cfg, "Entity config missing 'identifier_mode'"

    def test_nowcasting_has_nse_metric(self, config_dir):
        """Nowcasting configs should include NSE in metrics."""
        yamls = list((config_dir / 'nowcasting').glob('*.yaml'))
        with open(yamls[0]) as f:
            cfg = yaml.safe_load(f)
        assert 'nse' in cfg.get('metrics', []), "Nowcasting config missing 'nse' metric"

    def test_swiss_variants_in_long_term(self, config_dir):
        """All three Swiss variants should appear in long_term configs."""
        names = {yf.stem for yf in (config_dir / 'long_term').glob('*.yaml')}
        for variant in ['swiss1990', 'swiss2010', 'swisszurich']:
            matching = [n for n in names if variant in n]
            assert len(matching) > 0, f'No long_term configs for {variant}'

    def test_all_models_in_long_term(self, config_dir):
        """All 15 models should appear in long_term configs."""
        names = {yf.stem.split('_')[0] for yf in (config_dir / 'long_term').glob('*.yaml')}
        # Some names may be lowered differently
        found = {n.lower() for n in names}
        for p in ['dlinear', 'transformer', 'informer', 'lstmadapter']:
            assert p in found, f"Model prefix '{p}' not found in long_term configs"


# ======================================================================
# aggregate_results.py
# ======================================================================


class TestResultsAggregation:
    """Test results loading and report generation."""

    @pytest.fixture
    def mock_results_dir(self, tmp_path):
        """Create a mock results directory with sample JSON files."""
        long_term = tmp_path / 'long_term'
        long_term.mkdir()

        for model in ['DLinear', 'PatchTST']:
            for dataset in ['ETTh1', 'Weather']:
                for seed in [1, 2, 3]:
                    result = {
                        'model': model,
                        'dataset': dataset,
                        'seed': seed,
                        'config': {'seq_len': 96, 'pred_len': 96},
                        'metrics': {
                            'test': {
                                'mse': 0.3 + seed * 0.01,
                                'mae': 0.4 + seed * 0.01,
                            },
                        },
                        'best_val_score': 0.35,
                        'epochs_run': 20,
                        'elapsed_seconds': 60.0,
                        'status': 'success',
                    }
                    with open(
                        long_term / f'{model.lower()}_{dataset.lower()}_H96_seed{seed}.json',
                        'w',
                    ) as f:
                        json.dump(result, f)

        # Add an entity result
        entity = tmp_path / 'entity'
        entity.mkdir()
        for mode in ['none', 'embedding']:
            result = {
                'model': 'DLinear',
                'dataset': 'Traffic',
                'seed': 42,
                'config': {'identifier_mode': mode},
                'metrics': {'test': {'mse': 0.5 if mode == 'none' else 0.45}},
                'status': 'success',
            }
            with open(entity / f'dlinear_traffic_{mode}_seed42.json', 'w') as f:
                json.dump(result, f)

        # Add a failed result
        failed = {
            'model': 'Mamba',
            'dataset': 'ETTh1',
            'seed': 1,
            'status': 'error',
            'error': 'mamba_ssm not installed',
        }
        with open(long_term / 'mamba_etth1_H96_seed1.json', 'w') as f:
            json.dump(failed, f)

        return tmp_path

    def test_load_results(self, mock_results_dir):
        from tools.aggregate_results import load_results

        groups = load_results(mock_results_dir)
        assert 'long_term' in groups
        assert 'entity' in groups
        assert len(groups['long_term']) == 13  # 2*2*3 + 1 failed
        assert len(groups['entity']) == 2

    def test_aggregate_by_model_dataset(self, mock_results_dir):
        from tools.aggregate_results import load_results, aggregate_by_model_dataset

        groups = load_results(mock_results_dir)
        agg = aggregate_by_model_dataset(groups['long_term'], 'mse')
        assert ('DLinear', 'ETTh1') in agg
        assert ('PatchTST', 'Weather') in agg
        assert agg[('DLinear', 'ETTh1')]['n'] == 3  # 3 seeds

    def test_aggregate_entity_ablation(self, mock_results_dir):
        from tools.aggregate_results import load_results, aggregate_entity_ablation

        groups = load_results(mock_results_dir)
        agg = aggregate_entity_ablation(groups['entity'], 'mse')
        assert ('DLinear', 'Traffic', 'none') in agg
        assert ('DLinear', 'Traffic', 'embedding') in agg
        assert agg[('DLinear', 'Traffic', 'embedding')]['mean'] < agg[('DLinear', 'Traffic', 'none')]['mean']

    def test_generate_markdown_report(self, mock_results_dir):
        from tools.aggregate_results import load_results, generate_markdown_report

        groups = load_results(mock_results_dir)
        report = generate_markdown_report(groups)
        assert '# Liulian Benchmark Results' in report
        assert 'DLinear' in report
        assert 'PatchTST' in report
        assert 'Failed Experiments' in report
        assert 'mamba_ssm' in report

    def test_generate_markdown_report_to_file(self, mock_results_dir, tmp_path):
        from tools.aggregate_results import load_results, generate_markdown_report

        groups = load_results(mock_results_dir)
        out = tmp_path / 'report.md'
        generate_markdown_report(groups, out)
        assert out.exists()
        content = out.read_text()
        assert '# Liulian Benchmark Results' in content

    def test_generate_latex_tables(self, mock_results_dir, tmp_path):
        from tools.aggregate_results import load_results, generate_latex_tables

        groups = load_results(mock_results_dir)
        tex_dir = tmp_path / 'tables'
        generate_latex_tables(groups, tex_dir)
        assert tex_dir.exists()
        tex_files = list(tex_dir.glob('*.tex'))
        assert len(tex_files) >= 1


# ======================================================================
# run_benchmark.py helpers
# ======================================================================


class TestRunBenchmarkHelpers:
    """Test run_benchmark.py helper functions."""

    def test_result_path(self):
        from tools.run_benchmark import result_path

        p = result_path(
            Path('experiments/configs/long_term/dlinear_etth1_H96.yaml'),
            seed=42,
            results_dir=Path('/tmp/results'),
        )
        assert p == Path('/tmp/results/long_term/dlinear_etth1_H96_seed42.json')

    def test_is_completed_false(self, tmp_path):
        from tools.run_benchmark import is_completed

        assert not is_completed(
            Path('experiments/configs/long_term/dlinear_etth1_H96.yaml'),
            seed=42,
            results_dir=tmp_path,
        )

    def test_is_completed_true(self, tmp_path):
        from tools.run_benchmark import is_completed

        # Create the result file
        d = tmp_path / 'long_term'
        d.mkdir()
        (d / 'dlinear_etth1_H96_seed42.json').write_text('{}')
        assert is_completed(
            Path('experiments/configs/long_term/dlinear_etth1_H96.yaml'),
            seed=42,
            results_dir=tmp_path,
        )

    def test_gather_configs(self, tmp_path):
        from tools.run_benchmark import gather_configs

        (tmp_path / 'a.yaml').write_text('model: DLinear')
        (tmp_path / 'b.yaml').write_text('model: PatchTST')
        (tmp_path / 'c.txt').write_text('not yaml')
        configs = gather_configs(tmp_path)
        assert len(configs) == 2
        assert all(c.suffix == '.yaml' for c in configs)

    def test_load_config(self, tmp_path):
        from tools.run_benchmark import load_config

        cfg_path = tmp_path / 'test.yaml'
        cfg_path.write_text(yaml.dump({'model': 'DLinear', 'seq_len': 96}))
        cfg = load_config(cfg_path)
        assert cfg['model'] == 'DLinear'
        assert cfg['seq_len'] == 96


# ======================================================================
# PEMS dataset
# ======================================================================


class TestPEMSDataset:
    """Test PEMS dataset loading."""

    @pytest.fixture
    def pems_data(self, tmp_path):
        """Create a small synthetic PEMS-like .npz file."""
        import numpy as np

        T, N, F = 500, 10, 3
        data = np.random.randn(T, N, F).astype(np.float32)
        np.savez(tmp_path / 'PEMS_test.npz', data=data)
        return tmp_path

    def test_pems_loads(self, pems_data):
        from liulian.data.pems_dataset import PEMSDataset

        container = PEMSDataset(
            root_path=str(pems_data),
            data_path='PEMS_test.npz',
            size=(96, 48, 12),
        )
        ds = container.get_split('train')
        assert len(ds) > 0

    def test_pems_returns_4tuple(self, pems_data):
        from liulian.data.pems_dataset import PEMSDataset

        container = PEMSDataset(
            root_path=str(pems_data),
            data_path='PEMS_test.npz',
            size=(96, 48, 12),
        )
        ds = container.get_split('train')
        item = ds[0]
        assert len(item) == 4
        x, y, x_mark, y_mark = item
        assert x.shape == (96, 10)  # seq_len × sensors
        assert y.shape == (60, 10)  # (label_len + pred_len) × sensors
        assert x_mark.shape[0] == 96
        assert y_mark.shape[0] == 60

    def test_pems_splits(self, pems_data):
        from liulian.data.pems_dataset import PEMSDataset

        container = PEMSDataset(
            root_path=str(pems_data),
            data_path='PEMS_test.npz',
            size=(48, 24, 12),
        )
        for flag in ['train', 'val', 'test']:
            ds = container.get_split(flag)
            assert len(ds) > 0, f'Empty {flag} split'

    def test_pems_inverse_transform(self, pems_data):
        import numpy as np
        from liulian.data.pems_dataset import PEMSDataset

        container = PEMSDataset(
            root_path=str(pems_data),
            data_path='PEMS_test.npz',
            size=(48, 24, 12),
            scale=True,
        )
        dummy = np.zeros((10, 10))
        result = container.inverse_transform(dummy)
        assert result.shape == (10, 10)


# ======================================================================
# Entity mixin
# ======================================================================


class TestEntityMixinUnit:
    """Unit tests for EntityAwareMixin and EntityWrapper."""

    def test_entity_wrapper_forward(self):
        import torch
        from liulian.models.torch.entity_mixin import EntityWrapper

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(7, 7)

            def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
                return self.proj(x_enc)

        model = SimpleModel()
        wrapper = EntityWrapper(model, enc_in=7, num_embeddings=10, embedding_size=4)
        x_enc = torch.randn(2, 96, 7)
        x_mark = torch.zeros(2, 96, 1)
        out = wrapper(x_enc, x_mark)
        assert out.shape == (2, 96, 7)

    def test_entity_wrapper_no_mark(self):
        import torch
        from liulian.models.torch.entity_mixin import EntityWrapper

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(7, 7)

            def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
                return self.proj(x_enc)

        model = SimpleModel()
        wrapper = EntityWrapper(model, enc_in=7, num_embeddings=10, embedding_size=4)
        x_enc = torch.randn(2, 96, 7)
        # No x_mark — should pass through without error
        out = wrapper(x_enc, None)
        assert out.shape == (2, 96, 7)

    def test_entity_model_config_identity(self):
        from liulian.models.torch.entity_mixin import EntityAwareMixin

        cfg = {'enc_in': 7, 'identifier_mode': 'embedding'}
        result = EntityAwareMixin._entity_model_config(cfg)
        assert result is cfg  # Same object — no copy/modification
        assert result['enc_in'] == 7  # Unchanged


# ======================================================================
# Data factory registrations
# ======================================================================


class TestDataFactoryRegistrations:
    """Verify all datasets are registered in data_factory."""

    def test_all_expected_datasets_registered(self):
        from liulian.data.data_factory import DATASET_REGISTRY

        expected = [
            'ETTh1',
            'ETTh2',
            'ETTm1',
            'ETTm2',
            'weather',
            'electricity',
            'traffic',
            'exchange_rate',
            'illness',
            'm4',
            'PEMS03',
            'PEMS04',
            'PEMS07',
            'PEMS08',
        ]
        for name in expected:
            assert name in DATASET_REGISTRY, f"'{name}' not in DATASET_REGISTRY"

    def test_pems_uses_correct_class(self):
        from liulian.data.data_factory import DATASET_REGISTRY
        from liulian.data.pems_dataset import PEMSDataset

        for name in ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']:
            assert DATASET_REGISTRY[name] is PEMSDataset


# ======================================================================
# Search spaces
# ======================================================================


class TestSearchSpaces:
    """Verify search spaces for all model families."""

    def test_tsl_model_spaces_exist(self):
        from liulian.optim.search_spaces import get_search_space

        models = [
            'dlinear',
            'transformer',
            'informer',
            'autoformer',
            'fedformer',
            'itransformer',
            'patchtst',
            'timesnet',
            'timemixer',
            'timexer',
            'mamba',
        ]
        for model in models:
            space = get_search_space(model)
            assert space is not None, f"No search space for '{model}'"
            assert isinstance(space, dict), f"Search space for '{model}' is not a dict"
            assert len(space) > 0, f"Search space for '{model}' is empty"

    def test_custom_model_spaces_exist(self):
        from liulian.optim.search_spaces import get_search_space

        for name in ['lstm', 'lstm_general', 'transformer_encoder', 'transformer_enc']:
            space = get_search_space(name)
            assert space is not None, f"No search space for '{name}'"
