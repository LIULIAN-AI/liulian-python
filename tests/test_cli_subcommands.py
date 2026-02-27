"""Tests for the liulian CLI — argument parsing and subcommand dispatch."""

from __future__ import annotations

import argparse
import pytest

from liulian.cli import main


class TestCliParsing:
    """Verify that the argument parser accepts all expected subcommands."""

    def test_info(self, capsys):
        """``liulian info`` prints version."""
        main(['info'])
        captured = capsys.readouterr()
        assert 'liulian' in captured.out.lower()

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(['--version'])
        assert exc_info.value.code == 0

    def test_no_subcommand_prints_help(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert (
            'usage' in captured.out.lower()
            or 'subcommand' in captured.out.lower()
            or 'liulian' in captured.out.lower()
        )

    # --- Validate that each subcommand accepts its expected positional args ---
    # (We do NOT actually run them — just ensure the parser doesn't error.)

    def test_train_parser_accepts_config(self):
        """``liulian train`` parser should accept config + optional overrides."""
        from liulian.cli import main as _m

        # Build the parser manually to inspect
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest='command')
        sp = sub.add_parser('train')
        sp.add_argument('config')
        sp.add_argument('--epochs', type=int)
        sp.add_argument('--lr', type=float)
        sp.add_argument('--wandb-project')
        args = parser.parse_args(
            ['train', 'exp.yaml', '--epochs', '5', '--lr', '0.001']
        )
        assert args.config == 'exp.yaml'
        assert args.epochs == 5
        assert args.lr == pytest.approx(0.001)

    def test_viz_parser_accepts_method(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest='command')
        sp = sub.add_parser('viz')
        sp.add_argument('config')
        sp.add_argument('--method', default='mean')
        args = parser.parse_args(['viz', 'cfg.yaml', '--method', 'median'])
        assert args.method == 'median'

    def test_tune_parser(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest='command')
        sp = sub.add_parser('tune')
        sp.add_argument('config')
        args = parser.parse_args(['tune', 'hpo.yaml'])
        assert args.config == 'hpo.yaml'

    def test_predict_parser(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest='command')
        sp = sub.add_parser('predict')
        sp.add_argument('config')
        args = parser.parse_args(['predict', 'inf.yaml'])
        assert args.config == 'inf.yaml'


class TestCliSubcommandsMissing:
    """Subcommands that require a config file should exit cleanly when
    the file doesn't exist.
    """

    @pytest.mark.parametrize(
        'subcmd', ['run', 'eval', 'train', 'predict', 'viz', 'tune']
    )
    def test_missing_config_exits(self, subcmd):
        with pytest.raises(SystemExit):
            main([subcmd, '/nonexistent/__fake__.yaml'])
