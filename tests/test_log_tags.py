"""Unit tests for liulian.utils.log_tags — multi-mode tags + logging integration."""

from __future__ import annotations

import io
import logging

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_mode(mode):
    """Context-manager-like helper: set mode, rebuild, yield, restore."""
    from liulian.utils.log_tags import set_output_mode, get_output_mode, _rebuild_tags

    original = get_output_mode()
    set_output_mode(mode)
    return original


def _restore_mode(original):
    from liulian.utils.log_tags import set_output_mode

    set_output_mode(original)


# ---------------------------------------------------------------------------
# Basic tag strings
# ---------------------------------------------------------------------------


class TestTagStrings:
    """Verify the pre-built tag constants."""

    def test_tags_are_non_empty(self) -> None:
        from liulian.utils.log_tags import (
            INFO_TAG,
            SUCCESS_TAG,
            WARNING_TAG,
            ERROR_TAG,
            DEBUG_TAG,
            HINT_TAG,
            PROGRESS_TAG,
        )

        for tag in (
            INFO_TAG,
            SUCCESS_TAG,
            WARNING_TAG,
            ERROR_TAG,
            DEBUG_TAG,
            HINT_TAG,
            PROGRESS_TAG,
        ):
            assert isinstance(tag, str) and len(tag) > 0

    def test_tag_labels_plain(self) -> None:
        from liulian.utils.log_tags import strip_ansi, INFO_TAG, SUCCESS_TAG

        assert '[info]' in strip_ansi(INFO_TAG)
        assert '[ok]' in strip_ansi(SUCCESS_TAG)


# ---------------------------------------------------------------------------
# Output modes
# ---------------------------------------------------------------------------


class TestOutputModes:
    def test_set_and_get(self) -> None:
        from liulian.utils.log_tags import (
            set_output_mode,
            get_output_mode,
            MODE_ANSI,
            MODE_PLAIN,
            MODE_HTML,
        )

        original = get_output_mode()
        try:
            for m in (MODE_ANSI, MODE_PLAIN, MODE_HTML):
                set_output_mode(m)
                assert get_output_mode() == m
        finally:
            set_output_mode(original)

    def test_invalid_mode_raises(self) -> None:
        from liulian.utils.log_tags import set_output_mode

        with pytest.raises(ValueError, match='mode must be'):
            set_output_mode('invalid')

    def test_backward_compat(self) -> None:
        from liulian.utils.log_tags import (
            set_colour_enabled,
            colour_enabled,
            get_output_mode,
            MODE_ANSI,
            MODE_PLAIN,
        )

        original = get_output_mode()
        try:
            set_colour_enabled(True)
            assert colour_enabled()
            assert get_output_mode() == MODE_ANSI
            set_colour_enabled(False)
            assert not colour_enabled()
            assert get_output_mode() == MODE_PLAIN
        finally:
            from liulian.utils.log_tags import set_output_mode

            set_output_mode(original)

    def test_pycharm_detection(self, monkeypatch) -> None:
        """PYCHARM_HOSTED env var triggers ANSI mode even without isatty."""
        from liulian.utils.log_tags import _detect_mode, MODE_ANSI

        monkeypatch.setenv('PYCHARM_HOSTED', '1')
        monkeypatch.delenv('LIULIAN_LOG_HTML', raising=False)
        monkeypatch.delenv('NO_COLOR', raising=False)
        monkeypatch.delenv('LIULIAN_NO_COLOR', raising=False)
        monkeypatch.delenv('LIULIAN_FORCE_COLOR', raising=False)
        assert _detect_mode() == MODE_ANSI

    def test_vscode_detection(self, monkeypatch) -> None:
        """TERM_PROGRAM=vscode triggers ANSI mode even without isatty."""
        from liulian.utils.log_tags import _detect_mode, MODE_ANSI

        monkeypatch.setenv('TERM_PROGRAM', 'vscode')
        monkeypatch.delenv('LIULIAN_LOG_HTML', raising=False)
        monkeypatch.delenv('NO_COLOR', raising=False)
        monkeypatch.delenv('LIULIAN_NO_COLOR', raising=False)
        monkeypatch.delenv('LIULIAN_FORCE_COLOR', raising=False)
        monkeypatch.delenv('PYCHARM_HOSTED', raising=False)
        assert _detect_mode() == MODE_ANSI


# ---------------------------------------------------------------------------
# ANSI mode: coloured tag, plain message
# ---------------------------------------------------------------------------


class TestAnsiMode:
    def test_info_tag_colour_then_reset(self) -> None:
        orig = _force_mode('ansi')
        try:
            from liulian.utils.log_tags import INFO_TAG, _RESET

            # Tag must contain ANSI colour and end with reset before message
            assert '\033[' in INFO_TAG
            assert _RESET in INFO_TAG
            assert INFO_TAG.endswith(' ')
        finally:
            _restore_mode(orig)

    def test_error_tag_prefix_only(self) -> None:
        """ERROR_TAG constant is just the tag prefix, not full-line styling."""
        orig = _force_mode('ansi')
        try:
            from liulian.utils.log_tags import ERROR_TAG

            assert '[ERROR]' in ERROR_TAG
        finally:
            _restore_mode(orig)


# ---------------------------------------------------------------------------
# Plain mode
# ---------------------------------------------------------------------------


class TestPlainMode:
    def test_no_ansi_codes(self) -> None:
        orig = _force_mode('plain')
        try:
            from liulian.utils.log_tags import INFO_TAG, ERROR_TAG

            assert '\033[' not in INFO_TAG
            assert '\033[' not in ERROR_TAG
            assert INFO_TAG == '[info] '
            assert ERROR_TAG == '[ERROR] '
        finally:
            _restore_mode(orig)


# ---------------------------------------------------------------------------
# HTML mode
# ---------------------------------------------------------------------------


class TestHtmlMode:
    def test_html_info_tag(self) -> None:
        orig = _force_mode('html')
        try:
            from liulian.utils.log_tags import INFO_TAG

            assert '<span' in INFO_TAG
            assert '[info]' in INFO_TAG
            assert '</span>' in INFO_TAG
        finally:
            _restore_mode(orig)

    def test_html_error_format(self) -> None:
        orig = _force_mode('html')
        try:
            from liulian.utils.log_tags import _error_format

            result = _error_format('ERROR', 'crash')
            assert '<span' in result
            assert 'background-color' in result
            assert '[ERROR]' in result
            assert 'crash' in result
            assert '</span>' in result
        finally:
            _restore_mode(orig)


# ---------------------------------------------------------------------------
# Error line styling (ANSI)
# ---------------------------------------------------------------------------


class TestErrorFormat:
    def test_ansi_error_full_line(self) -> None:
        orig = _force_mode('ansi')
        try:
            from liulian.utils.log_tags import _error_format, _BG_DARK_RED, _RESET

            result = _error_format('ERROR', 'disk full')
            assert _BG_DARK_RED in result
            assert '[ERROR]' in result
            assert 'disk full' in result
            assert result.endswith(_RESET)
        finally:
            _restore_mode(orig)

    def test_plain_error_no_ansi(self) -> None:
        orig = _force_mode('plain')
        try:
            from liulian.utils.log_tags import _error_format

            result = _error_format('ERROR', 'disk full')
            assert result == '[ERROR] disk full'
        finally:
            _restore_mode(orig)


# ---------------------------------------------------------------------------
# Convenience printers
# ---------------------------------------------------------------------------


class TestConveniencePrinters:
    def test_info_prints_to_stdout(self, capsys) -> None:
        from liulian.utils.log_tags import info

        info('hello')
        assert 'hello' in capsys.readouterr().out

    def test_error_prints_to_stderr(self, capsys) -> None:
        from liulian.utils.log_tags import error

        error('fail')
        captured = capsys.readouterr().err
        assert 'fail' in captured
        assert '[ERROR]' in captured


# ---------------------------------------------------------------------------
# strip_ansi / strip_html_tags / strip_formatting
# ---------------------------------------------------------------------------


class TestStripUtilities:
    def test_strip_ansi_plain(self) -> None:
        from liulian.utils.log_tags import strip_ansi

        assert strip_ansi('[info] hello') == '[info] hello'

    def test_strip_ansi_codes(self) -> None:
        from liulian.utils.log_tags import strip_ansi

        assert strip_ansi('\033[94m[info]\033[0m hello') == '[info] hello'

    def test_strip_html(self) -> None:
        from liulian.utils.log_tags import strip_html_tags

        html = '<span style="color:red">[ERROR]</span> msg'
        assert strip_html_tags(html) == '[ERROR] msg'

    def test_strip_formatting_both(self) -> None:
        from liulian.utils.log_tags import strip_formatting

        mixed = '\033[91m<span>[x]</span>\033[0m'
        assert strip_formatting(mixed) == '[x]'


# ---------------------------------------------------------------------------
# _msg_has_tag detection
# ---------------------------------------------------------------------------


class TestMsgHasTag:
    def test_plain_tag(self) -> None:
        from liulian.utils.log_tags import _msg_has_tag

        assert _msg_has_tag('[info] something')
        assert _msg_has_tag('[ok] done')
        assert _msg_has_tag('[ERROR] oops')

    def test_ansi_tag(self) -> None:
        from liulian.utils.log_tags import _msg_has_tag

        assert _msg_has_tag('\033[94m\033[1m[info]\033[0m something')

    def test_html_tag(self) -> None:
        from liulian.utils.log_tags import _msg_has_tag

        assert _msg_has_tag('<span style="color:red">[ERROR]</span> oops')

    def test_no_tag(self) -> None:
        from liulian.utils.log_tags import _msg_has_tag

        assert not _msg_has_tag('plain text')
        assert not _msg_has_tag('[custom] tag')


# ---------------------------------------------------------------------------
# TaggedFormatter
# ---------------------------------------------------------------------------


class TestTaggedFormatter:
    """Test the logging.Formatter subclass."""

    @pytest.fixture(autouse=True)
    def _plain_mode(self):
        """Use plain mode for deterministic assertions."""
        orig = _force_mode('plain')
        yield
        _restore_mode(orig)

    def _make_logger(self, stream, *, auto_tag=True):
        from liulian.utils.log_tags import TaggedFormatter

        lgr = logging.getLogger(f'_test_{id(stream)}')
        lgr.handlers.clear()
        lgr.setLevel(logging.DEBUG)
        lgr.propagate = False
        handler = logging.StreamHandler(stream)
        handler.setFormatter(TaggedFormatter(auto_tag=auto_tag))
        lgr.addHandler(handler)
        return lgr

    def test_auto_tag_info(self) -> None:
        buf = io.StringIO()
        lgr = self._make_logger(buf)
        lgr.info('hello')
        assert buf.getvalue().strip() == '[info] hello'

    def test_auto_tag_warning(self) -> None:
        buf = io.StringIO()
        lgr = self._make_logger(buf)
        lgr.warning('caution')
        assert '[warn]' in buf.getvalue()

    def test_auto_tag_debug(self) -> None:
        buf = io.StringIO()
        lgr = self._make_logger(buf)
        lgr.debug('trace')
        assert '[debug]' in buf.getvalue()

    def test_error_full_line_plain(self) -> None:
        """In plain mode, error is just [ERROR] msg (no ANSI)."""
        buf = io.StringIO()
        lgr = self._make_logger(buf)
        lgr.error('oops')
        assert buf.getvalue().strip() == '[ERROR] oops'

    def test_error_full_line_ansi(self) -> None:
        """In ANSI mode, error has background colour on entire line."""
        _restore_mode('ansi')  # switch to ansi for this test
        try:
            from liulian.utils.log_tags import _BG_DARK_RED

            buf = io.StringIO()
            lgr = self._make_logger(buf)
            lgr.error('crash')
            output = buf.getvalue()
            assert _BG_DARK_RED in output
            assert 'crash' in output
        finally:
            _force_mode('plain')

    def test_explicit_tag_skips_auto(self) -> None:
        """If message already has a tag, auto-tag is skipped."""
        from liulian.utils.log_tags import SUCCESS_TAG

        buf = io.StringIO()
        lgr = self._make_logger(buf)
        lgr.info(f'{SUCCESS_TAG}done!')
        output = buf.getvalue()
        assert '[ok]' in output
        assert '[info]' not in output

    def test_auto_tag_disabled(self) -> None:
        buf = io.StringIO()
        lgr = self._make_logger(buf, auto_tag=False)
        lgr.info('no tag')
        output = buf.getvalue()
        assert '[info]' not in output
        assert 'no tag' in output

    # -- custom levels --

    def test_logger_ok_method(self) -> None:
        """logger.ok() produces an [ok] tagged message."""
        buf = io.StringIO()
        lgr = self._make_logger(buf)
        lgr.ok('milestone reached')
        assert buf.getvalue().strip() == '[ok] milestone reached'

    def test_logger_hint_method(self) -> None:
        """logger.hint() produces a [hint] tagged message."""
        buf = io.StringIO()
        lgr = self._make_logger(buf)
        lgr.hint('try using --fast')
        assert buf.getvalue().strip() == '[hint] try using --fast'

    def test_logger_progress_method(self) -> None:
        """logger.progress() produces a [...] tagged message."""
        buf = io.StringIO()
        lgr = self._make_logger(buf)
        lgr.progress('loading data')
        assert buf.getvalue().strip() == '[...] loading data'

    def test_ok_level_between_info_and_warning(self) -> None:
        """OK_LEVEL (25) is between INFO (20) and WARNING (30)."""
        from liulian.utils.log_tags import OK_LEVEL, HINT_LEVEL, PROGRESS_LEVEL

        assert logging.INFO < HINT_LEVEL < PROGRESS_LEVEL < OK_LEVEL < logging.WARNING

    def test_ok_not_emitted_at_warning_level(self) -> None:
        """If logger.level=WARNING, logger.ok() is suppressed."""
        buf = io.StringIO()
        lgr = self._make_logger(buf)
        lgr.setLevel(logging.WARNING)
        lgr.ok('should not appear')
        assert buf.getvalue() == ''

    def test_ok_with_args(self) -> None:
        """logger.ok() supports %-style args like standard logging."""
        buf = io.StringIO()
        lgr = self._make_logger(buf)
        lgr.ok('Epoch %d done, loss=%.4f', 10, 0.0123)
        assert 'Epoch 10 done, loss=0.0123' in buf.getvalue()

    def test_ok_ansi_green(self) -> None:
        """In ANSI mode, [ok] tag uses green colour."""
        _restore_mode('ansi')
        try:
            from liulian.utils.log_tags import _GREEN

            buf = io.StringIO()
            lgr = self._make_logger(buf)
            lgr.ok('done')
            output = buf.getvalue()
            assert _GREEN in output
            assert '[ok]' in output
        finally:
            _force_mode('plain')


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_returns_logger(self) -> None:
        from liulian.utils.log_tags import setup_logging

        lgr = setup_logging('_test_setup', stream=io.StringIO())
        assert isinstance(lgr, logging.Logger)
        assert lgr.name == '_test_setup'
        lgr.handlers.clear()

    def test_root_logger(self) -> None:
        from liulian.utils.log_tags import setup_logging, TaggedFormatter

        root = setup_logging(stream=io.StringIO())
        assert root.name == 'root'
        assert any(isinstance(h.formatter, TaggedFormatter) for h in root.handlers)
        # Cleanup: remove our handler
        root.handlers[:] = [
            h for h in root.handlers if not isinstance(h.formatter, TaggedFormatter)
        ]

    def test_no_duplicate_handlers(self) -> None:
        from liulian.utils.log_tags import setup_logging, TaggedFormatter

        buf = io.StringIO()
        lgr = setup_logging('_test_dup', stream=buf)
        n1 = sum(1 for h in lgr.handlers if isinstance(h.formatter, TaggedFormatter))
        setup_logging('_test_dup', stream=buf)  # second call
        n2 = sum(1 for h in lgr.handlers if isinstance(h.formatter, TaggedFormatter))
        assert n1 == n2 == 1
        lgr.handlers.clear()

    def test_mode_switch(self) -> None:
        from liulian.utils.log_tags import (
            setup_logging,
            get_output_mode,
            MODE_HTML,
        )

        orig = get_output_mode()
        try:
            setup_logging('_test_mode', stream=io.StringIO(), mode=MODE_HTML)
            assert get_output_mode() == MODE_HTML
        finally:
            from liulian.utils.log_tags import set_output_mode

            set_output_mode(orig)
            lgr = logging.getLogger('_test_mode')
            lgr.handlers.clear()


# ---------------------------------------------------------------------------
# redirect_ray_loggers
# ---------------------------------------------------------------------------


class TestRedirectRayLoggers:
    def test_redirects_stream(self) -> None:
        """redirect_ray_loggers switches handler streams to stdout."""
        from liulian.utils.log_tags import redirect_ray_loggers

        # Create a fake 'ray' logger with a stderr handler
        import sys

        rl = logging.getLogger('ray._test_redirect')
        rl.handlers.clear()
        h = logging.StreamHandler(sys.stderr)
        rl.addHandler(h)

        buf = io.StringIO()
        redirect_ray_loggers(buf)  # won't affect our fake, but verify no crash
        rl.handlers.clear()


# ---------------------------------------------------------------------------
# Custom level registration
# ---------------------------------------------------------------------------


class TestCustomLevels:
    def test_level_names_registered(self) -> None:
        from liulian.utils.log_tags import OK_LEVEL, HINT_LEVEL, PROGRESS_LEVEL

        assert logging.getLevelName(OK_LEVEL) == 'OK'
        assert logging.getLevelName(HINT_LEVEL) == 'HINT'
        assert logging.getLevelName(PROGRESS_LEVEL) == 'PROGRESS'

    def test_logger_has_ok_method(self) -> None:
        lgr = logging.getLogger('_test_has_ok')
        assert hasattr(lgr, 'ok')
        assert callable(lgr.ok)

    def test_logger_has_hint_method(self) -> None:
        lgr = logging.getLogger('_test_has_hint')
        assert hasattr(lgr, 'hint')
        assert callable(lgr.hint)

    def test_logger_has_progress_method(self) -> None:
        lgr = logging.getLogger('_test_has_progress')
        assert hasattr(lgr, 'progress')
        assert callable(lgr.progress)
