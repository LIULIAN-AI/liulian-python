"""Structured logging with coloured severity tags.

Multi-target tag system that works correctly across:

* **CLI / Terminal** — ANSI colour codes for instant visual distinction.
* **Jupyter notebooks** — identical ANSI rendering (most kernels support it).
* **Plain text / CI logs** — colour codes automatically stripped when stdout
  is not a TTY (or when ``NO_COLOR`` / ``LIULIAN_NO_COLOR`` env vars are set),
  leaving clean ``[info]``, ``[ok]``, … prefixes.
* **HTML** — styled ``<span>`` elements for HTML reports / notebook HTML
  output (activated by ``LIULIAN_LOG_HTML`` env var or ``set_output_mode('html')``).

Styling rules:

* **Normal levels** — only the ``[tag]`` bracket is coloured; the following
  message text stays in the default (plain) style.
* **ERROR / CRITICAL** — the *entire line* (tag + message) is rendered in
  red text on a dark-red background for maximum visibility.

Works with **both** ``print()`` and the standard :mod:`logging` module::

    # ── print-based usage ──
    from liulian.utils.log_tags import info, success, WARNING_TAG

    info('Ray initialised with 8 CPUs.')
    success('Training complete — best val MSE: 0.0123')
    print(f'{WARNING_TAG}checkpoint missing, retraining.')

    # ── logging-based usage ──
    import logging
    from liulian.utils.log_tags import setup_logging

    setup_logging()                                 # configures root logger
    logger = logging.getLogger(__name__)
    logger.info('Training started')                 # → [info] Training started
    logger.error('CUDA OOM')                        # → [ERROR] CUDA OOM  (all red)

Output mode is determined **once** at import time by :func:`_detect_mode`.
Override programmatically with :func:`set_output_mode` or env vars.
"""

from __future__ import annotations

import logging as _logging
import os
import re as _re
import sys
from typing import IO, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Custom log levels  (registered early so they're usable everywhere)
# ═══════════════════════════════════════════════════════════════════════════

HINT_LEVEL = 21       # advisory hint — just above INFO (20)
PROGRESS_LEVEL = 22   # progress marker
OK_LEVEL = 25         # success milestone — between INFO (20) and WARNING (30)

_logging.addLevelName(OK_LEVEL, 'OK')
_logging.addLevelName(HINT_LEVEL, 'HINT')
_logging.addLevelName(PROGRESS_LEVEL, 'PROGRESS')


def _log_ok(self: _logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(OK_LEVEL):
        self._log(OK_LEVEL, message, args, **kwargs)


def _log_hint(self: _logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(HINT_LEVEL):
        self._log(HINT_LEVEL, message, args, **kwargs)


def _log_progress(self: _logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(PROGRESS_LEVEL):
        self._log(PROGRESS_LEVEL, message, args, **kwargs)


# Monkey-patch Logger so every logger gains .ok(), .hint(), .progress()
_logging.Logger.ok = _log_ok            # type: ignore[attr-defined]
_logging.Logger.hint = _log_hint        # type: ignore[attr-defined]
_logging.Logger.progress = _log_progress  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════
# Output mode detection
# ═══════════════════════════════════════════════════════════════════════════

MODE_ANSI = 'ansi'
MODE_PLAIN = 'plain'
MODE_HTML = 'html'


def _detect_mode() -> str:
    """Auto-detect the best output mode.

    Resolution order:
    1. ``LIULIAN_LOG_HTML`` env var → HTML.
    2. ``NO_COLOR`` or ``LIULIAN_NO_COLOR`` → plain.
    3. ``LIULIAN_FORCE_COLOR`` → ANSI.
    4. ``stdout.isatty()`` → ANSI.
    5. IDE consoles (PyCharm, VS Code, etc.) that support ANSI but
       don't report ``isatty=True`` → ANSI.
    6. Running inside Jupyter → ANSI.
    7. Fallback → plain.
    """
    if os.environ.get('LIULIAN_LOG_HTML') is not None:
        return MODE_HTML
    if os.environ.get('NO_COLOR') is not None:
        return MODE_PLAIN
    if os.environ.get('LIULIAN_NO_COLOR') is not None:
        return MODE_PLAIN
    if os.environ.get('LIULIAN_FORCE_COLOR') is not None:
        return MODE_ANSI
    try:
        if sys.stdout.isatty():
            return MODE_ANSI
    except AttributeError:
        pass
    # IDE run consoles support ANSI colours but don't report isatty=True.
    # PyCharm sets PYCHARM_HOSTED=1; VS Code sets TERM_PROGRAM=vscode.
    if os.environ.get('PYCHARM_HOSTED') is not None:
        return MODE_ANSI
    if os.environ.get('TERM_PROGRAM') == 'vscode':
        return MODE_ANSI
    # Jupyter kernels don't always report isatty=True
    try:
        get_ipython  # type: ignore[name-defined]
        return MODE_ANSI
    except NameError:
        pass
    return MODE_PLAIN


_OUTPUT_MODE: str = _detect_mode()


def set_output_mode(mode: str) -> None:
    """Switch output mode (``'ansi'``, ``'plain'``, or ``'html'``)."""
    global _OUTPUT_MODE
    if mode not in (MODE_ANSI, MODE_PLAIN, MODE_HTML):
        raise ValueError(f"mode must be one of {MODE_ANSI!r}, "
                         f"{MODE_PLAIN!r}, {MODE_HTML!r}")
    _OUTPUT_MODE = mode
    _rebuild_tags()


def get_output_mode() -> str:
    """Return the current output mode string."""
    return _OUTPUT_MODE


# Backward compatibility
def set_colour_enabled(enabled: bool) -> None:
    """Legacy helper — calls ``set_output_mode``."""
    set_output_mode(MODE_ANSI if enabled else MODE_PLAIN)


def colour_enabled() -> bool:
    """Return True if ANSI colour output is currently active."""
    return _OUTPUT_MODE == MODE_ANSI


# ═══════════════════════════════════════════════════════════════════════════
# ANSI escape codes
# ═══════════════════════════════════════════════════════════════════════════

_RESET = '\033[0m'
_BOLD = '\033[1m'

# Foreground colours
_RED = '\033[91m'
_GREEN = '\033[92m'
_YELLOW = '\033[93m'
_BLUE = '\033[94m'
_MAGENTA = '\033[95m'
_CYAN = '\033[96m'
_GREY = '\033[90m'

# Background for errors  (256-colour deep maroon)
_BG_DARK_RED = '\033[48;5;52m'


# ═══════════════════════════════════════════════════════════════════════════
# HTML colour palette
# ═══════════════════════════════════════════════════════════════════════════

_HTML_COLOURS: dict[str, str] = {
    'blue': '#5c9eff',
    'green': '#4ec34e',
    'yellow': '#e0c346',
    'red': '#ff6b6b',
    'grey': '#888888',
    'cyan': '#56c8d8',
    'magenta': '#d670d6',
}

_HTML_ERROR_FG = '#ff6b6b'
_HTML_ERROR_BG = '#3d0000'


# ═══════════════════════════════════════════════════════════════════════════
# Tag builders
# ═══════════════════════════════════════════════════════════════════════════

def _tag(label: str, ansi_colour: str, html_colour_key: str) -> str:
    """Build a **prefix** tag ``[label] `` for *non-error* levels.

    Only the ``[label]`` portion is coloured; the message stays plain.
    """
    if _OUTPUT_MODE == MODE_ANSI:
        return f'{ansi_colour}{_BOLD}[{label}]{_RESET} '
    if _OUTPUT_MODE == MODE_HTML:
        c = _HTML_COLOURS.get(html_colour_key, html_colour_key)
        return (f'<span style="color:{c};font-weight:bold">'
                f'[{label}]</span> ')
    return f'[{label}] '


def _error_format(label: str, msg: str) -> str:
    """Return a *fully-styled* error string — tag **and** body are red.

    ANSI  — bright-red text on dark-red background.
    HTML  — ``<span>`` with equivalent colours.
    Plain — bare ``[ERROR] msg``.
    """
    if _OUTPUT_MODE == MODE_ANSI:
        return f'{_RED}{_BG_DARK_RED}{_BOLD}[{label}]{_RESET}{_RED}{_BG_DARK_RED} {msg}{_RESET}'
    if _OUTPUT_MODE == MODE_HTML:
        return (f'<span style="color:{_HTML_ERROR_FG};'
                f'background-color:{_HTML_ERROR_BG};'
                f'font-weight:bold">[{label}] {msg}</span>')
    return f'[{label}] {msg}'


# ═══════════════════════════════════════════════════════════════════════════
# Pre-built tag strings  (non-error levels)
# ═══════════════════════════════════════════════════════════════════════════

INFO_TAG: str = ''
SUCCESS_TAG: str = ''
WARNING_TAG: str = ''
ERROR_TAG: str = ''      # tag-only prefix (for print f-string use)
DEBUG_TAG: str = ''
HINT_TAG: str = ''
PROGRESS_TAG: str = ''


def _rebuild_tags() -> None:
    """(Re)build module-level tag strings from the current output mode."""
    global INFO_TAG, SUCCESS_TAG, WARNING_TAG, ERROR_TAG
    global DEBUG_TAG, HINT_TAG, PROGRESS_TAG
    INFO_TAG = _tag('info', _BLUE, 'blue')
    SUCCESS_TAG = _tag('ok', _GREEN, 'green')
    WARNING_TAG = _tag('warn', _YELLOW, 'yellow')
    ERROR_TAG = _tag('ERROR', _RED, 'red')
    DEBUG_TAG = _tag('debug', _GREY, 'grey')
    HINT_TAG = _tag('hint', _CYAN, 'cyan')
    PROGRESS_TAG = _tag('...', _MAGENTA, 'magenta')


_rebuild_tags()


# ═══════════════════════════════════════════════════════════════════════════
# Convenience printers
# ═══════════════════════════════════════════════════════════════════════════

def tagged(label: str, ansi_colour: str, html_colour_key: str,
           msg: str, *, file: Optional[IO] = None) -> None:
    """Print *msg* with a custom coloured ``[label]`` prefix."""
    print(f'{_tag(label, ansi_colour, html_colour_key)}{msg}',
          file=file or sys.stdout)


def info(msg: str, *, file: Optional[IO] = None) -> None:
    """``[info]`` tag (blue).  Message body is plain."""
    print(f'{INFO_TAG}{msg}', file=file or sys.stdout)


def success(msg: str, *, file: Optional[IO] = None) -> None:
    """``[ok]`` tag (green).  Message body is plain."""
    print(f'{SUCCESS_TAG}{msg}', file=file or sys.stdout)


def warning(msg: str, *, file: Optional[IO] = None) -> None:
    """``[warn]`` tag (yellow).  Message body is plain."""
    print(f'{WARNING_TAG}{msg}', file=file or sys.stdout)


def error(msg: str, *, file: Optional[IO] = None) -> None:
    """``[ERROR]`` — **entire line** is red on dark-red background."""
    print(_error_format('ERROR', msg), file=file or sys.stderr)


def debug(msg: str, *, file: Optional[IO] = None) -> None:
    """``[debug]`` tag (grey).  Message body is plain."""
    print(f'{DEBUG_TAG}{msg}', file=file or sys.stdout)


def hint(msg: str, *, file: Optional[IO] = None) -> None:
    """``[hint]`` tag (cyan).  Message body is plain."""
    print(f'{HINT_TAG}{msg}', file=file or sys.stdout)


def progress(msg: str, *, file: Optional[IO] = None) -> None:
    """``[...]`` tag (magenta).  Message body is plain."""
    print(f'{PROGRESS_TAG}{msg}', file=file or sys.stdout)


# ═══════════════════════════════════════════════════════════════════════════
# Text-stripping utilities
# ═══════════════════════════════════════════════════════════════════════════

_ANSI_RE = _re.compile(r'\033\[[0-9;]*m')
_HTML_SPAN_RE = _re.compile(r'</?span[^>]*>')


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from *text*."""
    return _ANSI_RE.sub('', text)


def strip_html_tags(text: str) -> str:
    """Remove ``<span …>`` / ``</span>`` wrappers from *text*."""
    return _HTML_SPAN_RE.sub('', text)


def strip_formatting(text: str) -> str:
    """Remove both ANSI and HTML formatting from *text*."""
    return strip_html_tags(strip_ansi(text))


# ═══════════════════════════════════════════════════════════════════════════
# Tag-detection (prevents double-tagging in TaggedFormatter)
# ═══════════════════════════════════════════════════════════════════════════

_TAG_LABELS = ('info', 'ok', 'warn', 'ERROR', 'CRIT', 'debug', 'hint', '...')

# Matches an optional ANSI leader or HTML <span> followed by [label]
_TAG_DETECT = _re.compile(
    r'(?:\033\[[0-9;]*m|<span[^>]*>)*\[(?:'
    + '|'.join(_re.escape(lbl) for lbl in _TAG_LABELS)
    + r')\]'
)


def _msg_has_tag(msg: str) -> bool:
    """Return True if *msg* already starts with a known tag."""
    return bool(_TAG_DETECT.match(msg))


# ═══════════════════════════════════════════════════════════════════════════
# Python ``logging`` integration
# ═══════════════════════════════════════════════════════════════════════════

class TaggedFormatter(_logging.Formatter):
    """Logging formatter that auto-prepends coloured severity tags.

    Styling rules:

    * **DEBUG / INFO / WARNING** — a coloured ``[tag]`` is prepended;
      the message text remains in the default (plain) style.
    * **ERROR / CRITICAL** — the *entire line* (tag **and** body) is
      rendered in red text on a dark-red background.
    * If the message already starts with a known tag (e.g.
      ``SUCCESS_TAG``), the auto-tag is **skipped** to avoid
      double-tagging.

    Set ``auto_tag=False`` to disable automatic tagging entirely.

    Example::

        import logging
        from liulian.utils.log_tags import TaggedFormatter

        handler = logging.StreamHandler()
        handler.setFormatter(TaggedFormatter())
        logger = logging.getLogger('demo')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info('starting')     # → [info] starting
        logger.error('crash')       # → [ERROR] crash  (all red)
    """

    _LEVEL_SPEC: dict[int, tuple[str, str, str]] = {
        _logging.DEBUG:    ('debug', _GREY, 'grey'),
        _logging.INFO:     ('info', _BLUE, 'blue'),
        HINT_LEVEL:        ('hint', _CYAN, 'cyan'),
        PROGRESS_LEVEL:    ('...', _MAGENTA, 'magenta'),
        OK_LEVEL:          ('ok', _GREEN, 'green'),
        _logging.WARNING:  ('warn', _YELLOW, 'yellow'),
        _logging.ERROR:    ('ERROR', _RED, 'red'),
        _logging.CRITICAL: ('CRIT', _RED, 'red'),
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        *,
        auto_tag: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(fmt or '%(message)s', datefmt, **kwargs)
        self.auto_tag = auto_tag

    def format(self, record: _logging.LogRecord) -> str:  # noqa: D401
        msg = super().format(record)

        if not self.auto_tag or _msg_has_tag(msg):
            return msg

        label, ansi_colour, html_key = self._LEVEL_SPEC.get(
            record.levelno, ('info', _BLUE, 'blue')
        )

        if record.levelno >= _logging.ERROR:
            return _error_format(label, msg)

        return f'{_tag(label, ansi_colour, html_key)}{msg}'


def setup_logging(
    name: Optional[str] = None,
    level: int = _logging.INFO,
    *,
    auto_tag: bool = True,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    stream: Optional[IO] = None,
    mode: Optional[str] = None,
) -> _logging.Logger:
    """Configure and return a logger that uses :class:`TaggedFormatter`.

    When *name* is ``None`` the **root** logger is configured, giving
    every child logger in the process automatic tagged output — a
    drop-in replacement for :func:`logging.basicConfig`.

    Args:
        name: Logger name (``None`` → root logger).
        level: Minimum log level.
        auto_tag: Automatically prepend a tag based on log level.
        fmt: Format string for :class:`TaggedFormatter`
             (default ``'%(message)s'``).
        datefmt: Date format string (passed through to the formatter).
        stream: Output stream (default ``sys.stdout``).  Using stdout
              rather than stderr prevents IDEs (PyCharm, VS Code) from
              rendering every log line in red.
        mode: Force output mode (``'ansi'``, ``'plain'``, ``'html'``).
              ``None`` keeps the auto-detected mode.

    Returns:
        The configured :class:`~logging.Logger`.
    """
    if mode is not None:
        set_output_mode(mode)

    lgr = _logging.getLogger(name)
    lgr.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not any(
        isinstance(h.formatter, TaggedFormatter)
        for h in lgr.handlers
    ):
        handler = _logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(
            TaggedFormatter(fmt=fmt, datefmt=datefmt, auto_tag=auto_tag),
        )
        lgr.addHandler(handler)

    return lgr


def redirect_ray_loggers(stream: Optional[IO] = None) -> None:
    """Force Ray's internal loggers to write to *stream* (default stdout).

    Ray emits logs via its own loggers (``ray``, ``ray.tune``,
    ``ray._private``, etc.) which default to stderr — rendering every
    line red in PyCharm.  Call this **after** ``ray.init()`` (or lazily
    the first time Ray logs) to redirect all existing and future Ray
    handlers to *stream*.
    """
    target = stream or sys.stdout
    for logger_name in ('ray', 'ray.tune', 'ray._private',
                        'ray.data', 'ray.train', 'ray.air',
                        'ray._private.worker'):
        rl = _logging.getLogger(logger_name)
        for h in rl.handlers:
            if isinstance(h, _logging.StreamHandler):
                h.stream = target
        # Ensure future handlers also use stdout by installing a
        # custom StreamHandler pointing at the right target if the
        # logger has no handlers yet.
        if not rl.handlers:
            sh = _logging.StreamHandler(target)
            rl.addHandler(sh)
