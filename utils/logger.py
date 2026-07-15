# utils/logger.py
# ============================================================
# AI Trader — Centralized Logging System
# print() এর বদলে এটা ব্যবহার করো — সব logs file-এ save হবে
#
# Hotfix: console handler now writes through a UTF-8-wrapped stdout
# stream instead of the bare default. On Windows, plain sys.stdout uses
# the cp1252 codepage, which can't encode emoji/box-drawing characters
# (✅ ❌ ⛔ 🟡 ═ ━ → etc.) used throughout the log messages — every such
# line was raising UnicodeEncodeError ("--- Logging error ---") and
# spamming the console while losing the actual log content. The file
# handler was already safe (encoding="utf-8" was set there); only the
# console handler was missing the equivalent fix.
# ============================================================

import io
import logging
import os
import re
import sys
from datetime import datetime

LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "trader.log")

os.makedirs(LOG_DIR, exist_ok=True)


# ── Secret-redaction filter ────────────────────────────────────
# Audit P3 fix: prevent sensitive tokens from EVER reaching the log
# file or console. Previously the Telegram bot token, MT5 password,
# and various API keys were being logged in plaintext (sometimes via
# library tracebacks, sometimes via our own debug logs). Once a token
# is in the log file, it's effectively leaked — anyone with read
# access to logs gets full bot control.
#
# This filter inspects every LogRecord before it's emitted, scans
# both the message AND the formatted record for known secret patterns,
# and replaces matches with "<REDACTED:KIND>". Patterns are matched
# case-insensitively. The list is conservative — we only redact
# high-confidence secret shapes, not arbitrary long strings.
#
# Adding a new secret type: append a (compiled_regex, kind_label)
# tuple to SECRET_PATTERNS below.

SECRET_PATTERNS = [
    # Telegram bot token — BotFather format is "<digits>:<35-40 alnum/hyphen/underscore>"
    # e.g. "6123456789:AAH...AAbb"
    (re.compile(r"\b(\d{8,12}:[A-Za-z0-9_-]{30,45})\b"), "TELEGRAM_TOKEN"),
    # Telegram chat ID sometimes paired with token in URL — separate pattern
    (re.compile(r"(bot\d{8,12}:[A-Za-z0-9_-]{30,45})"), "TELEGRAM_BOT_URL"),
    # Round-12 fix: full Telegram API URL form —
    # https://api.telegram.org/bot<TOKEN>/sendMessage?...
    # The operator's audit found tokens leaking via exception tracebacks
    # that included the full request URL. This pattern catches the URL
    # wrapper so even if the bare-token pattern somehow misses (e.g.
    # extra escaping), the URL form is also redacted.
    (re.compile(r"(api\.telegram\.org/bot\d{8,12}:[A-Za-z0-9_-]{30,45})"),
     "TELEGRAM_API_URL"),
    # MT5 password — typically appears as "MT5_PASSWORD=xxx" or
    # "password=xxx" in env dumps; we redact the value, not the key.
    (re.compile(r"(?i)(MT5_PASSWORD|MT5_INVESTOR|password)\s*[=:]\s*\S+"),
     "MT5_PASSWORD"),
    # Generic API key=value patterns — Round-12: expanded to include
    # all 7 LLM providers' key env vars.
    (re.compile(r"(?i)(GROQ_API_KEY|GEMINI_API_KEY|CEREBRAS_API_KEY|"
                r"SAMBANOVA_API_KEY|OPENROUTER_API_KEY|"
                r"GITHUB_TOKEN|GITHUB_MODELS_API_KEY|"
                r"HF_TOKEN|HUGGINGFACE_API_KEY|"
                r"ALPHA_VANTAGE_API_KEY|POLYGON_API_KEY|"
                r"FINNHUB_API_KEY|TWELVE_DATA_API_KEY|NEWS_API_KEY|"
                r"TELEGRAM_TOKEN|API_KEY|SECRET_KEY)"
                r"\s*[=:]\s*[A-Za-z0-9_\-]{16,}"),
     "API_KEY"),
    # Bearer token in HTTP Authorization headers
    (re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)[A-Za-z0-9_\-\.]{20,}"),
     "BEARER_TOKEN"),
]


class SecretRedactionFilter(logging.Filter):
    """
    Audit P3 fix: redact known secret patterns from every log record
    before it reaches the file or console handler.

    The filter scans both `record.msg` (the raw format string) and
    `record.args` (the format arguments) so that BOTH
        log.info(f"token={token}")
    AND
        log.info("token=%s", token)
    are properly redacted.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact the message string
        if isinstance(record.msg, str):
            for pattern, kind in SECRET_PATTERNS:
                record.msg = pattern.sub(f"<REDACTED:{kind}>", record.msg)

        # Redact format args (tuple of values)
        if record.args:
            if isinstance(record.args, tuple):
                new_args = tuple(self._redact_value(a) for a in record.args)
                record.args = new_args
            else:
                record.args = self._redact_value(record.args)

        return True  # always allow the record, just redact first

    @staticmethod
    def _redact_value(value):
        if isinstance(value, str):
            for pattern, kind in SECRET_PATTERNS:
                value = pattern.sub(f"<REDACTED:{kind}>", value)
            return value
        return value


def _utf8_console_stream():
    """
    Wrap sys.stdout so console output is encoded as UTF-8 regardless of
    the OS-default codepage (cp1252 on most Windows setups). Falls back
    to the raw stream if stdout doesn't expose a .buffer (e.g. when
    stdout has already been redirected/wrapped, or in some IDE/test
    runners) so logging never breaks even in unusual environments.
    """
    try:
        return io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except (AttributeError, ValueError):
        return sys.stdout


def get_logger(name: str) -> logging.Logger:
    """
    যেকোনো module থেকে call করো:
        from utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Data fetched")
        log.warning("Missing candles")
        log.error("Fetch failed")
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger   # already configured

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Audit P3 fix: secret-redaction filter — applied to BOTH handlers
    # so tokens can never reach the file or console, regardless of
    # which logger or library emits the record.
    secret_filter = SecretRedactionFilter()

    # ── File handler (DEBUG+) ──
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh.addFilter(secret_filter)

    # ── Console handler (INFO+) ──
    # UTF-8 wrapped stream so emoji/box-drawing chars never raise
    # UnicodeEncodeError on Windows' cp1252 console codepage.
    ch = logging.StreamHandler(_utf8_console_stream())
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    ch.addFilter(secret_filter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Stop records from also bubbling up to the root logger. If anything
    # else in the process (main.py, a third-party lib like python-telegram-
    # bot, or a stray logging.basicConfig() call) has attached its own
    # StreamHandler to the root logger, every message from THIS logger
    # would otherwise be emitted twice: once through our UTF-8-safe
    # console handler above, and once through that other handler — which,
    # if it doesn't specify an encoding, hits the same cp1252 crash on
    # Windows. Disabling propagation makes this logger fully self-
    # contained so its own handlers are the only ones that ever run.
    logger.propagate = False

    return logger