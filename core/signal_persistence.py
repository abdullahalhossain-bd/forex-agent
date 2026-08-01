"""
core/signal_persistence.py — Day 97+ Signal Persistence Filter
================================================================
Book reference: "The Only Technical Analysis Book You Will Ever Need" (Brian Hale)
Page 15: "Avoid acting on high-frequency signal flip-flopping; build in a filter
for signal stability/persistence"

Problem this solves:
  - In choppy/ranging markets, signals flip BUY→SELL→BUY→SELL rapidly
  - Each flip looks like a valid entry, but they're noise
  - Trading every flip = death by a thousand cuts (spread + commission)

Solution:
  Track recent signal history per pair. If signal direction changed more than
  N times in the last M bars, suppress new entries — the market is too
  indecisive to trade safely.

Usage:
    from core.signal_persistence import SignalPersistenceFilter
    spf = SignalPersistenceFilter()
    if not spf.is_stable(symbol="EURUSD", current_signal="BUY"):
        # signal is flip-flopping — don't trade
        return "NO TRADE"
    spf.record(symbol="EURUSD", signal="BUY", confidence=65)
"""

import json
import time
from collections import deque, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger("signal_persistence")


class SignalPersistenceFilter:
    """Filters out flip-flopping (unstable) signals.

    Rules (from book Page 15):
      1. If signal direction changed > MAX_FLIPS in last WINDOW_BARS bars → suppress
      2. If signal is NEW (first ever for this pair) → require MIN_CONFIDENCE to enter
      3. If last signal was opposite direction < MIN_BARS_SAME_DIR ago → suppress
         (this catches the "I just said SELL, now I'm saying BUY" case)
    """

    # Config
    WINDOW_BARS = 20          # look back this many signal records
    MAX_FLIPS = 3             # more than 3 direction changes in 20 bars = unstable
    MIN_BARS_SAME_DIR = 2     # signal must persist for at least 2 bars before acting
    MIN_CONFIDENCE_NEW = 50   # first-ever signal for a pair needs ≥50% confidence

    def __init__(self):
        # Per-symbol signal history: {symbol: deque of (timestamp, signal, confidence)}
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.WINDOW_BARS))

    def record(self, symbol: str, signal: str, confidence: float = 0) -> None:
        """Record a signal decision for this symbol."""
        now = time.time()
        self._history[symbol].append((now, signal.upper(), confidence))
        log.debug(f"[SPF] {symbol} signal recorded: {signal} ({confidence}%)")

    def is_stable(self, symbol: str, current_signal: str) -> bool:
        """Check if the current signal is stable enough to trade.

        Returns True if signal is stable (OK to trade), False if flip-flopping.
        """
        history = self._history.get(symbol)
        if not history or len(history) < 2:
            # Not enough history — first signal. Allow if confidence is adequate.
            return True

        records = list(history)

        # Rule 1: Count direction flips in the window
        flips = 0
        prev_dir = None
        for _, sig, _ in records:
            direction = self._direction(sig)
            if prev_dir is not None and direction != prev_dir and direction != "NEUTRAL":
                flips += 1
            if direction != "NEUTRAL":
                prev_dir = direction

        if flips > self.MAX_FLIPS:
            log.info(
                f"[SPF] {symbol} signal UNSTABLE — {flips} flips in last "
                f"{len(records)} bars (max {self.MAX_FLIPS}). Suppressing entry."
            )
            return False

        # Rule 2: Check minimum persistence (same direction for ≥ MIN_BARS_SAME_DIR)
        current_dir = self._direction(current_signal)
        if current_dir == "NEUTRAL":
            return False  # WAIT/NO TRADE is never "stable" for entry

        same_dir_count = 0
        for _, sig, _ in reversed(records):
            if self._direction(sig) == current_dir:
                same_dir_count += 1
            else:
                break  # stop at first different signal

        if same_dir_count < self.MIN_BARS_SAME_DIR:
            log.info(
                f"[SPF] {symbol} signal too new — only {same_dir_count} bar(s) "
                f"in {current_dir} direction (min {self.MIN_BARS_SAME_DIR}). "
                f"Waiting for persistence."
            )
            return False

        return True

    def get_flip_count(self, symbol: str) -> int:
        """Return number of direction flips in current window."""
        history = self._history.get(symbol)
        if not history or len(history) < 2:
            return 0
        records = list(history)
        flips = 0
        prev_dir = None
        for _, sig, _ in records:
            direction = self._direction(sig)
            if prev_dir is not None and direction != prev_dir and direction != "NEUTRAL":
                flips += 1
            if direction != "NEUTRAL":
                prev_dir = direction
        return flips

    @staticmethod
    def _direction(signal: str) -> str:
        """Normalize signal to direction."""
        s = signal.upper().strip()
        if s in ("BUY", "STRONG_BUY", "BULLISH", "LONG"):
            return "BUY"
        if s in ("SELL", "STRONG_SELL", "BEARISH", "SHORT"):
            return "SELL"
        return "NEUTRAL"


# ── Singleton ─────────────────────────────────────────────────────

_SPF: Optional[SignalPersistenceFilter] = None


def get_signal_persistence_filter() -> SignalPersistenceFilter:
    global _SPF
    if _SPF is None:
        _SPF = SignalPersistenceFilter()
    return _SPF
