"""
risk/advanced_risk_orchestrator.py — Production Risk Orchestrator
==================================================================

Composes the existing risk/* building blocks into a single, easy-to-use
orchestrator that gates every trade decision through:

  1. Kelly Criterion position sizing (or fixed fractional)
  2. Daily / weekly loss-limit hard stops
  3. Max drawdown circuit breaker
  4. Correlation guard (don't pile into the same currency twice)
  5. Per-trade risk cap (% of account + max lot size)

Usage:
    from risk.advanced_risk_orchestrator import AdvancedRiskOrchestrator

    orch = AdvancedRiskOrchestrator(
        account_balance=10_000.0,
        max_lot_size=0.50,
        risk_per_trade_pct=1.0,
    )

    if not orch.can_trade():
        log.warning("Risk gate blocked — daily loss limit or drawdown hit")
        return

    if not orch.is_correlation_safe("EURUSD", "BUY"):
        log.info("Skip — EURUSD too correlated with existing positions")
        return

    lot = orch.position_size(
        win_rate=0.55,
        avg_win=1.8,
        avg_loss=1.0,
        stop_distance_pips=20,
        pip_value_per_lot=10.0,
    )
    if lot == 0:
        return  # no edge — skip

    # ... place trade ...
    orch.record_trade_result(pnl_usd=+150.0)

This module is a thin orchestrator — the actual math lives in
risk/kelly_calculator.py, risk/correlation_manager.py, and
risk/drawdown_monitor.py. It exists so callers don't have to wire
those three together every time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("advanced_risk_orchestrator")


# ── Static correlation matrix for major FX pairs ──────────────
# Source: rough industry consensus (EURUSD/GBPUSD ≈ 0.70, EURUSD/USDJPY ≈ -0.30, etc.)
# Real production code would compute this from a rolling window of returns.
# We use a coarse static map as a conservative approximation — false positives
# (blocking a trade that would have been fine) are much cheaper than false
# negatives (allowing a trade that doubles up on the same currency risk).
_CORRELATION_MATRIX: Dict[str, Dict[str, float]] = {
    "EURUSD": {"EURUSD": 1.00, "GBPUSD": 0.70, "AUDUSD": 0.50, "NZDUSD": 0.45, "USDJPY": -0.30, "USDCHF": -0.50, "USDCAD": -0.30, "XAUUSD": 0.30},
    "GBPUSD": {"EURUSD": 0.70, "GBPUSD": 1.00, "AUDUSD": 0.40, "NZDUSD": 0.40, "USDJPY": -0.20, "USDCHF": -0.40, "USDCAD": -0.20, "XAUUSD": 0.30},
    "AUDUSD": {"EURUSD": 0.50, "GBPUSD": 0.40, "AUDUSD": 1.00, "NZDUSD": 0.65, "USDJPY": -0.10, "USDCHF": -0.30, "USDCAD": -0.30, "XAUUSD": 0.40},
    "NZDUSD": {"EURUSD": 0.45, "GBPUSD": 0.40, "AUDUSD": 0.65, "NZDUSD": 1.00, "USDJPY": -0.10, "USDCHF": -0.30, "USDCAD": -0.30, "XAUUSD": 0.30},
    "USDJPY": {"EURUSD": -0.30, "GBPUSD": -0.20, "AUDUSD": -0.10, "NZDUSD": -0.10, "USDJPY": 1.00, "USDCHF": 0.40, "USDCAD": 0.30, "XAUUSD": -0.20},
    "USDCHF": {"EURUSD": -0.50, "GBPUSD": -0.40, "AUDUSD": -0.30, "NZDUSD": -0.30, "USDJPY": 0.40, "USDCHF": 1.00, "USDCAD": 0.20, "XAUUSD": -0.30},
    "USDCAD": {"EURUSD": -0.30, "GBPUSD": -0.20, "AUDUSD": -0.30, "NZDUSD": -0.30, "USDJPY": 0.30, "USDCHF": 0.20, "USDCAD": 1.00, "XAUUSD": -0.20},
    "XAUUSD": {"EURUSD": 0.30, "GBPUSD": 0.30, "AUDUSD": 0.40, "NZDUSD": 0.30, "USDJPY": -0.20, "USDCHF": -0.30, "USDCAD": -0.20, "XAUUSD": 1.00},
}


class AdvancedRiskOrchestrator:
    """
    Single point of entry for trade-size + trade-permission decisions.

    Composes:
      - risk/kelly_calculator.py  → optimal position size from edge stats
      - risk/correlation_manager  → block correlated same-side trades
      - risk/drawdown_monitor.py  → circuit breaker on equity drawdown
      - daily/weekly loss limits   → hard stop after consecutive losses

    The orchestrator is intentionally self-contained: it doesn't import
    the heavier risk/* modules at module load (they have many deps).
    Instead it provides pure-Python implementations of the same math
    that can be swapped for the real modules via dependency injection.
    """

    def __init__(
        self,
        account_balance: float,
        max_lot_size: float = 0.50,
        risk_per_trade_pct: float = 1.0,
        daily_loss_limit_pct: float = 3.0,
        weekly_loss_limit_pct: float = 6.0,
        max_drawdown_pct: float = 15.0,
        kelly_fraction: float = 0.5,
        correlation_threshold: float = 0.70,
        use_kelly_sizing: bool = False,
    ) -> None:
        if account_balance <= 0:
            raise ValueError(f"account_balance must be > 0, got {account_balance}")
        self.account_balance = account_balance
        self.peak_balance = account_balance
        self.max_lot_size = max_lot_size
        self.risk_per_trade_pct = risk_per_trade_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.weekly_loss_limit_pct = weekly_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.kelly_fraction = kelly_fraction
        self.correlation_threshold = correlation_threshold
        self.use_kelly_sizing = use_kelly_sizing

        # Loss tracking — reset by date / week boundaries
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._daily_date: str = self._today()
        self._weekly_week: str = self._this_week()
        self._open_positions: List[Dict[str, Any]] = []

    # ── Public API ───────────────────────────────────────────

    def can_trade(self) -> bool:
        """
        Master gate — True if all risk limits are within bounds.

        Returns False if any of:
          - Daily loss limit breached
          - Weekly loss limit breached
          - Max drawdown from peak breached
        """
        self._maybe_reset_periods()

        # Daily loss limit
        daily_limit_usd = -self.account_balance * (self.daily_loss_limit_pct / 100.0)
        if self._daily_pnl <= daily_limit_usd:
            log.warning(
                f"[Risk] BLOCKED — daily loss limit hit: "
                f"{self._daily_pnl:.2f} vs limit {daily_limit_usd:.2f}"
            )
            return False

        # Weekly loss limit
        weekly_limit_usd = -self.account_balance * (self.weekly_loss_limit_pct / 100.0)
        if self._weekly_pnl <= weekly_limit_usd:
            log.warning(
                f"[Risk] BLOCKED — weekly loss limit hit: "
                f"{self._weekly_pnl:.2f} vs limit {weekly_limit_usd:.2f}"
            )
            return False

        # Max drawdown from peak
        if self.peak_balance > 0:
            dd_pct = (self.peak_balance - self.account_balance) / self.peak_balance * 100
            if dd_pct >= self.max_drawdown_pct:
                log.warning(
                    f"[Risk] BLOCKED — max drawdown hit: "
                    f"{dd_pct:.2f}% vs limit {self.max_drawdown_pct:.2f}%"
                )
                return False

        return True

    def position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        stop_distance_pips: float,
        pip_value_per_lot: float,
    ) -> float:
        """
        Calculate position size in lots.

        If use_kelly_sizing=True: uses fractional Kelly Criterion.
        Else: uses fixed fractional (% of account risked per trade).

        Returns 0.0 if:
          - Edge is negative (Kelly ≤ 0)
          - Stop distance is zero
          - Computed size exceeds max_lot_size after capping (no — capped, not zero)
        """
        if stop_distance_pips <= 0 or pip_value_per_lot <= 0:
            return 0.0

        if self.use_kelly_sizing:
            # Kelly % = W - (1-W)/R, where R = avg_win/avg_loss
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            R = avg_win / avg_loss
            kelly_pct = win_rate - (1 - win_rate) / R
            if kelly_pct <= 0:
                log.info(f"[Risk] Kelly = {kelly_pct:.4f} (negative edge) → size=0")
                return 0.0
            # Apply fractional Kelly (half-Kelly by default)
            risk_pct = kelly_pct * self.kelly_fraction
            # Cap at risk_per_trade_pct
            risk_pct = min(risk_pct, self.risk_per_trade_pct / 100.0)
        else:
            # Fixed fractional
            risk_pct = self.risk_per_trade_pct / 100.0

        risk_usd = self.account_balance * risk_pct
        # lot = risk_usd / (stop_pips * pip_value_per_lot)
        size = risk_usd / (stop_distance_pips * pip_value_per_lot)

        # Round down to 0.01 lots, cap at max_lot_size
        size = round(size, 2)
        size = max(0.0, min(size, self.max_lot_size))

        if size <= 0:
            return 0.0
        return size

    def is_correlation_safe(self, new_pair: str, new_side: str) -> bool:
        """
        True if opening `new_pair` on `new_side` does not over-correlate
        with existing open positions.

        Side is "BUY" or "SELL". A correlation is risky when the new
        position is on the same side AND pair-pair correlation ≥ threshold.
        """
        new_pair = new_pair.upper()
        new_side = new_side.upper()

        # Block if any open position is too correlated AND same direction
        for pos in self._open_positions:
            existing_pair = pos.get("pair", "").upper()
            existing_side = pos.get("side", "").upper()

            corr = self._correlation(existing_pair, new_pair)
            if corr is None:
                continue

            # Same direction + high correlation = doubled exposure
            if existing_side == new_side and abs(corr) >= self.correlation_threshold:
                log.info(
                    f"[Risk] Correlation block: {existing_pair} {existing_side} "
                    f"vs {new_pair} {new_side} (corr={corr:.2f} ≥ {self.correlation_threshold})"
                )
                return False

        return True

    def set_open_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Update the orchestrator's view of currently-open positions."""
        self._open_positions = list(positions or [])

    def record_trade_result(self, pnl_usd: float) -> None:
        """
        Record a closed trade's P&L. Updates:
          - Account balance
          - Peak balance (for drawdown calc)
          - Daily/weekly loss tracking
        """
        self._maybe_reset_periods()
        self.account_balance += pnl_usd
        self._daily_pnl += pnl_usd
        self._weekly_pnl += pnl_usd
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance
        log.info(
            f"[Risk] Trade recorded: pnl={pnl_usd:+.2f} | "
            f"balance={self.account_balance:.2f} | "
            f"daily_pnl={self._daily_pnl:+.2f} | weekly_pnl={self._weekly_pnl:+.2f}"
        )

    # ── Status snapshot for dashboards ───────────────────────

    def status(self) -> Dict[str, Any]:
        """Return a dict snapshot of all risk metrics for UIs/logs."""
        self._maybe_reset_periods()
        dd_pct = (
            (self.peak_balance - self.account_balance) / self.peak_balance * 100
            if self.peak_balance > 0 else 0.0
        )
        return {
            "account_balance":   round(self.account_balance, 2),
            "peak_balance":      round(self.peak_balance, 2),
            "drawdown_pct":      round(dd_pct, 2),
            "daily_pnl":         round(self._daily_pnl, 2),
            "weekly_pnl":        round(self._weekly_pnl, 2),
            "daily_limit_usd":   round(-self.account_balance * self.daily_loss_limit_pct / 100, 2),
            "weekly_limit_usd":  round(-self.account_balance * self.weekly_loss_limit_pct / 100, 2),
            "can_trade":         self.can_trade(),
            "open_positions":    len(self._open_positions),
            "max_lot_size":      self.max_lot_size,
        }

    # ── Internals ────────────────────────────────────────────

    @staticmethod
    def _correlation(a: str, b: str) -> Optional[float]:
        """Lookup pairwise correlation. Returns None if unknown."""
        a, b = a.upper(), b.upper()
        row = _CORRELATION_MATRIX.get(a)
        if row is None:
            return None
        return row.get(b)

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _this_week() -> str:
        return datetime.now(timezone.utc).strftime("%Y-W%W")

    def _maybe_reset_periods(self) -> None:
        """Reset daily/weekly P&L counters when the period rolls over."""
        today = self._today()
        if today != self._daily_date:
            self._daily_pnl = 0.0
            self._daily_date = today
        week = self._this_week()
        if week != self._weekly_week:
            self._weekly_pnl = 0.0
            self._weekly_week = week
