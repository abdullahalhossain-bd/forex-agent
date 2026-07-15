# risk/trading_controls.py — Pre-trade guard rails (ported from zipline)
# =============================================================================
# Ported from: https://github.com/quantopian/zipline/blob/master/zipline/finance/controls.py
# Original author: Quantopian — Apache 2.0 license
#
# Trading controls are PRE-TRADE GUARD RAILS that validate every order before
# it reaches the broker. If a guard is violated, the order is blocked (either
# raises an exception or logs a warning, configurable via `on_error`).
#
# Available controls:
#   - MaxOrderCount: limit number of orders per day
#   - MaxOrderSize: limit single order size (shares or notional)
#   - MaxPositionSize: limit total position size after order fills
#   - LongOnly: prohibit short positions
#
# Usage:
#     controls = TradingControls()
#     controls.add(MaxOrderCount(max_count=10))
#     controls.add(MaxPositionSize(max_notional=50000))
#     controls.add(LongOnly())
#
#     # Before each order:
#     controls.validate(asset="EURUSD", amount=1000, portfolio=portfolio_state)
# =============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("trading_controls")


class TradingControlViolation(Exception):
    """Raised when a trading control is violated."""
    def __init__(self, asset: str, amount: float, constraint: str):
        self.asset = asset
        self.amount = amount
        self.constraint = constraint
        super().__init__(
            f"Trading control violation: {constraint} "
            f"(asset={asset}, amount={amount})"
        )


@dataclass
class PortfolioState:
    """Minimal portfolio state needed by trading controls."""
    positions: dict[str, float] = field(default_factory=dict)  # symbol → shares
    cash: float = 0.0


class TradingControl(ABC):
    """Abstract base class for pre-trade guard rails."""

    def __init__(self, on_error: str = "fail"):
        """
        Parameters
        ----------
        on_error : "fail" (raise exception) or "log" (log warning + block).
        """
        if on_error not in ("fail", "log"):
            raise ValueError(f"on_error must be 'fail' or 'log', got {on_error!r}")
        self.on_error = on_error

    @abstractmethod
    def validate(
        self,
        asset: str,
        amount: float,
        portfolio: PortfolioState,
        current_price: float = 0.0,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if the order is allowed. Returns True if OK, False if blocked.
        If blocked and on_error="fail", raises TradingControlViolation.
        """
        ...

    def _fail(self, asset: str, amount: float, msg: str) -> bool:
        if self.on_error == "fail":
            raise TradingControlViolation(asset, amount, msg)
        else:
            log.warning(f"Trading control blocked: {msg} (asset={asset}, amount={amount})")
            return False


class MaxOrderCount(TradingControl):
    """Limit the number of orders placed per day."""

    def __init__(self, max_count: int = 100, on_error: str = "fail"):
        super().__init__(on_error)
        self.max_count = max_count
        self._current_date = None
        self._orders_today = 0

    def validate(self, asset, amount, portfolio, current_price=0.0, current_time=None):
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        today = current_time.date()
        if self._current_date != today:
            self._current_date = today
            self._orders_today = 0

        if self._orders_today >= self.max_count:
            return self._fail(asset, amount,
                f"MaxOrderCount: {self._orders_today}/{self.max_count} orders today")

        self._orders_today += 1
        return True


class MaxOrderSize(TradingControl):
    """Limit the size of any single order (shares or notional)."""

    def __init__(self, max_shares: Optional[float] = None,
                 max_notional: Optional[float] = None, on_error: str = "fail"):
        super().__init__(on_error)
        if max_shares is None and max_notional is None:
            raise ValueError("Must supply at least one of max_shares or max_notional")
        self.max_shares = max_shares
        self.max_notional = max_notional

    def validate(self, asset, amount, portfolio, current_price=0.0, current_time=None):
        if self.max_shares is not None and abs(amount) > self.max_shares:
            return self._fail(asset, amount,
                f"MaxOrderSize: {abs(amount)} shares > max {self.max_shares}")

        notional = abs(amount) * current_price
        if self.max_notional is not None and notional > self.max_notional:
            return self._fail(asset, amount,
                f"MaxOrderSize: ${notional:.2f} notional > max ${self.max_notional:.2f}")

        return True


class MaxPositionSize(TradingControl):
    """Limit the total position size after the order fills."""

    def __init__(self, max_shares: Optional[float] = None,
                 max_notional: Optional[float] = None, on_error: str = "fail"):
        super().__init__(on_error)
        if max_shares is None and max_notional is None:
            raise ValueError("Must supply at least one of max_shares or max_notional")
        self.max_shares = max_shares
        self.max_notional = max_notional

    def validate(self, asset, amount, portfolio, current_price=0.0, current_time=None):
        current_shares = portfolio.positions.get(asset, 0)
        post_order_shares = current_shares + amount

        if self.max_shares is not None and abs(post_order_shares) > self.max_shares:
            return self._fail(asset, amount,
                f"MaxPositionSize: {abs(post_order_shares)} shares > max {self.max_shares}")

        post_order_notional = abs(post_order_shares) * current_price
        if self.max_notional is not None and post_order_notional > self.max_notional:
            return self._fail(asset, amount,
                f"MaxPositionSize: ${post_order_notional:.2f} notional > max ${self.max_notional:.2f}")

        return True


class LongOnly(TradingControl):
    """Prohibit short positions (position cannot go negative)."""

    def __init__(self, on_error: str = "fail"):
        super().__init__(on_error)

    def validate(self, asset, amount, portfolio, current_price=0.0, current_time=None):
        current_shares = portfolio.positions.get(asset, 0)
        post_order_shares = current_shares + amount

        if post_order_shares < 0:
            return self._fail(asset, amount,
                f"LongOnly: position would be {post_order_shares} (negative)")

        return True


class TradingControls:
    """
    Container for multiple trading controls. Validates every order against
    all registered controls.

    Usage:
        controls = TradingControls()
        controls.add(MaxOrderCount(max_count=10))
        controls.add(MaxPositionSize(max_notional=50000))
        controls.add(LongOnly())

        # Before each order:
        if controls.validate("EURUSD", 1000, portfolio, current_price=1.085):
            # Safe to place order
            broker.place_order(...)
    """

    def __init__(self):
        self._controls: list[TradingControl] = []

    def add(self, control: TradingControl) -> None:
        self._controls.append(control)
        log.info(f"Registered trading control: {type(control).__name__}")

    def validate(
        self,
        asset: str,
        amount: float,
        portfolio: PortfolioState,
        current_price: float = 0.0,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """
        Validate order against all controls. Returns True if all pass.
        Raises TradingControlViolation if any control fails (on_error="fail").
        """
        for control in self._controls:
            control.validate(asset, amount, portfolio, current_price, current_time)
        return True

    def list_controls(self) -> list[str]:
        return [type(c).__name__ for c in self._controls]


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    controls = TradingControls()
    controls.add(MaxOrderCount(max_count=3))
    controls.add(MaxOrderSize(max_notional=50000))
    controls.add(MaxPositionSize(max_shares=1000))
    controls.add(LongOnly())

    portfolio = PortfolioState(positions={"EURUSD": 500}, cash=100000)

    # Valid order
    assert controls.validate("EURUSD", 200, portfolio, current_price=1.085)
    print("✓ Valid order passed")

    # Order size too large (notional)
    try:
        controls.validate("EURUSD", 100000, portfolio, current_price=1.085)
        assert False, "should have raised"
    except TradingControlViolation as e:
        print(f"✓ MaxOrderSize blocked: {e.constraint}")

    # Position size too large
    try:
        controls.validate("EURUSD", 600, portfolio, current_price=1.085)
        assert False, "should have raised"
    except TradingControlViolation as e:
        print(f"✓ MaxPositionSize blocked: {e.constraint}")

    # Short position blocked
    try:
        controls.validate("EURUSD", -600, portfolio, current_price=1.085)
        assert False, "should have raised"
    except TradingControlViolation as e:
        print(f"✓ LongOnly blocked: {e.constraint}")

    # Max order count (3 per day) — use a separate controls instance
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    count_controls = TradingControls()
    count_controls.add(MaxOrderCount(max_count=3))
    for i in range(3):
        count_controls.validate("GBPUSD", 10, PortfolioState(),
                                current_price=1.25, current_time=now)
    try:
        count_controls.validate("GBPUSD", 10, PortfolioState(),
                                current_price=1.25, current_time=now)
        assert False, "should have raised"
    except TradingControlViolation as e:
        print(f"✓ MaxOrderCount blocked: {e.constraint}")

    # Log mode (no exception)
    log_controls = TradingControls()
    log_controls.add(LongOnly(on_error="log"))
    result = log_controls.validate("EURUSD", -100, portfolio, current_price=1.085)
    print(f"✓ Log mode returned: {result}")

    print(f"\nRegistered controls: {controls.list_controls()}")
    print("\nTrading controls smoke test passed.")
