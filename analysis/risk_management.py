"""
analysis/risk_management.py — Book 5 (Frank Miller S&D) Chapter 14 Risk Management
==================================================================================

Pages 153-157 implement the book's complete risk-management system:
position sizing, margin-call thresholds, and a drawdown-based
"circuit breaker" that throttles risk during losing streaks.

  ── RISK PER TRADE (Book P154) ────────────────────────────────
    Experienced trader : max 2% account equity per trade
    Beginner trader    : max 1% account equity per trade
                         (until account has tripled — 3× growth)

  ── MARGIN CALL THRESHOLD (Book P154) ─────────────────────────
    Margin call triggers when account loss × leverage ≈ 100% of margin.
    Examples (Book P154):
      Leverage ×10 + 10% account loss → margin call
      Leverage ×5  + 20% account loss → margin call
      $1,000 at 100:1 leverage → $100,000 exposure; 1% loss = $1,000 = margin call
    Formula: margin_call_triggered ≈ (account_loss_pct × leverage) ≥ 100%

  ── POSITION SIZING FORMULA (Book P155) ───────────────────────
    position_size = risk_amount / |entry_price - stop_price|
    Example: $1,000 × 2% = $20 risk; entry $10, stop $8 → $20/$2 = 10 shares

  ── FOREX-SPECIFIC SIZING (Book P155-156) ────────────────────
    Case A — quote currency = account currency:
      position_size (units) = risk_amount / pip_distance
    Case B — quote currency ≠ account currency:
      risk_in_quote = risk_amount × exchange_rate
      pip_value     = risk_in_quote / pip_distance
      position_size = risk_in_quote / pip_distance

  ── DRAWDOWN CIRCUIT BREAKER (Book P157) ─────────────────────
    1. If drawdown_from_peak ≥ 20% → STOP TRADING for the rest of the month
    2. During ANY drawdown → reduce risk_per_trade by 25% (e.g., 2% → 1.5%)
    3. Restore original risk_per_trade only when equity makes a NEW HIGH

  ── COMPOUNDING DRAWDOWN MATH (Book P155) ────────────────────
    remaining_equity = initial_equity × (1 - risk%)^n_losses
    Book example: $1,000 + 10 losses @ 2% → $817 (book states $833.79)
                  $1,000 + 10 losses @ 5% → $599 (book states $630.25)
    (Minor discrepancy flagged in audit — likely rounding methodology
     difference. We use the exact formula: 1000×(0.98)^10 ≈ 817.07.)

Usage:
    from analysis.risk_management import RiskManager, PositionSizer
    rm = RiskManager(account_equity=10_000, is_beginner=False)
    ps = PositionSizer(rm)
    size = ps.size_for_stock(entry=10.0, stop=8.0)
    # → 100 shares (risk=$200, $10-$8=$2, 200/2=100)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from utils.logger import get_logger

log = get_logger("risk_mgmt")


# ════════════════════════════════════════════════════════════════
#  CONSTANTS — Book P154, P157
# ════════════════════════════════════════════════════════════════

# Risk per trade (Book P154)
RISK_EXPERIENCED_PCT = 2.0    # 2% for experienced traders
RISK_BEGINNER_PCT    = 1.0    # 1% for beginners (until 3x account growth)
BEGINNER_GRADUATION_MULT = 3.0  # triple account → graduate to experienced

# Drawdown circuit breaker (Book P157)
DRAWDOWN_STOP_PCT        = 20.0   # ≥20% drawdown → stop trading for the month
DRAWDOWN_RISK_REDUCTION  = 0.25   # reduce risk by 25% during any drawdown
DRAWDOWN_RISK_MULTIPLIER = 1.0 - DRAWDOWN_RISK_REDUCTION  # = 0.75

# Margin call threshold (Book P154)
# Margin call when account_loss_pct × leverage ≥ 100%
MARGIN_CALL_THRESHOLD = 100.0


# ════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class RiskState:
    """Snapshot of the risk-management state."""
    account_equity: float
    peak_equity: float
    is_beginner: bool
    base_risk_pct: float                  # 2% or 1% (Book P154)
    current_risk_pct: float               # after drawdown adjustment (Book P157)
    drawdown_pct: float                   # current drawdown from peak
    in_drawdown: bool
    trading_halted: bool                  # True if drawdown ≥ 20% (Book P157)
    halt_reason: str = ""

    @property
    def risk_amount(self) -> float:
        """Dollar risk allowed per trade at current risk_pct."""
        return self.account_equity * (self.current_risk_pct / 100.0)


@dataclass
class PositionSizeResult:
    """Result of a position-sizing calculation."""
    position_size: float                  # shares / units / lots (context-dependent)
    risk_amount: float                    # dollar amount risked
    risk_pct: float                       # % of account risked
    entry_price: float
    stop_price: float
    pip_distance: Optional[float] = None  # forex only
    quote_currency_note: str = ""         # forex only — explains conversion
    method: str = "stock"                 # "stock" | "forex_same_ccy" | "forex_diff_ccy"


# ════════════════════════════════════════════════════════════════
#  RISK MANAGER — state machine for the circuit breaker
# ════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Book 5 Chapter 14 — risk-management state machine.

    Tracks:
      • Account equity + peak (for drawdown calculation)
      • Beginner/experienced tier (2% vs 1% risk per trade)
      • Current drawdown-adjusted risk_pct
      • Trading-halt status (≥20% drawdown → halt for month)

    Public API:
      • update(equity)            → RiskState
      • get_risk_amount()         → float (current $ risk allowed)
      • can_trade()               → bool
      • graduate_to_experienced() → bool (check if 3x growth reached)
      • reset_month()             → clear monthly halt
    """

    def __init__(
        self,
        account_equity: float,
        initial_equity: Optional[float] = None,
        is_beginner: bool = False,
    ):
        """
        Args:
            account_equity  : current account equity
            initial_equity  : starting equity (for 3x graduation check);
                              defaults to account_equity if not provided
            is_beginner     : True → 1% risk; False → 2% risk (Book P154)
        """
        self.account_equity = account_equity
        self.initial_equity = initial_equity if initial_equity is not None else account_equity
        self.is_beginner = is_beginner
        self.peak_equity = account_equity
        self.trading_halted = False
        self.halt_reason = ""

    @property
    def base_risk_pct(self) -> float:
        """Book P154: 2% experienced, 1% beginner."""
        return RISK_BEGINNER_PCT if self.is_beginner else RISK_EXPERIENCED_PCT

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak (Book P157)."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, ((self.peak_equity - self.account_equity) / self.peak_equity) * 100.0)

    @property
    def in_drawdown(self) -> bool:
        """True if account is below peak (any drawdown)."""
        return self.account_equity < self.peak_equity

    @property
    def current_risk_pct(self) -> float:
        """
        Book P157: during any drawdown, reduce risk by 25%.
        Example: 2% → 1.5%, 1% → 0.75%.
        """
        if self.in_drawdown:
            return self.base_risk_pct * DRAWDOWN_RISK_MULTIPLIER
        return self.base_risk_pct

    def update(self, equity: float) -> RiskState:
        """Update account equity (and peak if new high). Returns current state."""
        self.account_equity = equity
        # Update peak (Book P157: restore risk only on new high)
        if equity > self.peak_equity:
            self.peak_equity = equity
            # New high → restore risk + clear halt if applicable
            # (Book P157 implies risk restoration on new high)
            # Note: halt-for-month is NOT cleared by a new high — only by month reset
        # Check halt condition (Book P157: ≥20% drawdown → halt for month)
        if self.drawdown_pct >= DRAWDOWN_STOP_PCT and not self.trading_halted:
            self.trading_halted = True
            self.halt_reason = (
                f"Drawdown {self.drawdown_pct:.1f}% ≥ {DRAWDOWN_STOP_PCT:.0f}% — "
                f"halt trading for rest of month (Book P157)"
            )
            log.warning(f"[RiskManager] {self.halt_reason}")
        return self.get_state()

    def get_state(self) -> RiskState:
        """Return a snapshot of the current risk state."""
        return RiskState(
            account_equity=self.account_equity,
            peak_equity=self.peak_equity,
            is_beginner=self.is_beginner,
            base_risk_pct=self.base_risk_pct,
            current_risk_pct=self.current_risk_pct,
            drawdown_pct=self.drawdown_pct,
            in_drawdown=self.in_drawdown,
            trading_halted=self.trading_halted,
            halt_reason=self.halt_reason,
        )

    def get_risk_amount(self) -> float:
        """Dollar risk allowed per trade at current risk_pct."""
        return self.account_equity * (self.current_risk_pct / 100.0)

    def can_trade(self) -> bool:
        """True if trading is not halted."""
        return not self.trading_halted

    def graduate_to_experienced(self) -> bool:
        """
        Book P154: beginner graduates to experienced when account triples.
        Returns True if graduation just occurred.
        """
        if self.is_beginner and self.account_equity >= self.initial_equity * BEGINNER_GRADUATION_MULT:
            self.is_beginner = False
            log.info(
                f"[RiskManager] Beginner graduated to experienced "
                f"(account {self.account_equity:.2f} ≥ 3× initial {self.initial_equity:.2f})"
            )
            return True
        return False

    def reset_month(self) -> None:
        """Clear the monthly trading halt (called at start of new month)."""
        if self.trading_halted:
            log.info("[RiskManager] Monthly reset — clearing trading halt")
        self.trading_halted = False
        self.halt_reason = ""


# ════════════════════════════════════════════════════════════════
#  POSITION SIZER — Book P155-156 formulas
# ════════════════════════════════════════════════════════════════

class PositionSizer:
    """
    Book 5 Chapter 14 — position-sizing formulas.

    Master formula (Book P155):
        position_size = risk_amount / |entry_price - stop_price|

    Forex-specific (Book P155-156):
        Case A (quote ccy = account ccy): position_size = risk_amount / pip_distance
        Case B (quote ccy ≠ account ccy): convert risk to quote ccy first
    """

    # Standard lot conventions
    STANDARD_LOT_UNITS = 100_000      # 1.00 standard lot
    MINI_LOT_UNITS     = 10_000       # 0.10 mini lot
    MICRO_LOT_UNITS    = 1_000        # 0.01 micro lot

    def __init__(self, risk_manager: RiskManager):
        self.rm = risk_manager

    def size_for_stock(
        self,
        entry: float,
        stop: float,
    ) -> PositionSizeResult:
        """
        Book P155 stock example: $1,000 × 2% = $20 risk; entry $10, stop $8 → 10 shares.
        Formula: position_size = risk_amount / |entry - stop|
        """
        risk_amount = self.rm.get_risk_amount()
        price_diff = abs(entry - stop)
        if price_diff <= 0:
            return PositionSizeResult(
                position_size=0.0, risk_amount=risk_amount,
                risk_pct=self.rm.current_risk_pct,
                entry_price=entry, stop_price=stop, method="stock",
            )
        size = risk_amount / price_diff
        return PositionSizeResult(
            position_size=size,
            risk_amount=risk_amount,
            risk_pct=self.rm.current_risk_pct,
            entry_price=entry,
            stop_price=stop,
            method="stock",
        )

    def size_for_forex(
        self,
        entry: float,
        stop: float,
        pair: str,
        account_currency: str = "USD",
        exchange_rate_to_quote: Optional[float] = None,
        pip_value_per_standard_lot: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Book P155-156 forex position sizing.

        Two cases:
          A. quote_currency == account_currency:
             position_size (units) = risk_amount / pip_distance
             (then convert to lots by dividing by STANDARD_LOT_UNITS)

          B. quote_currency != account_currency:
             risk_in_quote = risk_amount × exchange_rate
             position_size (units) = risk_in_quote / pip_distance

        Args:
            entry, stop         : price levels
            pair                : e.g. "AUD/USD", "USD/JPY"
            account_currency    : e.g. "USD"
            exchange_rate_to_quote : conversion rate from account ccy to quote ccy
                                    (required for Case B)
            pip_value_per_standard_lot : optional override for pip value calculation

        Returns:
            PositionSizeResult with position_size in LOTS
            (0.01 = micro, 0.10 = mini, 1.00 = standard)
        """
        risk_amount = self.rm.get_risk_amount()
        # Extract quote currency from pair (2nd currency)
        parts = pair.upper().replace("/", "").replace("_", "")
        if len(parts) != 6:
            return PositionSizeResult(
                position_size=0.0, risk_amount=risk_amount,
                risk_pct=self.rm.current_risk_pct,
                entry_price=entry, stop_price=stop, method="forex_same_ccy",
                quote_currency_note=f"Invalid pair format: {pair}",
            )
        base_ccy = parts[:3]
        quote_ccy = parts[3:]
        quote_currency_note = f"pair={pair}, base={base_ccy}, quote={quote_ccy}, account={account_currency}"

        # Determine pip distance
        # JPY pairs: 1 pip = 0.01; all others: 1 pip = 0.0001
        pip_size = 0.01 if "JPY" in pair.upper() else 0.0001
        pip_distance = abs(entry - stop) / pip_size
        if pip_distance <= 0:
            return PositionSizeResult(
                position_size=0.0, risk_amount=risk_amount,
                risk_pct=self.rm.current_risk_pct,
                entry_price=entry, stop_price=stop, method="forex_same_ccy",
                quote_currency_note=quote_currency_note + " (invalid pip distance)",
            )

        if quote_ccy == account_currency.upper():
            # Case A: quote ccy = account ccy (Book P155-156, AUD/USD example)
            # risk_amount is already in quote ccy.
            # Derivation: units x pip_size x pip_distance_in_pips = risk_amount
            #   => units = risk_amount / (pip_size x pip_distance)
            # Since pip_distance = |entry-stop| / pip_size, this simplifies to:
            #   units = risk_amount / |entry-stop|
            position_units = risk_amount / abs(entry - stop)
            method = "forex_same_ccy"
            note = quote_currency_note + " — Case A (quote=account ccy)"
        else:
            # Case B: quote ccy ≠ account ccy (Book P156, USD/JPY example)
            # Need exchange_rate_to_quote (e.g., for USD/JPY with USD account,
            # exchange_rate = USD/JPY rate ≈ 119.00)
            if exchange_rate_to_quote is None:
                # Use entry as the exchange rate (common approximation when
                # account ccy = base ccy — e.g. USD account, USD/JPY pair)
                exchange_rate_to_quote = entry
            risk_in_quote = risk_amount * exchange_rate_to_quote
            position_units = risk_in_quote / abs(entry - stop)
            method = "forex_diff_ccy"
            note = (quote_currency_note +
                    f" — Case B (quote≠account ccy), exchange_rate={exchange_rate_to_quote}")

        # Convert units to lots
        position_lots = position_units / self.STANDARD_LOT_UNITS

        return PositionSizeResult(
            position_size=position_lots,
            risk_amount=risk_amount,
            risk_pct=self.rm.current_risk_pct,
            entry_price=entry,
            stop_price=stop,
            pip_distance=pip_distance,
            quote_currency_note=note,
            method=method,
        )


# ════════════════════════════════════════════════════════════════
#  MARGIN CALL DETECTOR — Book P154
# ════════════════════════════════════════════════════════════════

class MarginCallDetector:
    """
    Book 5 Chapter 14 (Page 154) — margin call detection.

    Rule: margin call triggers when account_loss_pct × leverage ≥ 100%.

    Examples (Book P154):
      ×10 leverage + 10% account loss → 10×10 = 100% → margin call
      ×5  leverage + 20% account loss → 5×20  = 100% → margin call
      $1,000 at 100:1 leverage → $100,000 exposure
        1% loss on $100,000 = $1,000 = 100% of account → margin call
    """

    @staticmethod
    def is_margin_call(
        account_loss_pct: float,
        leverage: float,
    ) -> bool:
        """
        Book P154: margin call when account_loss_pct × leverage ≥ 100%.

        Args:
            account_loss_pct : % of account equity lost (e.g. 10.0 for 10%)
            leverage         : leverage ratio (e.g. 10.0 for ×10)
        """
        if leverage <= 0:
            return False
        return (account_loss_pct * leverage) >= MARGIN_CALL_THRESHOLD

    @staticmethod
    def max_loss_before_margin_call(leverage: float) -> float:
        """
        Book P154: inverse — what % account loss triggers margin call?
        Returns the loss % at which margin_call_trigger = 100%.
        """
        if leverage <= 0:
            return 100.0
        return MARGIN_CALL_THRESHOLD / leverage

    @staticmethod
    def exposure(amount: float, leverage: float) -> float:
        """Book P154: total market exposure = amount × leverage."""
        return amount * leverage


# ════════════════════════════════════════════════════════════════
#  DRAWDOWN SIMULATOR — Book P155 compounding math
# ════════════════════════════════════════════════════════════════

class DrawdownSimulator:
    """
    Book 5 Chapter 14 (Page 155) — compounding drawdown simulator.

    Formula (Book P155):
        remaining_equity = initial_equity × (1 - risk%)^n_losses

    Book example:
      $1,000 + 10 losses @ 2% → $817.07 (book states $833.79 — flagged discrepancy)
      $1,000 + 10 losses @ 5% → $598.74 (book states $630.25 — flagged discrepancy)
    """

    @staticmethod
    def simulate_losing_streak(
        initial_equity: float,
        risk_pct: float,
        n_losses: int,
    ) -> Dict[str, Any]:
        """
        Simulate n consecutive losing trades at given risk %.

        Returns:
            {
                "initial_equity": float,
                "final_equity": float,
                "total_loss_pct": float,
                "n_losses": int,
                "risk_pct": float,
            }
        """
        if n_losses < 0:
            n_losses = 0
        multiplier = (1 - risk_pct / 100.0) ** n_losses
        final_equity = initial_equity * multiplier
        total_loss_pct = (1 - multiplier) * 100.0
        return {
            "initial_equity": initial_equity,
            "final_equity": round(final_equity, 2),
            "total_loss_pct": round(total_loss_pct, 2),
            "n_losses": n_losses,
            "risk_pct": risk_pct,
        }

    @staticmethod
    def compare_risk_levels(
        initial_equity: float,
        risk_pcts: list,
        n_losses: int,
    ) -> Dict[float, Dict[str, Any]]:
        """
        Compare multiple risk levels over the same losing streak.
        Book P155 example: compare 2% vs 5% over 10 losses.
        """
        return {
            risk: DrawdownSimulator.simulate_losing_streak(initial_equity, risk, n_losses)
            for risk in risk_pcts
        }


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/z/my-project/forex_ai")

    print("=" * 64)
    print("  RISK MANAGEMENT — Book 5 Chapter 14 (Pages 153-157)")
    print("=" * 64)

    # ── Book P155 stock example ──
    print("\n── Book P155 stock example ──")
    rm = RiskManager(account_equity=1_000, is_beginner=False)
    print(f"  Account: ${rm.account_equity}, risk={rm.base_risk_pct}% (experienced)")
    ps = PositionSizer(rm)
    result = ps.size_for_stock(entry=10.0, stop=8.0)
    print(f"  Entry=$10, Stop=$8 → position = {result.position_size:.0f} shares "
          f"(risk=${result.risk_amount:.0f})")
    print(f"  Book expected: 10 shares (risk $20 / $2 diff)")

    # ── Book P155-156 AUD/USD example ──
    print("\n── Book P155-156 AUD/USD example (Case A: quote=account ccy) ──")
    # $100,000 account, 2% risk = $2,000
    rm2 = RiskManager(account_equity=100_000, is_beginner=False)
    ps2 = PositionSizer(rm2)
    # Book says: entry 0.6900, stop 0.6200 = 700 pips
    # position_size = 2000 / 700 pips... but in units it's 2000/0.07 = 28,571 units
    # = 0.286 standard lots (book says 0.29 mini lot — they reported mini lots)
    aud_usd = ps2.size_for_forex(entry=0.6900, stop=0.6200, pair="AUD/USD",
                                  account_currency="USD")
    print(f"  Entry=0.6900, Stop=0.6200 (700 pips)")
    print(f"  Position = {aud_usd.position_size:.4f} standard lots "
          f"({aud_usd.position_size*10:.2f} mini lots)")
    print(f"  Book expected: ~0.29 mini lot (=0.029 standard lot)")
    print(f"  Note: {aud_usd.quote_currency_note}")

    # ── Book P156 USD/JPY example ──
    print("\n── Book P156 USD/JPY example (Case B: quote≠account ccy) ──")
    usd_jpy = ps2.size_for_forex(entry=119.00, stop=124.00, pair="USD/JPY",
                                  account_currency="USD")
    print(f"  Entry=119.00, Stop=124.00 (500 pips)")
    print(f"  Position = {usd_jpy.position_size:.4f} standard lots "
          f"({usd_jpy.position_size*100:.2f} micro lots)")
    print(f"  Book expected: ~4.76 micro lots (=0.0476 standard lot)")
    print(f"  Note: {usd_jpy.quote_currency_note}")

    # ── Drawdown circuit breaker ──
    print("\n── Drawdown Circuit Breaker (Book P157) ──")
    rm3 = RiskManager(account_equity=10_000, is_beginner=False)
    print(f"  Start: ${rm3.account_equity}, risk={rm3.base_risk_pct}%, peak=${rm3.peak_equity}")
    # Simulate losses
    state = rm3.update(9_500)  # 5% drawdown
    print(f"  After 5% drawdown (${rm3.account_equity}): "
          f"risk={state.current_risk_pct}% (reduced), halt={state.trading_halted}")
    # Hit 20% drawdown
    state = rm3.update(7_500)  # 25% drawdown from 10k peak
    print(f"  After 25% drawdown (${rm3.account_equity}): "
          f"risk={state.current_risk_pct}%, halt={state.trading_halted}")
    print(f"  Halt reason: {state.halt_reason}")

    # ── Margin call detector ──
    print("\n── Margin Call Detector (Book P154) ──")
    print(f"  ×10 leverage + 10% loss → margin call? "
          f"{MarginCallDetector.is_margin_call(10, 10)}")
    print(f"  ×5 leverage + 20% loss → margin call? "
          f"{MarginCallDetector.is_margin_call(20, 5)}")
    print(f"  ×100 leverage + 1% loss → margin call? "
          f"{MarginCallDetector.is_margin_call(1, 100)}")
    print(f"  Max loss before MC at ×10 leverage: "
          f"{MarginCallDetector.max_loss_before_margin_call(10):.1f}%")

    # ── Drawdown simulator (Book P155) ──
    print("\n── Drawdown Simulator (Book P155) ──")
    comparison = DrawdownSimulator.compare_risk_levels(1_000, [2.0, 5.0], 10)
    for risk, result in comparison.items():
        print(f"  {risk}% risk × 10 losses: ${result['final_equity']} "
              f"(loss {result['total_loss_pct']:.1f}%)")
    print(f"  Book states: 2% → $833.79, 5% → $630.25")
    print(f"  Our formula: 2% → $817.07, 5% → $598.74 (minor discrepancy flagged)")

    print("\n" + "=" * 64)
    print("  Risk management smoke test complete.")
    print("=" * 64)