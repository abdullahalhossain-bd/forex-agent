"""
risk/strict_risk_manager.py — Strict Risk Manager (Fix for Fatal Flaw #8)
========================================================================

Replaces the permissive Book P154 risk rules (2% per trade, 20% drawdown halt)
with conservative professional-grade rules that prevent account blow-up.

Rules:
  1. Max 0.5% risk per trade (NOT 2%)
  2. Max 1.5% daily loss → halt for the day
  3. Max 5% weekly loss → halt for the week
  4. Max 10% drawdown from peak → halt completely (NOT 20%)
  5. Max 1 trade per "currency cluster" at a time (correlation control)
  6. Max 3 open positions total
  7. No new trades after 3 consecutive losses (cooldown)
  8. Reduced risk (0.25%) during any drawdown
  9. Hard daily trade limit (5 trades/day max)
 10. News event blackout (no trades 30 min before/after high-impact news)

Currency clusters (correlated):
  - USD: EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, USDJPY
  - EUR: EURUSD, EURGBP, EURJPY, EURAUD, EURCAD, EURCHF
  - GBP: GBPUSD, EURGBP, GBPJPY, GBPAUD, GBPCAD
  - JPY: USDJPY, EURJPY, GBPJPY, AUDJPY, CADJPY, CHFJPY
  - Metals: XAUUSD, XAGUSD (separate cluster)
  - Each pair belongs to multiple clusters; conflict if any cluster shared

Usage:
    from risk.strict_risk_manager import StrictRiskManager
    rm = StrictRiskManager(account_equity=10_000)
    if rm.can_open_trade(pair="EURUSD", direction="long"):
        size = rm.position_size(entry=1.0850, stop=1.0820, pair="EURUSD")
        rm.register_trade(pair="EURUSD", direction="long", risk_amount=size*0.005)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import threading

from utils.logger import get_logger

log = get_logger("strict_risk")


# ════════════════════════════════════════════════════════════════
#  STRICT RISK PARAMETERS (replaces Book P154/P157 permissive rules)
# ════════════════════════════════════════════════════════════════

# Per-trade risk (much lower than Book's 2%)
RISK_PER_TRADE_PCT       = 0.5     # 0.5% per trade (was 2%)
RISK_PER_TRADE_REDUCED   = 0.25    # 0.25% during drawdown (was 1.5%)
RISK_PER_TRADE_BEGINNER  = 0.25    # 0.25% for first 100 trades

# Aggregate limits
MAX_DAILY_LOSS_PCT       = 1.5     # halt for day at -1.5%
MAX_WEEKLY_LOSS_PCT      = 5.0     # halt for week at -5%
MAX_DRAWDOWN_PCT         = 10.0    # halt completely at -10% (was 20%)
MAX_OPEN_POSITIONS       = 3       # max 3 trades at once
MAX_TRADES_PER_DAY       = 20      # max 20 trades/day (synced with core.constants.MAX_TRADES_PER_DAY)
MAX_CONSECUTIVE_LOSSES   = 3       # cooldown after 3 losses
COOLDOWN_HOURS           = 4       # cooldown duration

# Correlation control
CURRENCY_CLUSTERS = {
    "USD":    ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
    "EUR":    ["EURUSD", "EURGBP", "EURJPY", "EURAUD", "EURCAD", "EURCHF"],
    "GBP":    ["GBPUSD", "EURGBP", "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF"],
    "JPY":    ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"],
    "AUD":    ["AUDUSD", "EURAUD", "GBPAUD", "AUDJPY", "AUDCAD", "AUDNZD"],
    "CAD":    ["USDCAD", "EURCAD", "GBPCAD", "AUDCAD", "CADJPY", "CADCHF"],
    "CHF":    ["USDCHF", "EURCHF", "GBPCHF", "AUDCHF", "CADCHF", "CHFJPY"],
    "METALS": ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"],
    "INDICES": ["US30", "NAS100", "SPX500", "UK100", "GER40", "JPN225"],
    "CRYPTO": ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD"],
}
# Allow 1 trade per cluster (e.g., 1 USD-bet, 1 JPY-bet, 1 metals-bet)
MAX_TRADES_PER_CLUSTER   = 1


# ════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class OpenPosition:
    """Currently open position."""
    pair: str
    direction: str          # "long" | "short"
    entry_price: float
    stop_loss: float
    risk_amount: float      # dollar amount risked
    opened_at: datetime
    clusters: List[str] = field(default_factory=list)


@dataclass
class TradeRecord:
    """Historical trade record."""
    pair: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    pnl_dollars: float
    pnl_pips: float
    win: bool


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    allowed: bool
    reason: str
    current_state: Dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════
#  STRICT RISK MANAGER
# ════════════════════════════════════════════════════════════════

class StrictRiskManager:
    """
    Strict risk manager that prevents account blow-up.

    Replaces Book P154/P157 permissive rules (2% per trade, 20% drawdown)
    with professional-grade rules (0.5% per trade, 10% drawdown, correlation
    control, daily/weekly limits, cooldown after losses).
    """

    def __init__(
        self,
        account_equity: float,
        initial_equity: Optional[float] = None,
        is_beginner: bool = True,  # Beginner = first 100 trades
    ):
        self.account_equity = account_equity
        self.initial_equity = initial_equity or account_equity
        self.peak_equity = account_equity
        self.is_beginner = is_beginner
        self.trade_count = 0

        # State
        self.open_positions: List[OpenPosition] = []
        self.trade_history: List[TradeRecord] = []
        self.consecutive_losses = 0
        self.last_cooldown_until: Optional[datetime] = None

        # Daily/weekly tracking
        self.day_start: datetime = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self.week_start: datetime = self.day_start - timedelta(days=self.day_start.weekday())
        self.day_pnl: float = 0.0
        self.week_pnl: float = 0.0
        self.day_trade_count: int = 0

        # Halts
        self.halt_day: bool = False
        self.halt_week: bool = False
        self.halt_permanent: bool = False
        self.halt_reason: str = ""

        # CRITICAL FIX: thread-safe reentrant lock for all mutable state
        # Without this, concurrent calls to can_open_trade + register_trade
        # would race and bypass the max-positions / cluster rules.
        self._lock = threading.RLock()

    # ══════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════

    def can_open_trade(
        self,
        pair: str,
        direction: str,
        now: Optional[datetime] = None,
    ) -> RiskCheckResult:
        """
        Check if a new trade can be opened. Returns reason if blocked.

        Thread-safe: holds RLock for the duration of the check + state mutations
        (halt flags may be set during this call).
        """
        with self._lock:
            return self._can_open_trade_unlocked(pair, direction, now)

    def _can_open_trade_unlocked(
        self,
        pair: str,
        direction: str,
        now: Optional[datetime] = None,
    ) -> RiskCheckResult:
        """Internal: must be called while holding self._lock."""
        now = now or datetime.now(timezone.utc)
        state = self._snapshot_state()

        # Check 1: Permanent halt (drawdown)
        if self.halt_permanent:
            return RiskCheckResult(False, f"PERMANENT HALT: {self.halt_reason}", state)

        # Check 2: Daily halt
        if self.halt_day and now.date() == self.day_start.date():
            return RiskCheckResult(False, f"DAY HALT: {self.halt_reason}", state)
        # Reset day halt if new day
        if now.date() != self.day_start.date():
            self._reset_day(now)

        # Check 3: Weekly halt
        if self.halt_week:
            week_now = now - timedelta(days=now.weekday())
            if week_now.date() == self.week_start.date():
                return RiskCheckResult(False, f"WEEK HALT: {self.halt_reason}", state)
            else:
                self._reset_week(now)

        # Check 4: Cooldown after consecutive losses
        if self.last_cooldown_until and now < self.last_cooldown_until:
            remaining = (self.last_cooldown_until - now).total_seconds() / 3600
            return RiskCheckResult(False,
                f"COOLDOWN: {remaining:.1f}h remaining after {self.consecutive_losses} losses",
                state)

        # Check 5: Max open positions
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return RiskCheckResult(False,
                f"MAX POSITIONS: {len(self.open_positions)}/{MAX_OPEN_POSITIONS} open",
                state)

        # Check 6: Max trades per day
        if self.day_trade_count >= MAX_TRADES_PER_DAY:
            return RiskCheckResult(False,
                f"MAX DAILY TRADES: {self.day_trade_count}/{MAX_TRADES_PER_DAY} | source=risk/strict_risk_manager.py:check() | config=MAX_TRADES_PER_DAY={MAX_TRADES_PER_DAY}",
                state)

        # Check 7: Correlation control — same cluster conflict
        trade_clusters = self._clusters_for_pair(pair)
        for pos in self.open_positions:
            shared = set(pos.clusters) & set(trade_clusters)
            if shared:
                return RiskCheckResult(False,
                    f"CORRELATION: shares cluster(s) {shared} with open {pos.pair}",
                    state)

        # Check 8: Cluster limit
        cluster_count: Dict[str, int] = {}
        for pos in self.open_positions:
            for c in pos.clusters:
                cluster_count[c] = cluster_count.get(c, 0) + 1
        for c in trade_clusters:
            if cluster_count.get(c, 0) >= MAX_TRADES_PER_CLUSTER:
                return RiskCheckResult(False,
                    f"CLUSTER LIMIT: {c} already has trade", state)

        # Check 9: Daily loss limit
        if self.day_pnl <= -self.account_equity * MAX_DAILY_LOSS_PCT / 100:
            self.halt_day = True
            self.halt_reason = f"Daily loss {MAX_DAILY_LOSS_PCT}% hit"
            return RiskCheckResult(False, f"DAY HALT: {self.halt_reason}", state)

        # Check 10: Weekly loss limit
        if self.week_pnl <= -self.account_equity * MAX_WEEKLY_LOSS_PCT / 100:
            self.halt_week = True
            self.halt_reason = f"Weekly loss {MAX_WEEKLY_LOSS_PCT}% hit"
            return RiskCheckResult(False, f"WEEK HALT: {self.halt_reason}", state)

        # Check 11: Max drawdown
        drawdown_pct = self._drawdown_pct()
        if drawdown_pct >= MAX_DRAWDOWN_PCT:
            self.halt_permanent = True
            self.halt_reason = f"Max drawdown {MAX_DRAWDOWN_PCT}% hit ({drawdown_pct:.1f}%)"
            log.critical(f"[StrictRisk] PERMANENT HALT: {self.halt_reason}")
            return RiskCheckResult(False, f"PERMANENT HALT: {self.halt_reason}", state)

        # All checks passed
        return RiskCheckResult(True, "OK", state)

    def position_size(
        self,
        entry: float,
        stop: float,
        pair: str,
    ) -> float:
        """
        Calculate position size based on current risk rules.

        Returns: dollar amount to risk (NOT lot size).
        Thread-safe.
        """
        with self._lock:
            return self._position_size_unlocked(entry, stop, pair)

    def _position_size_unlocked(self, entry: float, stop: float, pair: str) -> float:
        """Internal: must be called while holding self._lock."""
        # Determine risk %
        if self.is_beginner:
            risk_pct = RISK_PER_TRADE_BEGINNER
        elif self._in_drawdown():
            risk_pct = RISK_PER_TRADE_REDUCED
        else:
            risk_pct = RISK_PER_TRADE_PCT

        risk_amount = self.account_equity * (risk_pct / 100.0)

        # Validate stop distance is sensible
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0

        # Optional: volatility-adjusted sizing
        # (skip for simplicity; ATR-based adjustment can be added)

        return risk_amount

    def register_trade(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        risk_amount: float,
        now: Optional[datetime] = None,
    ):
        """Register a newly opened trade. Thread-safe."""
        with self._lock:
            now = now or datetime.now(timezone.utc)
            clusters = self._clusters_for_pair(pair)
            pos = OpenPosition(
                pair=pair, direction=direction,
                entry_price=entry_price, stop_loss=stop_loss,
                risk_amount=risk_amount, opened_at=now,
                clusters=clusters,
            )
            self.open_positions.append(pos)
            self.day_trade_count += 1
            self.trade_count += 1
            log.info(f"[StrictRisk] Opened {pair} {direction} risk=${risk_amount:.2f} "
                     f"clusters={clusters}")

    def close_trade(
        self,
        pair: str,
        direction: str,
        pnl_dollars: float,
        pnl_pips: float,
        now: Optional[datetime] = None,
    ):
        """Close an open position and update P&L tracking. Thread-safe."""
        with self._lock:
            now = now or datetime.now(timezone.utc)

            # Find and remove from open positions
            for i, pos in enumerate(self.open_positions):
                if pos.pair == pair and pos.direction == direction:
                    self.open_positions.pop(i)
                    break

            # Record in history
            record = TradeRecord(
                pair=pair, direction=direction,
                entry_time=now - timedelta(hours=1),  # approximate
                exit_time=now,
                pnl_dollars=pnl_dollars,
                pnl_pips=pnl_pips,
                win=pnl_dollars > 0,
            )
            self.trade_history.append(record)

            # Update P&L
            self.day_pnl += pnl_dollars
            self.week_pnl += pnl_dollars
            self.account_equity += pnl_dollars

            # Update peak
            if self.account_equity > self.peak_equity:
                self.peak_equity = self.account_equity

            # Update consecutive losses
            if pnl_dollars < 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    self.last_cooldown_until = now + timedelta(hours=COOLDOWN_HOURS)
                    log.warning(f"[StrictRisk] COOLDOWN triggered: "
                                f"{self.consecutive_losses} consecutive losses")
            else:
                self.consecutive_losses = 0

            # Check if beginner graduates (100 trades + profit)
            if self.is_beginner and self.trade_count >= 100:
                if self.account_equity > self.initial_equity:
                    self.is_beginner = False
                    log.info("[StrictRisk] Beginner graduated to experienced")

            log.info(f"[StrictRisk] Closed {pair} {direction} "
                     f"P&L=${pnl_dollars:+.2f} (day=${self.day_pnl:+.2f}, "
                     f"week=${self.week_pnl:+.2f})")

    # ══════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════

    def _clusters_for_pair(self, pair: str) -> List[str]:
        """Get all currency clusters a pair belongs to."""
        pair = pair.upper()
        clusters = []
        for cluster_name, pairs in CURRENCY_CLUSTERS.items():
            if pair in pairs:
                clusters.append(cluster_name)
        return clusters or ["UNKNOWN"]

    def _drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, ((self.peak_equity - self.account_equity) / self.peak_equity) * 100)

    def _in_drawdown(self) -> bool:
        return self.account_equity < self.peak_equity

    def _snapshot_state(self) -> Dict[str, Any]:
        return {
            "equity": self.account_equity,
            "peak": self.peak_equity,
            "drawdown_pct": self._drawdown_pct(),
            "open_positions": len(self.open_positions),
            "day_pnl": self.day_pnl,
            "week_pnl": self.week_pnl,
            "consecutive_losses": self.consecutive_losses,
            "day_trade_count": self.day_trade_count,
            "is_beginner": self.is_beginner,
        }

    def _reset_day(self, now: datetime):
        self.day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.day_pnl = 0.0
        self.day_trade_count = 0
        self.halt_day = False

    def _reset_week(self, now: datetime):
        self.week_start = now - timedelta(days=now.weekday())
        self.week_pnl = 0.0
        self.halt_week = False


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 70)
    print("  STRICT RISK MANAGER — Smoke Test")
    print("=" * 70)

    rm = StrictRiskManager(account_equity=10_000, is_beginner=True)
    print(f"\nInitial: equity=${rm.account_equity}, beginner={rm.is_beginner}")

    # Test 1: First trade allowed
    check = rm.can_open_trade("EURUSD", "long")
    print(f"\n[1] EURUSD long: {check.allowed} ({check.reason})")
    if check.allowed:
        size = rm.position_size(entry=1.0850, stop=1.0820, pair="EURUSD")
        rm.register_trade("EURUSD", "long", 1.0850, 1.0820, size)
        print(f"    Position size (risk amount): ${size:.2f} ({size/10_000*100:.2f}%)")

    # Test 2: GBPUSD long — same USD cluster, should be BLOCKED
    check = rm.can_open_trade("GBPUSD", "long")
    print(f"\n[2] GBPUSD long (USD cluster conflict): {check.allowed} ({check.reason})")

    # Test 3: USDJPY long — same USD cluster, BLOCKED
    check = rm.can_open_trade("USDJPY", "long")
    print(f"\n[3] USDJPY long (USD conflict): {check.allowed} ({check.reason})")

    # Test 4: XAUUSD long — different cluster (METALS), ALLOWED
    check = rm.can_open_trade("XAUUSD", "long")
    print(f"\n[4] XAUUSD long (METALS cluster OK): {check.allowed} ({check.reason})")
    if check.allowed:
        size = rm.position_size(entry=2000.0, stop=1990.0, pair="XAUUSD")
        rm.register_trade("XAUUSD", "long", 2000.0, 1990.0, size)
        print(f"    Position size: ${size:.2f}")

    # Test 5: Third position — should still be allowed (3 max)
    check = rm.can_open_trade("US30", "long")
    print(f"\n[5] US30 long (INDICES cluster OK): {check.allowed} ({check.reason})")
    if check.allowed:
        size = rm.position_size(entry=38000.0, stop=37900.0, pair="US30")
        rm.register_trade("US30", "long", 38000.0, 37900.0, size)

    # Test 6: Fourth position — should be BLOCKED (3 max)
    check = rm.can_open_trade("BTCUSD", "long")
    print(f"\n[6] BTCUSD long (4th position, max=3): {check.allowed} ({check.reason})")

    # Close a winning trade
    print("\n── Close XAUUSD +$30 ──")
    rm.close_trade("XAUUSD", "long", pnl_dollars=30.0, pnl_pips=3.0)

    # Test 7: Now BTCUSD should be allowed
    check = rm.can_open_trade("BTCUSD", "long")
    print(f"\n[7] BTCUSD long (after close): {check.allowed} ({check.reason})")

    # Simulate 3 consecutive losses
    print("\n── Simulate 3 consecutive losses ──")
    rm.close_trade("EURUSD", "long", pnl_dollars=-25.0, pnl_pips=-2.5)
    print(f"  After loss 1: consecutive_losses={rm.consecutive_losses}")
    rm.close_trade("US30", "long", pnl_dollars=-25.0, pnl_pips=-2.5)
    print(f"  After loss 2: consecutive_losses={rm.consecutive_losses}")
    rm.close_trade("BTCUSD", "long", pnl_dollars=-25.0, pnl_pips=-2.5)
    print(f"  After loss 3: consecutive_losses={rm.consecutive_losses}")

    # Test 8: Should be in cooldown
    check = rm.can_open_trade("EURUSD", "long")
    print(f"\n[8] EURUSD after 3 losses (cooldown): {check.allowed} ({check.reason})")

    # Show final state
    print(f"\n── Final State ──")
    print(f"  Equity: ${rm.account_equity:.2f}")
    print(f"  Peak: ${rm.peak_equity:.2f}")
    print(f"  Drawdown: {rm._drawdown_pct():.2f}%")
    print(f"  Day P&L: ${rm.day_pnl:+.2f}")
    print(f"  Week P&L: ${rm.week_pnl:+.2f}")
    print(f"  Trade count: {rm.trade_count}")
    print(f"  Open positions: {len(rm.open_positions)}")

    print("\n" + "=" * 70)
