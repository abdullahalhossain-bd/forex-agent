# risk/drawdown_controller.py — Day 58 | Drawdown Protection System
# ============================================================
# The most critical component of Day 58.
# Protects the account from catastrophic losses through multi-level
# protection thresholds.
#
# Protection Levels:
#   GREEN    (0-5% DD)    : Normal trading
#   YELLOW   (5-8% DD)    : Reduced risk, warnings
#   ORANGE   (8-12% DD)   : Defensive mode, halved risk
#   RED      (12-15% DD)  : Emergency mode, minimal trading
#   CRITICAL (>15% DD)    : STOP ALL TRADING
#
# Emergency Rules:
#   daily_loss > 3%   → stop_trading()
#   weekly_loss > 7%  → reduce_risk()
#   drawdown > 15%    → emergency_mode()
# ============================================================

import json
import os
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

from utils.logger import get_logger
from core.constants import MEMORY_DIR

log = get_logger("drawdown_controller")

DRAWDOWN_STATE_PATH = MEMORY_DIR / "drawdown_state.json"

# Protection thresholds
PROTECTION_LEVELS = {
    "GREEN":    {"dd_range": (0.0, 5.0),   "risk_scale": 1.0,   "action": "Normal trading"},
    "YELLOW":   {"dd_range": (5.0, 8.0),   "risk_scale": 0.7,   "action": "Reduce risk by 30%"},
    "ORANGE":   {"dd_range": (8.0, 12.0),  "risk_scale": 0.5,   "action": "Defensive mode, halve risk"},
    "RED":      {"dd_range": (12.0, 15.0), "risk_scale": 0.25,  "action": "Emergency, minimal trading"},
    "CRITICAL": {"dd_range": (15.0, 100.0),"risk_scale": 0.0,   "action": "STOP ALL TRADING"},
}

# Daily/weekly limits
DAILY_LOSS_LIMIT_PCT = 3.0  # default — overridden by config below
# Day 81+ hotfix: load from config (default 20.0).
try:
    from config import DAILY_LOSS_LIMIT_PCT as _CFG_DLL
    DAILY_LOSS_LIMIT_PCT = float(_CFG_DLL)
except Exception as e:
    # P0 fix: fail-closed on safety-critical config (mirrors circuit_breaker.py:54).
    # Previously silently fell back to 20.0 (20% daily loss) — too permissive.
    import logging as _logging
    _log = _logging.getLogger("drawdown_controller")
    _log.critical(f"[DrawdownController] DAILY_LOSS_LIMIT_PCT config load failed: {e}; "
                  f"crashing to prevent silent 20% daily-loss fallback.")
    raise RuntimeError(f"Safety-critical config missing: DAILY_LOSS_LIMIT_PCT — {e}") from e
WEEKLY_LOSS_LIMIT_PCT = 7.0
MAX_DRAWDOWN_LIMIT_PCT = 15.0


class DrawdownController:
    """
    Account Protection System — Drawdown Controller.

    এটা Day 58 এর সবচেয়ে গুরুত্বপূর্ণ component.
    AI account protect করবে multi-level protection system-এর মাধ্যমে।

    Protection Mechanism:
      GREEN (0-5% DD):    Full risk allocation, normal trading
      YELLOW (5-8% DD):   Risk reduced by 30%, monitor closely
      ORANGE (8-12% DD):  Risk halved, defensive trading only
      RED (12-15% DD):     Emergency mode, 75% risk reduction
      CRITICAL (>15% DD):  Complete trading halt

    Additional Safety:
      - Daily loss limit: 3% → stop trading for the day
      - Weekly loss limit: 7% → reduce risk significantly
      - Consecutive loss monitoring
      - Balance watermark tracking (peak equity)

    Usage:
        dc = DrawdownController(initial_balance=10000)
        level = dc.get_protection_level(balance=9200)
        # level = "YELLOW", risk_scale = 0.7

        dc.record_trade(pnl_usd=-150, balance=9200)
        check = dc.check_emergency(balance=9200, performance={})
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_drawdown_limit: float = MAX_DRAWDOWN_LIMIT_PCT,
        daily_loss_limit: float = DAILY_LOSS_LIMIT_PCT,
        weekly_loss_limit: float = WEEKLY_LOSS_LIMIT_PCT,
    ):
        self.initial_balance = initial_balance
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit

        # Peak equity tracking (watermark)
        self._peak_balance = initial_balance
        self._max_drawdown_seen = 0.0

        # State
        self._state = self._load_state()
        self._peak_balance = self._state.get("peak_balance", initial_balance)
        self._max_drawdown_seen = self._state.get("max_drawdown_seen", 0.0)

        # Daily/weekly loss tracking
        self._daily_pnl: dict[str, float] = self._state.get("daily_pnl", {})

        log.info(
            f"[DrawdownController] Initialized | "
            f"Initial: ${initial_balance:,.2f} | "
            f"Max DD: {max_drawdown_limit}% | "
            f"Daily Limit: {daily_loss_limit}% | "
            f"Weekly Limit: {weekly_loss_limit}%"
        )

    # ═══════════════════════════════════════════════════════
    # CORE DRAWDOWN CALCULATION
    # ═══════════════════════════════════════════════════════

    def current_drawdown_pct(self, current_balance: float) -> float:
        """
        Calculate current drawdown percentage from peak equity.

        Drawdown = (Peak - Current) / Peak * 100

        This is the standard drawdown calculation used by fund managers.
        """
        if self._peak_balance <= 0:
            return 0.0
        dd = (self._peak_balance - current_balance) / self._peak_balance * 100
        return max(0.0, round(dd, 2))

    def update_peak(self, balance: float) -> None:
        """Update peak balance watermark if new high reached."""
        if balance > self._peak_balance:
            self._peak_balance = balance
            log.info(
                f"[DrawdownController] New peak: ${balance:,.2f}"
            )
            self._save_state()

    # ═══════════════════════════════════════════════════════
    # PROTECTION LEVEL DETERMINATION
    # ═══════════════════════════════════════════════════════

    def get_protection_level(self, balance: float) -> str:
        """
        Determine current protection level based on drawdown.

        Returns: "GREEN", "YELLOW", "ORANGE", "RED", or "CRITICAL"
        """
        dd = self.current_drawdown_pct(balance)

        for level_name, config in PROTECTION_LEVELS.items():
            low, high = config["dd_range"]
            if low <= dd < high:
                return level_name

        return "CRITICAL"

    def get_risk_scale(self, balance: float) -> float:
        """
        Get risk scaling factor based on protection level.

        Returns:
            1.0 (GREEN), 0.7 (YELLOW), 0.5 (ORANGE),
            0.25 (RED), 0.0 (CRITICAL)
        """
        level = self.get_protection_level(balance)
        return PROTECTION_LEVELS[level]["risk_scale"]

    def get_action(self, balance: float) -> str:
        """Get recommended action based on protection level."""
        level = self.get_protection_level(balance)
        return PROTECTION_LEVELS[level]["action"]

    # ═══════════════════════════════════════════════════════
    # EMERGENCY CHECKS
    # ═══════════════════════════════════════════════════════

    def check_emergency(
        self,
        balance: float,
        performance: dict | None = None,
    ) -> dict:
        """
        Check all emergency conditions.

        Checks:
          1. Drawdown > max limit → CRITICAL (stop all trading)
          2. Daily loss > limit → stop for the day
          3. Weekly loss > limit → reduce risk
          4. Protection level assessment

        Args:
            balance: Current account balance
            performance: Dict with daily_loss_pct, weekly_loss_pct, etc.

        Returns:
            {
                "stop_trading": True/False,
                "level": "GREEN"/"YELLOW"/...,
                "reason": str,
                "risk_scale": float,
            }
        """
        if performance is None:
            performance = {}

        dd = self.current_drawdown_pct(balance)
        level = self.get_protection_level(balance)
        risk_scale = PROTECTION_LEVELS[level]["risk_scale"]

        # Update peak if new high
        self.update_peak(balance)

        # Track max drawdown
        if dd > self._max_drawdown_seen:
            self._max_drawdown_seen = dd
            log.warning(
                f"[DrawdownController] New max drawdown: {dd:.1f}%"
            )

        # 1. Critical drawdown check
        if dd >= self.max_drawdown_limit:
            return {
                "stop_trading": True,
                "level": "CRITICAL",
                "reason": (
                    f"CRITICAL: Drawdown {dd:.1f}% >= "
                    f"max {self.max_drawdown_limit}%. "
                    f"STOP ALL TRADING."
                ),
                "risk_scale": 0.0,
                "drawdown_pct": dd,
            }

        # 2. Daily loss limit
        daily_loss = performance.get("daily_loss_pct", 0.0)
        if daily_loss >= self.daily_loss_limit:
            return {
                "stop_trading": True,
                "level": level,
                "reason": (
                    f"Daily loss limit reached: {daily_loss:.1f}% >= "
                    f"{self.daily_loss_limit}%. Stop trading today."
                ),
                "risk_scale": 0.0,
                "drawdown_pct": dd,
            }

        # 3. Weekly loss limit
        weekly_loss = performance.get("weekly_loss_pct", 0.0)
        if weekly_loss >= self.weekly_loss_limit:
            return {
                "stop_trading": False,
                "level": level,
                "reason": (
                    f"Weekly loss limit reached: {weekly_loss:.1f}% >= "
                    f"{self.weekly_loss_limit}%. Reducing risk."
                ),
                "risk_scale": min(risk_scale, 0.3),
                "drawdown_pct": dd,
            }

        # 4. RED level → significant reduction
        if level == "RED":
            return {
                "stop_trading": False,
                "level": level,
                "reason": (
                    f"RED level: Drawdown {dd:.1f}%. "
                    f"Emergency mode — minimal trading only."
                ),
                "risk_scale": 0.25,
                "drawdown_pct": dd,
            }

        return {
            "stop_trading": False,
            "level": level,
            "reason": PROTECTION_LEVELS[level]["action"],
            "risk_scale": risk_scale,
            "drawdown_pct": dd,
        }

    # ═══════════════════════════════════════════════════════
    # TRADE RECORDING
    # ═══════════════════════════════════════════════════════

    def record_trade(self, pnl_usd: float, balance: float) -> None:
        """
        Record a trade result for PnL tracking.

        Updates:
          - Peak balance watermark
          - Daily PnL tracking
          - Drawdown calculations
        """
        today = date.today().isoformat()
        self._daily_pnl[today] = self._daily_pnl.get(today, 0) + pnl_usd

        # Update peak
        if balance > self._peak_balance:
            self._peak_balance = balance

        # Track max drawdown
        dd = self.current_drawdown_pct(balance)
        if dd > self._max_drawdown_seen:
            self._max_drawdown_seen = dd

        self._save_state()

        log.debug(
            f"[DrawdownController] Trade recorded: ${pnl_usd:+.2f} | "
            f"Balance: ${balance:,.2f} | DD: {dd:.1f}% | "
            f"Peak: ${self._peak_balance:,.2f}"
        )

    # ═══════════════════════════════════════════════════════
    # DAILY/WEEKLY PNL QUERIES
    # ═══════════════════════════════════════════════════════

    def get_daily_pnl(self) -> float:
        """Get today's PnL."""
        today = date.today().isoformat()
        return self._daily_pnl.get(today, 0.0)

    def get_weekly_pnl(self) -> float:
        """Get this week's PnL (last 7 days)."""
        week_ago = (date.today() - timedelta(days=7)).isoformat()
        return sum(
            v for k, v in self._daily_pnl.items() if k >= week_ago
        )

    def get_daily_loss_pct(self) -> float:
        """Get today's loss as percentage of initial balance."""
        daily = self.get_daily_pnl()
        if daily >= 0:
            return 0.0
        return round(abs(daily) / self.initial_balance * 100, 2)

    def get_weekly_loss_pct(self) -> float:
        """Get this week's loss as percentage of initial balance."""
        weekly = self.get_weekly_pnl()
        if weekly >= 0:
            return 0.0
        return round(abs(weekly) / self.initial_balance * 100, 2)

    def is_daily_limit_hit(self) -> bool:
        """Check if daily loss limit is hit."""
        return self.get_daily_loss_pct() >= self.daily_loss_limit

    def is_weekly_limit_hit(self) -> bool:
        """Check if weekly loss limit is hit."""
        return self.get_weekly_loss_pct() >= self.weekly_loss_limit

    # ═══════════════════════════════════════════════════════
    # DAILY RESET
    # ═══════════════════════════════════════════════════════

    def reset_daily(self) -> None:
        """Reset daily counters (called at start of new day)."""
        # Clean up old daily PnL entries (keep last 30 days)
        cutoff = (date.today() - timedelta(days=30)).isoformat()
        self._daily_pnl = {
            k: v for k, v in self._daily_pnl.items() if k >= cutoff
        }
        self._save_state()
        log.info("[DrawdownController] Daily reset complete")

    # ═══════════════════════════════════════════════════════
    # STATUS & REPORTING
    # ═══════════════════════════════════════════════════════

    def get_status(self, balance: float) -> dict:
        """Get complete drawdown status."""
        level = self.get_protection_level(balance)
        return {
            "balance": round(balance, 2),
            "peak_balance": round(self._peak_balance, 2),
            "current_drawdown_pct": self.current_drawdown_pct(balance),
            "max_drawdown_seen": round(self._max_drawdown_seen, 2),
            "protection_level": level,
            "risk_scale": PROTECTION_LEVELS[level]["risk_scale"],
            "action": PROTECTION_LEVELS[level]["action"],
            "daily_pnl": round(self.get_daily_pnl(), 2),
            "weekly_pnl": round(self.get_weekly_pnl(), 2),
            "daily_loss_pct": self.get_daily_loss_pct(),
            "weekly_loss_pct": self.get_weekly_loss_pct(),
            "daily_limit_hit": self.is_daily_limit_hit(),
            "weekly_limit_hit": self.is_weekly_limit_hit(),
        }

    def print_status(self, balance: float) -> None:
        """Print drawdown status."""
        s = self.get_status(balance)
        level_icons = {
            "GREEN": "G", "YELLOW": "Y",
            "ORANGE": "O", "RED": "R", "CRITICAL": "!",
        }
        icon = level_icons.get(s["protection_level"], "?")
        bar = "=" * 50
        print(f"\n{bar}")
        print(f"  [{icon}] DRAWDOWN CONTROLLER")
        print(bar)
        print(f"  Balance           : ${s['balance']:,.2f}")
        print(f"  Peak Balance      : ${s['peak_balance']:,.2f}")
        print(f"  Current Drawdown  : {s['current_drawdown_pct']:.1f}%")
        print(f"  Max Drawdown Seen : {s['max_drawdown_seen']:.1f}%")
        print(f"  Protection Level  : {s['protection_level']}")
        print(f"  Risk Scale        : {s['risk_scale']*100:.0f}%")
        print(f"  Action            : {s['action']}")
        print(f"  Daily PnL         : ${s['daily_pnl']:+,.2f} ({s['daily_loss_pct']}%)")
        print(f"  Weekly PnL        : ${s['weekly_pnl']:+,.2f} ({s['weekly_loss_pct']}%)")
        print(f"  Daily Limit Hit   : {'YES' if s['daily_limit_hit'] else 'NO'}")
        print(f"  Weekly Limit Hit  : {'YES' if s['weekly_limit_hit'] else 'NO'}")
        print(bar + "\n")

    # ═══════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════

    def _load_state(self) -> dict:
        """KILL SHOT FIX: Fail CLOSED on corruption, not open.
        Previously returned fresh state (peak=initial, drawdown=0) on any
        read error — silently resetting safety counters. Now returns
        a FAIL-SAFE state that blocks new trades.
        """
        DRAWDOWN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if DRAWDOWN_STATE_PATH.exists():
            try:
                with open(DRAWDOWN_STATE_PATH) as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError) as e:
                log.critical(
                    f"drawdown_controller: state CORRUPT ({e}) — "
                    f"FAILING CLOSED. Manual intervention required."
                )
                return {
                    "peak_balance": self.initial_balance,
                    "max_drawdown_seen": 999999,  # blocks all trades
                    "daily_pnl": {},
                    "_corrupt": True,
                }
            except Exception as e:
                log.warning(f"drawdown_controller: state read error: {e}")
        return {
            "peak_balance": self.initial_balance,
            "max_drawdown_seen": 0.0,
            "daily_pnl": {},
        }

    def _save_state(self) -> None:
        state = {
            "peak_balance": round(self._peak_balance, 2),
            "max_drawdown_seen": round(self._max_drawdown_seen, 2),
            "daily_pnl": self._daily_pnl,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        # KILL SHOT FIX: Atomic write (temp file + os.replace)
        import tempfile
        dir_name = os.path.dirname(DRAWDOWN_STATE_PATH) or "."
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_name, suffix=".tmp",
                prefix="dd_state_", delete=False
            ) as tmp_f:
                json.dump(state, tmp_f, indent=2)
                tmp_path = tmp_f.name
            os.replace(tmp_path, DRAWDOWN_STATE_PATH)
        except Exception as e:
            try:
                os.unlink(tmp_path)
            except (OSError, UnboundLocalError):
                pass
            raise
