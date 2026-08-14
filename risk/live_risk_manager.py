"""
risk/live_risk_manager.py — Live Risk Manager (Day 75)
========================================================

The central risk controller. Every trade MUST pass through this
before execution. It coordinates:

  1. Capital Tier System (Tier 1/2/3 — gradual risk increase)
  2. Kill Switch (3-level emergency brake)
  3. Drawdown Monitor (Capital Preservation Mode)
  4. Position Sizer (dynamic lot sizing)
  5. Exposure Manager (correlation + direction limits)
  6. Risk Reporter (event logging + Telegram alerts)

Permission flow:
  Signal → Confidence Check → Kill Switch → Drawdown Mode
         → Exposure Check → Position Size → Spread Check → Execute

Usage:
    mgr = get_live_risk_manager()
    permission = mgr.check_trade_permission(
        pair="EURUSD", direction="BUY", confidence=75,
        sl_pips=20, atr=0.001, balance=10000,
    )
    if permission.allowed:
        execute_trade(permission.lot, permission.sl, permission.tp)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

from risk.position_sizer import PositionSizer, PositionSizeResult, get_position_sizer
from risk.kill_switch import KillSwitch, get_kill_switch
from risk.exposure_manager import ExposureManager, get_exposure_manager
from risk.drawdown_monitor import DrawdownMonitor, DrawdownStatus, get_drawdown_monitor
from risk.risk_reporter import RiskReporter, get_risk_reporter

log = get_logger("live_risk_manager")


# ── Capital Tier System ─────────────────────────────────────────────

@dataclass
class CapitalTier:
    """One tier of the capital progression system.

    ⚠ CODE VERIFICATION REQUIRED (2026-08-14 forensic-audit finding):
    `approval_mode` below is stored on every tier (manual / semi_auto /
    fully_auto) but is NEVER read anywhere in this file --
    check_trade_permission() does not consult it, and nothing else in
    this module gates execution on it. Confirmed against a real trade:
    trader.log shows the EURUSD SELL that triggered this audit (ticket
    10015498153) ran under tier=1 -- "Initial Live", approval_mode=
    "manual" by design -- yet executed via "[Mode 3 — Autonomous]" with
    no human step. Either (a) manual/semi/fully-auto approval is enforced
    entirely by a *different* module (an "approval_mode"/Mode-1/2/3
    system referenced in trader.log that this class doesn't talk to at
    all, in which case this field is dead code misleadingly implying a
    protection that lives elsewhere or doesn't connect), or (b) it's
    supposed to be enforced here and the wiring was simply never written.
    Not fixed here: doing so blind, without seeing core/trader.py or
    whatever implements "Mode 3 — Autonomous", risks either duplicating
    an existing gate incorrectly or breaking live execution. Needs that
    source before this can be safely resolved.
    """
    tier: int
    name: str
    risk_per_trade: float        # 0.005 = 0.5%
    daily_loss_limit: float      # 0.015 = 1.5%
    max_trades_per_day: int
    min_confidence: float
    approval_mode: str           # manual / semi_auto / fully_auto -- see warning above: unenforced here
    tier_mult: float             # position size multiplier

# Import from the single source of truth (core.constants).
# Keeps TIERS in sync with get_max_trades_per_day().
def _max_trades() -> int:
    try:
        from core.constants import get_max_trades_per_day
        return get_max_trades_per_day()
    except Exception:
        return 20


def _build_tiers() -> dict:
    """Bug #14 fix: lazy TIERS construction.

    Previously TIERS was built at MODULE LOAD TIME, calling
    _max_trades() which imports core.constants. If that import
    failed (circular dependency, missing module), the entire
    live_risk_manager module would fail to import — killing the
    entire trading pipeline.
    Now TIERS is a lazy-initialized dict, built on first access.
    """
    mt = _max_trades()
    # 2026-08-13: lowered min_confidence from 70 → 55 for all tiers.
    # Per-pair strategies produce 55-85% confidence signals; the old 70%
    # floor blocked ALL of them. 55% matches per-pair profile min_confidence.
    #
    # 2026-08-14 forensic-audit correction: the line below was never
    # actually updated to 55 -- it reads 50.0 for all three tiers, and
    # trader.log confirms 50.0 is what's really enforced in production
    # ("Min confidence 63.8% (per-pair min 50%)", tier=1). This comment
    # was therefore describing a value the code never had. Also relevant
    # to the audit's open question about the Devil's Advocate module
    # referencing a "typical threshold of 70": that 70% does not exist
    # anywhere in the current tier config -- the real, enforced floor is
    # 50% and is IDENTICAL across all three tiers, so tier no longer
    # differentiates confidence requirements at all (see the now-inaccurate
    # action_taken log strings in maybe_promote_tier() below, fixed
    # alongside this comment). Not changing the actual 50.0 value here --
    # that's a live risk parameter and, per the audit's own methodology
    # (§19), shouldn't move without a baseline-vs-modified backtest; this
    # only fixes the comment to describe what the code actually does.
    return {
        1: CapitalTier(1, "Initial Live", 0.005, 0.015, mt, 50.0, "manual", 0.5),
        2: CapitalTier(2, "Controlled Automation", 0.01, 0.03, mt, 50.0, "semi_auto", 0.8),
        3: CapitalTier(3, "Mature System", 0.01, 0.03, mt, 50.0, "fully_auto", 1.0),
    }


class _LazyTiers:
    """Dict-like proxy that builds TIERS on first access."""
    def __init__(self):
        self._data = None

    def _ensure(self):
        if self._data is None:
            self._data = _build_tiers()

    def get(self, key, default=None):
        self._ensure()
        return self._data.get(key, default)

    def __getitem__(self, key):
        self._ensure()
        return self._data[key]

    def __contains__(self, key):
        self._ensure()
        return key in self._data


TIERS = _LazyTiers()


@dataclass
class TradePermissionResult:
    """Result of LiveRiskManager's trade permission check.

    H6 ARCHITECTURAL FIX: renamed from `TradePermission` to
    `TradePermissionResult` to disambiguate from
    `risk/trade_permission.py::TradePermission` (the final permission
    gate class). The two classes serve different purposes:
      - `TradePermissionResult` (this class) — the result dataclass
        returned by `LiveRiskManager.check_trade_permission()`
      - `TradePermission` (in risk/trade_permission.py) — the gate
        class that runs the final checklist

    A backward-compat alias `TradePermission = TradePermissionResult`
    is kept below so existing imports continue to work, but new code
    should use `TradePermissionResult`.
    """
    allowed: bool = False
    lot: float = 0.0
    risk_amount_usd: float = 0.0
    risk_pct: float = 0.0
    sl_pips: float = 0.0
    emergency_close_required: bool = False
    tp_pips: float = 0.0
    reject_reason: str = ""
    tier: int = 1
    mode: str = "NORMAL"
    checks: List[Dict[str, Any]] = field(default_factory=list)
    position_sizing: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backward-compat alias — existing imports of `TradePermission` from
# this module continue to work. New code should import `TradePermissionResult`.
TradePermission = TradePermissionResult


class LiveRiskManager:
    """Central risk controller — every trade passes through here."""

    def __init__(self, initial_balance: float = 10000.0, tier: int = 1):
        """
        Args:
            initial_balance: Starting account balance.
            tier: Initial capital tier (1, 2, or 3).

        Round-7 audit fix: default tier changed from 3 → 1.

        Previously: `tier: int = 3` meant a fresh account with 0
        trades was treated as a "Mature System" (TIERS[3]) with
        min_confidence=50.0, fully_auto approval, and 1.0× position
        multiplier. That's exactly backwards — a new account should
        START at Tier 1 ("Initial Live": manual approval, 80%
        min_confidence, 0.5× multiplier) and EARN its way up via
        maybe_promote_tier() as trade count + win rate grow.

        The operator's audit caught this: "LiveRiskManager(tier=3)
        default, TIERS[3].min_confidence = 50.0। fresh account (0
        trade) কেও সরাসরি 'Mature System' ধরে নেয়।"

        To override the default (e.g. for a proven account being re-
        deployed), pass tier=3 explicitly OR set the
        `LRM_INITIAL_TIER` env var.
        """
        import os as _os
        # Env var override — lets operators pin a tier without code changes
        env_tier = _os.getenv("LRM_INITIAL_TIER", "").strip()
        if env_tier.isdigit():
            tier = int(env_tier)
        self.initial_balance = initial_balance
        # Round-7: default fallback also changed 3 → 1 (defensive)
        self.current_tier = TIERS.get(tier, TIERS[1])
        self.position_sizer = get_position_sizer()
        self.kill_switch = get_kill_switch()
        self.exposure_mgr = get_exposure_manager()
        self.drawdown_monitor = get_drawdown_monitor()
        self.risk_reporter = get_risk_reporter()
        self._trades_today = 0
        self._consecutive_losses = 0  # legacy in-memory cache — reads now go through StreakTracker
        # H5 FIX: optional learning agent for tier promotion stats
        self.learning_agent = None

    def set_tier(self, tier: int) -> None:
        """Set the capital tier (1, 2, or 3)."""
        if tier in TIERS:
            old_tier = self.current_tier.tier
            self.current_tier = TIERS[tier]
            if old_tier != tier:
                log.info(
                    f"[LiveRiskManager] Tier changed {old_tier} → {tier} "
                    f"({self.current_tier.name}) | min_conf={self.current_tier.min_confidence}% | "
                    f"risk/trade={self.current_tier.risk_per_trade:.3f} | "
                    f"mult={self.current_tier.tier_mult}"
                )

    def maybe_promote_tier(
        self,
        total_closed_trades: int,
        win_rate: float,
    ) -> bool:
        """
        Auto-promote the capital tier based on closed-trade count + win rate.

        Round-7 audit fix: previously, set_tier() existed but NOTHING
        in the codebase called it — a fresh account stayed at its
        initial tier forever. The operator's audit caught this:
        "কোথাও tier auto-promotion নেই — set_tier() মেথড আছে কিন্তু
        trade count বেড়ে গেলে কে call করবে সেটা কোথাও নেই।"

        Promotion rules (conservative — favor capital preservation):
          Tier 1 → Tier 2: ≥ 10 closed trades AND win_rate ≥ 45%
          Tier 2 → Tier 3: ≥ 30 closed trades AND win_rate ≥ 50%

        Demotion (when performance degrades):
          Tier 3 → Tier 2: win_rate drops below 40% (after ≥ 20 trades)
          Tier 2 → Tier 1: win_rate drops below 35% (after ≥ 10 trades)

        This method is idempotent — calling it repeatedly with the same
        stats is safe. It only logs when the tier actually changes.

        Args:
            total_closed_trades: Lifetime count of closed trades (wins + losses).
            win_rate: Win rate as a percentage (0-100). 0 if no trades.

        Returns:
            True if tier was changed, False otherwise.
        """
        current = self.current_tier.tier

        # ── Promotion path ──
        if current == 1 and total_closed_trades >= 10 and win_rate >= 45:
            self.set_tier(2)
            self.risk_reporter.record_event(
                "TIER_PROMOTION",
                trigger_value=f"Tier 1 → 2 (trades={total_closed_trades}, wr={win_rate:.1f}%)",
                # 2026-08-14 forensic-audit fix: this used to say
                # "min_conf 80%→70%" -- a claim from before the 2026-08-13
                # change that flattened min_confidence to 50.0 on every
                # tier. Promotion no longer changes min_confidence at all;
                # logging that it does misleads anyone reading the risk
                # event trail (including a future audit) into thinking the
                # confidence gate tightens or loosens on promotion when it
                # is now identical (50%) before and after. State what
                # actually changes: risk-per-trade and position multiplier.
                action_taken=f"risk 0.5%→1.0%, mult 0.5→0.8 (min_conf unchanged at "
                             f"{TIERS[2].min_confidence:.0f}% -- see 2026-08-13 flattening)",
                send_telegram=True,
            )
            return True
        if current == 2 and total_closed_trades >= 30 and win_rate >= 50:
            self.set_tier(3)
            self.risk_reporter.record_event(
                "TIER_PROMOTION",
                trigger_value=f"Tier 2 → 3 (trades={total_closed_trades}, wr={win_rate:.1f}%)",
                # 2026-08-14 forensic-audit fix: same correction as above --
                # was "min_conf 70%→55%, approval semi_auto→fully_auto".
                # min_confidence does not change (flat 50% across tiers).
                # approval_mode is left in the message because that field
                # genuinely differs per tier in CapitalTier -- but see the
                # separate CODE VERIFICATION REQUIRED note where this class
                # is defined: approval_mode is not actually read/enforced
                # anywhere in this file, so this log line describes the
                # *intended* design, not a verified live behavior change.
                action_taken=f"mult 0.8→1.0 (min_conf unchanged at "
                             f"{TIERS[3].min_confidence:.0f}% -- see 2026-08-13 flattening; "
                             f"approval_mode field says semi_auto→fully_auto but is NOT "
                             f"currently enforced by this module -- CODE VERIFICATION "
                             f"REQUIRED against whatever does gate manual/auto execution)",
                send_telegram=True,
            )
            return True

        # ── Demotion path (capital preservation) ──
        if current == 3 and total_closed_trades >= 20 and win_rate < 40:
            self.set_tier(2)
            self.risk_reporter.record_event(
                "TIER_DEMOTION",
                trigger_value=f"Tier 3 → 2 (trades={total_closed_trades}, wr={win_rate:.1f}%)",
                action_taken=f"win_rate below 40% — reducing risk until performance recovers",
                send_telegram=True,
            )
            return True
        if current == 2 and total_closed_trades >= 10 and win_rate < 35:
            self.set_tier(1)
            self.risk_reporter.record_event(
                "TIER_DEMOTION",
                trigger_value=f"Tier 2 → 1 (trades={total_closed_trades}, wr={win_rate:.1f}%)",
                action_taken=f"win_rate below 35% — reducing risk to minimum",
                send_telegram=True,
            )
            return True

        return False

    def _get_consecutive_losses(self) -> int:
        """Co-founder fix: READ from the authoritative StreakTracker
        (which reads CircuitBreaker's persisted state) instead of our
        own in-memory counter. Our in-memory counter resets on restart
        and drifts from the real count; StreakTracker always reflects
        the true persisted value."""
        try:
            from risk.streak_tracker import get_consecutive_losses
            return get_consecutive_losses()
        except Exception:
            # Fallback to in-memory cache if StreakTracker unavailable
            return self._consecutive_losses

    def record_trade_result(self, won: bool, pnl_usd: float = 0.0) -> None:
        """Record a trade outcome for streak tracking + tier promotion.

        Co-founder fix: we still update our local cache for backward
        compat, but reads now go through StreakTracker (which reads
        CircuitBreaker's persisted state). CircuitBreaker.record_result
        is the SINGLE WRITER — this method is now just a local mirror
        for modules that haven't been migrated yet.

        H5 ARCHITECTURAL FIX: this method now ALSO triggers
        `maybe_promote_tier()` after each closed trade. Previously,
        `maybe_promote_tier()` existed but was NEVER called anywhere in
        the codebase — a fresh account stayed at its initial tier forever.
        Now: after each closed trade, we pull cumulative stats from the
        learning agent (via the stats callback) and attempt tier
        promotion / demotion. This is the wire-up the audit requested.

        Args:
            won: True if the trade was a win, False if loss.
            pnl_usd: Profit/loss in USD (used for stats display, optional).
        """
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 3:
                self.risk_reporter.record_event(
                    "LOSS_STREAK_WARNING",
                    trigger_value=f"{self._consecutive_losses} consecutive losses",
                    action_taken="Position size reduced",
                )

        # ── H5 FIX: Auto-promote/demote tier based on cumulative stats ──
        # Pull lifetime trade count + win rate from the learning agent.
        # The stats callback is optional — if not wired, we skip the
        # tier check (graceful degradation).
        try:
            stats = self._get_lifetime_stats()
            if stats and stats.get("total", 0) > 0:
                self.maybe_promote_tier(
                    total_closed_trades=stats["total"],
                    win_rate=stats.get("win_rate", 0.0),
                )
        except Exception as _e_tier:
            log.debug(f"[LiveRiskManager] Tier promotion check skipped: {_e_tier}")

    def _get_lifetime_stats(self) -> Optional[Dict[str, Any]]:
        """Pull lifetime trade stats for tier promotion check.

        H5 FIX: tries multiple sources in order:
          1. learning_agent (if wired by trader.py)
          2. circuit_breaker (always available, has win/loss counts)
          3. Returns None if neither is available (graceful degradation)

        Returns:
            {"total": int, "wins": int, "losses": int, "win_rate": float}
            or None if no stats source available.
        """
        # Source 1: learning agent (most accurate — has full trade history)
        if getattr(self, "learning_agent", None) is not None:
            try:
                ls = self.learning_agent.get_performance_stats()
                if isinstance(ls, dict):
                    total = int(ls.get("total_decisions", 0) or ls.get("closed_trades", 0) or 0)
                    if total > 0:
                        wr = float(ls.get("win_rate", 0) or 0)
                        # Strip trailing % if present
                        if isinstance(wr, str) and wr.endswith("%"):
                            wr = float(wr[:-1])
                        return {
                            "total": total,
                            "wins": int(wr * total / 100) if wr > 0 else 0,
                            "losses": int(total * (100 - wr) / 100) if wr > 0 else total,
                            "win_rate": float(wr),
                        }
            except Exception:
                pass

        # Source 2: circuit breaker (always available — has its own counters)
        try:
            cb = self.kill_switch
            if cb is not None and hasattr(cb, "get_stats"):
                cs = cb.get_stats()
                if isinstance(cs, dict):
                    total = int(cs.get("total_trades", 0) or 0)
                    wins = int(cs.get("wins", 0) or 0)
                    if total > 0:
                        return {
                            "total": total,
                            "wins": wins,
                            "losses": total - wins,
                            "win_rate": float(wins * 100 / total),
                        }
        except Exception:
            pass

        return None

    def attach_learning_agent(self, learning_agent) -> None:
        """Wire the learning agent for tier promotion stats.

        H5 FIX: trader.py should call this once at boot so
        record_trade_result() can pull lifetime win/loss stats.
        """
        self.learning_agent = learning_agent
        log.info("[LiveRiskManager] Learning agent attached for tier promotion stats")

    def reset_daily(self) -> None:
        """Reset daily counters (called at start of each trading day)."""
        self._trades_today = 0

    def check_trade_permission(
        self,
        pair: str,
        direction: str,
        confidence: float,
        sl_pips: float,
        tp_pips: float,
        balance: float,
        atr: float = 0.001,
        atr_median: float = 0.001,
        spread_pips: float = 1.5,
        open_positions: Optional[List[Dict]] = None,
        daily_pnl: float = 0.0,
        weekly_pnl: float = 0.0,
    ) -> TradePermission:
        """Run ALL risk checks before allowing a trade.

        This is the FINAL gate before MT5 execution.

        Args:
            pair: Trading pair.
            direction: BUY or SELL.
            confidence: Master Decision confidence (0-100).
            sl_pips: Stop loss in pips.
            tp_pips: Take profit in pips.
            balance: Current account balance.
            atr: Current ATR.
            atr_median: Median ATR (for volatility comparison).
            spread_pips: Current spread in pips.
            open_positions: List of currently open positions.
            daily_pnl: Today's PnL (negative = loss).
            weekly_pnl: This week's PnL.

        Returns:
            TradePermission with allowed + lot + reject_reason.
        """
        perm = TradePermission(sl_pips=sl_pips, tp_pips=tp_pips)
        tier = self.current_tier

        # Update exposure manager
        self.exposure_mgr.update_positions(open_positions or [])

        # Update drawdown monitor
        dd_status = self.drawdown_monitor.update(balance, self.initial_balance)
        perm.mode = dd_status.mode

        # ── Check 1: Kill Switch ────────────────────────────────────
        ks = self.kill_switch.check(balance, self.initial_balance, daily_pnl, weekly_pnl)
        perm.checks.append({"check": "kill_switch", "passed": ks["trading_allowed"], "detail": ks["reason"]})
        if not ks["trading_allowed"]:
            perm.reject_reason = f"Kill Switch L{ks['level']}: {ks['reason']}"
            # Level 3 = max drawdown = full stop. Previously this only
            # blocked NEW trades — existing open positions stayed open
            # indefinitely with no automatic mechanism to close them
            # (the only close-all path was a manual webhook command).
            # Flag it here so the caller (core/trader.py, which has the
            # paper_trader/mt5_connection/notifier this module doesn't)
            # can trigger execution/emergency_exit.py's close_all_positions().
            if ks.get("level") == 3:
                perm.emergency_close_required = True
            self.risk_reporter.record_event(
                f"KILL_SWITCH_L{ks['level']}",
                trigger_value=ks["reason"],
                action_taken="Trade blocked",
            )
            return perm

        # ── Check 2: Confidence floor (tier + drawdown adjusted) ────
        min_conf = max(tier.min_confidence, dd_status.min_confidence_required)
        if confidence < min_conf:
            perm.reject_reason = f"Confidence {confidence:.0f}% < {min_conf:.0f}% (tier={tier.tier}, mode={dd_status.mode})"
            perm.checks.append({"check": "confidence", "passed": False, "detail": perm.reject_reason})
            return perm
        perm.checks.append({"check": "confidence", "passed": True, "detail": f"{confidence:.0f}% ≥ {min_conf:.0f}%"})

        # ── Check 3: Daily trade count ──────────────────────────────
        _max_daily = tier.max_trades_per_day
        if self._trades_today >= _max_daily:
            perm.reject_reason = (
                f"Max trades/day reached ({self._trades_today}/{_max_daily}) | "
                f"source=risk/live_risk_manager.py:check_trade_permission() | "
                f"config=TIERS[tier={tier.tier}].max_trades_per_day={_max_daily}"
            )
            perm.checks.append({"check": "daily_trades", "passed": False, "detail": perm.reject_reason})
            return perm
        perm.checks.append({"check": "daily_trades", "passed": True, "detail": f"{self._trades_today}/{_max_daily}"})

        # ── Check 4: Spread check ───────────────────────────────────
        max_spread = 5.0  # max 5 pips spread
        if spread_pips > max_spread:
            perm.reject_reason = f"Spread too high: {spread_pips:.1f} > {max_spread}"
            perm.checks.append({"check": "spread", "passed": False, "detail": perm.reject_reason})
            return perm
        perm.checks.append({"check": "spread", "passed": True, "detail": f"{spread_pips:.1f} pips"})

        # ── Check 5: Exposure / correlation ─────────────────────────
        # Estimate risk_usd for exposure check
        est_risk = balance * tier.risk_per_trade * tier.tier_mult
        exp = self.exposure_mgr.check(pair, direction, lot=0.1, risk_usd=est_risk, balance=balance)
        perm.checks.append({"check": "exposure", "passed": exp.allowed, "detail": exp.reason})
        if not exp.allowed:
            perm.reject_reason = f"Exposure: {exp.reason}"
            self.risk_reporter.record_event("EXPOSURE_REJECTED", trigger_value=exp.reason, action_taken="Trade blocked")
            return perm

        # ── Check 6: Position sizing ────────────────────────────────
        # FIX (2026-08-08, cent-account audit): this was hardcoded
        # `10.0 if not pair.endswith("JPY") else 9.0` — a static
        # real-USD pip value. On a Cent account (Exness Cent etc.),
        # `balance` above is already synced from live MT5 in cents
        # (see AITrader._sync_balance), so mixing it with a real-USD
        # pip value here creates the same ~100x unit mismatch that
        # core/constants.get_live_pip_value_per_lot() was written to
        # fix — core/trader.py already uses that live lookup for the
        # trade's actual final lot size, but this function's own
        # internal sizing check was never switched over, so it could
        # wrongly reject (sizing.lot <= 0) a valid cent-account trade
        # even though the real trade would have sized correctly.
        try:
            from core.constants import get_live_pip_value_per_lot
            pip_value = get_live_pip_value_per_lot(pair)
        except Exception as e:
            pip_value = 10.0 if not pair.endswith("JPY") else 9.0
            log.warning(
                f"[LiveRiskManager] get_live_pip_value_per_lot failed for "
                f"{pair} ({e}) — using static pip_value={pip_value}. On a "
                f"Cent account this can misjudge (not miss-execute) the "
                f"position_size permission check."
            )
        sizing = self.position_sizer.calculate(
            balance=balance,
            risk_pct=tier.risk_per_trade,
            sl_pips=sl_pips,
            pip_value_per_lot=pip_value,
            confidence=confidence,
            atr=atr,
            atr_median=atr_median,
            consecutive_losses=self._get_consecutive_losses(),  # Co-founder fix: read from StreakTracker
            tier_mult=tier.tier_mult * dd_status.position_multiplier,
        )
        perm.position_sizing = sizing.to_dict()

        if sizing.lot <= 0:
            perm.reject_reason = f"Position sizing: {sizing.reason}"
            perm.checks.append({"check": "position_size", "passed": False, "detail": sizing.reason})
            return perm

        perm.checks.append({"check": "position_size", "passed": True, "detail": f"lot={sizing.lot}, risk=${sizing.risk_amount_usd}"})

        # ── ALL CHECKS PASSED ───────────────────────────────────────
        perm.allowed = True
        perm.lot = sizing.lot
        perm.risk_amount_usd = sizing.risk_amount_usd
        perm.risk_pct = sizing.risk_pct
        perm.tier = tier.tier
        self._trades_today += 1

        # Report capital preservation mode if active
        if dd_status.mode != "NORMAL":
            self.risk_reporter.record_event(
                "CAPITAL_PRESERVATION_ACTIVATED",
                trigger_value=f"DD={dd_status.current_drawdown_pct:.1%}, mode={dd_status.mode}",
                action_taken=f"min_conf={dd_status.min_confidence_required}, pos_mult={dd_status.position_multiplier}",
                send_telegram=False,
            )

        log.info(
            f"[LiveRiskManager] APPROVED {pair} {direction} | "
            f"lot={perm.lot} | risk=${perm.risk_amount_usd} ({perm.risk_pct:.2f}%) | "
            f"tier={perm.tier} | mode={perm.mode} | conf={confidence:.0f}%"
        )
        return perm

    def status(self) -> Dict[str, Any]:
        """Return full risk status for dashboard."""
        return {
            "tier": self.current_tier.tier,
            "tier_name": self.current_tier.name,
            "risk_per_trade": self.current_tier.risk_per_trade,
            "daily_loss_limit": self.current_tier.daily_loss_limit,
            "max_trades_day": self.current_tier.max_trades_per_day,
            "trades_today": self._trades_today,
            "consecutive_losses": self._get_consecutive_losses(),  # Co-founder fix: authoritative source
            "kill_switch": self.kill_switch.status(),
            "drawdown": self.drawdown_monitor.status(),
            "exposure": self.exposure_mgr.status(),
            "risk_events": self.risk_reporter.stats(),
        }


# ── Singleton ───────────────────────────────────────────────────────

_MANAGER: Optional[LiveRiskManager] = None


def get_live_risk_manager() -> LiveRiskManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = LiveRiskManager()
    return _MANAGER