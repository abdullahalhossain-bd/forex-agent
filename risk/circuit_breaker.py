# risk/circuit_breaker.py  —  Week 3 Upgrade | AI Kill Switch
# ============================================================
# Consecutive loss tracker + auto trading pause
# AI নিজেকে রক্ষা করতে শিখবে
# ============================================================

import json
import os
from datetime import datetime, date, timedelta
from utils.logger import get_logger

log = get_logger("circuit_breaker")

CB_STATE_DIR  = "memory"
CB_STATE_PATH = "memory/circuit_breaker_state.json"   # legacy/global path — kept only
                                                        # for symbol=None backward-compat


class CircuitBreaker:
    """
    AI-এর automatic protection system।

    Triggers:
      1. Consecutive losses >= threshold → pause trading
      2. Daily loss % >= limit → pause trading
      3. Abnormal volatility detected → pause trading
      4. Win rate drops severely → learning mode

    Modes after trigger:
      - TRADING   : Normal operation
      - PAUSED    : No new trades, existing managed
      - LEARNING  : Analyze mistakes, no trading
      - COOLDOWN  : Wait N hours before resume

    Usage:
        cb = CircuitBreaker(symbol="USDCAD")
        if cb.allow_trade():
            ... take trade ...
        cb.record_result("LOSS")

    BUG FIX (per-symbol isolation): CircuitBreaker used to be backed by a
    single shared state file (memory/circuit_breaker_state.json) no matter
    which pair called it. With 60+ pairs all running through one instance,
    a losing streak on ONE symbol (or even stale leftover data — see the
    staleness fix below) tripped LEARNING/PAUSED mode for every symbol
    simultaneously. Logs showed the exact same "Win rate dropped to 0.0%"
    trigger firing back-to-back for EURUSD, USDCAD, GBPJPY, XAUUSD... all
    in the same boot cycle. Callers should now construct one CircuitBreaker
    per symbol (`CircuitBreaker(symbol=pair, balance=...)`), which writes
    to its own state file under memory/circuit_breaker/<symbol>.json.
    `symbol=None` (or omitted) falls back to the old shared/global file so
    existing single-account-wide callers (e.g. a global daily-loss gate)
    keep working unchanged.
    """

    # Thresholds
    MAX_CONSECUTIVE_LOSSES = 3      # ৩টা loss → pause

    # P1 fix (audit §3.2): this used to silently fall back to
    # MAX_DAILY_LOSS_PCT = 20.0 if `from config import DAILY_LOSS_LIMIT_PCT`
    # failed — a permissive default 6-13x looser than the 1.5-3% every
    # other risk file in this codebase (risk_engine.py, live_risk_manager.py)
    # assumes. risk_engine.py's own P0-2 fix already established the
    # correct pattern for this exact situation: "Config loading must NOT
    # be wrapped in try/except. If config.py fails to import, the system
    # MUST crash on boot — silently trading with wrong risk parameters is
    # far more dangerous." This mirrors that pattern so CircuitBreaker and
    # RiskEngine can never silently disagree about the daily-loss ceiling
    # because one of them quietly downgraded to a different default.
    from config import DAILY_LOSS_LIMIT_PCT as _CFG_DLL
    MAX_DAILY_LOSS_PCT = float(_CFG_DLL)

    # 2026-07-23 addition: weekly loss halt. Wired in from risk/strict_risk_manager.py
    # (which was never imported anywhere live) — that file's 5% weekly-loss rule was
    # the one piece of its 10-rule bundle that had NO live equivalent (daily loss halt
    # existed here already; drawdown halt exists in live_risk_manager.py's
    # DrawdownMonitor; correlation/cooldown/trade-caps all exist elsewhere too — see
    # core/obsolete.py for the full redundancy breakdown). Added here, not as a
    # separate StrictRiskManager instance, because that class keeps its own parallel
    # equity/open-positions state that would start at zero and drift from the real
    # broker/DB state — a second source of truth is worse than no gate at all.
    MAX_WEEKLY_LOSS_PCT = 5.0

    MIN_WIN_RATE_THRESHOLD = 30.0   # ৩০% এর নিচে → learning mode
    COOLDOWN_HOURS         = 4      # pause এর পরে কত ঘণ্টা wait
    LOOKBACK_TRADES        = 10     # win rate চেক করার জন্য কতটা পিছনে

    # BUG FIX: LEARNING mode used to be a permanent trap — allow_trade()
    # just kept returning False forever with no expiry, because (unlike
    # PAUSED/COOLDOWN, which carries a cooldown_until timestamp) nothing
    # ever set the mode back to TRADING. Logs showed thousands of blocked
    # cycles per hour, each one still paying for a full market-data fetch
    # before being rejected. LEARNING_HOURS gives it the same kind of
    # timed expiry COOLDOWN already had, so the bot re-evaluates itself
    # instead of freezing until a human calls manual_resume().
    LEARNING_HOURS         = 2      # learning mode এর পরে কত ঘণ্টা wait before re-check

    # BUG FIX (stale recent_results never expired): recent_results was
    # persisted to disk indefinitely (last 50 entries, no age limit) and
    # only daily_loss_usd/date were reset on a new day. A losing streak
    # from days ago — or from a previous demo/backtest session — stayed in
    # recent_results forever and could immediately trip LEARNING mode on a
    # completely fresh boot, before a single trade had been taken this
    # session (logs showed "Win rate dropped to 0.0%" in the very first
    # cycle while the AI's own trade journal reported "0 closed / 0 total"
    # for the session). Entries older than RESULTS_MAX_AGE_HOURS are now
    # pruned before every win-rate/consecutive-loss evaluation, so only
    # recent, relevant trade outcomes can influence the breaker.
    RESULTS_MAX_AGE_HOURS   = 24    # এর চেয়ে পুরনো result গোনা হবে না

    def __init__(self, symbol: str = None, balance: float = 1000.0):
        self.symbol     = symbol
        self.balance    = balance
        self.state_path = self._state_path_for(symbol)
        self._state     = self._load_state()

    # ── State path resolution ─────────────────────────────────

    def _state_path_for(self, symbol: str) -> str:
        """Per-symbol state file so one pair's losing streak can't trip
        the breaker for every other pair. symbol=None keeps the legacy
        single shared file for backward compatibility."""
        if symbol is None:
            return CB_STATE_PATH
        safe = "".join(c for c in symbol.upper() if c.isalnum() or c in ("_", "-"))
        return os.path.join(CB_STATE_DIR, "circuit_breaker", f"{safe}.json")

    # ── Main Gate ──────────────────────────────────────────────

    def allow_trade(self) -> dict:
        """
        Trade নেওয়ার আগে এটা call করো।

        Returns:
            {
                "allowed": True/False,
                "mode":    "TRADING" | "PAUSED" | "LEARNING" | "COOLDOWN",
                "reason":  str,
                "stats":   dict,
            }
        """
        mode   = self._state.get("mode", "TRADING")
        reason = ""

        # Cooldown check
        if mode == "COOLDOWN":
            cooldown_until = self._state.get("cooldown_until")
            if cooldown_until:
                until_dt = datetime.fromisoformat(cooldown_until)
                if datetime.utcnow() < until_dt:
                    remaining = (until_dt - datetime.utcnow()).seconds // 60
                    return self._response(
                        False, "COOLDOWN",
                        f"Cooldown active — {remaining} min remaining. "
                        f"Resume after {until_dt.strftime('%H:%M UTC')}",
                    )
                else:
                    # Cooldown expired → resume
                    self._set_mode("TRADING", "Cooldown expired — resuming")

        if mode == "PAUSED":
            reason = self._state.get("pause_reason", "Circuit breaker active")
            return self._response(False, mode, reason)

        # BUG FIX: LEARNING mode now expires after LEARNING_HOURS instead of
        # blocking every cycle forever. Once expired we drop back to
        # TRADING so the next allow_trade() call re-evaluates win rate on
        # fresh data — if it's still bad, it will simply re-trigger.
        if mode == "LEARNING":
            learning_since = self._state.get("learning_since")
            if learning_since:
                since_dt = datetime.fromisoformat(learning_since)
                elapsed_hours = (datetime.utcnow() - since_dt).total_seconds() / 3600
                if elapsed_hours < self.LEARNING_HOURS:
                    reason = self._state.get("pause_reason", "Circuit breaker active")
                    return self._response(False, mode, reason)
                else:
                    self._set_mode("TRADING", "Learning mode expired — resuming for re-evaluation")
            else:
                # Legacy state file with no timestamp — stamp it now so it
                # doesn't stay stuck forever either.
                self._state["learning_since"] = datetime.utcnow().isoformat()
                self._save_state()
                reason = self._state.get("pause_reason", "Circuit breaker active")
                return self._response(False, mode, reason)

        # Real-time checks
        consec = self._state.get("consecutive_losses", 0)
        if consec >= self.MAX_CONSECUTIVE_LOSSES:
            self._trigger_pause(
                f"🛑 {consec} consecutive losses — entering cooldown"
            )
            return self._response(
                False, "PAUSED",
                f"{consec} consecutive losses hit threshold ({self.MAX_CONSECUTIVE_LOSSES})"
            )

        # 2026-07-23: weekly loss halt (see MAX_WEEKLY_LOSS_PCT comment above).
        self._reset_week_if_needed()
        weekly_loss_pct = self._state.get("weekly_loss_usd", 0) / self.balance * 100
        if weekly_loss_pct >= self.MAX_WEEKLY_LOSS_PCT:
            self._trigger_pause(
                f"🛑 Weekly loss limit reached: {weekly_loss_pct:.1f}% "
                f"(max {self.MAX_WEEKLY_LOSS_PCT}%)"
            )
            return self._response(
                False, "PAUSED",
                f"Weekly loss {weekly_loss_pct:.1f}% >= {self.MAX_WEEKLY_LOSS_PCT}% — halted for the week"
            )

        # BUG FIX: win rate must be computed over trades with a directional
        # outcome only. recent_results also contains "BREAKEVEN" entries
        # (PositionManager moves SL to entry once a trade reaches 50% of
        # TP distance — see broker/position_manager.py BREAKEVEN_TRIGGER_PC
        # — a protective, capital-preserving action, not a loss). The old
        # formula divided WIN count by len(recent), which *included*
        # BREAKEVEN in the denominator without ever counting it in the
        # numerator. A run of well-managed breakeven exits — exactly what
        # happens right after a trend stalls — could crater the reported
        # win rate toward 0% and trip LEARNING mode even though the
        # account hadn't actually lost money. Now scratch/breakeven trades
        # are excluded from the population, matching standard win-rate
        # convention: WIN / (WIN + LOSS).
        #
        # BUG FIX (staleness): decisive results are now drawn from
        # _recent_decisive(), which prunes entries older than
        # RESULTS_MAX_AGE_HOURS before slicing to LOOKBACK_TRADES — see
        # class docstring / RESULTS_MAX_AGE_HOURS comment above.
        decisive = self._recent_decisive()
        if len(decisive) >= 5:
            wr = decisive.count("WIN") / len(decisive) * 100
            if wr < self.MIN_WIN_RATE_THRESHOLD:
                self._trigger_learning(
                    f"Win rate dropped to {wr:.1f}% — entering learning mode"
                )
                return self._response(
                    False, "LEARNING",
                    f"Win rate {wr:.1f}% below minimum {self.MIN_WIN_RATE_THRESHOLD}%"
                )

        return self._response(True, "TRADING", "All checks passed")

    # ── Record Result ──────────────────────────────────────────

    def record_result(self, result: str, pnl_usd: float = 0.0):
        """
        Trade close হওয়ার পরে call করো।
        result: 'WIN' | 'LOSS' | 'BREAKEVEN'

        Day 81+ hotfix: sync daily_loss_usd with daily_risk.json so CB
        and RiskEngine always agree on the day's loss total.  Previously
        CB tracked its own running sum (incremental `+= abs(pnl_usd)`)
        while RiskEngine tracked a separate `total_loss_usd` in
        daily_risk.json — they drifted, and CB could trigger at $435
        while RiskEngine thought loss was only $128.

        Day 102+ hotfix: REAL-TIME THRESHOLD TRIGGER. Previously, the
        consecutive-loss / win-rate checks only ran inside allow_trade()
        — meaning if N trades closed in a single cycle (e.g. 6 pairs all
        hit SL on the same candle), record_result() would silently bump
        consecutive_losses to N without ever triggering the pause. The
        pause only kicked in on the NEXT cycle's allow_trade() call,
        letting an extra batch of losses slip through. Now we check the
        threshold immediately after incrementing, so the breaker trips
        the moment the limit is crossed — no extra cycle of damage.

        BUG FIX (staleness): each result is now stored with a UTC
        timestamp so _recent_decisive() can age it out after
        RESULTS_MAX_AGE_HOURS instead of it living forever in the last-50
        window. See class docstring for the bug this caused.
        """
        recent = self._state.get("recent_results", [])
        recent.append({
            "result": result,
            "ts":     datetime.utcnow().isoformat(),
        })
        self._state["recent_results"] = recent[-50:]  # শেষ ৫০টা রাখো

        if result == "LOSS":
            self._state["consecutive_losses"] = \
                self._state.get("consecutive_losses", 0) + 1
            # Day 81+ hotfix: do NOT incrementally track daily_loss_usd
            # here — instead sync from daily_risk.json (single source of
            # truth).  RiskEngine._save_daily() is called by the same
            # trade-close path and writes the authoritative total.
            self._sync_daily_loss_from_risk_engine()
            # 2026-07-23: weekly_loss_usd has no separate authoritative
            # file to sync from (unlike daily_risk.json for the daily
            # figure), so it's accumulated directly here. Reset on ISO
            # week rollover is handled by _reset_week_if_needed().
            self._reset_week_if_needed()
            self._state["weekly_loss_usd"] = \
                self._state.get("weekly_loss_usd", 0.0) + abs(pnl_usd)
            log.info(
                f"[CB{self._log_tag()}] LOSS recorded | consecutive={self._state['consecutive_losses']} "
                f"| daily_loss=${self._state['daily_loss_usd']:.2f}"
            )
        else:
            self._state["consecutive_losses"] = 0
            # Minor fix: this branch fires for BREAKEVEN too (anything
            # != "LOSS"), but used to always log "WIN recorded" — logging
            # the actual result avoids masking breakeven closes as wins
            # in the logs, which made this exact bug harder to spot.
            log.info(f"[CB{self._log_tag()}] {result} recorded | loss streak reset")

        # Daily loss check
        daily_loss_pct = self._state.get("daily_loss_usd", 0) / self.balance * 100
        if daily_loss_pct >= self.MAX_DAILY_LOSS_PCT:
            self._trigger_pause(
                f"Daily loss limit reached: {daily_loss_pct:.1f}%"
            )
            self._save_state()
            return  # already paused — skip redundant checks below

        # 2026-07-23 REAL-TIME WEEKLY-LOSS CHECK (mirrors the daily check above)
        weekly_loss_pct = self._state.get("weekly_loss_usd", 0) / self.balance * 100
        if weekly_loss_pct >= self.MAX_WEEKLY_LOSS_PCT:
            self._trigger_pause(
                f"Weekly loss limit reached: {weekly_loss_pct:.1f}%"
            )
            self._save_state()
            return  # already paused — skip redundant checks below

        # Day 102+ REAL-TIME CONSECUTIVE-LOSS CHECK
        # Trip immediately instead of waiting for the next allow_trade().
        consec = self._state.get("consecutive_losses", 0)
        if consec >= self.MAX_CONSECUTIVE_LOSSES:
            self._trigger_pause(
                f"🛑 {consec} consecutive losses — entering cooldown"
            )
            self._save_state()
            return

        # Day 102+ REAL-TIME WIN-RATE CHECK
        # If recent win rate drops below threshold, enter learning mode
        # immediately instead of waiting for the next cycle.
        # BUG FIX: same BREAKEVEN-exclusion + staleness-pruning logic as
        # allow_trade() — see the comments there. This real-time check has
        # to match that logic exactly, or the two paths could disagree on
        # whether the win rate has actually dropped below threshold.
        decisive_window = self._recent_decisive()
        if len(decisive_window) >= 5:
            wr = decisive_window.count("WIN") / len(decisive_window) * 100
            if wr < self.MIN_WIN_RATE_THRESHOLD:
                self._trigger_learning(
                    f"Win rate dropped to {wr:.1f}% — entering learning mode"
                )
                self._save_state()
                return

        self._save_state()

    # ── Manual Controls ────────────────────────────────────────

    def manual_resume(self, reason: str = "Manual override") -> dict:
        """Human manually resume করলে।"""
        self._set_mode("TRADING", reason)
        self._state["consecutive_losses"] = 0
        self._save_state()
        log.info(f"[CB{self._log_tag()}] Manually resumed: {reason}")
        return {"mode": "TRADING", "reason": reason}

    def force_learning_mode(self) -> dict:
        """Human manually learning mode-এ পাঠালে।"""
        self._trigger_learning("Manual learning mode activation")
        return {"mode": "LEARNING"}

    def reset_daily(self):
        """নতুন দিনে daily counters reset করো।"""
        today = date.today().isoformat()
        if self._state.get("date") != today:
            self._state["date"]           = today
            self._state["daily_loss_usd"] = 0.0
            log.info(f"[CB{self._log_tag()}] Daily reset — new trading day")
            self._save_state()

    # BUG FIX (2026-07-23 weekly-loss halt crash): allow_trade() and
    # record_result() were wired to call self._reset_week_if_needed()
    # when the MAX_WEEKLY_LOSS_PCT gate was added (see comment above),
    # but the method itself was never actually written — the weekly
    # rule was ported over from strict_risk_manager.py's reset logic
    # in name only. Every call raised AttributeError, which the outer
    # loop was catching and turning into a COOLDOWN, so the bot never
    # took a trade after boot. This mirrors reset_daily()'s pattern
    # exactly, but keyed on ISO year-week instead of the calendar date,
    # so weekly_loss_usd rolls over to 0 at the start of each new
    # ISO week (Monday) instead of accumulating forever.
    def _reset_week_if_needed(self):
        """নতুন সপ্তাহে weekly_loss_usd reset করো (ISO week, Mon–Sun)."""
        iso = date.today().isocalendar()
        current_week = f"{iso[0]}-W{iso[1]:02d}"
        if self._state.get("week") != current_week:
            self._state["week"]            = current_week
            self._state["weekly_loss_usd"] = 0.0
            log.info(f"[CB{self._log_tag()}] Weekly reset — new ISO week {current_week}")
            self._save_state()

    # ── Status ─────────────────────────────────────────────────

    def get_status(self) -> dict:
        # BUG FIX: exclude BREAKEVEN so the dashboard's "recent_win_rate"
        # actually reflects the same number allow_trade()/record_result()
        # gate on — see the fix comment in allow_trade(). Also uses the
        # same staleness-pruned window (_recent_decisive) so the dashboard
        # can't show a stale/expired win rate that the gate itself has
        # already stopped honoring.
        decisive = self._recent_decisive()
        wr       = (
            decisive.count("WIN") / len(decisive) * 100
            if decisive else 0
        )
        return {
            "mode":               self._state.get("mode", "TRADING"),
            "symbol":             self.symbol,
            "consecutive_losses": self._state.get("consecutive_losses", 0),
            "daily_loss_sync_stale": self._state.get("daily_loss_sync_stale", False),
            "recent_win_rate":    round(wr, 1),
            "daily_loss_usd":     self._state.get("daily_loss_usd", 0),
            "daily_loss_pct":     round(
                self._state.get("daily_loss_usd", 0) / self.balance * 100, 2
            ),
            "pause_reason":       self._state.get("pause_reason", ""),
            "total_trades":       len(self._state.get("recent_results", [])),
        }

    def print_status(self):
        s    = self.get_status()
        icon = {"TRADING": "🟢", "PAUSED": "🔴", "LEARNING": "🧠", "COOLDOWN": "⏳"}
        bar  = "═" * 46
        print(f"\n{bar}")
        label = f" ({s['symbol']})" if s.get("symbol") else ""
        print(f"  {icon.get(s['mode'], '⚪')}  CIRCUIT BREAKER STATUS{label}")
        print(bar)
        print(f"  Mode              : {s['mode']}")
        print(f"  Consecutive Loss  : {s['consecutive_losses']} / {self.MAX_CONSECUTIVE_LOSSES}")
        print(f"  Recent Win Rate   : {s['recent_win_rate']}%  (last {self.LOOKBACK_TRADES} trades, "
              f"< {self.RESULTS_MAX_AGE_HOURS}h old)")
        print(f"  Daily Loss        : ${s['daily_loss_usd']:.2f}  ({s['daily_loss_pct']}%)")
        if s["pause_reason"]:
            print(f"  Pause Reason      : {s['pause_reason']}")
        print(bar + "\n")

    # ── Internal ───────────────────────────────────────────────

    def _log_tag(self) -> str:
        return f":{self.symbol}" if self.symbol else ""

    def _recent_decisive(self) -> list:
        """Return WIN/LOSS results (BREAKEVEN excluded, stale entries
        excluded), most-recent-first-trimmed to LOOKBACK_TRADES.

        Centralizes the exact filtering logic that allow_trade(),
        record_result(), and get_status() must all agree on — previously
        each site duplicated a slightly different version of this, which
        is how the staleness bug went unnoticed for so long.
        """
        raw = self._state.get("recent_results", [])
        now = datetime.utcnow()
        fresh = []
        for entry in raw:
            # Backward compat: older state files stored plain strings
            # ("WIN"/"LOSS"/"BREAKEVEN") with no timestamp. We can't know
            # their age, so — matching the intent of this fix — treat
            # untimestamped legacy entries as stale/expired rather than
            # letting them silently keep influencing a fresh session.
            if not isinstance(entry, dict):
                continue
            ts = entry.get("ts")
            if not ts:
                continue
            try:
                age_hours = (now - datetime.fromisoformat(ts)).total_seconds() / 3600
            except ValueError:
                continue
            if age_hours <= self.RESULTS_MAX_AGE_HOURS:
                fresh.append(entry)

        windowed = fresh[-self.LOOKBACK_TRADES:]
        return [e["result"] for e in windowed if e.get("result") != "BREAKEVEN"]

    def _trigger_pause(self, reason: str):
        cooldown_until = (
            datetime.utcnow() + timedelta(hours=self.COOLDOWN_HOURS)
        ).isoformat()
        self._state["mode"]           = "COOLDOWN"
        self._state["pause_reason"]   = reason
        self._state["cooldown_until"] = cooldown_until
        self._save_state()
        log.warning(f"[CB{self._log_tag()}] ⛔ TRADING PAUSED: {reason}")
        log.warning(f"[CB{self._log_tag()}] Cooldown until: {cooldown_until}")

    def _trigger_learning(self, reason: str):
        self._state["mode"]           = "LEARNING"
        self._state["pause_reason"]   = reason
        self._state["learning_since"] = datetime.utcnow().isoformat()
        self._save_state()
        log.warning(f"[CB{self._log_tag()}] 🧠 LEARNING MODE: {reason}")

    def _set_mode(self, mode: str, reason: str):
        self._state["mode"]         = mode
        self._state["pause_reason"] = reason
        self._state.pop("cooldown_until", None)
        self._state.pop("learning_since", None)
        self._save_state()

    def _response(self, allowed: bool, mode: str, reason: str) -> dict:
        return {
            "allowed": allowed,
            "mode":    mode,
            "reason":  reason,
            "stats":   self.get_status(),
        }

    def _load_state(self) -> dict:
        """Load circuit breaker state from disk.

        CRITICAL FIX: Fail CLOSED on corruption, not open.
        Previously, any read error returned a fresh "TRADING" state —
        silently resetting consecutive_losses and daily_loss_usd to 0.
        A crash near the loss limit would reset the breaker, allowing more losses.
        Now: on corruption, return a PAUSED state that blocks new trades.
        """
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    state = json.load(f)
                # Daily reset check
                if state.get("date") != date.today().isoformat():
                    state["daily_loss_usd"] = 0.0
                    state["date"]           = date.today().isoformat()
                return state
            except (json.JSONDecodeError, KeyError) as e:
                # Corrupt JSON — fail CLOSED: PAUSE all trading
                log.critical(
                    f"circuit_breaker{self._log_tag()}: state file CORRUPT ({e}) — "
                    f"FAILING CLOSED (PAUSED mode). Manual intervention required."
                )
                return {
                    "mode":               "PAUSED",
                    "consecutive_losses": 99,  # blocks all trades
                    "daily_loss_usd":     999999,
                    "date":               date.today().isoformat(),
                    "recent_results":     [],
                    "pause_reason":       "STATE FILE CORRUPT — manual intervention required",
                    "_corrupt":           True,
                }
            except Exception as e:
                log.critical(
                    f"circuit_breaker{self._log_tag()}: state file read error ({e}) — "
                    f"FAILING CLOSED. Manual intervention required."
                )
                return {
                    "mode":               "PAUSED",
                    "consecutive_losses": 99,
                    "daily_loss_usd":     999999,
                    "date":               date.today().isoformat(),
                    "recent_results":     [],
                    "pause_reason":       f"STATE FILE ERROR: {e}",
                    "_corrupt":           True,
                }
        return {
            "mode":               "TRADING",
            "consecutive_losses": 0,
            "daily_loss_usd":     0.0,
            "date":               date.today().isoformat(),
            "recent_results":     [],
            "pause_reason":       "",
        }

    def _save_state(self):
        """CRITICAL FIX: Atomic write using temp file + os.replace().
        Prevents corruption from crash mid-write.
        """
        import tempfile
        dir_name = os.path.dirname(self.state_path) or "."
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_name, suffix=".tmp",
                prefix="cb_state_", delete=False
            ) as tmp_f:
                json.dump(self._state, tmp_f, indent=2)
                tmp_path = tmp_f.name
            os.replace(tmp_path, self.state_path)
        except Exception:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise

    def _sync_daily_loss_from_risk_engine(self):
        """Day 81+ hotfix: read daily_loss_usd from daily_risk.json
        (the RiskEngine's authoritative state file) so CB and RiskEngine
        always agree on the day's loss total.

        Previously CB tracked its own running sum which drifted from
        RiskEngine's total — CB could trigger at $435 while RiskEngine
        thought loss was only $128.  Now both read from the same file.
        """
        try:
            risk_path = "memory/daily_risk.json"
            with open(risk_path) as f:
                risk_state = json.load(f)
            # Only sync if the date matches today (RiskEngine resets on
            # date rollover; CB should follow the same convention).
            today = date.today().isoformat()
            if risk_state.get("date") == today:
                self._state["daily_loss_usd"] = float(
                    risk_state.get("total_loss_usd", 0)
                )
                self._state["daily_loss_sync_stale"] = False
        except Exception as e:
            # P1 fix (audit §5.3): this used to be log.debug() — silent at
            # normal log levels — meaning daily_loss_usd could go stale
            # with no warning above debug, undermining the exact
            # consistency guarantee this method exists to provide (CB and
            # RiskEngine must agree on the day's loss total). Escalated to
            # warning, and the staleness is now recorded in state so
            # `status()`/dashboards can surface it instead of silently
            # trusting a possibly-stale daily_loss_usd.
            self._state["daily_loss_sync_stale"] = True
            log.warning(
                f"[CB{self._log_tag()}] sync from daily_risk.json failed — daily_loss_usd may be "
                f"STALE and out of sync with RiskEngine: {e}"
            )