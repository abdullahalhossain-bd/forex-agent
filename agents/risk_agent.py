# agents/risk_agent.py  —  Day 12 | Risk Management Agent
# Production-hardened: uses core.constants, consistent key naming
# Key naming convention matches RiskEngine: "lot" (not "lot_size"),
# "risk_pc" (not "risk_percent"), "risk_usd" (not "risk_amount_usd")

from utils.logger import get_logger
from core.constants import (
    get_pip_size,
    get_pip_value_usd,
    get_live_pip_value_per_lot,
    clean_symbol,
)

log = get_logger("risk_agent")

# Day 96+ hotfix: risk parameters were hardcoded in this module ($1000
# balance default, 1% max risk, 0.01–10.0 lot bounds) with no way to
# override them without editing code, and no way for them to stay in
# sync with risk/position_sizer.py's equivalents. Now sourced from
# config.py (single source of truth), with safe fallbacks — never
# silent — if config import fails for any reason.
try:
    from config import (
        INITIAL_BALANCE as _CFG_BALANCE,
        RISK_PER_TRADE as _CFG_RISK_PER_TRADE,
        MAX_RISK_PCT as _CFG_MAX_RISK_PCT,
        MIN_LOT as _CFG_MIN_LOT,
        MAX_LOT as _CFG_MAX_LOT,
    )
except Exception as e:
    _CFG_BALANCE = 1000.0
    _CFG_RISK_PER_TRADE = 0.01
    _CFG_MAX_RISK_PCT = 0.01
    _CFG_MIN_LOT = 0.01
    _CFG_MAX_LOT = 10.0
    log.warning(
        f"[risk_agent] config risk settings unavailable ({e}) — "
        f"falling back to hardcoded defaults (balance=${_CFG_BALANCE}, "
        f"risk={_CFG_RISK_PER_TRADE:.2%}, cap={_CFG_MAX_RISK_PCT:.2%}, "
        f"lot=[{_CFG_MIN_LOT}, {_CFG_MAX_LOT}])."
    )


class RiskAgent:
    """
    Risk Management Agent — calculates lot size, SL, TP from signal + ATR.
    Enforces the account's configured risk-per-trade rule and a hard
    risk cap that can never be exceeded regardless of lot rounding.
    Tracks daily loss limit.

    This is a simpler alternative to RiskEngine. The main pipeline
    uses RiskEngine by default; this agent is available for
    lightweight or per-pair risk calculations.
    """

    # Target risk % per trade — config-driven (config.RISK_PER_TRADE),
    # was hardcoded to 1.0.
    MAX_RISK_PERCENT    = float(_CFG_RISK_PER_TRADE) * 100
    # Absolute hard cap (fraction) — config-driven (config.MAX_RISK_PCT).
    # This is re-checked against the FINAL executable lot below (both
    # before and after the MIN_LOT floor is applied), so a lot-size
    # rounding/clamping step can never silently push real risk past it.
    MAX_RISK_PCT         = float(_CFG_MAX_RISK_PCT)
    # Broker-imposed lot bounds — config-driven (config.MIN_LOT / MAX_LOT),
    # were hardcoded to 0.01 / 10.0.
    MIN_LOT              = float(_CFG_MIN_LOT)
    MAX_LOT              = float(_CFG_MAX_LOT)
    # Day 81+ hotfix: load DAILY_LOSS_LIMIT from config (default 20.0).
    # Was hard-coded 3.0 — user wants 20.0.
    try:
        from config import DAILY_LOSS_LIMIT_PCT as _CFG_DLL
        DAILY_LOSS_LIMIT = float(_CFG_DLL)
    except Exception:
        DAILY_LOSS_LIMIT = 20.0
    # 2026-08-31: Fixed 1:2 R:R — SL=1.5 ATR, TP=3.0 ATR
    MIN_RR              = 2.0    # minimum risk:reward (fixed 1:2 policy)
    ATR_SL_MULTIPLIER   = 1.5   # SL = ATR * 1.5 (~25 pips on EURUSD H1)

    def __init__(self, account_balance: float = None):
        # config-driven default (config.INITIAL_BALANCE) instead of a
        # hardcoded $1000 — callers that pass account_balance explicitly
        # are unaffected.
        self.balance       = account_balance if account_balance is not None else _CFG_BALANCE
        self.daily_loss_pc = 0.0

    def calculate(
        self,
        signal:   str,
        entry:    float,
        ind_ctx:  dict,
        regime:   dict,
        symbol:   str = "EURUSD",
    ) -> dict:
        """Calculate full risk parameters from signal + entry + ATR."""

        if signal == "NO TRADE":
            return self._no_trade("Signal is NO TRADE")

        if self.daily_loss_pc >= self.DAILY_LOSS_LIMIT:
            return self._no_trade(
                f"Daily loss limit hit ({self.daily_loss_pc:.1f}%)"
            )

        # FIX (review): previously defaulted missing ATR to a hardcoded
        # 0.0005 — a plausible value for a 4/5-digit pair like EURUSD, but
        # wrong by ~100x for JPY pairs and by 1000x+ for metals/indices/
        # crypto. A silent wrong-scale guess produces a too-tight SL and
        # an oversized lot for the real risk taken — exactly backwards
        # for a risk-management module. Fail safe instead: if ATR is
        # missing or non-positive, refuse to guess and reject the trade
        # with an explicit reason rather than silently mis-sizing it.
        atr = ind_ctx.get("atr")
        if atr is None or atr <= 0:
            return self._no_trade(
                f"ATR unavailable or invalid for {symbol} "
                f"(atr={atr!r}) — refusing to guess a default SL distance"
            )
        csym = clean_symbol(symbol)
        pip = get_pip_size(csym)
        # Live MT5 pip value — was the static USD table (get_pip_value_usd)
        # unconditionally, which is only correct on a real-money, USD,
        # Standard account and is silently ~100x wrong on a Cent account.
        # get_live_pip_value_per_lot() reads it straight from the broker's
        # own symbol_info() for the exact (suffixed) symbol traded, and
        # falls back to the static table itself (with a loud warning) only
        # if no MT5 connection is available — so this is never worse than
        # the previous behavior, only more correct when MT5 is live.
        pip_val_std = get_live_pip_value_per_lot(csym)

        # FIX (review): pip_val_std is used as a divisor below
        # (risk_amount / (sl_pips * pip_val_std)). Previously there was
        # no guard for pip_val_std == 0 — an unrecognized/mis-cleaned
        # symbol returning 0 from get_pip_value_usd() would raise an
        # unhandled ZeroDivisionError mid-cycle instead of failing safe.
        if pip_val_std <= 0:
            return self._no_trade(
                f"Invalid pip value for {symbol} (pip_val_std={pip_val_std!r}) "
                f"— cannot size position, refusing to guess"
            )

        # Regime-based SL multiplier
        volatility = regime.get("volatility", "NORMAL")
        sl_mult = {
            "LOW_VOLATILITY":  1.2,
            "NORMAL":          self.ATR_SL_MULTIPLIER,
            "HIGH_VOLATILITY": 2.0,
        }.get(volatility, self.ATR_SL_MULTIPLIER)

        sl_distance = round(atr * sl_mult, 5)
        sl_pips     = round(sl_distance / pip) if pip > 0 else 10

        # Guard: if sl_pips is 0 or sl_distance is too small, use defaults
        if sl_pips < 1:
            sl_pips = 10
            sl_distance = sl_pips * pip

        # TP = SL * min RR
        tp_distance = round(sl_distance * self.MIN_RR, 5)
        tp_pips     = round(tp_distance / pip)

        # SL/TP price levels
        if signal == "BUY":
            sl_price = round(entry - sl_distance, 5)
            tp_price = round(entry + tp_distance, 5)
        else:   # SELL
            sl_price = round(entry + sl_distance, 5)
            tp_price = round(entry - tp_distance, 5)

        # Lot size — config-driven risk rule (config.RISK_PER_TRADE),
        # was hardcoded to a flat 1%.
        risk_amount = self.balance * (self.MAX_RISK_PERCENT / 100)
        lot_raw     = risk_amount / (sl_pips * pip_val_std) if sl_pips > 0 else 0

        # ── Pre-floor safety gate ───────────────────────────────────
        # Mirrors risk/position_sizer.py's tiny-balance gate. MIN_LOT is
        # a broker-imposed FLOOR — you cannot submit a smaller order. On
        # a small enough balance, flooring lot_raw up to MIN_LOT can
        # silently push the REAL risk taken past MAX_RISK_PCT (the risk
        # the account can actually afford is below what even the
        # smallest tradeable lot risks). Check the floor's OWN risk
        # before clamping, and refuse rather than silently over-risk.
        min_lot_risk = self.MIN_LOT * sl_pips * pip_val_std
        min_lot_risk_pct = (min_lot_risk / self.balance) if self.balance > 0 else float("inf")
        if min_lot_risk_pct > self.MAX_RISK_PCT:
            return self._no_trade(
                f"Balance too small for {symbol}: even the broker minimum "
                f"lot ({self.MIN_LOT}) risks ${min_lot_risk:.2f} "
                f"({min_lot_risk_pct:.2%}) of the ${self.balance:.2f} "
                f"balance, above the {self.MAX_RISK_PCT:.2%} hard cap "
                f"(sl_pips={sl_pips:.1f})"
            )

        lot = round(max(self.MIN_LOT, min(lot_raw, self.MAX_LOT)), 2)

        # ── Post-round safety check ─────────────────────────────────
        # Re-verify the ACTUAL risk of the FINAL, rounded/clamped
        # executable lot — rounding to 2dp and clamping to [MIN_LOT,
        # MAX_LOT] can move the real risk % away from what MAX_RISK_PERCENT
        # intended. Reject rather than silently trade over the hard cap.
        actual_risk = lot * sl_pips * pip_val_std
        actual_risk_pct = (actual_risk / self.balance) if self.balance > 0 else float("inf")
        if actual_risk_pct > self.MAX_RISK_PCT:
            return self._no_trade(
                f"Final lot {lot} for {symbol} risks ${actual_risk:.2f} "
                f"({actual_risk_pct:.2%}) of the ${self.balance:.2f} "
                f"balance, above the {self.MAX_RISK_PCT:.2%} hard cap"
            )

        rr_ratio    = round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0

        result = {
            "approved":       True,
            "signal":         signal,
            "entry":          entry,
            "sl_price":       sl_price,
            "tp_price":       tp_price,
            "sl_pips":        sl_pips,
            "tp_pips":        tp_pips,
            "lot":            lot,         # Consistent key name with RiskEngine
            "lot_size":       lot,         # Backward compat alias
            "risk_pc":        round(actual_risk_pct * 100, 4),  # ACTUAL risk % of final lot
            "risk_percent":   round(actual_risk_pct * 100, 4),  # Backward compat alias
            "risk_usd":       round(actual_risk, 2),   # ACTUAL risk $ of final lot
            "risk_amount_usd": round(actual_risk, 2),  # Backward compat alias
            "rr_ratio":       rr_ratio,
            "balance":        self.balance,
            "reject_reason":  None,
        }

        # Final RR check
        if rr_ratio < self.MIN_RR:
            result["approved"]      = False
            result["reject_reason"] = f"RR {rr_ratio} < min {self.MIN_RR}"

        log.info(
            f"[RiskAgent] {signal} | Entry: {entry} | "
            f"SL: {sl_price} ({sl_pips}p) | "
            f"TP: {tp_price} ({tp_pips}p) | "
            f"Lot: {lot} | RR: {rr_ratio} | "
            f"Approved: {result['approved']}"
        )
        return result

    def _no_trade(self, reason: str) -> dict:
        log.info(f"[RiskAgent] No trade — {reason}")
        return {
            "approved":        False,
            "signal":          "NO TRADE",
            "reject_reason":   reason,
            "entry":           None,
            "sl_price":        None,
            "tp_price":        None,
            "lot":             0,
            "lot_size":        0,
            "sl_pips":         0,
            "tp_pips":         0,
            "rr_ratio":        0,
            "risk_usd":        0,
            "risk_amount_usd": 0,
            # FIX (review): the approved-path dict in calculate() includes
            # "balance", "risk_pc", and "risk_percent" — this dict didn't,
            # so any downstream code expecting a uniform schema regardless
            # of approval status would KeyError on the no-trade path.
            "balance":         self.balance,
            "risk_pc":         self.MAX_RISK_PERCENT,
            "risk_percent":    self.MAX_RISK_PERCENT,
        }

    def print_summary(self, result: dict) -> None:
        bar  = "=" * 44
        icon = "[OK]" if result["approved"] else "[REJECT]"
        log.info(bar)
        log.info(f"  {icon}  RISK AGENT")
        log.info(bar)
        log.info(f"  Approved    : {result['approved']}")
        if result.get("reject_reason"):
            log.info(f"  Rejected    : {result['reject_reason']}")
        if result["approved"]:
            log.info(f"  Signal      : {result['signal']}")
            log.info(f"  Entry       : {result['entry']}")
            log.info(f"  SL          : {result['sl_price']}  ({result['sl_pips']} pips)")
            log.info(f"  TP          : {result['tp_price']}  ({result['tp_pips']} pips)")
            log.info(f"  Lot         : {result.get('lot', result.get('lot_size', 0))}")
            log.info(f"  Risk        : {result.get('risk_pc', result.get('risk_percent', 0))}%  (${result.get('risk_usd', result.get('risk_amount_usd', 0))})")
            log.info(f"  R:R         : 1:{result['rr_ratio']}")
        log.info(bar)

    def get_ai_context(self, result: dict) -> dict:
        return {
            "risk_approved":  result["approved"],
            "risk_lot":       result.get("lot", result.get("lot_size", 0)),
            "risk_sl_pips":   result.get("sl_pips", 0),
            "risk_tp_pips":   result.get("tp_pips", 0),
            "risk_rr":        result.get("rr_ratio", 0),
            "risk_reject":    result.get("reject_reason"),
        }