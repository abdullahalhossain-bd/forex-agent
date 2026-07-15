# agents/risk_agent.py  —  Day 12 | Risk Management Agent
# Production-hardened: uses core.constants, consistent key naming
# Key naming convention matches RiskEngine: "lot" (not "lot_size"),
# "risk_pc" (not "risk_percent"), "risk_usd" (not "risk_amount_usd")

from utils.logger import get_logger
from core.constants import get_pip_size, get_pip_value_usd, clean_symbol

log = get_logger("risk_agent")


class RiskAgent:
    """
    Risk Management Agent — calculates lot size, SL, TP from signal + ATR.
    Enforces 1% risk rule. Tracks daily loss limit.

    This is a simpler alternative to RiskEngine. The main pipeline
    uses RiskEngine by default; this agent is available for
    lightweight or per-pair risk calculations.
    """

    MAX_RISK_PERCENT    = 1.0    # max 1% account risk per trade
    # Day 81+ hotfix: load DAILY_LOSS_LIMIT from config (default 20.0).
    # Was hard-coded 3.0 — user wants 20.0.
    try:
        from config import DAILY_LOSS_LIMIT_PCT as _CFG_DLL
        DAILY_LOSS_LIMIT = float(_CFG_DLL)
    except Exception:
        DAILY_LOSS_LIMIT = 20.0
    MIN_RR              = 1.5    # minimum risk:reward ratio
    ATR_SL_MULTIPLIER   = 1.5   # SL = ATR * 1.5

    def __init__(self, account_balance: float = 1000.0):
        self.balance       = account_balance
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

        atr = ind_ctx.get("atr", 0.0005)
        csym = clean_symbol(symbol)
        pip = get_pip_size(csym)
        pip_val_std = get_pip_value_usd(csym)

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

        # Lot size — 1% risk rule
        risk_amount = self.balance * (self.MAX_RISK_PERCENT / 100)
        lot_raw     = risk_amount / (sl_pips * pip_val_std) if sl_pips > 0 else 0
        lot         = round(max(0.01, min(lot_raw, 10.0)), 2)

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
            "risk_pc":        self.MAX_RISK_PERCENT,  # Consistent key name
            "risk_percent":   self.MAX_RISK_PERCENT,  # Backward compat alias
            "risk_usd":       round(risk_amount, 2),   # Consistent key name
            "risk_amount_usd": round(risk_amount, 2),  # Backward compat alias
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
