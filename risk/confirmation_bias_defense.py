"""risk/confirmation_bias_defense.py — Confirmation bias defense"""
from __future__ import annotations
from dataclasses import dataclass, field
from utils.logger import get_logger
log = get_logger("confirmation_bias_defense")

@dataclass
class DisconfirmationResult:
    signal: str = ""; confirming_factors: list = field(default_factory=list)
    disconfirming_factors: list = field(default_factory=list); disconfirming_count: int = 0
    blocked: bool = False; reason: str = ""

def check_disconfirming_evidence(signal, df=None, ind_ctx=None, market_bias=None, mtf_bias=None, max_disconfirming_allowed=3):
    result = DisconfirmationResult(signal=signal)
    ind_ctx = ind_ctx or {}; signal = signal.upper()
    if signal == "BUY":
        rsi = float(ind_ctx.get("rsi",50) or 50)
        if rsi >= 70: result.disconfirming_factors.append(f"RSI overbought ({rsi:.1f})")
        if "bearish" in str(ind_ctx.get("macd_cross","")).lower(): result.disconfirming_factors.append("MACD bearish cross")
        if "bear" in str(ind_ctx.get("trend","")).lower(): result.disconfirming_factors.append("Bearish trend")
        if market_bias and "bear" in str(market_bias).lower(): result.disconfirming_factors.append("Market bias bearish")
        if mtf_bias and "bear" in str(mtf_bias).lower(): result.disconfirming_factors.append("MTF bias bearish")
        if rsi <= 35: result.confirming_factors.append("RSI oversold")
    elif signal == "SELL":
        rsi = float(ind_ctx.get("rsi",50) or 50)
        if rsi <= 30: result.disconfirming_factors.append(f"RSI oversold ({rsi:.1f})")
        if "bullish" in str(ind_ctx.get("macd_cross","")).lower(): result.disconfirming_factors.append("MACD bullish cross")
        if "bull" in str(ind_ctx.get("trend","")).lower(): result.disconfirming_factors.append("Bullish trend")
        if market_bias and "bull" in str(market_bias).lower(): result.disconfirming_factors.append("Market bias bullish")
        if mtf_bias and "bull" in str(mtf_bias).lower(): result.disconfirming_factors.append("MTF bias bullish")
        if rsi >= 65: result.confirming_factors.append("RSI overbought")
    result.disconfirming_count = len(result.disconfirming_factors)
    if result.disconfirming_count >= max_disconfirming_allowed:
        result.blocked = True; result.reason = f"{result.disconfirming_count} disconfirming factors — BLOCKED"
    else: result.reason = f"{result.disconfirming_count} disconfirming, {len(result.confirming_factors)} confirming"
    if result.blocked: log.warning(f"[ConfirmationBias] BLOCKED {signal}: {result.reason}")
    return result
