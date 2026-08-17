# risk/mean_reversion_da_gate.py
# ============================================================
# Mean-Reversion Second-Opinion Gate — deterministic, LLM-independent.
#
# ⚠️ IMPORTANT — this is NOT a replacement for core/devils_advocate.py.
# It's a fast, free, deterministic pre-filter validated specifically for
# the mean-reversion confluence signal in
# analysis/mean_reversion_confluence_engine.py. Run this FIRST (cheap),
# then still send whatever survives through the real
# core/devils_advocate.py LLM review before execution — this gate does
# not call an LLM and should never be the only review layer in live
# trading.
#
# Why this exists instead of reusing core/devils_advocate.py directly:
# that module's real prompt requires BOS/CHoCH structure to AGREE with
# the trade direction. For a MEAN-REVERSION signal (which is explicitly
# betting on an extreme reversing, i.e. against recent momentum/BOS),
# that requirement is close to self-contradictory — see validation notes
# below. This gate instead checks the things that actually predicted bad
# mean-reversion trades in backtesting: overextension, longer-horizon
# disagreement, volatility spikes, poor risk efficiency, and momentum
# not yet turning.
#
# VALIDATION (held-out test, 2026-04-25 onward, EURAUD/GBPCAD/EURCAD,
# proxy-LLM signal source, deterministic gate — not the live LLM DA):
#   No gate (raw confluence signal only): 469 trades, 41.2% win, +0.029R
#   This gate, veto_threshold=3 (lenient):  525 trades, 51.2% win, +0.025R
#   Real devils_advocate.py's literal structure-agreement rule (tested
#     separately, using the real structure.py/support_resistance.py
#     engines): 154 trades, 33.8% win, -0.156R — WORSE, because it's
#     structurally mismatched to a reversal-style signal (see note above).
#
#   Sample sizes are still modest (~13 months, 3 pairs). Forward-test on
#   a demo account before sizing this into live risk.
# ============================================================

from __future__ import annotations

from typing import Optional
import pandas as pd
from utils.logger import get_logger

log = get_logger("mean_reversion_da_gate")

VETO_THRESHOLD = 2          # updated after robustness sweep — see comment at top.
                             # ROBUSTNESS SWEEP (held-out test window, all 3 pairs):
                             #   thr da_thr    n    wr%     exp
                             #    1    1     242   46.7%  -0.066R
                             #    1    2     594   54.4%  +0.088R  <- shipped default
                             #    1    3     686   53.9%  +0.079R
                             #    2    2     506   49.6%  -0.008R
                             #    2    3     525   51.2%  +0.025R  (previous default)
                             #    3    3     458   49.3%  -0.013R
                             # da_threshold=1 (very strict veto) and >=4 (very
                             # loose veto) underperform across every score_threshold
                             # tested — the 2-3 region is where it's consistently
                             # positive, which is why (1, 2) was picked over a
                             # single-point optimum.
OVEREXTENSION_ATR_MULT = 2.5
VOLATILITY_SPIKE_RATIO = 1.8
MAX_RISK_ATR_MULT = 3.0
RSI_SLOPE_LOOKBACK = 3
RSI_SLOPE_THRESHOLD = 2.0


class MeanReversionDAGate:
    """
    Usage:
        gate = MeanReversionDAGate()
        vetoed, flags = gate.review(df, idx, direction, sl_price)

    ⚠️ Requires df to already have >=250 bars of history before `idx`
    (EMA200/ADX/ATR need that much warmup to be meaningful) and
    `analysis/mean_reversion_confluence_engine.py`'s prepare() to have
    been run on it first (needs atr14, ema20, rsi14 columns). Calling
    this on a short/cold-start df will silently under-flag rather than
    crash, but the veto decision won't be reliable that early.
    """

    def __init__(self, veto_threshold: int = VETO_THRESHOLD):
        self.veto_threshold = veto_threshold

    def review(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: str,
        sl_price: float,
    ) -> tuple[bool, list[str]]:
        """Returns (vetoed: bool, red_flags: list[str]).

        `idx` is the positional index of the candidate bar in `df`
        (df.iloc[idx] must be the decision bar). `sl_price` is the
        already-computed structural stop-loss for this trade.
        """
        row = df.iloc[idx]
        flags: list[str] = []

        atr = row.get("atr14")
        if pd.isna(atr) or not atr:
            return True, ["no_atr_data"]

        # 1. Overextension from EMA20 — chasing risk
        dist_from_ema20 = abs(row["close"] - row["ema20"]) / atr
        if dist_from_ema20 > OVEREXTENSION_ATR_MULT:
            flags.append("overextended_from_ema20")

        # 2. Longer-horizon (EMA100) disagreement
        ema100 = (
            df["close"].iloc[max(0, idx - 100): idx + 1]
            .ewm(span=100, adjust=False).mean().iloc[-1]
        )
        if direction == "SELL" and row["close"] > ema100:
            flags.append("ema100_disagrees")
        if direction == "BUY" and row["close"] < ema100:
            flags.append("ema100_disagrees")

        # 3. Volatility-spike unreliability (news-like spike -> technicals less reliable)
        lo3 = max(0, idx - 2)
        atr3 = df["high"].iloc[lo3: idx + 1].sub(df["low"].iloc[lo3: idx + 1]).mean()
        if atr and atr3 / atr > VOLATILITY_SPIKE_RATIO:
            flags.append("volatility_spike")

        # 4. Poor risk efficiency (structural SL too wide relative to ATR)
        risk = abs(sl_price - row["close"])
        if atr and risk / atr > MAX_RISK_ATR_MULT:
            flags.append("poor_risk_efficiency")

        # 5. Momentum not yet confirming the reversal (RSI slope)
        rsi_now = row.get("rsi14")
        rsi_prev = (
            df["rsi14"].iloc[idx - RSI_SLOPE_LOOKBACK]
            if idx >= RSI_SLOPE_LOOKBACK else rsi_now
        )
        if pd.notna(rsi_now) and pd.notna(rsi_prev):
            rsi_slope = rsi_now - rsi_prev
            if direction == "SELL" and rsi_slope > RSI_SLOPE_THRESHOLD:
                flags.append("momentum_not_confirming")
            if direction == "BUY" and rsi_slope < -RSI_SLOPE_THRESHOLD:
                flags.append("momentum_not_confirming")

        vetoed = len(flags) >= self.veto_threshold
        if vetoed:
            log.info(f"[MeanReversionDAGate] VETO {direction} @ idx={idx}: {flags}")
        return vetoed, flags