"""Deterministic Devil's Advocate surrogate for backtest.

In live trading, Devil's Advocate calls an LLM to adversarially review
approved trades. In backtest, we cannot call LLM APIs (cost + non-determinism).
This module provides a deterministic, rule-based surrogate that uses
historical features (NO future data) to generate counter-evidence.

Output: DA_APPROVE / DA_REJECT / DA_UNCERTAIN — same semantics as live.

This is NOT a substitute for the live LLM DA. It is a documented parity
gap — backtest WR is an upper bound on live WR. The surrogate catches
OBVIOUS red flags (counter-trend, over-extended, near S/R, high spread,
low liquidity) that the live LLM would also catch, but it cannot catch
subtle narrative/contextual red flags that require language understanding.

Counter-evidence factors (all derived from historical data available at
the bar timestamp — NO look-ahead):

1. TREND CONTRADICTION: signal direction vs H4/H1 trend alignment
2. MTF DISAGREEMENT: H4 vs H1 vs M15 trend conflict
3. OVEREXTENSION: price far from SMA (e.g. > 2 std dev)
4. NEARBY RESISTANCE/SUPPORT: signal direction into nearby S/R zone
5. ABNORMAL VOLATILITY: ATR > 2x historical average (news/event risk)
6. SPREAD CONDITION: spread > 1.5x symbol default (poor execution)
7. LIQUIDITY CONDITION: tick_volume < 50% of 20-bar average (thin market)
8. MOMENTUM EXHAUSTION: RSI extreme (>70 for BUY, <30 for SELL) — reversal risk

Decision logic:
- 0 counter-evidence factors → DA_APPROVE
- 1-2 factors → DA_UNCERTAIN (resolve to REJECT — conservative)
- 3+ factors → DA_REJECT

This matches the live DA's "HARD BLOCK on 3+ disconfirming factors" semantics
documented in risk/confirmation_bias_defense.py.
"""
import os
import sys
import logging
from typing import Dict, Any

log = logging.getLogger("da_surrogate")


class DevilAdvocateSurrogate:
    """Deterministic, rule-based Devil's Advocate for backtest."""

    def __init__(self, symbol: str = "", timeframe: str = "H1"):
        self.symbol = symbol
        self.timeframe = timeframe

    def review(self, signal: str, market_out: Dict[str, Any],
               analysis_out: Dict[str, Any], risk_out: Dict[str, Any],
               decision_out: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a deterministic DA verdict.

        Args:
            signal: "BUY" / "SELL"
            market_out: from MarketAgent.run()
            analysis_out: from AnalysisAgent.run()
            risk_out: from RiskEngine.check()
            decision_out: from DecisionAgent.decide()

        Returns:
            {
                "decision": "TAKE" | "REJECT" | "UNCERTAIN",
                "confidence": float (0-1),
                "counter_evidence": list[str],
                "reasoning": str,
                "surrogate": True,  # flag that this is NOT the live LLM DA
            }
        """
        if signal not in ("BUY", "SELL"):
            return {
                "decision": "TAKE",  # not a reviewable trade
                "confidence": 1.0,
                "counter_evidence": [],
                "reasoning": "Not a reviewable trade (signal not BUY/SELL)",
                "surrogate": True,
            }

        counter_evidence = []

        # 1. Trend contradiction
        mtf_bias = market_out.get("mtf_bias", "")
        if isinstance(mtf_bias, dict):
            mtf_bias = mtf_bias.get("bias", "")
        mtf_bias = str(mtf_bias).lower()
        if signal == "BUY" and "bearish" in mtf_bias:
            counter_evidence.append(f"MTF bias is bearish ({mtf_bias}) — counter-trend BUY")
        elif signal == "SELL" and "bullish" in mtf_bias:
            counter_evidence.append(f"MTF bias is bullish ({mtf_bias}) — counter-trend SELL")

        # 2. MTF disagreement (from analysis_out)
        mtf_ctx = analysis_out.get("mtf_structure_ctx", {}) if analysis_out else {}
        h4_trend = str(mtf_ctx.get("h4_trend", "")).lower()
        h1_trend = str(mtf_ctx.get("h1_trend", "")).lower()
        if h4_trend and h1_trend and h4_trend != h1_trend:
            counter_evidence.append(f"MTF disagreement: H4={h4_trend} vs H1={h1_trend}")

        # 3. Overextension (price vs SMA)
        ind_ctx = market_out.get("ind_ctx", {}) if market_out else {}
        price = float(ind_ctx.get("price", 0) or 0)
        sma = float(ind_ctx.get("sma_20", 0) or 0)
        if price > 0 and sma > 0:
            deviation_pct = abs(price - sma) / sma * 100
            if deviation_pct > 1.0:  # >1% from SMA — overextended
                counter_evidence.append(f"Price {deviation_pct:.2f}% from SMA20 — overextended")

        # 4. Nearby S/R
        sr_ctx = analysis_out.get("sr_ctx", {}) if analysis_out else {}
        dist_to_support = sr_ctx.get("dist_to_support_pips")
        dist_to_resistance = sr_ctx.get("dist_to_resistance_pips")
        if signal == "BUY" and dist_to_resistance is not None and dist_to_resistance < 20:
            counter_evidence.append(f"BUY near resistance ({dist_to_resistance:.0f} pips) — limited upside")
        elif signal == "SELL" and dist_to_support is not None and dist_to_support < 20:
            counter_evidence.append(f"SELL near support ({dist_to_support:.0f} pips) — limited downside")

        # 5. Abnormal volatility (ATR > 2x average)
        regime = market_out.get("regime", {}) if market_out else {}
        atr = float(regime.get("atr", 0) or 0)
        atr_avg = float(regime.get("atr_avg", 0) or 0)
        if atr > 0 and atr_avg > 0 and atr > atr_avg * 2:
            counter_evidence.append(f"Abnormal volatility: ATR {atr:.5f} > 2x avg {atr_avg:.5f}")

        # 6. Spread condition
        spread_pips = float(market_out.get("spread_pips", 0) or 0) if market_out else 0
        symbol_defaults = {"EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 1.5, "AUDUSD": 1.8,
                          "USDCAD": 2.0, "USDCHF": 2.0, "NZDUSD": 2.0, "XAUUSD": 25.0}
        default_spread = symbol_defaults.get(self.symbol, 2.0)
        if spread_pips > default_spread * 1.5:
            counter_evidence.append(f"High spread {spread_pips:.1f} pips > 1.5x default {default_spread}")

        # 7. Liquidity condition
        df = market_out.get("df") if market_out else None
        if df is not None and len(df) >= 20:
            try:
                vol = df["volume"].tail(20)
                current_vol = float(vol.iloc[-1])
                avg_vol = float(vol.mean())
                if avg_vol > 0 and current_vol < avg_vol * 0.5:
                    counter_evidence.append(f"Low liquidity: volume {current_vol:.0f} < 50% avg {avg_vol:.0f}")
            except Exception:
                pass

        # 8. Momentum exhaustion (RSI extreme)
        rsi = float(ind_ctx.get("rsi", 50) or 50)
        if signal == "BUY" and rsi > 70:
            counter_evidence.append(f"RSI {rsi:.0f} > 70 — overbought, reversal risk for BUY")
        elif signal == "SELL" and rsi < 30:
            counter_evidence.append(f"RSI {rsi:.0f} < 30 — oversold, reversal risk for SELL")

        # Decision logic
        n_factors = len(counter_evidence)
        if n_factors == 0:
            decision = "TAKE"
            confidence = 0.8
            reasoning = "No counter-evidence factors detected"
        elif n_factors <= 2:
            decision = "UNCERTAIN"
            confidence = 0.4
            reasoning = f"{n_factors} counter-evidence factor(s) — resolving to REJECT (conservative)"
        else:
            decision = "REJECT"
            confidence = 0.2
            reasoning = f"{n_factors} counter-evidence factors — HARD BLOCK"

        # UNCERTAIN resolves to REJECT (matches live DA's _resolve_uncertain fail-closed behavior)
        final_decision = "REJECT" if decision in ("UNCERTAIN", "REJECT") else "TAKE"

        return {
            "decision": final_decision,
            "raw_decision": decision,
            "confidence": confidence,
            "counter_evidence": counter_evidence,
            "n_factors": n_factors,
            "reasoning": reasoning,
            "surrogate": True,
        }
