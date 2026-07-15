"""
analysis/final_frontier.py — Final Research Domain Implementations
====================================================================
13 missing research domains from the ultimate institutional AI roadmap:

3. Market Ecology — participant type analysis
4. Strategy Decay — edge erosion tracking
5. Alpha Attribution — profit source breakdown
6. Capacity Analysis — strategy scalability
7. Transaction Cost Analysis (TCA) — full cost modeling
8. Latency Analysis — execution speed tracking
9. Data Provenance — source quality tracking
10. Market Calendar Intelligence — holidays, expiries, rebalancing
14. Market Maker Behavior — inventory/quote modeling
15. Regime Probability — soft regime distribution
16. Causal Inference — Granger causality
17. Digital Twin — virtual account simulation
19. Failure Analysis — loss classification
20. Edge Preservation — crowded strategy detection
21. AI Governance — decision logging + rollback

USAGE:
    from analysis.final_frontier import (
        MarketEcology, StrategyDecayTracker, AlphaAttribution,
        TransactionCostAnalyzer, LatencyAnalyzer, DataProvenance,
        MarketCalendar, RegimeProbability, CausalInference,
        DigitalTwin, FailureAnalyzer, EdgePreservation,
    )

WIRING STATUS (institutional review, item #6):
    As of this review, no other module in analysis/ imports from this file.
    These classes are self-contained utilities, not (yet) part of the live
    decision pipeline (dat_framework / decision_bridge). Treat this as a
    reference/utility library until something explicitly calls into it —
    do not assume any of the analysis here (regime probability, cost
    modeling, etc.) is currently influencing live trade decisions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path
from utils.logger import get_logger
from utils.pip_utils import pip_size, units_per_lot

log = get_logger("final_frontier")


# ════════════════════════════════════════════════════════════════════
# 3. MARKET ECOLOGY
# ════════════════════════════════════════════════════════════════════

class MarketEcology:
    """Analyze who is participating in the market and their likely behavior.

    Identifies market participant types based on order flow patterns:
    - Retail: small orders, emotional, trend-following
    - Banks: large institutional orders, session-based
    - HFT: ultra-fast, small spreads, high frequency
    - Market Makers: two-sided quotes, inventory management
    """

    @staticmethod
    def classify_participants(df: pd.DataFrame, volume_col: str = "volume") -> dict:
        """Classify market participants based on volume and price patterns.

        Returns:
            {"dominant_participant": str, "retail_pct": float, ...}
        """
        if df is None or len(df) < 20:
            return {"dominant_participant": "UNKNOWN", "reason": "Insufficient data"}

        vol = df[volume_col].values if volume_col in df.columns else np.ones(len(df))
        ranges = (df["high"] - df["low"]).values

        # Retail signature: small volume, high range/volume ratio (emotional)
        avg_vol = np.mean(vol)
        small_vol_bars = np.sum(vol < avg_vol * 0.5)
        retail_pct = small_vol_bars / len(vol)

        # Institutional signature: large volume spikes
        large_vol_bars = np.sum(vol > avg_vol * 2)
        institutional_pct = large_vol_bars / len(vol)

        # HFT signature: many small bars with tiny ranges
        tiny_range = np.sum(ranges < np.median(ranges) * 0.3)
        hft_pct = tiny_range / len(ranges)

        # Determine dominant
        if institutional_pct > 0.15:
            dominant = "INSTITUTIONAL"
            reason = f"{institutional_pct:.0%} bars have 2×+ average volume — institutional activity"
        elif retail_pct > 0.5:
            dominant = "RETAIL"
            reason = f"{retail_pct:.0%} bars have low volume — retail-dominated market"
        elif hft_pct > 0.3:
            dominant = "HFT"
            reason = f"{hft_pct:.0%} bars have tiny ranges — HFT activity"
        else:
            dominant = "MIXED"
            reason = "No single participant type dominant"

        return {
            "dominant_participant": dominant,
            "retail_pct": round(retail_pct, 2),
            "institutional_pct": round(institutional_pct, 2),
            "hft_pct": round(hft_pct, 2),
            "avg_volume": round(float(avg_vol), 1),
            "reason": reason,
        }


# ════════════════════════════════════════════════════════════════════
# 4. STRATEGY DECAY TRACKER
# ════════════════════════════════════════════════════════════════════

class StrategyDecayTracker:
    """Track strategy performance decay over time.

    Every strategy has a lifespan. This module detects when a strategy
    is losing its edge by tracking:
    - Win rate decline
    - Sharpe ratio decline
    - Profit factor decline
    """

    DECAY_THRESHOLD = 0.20  # 20% decline = significant

    def __init__(self):
        self._history: Dict[str, List[dict]] = defaultdict(list)

    def record(self, strategy: str, win_rate: float, sharpe: float,
               profit_factor: float, timestamp: str = None):
        """Record a performance snapshot for a strategy."""
        entry = {
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "win_rate": win_rate,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
        }
        self._history[strategy].append(entry)

    def check_decay(self, strategy: str, window: int = 20) -> dict:
        """Check if a strategy is decaying.

        Returns:
            {"decaying": bool, "wr_decline": float, "sharpe_decline": float, ...}
        """
        history = self._history.get(strategy, [])
        if len(history) < window * 2:
            return {"decaying": False, "reason": "Insufficient history"}

        recent = history[-window:]
        prior = history[-window * 2:-window]

        recent_wr = np.mean([e["win_rate"] for e in recent])
        prior_wr = np.mean([e["win_rate"] for e in prior])
        wr_decline = (prior_wr - recent_wr) / prior_wr if prior_wr > 0 else 0

        recent_sharpe = np.mean([e["sharpe"] for e in recent])
        prior_sharpe = np.mean([e["sharpe"] for e in prior])
        sharpe_decline = (prior_sharpe - recent_sharpe) / abs(prior_sharpe) if prior_sharpe != 0 else 0

        decaying = wr_decline > self.DECAY_THRESHOLD or sharpe_decline > self.DECAY_THRESHOLD

        return {
            "decaying": decaying,
            "wr_decline": round(wr_decline, 3),
            "sharpe_decline": round(sharpe_decline, 3),
            "recent_wr": round(recent_wr, 3),
            "prior_wr": round(prior_wr, 3),
            "recommendation": "Reduce allocation or retire strategy" if decaying else "Strategy healthy",
        }


# ════════════════════════════════════════════════════════════════════
# 5. ALPHA ATTRIBUTION
# ════════════════════════════════════════════════════════════════════

class AlphaAttribution:
    """Attribute profit/loss to specific alpha sources.

    Breaks down returns into:
    - Trend alpha (from trend-following)
    - Momentum alpha (from momentum signals)
    - Mean reversion alpha (from range trading)
    - News alpha (from news-based trades)
    - Liquidity alpha (from liquidity sweep entries)
    """

    @staticmethod
    def attribute(trades: List[dict]) -> dict:
        """Attribute P&L to alpha sources.

        Args:
            trades: List of trade dicts with 'strategy', 'pnl', 'reason'.

        Returns:
            {"by_source": dict, "total_pnl": float, "best_source": str, ...}
        """
        if not trades:
            return {"by_source": {}, "total_pnl": 0, "best_source": "NONE"}

        by_source = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})

        for t in trades:
            # Determine alpha source from strategy/reason
            strategy = t.get("strategy", "UNKNOWN").upper()
            reason = t.get("reason", "").lower()
            pnl = float(t.get("pnl", 0) or 0)

            if "TREND" in strategy:
                source = "trend"
            elif "MOMENTUM" in strategy:
                source = "momentum"
            elif "MEAN" in strategy or "RANGE" in strategy:
                source = "mean_reversion"
            elif "BREAKOUT" in strategy or "RETEST" in strategy:
                source = "breakout"
            elif "PULLBACK" in strategy or "SMC" in strategy:
                source = "liquidity"
            elif "NEWS" in reason or "sentiment" in reason:
                source = "news"
            else:
                source = "other"

            by_source[source]["pnl"] += pnl
            by_source[source]["trades"] += 1
            if pnl > 0:
                by_source[source]["wins"] += 1

        # Compute summary
        total_pnl = sum(v["pnl"] for v in by_source.values())
        best_source = max(by_source.items(), key=lambda x: x[1]["pnl"])[0] if by_source else "NONE"

        result = {}
        for source, stats in by_source.items():
            wr = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            result[source] = {
                "pnl": round(stats["pnl"], 2),
                "trades": stats["trades"],
                "win_rate": round(wr, 2),
                "pct_of_total": round(stats["pnl"] / total_pnl * 100, 1) if total_pnl != 0 else 0,
            }

        return {
            "by_source": result,
            "total_pnl": round(total_pnl, 2),
            "best_source": best_source,
            "worst_source": min(by_source.items(), key=lambda x: x[1]["pnl"])[0] if by_source else "NONE",
        }


# ════════════════════════════════════════════════════════════════════
# 7. TRANSACTION COST ANALYSIS (TCA)
# ════════════════════════════════════════════════════════════════════

class TransactionCostAnalyzer:
    """Full transaction cost analysis for every trade.

    Components:
    - Commission: broker fee per lot
    - Spread: bid-ask spread cost
    - Slippage: difference between expected and actual fill price
    - Market impact: price movement caused by our order
    - Opportunity cost: profit missed by not executing faster
    """

    @staticmethod
    def analyze(
        entry_expected: float,
        entry_actual: float,
        exit_expected: float,
        exit_actual: float,
        lot: float,
        spread_pips: float,
        commission_per_lot: float = 7.0,
        pip_value: float = 10.0,
        symbol: str = None,
    ) -> dict:
        """Analyze transaction costs for a completed trade.

        Args:
            symbol: optional instrument symbol (e.g. "USDJPY", "XAUUSD").
                FIX (institutional review, item #1): entry/exit slippage-in-
                pips used to be computed with a hardcoded pip size of 0.0001,
                which is wrong by ~100x on JPY crosses (pip = 0.01) and on
                metals. Passing `symbol` makes that conversion correct for
                the instrument; omitting it preserves the old 0.0001 default
                so existing callers are unaffected.

        Returns:
            {"total_cost": float, "commission": float, "spread_cost": float, ...}
        """
        # Commission
        commission = commission_per_lot * lot

        # Spread cost (paid on entry and exit)
        spread_cost = spread_pips * pip_value * lot * 2  # entry + exit

        # Slippage (raw money terms — units-per-lot conversion, not
        # pip-size-dependent, so this was already correct)
        entry_slippage = abs(entry_actual - entry_expected) * lot * units_per_lot()  # approximate
        exit_slippage = abs(exit_actual - exit_expected) * lot * units_per_lot()
        total_slippage = entry_slippage + exit_slippage

        # Market impact (simplified — 0 if small order)
        market_impact = 0
        if lot > 1.0:
            market_impact = (lot - 1.0) * spread_pips * 0.5 * pip_value

        total_cost = commission + spread_cost + total_slippage + market_impact

        size = pip_size(symbol)
        return {
            "total_cost": round(total_cost, 2),
            "commission": round(commission, 2),
            "spread_cost": round(spread_cost, 2),
            "slippage": round(total_slippage, 2),
            "market_impact": round(market_impact, 2),
            "entry_slippage_pips": round(abs(entry_actual - entry_expected) / size, 1),
            "exit_slippage_pips": round(abs(exit_actual - exit_expected) / size, 1),
            "cost_per_lot": round(total_cost / lot, 2) if lot > 0 else 0,
        }


# ════════════════════════════════════════════════════════════════════
# 8. LATENCY ANALYZER
# ════════════════════════════════════════════════════════════════════

class LatencyAnalyzer:
    """Track execution latency for performance optimization.

    Measures:
    - Signal latency: time from candle close to signal generation
    - Order latency: time from signal to order send
    - Fill latency: time from order send to fill confirmation
    - Total latency: end-to-end
    """

    def __init__(self):
        self._records: List[dict] = []

    def record(self, signal_time: float, order_time: float, fill_time: float,
               pair: str = ""):
        """Record latency for one trade execution."""
        signal_latency = order_time - signal_time
        order_latency = fill_time - order_time
        total_latency = fill_time - signal_time

        entry = {
            "pair": pair,
            "signal_latency_ms": round(signal_latency * 1000, 1),
            "order_latency_ms": round(order_latency * 1000, 1),
            "total_latency_ms": round(total_latency * 1000, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._records.append(entry)

    def summary(self) -> dict:
        """Get latency summary statistics."""
        if not self._records:
            return {"avg_total_ms": 0, "reason": "No latency data"}

        totals = [r["total_latency_ms"] for r in self._records]
        return {
            "avg_total_ms": round(np.mean(totals), 1),
            "p50_total_ms": round(np.percentile(totals, 50), 1),
            "p95_total_ms": round(np.percentile(totals, 95), 1),
            "max_total_ms": round(np.max(totals), 1),
            "avg_signal_ms": round(np.mean([r["signal_latency_ms"] for r in self._records]), 1),
            "avg_order_ms": round(np.mean([r["order_latency_ms"] for r in self._records]), 1),
            "n_trades": len(self._records),
        }


# ════════════════════════════════════════════════════════════════════
# 9. DATA PROVENANCE
# ════════════════════════════════════════════════════════════════════

class DataProvenance:
    """Track the source, quality, and reliability of every data point.

    Every piece of data gets a provenance tag:
    - Source (MT5, yfinance, API, synthetic)
    - Timestamp (when fetched)
    - Quality score (0-1)
    - Reliability (HIGH/MEDIUM/LOW)
    """

    def __init__(self):
        self._records: Dict[str, dict] = {}

    def tag(self, key: str, source: str, quality: float = 1.0,
            reliability: str = "HIGH", metadata: dict = None):
        """Tag a data point with provenance information."""
        self._records[key] = {
            "source": source,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "quality": quality,
            "reliability": reliability,
            "metadata": metadata or {},
        }

    def get(self, key: str) -> dict:
        """Get provenance for a data point."""
        return self._records.get(key, {"source": "UNKNOWN", "quality": 0, "reliability": "UNKNOWN"})

    def quality_report(self) -> dict:
        """Generate a data quality report."""
        if not self._records:
            return {"total": 0, "avg_quality": 0, "sources": {}}

        sources = defaultdict(list)
        for record in self._records.values():
            sources[record["source"]].append(record["quality"])

        return {
            "total": len(self._records),
            "avg_quality": round(np.mean([r["quality"] for r in self._records.values()]), 3),
            "by_source": {
                source: {
                    "count": len(quals),
                    "avg_quality": round(np.mean(quals), 3),
                }
                for source, quals in sources.items()
            },
        }


# ════════════════════════════════════════════════════════════════════
# 10. MARKET CALENDAR INTELLIGENCE
# ════════════════════════════════════════════════════════════════════

class MarketCalendar:
    """Market calendar intelligence — holidays, expiries, rebalancing.

    Tracks:
    - Bank holidays (US, UK, EU, JP)
    - Contract rollover dates
    - Options/futures expiry
    - Month-end / quarter-end / year-end rebalancing
    - Half trading days
    """

    # Major holidays (simplified — month/day)
    HOLIDAYS = {
        "US": [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (12, 24), # Christmas Eve (half day)
        ],
        "UK": [(1, 1), (12, 25), (12, 26)],
        "JP": [(1, 1), (1, 2), (12, 23), (12, 31)],
    }

    @staticmethod
    def check_calendar(dt: datetime = None) -> dict:
        """Check market calendar for a given date.

        Returns:
            {"is_holiday": bool, "is_half_day": bool, "is_month_end": bool, ...}
        """
        dt = dt or datetime.now(timezone.utc)
        month, day = dt.month, dt.day
        weekday = dt.weekday()  # 0=Monday, 6=Sunday

        # Weekend
        is_weekend = weekday >= 5

        # Holiday check
        is_holiday = False
        holiday_region = ""
        for region, holidays in MarketCalendar.HOLIDAYS.items():
            if (month, day) in holidays:
                is_holiday = True
                holiday_region = region
                break

        # Half day (Christmas Eve, day before Independence Day)
        is_half_day = (month == 12 and day == 24) or (month == 7 and day == 3)

        # Month-end / quarter-end / year-end
        is_month_end = dt.day >= 28  # simplified
        is_quarter_end = dt.month in (3, 6, 9, 12) and dt.day >= 28
        is_year_end = dt.month == 12 and dt.day >= 28

        # Friday (position closing considerations)
        is_friday = weekday == 4

        # Trade allowed?
        trade_allowed = not is_weekend and not is_holiday

        alerts = []
        if is_holiday:
            alerts.append(f"{holiday_region} holiday — market closed")
        if is_half_day:
            alerts.append("Half trading day — reduced liquidity expected")
        if is_friday:
            alerts.append("Friday — consider closing positions before weekend")
        if is_month_end:
            alerts.append("Month-end — possible rebalancing flows")
        if is_quarter_end:
            alerts.append("Quarter-end — large institutional rebalancing likely")

        return {
            "is_holiday": is_holiday,
            "is_half_day": is_half_day,
            "is_weekend": is_weekend,
            "is_month_end": is_month_end,
            "is_quarter_end": is_quarter_end,
            "is_year_end": is_year_end,
            "is_friday": is_friday,
            "trade_allowed": trade_allowed,
            "alerts": alerts,
        }


# ════════════════════════════════════════════════════════════════════
# 15. REGIME PROBABILITY
# ════════════════════════════════════════════════════════════════════

class RegimeProbability:
    """Soft regime classification — probability distribution over regimes.

    Instead of "market is TRENDING", returns:
    {"trending": 0.65, "ranging": 0.20, "volatile": 0.15}
    """

    @staticmethod
    def compute(df: pd.DataFrame, adx: float = 25, atr_ratio: float = 1.0) -> dict:
        """Compute regime probability distribution.

        Args:
            df: OHLCV DataFrame.
            adx: Current ADX value.
            atr_ratio: Current ATR / average ATR ratio.

        Returns:
            {"trending": float, "ranging": float, "volatile": float, "dominant": str}
        """
        # Base probabilities
        trending_prob = 0.33
        ranging_prob = 0.33
        volatile_prob = 0.34

        # Adjust based on ADX
        if adx > 30:
            trending_prob += 0.3
            ranging_prob -= 0.15
            volatile_prob -= 0.15
        elif adx < 20:
            ranging_prob += 0.3
            trending_prob -= 0.15
            volatile_prob -= 0.15

        # Adjust based on ATR ratio
        if atr_ratio > 1.5:
            volatile_prob += 0.25
            trending_prob -= 0.1
            ranging_prob -= 0.15
        elif atr_ratio < 0.7:
            ranging_prob += 0.15
            volatile_prob -= 0.1

        # Normalize
        total = trending_prob + ranging_prob + volatile_prob
        trending_prob /= total
        ranging_prob /= total
        volatile_prob /= total

        dominant = max([("trending", trending_prob),
                       ("ranging", ranging_prob),
                       ("volatile", volatile_prob)],
                      key=lambda x: x[1])[0]

        return {
            "trending": round(trending_prob, 3),
            "ranging": round(ranging_prob, 3),
            "volatile": round(volatile_prob, 3),
            "dominant": dominant,
            "confidence": round(max(trending_prob, ranging_prob, volatile_prob), 3),
        }


# ════════════════════════════════════════════════════════════════════
# 16. CAUSAL INFERENCE
# ════════════════════════════════════════════════════════════════════

class CausalInference:
    """Granger causality test for causal inference.

    Tests whether one time series Granger-causes another.
    "X Granger-causes Y if past values of X help predict Y better
    than past values of Y alone."
    """

    @staticmethod
    def granger_causality(cause: np.ndarray, effect: np.ndarray,
                          max_lag: int = 5) -> dict:
        """Simplified Granger causality test.

        Returns:
            {"causes": bool, "f_statistic": float, "p_value": float, ...}
        """
        if len(cause) < max_lag + 10 or len(effect) < max_lag + 10:
            return {"causes": False, "reason": "Insufficient data"}

        n = min(len(cause), len(effect))
        cause = cause[:n]
        effect = effect[:n]

        # Build restricted model (Y ~ Y_lag)
        Y = effect[max_lag:]
        X_restricted = np.column_stack([effect[max_lag - i - 1:n - i - 1] for i in range(max_lag)])
        X_restricted = np.column_stack([np.ones(len(Y)), X_restricted])

        # Build unrestricted model (Y ~ Y_lag + X_lag)
        X_unrestricted = np.column_stack([
            np.ones(len(Y)),
            *[effect[max_lag - i - 1:n - i - 1] for i in range(max_lag)],
            *[cause[max_lag - i - 1:n - i - 1] for i in range(max_lag)],
        ])

        try:
            # Fit restricted
            beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            resid_r = Y - X_restricted @ beta_r
            ssr_r = np.sum(resid_r ** 2)

            # Fit unrestricted
            beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
            resid_u = Y - X_unrestricted @ beta_u
            ssr_u = np.sum(resid_u ** 2)

            # F-statistic
            n_params = len(Y)
            p_r = X_restricted.shape[1]
            p_u = X_unrestricted.shape[1]
            df1 = p_u - p_r
            df2 = n_params - p_u

            if df2 > 0 and ssr_u > 0:
                f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
                # Approximate p-value (simplified)
                p_value = max(0, min(1, np.exp(-f_stat / 2)))
            else:
                f_stat = 0
                p_value = 1

            causes = p_value < 0.05

            return {
                "causes": causes,
                "f_statistic": round(f_stat, 3),
                "p_value": round(p_value, 4),
                "ssr_restricted": round(ssr_r, 6),
                "ssr_unrestricted": round(ssr_u, 6),
                "interpretation": f"Granger causes (p={p_value:.3f})" if causes else f"No causality (p={p_value:.3f})",
            }
        except Exception as e:
            return {"causes": False, "reason": f"Computation failed: {e}"}


# ════════════════════════════════════════════════════════════════════
# 17. DIGITAL TWIN
# ════════════════════════════════════════════════════════════════════

class DigitalTwin:
    """Virtual clone of the real account for pre-trade testing.

    Before executing a trade on the real account, simulate it on the
    digital twin to estimate:
    - Expected P&L
    - Risk exposure
    - Portfolio impact
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions: List[dict] = []
        self.trade_history: List[dict] = []

    def simulate_trade(self, direction: str, entry: float, sl: float,
                       tp: float, lot: float, pair: str = "EURUSD") -> dict:
        """Simulate a trade on the digital twin.

        Returns:
            {"projected_pnl": float, "projected_risk": float, "recommendation": str}
        """
        if direction.upper() == "BUY":
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp

        risk_usd = risk * lot * 100000  # simplified pip value
        reward_usd = reward * lot * 100000
        rr = reward / risk if risk > 0 else 0

        # Check if risk exceeds 2% of balance
        risk_pct = risk_usd / self.balance * 100 if self.balance > 0 else 100

        if risk_pct > 2:
            recommendation = "SKIP — risk too high"
        elif rr < 2:
            recommendation = "SKIP — R/R too low"
        elif risk_pct < 0.3:
            recommendation = "OK — very low risk, consider increasing size"
        else:
            recommendation = "OK — execute trade"

        return {
            "projected_pnl": round(reward_usd, 2),
            "projected_risk": round(risk_usd, 2),
            "projected_rr": round(rr, 2),
            "risk_pct": round(risk_pct, 2),
            "balance_after_win": round(self.balance + reward_usd, 2),
            "balance_after_loss": round(self.balance - risk_usd, 2),
            "recommendation": recommendation,
        }


# ════════════════════════════════════════════════════════════════════
# 19. FAILURE ANALYSIS
# ════════════════════════════════════════════════════════════════════

class FailureAnalyzer:
    """Classify every loss to understand WHY it happened.

    Categories:
    - BAD_PREDICTION: signal was wrong (direction incorrect)
    - BAD_TIMING: right direction but entered too early/late
    - BAD_EXECUTION: slippage/spread ate the profit
    - BAD_EXIT: TP/SL placement was wrong
    - UNEXPECTED_NEWS: unforeseen news event
    - RANDOM_NOISE: market noise (unavoidable)
    """

    @staticmethod
    def classify_loss(trade: dict, market_context: dict = None) -> dict:
        """Classify a losing trade.

        Args:
            trade: Trade dict with entry, exit, sl, tp, pnl, direction.
            market_context: Market state at time of trade (optional).

        Returns:
            {"category": str, "reason": str, "lesson": str}
        """
        direction = trade.get("direction", trade.get("type", "")).upper()
        entry = float(trade.get("entry", 0) or 0)
        exit_price = float(trade.get("exit_price", trade.get("exit", 0)) or 0)
        sl = float(trade.get("sl", 0) or 0)
        tp = float(trade.get("tp", 0) or 0)
        pnl = float(trade.get("pnl", 0) or 0)

        # Check if direction was correct but timing was wrong
        # (price eventually went in our direction but hit SL first)
        if direction == "BUY" and exit_price <= sl:
            # Did price eventually go above entry?
            if market_context and market_context.get("price_after_exit", 0) > entry:
                return {
                    "category": "BAD_TIMING",
                    "reason": "Direction correct but SL hit before move — entered too early",
                    "lesson": "Wait for better entry (pullback/confirmation) before entering",
                }

        if direction == "SELL" and exit_price >= sl:
            if market_context and market_context.get("price_after_exit", 0) < entry:
                return {
                    "category": "BAD_TIMING",
                    "reason": "Direction correct but SL hit before drop — entered too early",
                    "lesson": "Wait for confirmation before shorting",
                }

        # Check if SL was too tight
        if sl > 0 and entry > 0:
            sl_distance = abs(entry - sl)
            atr = market_context.get("atr", 0.001) if market_context else 0.001
            if sl_distance < atr * 0.5:
                return {
                    "category": "BAD_EXIT",
                    "reason": f"SL too tight ({sl_distance:.5f} < 0.5×ATR {atr:.5f})",
                    "lesson": "Use wider SL (at least 1×ATR) to avoid noise stop-outs",
                }

        # Check if it was news-related
        if market_context and market_context.get("news_during_trade", False):
            return {
                "category": "UNEXPECTED_NEWS",
                "reason": "News event during trade caused adverse move",
                "lesson": "Check news calendar more carefully before entering",
            }

        # Default: random noise or bad prediction
        if market_context and market_context.get("final_direction", "") == direction:
            return {
                "category": "RANDOM_NOISE",
                "reason": "Direction was ultimately correct but noise caused loss",
                "lesson": "Unavoidable loss — maintain risk management",
            }
        else:
            return {
                "category": "BAD_PREDICTION",
                "reason": "Signal direction was wrong — market moved against thesis",
                "lesson": "Review signal logic and improve confluence requirements",
            }


# ════════════════════════════════════════════════════════════════════
# 20. EDGE PRESERVATION
# ════════════════════════════════════════════════════════════════════

class EdgePreservation:
    """Detect when a strategy's edge is decaying (alpha decay).

    Tracks:
    - Win rate trend (is it declining?)
    - Average profit per trade (is it shrinking?)
    - Strategy crowding (are too many trades using the same setup?)
    """

    @staticmethod
    def check_edge_decay(trades: List[dict], window: int = 20) -> dict:
        """Check if a strategy is losing its edge.

        Returns:
            {"edge_decaying": bool, "wr_trend": str, "recommendation": str}
        """
        if len(trades) < window * 2:
            return {"edge_decaying": False, "reason": "Insufficient data"}

        recent = trades[-window:]
        prior = trades[-window * 2:-window]

        recent_wr = sum(1 for t in recent if t.get("result") == "WIN") / len(recent)
        prior_wr = sum(1 for t in prior if t.get("result") == "WIN") / len(prior)

        recent_avg_pnl = np.mean([float(t.get("pnl", 0) or 0) for t in recent])
        prior_avg_pnl = np.mean([float(t.get("pnl", 0) or 0) for t in prior])

        wr_trend = "DECLINING" if recent_wr < prior_wr - 0.1 else "STABLE" if abs(recent_wr - prior_wr) < 0.1 else "IMPROVING"
        pnl_trend = "DECLINING" if recent_avg_pnl < prior_avg_pnl * 0.7 else "STABLE"

        edge_decaying = wr_trend == "DECLINING" and pnl_trend == "DECLINING"

        if edge_decaying:
            recommendation = "Strategy edge decaying — reduce allocation, investigate cause"
        elif wr_trend == "DECLINING":
            recommendation = "Win rate declining but P&L stable — monitor"
        else:
            recommendation = "Edge preserved — strategy healthy"

        return {
            "edge_decaying": edge_decaying,
            "recent_wr": round(recent_wr, 3),
            "prior_wr": round(prior_wr, 3),
            "wr_trend": wr_trend,
            "pnl_trend": pnl_trend,
            "recent_avg_pnl": round(recent_avg_pnl, 2),
            "prior_avg_pnl": round(prior_avg_pnl, 2),
            "recommendation": recommendation,
        }


# ════════════════════════════════════════════════════════════════════
# SMOKE TESTS
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    print("=== Market Ecology ===")
    n = 100
    df = pd.DataFrame({
        "high": 1.1 + np.random.randn(n) * 0.001,
        "low": 1.099 + np.random.randn(n) * 0.001,
        "close": 1.1 + np.random.randn(n) * 0.0005,
        "volume": np.random.randint(50, 500, n),
    })
    eco = MarketEcology.classify_participants(df)
    print(f"  {eco['dominant_participant']} — {eco['reason']}")

    print("\n=== Strategy Decay ===")
    sdt = StrategyDecayTracker()
    for i in range(40):
        wr = 0.6 - i * 0.005  # declining win rate
        sdt.record("TREND_FOLLOW", wr, 1.5 - i * 0.02, 1.8 - i * 0.03)
    decay = sdt.check_decay("TREND_FOLLOW", window=10)
    print(f"  Decaying: {decay['decaying']} — WR decline: {decay['wr_decline']:.1%}")

    print("\n=== Alpha Attribution ===")
    trades = [
        {"strategy": "TREND_FOLLOW", "pnl": 50, "result": "WIN"},
        {"strategy": "MEAN_REVERSION", "pnl": -20, "result": "LOSS"},
        {"strategy": "TREND_FOLLOW", "pnl": 30, "result": "WIN"},
        {"strategy": "BREAKOUT", "pnl": -15, "result": "LOSS"},
    ]
    alpha = AlphaAttribution.attribute(trades)
    print(f"  Best source: {alpha['best_source']}, Total: ${alpha['total_pnl']}")

    print("\n=== TCA ===")
    tca = TransactionCostAnalyzer.analyze(
        entry_expected=1.1000, entry_actual=1.1001,
        exit_expected=1.1060, exit_actual=1.1059,
        lot=0.1, spread_pips=1.5,
    )
    print(f"  Total cost: ${tca['total_cost']} (spread=${tca['spread_cost']}, slippage=${tca['slippage']})")

    print("\n=== Regime Probability ===")
    rp = RegimeProbability.compute(df, adx=32, atr_ratio=1.6)
    print(f"  Dominant: {rp['dominant']} (conf={rp['confidence']:.0%}) — {rp}")

    print("\n=== Market Calendar ===")
    from datetime import datetime
    cal = MarketCalendar.check_calendar(datetime(2024, 12, 24))  # Christmas Eve
    print(f"  Holiday: {cal['is_holiday']}, Half day: {cal['is_half_day']}")
    print(f"  Alerts: {cal['alerts']}")

    print("\n=== Granger Causality ===")
    cause = np.cumsum(np.random.randn(100))  # random walk
    effect = cause[1:] + np.random.randn(99) * 0.5  # effect lags cause
    gc = CausalInference.granger_causality(cause[:-1], effect, max_lag=3)
    print(f"  Causes: {gc['causes']} (p={gc['p_value']:.3f})")

    print("\n=== Digital Twin ===")
    twin = DigitalTwin(10000)
    sim = twin.simulate_trade("BUY", 1.1000, 1.0950, 1.1120, 0.1)
    print(f"  Projected PnL: ${sim['projected_pnl']}, Risk: ${sim['projected_risk']}")
    print(f"  Recommendation: {sim['recommendation']}")

    print("\n=== Failure Analysis ===")
    loss_trade = {"direction": "BUY", "entry": 1.1000, "exit_price": 1.0950,
                  "sl": 1.0970, "tp": 1.1060, "pnl": -30}
    failure = FailureAnalyzer.classify_loss(loss_trade, {"atr": 0.003})
    print(f"  Category: {failure['category']} — {failure['reason']}")

    print("\n=== Edge Preservation ===")
    edge_trades = [{"result": "WIN", "pnl": 50}] * 15 + [{"result": "LOSS", "pnl": -30}] * 15 + \
                  [{"result": "LOSS", "pnl": -20}] * 10
    edge = EdgePreservation.check_edge_decay(edge_trades, window=15)
    print(f"  Edge decaying: {edge['edge_decaying']} — {edge['recommendation']}")

    print("\nAll final frontier smoke tests passed.")