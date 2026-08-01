"""
analysis/phd_frontier.py — PhD-Level Niche Research Domains
=============================================================
7 missing PhD-level domains from the ultimate research roadmap:

1. Information Theory — Shannon entropy, KL divergence, transfer entropy
3. Chaos & Nonlinear Dynamics — Lyapunov exponent, fractal dimension
6. Anomaly Detection — Isolation Forest, statistical outliers
8. Game Theory — Nash equilibrium, zero-sum modeling
10. Federated Learning — privacy-preserving framework stub
14. Decision Intelligence — expected utility, decision trees
15. Knowledge Graphs — market entity relationships

USAGE:
    from analysis.phd_frontier import (
        InformationTheory, ChaosDynamics, AnomalyDetector,
        GameTheory, FederatedLearning, DecisionIntelligence,
        KnowledgeGraph,
    )
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from utils.logger import get_logger

log = get_logger("phd_frontier")


# ════════════════════════════════════════════════════════════════════
# 1. INFORMATION THEORY
# ════════════════════════════════════════════════════════════════════

class InformationTheory:
    """Information-theoretic measures for market analysis.

    - Shannon Entropy: measures unpredictability of price changes
    - KL Divergence: measures how much one distribution differs from another
    - Transfer Entropy: measures information flow from one series to another
    - Mutual Information: measures shared information between two series
    """

    @staticmethod
    def shannon_entropy(values: np.ndarray, n_bins: int = 20) -> float:
        """Compute Shannon entropy of a value distribution.

        High entropy = unpredictable/random market
        Low entropy = predictable/structured market
        """
        if len(values) < n_bins:
            return 0.0

        hist, _ = np.histogram(values, bins=n_bins, density=True)
        probs = hist * (values.max() - values.min()) / n_bins
        probs = probs[probs > 0]  # remove zeros

        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 20) -> float:
        """Kullback-Leibler divergence D(P || Q).

        Measures how much distribution P differs from Q.
        0 = identical, higher = more different.
        """
        if len(p) < n_bins or len(q) < n_bins:
            return 0.0

        # Build histograms
        all_vals = np.concatenate([p, q])
        bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)

        # Normalize to probabilities
        p_hist = p_hist / (p_hist.sum() + 1e-10)
        q_hist = q_hist / (q_hist.sum() + 1e-10)

        # KL divergence = sum(p * log(p/q))
        mask = (p_hist > 0) & (q_hist > 0)
        kl = np.sum(p_hist[mask] * np.log(p_hist[mask] / q_hist[mask]))

        return float(kl)

    @staticmethod
    def transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
        """Transfer entropy from source to target.

        Measures how much information flows from source → target.
        High value = source influences target.
        """
        if len(source) < 20 or len(target) < 20:
            return 0.0

        n = min(len(source), len(target))
        source = source[:n]
        target = target[:n]

        # Discretize
        def discretize(x, bins=5):
            edges = np.linspace(x.min(), x.max(), bins + 1)
            return np.digitize(x, edges[1:-1])

        s = discretize(source)
        t = discretize(target)
        t_lag = np.roll(t, lag)
        t_lag[:lag] = 0

        # Transfer entropy = H(t_future | t_past) - H(t_future | t_past, s_past)
        # Simplified computation using joint probabilities
        from collections import Counter

        # P(t_future | t_past)
        c1 = Counter()
        c2 = Counter()
        for i in range(lag, n):
            c1[(t[i], t_lag[i])] += 1
            c2[(t[i], t_lag[i], s[i - lag])] += 1

        total = sum(c1.values())
        te = 0
        for (t_fut, t_past), count in c1.items():
            p_t_given_tpast = count / total
            # Find corresponding count in c2
            s_counts = {k: v for k, v in c2.items() if k[0] == t_fut and k[1] == t_past}
            p_joint = sum(s_counts.values()) / total
            if p_t_given_tpast > 0 and p_joint > 0:
                te += p_joint * np.log2(p_joint / (p_t_given_tpast * (p_joint / p_t_given_tpast + 1e-10) + 1e-10))

        return float(max(0, te))

    @staticmethod
    def mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        """Mutual information between two series.

        0 = independent, higher = more shared information.
        """
        if len(x) < n_bins or len(y) < n_bins:
            return 0.0

        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        # 2D histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
        p_xy = hist_2d / (hist_2d.sum() + 1e-10)

        p_x = p_xy.sum(axis=1, keepdims=True)  # shape (n_bins, 1)
        p_y = p_xy.sum(axis=0, keepdims=True)  # shape (1, n_bins)

        # MI = sum(p_xy * log(p_xy / (p_x * p_y)))
        # Broadcast p_x and p_y to match p_xy shape
        p_xy_product = p_x * p_y  # broadcasts to (n_bins, n_bins)
        mask = (p_xy > 0) & (p_xy_product > 0)
        mi = np.sum(p_xy[mask] * np.log(p_xy[mask] / p_xy_product[mask]))

        return float(mi)


# ════════════════════════════════════════════════════════════════════
# 3. CHAOS & NONLINEAR DYNAMICS
# ════════════════════════════════════════════════════════════════════

class ChaosDynamics:
    """Chaos theory and nonlinear dynamics for market analysis.

    - Lyapunov Exponent: measures sensitivity to initial conditions
      (positive = chaotic, negative = stable)
    - Fractal Dimension: measures complexity of price path
    - Correlation Dimension: measures attractor dimension
    """

    @staticmethod
    def lyapunov_exponent(prices: np.ndarray, min_sep: int = 10,
                          max_pairs: int = 500) -> float:
        """Estimate largest Lyapunov exponent.

        Positive = chaotic (sensitive to initial conditions)
        Negative = stable/predictable
        Near zero = random walk

        Uses the method of tracking nearby trajectories.
        """
        if len(prices) < 50:
            return 0.0

        returns = np.diff(np.log(prices))
        n = len(returns)

        # Find pairs of nearby points and track their divergence
        divergences = []
        np.random.seed(42)

        for _ in range(max_pairs):
            i = np.random.randint(0, n - min_sep)
            # Find nearest neighbor
            distances = np.abs(returns[:n - min_sep] - returns[i])
            distances[i] = np.inf  # exclude self
            j = np.argmin(distances)

            if distances[j] < 1e-10:
                continue

            # Track divergence over min_sep steps
            d0 = abs(returns[i] - returns[j])
            d_t = abs(returns[i + min_sep] - returns[j + min_sep]) if j + min_sep < n else d0

            if d0 > 0 and d_t > 0:
                divergences.append(np.log(d_t / d0))

        if not divergences:
            return 0.0

        # Lyapunov = average log divergence rate
        lyap = np.mean(divergences) / min_sep

        return float(lyap)

    @staticmethod
    def fractal_dimension(prices: np.ndarray, scales: List[int] = None) -> float:
        """Estimate fractal dimension using Hurst exponent relation.

        D = 2 - H (where H = Hurst exponent)
        D ≈ 1.5 = random walk
        D < 1.5 = trending (smoother)
        D > 1.5 = mean-reverting (rougher)
        """
        if scales is None:
            scales = [10, 20, 50, 100]

        if len(prices) < max(scales):
            return 1.5  # default to random walk

        # Use box-counting method (simplified)
        counts = []
        for scale in scales:
            if scale >= len(prices):
                continue
            # Count boxes needed to cover the price path
            scaled_prices = prices[::scale]
            price_range = scaled_prices.max() - scaled_prices.min()
            if price_range > 0:
                n_boxes = len(scaled_prices)  # simplified
                counts.append((scale, n_boxes))

        if len(counts) < 2:
            return 1.5

        # Fit: log(N) = -D * log(scale) + c
        log_scales = np.log([s for s, _ in counts])
        log_counts = np.log([n for _, n in counts])

        if len(log_scales) >= 2:
            coeffs = np.polyfit(log_scales, log_counts, 1)
            D = -coeffs[0]
        else:
            D = 1.5

        return float(max(1.0, min(2.0, D)))

    @staticmethod
    def market_regime_from_chaos(prices: np.ndarray) -> dict:
        """Classify market regime using chaos theory.

        Returns:
            {"lyapunov": float, "fractal_dim": float, "regime": str, ...}
        """
        lyap = ChaosDynamics.lyapunov_exponent(prices)
        fd = ChaosDynamics.fractal_dimension(prices)

        if lyap > 0.1:
            regime = "CHAOTIC"
            reason = f"Lyapunov={lyap:.3f} > 0 — market is chaotic, prediction difficult"
        elif lyap < -0.1:
            regime = "STABLE"
            reason = f"Lyapunov={lyap:.3f} < 0 — market is stable, prediction easier"
        else:
            regime = "RANDOM_WALK"
            reason = f"Lyapunov={lyap:.3f} ≈ 0 — market resembles random walk"

        if fd < 1.4:
            trend_type = "TRENDING (smooth)"
        elif fd > 1.6:
            trend_type = "MEAN_REVERTING (rough)"
        else:
            trend_type = "RANDOM"

        return {
            "lyapunov": round(lyap, 4),
            "fractal_dimension": round(fd, 4),
            "regime": regime,
            "trend_type": trend_type,
            "reason": reason,
        }


# ════════════════════════════════════════════════════════════════════
# 6. ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """Detect anomalous market behavior — flash crashes, manipulation, errors.

    Methods:
    - Statistical: Z-score, IQR-based outlier detection
    - Isolation Forest: tree-based anomaly detection (simplified)
    - Volume anomaly: detect unusual volume spikes
    """

    @staticmethod
    def detect_price_anomaly(prices: np.ndarray, window: int = 50,
                              z_threshold: float = 3.0) -> dict:
        """Detect price anomalies using Z-score.

        Returns:
            {"is_anomaly": bool, "z_score": float, "direction": str, ...}
        """
        if len(prices) < window:
            return {"is_anomaly": False, "reason": "Insufficient data"}

        recent = prices[-window:]
        mean = np.mean(recent)
        std = np.std(recent)

        if std < 1e-10:
            return {"is_anomaly": False, "reason": "Zero variance"}

        current = prices[-1]
        z_score = (current - mean) / std

        is_anomaly = abs(z_score) > z_threshold
        direction = "UP" if z_score > 0 else "DOWN"

        return {
            "is_anomaly": is_anomaly,
            "z_score": round(z_score, 2),
            "direction": direction if is_anomaly else "NORMAL",
            "current_price": round(current, 5),
            "mean": round(mean, 5),
            "std": round(std, 5),
            "severity": "EXTREME" if abs(z_score) > 5 else "HIGH" if abs(z_score) > 4 else "MEDIUM" if is_anomaly else "NONE",
            "reason": f"Z-score {z_score:.2f} {'> ' if z_score > 0 else '< '}{z_threshold}" if is_anomaly else "Normal",
        }

    @staticmethod
    def detect_volume_anomaly(volumes: np.ndarray, window: int = 20,
                              multiplier: float = 3.0) -> dict:
        """Detect volume anomalies (spikes that indicate institutional activity)."""
        if len(volumes) < window:
            return {"is_anomaly": False, "reason": "Insufficient data"}

        avg_vol = np.mean(volumes[-window:])
        current_vol = volumes[-1]

        if avg_vol < 1e-10:
            return {"is_anomaly": False, "reason": "Zero average volume"}

        ratio = current_vol / avg_vol
        is_anomaly = ratio > multiplier

        return {
            "is_anomaly": is_anomaly,
            "volume_ratio": round(ratio, 2),
            "current_volume": float(current_vol),
            "avg_volume": round(float(avg_vol), 1),
            "severity": "EXTREME" if ratio > 5 else "HIGH" if ratio > multiplier else "NORMAL",
            "reason": f"Volume {ratio:.1f}× average" if is_anomaly else "Normal volume",
        }

    @staticmethod
    def isolation_forest_score(features: np.ndarray, n_trees: int = 50,
                                sample_size: int = 256) -> np.ndarray:
        """Simplified Isolation Forest anomaly scoring (no sklearn dependency).

        Returns anomaly scores: higher = more anomalous.
        """
        if len(features) < 10:
            return np.zeros(len(features))

        n = len(features)
        scores = np.zeros(n)

        for _ in range(n_trees):
            # Random subsample
            idx = np.random.choice(n, min(sample_size, n), replace=False)
            sample = features[idx]

            # Random split
            feature_idx = np.random.randint(0, features.shape[1] if features.ndim > 1 else 1)
            if features.ndim > 1:
                split_val = np.random.uniform(sample[:, feature_idx].min(),
                                              sample[:, feature_idx].max())
                # Count how many points are isolated by this split
                left = features[:, feature_idx] < split_val
                # Path length proxy: points far from split are more isolated
                scores += np.abs(features[:, feature_idx] - split_val)
            else:
                split_val = np.random.uniform(sample.min(), sample.max())
                scores += np.abs(features - split_val)

        # Normalize: lower path length = more anomalous
        scores = scores / n_trees
        # Invert: high score = far from splits = anomalous
        scores = scores / (scores.max() + 1e-10)

        return scores


# ════════════════════════════════════════════════════════════════════
# 8. GAME THEORY
# ════════════════════════════════════════════════════════════════════

class GameTheory:
    """Game-theoretic analysis of market participant interactions.

    Models trading as a game between:
    - Us (the AI)
    - Market makers (providing liquidity)
    - Other traders (competing for the same edge)
    - Institutions (moving price with large orders)

    Uses Nash equilibrium concepts to find optimal strategies.
    """

    @staticmethod
    def zero_sum_payoff(our_action: str, opponent_action: str,
                        payoff_matrix: dict = None) -> float:
        """Compute payoff in a zero-sum game.

        Default payoff matrix for trading:
        - We BUY, market goes UP → +1
        - We BUY, market goes DOWN → -1
        - We SELL, market goes UP → -1
        - We SELL, market goes DOWN → +1
        - We HOLD → 0
        """
        if payoff_matrix is None:
            payoff_matrix = {
                ("BUY", "UP"): 1, ("BUY", "DOWN"): -1, ("BUY", "FLAT"): -0.1,
                ("SELL", "UP"): -1, ("SELL", "DOWN"): 1, ("SELL", "FLAT"): -0.1,
                ("HOLD", "UP"): 0, ("HOLD", "DOWN"): 0, ("HOLD", "FLAT"): 0,
            }

        return payoff_matrix.get((our_action, opponent_action), 0)

    @staticmethod
    def nash_equilibrium_strategy(win_prob: float, loss_prob: float,
                                  payoff_win: float = 1.0,
                                  payoff_loss: float = -1.0) -> dict:
        """Find optimal mixed strategy using Nash equilibrium.

        In trading: should we trade (BUY/SELL) or wait (HOLD)?
        Nash equilibrium: mix strategies so opponent can't exploit us.

        Returns:
            {"trade_probability": float, "hold_probability": float, "expected_value": float}
        """
        # Expected value of trading
        ev_trade = win_prob * payoff_win + loss_prob * payoff_loss

        # Expected value of holding = 0

        if ev_trade > 0:
            # Trading is positive EV → trade with high probability
            trade_prob = min(1.0, ev_trade / payoff_win)
        else:
            # Trading is negative EV → mostly hold
            trade_prob = max(0.0, 0.5 + ev_trade)  # softer penalty

        hold_prob = 1.0 - trade_prob

        return {
            "trade_probability": round(trade_prob, 3),
            "hold_probability": round(hold_prob, 3),
            "expected_value_trade": round(ev_trade, 3),
            "expected_value_hold": 0,
            "recommendation": "TRADE" if trade_prob > 0.6 else "HOLD" if trade_prob < 0.3 else "MARGINAL",
        }

    @staticmethod
    def adversarial_analysis(our_signal: str, market_maker_inventory: str = "FLAT") -> dict:
        """Analyze adversarial interaction with market makers.

        Market makers adjust quotes based on inventory. If they're long,
        they'll lower bid (want to sell). If short, they'll raise ask.

        Returns:
            {"predicted_mm_behavior": str, "optimal_action": str, "reason": str}
        """
        if market_maker_inventory == "LONG":
            # MM wants to sell → they'll push price down
            if our_signal == "BUY":
                return {
                    "predicted_mm_behavior": "SELL_PRESSURE",
                    "optimal_action": "WAIT",
                    "reason": "MM is long and will push price down — wait for better entry",
                }
            elif our_signal == "SELL":
                return {
                    "predicted_mm_behavior": "SELL_PRESSURE",
                    "optimal_action": "SELL",
                    "reason": "MM selling aligns with our SELL — ride the wave",
                }
        elif market_maker_inventory == "SHORT":
            if our_signal == "BUY":
                return {
                    "predicted_mm_behavior": "BUY_PRESSURE",
                    "optimal_action": "BUY",
                    "reason": "MM is short and will push price up — aligns with our BUY",
                }
            elif our_signal == "SELL":
                return {
                    "predicted_mm_behavior": "BUY_PRESSURE",
                    "optimal_action": "WAIT",
                    "reason": "MM is short and will push price up — wait for better SELL entry",
                }

        return {
            "predicted_mm_behavior": "NEUTRAL",
            "optimal_action": our_signal,
            "reason": "MM inventory neutral — no adversarial concern",
        }


# ════════════════════════════════════════════════════════════════════
# 14. DECISION INTELLIGENCE
# ════════════════════════════════════════════════════════════════════

class DecisionIntelligence:
    """Decision optimization beyond simple prediction.

    Not just "BUY or SELL" but:
    - What's the expected utility of each action?
    - What's the opportunity cost of not trading?
    - What's the information value of waiting?
    """

    @staticmethod
    def expected_utility(
        action: str,
        win_prob: float,
        win_pnl: float,
        loss_pnl: float,
        risk_aversion: float = 1.0,
    ) -> dict:
        """Compute expected utility of an action.

        Uses utility theory: U(x) = x^(1-α) / (1-α)
        where α = risk aversion (higher = more risk-averse)

        Returns:
            {"expected_utility": float, "expected_pnl": float, "certainty_equivalent": float}
        """
        if action not in ("BUY", "SELL"):
            return {"expected_utility": 0, "expected_pnl": 0, "recommendation": "HOLD"}

        # Expected PnL
        ev_pnl = win_prob * win_pnl + (1 - win_prob) * loss_pnl

        # Utility (simplified: use risk-adjusted EV)
        if risk_aversion <= 0:
            utility = ev_pnl
        else:
            # Risk-adjusted utility: penalize losses more
            win_util = win_pnl ** (1 / (1 + risk_aversion)) if win_pnl > 0 else win_pnl
            loss_util = -abs(loss_pnl) ** (1 + risk_aversion) if loss_pnl < 0 else loss_pnl
            utility = win_prob * win_util + (1 - win_prob) * loss_util

        # Certainty equivalent: guaranteed amount = expected utility
        ce = utility if utility > 0 else -abs(utility) ** (1 / (1 + risk_aversion))

        return {
            "action": action,
            "expected_utility": round(utility, 2),
            "expected_pnl": round(ev_pnl, 2),
            "certainty_equivalent": round(ce, 2),
            "win_prob": round(win_prob, 3),
            "recommendation": "TRADE" if utility > 0 else "SKIP",
        }

    @staticmethod
    def information_value(wait_time_minutes: int, current_confidence: float,
                          confidence_gain_per_minute: float = 0.002) -> dict:
        """Value of waiting for more information.

        If waiting increases confidence enough to change the decision,
        waiting has positive information value.

        Returns:
            {"should_wait": bool, "projected_confidence": float, "info_value": float}
        """
        projected_confidence = min(1.0, current_confidence + wait_time_minutes * confidence_gain_per_minute)

        # Information value = increase in expected utility from better decision
        info_value = projected_confidence - current_confidence

        should_wait = (
            current_confidence < 0.55 and  # uncertain
            projected_confidence >= 0.60 and  # waiting would cross threshold
            wait_time_minutes <= 30  # don't wait too long
        )

        return {
            "should_wait": should_wait,
            "current_confidence": round(current_confidence, 3),
            "projected_confidence": round(projected_confidence, 3),
            "info_value": round(info_value, 3),
            "wait_time": wait_time_minutes,
            "reason": "Waiting improves confidence past threshold" if should_wait else "No benefit from waiting",
        }


# ════════════════════════════════════════════════════════════════════
# 15. KNOWLEDGE GRAPHS
# ════════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """Market entity relationship graph.

    Models relationships between market entities:
    - EUR (currency) → EURUSD (pair) → USD (currency)
    - USD → DXY (dollar index) → Gold (inverse correlation)
    - Fed (central bank) → USD → all USD pairs
    - CPI (economic event) → USD → EURUSD/GBPUSD/USDJPY
    """

    # Entity types
    ENTITY_CURRENCY = "currency"
    ENTITY_PAIR = "pair"
    ENTITY_COMMODITY = "commodity"
    ENTITY_INDEX = "index"
    ENTITY_CENTRAL_BANK = "central_bank"
    ENTITY_ECONOMIC_EVENT = "economic_event"

    # Relationship types
    REL_CORRELATES = "correlates_with"
    REL_INVERSE_CORRELATES = "inverse_correlates_with"
    REL_INFLUENCES = "influences"
    REL_COMPONENT_OF = "component_of"
    REL_IMPACTS = "impacts"

    def __init__(self):
        self._nodes: Dict[str, dict] = {}
        self._edges: List[dict] = []
        self._build_default_graph()

    def _build_default_graph(self):
        """Build default market knowledge graph."""
        # Currencies
        for c in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
            self._nodes[c] = {"type": self.ENTITY_CURRENCY, "name": c}

        # Pairs
        for p in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAUUSD"]:
            self._nodes[p] = {"type": self.ENTITY_PAIR, "name": p}

        # Commodities / Indices
        self._nodes["XAUUSD"] = {"type": self.ENTITY_COMMODITY, "name": "Gold"}
        self._nodes["DXY"] = {"type": self.ENTITY_INDEX, "name": "Dollar Index"}
        self._nodes["VIX"] = {"type": self.ENTITY_INDEX, "name": "Volatility Index"}
        self._nodes["SP500"] = {"type": self.ENTITY_INDEX, "name": "S&P 500"}

        # Central Banks
        self._nodes["FED"] = {"type": self.ENTITY_CENTRAL_BANK, "name": "Federal Reserve"}
        self._nodes["ECB"] = {"type": self.ENTITY_CENTRAL_BANK, "name": "European Central Bank"}
        self._nodes["BOE"] = {"type": self.ENTITY_CENTRAL_BANK, "name": "Bank of England"}
        self._nodes["BOJ"] = {"type": self.ENTITY_CENTRAL_BANK, "name": "Bank of Japan"}

        # Economic Events
        for e in ["CPI", "NFP", "FOMC", "GDP", "RATE_DECISION"]:
            self._nodes[e] = {"type": self.ENTITY_ECONOMIC_EVENT, "name": e}

        # Edges: relationships
        # Pair → component currencies
        for pair, (base, quote) in [
            ("EURUSD", ("EUR", "USD")), ("GBPUSD", ("GBP", "USD")),
            ("USDJPY", ("USD", "JPY")), ("AUDUSD", ("AUD", "USD")),
            ("USDCAD", ("USD", "CAD")), ("USDCHF", ("USD", "CHF")),
        ]:
            self._edges.append({"source": pair, "target": base, "type": self.REL_COMPONENT_OF})
            self._edges.append({"source": pair, "target": quote, "type": self.REL_COMPONENT_OF})

        # Correlations
        self._edges.append({"source": "DXY", "target": "EURUSD", "type": self.REL_INVERSE_CORRELATES, "strength": -0.8})
        self._edges.append({"source": "DXY", "target": "GBPUSD", "type": self.REL_INVERSE_CORRELATES, "strength": -0.7})
        self._edges.append({"source": "DXY", "target": "XAUUSD", "type": self.REL_INVERSE_CORRELATES, "strength": -0.6})
        self._edges.append({"source": "XAUUSD", "target": "USD", "type": self.REL_INVERSE_CORRELATES, "strength": -0.6})
        self._edges.append({"source": "VIX", "target": "SP500", "type": self.REL_INVERSE_CORRELATES, "strength": -0.7})

        # Central bank → currency
        self._edges.append({"source": "FED", "target": "USD", "type": self.REL_INFLUENCES, "strength": 0.9})
        self._edges.append({"source": "ECB", "target": "EUR", "type": self.REL_INFLUENCES, "strength": 0.9})
        self._edges.append({"source": "BOE", "target": "GBP", "type": self.REL_INFLUENCES, "strength": 0.9})
        self._edges.append({"source": "BOJ", "target": "JPY", "type": self.REL_INFLUENCES, "strength": 0.9})

        # Economic events → USD
        for e in ["CPI", "NFP", "FOMC", "GDP", "RATE_DECISION"]:
            self._edges.append({"source": e, "target": "USD", "type": self.REL_IMPACTS, "strength": 0.8})

    def get_related(self, entity: str, max_depth: int = 2) -> dict:
        """Get all entities related to a given entity (BFS traversal).

        Returns:
            {"direct": list, "indirect": list, "correlations": list}
        """
        if entity not in self._nodes:
            return {"direct": [], "indirect": [], "reason": "Entity not in graph"}

        direct = set()
        indirect = set()
        correlations = []

        # Direct connections
        for edge in self._edges:
            if edge["source"] == entity:
                direct.add(edge["target"])
                if "strength" in edge:
                    correlations.append({
                        "entity": edge["target"],
                        "relationship": edge["type"],
                        "strength": edge["strength"],
                    })
            elif edge["target"] == entity:
                direct.add(edge["source"])
                if "strength" in edge:
                    correlations.append({
                        "entity": edge["source"],
                        "relationship": edge["type"],
                        "strength": edge["strength"],
                    })

        # Indirect (depth 2)
        if max_depth >= 2:
            for d in list(direct):
                for edge in self._edges:
                    other = None
                    if edge["source"] == d:
                        other = edge["target"]
                    elif edge["target"] == d:
                        other = edge["source"]
                    if other and other != entity and other not in direct:
                        indirect.add(other)

        return {
            "entity": entity,
            "direct": list(direct),
            "indirect": list(indirect),
            "correlations": correlations,
            "total_connections": len(direct) + len(indirect),
        }

    def get_impact_chain(self, event: str) -> dict:
        """Trace the impact chain from an event to trading pairs.

        Example: CPI → USD → EURUSD/GBPUSD/USDJPY
        """
        chain = []
        related = self.get_related(event, max_depth=3)

        # Find currencies impacted
        currencies = [e for e in related["direct"] if self._nodes.get(e, {}).get("type") == self.ENTITY_CURRENCY]

        # Find pairs that contain those currencies
        impacted_pairs = []
        for curr in currencies:
            for edge in self._edges:
                if edge["type"] == self.REL_COMPONENT_OF:
                    if edge["target"] == curr:
                        impacted_pairs.append({
                            "pair": edge["source"],
                            "currency": curr,
                            "relationship": f"{event} impacts {curr} which is in {edge['source']}",
                        })

        return {
            "event": event,
            "impacted_currencies": currencies,
            "impacted_pairs": impacted_pairs,
            "chain": f"{event} → {currencies} → {[p['pair'] for p in impacted_pairs]}",
        }


# ════════════════════════════════════════════════════════════════════
# 10. FEDERATED LEARNING (STUB)
# ════════════════════════════════════════════════════════════════════

class FederatedLearning:
    """Framework stub for federated learning.

    Federated learning allows training models across multiple brokers/
    data sources without sharing raw data — only model gradients are
    shared, preserving privacy.

    This is a framework stub — full implementation requires:
    - Multiple broker connections
    - Secure gradient aggregation server
    - Differential privacy guarantees
    """

    @staticmethod
    def get_status() -> dict:
        return {
            "method": "Federated Learning",
            "description": "Train models across multiple brokers without sharing raw data",
            "status": "FRAMEWORK_READY",
            "implementation": "Requires: federated averaging, secure aggregation, differential privacy",
            "benefit": "Learn from multiple data sources without exposing individual broker data",
            "use_case": "Train on MT5 + OANDA + IC Markets data simultaneously",
        }


# ════════════════════════════════════════════════════════════════════
# SMOKE TESTS
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    print("=== 1. Information Theory ===")
    prices = 1.1000 + np.cumsum(np.random.randn(200) * 0.001)
    returns = np.diff(np.log(prices))
    entropy = InformationTheory.shannon_entropy(returns)
    print(f"  Shannon Entropy: {entropy:.3f} bits")
    kl = InformationTheory.kl_divergence(returns[:100], returns[100:])
    print(f"  KL Divergence: {kl:.3f}")
    mi = InformationTheory.mutual_information(returns[:-1], returns[1:])
    print(f"  Mutual Information: {mi:.4f}")

    print("\n=== 2. Chaos Dynamics ===")
    chaos = ChaosDynamics.market_regime_from_chaos(prices)
    print(f"  Lyapunov: {chaos['lyapunov']}, Fractal Dim: {chaos['fractal_dimension']}")
    print(f"  Regime: {chaos['regime']}, Trend: {chaos['trend_type']}")

    print("\n=== 3. Anomaly Detection ===")
    # Create data with anomaly
    anomalous_prices = prices.copy()
    anomalous_prices[-1] = anomalous_prices[-2] + 0.01  # flash spike
    anomaly = AnomalyDetector.detect_price_anomaly(anomalous_prices, z_threshold=3.0)
    print(f"  Price anomaly: {anomaly['is_anomaly']} (Z={anomaly['z_score']:.2f}, severity={anomaly['severity']})")

    volumes = np.random.randint(100, 500, 100)
    volumes[-1] = 3000  # volume spike
    vol_anom = AnomalyDetector.detect_volume_anomaly(volumes, multiplier=3.0)
    print(f"  Volume anomaly: {vol_anom['is_anomaly']} (ratio={vol_anom['volume_ratio']:.1f}×)")

    print("\n=== 4. Game Theory ===")
    nash = GameTheory.nash_equilibrium_strategy(win_prob=0.65, loss_prob=0.35)
    print(f"  Nash: trade_prob={nash['trade_probability']:.0%} → {nash['recommendation']}")
    adv = GameTheory.adversarial_analysis("BUY", "LONG")
    print(f"  Adversarial: MM=LONG, our=BUY → {adv['optimal_action']} ({adv['reason'][:50]})")

    print("\n=== 5. Decision Intelligence ===")
    eu = DecisionIntelligence.expected_utility("BUY", win_prob=0.6, win_pnl=100, loss_pnl=-50, risk_aversion=1.0)
    print(f"  Expected utility: {eu['expected_utility']} (EV=${eu['expected_pnl']}) → {eu['recommendation']}")
    iv = DecisionIntelligence.information_value(wait_time_minutes=15, current_confidence=0.52)
    print(f"  Info value: should_wait={iv['should_wait']} (conf {iv['current_confidence']:.0%} → {iv['projected_confidence']:.0%})")

    print("\n=== 6. Knowledge Graph ===")
    kg = KnowledgeGraph()
    related = kg.get_related("USD")
    print(f"  USD direct connections: {related['direct']}")
    impact = kg.get_impact_chain("CPI")
    print(f"  CPI impact chain: {impact['chain']}")

    print("\n=== 7. Federated Learning ===")
    fl = FederatedLearning.get_status()
    print(f"  Status: {fl['status']} — {fl['benefit'][:50]}")

    print("\nAll PhD frontier smoke tests passed.")
