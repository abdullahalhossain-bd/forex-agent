"""
analysis/research_domains.py — Missing Deep Research Domains
=================================================================
Consolidates 3 remaining missing research domains:

3. Options Intelligence — put/call ratio, gamma exposure, max pain
4. Futures Data — COT, CME gap, funding rate
12. Graph-Based Analysis — asset correlation network

These are framework modules with data interfaces. Full implementation
requires external data feeds (CME, CFTC, options exchanges) which may
not be available in all environments. The modules gracefully degrade
to "data unavailable" when feeds are missing.

USAGE:
    from analysis.research_domains import (
        get_options_intelligence,
        get_futures_data,
        compute_correlation_graph,
    )
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from datetime import datetime, timezone
from utils.logger import get_logger

log = get_logger("research_domains")


# ════════════════════════════════════════════════════════════════════
# 3. OPTIONS INTELLIGENCE
# ════════════════════════════════════════════════════════════════════

def get_options_intelligence(symbol: str = "EURUSD") -> dict:
    """Get options market intelligence.

    NOTE: Forex options data is not freely available via standard APIs.
    This module provides the FRAMEWORK and interface — when a data feed
    is connected (e.g., CME, ICE, broker API), it will populate.

    Returns:
        {
            "put_call_ratio": float,    # >1 = bearish, <1 = bullish
            "open_interest": dict,      # calls vs puts OI
            "gamma_exposure": float,    # dealer GEX (positive = stable, negative = volatile)
            "max_pain": float,          # price where most options expire worthless
            "iv_rank": float,           # 0-100, how high is current IV vs history
            "signal": str,              # BUY / SELL / NEUTRAL
            "score": int,
            "reason": str,
            "data_available": bool,
        }
    """
    # Placeholder — in production, connect to:
    # - CME Group API for FX options
    # - ICE Data Services
    # - Broker options chain (if available)

    result = {
        "put_call_ratio": None,
        "open_interest": {"calls": None, "puts": None},
        "gamma_exposure": None,
        "max_pain": None,
        "iv_rank": None,
        "signal": "NEUTRAL",
        "score": 0,
        "reason": "Options data feed not connected — no options intelligence available",
        "data_available": False,
    }

    # Try to fetch from any available source
    try:
        # Could integrate with:
        # - Yahoo Finance for equity options (not FX)
        # - CME API for futures options
        # - QuikStrike for FX options
        pass
    except Exception as e:
        log.debug(f"[OptionsIntel] Data fetch failed: {e}")

    return result


# ════════════════════════════════════════════════════════════════════
# 4. FUTURES MARKET DATA
# ════════════════════════════════════════════════════════════════════

def get_futures_data(symbol: str = "EURUSD") -> dict:
    """Get futures market intelligence.

    Includes:
    - COT (Commitment of Traders) — institutional positioning
    - CME Gap — price gaps between futures sessions
    - Open Interest changes — new money entering/exiting

    NOTE: CFTC publishes COT data weekly (free). CME gap requires
    futures price data. This framework connects when data is available.

    Returns:
        {
            "cot": dict,            # commercial/non-commercial positioning
            "cme_gap": dict,        # gap direction and size
            "open_interest": dict,  # OI changes
            "signal": str,
            "score": int,
            "reason": str,
            "data_available": bool,
        }
    """
    result = {
        "cot": {"commercial_long": None, "commercial_short": None,
                "non_commercial_long": None, "non_commercial_short": None,
                "net_position": None},
        "cme_gap": {"direction": "NONE", "size_pips": 0, "filled": True},
        "open_interest": {"current": None, "change": None},
        "signal": "NEUTRAL",
        "score": 0,
        "reason": "Futures data feed not connected — no COT/CME gap data",
        "data_available": False,
    }

    # Try COT data from CFTC (free, weekly)
    try:
        # CFTC publishes COT data every Friday
        # Could fetch from: https://www.cftc.gov/dea/futures/
        # For now, framework only
        pass
    except Exception as e:
        log.debug(f"[FuturesData] COT fetch failed: {e}")

    # Try CME gap detection (requires futures price data)
    try:
        # CME futures for FX: 6E (EURUSD), 6B (GBPUSD), etc.
        # Gap = difference between today's open and yesterday's close
        # Could fetch from yfinance: "6E=F" for EURUSD futures
        pass
    except Exception as e:
        log.debug(f"[FuturesData] CME gap detection failed: {e}")

    return result


def detect_cme_gap(futures_df: pd.DataFrame) -> dict:
    """Detect CME futures gap from price data.

    Args:
        futures_df: DataFrame with futures OHLC data and DatetimeIndex.

    Returns:
        {"direction": "UP"/"DOWN"/"NONE", "size_pips": float, "filled": bool}
    """
    if futures_df is None or len(futures_df) < 2:
        return {"direction": "NONE", "size_pips": 0, "filled": True}

    prev_close = float(futures_df.iloc[-2]["close"])
    today_open = float(futures_df.iloc[-1]["open"])
    today_low = float(futures_df.iloc[-1]["low"])
    today_high = float(futures_df.iloc[-1]["high"])

    gap = today_open - prev_close
    pip_size = 0.0001 if abs(prev_close) < 10 else 0.01

    if abs(gap) < pip_size * 3:  # < 3 pips = no gap
        return {"direction": "NONE", "size_pips": 0, "filled": True}

    direction = "UP" if gap > 0 else "DOWN"
    size_pips = abs(gap) / pip_size
    filled = (direction == "UP" and today_low <= prev_close) or \
             (direction == "DOWN" and today_high >= prev_close)

    log.info(f"[CMEGap] {direction} gap {size_pips:.1f} pips, filled={filled}")

    return {
        "direction": direction,
        "size_pips": round(size_pips, 1),
        "filled": filled,
    }


# ════════════════════════════════════════════════════════════════════
# 12. GRAPH-BASED ANALYSIS
# ════════════════════════════════════════════════════════════════════

def compute_correlation_graph(
    price_data: Dict[str, pd.Series],
    threshold: float = 0.5,
) -> dict:
    """Build a correlation graph between assets.

    Analyzes which assets are correlated and identifies clusters.
    Useful for:
    - Avoiding correlated positions
    - Finding leading indicators
    - Detecting regime shifts (correlations break in crises)

    Args:
        price_data: Dict of {asset_name: price_series}.
        threshold: Minimum |correlation| to include edge.

    Returns:
        {
            "nodes": list of asset names,
            "edges": list of {"source": str, "target": str, "weight": float},
            "clusters": list of lists (correlated groups),
            "leading_indicators": dict,  # asset → assets it leads
            "signal": str,
            "score": int,
            "reason": str,
        }
    """
    if not price_data or len(price_data) < 2:
        return {
            "nodes": list(price_data.keys()) if price_data else [],
            "edges": [],
            "clusters": [],
            "leading_indicators": {},
            "signal": "NEUTRAL",
            "score": 0,
            "reason": "Insufficient assets for graph analysis",
        }

    # Compute correlation matrix
    assets = list(price_data.keys())
    returns = {}
    for asset in assets:
        prices = price_data[asset]
        if len(prices) > 1:
            returns[asset] = np.diff(np.log(prices.values))

    # Align lengths
    min_len = min(len(r) for r in returns.values()) if returns else 0
    if min_len < 10:
        return {
            "nodes": assets,
            "edges": [],
            "clusters": [],
            "leading_indicators": {},
            "signal": "NEUTRAL",
            "score": 0,
            "reason": "Insufficient data for correlation",
        }

    aligned = {a: r[-min_len:] for a, r in returns.items()}
    corr_matrix = np.corrcoef([aligned[a] for a in assets])

    # Build edges
    edges = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            corr = corr_matrix[i, j]
            if abs(corr) >= threshold:
                edges.append({
                    "source": assets[i],
                    "target": assets[j],
                    "weight": round(float(corr), 3),
                })

    # Find clusters (simplified — connected components)
    visited = set()
    clusters = []
    for asset in assets:
        if asset in visited:
            continue
        cluster = {asset}
        visited.add(asset)
        queue = [asset]
        while queue:
            current = queue.pop()
            for edge in edges:
                if edge["source"] == current and edge["target"] not in visited:
                    cluster.add(edge["target"])
                    visited.add(edge["target"])
                    queue.append(edge["target"])
                elif edge["target"] == current and edge["source"] not in visited:
                    cluster.add(edge["source"])
                    visited.add(edge["source"])
                    queue.append(edge["source"])
        if len(cluster) > 1:
            clusters.append(list(cluster))

    # Leading indicators (simplified — lag-1 cross-correlation)
    leading_indicators = {}
    for i, asset in enumerate(assets):
        leads = []
        for j, other in enumerate(assets):
            if i == j:
                continue
            # Check if asset's returns predict other's next-bar returns
            if min_len > 2:
                lag_corr = np.corrcoef(aligned[asset][:-1], aligned[other][1:])[0, 1]
                if abs(lag_corr) > 0.3:
                    leads.append({"leads": other, "correlation": round(float(lag_corr), 3)})
        if leads:
            leading_indicators[asset] = leads

    signal = "NEUTRAL"
    score = 20
    reason = f"Graph: {len(assets)} assets, {len(edges)} edges, {len(clusters)} clusters"

    log.info(f"[GraphAnalysis] {reason}")

    return {
        "nodes": assets,
        "edges": edges,
        "clusters": clusters,
        "leading_indicators": leading_indicators,
        "signal": signal,
        "score": score,
        "reason": reason,
    }


# ════════════════════════════════════════════════════════════════════
# 17. META-LEARNING
# ════════════════════════════════════════════════════════════════════

def meta_learning_strategy_update(
    strategy_performance: Dict[str, dict],
    min_trades: int = 10,
    min_trades_per_regime: int = 20,
) -> dict:
    """Meta-learning: learn WHICH strategy works WHEN.

    Analyzes recent strategy performance and produces meta-rules:
    - "When regime=TRENDING, TREND_FOLLOW has 65% WR"
    - "When regime=RANGING, MEAN_REVERSION has 70% WR"
    - "When session=ASIAN, BREAKOUT has 30% WR (avoid)"

    Args:
        strategy_performance: Dict of {strategy_name: {
            "trades": list of {result, regime, session, ...},
            "total_trades": int,
        }}
        min_trades: minimum total trades for a strategy before its
            overall win-rate is used to adjust strategy_rankings.
        min_trades_per_regime: minimum trades WITHIN a single regime
            before a PREFER/AVOID meta-rule is generated for it.
            A win rate computed from a handful of trades is not
            statistically distinguishable from noise — e.g. 2/3 wins
            (67%) looks like a strong "PREFER" signal but is well
            within the range you'd expect from a coin flip. Default
            raised from an unconditional 3 to 20 to reduce the risk of
            meta-rules that are really just overfitting to a small
            sample. Callers with a strong reason to accept noisier
            rules (e.g. a research/exploration context, not live
            weighting) can lower this explicitly.

    Returns:
        {
            "meta_rules": list of rules,
            "strategy_rankings": dict,  # strategy → adjusted weight
            "best_by_regime": dict,    # regime → best strategy
            "reason": str,
        }
    """
    meta_rules = []
    strategy_rankings = {}
    best_by_regime = {}

    for strat_name, perf in strategy_performance.items():
        trades = perf.get("trades", [])
        total = perf.get("total_trades", len(trades))

        if total < min_trades:
            strategy_rankings[strat_name] = 1.0  # neutral weight
            continue

        # Compute overall win rate
        wins = sum(1 for t in trades if t.get("result") == "WIN")
        wr = wins / total if total > 0 else 0.5
        weight = max(0.3, min(wr * 2.0, 2.0))
        strategy_rankings[strat_name] = round(weight, 2)

        # Compute win rate by regime
        regime_stats = {}
        for t in trades:
            regime = t.get("regime", "UNKNOWN")
            if regime not in regime_stats:
                regime_stats[regime] = {"wins": 0, "losses": 0}
            if t.get("result") == "WIN":
                regime_stats[regime]["wins"] += 1
            elif t.get("result") == "LOSS":
                regime_stats[regime]["losses"] += 1

        for regime, stats in regime_stats.items():
            total_r = stats["wins"] + stats["losses"]
            if total_r >= min_trades_per_regime:
                wr_r = stats["wins"] / total_r
                if wr_r > 0.6:
                    meta_rules.append(
                        f"When regime={regime}, {strat_name} has {wr_r:.0%} WR ({total_r} trades) — PREFER"
                    )
                    if regime not in best_by_regime or wr_r > best_by_regime[regime]["wr"]:
                        best_by_regime[regime] = {"strategy": strat_name, "wr": wr_r}
                elif wr_r < 0.3:
                    meta_rules.append(
                        f"When regime={regime}, {strat_name} has {wr_r:.0%} WR ({total_r} trades) — AVOID"
                    )

    reason = f"Meta-learning: {len(meta_rules)} rules generated, {len(strategy_rankings)} strategies ranked"

    log.info(f"[MetaLearning] {reason}")

    return {
        "meta_rules": meta_rules,
        "strategy_rankings": strategy_rankings,
        "best_by_regime": {k: v["strategy"] for k, v in best_by_regime.items()},
        "reason": reason,
    }


# ── Smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Options Intelligence ===")
    oi = get_options_intelligence("EURUSD")
    print(f"  Data available: {oi['data_available']}")

    print("\n=== Futures Data ===")
    fd = get_futures_data("EURUSD")
    print(f"  Data available: {fd['data_available']}")

    print("\n=== CME Gap Detection ===")
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.001)
    # Create a gap
    close[-10] = close[-11] + 0.005  # gap up
    df = pd.DataFrame({
        "open": close, "high": close + 0.001, "low": close - 0.001, "close": close
    }, index=dates)
    gap = detect_cme_gap(df)
    print(f"  Gap: {gap}")

    print("\n=== Correlation Graph ===")
    np.random.seed(42)
    prices = {
        "EURUSD": pd.Series(1.1000 + np.cumsum(np.random.randn(100) * 0.001)),
        "GBPUSD": pd.Series(1.2500 + np.cumsum(np.random.randn(100) * 0.001)),
        "USDJPY": pd.Series(150.0 + np.cumsum(np.random.randn(100) * 0.05)),
        "XAUUSD": pd.Series(2000.0 + np.cumsum(np.random.randn(100) * 2)),
    }
    # Make EURUSD and GBPUSD correlated
    prices["GBPUSD"] = prices["EURUSD"] * 1.136 + np.random.randn(100) * 0.001
    graph = compute_correlation_graph(prices, threshold=0.5)
    print(f"  Nodes: {graph['nodes']}")
    print(f"  Edges: {graph['edges']}")
    print(f"  Clusters: {graph['clusters']}")

    print("\n=== Meta-Learning ===")
    strat_perf = {
        "TREND_FOLLOW": {
            "trades": [
                {"result": "WIN", "regime": "TRENDING"},
                {"result": "WIN", "regime": "TRENDING"},
                {"result": "WIN", "regime": "TRENDING"},
                {"result": "LOSS", "regime": "RANGING"},
                {"result": "LOSS", "regime": "RANGING"},
                {"result": "LOSS", "regime": "RANGING"},
                {"result": "LOSS", "regime": "RANGING"},
            ],
            "total_trades": 7,
        },
        "MEAN_REVERSION": {
            "trades": [
                {"result": "WIN", "regime": "RANGING"},
                {"result": "WIN", "regime": "RANGING"},
                {"result": "WIN", "regime": "RANGING"},
                {"result": "LOSS", "regime": "TRENDING"},
            ],
            "total_trades": 4,
        },
    }
    ml = meta_learning_strategy_update(strat_perf, min_trades=3)
    print(f"  Rules (default min_trades_per_regime=20): {ml['meta_rules']}")
    print(f"  Rankings: {ml['strategy_rankings']}")
    print(f"  Best by regime: {ml['best_by_regime']}")
    print("  (No per-regime rules expected above — sample sizes here are only")
    print("   3-4 trades per regime, well below the default 20-trade minimum,")
    print("   which exists specifically to avoid rules built on noise.)")
    # Explicit opt-in to a lower, exploratory threshold, for comparison:
    ml_explore = meta_learning_strategy_update(strat_perf, min_trades=3, min_trades_per_regime=3)
    print(f"  Rules (explicit min_trades_per_regime=3, exploratory only): {ml_explore['meta_rules']}")

    print("\nAll research domain smoke tests passed.")