# analysis/dna_journal.py
# ============================================================
# Market DNA — cluster trade-journal statistics.
#
# Policy from architecture review (non-negotiable):
#
#   REBUILD, DON'T MIGRATE. When a MarketDNADetector is refit,
#   old cluster IDs are NOT mapped onto new cluster IDs. A new
#   HDBSCAN fit can shift cluster *boundaries*, not just labels —
#   so "Cluster 5 becomes Cluster 13" is not just a renaming
#   problem, membership itself can change. Every time a new
#   model_id is frozen, its journal is rebuilt from scratch by
#   re-running historical trades through the NEW detector's
#   predict_live(). This costs compute, not correctness.
#
#   EVIDENCE TIERS, NOT A SINGLE THRESHOLD. Point-estimate win
#   rate on <300 trades is close to noise. Every stat surfaced
#   downstream carries a Wilson-score confidence interval and an
#   explicit tier so callers can tell "no data" from "bad edge"
#   from "reliable edge".
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import beta

from utils.logger import get_logger

log = get_logger(__name__)


# ── Evidence tiers (from review round 3) ────────────────────
TRADE_COUNT_TIERS = (
    (0, 100, "NO_STATISTICAL_EDGE"),
    (100, 300, "WEAK_EVIDENCE"),
    (300, 1000, "RELIABLE"),
    (1000, float("inf"), "HIGH_CONFIDENCE"),
)

# Position multiplier by tier — conservative default; a system can
# override with values informed by its own capital/risk policy, but
# NO_STATISTICAL_EDGE must never exceed 1.0x and should generally be
# well below it.
DEFAULT_TIER_MULTIPLIER = {
    "NO_STATISTICAL_EDGE": 0.25,
    "WEAK_EVIDENCE": 0.5,
    "RELIABLE": 1.0,
    "HIGH_CONFIDENCE": 1.0,   # upside is capped elsewhere (win-rate scaling), not here
}


def trade_count_tier(n_trades: int) -> str:
    for lo, hi, tier in TRADE_COUNT_TIERS:
        if lo <= n_trades < hi:
            return tier
    return "NO_STATISTICAL_EDGE"


def wilson_ci(wins: int, total: int, alpha: float = 0.05) -> tuple:
    """
    Beta-distribution (Jeffreys-interval-style) 95% CI for a win
    rate. More honest than a bare point estimate at low sample
    sizes, which is exactly where this module is most likely to be
    consulted for a rarely-seen cluster.
    """
    if total == 0:
        return (0.0, 1.0)
    lo, hi = beta.ppf([alpha / 2, 1 - alpha / 2], wins + 0.5, total - wins + 0.5)
    return (float(lo), float(hi))


@dataclass
class ClusterStats:
    model_id: str
    cluster_id: int
    trades: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    profit_factor: Optional[float]
    expectancy_r: Optional[float]     # expectancy in R-multiples, if available
    avg_holding_bars: Optional[float]
    tier: str
    position_multiplier: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def build_cluster_journal(
    trades_with_clusters: pd.DataFrame,
    model_id: str,
    *,
    result_col: str = "result",       # expects 'WIN' / 'LOSS' (or truthy pnl)
    pnl_col: str = "pnl",
    cluster_col: str = "cluster_id",
    tier_multiplier: dict = None,
) -> list[ClusterStats]:
    """
    Rebuild cluster statistics FROM SCRATCH for a given model_id.

    `trades_with_clusters` must already have a `cluster_col` assigned
    by running each historical trade's entry-bar features through
    THIS model_id's MarketDNADetector.predict_live() — this function
    does no clustering itself, it only aggregates.

    Rows where cluster_col is null/UNKNOWN should be excluded before
    calling this (UNKNOWN has no meaningful per-cluster stats by
    definition).
    """
    tier_multiplier = tier_multiplier or DEFAULT_TIER_MULTIPLIER
    stats: list[ClusterStats] = []

    df = trades_with_clusters.dropna(subset=[cluster_col])
    for cid, g in df.groupby(cluster_col):
        n = len(g)
        wins_mask = (g[result_col].astype(str).str.upper() == "WIN") if result_col in g else (g[pnl_col] > 0)
        wins = int(wins_mask.sum())
        win_rate = wins / n if n else 0.0
        ci_low, ci_high = wilson_ci(wins, n)

        gross_profit = g.loc[g[pnl_col] > 0, pnl_col].sum() if pnl_col in g else np.nan
        gross_loss = -g.loc[g[pnl_col] < 0, pnl_col].sum() if pnl_col in g else np.nan
        profit_factor = (
            float(gross_profit / gross_loss)
            if pnl_col in g and gross_loss and gross_loss > 0
            else None
        )
        expectancy_r = float(g[pnl_col].mean()) if pnl_col in g and n else None

        tier = trade_count_tier(n)
        stats.append(ClusterStats(
            model_id=model_id,
            cluster_id=int(cid),
            trades=n,
            wins=wins,
            win_rate=round(win_rate, 4),
            ci_low=round(ci_low, 4),
            ci_high=round(ci_high, 4),
            profit_factor=None if profit_factor is None else round(profit_factor, 3),
            expectancy_r=None if expectancy_r is None else round(expectancy_r, 5),
            avg_holding_bars=None,
            tier=tier,
            position_multiplier=tier_multiplier.get(tier, 0.25),
        ))

    log.info(
        f"[dna_journal] Rebuilt journal for model_id={model_id}: "
        f"{len(stats)} clusters, {len(df)} trades aggregated."
    )
    return stats


def lookup(stats: list[ClusterStats], cluster_id: int) -> Optional[ClusterStats]:
    for s in stats:
        if s.cluster_id == cluster_id:
            return s
    return None


def decision_context(cluster_stat: Optional[ClusterStats]) -> dict:
    """
    Translate a ClusterStats row into the ONLY vocabulary this
    module is allowed to speak in: known/unknown, reliable/
    unreliable, and a position-size multiplier. Never BUY/SELL.
    """
    if cluster_stat is None:
        return {
            "state": "UNKNOWN",
            "recommendation": "REDUCE_SIZE",
            "position_multiplier": DEFAULT_TIER_MULTIPLIER["NO_STATISTICAL_EDGE"],
            "reason": "no journal entry for this cluster yet",
        }
    if cluster_stat.tier == "NO_STATISTICAL_EDGE":
        rec = "REDUCE_SIZE"
    elif cluster_stat.expectancy_r is not None and cluster_stat.expectancy_r <= 0:
        # Expectancy (mean R-multiple per trade) is the correct
        # profitability test — unlike a flat 50% win-rate check, it's
        # valid regardless of the underlying strategy's reward:risk
        # shape. A strategy with a 2.5:1.5 reward:risk ratio, for
        # example, is profitable well below a 50% win rate; comparing
        # win_rate/ci_high to a flat coin-flip line would reject
        # genuinely profitable clusters like that.
        rec = "REJECT"
    elif cluster_stat.expectancy_r is None and cluster_stat.ci_high < 0.5:
        # No per-trade R data available (e.g. result-only history) —
        # fall back to the win-rate-vs-coinflip heuristic, which is
        # only meaningful when we don't know the payoff shape.
        rec = "REJECT"
    else:
        rec = "APPROVE"

    return {
        "state": "KNOWN",
        "cluster_id": cluster_stat.cluster_id,
        "tier": cluster_stat.tier,
        "win_rate": cluster_stat.win_rate,
        "ci": [cluster_stat.ci_low, cluster_stat.ci_high],
        "profit_factor": cluster_stat.profit_factor,
        "recommendation": rec,
        "position_multiplier": cluster_stat.position_multiplier,
    }