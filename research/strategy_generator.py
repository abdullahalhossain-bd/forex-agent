"""
research/strategy_generator.py — Minimal stub (Day 57 placeholder)
===================================================================

This file exists to satisfy the imports in:

    research/research_agent.py        : from research.strategy_generator import StrategyGenerator
    research/experiment_runner.py     : from research.strategy_generator import StrategyGenerator
    research/hypothesis_engine.py     : from research.strategy_generator import FILTER_COMPONENTS, ENTRY_COMPONENTS
    tests/test_pipeline.py            : from research.strategy_generator import ...

The full StrategyGenerator logic was never committed upstream. This stub
provides the minimum API surface (class + the two component registries)
so all three modules import cleanly. Strategy generation returns a simple
rule-based strategy that the experiment runner can backtest.

Marked LEGACY_STUB (created to unblock imports).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


# ── Component registries ────────────────────────────────────────────
# These power HypothesisEngine's filter_addition / filter_removal /
# entry_combination templates. Each entry is {name, description, callable}.

FILTER_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "min_atr": {
        "name": "min_atr",
        "description": "Skip setups where ATR(pips) < threshold.",
        "params": {"atr_pips": 8},
        "callable": lambda df, ctx: df["atr"] >= ctx.get("atr_pips", 8),
    },
    "trend_filter": {
        "name": "trend_filter",
        "description": "Only trade in EMA(50) direction.",
        "params": {"ema_period": 50},
        "callable": lambda df, ctx: (
            (df["close"] > df["ema_50"]) | (df["close"] < df["ema_50"])
        ),
    },
    "session_filter": {
        "name": "session_filter",
        "description": "Restrict trading to London + NY sessions.",
        "params": {"sessions": ["london", "new_york"]},
        "callable": lambda df, ctx: df["session"].isin(ctx.get("sessions", [])),
    },
    "news_filter": {
        "name": "news_filter",
        "description": "Skip entries within ±30min of high-impact news.",
        "params": {"window_minutes": 30},
        "callable": lambda df, ctx: ~df.get("news_blackout", False),
    },
    "rr_min": {
        "name": "rr_min",
        "description": "Reject setups with R:R below threshold.",
        "params": {"min_rr": 1.5},
        "callable": lambda df, ctx: df["rr"] >= ctx.get("min_rr", 1.5),
    },
}

ENTRY_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "ema_crossover": {
        "name": "ema_crossover",
        "description": "BUY when EMA(20) crosses above EMA(50); SELL when below.",
        "params": {"fast": 20, "slow": 50},
        "callable": lambda df, ctx: (
            (df["ema_20"] > df["ema_50"]).astype(int) - (df["ema_20"] < df["ema_50"]).astype(int)
        ),
    },
    "rsi_extreme": {
        "name": "rsi_extreme",
        "description": "BUY when RSI < 30; SELL when RSI > 70.",
        "params": {"oversold": 30, "overbought": 70},
        "callable": lambda df, ctx: (
            (df["rsi"] < ctx.get("oversold", 30)).astype(int) -
            (df["rsi"] > ctx.get("overbought", 70)).astype(int)
        ),
    },
    "breakout": {
        "name": "breakout",
        "description": "BUY on break of 20-bar high; SELL on break of 20-bar low.",
        "params": {"window": 20},
        "callable": lambda df, ctx: (
            (df["close"] > df["high"].rolling(ctx.get("window", 20)).max().shift(1)).astype(int) -
            (df["close"] < df["low"].rolling(ctx.get("window", 20)).min().shift(1)).astype(int)
        ),
    },
    "engulfing": {
        "name": "engulfing",
        "description": "BUY on bullish engulfing; SELL on bearish engulfing.",
        "params": {},
        "callable": lambda df, ctx: df.get("engulfing_signal", 0),
    },
    "smc_bos": {
        "name": "smc_bos",
        "description": "Enter on SMC Break-of-Structure event.",
        "params": {},
        "callable": lambda df, ctx: df.get("bos_signal", 0),
    },
}


@dataclass
class GeneratedStrategy:
    """A strategy produced by StrategyGenerator."""
    id: str
    name: str
    description: str
    filters: List[str] = field(default_factory=list)
    entries: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "filters": list(self.filters),
            "entries": list(self.entries),
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }


class StrategyGenerator:
    """Generates trading strategies by combining filter and entry components.

    The full version would use evolutionary search / Bayesian optimization.
    This stub supports three operations:
      * random_strategy()    — pick random filters + entries
      * mutate(strategy)     — add/remove one component
      * combine(s1, s2)      — union of components
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = __import__("random").Random(seed)

    def random_strategy(self, name: Optional[str] = None) -> GeneratedStrategy:
        n_filters = self.rng.randint(1, min(3, len(FILTER_COMPONENTS)))
        n_entries = self.rng.randint(1, min(2, len(ENTRY_COMPONENTS)))
        filters = self.rng.sample(sorted(FILTER_COMPONENTS.keys()), n_filters)
        entries = self.rng.sample(sorted(ENTRY_COMPONENTS.keys()), n_entries)
        return GeneratedStrategy(
            id=str(uuid.uuid4())[:8],
            name=name or f"gen_{'_'.join(entries)}",
            description=f"Generated: filters={filters}, entries={entries}",
            filters=filters,
            entries=entries,
            params={},
        )

    def mutate(self, strategy: GeneratedStrategy) -> GeneratedStrategy:
        """Return a mutated copy with one component added or removed."""
        new_filters = list(strategy.filters)
        new_entries = list(strategy.entries)
        if self.rng.random() < 0.5 and FILTER_COMPONENTS:
            unused = [f for f in FILTER_COMPONENTS if f not in new_filters]
            if unused and self.rng.random() < 0.5:
                new_filters.append(self.rng.choice(unused))
            elif new_filters:
                new_filters.remove(self.rng.choice(new_filters))
        elif ENTRY_COMPONENTS:
            unused = [e for e in ENTRY_COMPONENTS if e not in new_entries]
            if unused and self.rng.random() < 0.5:
                new_entries.append(self.rng.choice(unused))
            elif new_entries:
                new_entries.remove(self.rng.choice(new_entries))
        return GeneratedStrategy(
            id=str(uuid.uuid4())[:8],
            name=f"{strategy.name}_mut",
            description=f"Mutated from {strategy.id}",
            filters=new_filters,
            entries=new_entries,
            params=dict(strategy.params),
        )

    def combine(self, s1: GeneratedStrategy, s2: GeneratedStrategy) -> GeneratedStrategy:
        """Union of two strategies' components."""
        return GeneratedStrategy(
            id=str(uuid.uuid4())[:8],
            name=f"{s1.name}+{s2.name}",
            description=f"Combination of {s1.id} and {s2.id}",
            filters=sorted(set(s1.filters) | set(s2.filters)),
            entries=sorted(set(s1.entries) | set(s2.entries)),
            params={**s1.params, **s2.params},
        )

    def list_components(self) -> Dict[str, List[str]]:
        return {
            "filters": sorted(FILTER_COMPONENTS.keys()),
            "entries": sorted(ENTRY_COMPONENTS.keys()),
        }
