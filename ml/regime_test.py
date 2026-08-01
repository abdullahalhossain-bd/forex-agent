"""
ml/regime_test.py — Market Regime Robustness Test (Day 72)
=============================================================

Tests whether a model/strategy performs consistently across different
market regimes:
  - Trending (strong directional, ADX > 25)
  - Ranging (sideways, ADX < 20)
  - High volatility (ATR > 1.5× median)
  - Low volatility (ATR < 0.7× median)

A model that only works in trending markets but fails in ranging is
OVERFIT to one regime. A robust model works (even if not equally well)
across ALL regimes.

Output per regime:
    {profit_factor, win_rate, trade_count}

Final score: average PF across all regimes × consistency bonus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("regime_test")


@dataclass
class RegimeResult:
    """One regime's performance."""
    regime: str
    trade_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


@dataclass
class RegimeTestResult:
    """Aggregated regime robustness results."""
    regimes: List[RegimeResult] = field(default_factory=list)
    avg_profit_factor: float = 0.0
    min_profit_factor: float = 0.0
    consistency: float = 0.0          # std dev of PFs (lower = more consistent)
    regimes_passed: int = 0
    total_regimes: int = 0
    score: float = 0.0
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regimes": [r.to_dict() for r in self.regimes],
            "avg_profit_factor": round(self.avg_profit_factor, 3),
            "min_profit_factor": round(self.min_profit_factor, 3),
            "consistency": round(self.consistency, 3),
            "regimes_passed": self.regimes_passed,
            "total_regimes": self.total_regimes,
            "score": round(self.score, 1),
            "passed": self.passed,
        }


class RegimeTester:
    """Tests model robustness across market regimes."""

    def test(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        pnl_per_trade: Optional[List[float]] = None,
    ) -> RegimeTestResult:
        """Test strategy performance across regimes.

        Args:
            df: Original OHLCV dataframe (must have 'adx' and 'atr' columns,
                or will try to compute them).
            y_pred: Predicted signals (1=BUY, 0=SELL/no-trade).
            y_true: Actual labels.
            pnl_per_trade: Optional PnL per trade for profit factor calc.

        Returns:
            RegimeTestResult.
        """
        result = RegimeTestResult()

        if df is None or len(df) == 0:
            return result

        # Ensure ADX + ATR columns
        df = df.copy()
        if "adx" not in df.columns or "atr" not in df.columns:
            try:
                from data.indicators import Indicators
                df = Indicators().add_all(df)
            except Exception:
                pass

        # Classify each row into a regime
        regimes = self._classify_regimes(df)

        # Split predictions by regime
        regime_names = ["TRENDING", "RANGING", "HIGH_VOLATILITY", "LOW_VOLATILITY"]
        result.total_regimes = len(regime_names)

        pfs: List[float] = []

        for regime_name in regime_names:
            mask = regimes == regime_name
            if mask.sum() < 5:
                result.regimes.append(RegimeResult(regime=regime_name, trade_count=0))
                continue

            regime_pred = y_pred[mask]
            regime_true = y_true[mask]

            tp = int(np.sum((regime_pred == 1) & (regime_true == 1)))
            fp = int(np.sum((regime_pred == 1) & (regime_true == 0)))
            wr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            pf = tp / fp if fp > 0 else (float("inf") if tp > 0 else 0.0)
            pf = min(pf, 10.0)

            rr = RegimeResult(
                regime=regime_name, trade_count=int(mask.sum()),
                win_rate=wr, profit_factor=pf,
            )
            result.regimes.append(rr)
            pfs.append(pf)

            if pf >= 1.0:
                result.regimes_passed += 1

            log.info(f"[RegimeTest] {regime_name}: trades={mask.sum()} WR={wr:.1%} PF={pf:.2f}")

        if pfs:
            result.avg_profit_factor = float(np.mean(pfs))
            result.min_profit_factor = float(np.min(pfs))
            result.consistency = float(np.std(pfs))

            # Score: avg PF + min PF + consistency + pass ratio
            avg_pf_score = min(100, result.avg_profit_factor * 50)
            min_pf_score = min(100, result.min_profit_factor * 80)
            pass_ratio = (result.regimes_passed / result.total_regimes) * 100
            consistency_score = max(0, 100 - result.consistency * 50)
            result.score = (avg_pf_score * 0.3 + min_pf_score * 0.3 +
                           pass_ratio * 0.25 + consistency_score * 0.15)
            result.passed = (
                result.score >= 55
                and result.regimes_passed >= 2
                and result.min_profit_factor >= 0.8
            )

        log.info(
            f"[RegimeTest] {result.regimes_passed}/{result.total_regimes} regimes passed | "
            f"avg PF={result.avg_profit_factor:.2f} min PF={result.min_profit_factor:.2f} | "
            f"score={result.score:.1f} passed={result.passed}"
        )
        return result

    def _classify_regimes(self, df: pd.DataFrame) -> pd.Series:
        """Classify each row into a market regime."""
        n = len(df)
        regimes = pd.Series("RANGING", index=df.index)

        adx = df.get("adx", pd.Series(20, index=df.index))
        atr = df.get("atr", pd.Series(0.001, index=df.index))
        atr_median = atr.median() if len(atr) > 0 else 0.001

        # Trending: ADX > 25
        regimes[adx > 25] = "TRENDING"
        # Ranging: ADX < 20 (default)
        regimes[adx < 20] = "RANGING"
        # High volatility: ATR > 1.5× median
        regimes[atr > atr_median * 1.5] = "HIGH_VOLATILITY"
        # Low volatility: ATR < 0.7× median (overrides RANGING only)
        low_vol_mask = atr < atr_median * 0.7
        regimes[low_vol_mask] = regimes[low_vol_mask].apply(
            lambda x: "LOW_VOLATILITY" if x == "RANGING" else x
        )

        return regimes


# ── Singleton ───────────────────────────────────────────────────────

_TESTER: Optional[RegimeTester] = None


def get_regime_tester() -> RegimeTester:
    global _TESTER
    if _TESTER is None:
        _TESTER = RegimeTester()
    return _TESTER
