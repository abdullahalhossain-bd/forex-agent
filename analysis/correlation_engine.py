"""
analysis/correlation_engine.py — Day 96 Correlation & Volatility Risk Engine
=============================================================================
Computes currency pair correlations + volatility state to adjust position
sizing and detect hidden risk from correlated exposures.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("correlation_engine")


# ── Known correlation groups (fallback when live computation fails) ──
CORRELATION_GROUPS = [
    ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
    ["USDJPY", "USDCHF", "USDCAD"],
    ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],
    ["EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD"],
    ["GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD"],
    ["XAUUSD", "EURUSD", "GBPUSD", "AUDUSD"],
]

STATIC_CORRELATIONS = {
    ("EURUSD", "GBPUSD"):  0.84,
    ("EURUSD", "USDCHF"): -0.85,
    ("EURUSD", "AUDUSD"):  0.70,
    ("EURUSD", "USDJPY"): -0.55,
    ("EURUSD", "USDCAD"): -0.65,
    ("GBPUSD", "AUDUSD"):  0.72,
    ("GBPUSD", "USDJPY"): -0.50,
    ("USDJPY", "USDCHF"):  0.60,
    ("USDJPY", "USDCAD"):  0.45,
    ("XAUUSD", "EURUSD"):  0.55,
    ("XAUUSD", "GBPUSD"):  0.50,
    ("XAUUSD", "USDJPY"): -0.60,
}


class CorrelationEngine:
    """Currency correlation + volatility risk engine."""

    def __init__(
        self,
        correlation_threshold: float = 0.70,
        volatility_spike_mult: float = 2.0,
        volatility_compress_mult: float = 0.7,
        lookback_periods: int = 50,
    ):
        self.correlation_threshold = correlation_threshold
        self.volatility_spike_mult = volatility_spike_mult
        self.volatility_compress_mult = volatility_compress_mult
        self.lookback_periods = lookback_periods
        self._open_pairs: List[str] = []  # FIX: track open pairs in-memory

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════

    def analyze(
        self,
        pair: str,
        df: pd.DataFrame,
        open_pairs: List[str] = None,
    ) -> Dict[str, Any]:
        """Analyze correlation risk + volatility for a pair."""
        if open_pairs is None:
            open_pairs = list(self._open_pairs)

        vol_result = self._analyze_volatility(df)
        corr_result = self._analyze_correlation(pair, open_pairs, df)

        vol_adj = vol_result["risk_adjustment"]
        corr_adj = corr_result["risk_adjustment"]
        combined_adj = min(vol_adj, corr_adj)

        reasons = []
        if vol_result["volatility_state"] in ("HIGH", "EXTREME"):
            reasons.append(f"ATR {vol_result['atr_ratio']:.1f}x avg ({vol_result['volatility_state']})")
        if corr_result["correlated_pairs"]:
            reasons.append(f"correlation overlap with {','.join(corr_result['correlated_pairs'])}")
        reason = " + ".join(reasons) if reasons else "normal risk"

        result = {
            "pair":               pair,
            "correlation_risk":   corr_result["correlation_risk"],
            "correlated_pairs":   corr_result["correlated_pairs"],
            "correlations":       corr_result["correlations"],
            "volatility_state":   vol_result["volatility_state"],
            "atr_current":        vol_result["atr_current"],
            "atr_average":        vol_result["atr_average"],
            "atr_ratio":          vol_result["atr_ratio"],
            "volatility_adj":     vol_adj,
            "correlation_adj":    corr_adj,
            "risk_adjustment":    combined_adj,
            "reason":             reason,
            # FIX (institutional review, item #4): {other_pair: error_msg}
            # for any pair where live correlation computation hit a real
            # exception (vs. falling back to static correlation simply
            # because there wasn't enough history yet). Empty dict = no
            # infra failures this call.
            "live_computation_errors": corr_result.get("live_computation_errors", {}),
        }

        log.info(
            f"[CorrEngine] {pair} | vol={vol_result['volatility_state']} "
            f"(ATR {vol_result['atr_ratio']:.1f}x) | "
            f"corr_risk={corr_result['correlation_risk']:.2f} | "
            f"adj={combined_adj:.2f} | {reason}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # FIX: build_matrix — was missing, caused AttributeError
    # ═══════════════════════════════════════════════════════

    def build_matrix(self, pairs: List[str], df_dict: Dict[str, pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """Build a correlation matrix for a list of pairs.

        Args:
            pairs:   list of pair symbols e.g. ["EURUSD", "GBPUSD", ...]
            df_dict: optional dict of {pair: DataFrame} for live computation

        Returns: nested dict — matrix[pair_a][pair_b] = correlation float
        """
        matrix: Dict[str, Dict[str, float]] = {}

        for pair_a in pairs:
            matrix[pair_a] = {}
            for pair_b in pairs:
                if pair_a == pair_b:
                    matrix[pair_a][pair_b] = 1.0
                    continue

                # Try live correlation if dataframes provided
                r = None
                if df_dict and pair_a in df_dict and pair_b in df_dict:
                    try:
                        df_a = df_dict[pair_a]
                        df_b = df_dict[pair_b]
                        n = min(len(df_a), len(df_b), self.lookback_periods)
                        if n >= 10:
                            ret_a = df_a["close"].tail(n).pct_change().dropna().values
                            ret_b = df_b["close"].tail(n).pct_change().dropna().values
                            min_len = min(len(ret_a), len(ret_b))
                            if min_len >= 5:
                                r_val = np.corrcoef(ret_a[:min_len], ret_b[:min_len])[0, 1]
                                if not np.isnan(r_val):
                                    r = float(r_val)
                    except Exception as e:
                        r = None

                # Fall back to static
                if r is None:
                    r = self._get_static_correlation(pair_a, pair_b)

                matrix[pair_a][pair_b] = round(r, 2) if r is not None else 0.0

        return matrix

    # ═══════════════════════════════════════════════════════
    # FIX: sync_open — was missing, called by safety_guard
    # ═══════════════════════════════════════════════════════

    def sync_open(self, open_pairs: List[str]) -> None:
        """Sync currently open pairs list (called by safety_guard & market_scanner).

        Args:
            open_pairs: list of currently open pair symbols
        """
        self._open_pairs = [p.upper() for p in (open_pairs or [])]
        log.debug(f"[CorrEngine] synced open_pairs={self._open_pairs}")

    # ═══════════════════════════════════════════════════════
    # VOLATILITY ANALYSIS
    # ═══════════════════════════════════════════════════════

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare current ATR vs historical average."""
        if df is None or len(df) < 20:
            return self._default_volatility()

        try:
            if "atr" in df.columns:
                atr_series = df["atr"].dropna()
            else:
                atr_series = self._compute_atr(df)

            if len(atr_series) < 10:
                return self._default_volatility()

            atr_current = float(atr_series.iloc[-1])
            avg_window = min(len(atr_series), 100)
            atr_average = float(atr_series.tail(avg_window).mean())

            if atr_average == 0:
                return self._default_volatility()

            ratio = atr_current / atr_average

            if ratio >= 3.0:
                state = "EXTREME"
                adj = 0.25
            elif ratio >= self.volatility_spike_mult:
                state = "HIGH"
                adj = 0.50
            elif ratio <= self.volatility_compress_mult:
                state = "LOW"
                adj = 0.80
            else:
                state = "NORMAL"
                adj = 1.0

            return {
                "volatility_state": state,
                "atr_current":      round(atr_current, 6),
                "atr_average":      round(atr_average, 6),
                "atr_ratio":        round(ratio, 2),
                "risk_adjustment":  adj,
            }
        except Exception as e:
            log.warning(f"[CorrEngine] volatility analysis failed: {e}")
            return self._default_volatility()

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR from OHLC data."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def _default_volatility() -> Dict[str, Any]:
        return {
            "volatility_state": "NORMAL",
            "atr_current":      0.0,
            "atr_average":      0.0,
            "atr_ratio":        1.0,
            "risk_adjustment":  1.0,
        }

    # ═══════════════════════════════════════════════════════
    # CORRELATION ANALYSIS
    # ═══════════════════════════════════════════════════════

    def _analyze_correlation(
        self, pair: str, open_pairs: List[str], df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check correlation between this pair and currently-open pairs."""
        if not open_pairs:
            return {
                "correlation_risk":  0.0,
                "correlated_pairs":  [],
                "correlations":      {},
                "risk_adjustment":   1.0,
            }

        other_pairs = [p for p in open_pairs if p.upper() != pair.upper()]

        if not other_pairs:
            return {
                "correlation_risk":  0.0,
                "correlated_pairs":  [],
                "correlations":      {},
                "risk_adjustment":   1.0,
            }

        correlations = {}
        correlated_pairs = []
        max_abs_corr = 0.0
        # FIX (institutional review, item #4): track which pairs fell back
        # to the static correlation table because live computation raised
        # an actual exception (data feed / orchestrator problem), as
        # opposed to falling back because there simply wasn't enough
        # history yet. Both used to be silently indistinguishable "None"
        # results — see _compute_live_correlation below.
        live_computation_errors: Dict[str, str] = {}

        for other in other_pairs:
            r, live_error = self._compute_live_correlation(pair, other, df)
            if live_error:
                live_computation_errors[other] = live_error
            if r is None:
                r = self._get_static_correlation(pair, other)

            if r is not None:
                abs_r = abs(r)
                correlations[other] = round(r, 2)
                if abs_r > max_abs_corr:
                    max_abs_corr = abs_r
                if abs_r >= self.correlation_threshold:
                    correlated_pairs.append(other)

        if max_abs_corr >= 0.90:
            adj = 0.25
        elif max_abs_corr >= 0.80:
            adj = 0.40
        elif max_abs_corr >= 0.70:
            adj = 0.50
        elif max_abs_corr >= 0.50:
            adj = 0.75
        else:
            adj = 1.0

        return {
            "correlation_risk":       round(max_abs_corr, 2),
            "correlated_pairs":       correlated_pairs,
            "correlations":           correlations,
            "risk_adjustment":        adj,
            "live_computation_errors": live_computation_errors,  # {other_pair: error_msg} — empty dict = no infra failures
        }

    def _compute_live_correlation(
        self, pair_a: str, pair_b: str, df_a: pd.DataFrame
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Try to compute live Pearson correlation between two pairs.

        Returns (correlation, error). `error` is None for the normal cases
        (not enough history yet, or a genuinely NaN correlation) and is set
        to a descriptive string ONLY when something actually went wrong
        (data orchestrator unavailable, unexpected exception, etc.) — see
        item #4. Previously all of these paths returned a bare `None` with
        just a generic "Suppressed exception at line 338" log line, making
        "no data yet" indistinguishable from "the data feed is broken."
        """
        try:
            from data.data_orchestrator import get_data_orchestrator
        except Exception as e:
            return None, f"data_orchestrator import failed: {e}"

        try:
            orch = get_data_orchestrator()
            df_b = orch.get_candles(pair_b, "M15", limit=self.lookback_periods)
        except Exception as e:
            return None, f"get_candles({pair_b}) failed: {e}"

        if df_b is None or len(df_b) < 20:
            return None, None  # normal: not enough history yet, not an error

        n = min(len(df_a), len(df_b), self.lookback_periods)
        closes_a = df_a["close"].tail(n).values
        closes_b = df_b["close"].tail(n).values

        if len(closes_a) < 10 or len(closes_b) < 10:
            return None, None  # normal: not enough overlapping history

        try:
            ret_a = np.diff(closes_a) / closes_a[:-1]
            ret_b = np.diff(closes_b) / closes_b[:-1]

            if len(ret_a) < 5 or len(ret_b) < 5:
                return None, None  # normal: not enough return observations

            r = np.corrcoef(ret_a, ret_b)[0, 1]
            if np.isnan(r):
                return None, None  # normal: degenerate input (e.g. zero variance)
            return float(r), None
        except Exception as e:
            return None, f"correlation computation failed for {pair_a}/{pair_b}: {e}"

    @staticmethod
    def _get_static_correlation(pair_a: str, pair_b: str) -> Optional[float]:
        """Look up static correlation estimate."""
        a = pair_a.upper()
        b = pair_b.upper()
        if (a, b) in STATIC_CORRELATIONS:
            return STATIC_CORRELATIONS[(a, b)]
        if (b, a) in STATIC_CORRELATIONS:
            return STATIC_CORRELATIONS[(b, a)]
        for group in CORRELATION_GROUPS:
            if a in group and b in group:
                return 0.70
        return None

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compact context for MasterAnalyst / RiskEngine."""
        return {
            "corr_risk":       result.get("correlation_risk", 0),
            "corr_pairs":      result.get("correlated_pairs", []),
            "vol_state":       result.get("volatility_state", "NORMAL"),
            "atr_ratio":       result.get("atr_ratio", 1.0),
            "risk_adjustment": result.get("risk_adjustment", 1.0),
            "risk_reason":     result.get("reason", "normal"),
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  🔗  CORRELATION + VOLATILITY  (Day 96)")
        log.info(bar)
        log.info(f"  Pair             : {result.get('pair','?')}")
        log.info(f"  Volatility state : {result.get('volatility_state','?')}")
        log.info(f"  ATR current      : {result.get('atr_current',0):.6f}")
        log.info(f"  ATR average      : {result.get('atr_average',0):.6f}")
        log.info(f"  ATR ratio        : {result.get('atr_ratio',0):.2f}x")
        log.info(f"  Correlation risk : {result.get('correlation_risk',0):.2f}")
        if result.get("correlated_pairs"):
            log.info(f"  Correlated with  : {', '.join(result['correlated_pairs'])}")
        log.info(f"  Risk adjustment  : {result.get('risk_adjustment',1.0):.2f}x")
        log.info(f"  Reason           : {result.get('reason','')}")
        log.info(bar)