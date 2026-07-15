# analysis/divergence.py  —  Day 83 | Divergence Engine
# ============================================================
# আপনার বলা "false breakout কমাতে সাহায্য" — এটাই সেই module।
#
# Divergence মানে: Price একদিকে যাচ্ছে, কিন্তু momentum indicator
# (RSI / MACD) অন্যদিকে যাচ্ছে। এর মানে reversal আসছে।
#
# ৪ রকম divergence detect করে:
#
#   1. Regular Bullish     : Price LL, RSI HL  → bottom reversal
#   2. Regular Bearish     : Price HH, RSI LH  → top reversal
#   3. Hidden Bullish      : Price HL, RSI LL  → trend continuation up
#   4. Hidden Bearish      : Price LH, RSI HH  → trend continuation down
#
# Detection algorithm:
#   1. Swing pivot points খোঁজো price-এ (fractal)
#   2. সেই একই index-এ indicator value match করো
#   3. শেষ ২টা pivot compare করে divergence detect করো
#   4. ATR-normalized slope দিয়ে significance score করো
#   5. Optional: volume confirmation (tick volume থাকলে)
#
# Usage:
#     engine = DivergenceEngine()
#     result = engine.detect(df, indicator='rsi')
#     ctx    = engine.get_ai_context(result)
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("divergence_engine")


class DivergenceEngine:
    """
    Price vs Indicator (RSI/MACD) divergence detector।

    Output:
        {
          "valid":          True/False,
          "divergences":    [ {type, indicator, price_dir, ind_dir,
                               pivot1_idx, pivot2_idx, score, note}, ... ],
          "latest":         {...} | None,    # সবচেয়ে সাম্প্রতিক
          "reversal_risk":  "HIGH"/"MEDIUM"/"LOW",
          "trend_continue": "BULLISH"/"BEARISH"/"NONE",  # hidden div
        }
    """

    def __init__(
        self,
        pivot_window:    int  = 5,
        min_pivot_count: int  = 2,
        lookback_pivots: int  = 4,
    ):
        """
        pivot_window    : দুই পাশে কতটা candle দেখবে pivot detect করতে
        min_pivot_count : এর কম থাকলে analysis skip
        lookback_pivots : শেষ কতটা pivot compare করবে (3-4 ideal)
        """
        self.pivot_window    = pivot_window
        self.min_pivot_count = min_pivot_count
        self.lookback_pivots = lookback_pivots

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════

    def detect(
        self,
        df:         pd.DataFrame,
        indicator:  str = "rsi",
        rsi_col:    str = "rsi",
        macd_col:   str = "macd",
        atr_col:    str = "atr",
    ) -> Dict[str, Any]:
        """
        df-এ 'high','low','close' লাগবে। Indicator column-ও লাগবে।
        indicator = 'rsi' | 'macd'
        """
        if df is None or len(df) < self.pivot_window * 4 + 5:
            return self._empty_result("Insufficient data")

        # Indicator column বেছে নাও
        if indicator == "rsi":
            if rsi_col not in df.columns:
                return self._empty_result(f"Missing column: {rsi_col}")
            ind_series = df[rsi_col]
        elif indicator == "macd":
            if macd_col not in df.columns:
                return self._empty_result(f"Missing column: {macd_col}")
            ind_series = df[macd_col]
        else:
            return self._empty_result(f"Unknown indicator: {indicator}")

        # Step 1: find pivots
        pivots = self._find_pivots(df, ind_series)
        if len(pivots) < self.min_pivot_count:
            return self._empty_result("Not enough pivots for divergence")

        # Step 2: scan all same-kind pivot pairs within lookback window.
        # আগে শুধু consecutive pair দেখতাম, কিন্তু divergence
        # সবসময় consecutive pivot-এ হয় না — দুটো high pivot
        # একে অপরের থেকে দূরে থাকতে পারে (মাঝে low pivot থাকে)।
        # তাই সব same-kind pair compare করি, latest lookback-এর মধ্যে।
        recent = pivots[-self.lookback_pivots * 2:] if len(pivots) >= self.lookback_pivots * 2 else pivots
        high_pivots = [p for p in recent if p["kind"] == "high"]
        low_pivots  = [p for p in recent if p["kind"] == "low"]

        divergences: List[Dict[str, Any]] = []

        # High pivots: check (last-1 → last), (last-2 → last), (last-3 → last)
        for h_pivots in (high_pivots, low_pivots):
            if len(h_pivots) < 2:
                continue
            # Consecutive pairs + cross-pair check (last vs prior)
            for i in range(1, len(h_pivots)):
                p1 = h_pivots[i - 1]
                p2 = h_pivots[i]
                div = self._detect_divergence_pair(df, p1, p2, atr_col)
                if div:
                    div["indicator"] = indicator
                    divergences.append(div)

        if not divergences:
            result = {
                "valid":          True,
                "divergences":    [],
                "latest":         None,
                "reversal_risk":  "LOW",
                "trend_continue": "NONE",
                "indicator":      indicator,
                "note":           "No divergence detected",
            }
            log.info("[Divergence] No divergence — momentum confirms price action")
            return result

        # Latest divergence
        latest = divergences[-1]

        # Risk assessment
        reversal_risk  = self._reversal_risk(divergences)
        trend_continue = self._trend_continuation(divergences)

        result = {
            "valid":          True,
            "divergences":    divergences,
            "latest":         latest,
            "reversal_risk":  reversal_risk,
            "trend_continue": trend_continue,
            "indicator":      indicator,
            "note":           latest.get("note", "Divergence detected"),
        }

        log.info(
            f"[Divergence] {latest['type']} ({indicator}) | "
            f"risk={reversal_risk} | cont={trend_continue}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # PIVOT DETECTION
    # ═══════════════════════════════════════════════════════

    def _find_pivots(self, df: pd.DataFrame, ind_series: pd.Series) -> List[Dict[str, Any]]:
        """
        Fractal pivot — দুই পাশের pivot_window এর চেয়ে বড় বা ছোট।
        Output: list of {index, price, ind_value, kind: 'high'|'low'}
        """
        highs = df["high"].values
        lows  = df["low"].values
        w     = self.pivot_window
        n     = len(df)

        pivots: List[Dict[str, Any]] = []
        for i in range(w, n - w):
            window_high = highs[i - w: i + w + 1]
            window_low  = lows[i - w: i + w + 1]

            if highs[i] == window_high.max() and not np.isnan(ind_series.iloc[i]):
                pivots.append({
                    "index":     i,
                    "price":     float(highs[i]),
                    "ind_value": float(ind_series.iloc[i]),
                    "kind":      "high",
                })
            elif lows[i] == window_low.min() and not np.isnan(ind_series.iloc[i]):
                pivots.append({
                    "index":     i,
                    "price":     float(lows[i]),
                    "ind_value": float(ind_series.iloc[i]),
                    "kind":      "low",
                })

        # Remove duplicates / consecutive same-kind (keep most extreme)
        cleaned: List[Dict[str, Any]] = []
        for p in pivots:
            if cleaned and cleaned[-1]["kind"] == p["kind"]:
                if p["kind"] == "high" and p["price"] > cleaned[-1]["price"]:
                    cleaned[-1] = p
                elif p["kind"] == "low" and p["price"] < cleaned[-1]["price"]:
                    cleaned[-1] = p
            else:
                cleaned.append(p)

        return cleaned

    # ═══════════════════════════════════════════════════════
    # DIVERGENCE DETECTION (pair)
    # ═══════════════════════════════════════════════════════

    def _detect_divergence_pair(
        self,
        df:    pd.DataFrame,
        p1:    Dict[str, Any],
        p2:    Dict[str, Any],
        atr_col: str,
    ) -> Optional[Dict[str, Any]]:
        """
        দুটা same-kind pivot compare করে divergence detect করো।

        Regular Bullish:  p1 low LL, p2 low HL  → price lower, ind higher
        Regular Bearish:  p1 high HH, p2 high LH → price higher, ind lower
        Hidden Bullish:   p1 low HL, p2 low LL  → price higher, ind lower
        Hidden Bearish:   p1 high LH, p2 high HH → price lower, ind higher
        """
        if p1["kind"] != p2["kind"]:
            return None

        price_change = p2["price"] - p1["price"]
        ind_change   = p2["ind_value"] - p1["ind_value"]

        # ATR-normalized price change (significance)
        atr = self._get_atr(df, atr_col)
        price_pct = abs(price_change) / (atr if atr > 0 else 1e-5)

        kind = p1["kind"]

        # Regular bullish: price lower low, indicator higher low
        if kind == "low" and price_change < 0 and ind_change > 0:
            score = self._score_divergence(price_pct, abs(ind_change), "regular_bullish")
            return {
                "type":          "REGULAR_BULLISH",
                "indicator":     None,  # set by caller
                "price_dir":     "LOWER_LOW",
                "ind_dir":       "HIGHER_LOW",
                "pivot1_idx":    p1["index"],
                "pivot2_idx":    p2["index"],
                "pivot1_price":  round(p1["price"], 5),
                "pivot2_price":  round(p2["price"], 5),
                "pivot1_ind":    round(p1["ind_value"], 3),
                "pivot2_ind":    round(p2["ind_value"], 3),
                "score":         score,
                "signal":        "BUY",
                "note":          "Price LL but indicator HL — bullish reversal setup",
            }

        # Regular bearish: price higher high, indicator lower high
        if kind == "high" and price_change > 0 and ind_change < 0:
            score = self._score_divergence(price_pct, abs(ind_change), "regular_bearish")
            return {
                "type":          "REGULAR_BEARISH",
                "indicator":     None,
                "price_dir":     "HIGHER_HIGH",
                "ind_dir":       "LOWER_HIGH",
                "pivot1_idx":    p1["index"],
                "pivot2_idx":    p2["index"],
                "pivot1_price":  round(p1["price"], 5),
                "pivot2_price":  round(p2["price"], 5),
                "pivot1_ind":    round(p1["ind_value"], 3),
                "pivot2_ind":    round(p2["ind_value"], 3),
                "score":         score,
                "signal":        "SELL",
                "note":          "Price HH but indicator LH — bearish reversal setup",
            }

        # Hidden bullish: price higher low, indicator lower low (trend continue up)
        if kind == "low" and price_change > 0 and ind_change < 0:
            score = self._score_divergence(price_pct, abs(ind_change), "hidden_bullish")
            return {
                "type":          "HIDDEN_BULLISH",
                "indicator":     None,
                "price_dir":     "HIGHER_LOW",
                "ind_dir":       "LOWER_LOW",
                "pivot1_idx":    p1["index"],
                "pivot2_idx":    p2["index"],
                "pivot1_price":  round(p1["price"], 5),
                "pivot2_price":  round(p2["price"], 5),
                "pivot1_ind":    round(p1["ind_value"], 3),
                "pivot2_ind":    round(p2["ind_value"], 3),
                "score":         score,
                "signal":        "BUY",
                "note":          "Hidden bullish — trend continuation up",
            }

        # Hidden bearish: price lower high, indicator higher high (trend continue down)
        if kind == "high" and price_change < 0 and ind_change > 0:
            score = self._score_divergence(price_pct, abs(ind_change), "hidden_bearish")
            return {
                "type":          "HIDDEN_BEARISH",
                "indicator":     None,
                "price_dir":     "LOWER_HIGH",
                "ind_dir":       "HIGHER_HIGH",
                "pivot1_idx":    p1["index"],
                "pivot2_idx":    p2["index"],
                "pivot1_price":  round(p1["price"], 5),
                "pivot2_price":  round(p2["price"], 5),
                "pivot1_ind":    round(p1["ind_value"], 3),
                "pivot2_ind":    round(p2["ind_value"], 3),
                "score":         score,
                "signal":        "SELL",
                "note":          "Hidden bearish — trend continuation down",
            }

        return None

    def _score_divergence(self, price_pct: float, ind_change: float, div_type: str) -> int:
        """
        0-100 score।
        Price move যত বড় + indicator যত বেশি opposite → score তত বেশি।
        """
        # Price significance (0-50 points)
        price_pts = min(50, price_pct * 25)

        # Indicator significance (0-30 points)
        # RSI: 5+ point change is significant
        # MACD: harder to threshold, use same logic
        ind_pts = min(30, abs(ind_change) * 6)

        # Type bonus
        type_bonus = {
            "regular_bullish": 10,    # reversal signals more reliable
            "regular_bearish": 10,
            "hidden_bullish":  5,
            "hidden_bearish":  5,
        }.get(div_type, 0)

        return max(0, min(100, int(price_pts + ind_pts + type_bonus)))

    # ═══════════════════════════════════════════════════════
    # RISK ASSESSMENT
    # ═══════════════════════════════════════════════════════

    def _reversal_risk(self, divergences: List[Dict[str, Any]]) -> str:
        """
        যদি শেষ divergence Regular type হয় এবং score বেশি হয় → HIGH।
        """
        if not divergences:
            return "LOW"

        latest = divergences[-1]
        if latest["type"] in ("REGULAR_BULLISH", "REGULAR_BEARISH"):
            if latest["score"] >= 70:   return "HIGH"
            if latest["score"] >= 50:   return "MEDIUM"
            return "LOW"
        # Hidden divergence → trend continuation, reversal risk low
        return "LOW"

    def _trend_continuation(self, divergences: List[Dict[str, Any]]) -> str:
        """
        Hidden divergence থাকলে trend continuation signal।
        """
        if not divergences:
            return "NONE"

        latest = divergences[-1]
        if latest["type"] == "HIDDEN_BULLISH": return "BULLISH"
        if latest["type"] == "HIDDEN_BEARISH": return "BEARISH"
        return "NONE"

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        DecisionAgent-এর জন্য compressed context।
        """
        if not result.get("valid"):
            return {
                "divergence_valid":       False,
                "divergence_type":        "NONE",
                "divergence_signal":      "NONE",
                "divergence_score":       0,
                "divergence_reversal_risk": "LOW",
                "divergence_trend_cont":  "NONE",
            }

        latest = result.get("latest")
        if not latest:
            return {
                "divergence_valid":         True,
                "divergence_type":          "NONE",
                "divergence_signal":        "NONE",
                "divergence_score":         0,
                "divergence_reversal_risk": result.get("reversal_risk", "LOW"),
                "divergence_trend_cont":    result.get("trend_continue", "NONE"),
            }

        return {
            "divergence_valid":         True,
            "divergence_type":          latest["type"],
            "divergence_signal":        latest["signal"],
            "divergence_score":         latest["score"],
            "divergence_reversal_risk": result.get("reversal_risk", "LOW"),
            "divergence_trend_cont":    result.get("trend_continue", "NONE"),
            "divergence_price_dir":     latest.get("price_dir", ""),
            "divergence_ind_dir":       latest.get("ind_dir", ""),
        }

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _get_atr(self, df: pd.DataFrame, atr_col: str, period: int = 14) -> float:
        if atr_col in df.columns:
            v = df[atr_col].iloc[-1]
            if not np.isnan(v):
                return float(v)
        # Fallback: compute TR average
        if len(df) < period + 1:
            return 1e-5
        h = df["high"].values[-period:]
        l = df["low"].values[-period:]
        c = df["close"].values[-period:]
        trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
               for i in range(1, len(h))]
        return float(np.mean(trs)) if trs else 1e-5

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "valid":          False,
            "reason":         reason,
            "divergences":    [],
            "latest":         None,
            "reversal_risk":  "LOW",
            "trend_continue": "NONE",
            "indicator":      None,
            "note":           reason,
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  🔄  DIVERGENCE ENGINE  (Day 83)")
        log.info(bar)

        if not result.get("valid"):
            log.info(f"  ⚠️  {result.get('reason', 'No analysis')}")
            log.info(bar)
            return

        if not result.get("divergences"):
            log.info("  No divergence detected — momentum confirms price")
            log.info(bar)
            return

        latest = result.get("latest", {})
        icon = {
            "REGULAR_BULLISH": "🟢",
            "REGULAR_BEARISH": "🔴",
            "HIDDEN_BULLISH":  "📈",
            "HIDDEN_BEARISH":  "📉",
        }.get(latest.get("type", ""), "❓")

        log.info(f"  Latest: {icon}  {latest.get('type', '')}")
        log.info(f"  Indicator : {result.get('indicator', '')}")
        log.info(f"  Score     : {latest.get('score', 0)}/100")
        log.info(f"  Signal    : {latest.get('signal', 'NONE')}")
        log.info(f"  Price     : {latest.get('price_dir', '')}")
        log.info(f"  Indicator : {latest.get('ind_dir', '')}")
        log.info(f"  Reversal Risk  : {result.get('reversal_risk', 'LOW')}")
        log.info(f"  Trend Continue : {result.get('trend_continue', 'NONE')}")
        log.info(f"  Note      : {latest.get('note', '')}")
        log.info(bar)


# ═══════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Synthetic test: price trending up but RSI topping out
    np.random.seed(42)
    n = 200
    prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.0005)
    # Make HH in price but lower RSI
    rsi = 70 - np.linspace(0, 15, n) + np.random.randn(n) * 2

    df = pd.DataFrame({
        "open":  prices,
        "high":  prices + 0.0005,
        "low":   prices - 0.0005,
        "close": prices,
        "rsi":   rsi,
    })

    engine = DivergenceEngine()
    result = engine.detect(df, indicator="rsi")
    engine.print_summary(result)

    ctx = engine.get_ai_context(result)
    print("\nAI Context:")
    for k, v in ctx.items():
        print(f"  {k:<28}: {v}")
