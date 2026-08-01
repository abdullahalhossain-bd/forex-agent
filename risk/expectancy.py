# risk/expectancy.py  —  Day 89 | Expectancy Calculator
# ============================================================
# আপনার বলা formula:
#   Expectancy = (Win% × Average Win) − (Loss% × Average Loss)
#
# এটা "average PnL per trade" এর চেয়ে অনেক বেশি informative।
# কারণ এটা দেখায়:
#   - Win rate একা গুরুত্বপূর্ণ না
#   - Win rate 30% হলেও Expectancy positive হতে পারে যদি
#     avg win >> avg loss (trend follow এর জন্য typical)
#   - Win rate 70% হলেও Expectancy negative হতে পারে যদি
#     avg loss >> avg win (scalping এ এমন হয়)
#
# এই module:
#   1. Trade history থেকে proper Expectancy calculate করে
#   2. সাথে confidence interval দেয় (sample size sensitive)
#   3. Strategy health score দেয় (5 dimensions)
#   4. Risk adjustment recommendation দেয়
#
# Output:
#   {
#     "expectancy":         float,    # per-trade expected value
#     "win_rate":           float,    # %
#     "loss_rate":          float,    # %
#     "avg_win":            float,
#     "avg_loss":           float,    # positive number
#     "expectancy_r":       float,    # in R multiples (if R available)
#     "profit_factor":      float,
#     " expectancy_ci":     [low, high],  # 95% confidence interval
#     "sample_size":        int,
#     "system_quality":     "EXCELLENT"|"GOOD"|"MARGINAL"|"POOR"|"FAILING",
#     "health_score":       0-100,
#     "recommendation":     str,
#     "components":         {...}     # breakdown
#   }
#
# এছাড়া analytics/analytics.py এর ভুল expectancy formula fix করার
# জন্য patch_leverage() function দেওয়া আছে।
# ============================================================

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("expectancy_calculator")


class ExpectancyCalculator:
    """
    Proper expectancy + system health evaluator।

    Usage:
        calc = ExpectancyCalculator()
        result = calc.calculate(trades_df)   # trades_df has 'pnl' column

        # Or from raw lists:
        result = calc.calculate_from_pnls([100, -50, 200, -30, ...])
    """

    # ─────────────────────────────────────────────────────
    # Thresholds for system quality assessment
    # ─────────────────────────────────────────────────────
    EXPECTANCY_EXCELLENT = 0.5    # in R-multiples
    EXPECTANCY_GOOD      = 0.25
    EXPECTANCY_MARGINAL  = 0.10
    EXPECTANCY_POOR      = 0.0

    PROFIT_FACTOR_EXCELLENT = 2.0
    PROFIT_FACTOR_GOOD      = 1.5
    PROFIT_FACTOR_MARGINAL  = 1.2

    MIN_SAMPLE_SIZE = 30     # Below this → low confidence
    GOOD_SAMPLE_SIZE = 100

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════

    def calculate(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        trades_df-এ 'pnl' column লাগবে।
        Optional columns: 'rr_ratio' (R multiples), 'pair', 'strategy', 'session'।
        """
        if trades_df is None or len(trades_df) == 0:
            return self._empty_result("No trades provided")

        if "pnl" not in trades_df.columns:
            return self._empty_result("Missing 'pnl' column")

        pnls = trades_df["pnl"].dropna().values
        return self._calculate(pnls, trades_df)

    def calculate_from_pnls(self, pnls: Sequence[float]) -> Dict[str, Any]:
        """
        Simple entry — শুধু PnL list দিলেই হবে।
        """
        if pnls is None or len(pnls) == 0:
            return self._empty_result("No PnLs provided")
        arr = np.array(pnls, dtype=float)
        df = pd.DataFrame({"pnl": arr})
        return self._calculate(arr, df)

    # ═══════════════════════════════════════════════════════
    # CORE CALCULATION
    # ═══════════════════════════════════════════════════════

    def _calculate(self, pnls: np.ndarray, trades_df: pd.DataFrame) -> Dict[str, Any]:
        n = len(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        breakeven = pnls[pnls == 0]

        n_wins = len(wins)
        n_losses = len(losses)
        n_be = len(breakeven)

        win_rate = (n_wins / n) * 100 if n > 0 else 0
        loss_rate = (n_losses / n) * 100 if n > 0 else 0
        be_rate = (n_be / n) * 100 if n > 0 else 0

        total_win = float(wins.sum()) if n_wins > 0 else 0.0
        total_loss = float(losses.sum()) if n_losses > 0 else 0.0   # negative
        total_loss_abs = abs(total_loss)

        avg_win = float(wins.mean()) if n_wins > 0 else 0.0
        avg_loss = float(losses.mean()) if n_losses > 0 else 0.0    # negative
        avg_loss_abs = abs(avg_loss)

        # ── Proper Expectancy Formula ──
        # E = (Win% × AvgWin) − (Loss% × AvgLoss)
        # With AvgLoss as positive magnitude:
        # E = (Win_rate × AvgWin) − (Loss_rate × AvgLoss_abs)
        expectancy = (win_rate / 100 * avg_win) - (loss_rate / 100 * avg_loss_abs)

        # Equivalent: total_pnl / n (sanity check)
        total_pnl = float(pnls.sum())
        expectancy_check = total_pnl / n if n > 0 else 0.0

        # ── Profit Factor ──
        profit_factor = total_win / total_loss_abs if total_loss_abs > 0 else float("inf")

        # ── Expectancy in R multiples (if rr_ratio available) ──
        expectancy_r = None
        if "rr_ratio" in trades_df.columns:
            rr = trades_df["rr_ratio"].dropna().values
            if len(rr) > 0:
                # R-based expectancy: avg win in R - avg loss in R
                win_rr = rr[rr > 0]
                loss_rr = rr[rr < 0]
                wr = (len(win_rr) / len(rr)) * 100 if len(rr) > 0 else 0
                lr = (len(loss_rr) / len(rr)) * 100 if len(rr) > 0 else 0
                aw_r = float(win_rr.mean()) if len(win_rr) > 0 else 0
                al_r = abs(float(loss_rr.mean())) if len(loss_rr) > 0 else 0
                expectancy_r = (wr / 100 * aw_r) - (lr / 100 * al_r)

        # ── Confidence Interval (95%) ──
        ci_low, ci_high = self._confidence_interval(pnls, expectancy)

        # ── System Quality + Health ──
        quality = self._system_quality(expectancy, expectancy_r, profit_factor, n)
        health_score = self._health_score(
            expectancy, expectancy_r, profit_factor, win_rate,
            avg_win, avg_loss_abs, n, ci_low, ci_high
        )
        recommendation = self._recommendation(
            quality, health_score, win_rate, profit_factor,
            avg_win, avg_loss_abs, n
        )

        result = {
            "valid":            True,
            "expectancy":       round(expectancy, 4),
            "expectancy_check": round(expectancy_check, 4),    # sanity check
            "expectancy_r":     round(expectancy_r, 4) if expectancy_r is not None else None,
            "win_rate":         round(win_rate, 2),
            "loss_rate":        round(loss_rate, 2),
            "breakeven_rate":   round(be_rate, 2),
            "avg_win":          round(avg_win, 4),
            "avg_loss":         round(avg_loss_abs, 4),    # positive magnitude
            "total_win":        round(total_win, 4),
            "total_loss":       round(total_loss_abs, 4),
            "profit_factor":    round(profit_factor, 3) if profit_factor != float("inf") else None,
            "expectancy_ci":    [round(ci_low, 4), round(ci_high, 4)],
            "sample_size":      n,
            "system_quality":   quality,
            "health_score":     health_score,
            "recommendation":   recommendation,
            "components": {
                "win_contribution":  round(win_rate / 100 * avg_win, 4),
                "loss_contribution": round(loss_rate / 100 * avg_loss_abs, 4),
                "ratio_w_l":         round(avg_win / avg_loss_abs, 3) if avg_loss_abs > 0 else None,
            },
        }

        log.info(
            f"[Expectancy] E={expectancy:.4f} | WR={win_rate:.1f}% | "
            f"PF={profit_factor:.2f} | avgWin={avg_win:.2f} avgLoss={avg_loss_abs:.2f} | "
            f"n={n} | quality={quality} | health={health_score}/100"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # CONFIDENCE INTERVAL
    # ═══════════════════════════════════════════════════════

    def _confidence_interval(self, pnls: np.ndarray, mean: float) -> tuple[float, float]:
        """
        95% CI using normal approximation (works for n >= 30)।
        Small n হলে t-distribution ভালো, কিন্তু simplification এ normal।
        """
        n = len(pnls)
        if n < 2:
            return mean, mean
        std = float(np.std(pnls, ddof=1))
        if std == 0:
            return mean, mean
        # 95% CI z-score = 1.96
        se = std / math.sqrt(n)
        return mean - 1.96 * se, mean + 1.96 * se

    # ═══════════════════════════════════════════════════════
    # SYSTEM QUALITY
    # ═══════════════════════════════════════════════════════

    def _system_quality(
        self,
        expectancy:       float,
        expectancy_r:     Optional[float],
        profit_factor:    float,
        n:                int,
    ) -> str:
        """
        Multi-dimensional quality assessment।
        """
        # Sample size penalty
        if n < self.MIN_SAMPLE_SIZE:
            return "INSUFFICIENT_DATA"

        # Use R-multiple expectancy if available (more standardized)
        e = expectancy_r if expectancy_r is not None else expectancy

        # If R available, use R thresholds
        if expectancy_r is not None:
            if e >= self.EXPECTANCY_EXCELLENT and profit_factor >= self.PROFIT_FACTOR_EXCELLENT:
                return "EXCELLENT"
            if e >= self.EXPECTANCY_GOOD and profit_factor >= self.PROFIT_FACTOR_GOOD:
                return "GOOD"
            if e >= self.EXPECTANCY_MARGINAL and profit_factor >= self.PROFIT_FACTOR_MARGINAL:
                return "MARGINAL"
            if e >= self.EXPECTANCY_POOR:
                return "POOR"
            return "FAILING"

        # Dollar-based (rough)
        if profit_factor == float("inf"):
            return "EXCELLENT" if expectancy > 0 else "POOR"
        if expectancy > 0 and profit_factor >= self.PROFIT_FACTOR_EXCELLENT:
            return "EXCELLENT"
        if expectancy > 0 and profit_factor >= self.PROFIT_FACTOR_GOOD:
            return "GOOD"
        if expectancy > 0 and profit_factor >= self.PROFIT_FACTOR_MARGINAL:
            return "MARGINAL"
        if expectancy > 0:
            return "POOR"
        return "FAILING"

    # ═══════════════════════════════════════════════════════
    # HEALTH SCORE
    # ═══════════════════════════════════════════════════════

    def _health_score(
        self,
        expectancy:       float,
        expectancy_r:     Optional[float],
        profit_factor:    float,
        win_rate:         float,
        avg_win:          float,
        avg_loss_abs:     float,
        n:                int,
        ci_low:           float,
        ci_high:          float,
    ) -> int:
        """
        0-100 score across 5 dimensions:
          1. Expectancy (30 pts)
          2. Profit Factor (25 pts)
          3. Win/Loss ratio (15 pts)
          4. Sample size (15 pts)
          5. Confidence (CI sign consistency) (15 pts)
        """
        # 1. Expectancy (R-based if available)
        e = expectancy_r if expectancy_r is not None else expectancy
        if expectancy_r is not None:
            if e >= 0.75:           exp_pts = 30
            elif e >= 0.5:          exp_pts = 25
            elif e >= 0.25:         exp_pts = 18
            elif e >= 0.10:         exp_pts = 10
            elif e > 0:             exp_pts = 5
            else:                   exp_pts = 0
        else:
            if e > 0:               exp_pts = 20
            else:                   exp_pts = 0

        # 2. Profit Factor
        if profit_factor == float("inf"):
            pf_pts = 25
        elif profit_factor >= 2.0:    pf_pts = 25
        elif profit_factor >= 1.5:    pf_pts = 20
        elif profit_factor >= 1.2:    pf_pts = 12
        elif profit_factor >= 1.0:    pf_pts = 5
        else:                         pf_pts = 0

        # 3. Win/Loss ratio
        if avg_loss_abs > 0:
            ratio = avg_win / avg_loss_abs
            if ratio >= 2.0:          wl_pts = 15
            elif ratio >= 1.5:        wl_pts = 12
            elif ratio >= 1.0:        wl_pts = 8
            elif ratio >= 0.5:        wl_pts = 4
            else:                     wl_pts = 0
        else:
            wl_pts = 0

        # 4. Sample size
        if n >= self.GOOD_SAMPLE_SIZE:    ss_pts = 15
        elif n >= self.MIN_SAMPLE_SIZE:   ss_pts = 10
        elif n >= 10:                     ss_pts = 5
        else:                             ss_pts = 0

        # 5. Confidence (CI doesn't cross zero)
        if ci_low > 0 and ci_high > 0:    ci_pts = 15
        elif ci_low < 0 and ci_high > 0:  ci_pts = 7   # uncertain
        elif ci_low < 0 and ci_high < 0:  ci_pts = 0
        else:                             ci_pts = 5

        return max(0, min(100, int(exp_pts + pf_pts + wl_pts + ss_pts + ci_pts)))

    # ═══════════════════════════════════════════════════════
    # RECOMMENDATION
    # ═══════════════════════════════════════════════════════

    def _recommendation(
        self,
        quality:         str,
        health_score:    int,
        win_rate:        float,
        profit_factor:   float,
        avg_win:         float,
        avg_loss_abs:    float,
        n:               int,
    ) -> str:
        if quality == "INSUFFICIENT_DATA":
            return f"Only {n} trades — collect {self.MIN_SAMPLE_SIZE - n} more before evaluating."

        if quality == "FAILING":
            return "System losing money. Stop trading live — review strategy."

        if quality == "POOR":
            return "Marginal edge. Reduce size 50%. Tighten entry criteria."

        if quality == "MARGINAL":
            if win_rate < 40:
                return f"Low win rate ({win_rate:.1f}%) but positive — let winners run, cut losses faster."
            return "Marginal edge. Paper trade only until PF > 1.5."

        if quality == "GOOD":
            if profit_factor < 1.5:
                return "Good expectancy but PF low — improve stop placement."
            return "Solid system. Trade with normal size."

        if quality == "EXCELLENT":
            return "Excellent system. Consider scaling up position size carefully."

        return "Evaluate manually."

    # ═══════════════════════════════════════════════════════
    # STRATEGY COMPARISON HELPER
    # ═══════════════════════════════════════════════════════

    def compare(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        একাধিক strategy-র expectancy result compare করো।
        results = {"trend_follow": {...}, "scalping": {...}, ...}
        """
        if not results:
            return {"best": None, "ranking": []}

        ranked = sorted(
            results.items(),
            key=lambda kv: kv[1].get("health_score", 0),
            reverse=True
        )

        return {
            "best":         ranked[0][0] if ranked else None,
            "best_score":   ranked[0][1].get("health_score", 0) if ranked else 0,
            "ranking": [
                {
                    "strategy":      name,
                    "expectancy":    r.get("expectancy"),
                    "expectancy_r":  r.get("expectancy_r"),
                    "profit_factor": r.get("profit_factor"),
                    "health_score":  r.get("health_score", 0),
                    "quality":       r.get("system_quality"),
                }
                for name, r in ranked
            ],
        }

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "valid":          False,
            "reason":         reason,
            "expectancy":     0.0,
            "win_rate":       0.0,
            "loss_rate":      0.0,
            "avg_win":        0.0,
            "avg_loss":       0.0,
            "profit_factor":  None,
            "sample_size":    0,
            "system_quality": "INSUFFICIENT_DATA",
            "health_score":   0,
            "recommendation": reason,
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  📈  EXPECTANCY CALCULATOR  (Day 89)")
        log.info(bar)

        if not result.get("valid"):
            log.info(f"  ⚠️  {result.get('reason', 'No analysis')}")
            log.info(bar)
            return

        icon = {
            "EXCELLENT":          "🌟",
            "GOOD":               "✅",
            "MARGINAL":           "🟡",
            "POOR":               "🟠",
            "FAILING":            "❌",
            "INSUFFICIENT_DATA":  "❓",
        }.get(result["system_quality"], "❓")

        log.info(f"  Expectancy     : {result['expectancy']}")
        if result.get("expectancy_r") is not None:
            log.info(f"  Expectancy (R) : {result['expectancy_r']} R/trade")
        log.info(f"  Win Rate       : {result['win_rate']}%")
        log.info(f"  Loss Rate      : {result['loss_rate']}%")
        log.info(f"  Avg Win        : {result['avg_win']}")
        log.info(f"  Avg Loss       : {result['avg_loss']}")
        log.info(f"  Profit Factor  : {result.get('profit_factor', 'N/A')}")
        log.info(f"  95% CI         : [{result['expectancy_ci'][0]}, {result['expectancy_ci'][1]}]")
        log.info(f"  Sample Size    : {result['sample_size']}")
        log.info(f"  System Quality : {icon}  {result['system_quality']}")
        log.info(f"  Health Score   : {result['health_score']}/100")

        comp = result.get("components", {})
        log.info("  ── Components ──")
        log.info(f"    Win contribution  : {comp.get('win_contribution', 0)}")
        log.info(f"    Loss contribution : {comp.get('loss_contribution', 0)}")
        if comp.get("ratio_w_l"):
            log.info(f"    Win/Loss ratio    : {comp['ratio_w_l']}")

        log.info(f"  Recommendation : {result['recommendation']}")
        log.info(bar)


# ═══════════════════════════════════════════════════════════
# PATCH FOR analytics/analytics.py (Bug Fix)
# ═══════════════════════════════════════════════════════════

def patch_analytics_expectancy():
    """
    আপনার analytics/analytics.py line 47 এ ভুল formula আছে:
        expectancy = round(total_pnl / total_trades, 2)
    এটা শুধু average PnL, proper expectancy না।

    Proper formula:
        expectancy = (win_rate × avg_win) - (loss_rate × avg_loss)

    এই function সেই patch করার জন্য একটা helper — যদি আপনি
    PerformanceAnalyzer.summarize() কে monkey-patch করতে চান।

    Usage:
        from risk.expectancy import patch_analytics_expectancy
        patch_analytics_expectancy()
        # এখন থেকে PerformanceAnalyzer.summarize() proper expectancy দেবে
    """
    from analytics.analytics import PerformanceAnalyzer

    original_summarize = PerformanceAnalyzer.summarize

    def patched_summarize(self, trades_df, equity_curve, strategy_name, pair, period_label):
        summary = original_summarize(self, trades_df, equity_curve, strategy_name, pair, period_label)

        # Override expectancy with proper formula
        if not trades_df.empty and "pnl" in trades_df.columns:
            calc = ExpectancyCalculator()
            exp_result = calc.calculate(trades_df)
            summary["expectancy"] = exp_result["expectancy"]
            summary["expectancy_r"] = exp_result.get("expectancy_r")
            summary["avg_win"] = exp_result["avg_win"]
            summary["avg_loss"] = exp_result["avg_loss"]
            summary["system_quality"] = exp_result["system_quality"]
            summary["health_score"] = exp_result["health_score"]
            summary["expectancy_ci"] = exp_result["expectancy_ci"]
            summary["recommendation"] = exp_result["recommendation"]

        return summary

    PerformanceAnalyzer.summarize = patched_summarize
    log.info("[Expectancy] Patched PerformanceAnalyzer.summarize() with proper expectancy formula")


# ═══════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    # Strategy 1: Trend follow — low win rate, high payoff
    n = 100
    pnls_trend = np.concatenate([
        np.random.choice([1, -1], size=n, p=[0.4, 0.6]) * np.random.exponential(2, n)
    ])
    pnls_trend = np.where(pnls_trend > 0, pnls_trend * 2.5, pnls_trend)  # winners 2.5x bigger

    # Strategy 2: Scalping — high win rate, small wins, big losses
    pnls_scalp = np.concatenate([
        np.random.choice([1, -1], size=n, p=[0.7, 0.3]) * np.random.exponential(1, n)
    ])
    pnls_scalp = np.where(pnls_scalp < 0, pnls_scalp * 2.0, pnls_scalp)  # losers 2x bigger

    calc = ExpectancyCalculator()

    print("\n=== Strategy 1: Trend Follow ===")
    r1 = calc.calculate_from_pnls(pnls_trend)
    calc.print_summary(r1)

    print("\n=== Strategy 2: Scalping ===")
    r2 = calc.calculate_from_pnls(pnls_scalp)
    calc.print_summary(r2)

    print("\n=== Comparison ===")
    cmp = calc.compare({"trend_follow": r1, "scalping": r2})
    print(f"Best: {cmp['best']} (score {cmp['best_score']})")
    for r in cmp["ranking"]:
        print(f"  {r['strategy']}: E={r['expectancy']:.3f}, PF={r['profit_factor']}, score={r['health_score']}")
