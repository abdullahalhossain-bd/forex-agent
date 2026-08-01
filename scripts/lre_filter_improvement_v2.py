"""LRE Filter Improvement v2 — failure_cascade + regime_transition

Walk-forward validation with root-cause analysis.
Run with ALL 10 filters active (matching production).
Focus: Increase WPR from 68.9% to >=95%.
"""
from __future__ import annotations
import sys, os, json, logging, copy, datetime, traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("lre_improve")
log.setLevel(logging.INFO)

_PIP = 0.0001
_EURUSD_H1_ATR = 0.0065


def load_trades(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["entry_time", "exit_time"])
    df["is_win"] = df["pnl_usd"] > 0  # economic winner (after costs)
    df["sl_dist_pips"] = np.abs(df["entry_price"] - df["stop_loss"]) * 10000
    df["tp_dist_pips"] = np.abs(df["take_profit"] - df["entry_price"]) * 10000
    df["rr"] = df["tp_dist_pips"] / df["sl_dist_pips"].replace(0, np.nan)
    df = df.sort_values("entry_time").reset_index(drop=True)
    return df


def _build_context_no_leak(row: pd.Series, trade_idx: int) -> Tuple[Dict, Dict, Dict]:
    """Reconstruct context WITHOUT using trade outcome."""
    direction = row["direction"]
    entry = row["entry_price"]
    sl = row["stop_loss"]
    tp = row["take_profit"]
    conf = row["confidence"]
    rr = row["rr"] if not np.isnan(row["rr"]) else 2.0
    sl_pips = abs(entry - sl) / _PIP
    tp_pips = abs(tp - entry) / _PIP
    entry_time = pd.Timestamp(row["entry_time"])
    hour = entry_time.hour
    strategy = row["strategy"]

    atr = abs(entry - sl) / 2.0
    atr = max(atr, _EURUSD_H1_ATR * 0.5)
    atr = min(atr, _EURUSD_H1_ATR * 2.0)

    # RSI: deterministic from direction and index, NOT outcome
    seed = hash(f"rsi_{trade_idx}") % 21 - 10
    if direction == "BUY":
        rsi = 45 + seed
    else:
        rsi = 55 + seed
    rsi = max(30, min(70, rsi))

    # MACD aligned with direction
    macd_val = 0.00015 * (1 if direction == "BUY" else -1)

    dec_out = {
        "decision": direction, "entry": entry, "confidence": float(conf),
        "rr": rr, "sl_pips": sl_pips, "tp_pips": tp_pips,
        "sl_price": sl, "tp_price": tp, "strategy": strategy,
    }

    ind_ctx = {
        "atr": {"value": atr}, "ATR": atr,
        "rsi": {"value": rsi}, "RSI": rsi,
        "macd": {"value": macd_val, "signal": macd_val * 0.7},
        "bb": {"upper": entry + atr * 2, "lower": entry - atr * 2},
    }

    # Regime from trade parameters (NOT outcome)
    sl_atr_norm = sl_pips / (atr * 10000)
    if sl_atr_norm > 2.5:
        regime_type = "volatile"
        regime_conf = 0.35
        trend_str = 0.25
        vol_label = "HIGH"
    elif conf >= 80 and sl_atr_norm < 1.5:
        regime_type = "trending"
        regime_conf = 0.65
        trend_str = 0.6
        vol_label = "LOW"
    elif conf >= 60:
        regime_type = "trending"
        regime_conf = 0.55
        trend_str = 0.5
        vol_label = "LOW"
    else:
        regime_type = "ranging"
        regime_conf = 0.5
        trend_str = 0.35
        vol_label = "NORMAL"

    regime = {
        "regime": regime_type, "label": regime_type,
        "confidence": regime_conf, "volatility": vol_label,
        "trend_strength": trend_str,
    }

    smc_score = 3.0 + conf / 20.0
    smc = {
        "score": smc_score, "total_score": smc_score,
        "bos": {"direction": f"bullish_{direction.lower()}", "type": "BOS"},
        "order_block": conf >= 70, "fvg": conf >= 75,
        "sweep_detected": False, "liquidity_sweep": False,
    }

    # SR levels: generic, no outcome info
    sr_levels = []
    for k in range(3):
        offset = atr * (0.5 + 0.5 * k)
        if direction == "BUY":
            sr_levels.append({"price": entry - offset, "type": "support"})
            sr_levels.append({"price": entry + atr * (1.5 + k), "type": "resistance"})
        else:
            sr_levels.append({"price": entry + offset, "type": "resistance"})
            sr_levels.append({"price": entry - atr * (1.5 + k), "type": "support"})
    sr_ctx = {"levels": sr_levels}

    if 7 <= hour <= 9 or 13 <= hour <= 17:
        session_quality = "HIGH"
    elif 0 <= hour <= 6 or 20 <= hour <= 23:
        session_quality = "LOW"
    else:
        session_quality = "MEDIUM"

    sentiment_ctx = {"retail_long_pct": 0.50, "long_pct": 0.50, "long_ratio": 1.0, "agreement": 0.5, "fg_index": 50.0}
    mtf_bias = {"bias": direction}
    news_ctx = {"high_impact_nearby": False}
    liquidity_ctx = {"grade": "NORMAL"}

    market_out = {
        "ind_ctx": ind_ctx, "regime": regime, "mtf_bias": mtf_bias,
        "spread": 1.5, "avg_spread": 1.5, "liquidity_ctx": liquidity_ctx,
    }

    analysis_out = {
        "sr": sr_ctx, "sr_ctx": sr_ctx,
        "liquidity": liquidity_ctx, "liquidity_ctx": liquidity_ctx,
        "smc": smc, "smc_ctx": smc,
        "session": {"quality": session_quality, "session_quality": session_quality},
        "session_ctx": {"quality": session_quality, "session_quality": session_quality},
        "sentiment": sentiment_ctx, "sentiment_ctx": sentiment_ctx,
        "news": news_ctx, "divergence": {},
    }

    return dec_out, analysis_out, market_out


@dataclass
class TradeEvalResult:
    trade_id: int
    is_win: bool
    pnl_pips: float
    pnl_usd: float
    direction: str
    entry_time: str
    exit_reason: str
    hold_bars: int
    strategy: str
    confidence: int
    l1_verdict: str
    l1_composite: float
    l1_primary_reason: str
    l1_filter_scores: Dict[str, float]
    blocked: bool
    failure_cascade_score: float = 0.0
    failure_cascade_data: Dict[str, Any] = field(default_factory=dict)
    regime_transition_score: float = 0.0
    regime_transition_data: Dict[str, Any] = field(default_factory=dict)


def run_walk_forward(trades_df, use_improved=False) -> List[TradeEvalResult]:
    """Run walk-forward validation with ALL 10 filters active."""
    from core.loss_rejection_engine.layer1_structural_filters import (
        StructuralFilterLayer, FilterResult, FILTER_WEIGHTS, LAYER1_REJECT_THRESHOLD,
    )

    layer = StructuralFilterLayer()

    if use_improved:
        from scripts.lre_improved_filters import (
            ImprovedFailureCascadeDetector,
            ImprovedRegimeTransitionFilter,
        )
        layer.failure_cascade = ImprovedFailureCascadeDetector()
        layer._filters["failure_cascade"] = layer.failure_cascade
        layer.regime_transition = ImprovedRegimeTransitionFilter()
        layer._filters["regime_transition"] = layer.regime_transition

    results = []

    for idx, row in trades_df.iterrows():
        dec_out, analysis_out, market_out = _build_context_no_leak(row, idx)
        symbol = row["symbol"]

        l1_out = layer.evaluate(dec_out, analysis_out, market_out, symbol=symbol)
        filter_scores = {f.name: f.rejection_score for f in l1_out.filters}

        fc_data, rt_data = {}, {}
        for f in l1_out.filters:
            if f.name == "failure_cascade": fc_data = f.data
            elif f.name == "regime_transition": rt_data = f.data

        blocked = not l1_out.pass_through

        result = TradeEvalResult(
            trade_id=row["trade_id"], is_win=row["is_win"],
            pnl_pips=row["pnl_pips"], pnl_usd=row["pnl_usd"],
            direction=row["direction"], entry_time=str(row["entry_time"]),
            exit_reason=row["exit_reason"], hold_bars=row["hold_bars"],
            strategy=row["strategy"], confidence=row["confidence"],
            l1_verdict=l1_out.verdict, l1_composite=l1_out.composite_score,
            l1_primary_reason=l1_out.primary_reason, l1_filter_scores=filter_scores,
            blocked=blocked,
            failure_cascade_score=filter_scores.get("failure_cascade", 0),
            failure_cascade_data=fc_data,
            regime_transition_score=filter_scores.get("regime_transition", 0),
            regime_transition_data=rt_data,
        )
        results.append(result)

        # Record outcome AFTER evaluation (walk-forward)
        # Use pnl_usd for consistency: economic outcome determines win/loss
        pnl = row["pnl_usd"]
        d = row["direction"]
        price_zone = "mid"
        regime_label = market_out.get("regime", {}).get("regime", "unknown")
        layer.record_trade_outcome(symbol, d, price_zone, regime_label, pnl)

    return results


def compute_metrics(results, label="") -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"label": label}

    winners = [r for r in results if r.is_win]
    losers = [r for r in results if not r.is_win]
    n_winners = len(winners)
    n_losers = len(losers)

    blocked_winners = [r for r in winners if r.blocked]
    blocked_losers = [r for r in losers if r.blocked]
    kept_winners = [r for r in winners if not r.blocked]
    kept_losers = [r for r in losers if not r.blocked]

    n_bw = len(blocked_winners)
    n_bl = len(blocked_losers)
    n_kw = len(kept_winners)
    n_kl = len(kept_losers)

    wpr = n_kw / n_winners * 100 if n_winners > 0 else 100.0
    lrr = n_bl / n_losers * 100 if n_losers > 0 else 0.0

    post_trades = kept_winners + kept_losers
    n_post = len(post_trades)

    win_rate = n_kw / n_post * 100 if n_post > 0 else 0.0
    gross_profit = sum(r.pnl_usd for r in kept_winners)
    gross_loss = abs(sum(r.pnl_usd for r in kept_losers))
    net_profit = gross_profit - gross_loss
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = np.mean([r.pnl_usd for r in kept_winners]) if kept_winners else 0
    avg_loss = np.mean([r.pnl_usd for r in kept_losers]) if kept_losers else 0
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss) if n_post > 0 else 0

    tp = n_bl
    fp = n_bw
    tn = n_kw
    fn = n_kl

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = (recall + (tn / (tn + fp))) / 2 if (tn + fp) > 0 else 0.0

    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0.0

    equity = [0.0]
    for r in results:
        if not r.blocked:
            equity.append(equity[-1] + r.pnl_usd)
    equity = equity[1:]

    max_dd = 0.0
    peak = 0.0
    for e in equity:
        if e > peak: peak = e
        dd = peak - e
        if dd > max_dd: max_dd = dd

    return {
        "label": label,
        "total_trades": total,
        "n_winners": n_winners, "n_losers": n_losers,
        "blocked_winners": n_bw, "blocked_losers": n_bl,
        "kept_winners": n_kw, "kept_losers": n_kl,
        "post_filter_trades": n_post,
        "wpr": round(wpr, 1), "lrr": round(lrr, 1),
        "win_rate_post": round(win_rate, 1),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_profit": round(net_profit, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "max_drawdown": round(max_dd, 2),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(f1, 3), "balanced_accuracy": round(balanced_acc, 3),
        "mcc": round(mcc, 3),
        "equity_curve": equity,
    }


def categorize_false_positives(results) -> List[Dict]:
    fps = []
    for r in results:
        if r.is_win and r.blocked:
            # Find which filter(s) caused the block
            blocking_filters = []
            for fname, score in r.l1_filter_scores.items():
                if score >= 70:
                    blocking_filters.append((fname, score))
            blocking_filters.sort(key=lambda x: -x[1])

            if r.hold_bars <= 2:
                category = "quick_scalp"
            elif r.hold_bars > 50:
                category = "long_hold_runner"
            elif r.strategy == "ict_amd":
                category = "ict_amd_winner"
            elif r.strategy == "stop_hunt":
                category = "stop_hunt_winner"
            else:
                category = "pa_winner"

            fps.append({
                "trade_id": r.trade_id, "direction": r.direction,
                "pnl_pips": r.pnl_pips, "pnl_usd": r.pnl_usd,
                "hold_bars": r.hold_bars, "strategy": r.strategy,
                "confidence": r.confidence, "entry_time": r.entry_time,
                "blocking_filters": blocking_filters,
                "failure_cascade_score": r.failure_cascade_score,
                "regime_transition_score": r.regime_transition_score,
                "failure_cascade_data": r.failure_cascade_data,
                "regime_transition_data": r.regime_transition_data,
                "l1_reason": r.l1_primary_reason,
                "l1_composite": r.l1_composite,
                "category": category,
            })
    return fps


def print_fp_report(fps, title="FALSE POSITIVE ROOT-CAUSE ANALYSIS"):
    log.info(f"\n{'='*80}")
    log.info(title)
    log.info(f"{'='*80}")
    log.info(f"Total false positives (rejected winners): {len(fps)}")

    # Group by blocking filter
    by_filter = {}
    for fp in fps:
        for fname, score in fp["blocking_filters"]:
            by_filter.setdefault(fname, []).append(fp)

    for fname, fplist in sorted(by_filter.items(), key=lambda x: -len(x[1])):
        log.info(f"\n--- {fname.upper()} ({len(fplist)} false positives) ---")
        for fp in fplist:
            all_scores = ", ".join(f"{fn}={sc:.0f}" for fn, sc in fp["blocking_filters"])
            log.info(f"  Trade #{fp['trade_id']:>2d} | {fp['direction']:4s} | "
                    f"PnL={fp['pnl_pips']:>7.1f} pips (${fp['pnl_usd']:>9.2f}) | "
                    f"hold={fp['hold_bars']:>3d} | conf={fp['confidence']}% | "
                    f"strat={fp['strategy']:>10s} | cat={fp['category']}")
            log.info(f"          Scores: {all_scores}")
            log.info(f"          Reason: {fp['l1_reason']}")

    # By category
    by_cat = {}
    for fp in fps:
        by_cat.setdefault(fp["category"], []).append(fp)
    log.info(f"\n--- CATEGORIES ---")
    for cat, fplist in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        avg_pnl = np.mean([f["pnl_usd"] for f in fplist])
        log.info(f"  {cat:>25s}: {len(fplist):>2d} FPs | avg PnL=${avg_pnl:>9.2f}")


def print_comparison(before, after):
    log.info(f"\n{'='*80}")
    log.info("BEFORE vs AFTER COMPARISON")
    log.info(f"{'='*80}")
    log.info(f"{'Metric':<25s} {'BEFORE':>12s} {'AFTER':>12s} {'CHANGE':>12s}")
    log.info("-" * 65)
    for name, key in [
        ("WPR (%)", "wpr"), ("LRR (%)", "lrr"), ("Profit Factor", "profit_factor"),
        ("Expectancy ($)", "expectancy"), ("Max Drawdown ($)", "max_drawdown"),
        ("Trade Count", "post_filter_trades"), ("Net Profit ($)", "net_profit"),
        ("MCC", "mcc"), ("F1", "f1"), ("Precision", "precision"), ("Recall", "recall"),
    ]:
        b, a = before[key], after[key]
        change = a - b
        sign = "+" if change > 0 else ""
        log.info(f"{name:<25s} {b:>12.2f} {a:>12.2f} {sign}{change:>11.2f}")

    log.info(f"\nCM BEFORE: TP={before['confusion_matrix']['TP']} FP={before['confusion_matrix']['FP']} "
            f"TN={before['confusion_matrix']['TN']} FN={before['confusion_matrix']['FN']}")
    log.info(f"CM AFTER:  TP={after['confusion_matrix']['TP']} FP={after['confusion_matrix']['FP']} "
            f"TN={after['confusion_matrix']['TN']} FN={after['confusion_matrix']['FN']}")


def main():
    CSV_PATH = PROJECT_ROOT / "backtest" / "results_EURUSD_H1.csv"
    REPORT_PATH = PROJECT_ROOT / "download" / "lre_filter_improvement_report.json"

    trades_df = load_trades(str(CSV_PATH))
    log.info(f"Loaded {len(trades_df)} trades: {trades_df['is_win'].sum()}W / {(~trades_df['is_win']).sum()}L")

    # ═══ PHASE 1: BASELINE (original all 10 filters) ═══
    log.info(f"\n{'#'*80}")
    log.info("PHASE 1: BASELINE — All 10 original filters")
    log.info(f"{'#'*80}")

    results_before = run_walk_forward(trades_df, use_improved=False)
    metrics_before = compute_metrics(results_before, label="BEFORE")
    fps_before = categorize_false_positives(results_before)

    log.info(f"\nWPR: {metrics_before['wpr']}% | LRR: {metrics_before['lrr']}% | PF: {metrics_before['profit_factor']}")
    log.info(f"Expectancy: ${metrics_before['expectancy']} | MaxDD: ${metrics_before['max_drawdown']}")
    log.info(f"Trades: {metrics_before['post_filter_trades']} | MCC: {metrics_before['mcc']}")
    log.info(f"CM: TP={metrics_before['confusion_matrix']['TP']} FP={metrics_before['confusion_matrix']['FP']} "
            f"TN={metrics_before['confusion_matrix']['TN']} FN={metrics_before['confusion_matrix']['FN']}")

    print_fp_report(fps_before)

    # ═══ PHASE 2: IMPROVED ═══
    log.info(f"\n{'#'*80}")
    log.info("PHASE 2: IMPROVED — Modified failure_cascade + regime_transition")
    log.info(f"{'#'*80}")

    results_after = run_walk_forward(trades_df, use_improved=True)
    metrics_after = compute_metrics(results_after, label="AFTER")
    fps_after = categorize_false_positives(results_after)

    log.info(f"\nWPR: {metrics_after['wpr']}% | LRR: {metrics_after['lrr']}% | PF: {metrics_after['profit_factor']}")
    log.info(f"Expectancy: ${metrics_after['expectancy']} | MaxDD: ${metrics_after['max_drawdown']}")
    log.info(f"Trades: {metrics_after['post_filter_trades']} | MCC: {metrics_after['mcc']}")
    log.info(f"CM: TP={metrics_after['confusion_matrix']['TP']} FP={metrics_after['confusion_matrix']['FP']} "
            f"TN={metrics_after['confusion_matrix']['TN']} FN={metrics_after['confusion_matrix']['FN']}")

    if fps_after:
        print_fp_report(fps_after, "REMAINING FALSE POSITIVES")
    else:
        log.info("\nNo false positives remaining!")

    print_comparison(metrics_before, metrics_after)

    # Save report
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "before": {k: v for k, v in metrics_before.items() if k != "equity_curve"},
        "after": {k: v for k, v in metrics_after.items() if k != "equity_curve"},
        "false_positives_before": fps_before,
        "false_positives_after": fps_after,
    }
    with open(str(REPORT_PATH), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"\nReport saved to {REPORT_PATH}")

    # Final verdict
    log.info(f"\n{'='*80}")
    log.info("FINAL VERDICT")
    log.info(f"{'='*80}")
    log.info(f"WPR >= 95%: {'YES' if metrics_after['wpr'] >= 95.0 else 'NO'} ({metrics_after['wpr']}%)")
    log.info(f"LRR: {metrics_after['lrr']}%")
    log.info(f"Expectancy: ${metrics_after['expectancy']} (baseline: ${metrics_before['expectancy']})")

    if metrics_after['wpr'] >= 95.0 and metrics_after['lrr'] > metrics_before['lrr'] * 0.5:
        log.info("\nRECOMMENDATION: MERGE into production.")
    elif metrics_after['wpr'] >= 95.0:
        log.info("\nRECOMMENDATION: MERGE with monitoring (LRR decreased but WPR target met).")
    else:
        log.info("\nRECOMMENDATION: Further improvement needed.")

    return metrics_before, metrics_after, fps_before, fps_after


if __name__ == "__main__":
    main()
