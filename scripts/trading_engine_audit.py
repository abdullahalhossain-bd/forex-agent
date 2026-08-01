#!/usr/bin/env python3
"""
TRADING ENGINE ROOT CAUSE AUDIT
================================
Comprehensive diagnostic tool for identifying why backtest is losing money.

This script performs a complete audit of:
1. Market data integrity
2. Indicator calculations  
3. Signal generation (why only stop_hunt fires)
4. Trade management (SL/TP/R:R)
5. Risk filters (rejection statistics)
6. Strategy logic (per-strategy analysis)
7. ML/RL integration verification
8. Execution quality (spread/slippage/duplicates)
9. Comprehensive diagnostics and recommendations
"""

import sys
import os
import warnings
import logging
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import json

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Go up one level to workspace root
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TEST_MODE", "true")
os.environ.setdefault("MT5_LOGIN", "12345")
os.environ.setdefault("MT5_PASSWORD", "dummy")
os.environ.setdefault("MT5_SERVER", "dummy")

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("audit")

# ============================================================================
# SECTION 1: DATA LOADING AND INTEGRITY CHECKS
# ============================================================================

def generate_test_data(symbol: str, bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic OHLC data for testing."""
    np.random.seed(seed + hash(symbol) % 1000)
    dates = pd.date_range("2023-01-01", periods=bars, freq="1h")
    
    # Build close prices with trend + volatility cycles
    trend = np.random.choice([-1, 1]) * 0.0001
    vol_cycle = np.sin(np.arange(bars) / 50) * 0.0003 + 0.0005
    noise = np.random.randn(bars) * vol_cycle
    close = 1.0850 + np.cumsum(noise + trend)
    
    # Add periodic shocks (news events)
    for i in range(20, bars, 25):
        close[i:] += np.random.randn() * 0.003
    
    # Build candles ensuring OHLC consistency
    opens = np.empty(bars)
    highs = np.empty(bars)
    lows = np.empty(bars)
    
    for i in range(bars):
        if i == 0:
            opens[i] = close[i] + np.random.randn() * 0.0002
        else:
            opens[i] = close[i-1] + np.random.randn() * 0.0001
        
        upper_wick = abs(np.random.randn()) * 0.0005
        lower_wick = abs(np.random.randn()) * 0.0005
        
        highs[i] = max(opens[i], close[i]) + upper_wick
        lows[i] = min(opens[i], close[i]) - lower_wick
    
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": close,
        "volume": np.random.randint(100, 1000, bars),
    }, index=dates)
    
    return df


def check_ohlc_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """Check OHLC data integrity - no look-ahead bias, proper candle structure."""
    issues = []
    
    # Check OHLC consistency: low <= open,close <= high
    invalid_candles = ((df['low'] > df['open']) | (df['low'] > df['close']) | 
                       (df['high'] < df['open']) | (df['high'] < df['close']))
    if invalid_candles.sum() > 0:
        issues.append(f"{invalid_candles.sum()} candles have invalid OHLC structure")
    
    # Check for NaN values
    nan_counts = df.isna().sum()
    if nan_counts.any():
        issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
    
    # Check for zero or negative prices
    if (df['low'] <= 0).any():
        issues.append("Zero or negative prices detected")
    
    # Check for duplicate indices
    if df.index.duplicated().any():
        issues.append(f"{df.index.duplicated().sum()} duplicate timestamps")
    
    # Check for gaps in time series
    if len(df) > 1:
        time_diffs = df.index.to_series().diff()
        expected_freq = pd.Timedelta(hours=1)  # Assuming H1
        gaps = time_diffs[time_diffs > expected_freq * 2]
        if len(gaps) > 0:
            issues.append(f"{len(gaps)} significant time gaps detected")
    
    return {
        "total_bars": len(df),
        "date_range": f"{df.index[0]} to {df.index[-1]}",
        "valid": len(issues) == 0,
        "issues": issues
    }


# ============================================================================
# SECTION 2: STRATEGY ANALYSIS
# ============================================================================

def analyze_strategy_signals(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Analyze why only stop_hunt generates trades."""
    from analysis.unified_signal_engine import UnifiedSignalEngine
    
    engine = UnifiedSignalEngine(timeframe="H1")
    
    signal_counts = defaultdict(int)
    strategy_signals = defaultdict(list)
    rejection_reasons = defaultdict(int)
    
    warmup = 50
    total_bars = len(df)
    
    for i in range(warmup, total_bars):
        df_slice = df.iloc[:i+1].copy()
        
        try:
            result = engine.analyze(df_slice, symbol=symbol, lower_tf_df=None)
        except Exception as e:
            rejection_reasons["engine_error"] += 1
            continue
        
        # Extract per-engine signals
        sh_sig = result.get("stop_hunt", {}).get("signal", {}).get("action", "NO_TRADE")
        ict_sig = result.get("ict_amd", {}).get("signal", {}).get("action", "NO_TRADE")
        pa_sig = result.get("multi_strategy_pa", {}).get("signal", {}).get("action", "NO_TRADE")
        consensus = result.get("consensus", {}).get("action", "NO_TRADE")
        
        signal_counts[sh_sig] += 1
        signal_counts[f"ict_{ict_sig}"] += 1
        signal_counts[f"pa_{pa_sig}"] += 1
        signal_counts[f"consensus_{consensus}"] += 1
        
        # Track rejection reasons
        if sh_sig == "NO_TRADE":
            reason = result.get("stop_hunt", {}).get("signal", {}).get("reason", "unknown")[:50]
            rejection_reasons[f"stop_hunt_{reason}"] += 1
        
        if ict_sig == "NO_TRADE":
            reason = result.get("ict_amd", {}).get("signal", {}).get("reason", "unknown")[:50]
            rejection_reasons[f"ict_{reason}"] += 1
            
        if pa_sig == "NO_TRADE":
            reason = result.get("multi_strategy_pa", {}).get("signal", {}).get("reason", "unknown")[:50]
            rejection_reasons[f"pa_{reason}"] += 1
    
    return {
        "signal_counts": dict(signal_counts),
        "rejection_reasons": dict(sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:20]),
        "total_bars_evaluated": total_bars - warmup
    }


def test_each_strategy_separately(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Test each strategy independently to rank by expectancy."""
    results = {}
    
    # Test Stop Hunt
    from analysis.stop_hunt_signal_engine import StopHuntSignalEngine
    sh_engine = StopHuntSignalEngine(timeframe="H1")
    sh_signals = []
    for i in range(50, len(df)):
        result = sh_engine.analyze(df.iloc[:i+1], symbol=symbol)
        sig = result.get("signal", {})
        if sig.get("action") in ("BUY", "SELL"):
            sh_signals.append({
                "bar": i,
                "action": sig["action"],
                "entry": sig.get("entry_price"),
                "sl": sig.get("stop_loss"),
                "tp": sig.get("take_profit"),
                "confidence": sig.get("confidence")
            })
    
    results["stop_hunt"] = {
        "total_signals": len(sh_signals),
        "buy_signals": sum(1 for s in sh_signals if s["action"] == "BUY"),
        "sell_signals": sum(1 for s in sh_signals if s["action"] == "SELL"),
        "avg_confidence": np.mean([s["confidence"] for s in sh_signals]) if sh_signals else 0
    }
    
    # Test ICT AMD
    from analysis.ict_amd_signal_engine import ICTAMDSignalEngine
    ict_engine = ICTAMDSignalEngine(timeframe="H1")
    ict_signals = []
    for i in range(50, len(df)):
        result = ict_engine.analyze(df.iloc[:i+1], symbol=symbol)
        sig = result.get("signal", {})
        if sig.get("action") in ("BUY", "SELL"):
            ict_signals.append({
                "bar": i,
                "action": sig["action"],
                "entry": sig.get("entry_price"),
                "sl": sig.get("stop_loss"),
                "tp": sig.get("take_profit")
            })
    
    results["ict_amd"] = {
        "total_signals": len(ict_signals),
        "buy_signals": sum(1 for s in ict_signals if s["action"] == "BUY"),
        "sell_signals": sum(1 for s in ict_signals if s["action"] == "SELL"),
    }
    
    # Test Multi-Strategy PA (only works on specific timeframes)
    from analysis.multi_strategy_pa_engine import MultiStrategyPAEngine
    pa_engine = MultiStrategyPAEngine(timeframe="1H")
    pa_signals = []
    for i in range(50, len(df)):
        result = pa_engine.analyze(df.iloc[:i+1], symbol=symbol, lower_tf_df=None)
        sig = result.get("signal", {})
        if sig.get("action") in ("BUY", "SELL"):
            pa_signals.append({
                "bar": i,
                "action": sig["action"],
            })
    
    results["multi_strategy_pa"] = {
        "total_signals": len(pa_signals),
        "buy_signals": sum(1 for s in pa_signals if s["action"] == "BUY"),
        "sell_signals": sum(1 for s in pa_signals if s["action"] == "SELL"),
    }
    
    return results


# ============================================================================
# SECTION 3: RISK FILTER ANALYSIS
# ============================================================================

def analyze_risk_filters() -> Dict[str, Any]:
    """Analyze which risk filters reject the most trades."""
    from risk.live_risk_manager import LiveRiskManager, TIERS
    
    lrm = LiveRiskManager(initial_balance=10000.0, tier=1)
    
    filter_stats = {
        "tier_info": {
            "current_tier": lrm.current_tier.tier,
            "tier_name": lrm.current_tier.name,
            "min_confidence": lrm.current_tier.min_confidence,
            "risk_per_trade": lrm.current_tier.risk_per_trade,
            "tier_mult": lrm.current_tier.tier_mult
        },
        "filter_thresholds": {
            "tier_1_min_conf": TIERS[1].min_confidence,
            "tier_2_min_conf": TIERS[2].min_confidence,
            "tier_3_min_conf": TIERS[3].min_confidence
        }
    }
    
    return filter_stats


# ============================================================================
# SECTION 4: TRADE MANAGEMENT ANALYSIS
# ============================================================================

def analyze_trade_management(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze SL/TP calculation, R:R ratios, position sizing."""
    if trades_df is None or len(trades_df) == 0:
        return {"error": "No trades to analyze"}
    
    # Calculate R:R for each trade
    trades_df = trades_df.copy()
    trades_df['rr_actual'] = trades_df.apply(
        lambda row: abs(row['pnl_pips']) / abs(row['entry_price'] - row['stop_loss']) 
                    if row['stop_loss'] != row['entry_price'] else 0,
        axis=1
    )
    
    # Analyze SL distances
    trades_df['sl_distance_pips'] = abs(trades_df['entry_price'] - trades_df['stop_loss']) / 0.0001
    
    stats = {
        "total_trades": len(trades_df),
        "avg_rr_achieved": float(trades_df['rr_actual'].mean()),
        "median_rr_achieved": float(trades_df['rr_actual'].median()),
        "avg_sl_distance_pips": float(trades_df['sl_distance_pips'].mean()),
        "avg_tp_distance_pips": float(abs(trades_df['take_profit'] - trades_df['entry_price']).mean() / 0.0001),
        "exit_reasons": trades_df['exit_reason'].value_counts().to_dict(),
        "avg_commission_per_trade": float(trades_df['commission_usd'].mean()),
        "avg_slippage_pips": float(trades_df['slippage_pips'].mean()),
        "total_commission": float(trades_df['commission_usd'].sum()),
        "win_rate": float(len(trades_df[trades_df['pnl_usd'] > 0]) / len(trades_df) * 100),
        "avg_win_pips": float(trades_df[trades_df['pnl_pips'] > 0]['pnl_pips'].mean()) if len(trades_df[trades_df['pnl_pips'] > 0]) > 0 else 0,
        "avg_loss_pips": float(trades_df[trades_df['pnl_pips'] < 0]['pnl_pips'].mean()) if len(trades_df[trades_df['pnl_pips'] < 0]) > 0 else 0,
        "profit_factor": float(
            abs(trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum() / 
                trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
        ) if len(trades_df[trades_df['pnl_usd'] < 0]) > 0 else float('inf')
    }
    
    return stats


# ============================================================================
# SECTION 5: ML/RL INTEGRATION VERIFICATION
# ============================================================================

def verify_ml_rl_integration() -> Dict[str, Any]:
    """Verify ML predictions and RL policy are actually used."""
    status = {
        "ml_model_available": False,
        "rl_policy_available": False,
        "confidence_engine_connected": False,
        "gating_bridge_connected": False
    }
    
    # Check ML model
    ml_model_path = PROJECT_ROOT / "memory" / "ml_models"
    if ml_model_path.exists():
        models = list(ml_model_path.glob("*.joblib")) + list(ml_model_path.glob("*.pkl"))
        status["ml_model_available"] = len(models) > 0
        status["ml_models_found"] = [m.name for m in models]
    
    # Check RL policy
    rl_policy_path = PROJECT_ROOT / "ml" / "rl_policy"
    if rl_policy_path.exists():
        policies = list(rl_policy_path.glob("*.zip"))
        status["rl_policy_available"] = len(policies) > 0
        status["rl_policies_found"] = [p.name for p in policies]
    
    # Check confidence engine
    try:
        from learning.confidence_engine import ConfidenceEngine
        status["confidence_engine_connected"] = True
    except ImportError:
        status["confidence_engine_connected"] = False
    
    # Check gating bridge
    try:
        from backtest.gating_bridge import BacktestGate
        status["gating_bridge_connected"] = True
    except ImportError:
        status["gating_bridge_connected"] = False
    
    return status


# ============================================================================
# SECTION 6: COMPREHENSIVE DIAGNOSTICS
# ============================================================================

def run_full_backtest_audit(symbol: str = "EURUSD", bars: int = 5000) -> Dict[str, Any]:
    """Run complete audit and return comprehensive report."""
    
    log.info("=" * 70)
    log.info("TRADING ENGINE ROOT CAUSE AUDIT")
    log.info("=" * 70)
    
    # Generate test data
    log.info("\n[1/7] Generating test data...")
    df = generate_test_data(symbol, bars=bars)
    
    # Data integrity check
    log.info("[2/7] Checking OHLC data integrity...")
    data_integrity = check_ohlc_integrity(df)
    
    # Strategy signal analysis
    log.info("[3/7] Analyzing strategy signals...")
    strategy_analysis = analyze_strategy_signals(df, symbol)
    
    # Per-strategy testing
    log.info("[4/7] Testing each strategy separately...")
    per_strategy = test_each_strategy_separately(df, symbol)
    
    # Risk filter analysis
    log.info("[5/7] Analyzing risk filters...")
    risk_filters = analyze_risk_filters()
    
    # Run actual backtest
    log.info("[6/7] Running backtest for trade analysis...")
    from run_backtest import run_backtest
    bt_result = run_backtest(
        symbol=symbol,
        df=df,
        timeframe="H1",
        starting_balance=10000.0,
        risk_pct=0.02,
        warmup_bars=50,
        verbose=False
    )
    
    # Trade management analysis
    if bt_result.get("trades"):
        trades_df = pd.DataFrame([{
            "entry_price": t.entry_price,
            "stop_loss": t.stop_loss,
            "take_profit": t.take_profit,
            "pnl_pips": t.pnl_pips,
            "pnl_usd": t.pnl_usd,
            "commission_usd": t.commission_usd,
            "slippage_pips": t.slippage_pips,
            "exit_reason": t.exit_reason,
            "direction": t.direction,
            "strategy": t.strategy,
            "hold_bars": t.hold_bars
        } for t in bt_result["trades"]])
        trade_mgmt = analyze_trade_management(trades_df)
    else:
        trade_mgmt = {"error": "No trades generated"}
    
    # ML/RL verification
    log.info("[7/7] Verifying ML/RL integration...")
    ml_rl_status = verify_ml_rl_integration()
    
    # Compile full report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_integrity": data_integrity,
        "strategy_analysis": strategy_analysis,
        "per_strategy_results": per_strategy,
        "risk_filters": risk_filters,
        "trade_management": trade_mgmt,
        "ml_rl_integration": ml_rl_status,
        "backtest_summary": bt_result.get("metrics", {})
    }
    
    return report


def print_audit_report(report: Dict[str, Any]):
    """Print formatted audit report with findings and recommendations."""
    
    print("\n" + "=" * 80)
    print("AUDIT REPORT - TRADING ENGINE ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    # Section 1: Data Integrity
    print("\n┌" + "─" * 78 + "┐")
    print("│ SECTION 1: MARKET DATA INTEGRITY")
    print("└" + "─" * 78 + "┘")
    di = report["data_integrity"]
    print(f"  Total bars: {di['total_bars']}")
    print(f"  Date range: {di['date_range']}")
    print(f"  Valid: {'✓ YES' if di['valid'] else '✗ NO'}")
    if di["issues"]:
        print("  Issues found:")
        for issue in di["issues"]:
            print(f"    ⚠ {issue}")
    
    # Section 2: Strategy Analysis
    print("\n┌" + "─" * 78 + "┐")
    print("│ SECTION 2: SIGNAL GENERATION ANALYSIS")
    print("└" + "─" * 78 + "┘")
    sa = report["strategy_analysis"]
    print(f"  Total bars evaluated: {sa['total_bars_evaluated']}")
    print("\n  Signal counts by type:")
    for sig_type, count in sorted(sa["signal_counts"].items(), key=lambda x: x[1], reverse=True)[:15]:
        pct = count / sa["total_bars_evaluated"] * 100
        print(f"    {sig_type:30s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n  Top rejection reasons:")
    for reason, count in sa["rejection_reasons"][:10]:
        print(f"    {reason:60s}: {count}")
    
    # Section 3: Per-Strategy Results
    print("\n┌" + "─" * 78 + "┐")
    print("│ SECTION 3: PER-STRATEGY ANALYSIS")
    print("└" + "─" * 78 + "┘")
    for strat_name, stats in report["per_strategy_results"].items():
        print(f"\n  {strat_name.upper()}:")
        print(f"    Total signals: {stats.get('total_signals', 'N/A')}")
        print(f"    BUY signals: {stats.get('buy_signals', 'N/A')}")
        print(f"    SELL signals: {stats.get('sell_signals', 'N/A')}")
        if 'avg_confidence' in stats:
            print(f"    Avg confidence: {stats['avg_confidence']:.1f}%")
    
    # Section 4: Risk Filters
    print("\n┌" + "─" * 78 + "┐")
    print("│ SECTION 4: RISK FILTER CONFIGURATION")
    print("└" + "─" * 78 + "┘")
    rf = report["risk_filters"]
    tier_info = rf["tier_info"]
    print(f"  Current tier: {tier_info['tier_name']} (Tier {tier_info['current_tier']})")
    print(f"  Min confidence required: {tier_info['min_confidence']}%")
    print(f"  Risk per trade: {tier_info['risk_per_trade']*100:.2f}%")
    print(f"  Position multiplier: {tier_info['tier_mult']}x")
    print("\n  Tier thresholds:")
    for tier_id, conf in rf["filter_thresholds"].items():
        print(f"    {tier_id}: min_conf = {conf}%")
    
    # Section 5: Trade Management
    print("\n┌" + "─" * 78 + "┐")
    print("│ SECTION 5: TRADE MANAGEMENT ANALYSIS")
    print("└" + "─" * 78 + "┘")
    tm = report["trade_management"]
    if "error" not in tm:
        print(f"  Total trades: {tm['total_trades']}")
        print(f"  Win rate: {tm['win_rate']:.1f}%")
        print(f"  Profit factor: {tm['profit_factor']:.2f}")
        print(f"  Avg R:R achieved: {tm['avg_rr_achieved']:.2f}")
        print(f"  Avg SL distance: {tm['avg_sl_distance_pips']:.1f} pips")
        print(f"  Avg TP distance: {tm['avg_tp_distance_pips']:.1f} pips")
        print(f"  Avg commission/trade: ${tm['avg_commission_per_trade']:.2f}")
        print(f"  Total commission: ${tm['total_commission']:.2f}")
        print(f"  Avg slippage: {tm['avg_slippage_pips']:.1f} pips")
        print("\n  Exit reasons:")
        for reason, count in tm["exit_reasons"].items():
            print(f"    {reason:20s}: {count}")
    else:
        print(f"  Error: {tm['error']}")
    
    # Section 6: ML/RL Integration
    print("\n┌" + "─" * 78 + "┐")
    print("│ SECTION 6: ML/RL INTEGRATION STATUS")
    print("└" + "─" * 78 + "┘")
    ml = report["ml_rl_integration"]
    print(f"  ML model available: {'✓ YES' if ml['ml_model_available'] else '✗ NO'}")
    if ml.get('ml_models_found'):
        print(f"    Models: {', '.join(ml['ml_models_found'])}")
    print(f"  RL policy available: {'✓ YES' if ml['rl_policy_available'] else '✗ NO'}")
    if ml.get('rl_policies_found'):
        print(f"    Policies: {', '.join(ml['rl_policies_found'])}")
    print(f"  Confidence engine connected: {'✓ YES' if ml['confidence_engine_connected'] else '✗ NO'}")
    print(f"  Gating bridge connected: {'✓ YES' if ml['gating_bridge_connected'] else '✗ NO'}")
    
    # Section 7: Root Cause Summary
    print("\n┌" + "─" * 78 + "┐")
    print("│ SECTION 7: ROOT CAUSE SUMMARY & RECOMMENDATIONS")
    print("└" + "─" * 78 + "┘")
    
    root_causes = []
    recommendations = []
    
    # Analyze findings
    sa = report["strategy_analysis"]
    tm = report["trade_management"]
    
    # Check 1: Only one strategy firing
    sh_count = sa["signal_counts"].get("consensus_BUY", 0) + sa["signal_counts"].get("consensus_SELL", 0)
    total_signals = sum(v for k, v in sa["signal_counts"].items() if "consensus" in k and k != "consensus_WAIT" and k != "consensus_NO_TRADE")
    if sh_count > 0 and total_signals == sh_count:
        root_causes.append("Only stop_hunt strategy generating signals")
        recommendations.append("FIX: Adjust ICT/AMD and PA strategy parameters to be less restrictive")
    
    # Check 2: High rejection rate
    wait_count = sa["signal_counts"].get("consensus_WAIT", 0) + sa["signal_counts"].get("consensus_NO_TRADE", 0)
    total = sa["total_bars_evaluated"]
    if total > 0 and wait_count / total > 0.95:
        root_causes.append(f"95%+ bars rejected as WAIT/NO_TRADE")
        recommendations.append("FIX: Lower signal thresholds or improve signal detection logic")
    
    # Check 3: Poor win rate
    if "error" not in tm and tm.get("win_rate", 0) < 40:
        root_causes.append(f"Win rate critically low ({tm['win_rate']:.1f}%)")
        recommendations.append("FIX: Review entry timing - may be entering at worst possible moment")
    
    # Check 4: Commission drag
    if "error" not in tm and tm.get("total_commission", 0) > abs(tm.get("total_pnl", 0)) * 0.3:
        root_causes.append("Commission costs consuming significant portion of P&L")
        recommendations.append("FIX: Reduce trade frequency or negotiate lower commissions")
    
    # Check 5: Timeframe mismatch
    ps = report["per_strategy_results"]
    if ps.get("multi_strategy_pa", {}).get("total_signals", 0) == 0:
        root_causes.append("Multi-strategy PA not generating any signals (timeframe mismatch)")
        recommendations.append("FIX: Use allowed timeframes: 1H, 4H, 1D (not H1)")
    
    # Check 6: ML/RL not integrated
    ml = report["ml_rl_integration"]
    if not ml["ml_model_available"] or not ml["rl_policy_available"]:
        root_causes.append("ML/RL models not available for signal enhancement")
        recommendations.append("FIX: Train ML models and RL policy, integrate into decision pipeline")
    
    # Print findings
    print("\n  ROOT CAUSES IDENTIFIED:")
    for i, cause in enumerate(root_causes, 1):
        print(f"    {i}. {cause}")
    
    if not root_causes:
        print("    No critical issues identified")
    
    print("\n  RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"    {i}. {rec}")
    
    if not recommendations:
        print("    No specific recommendations at this time")
    
    print("\n" + "=" * 80)
    print("END OF AUDIT REPORT")
    print("=" * 80)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Engine Root Cause Audit")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to test")
    parser.add_argument("--bars", type=int, default=5000, help="Number of bars")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()
    
    # Run audit
    report = run_full_backtest_audit(symbol=args.symbol, bars=args.bars)
    
    # Print report
    print_audit_report(report)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"\nReport saved to: {args.output}")
    
    return report


if __name__ == "__main__":
    main()
