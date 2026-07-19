#!/usr/bin/env python3
"""run_backtest.py — Event-Driven Backtest Runner"""
import argparse, json, sys, os, warnings, logging
from pathlib import Path
from datetime import datetime, timezone
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TEST_MODE", "true")
os.environ.setdefault("MT5_LOGIN", "108647429")
os.environ.setdefault("MT5_PASSWORD", "O@L5PnXe")
os.environ.setdefault("MT5_SERVER", "MetaQuotes-Demo")
import numpy as np, pandas as pd
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("backtest_runner")
log.setLevel(logging.INFO)

def _stable_symbol_hash(symbol: str) -> int:
    """CRITICAL FIX: Python's built-in hash() randomizes string hashes per
    process (PYTHONHASHSEED, on by default since 3.3) for security reasons.
    Using hash(symbol) as part of a "reproducible" seed meant the *same*
    --synthetic backtest command produced a genuinely different random
    price series on every single run -- explaining why repeated "identical"
    backtests showed wildly different win rates/PF/drawdown (29 trades
    WR=55% one run, 23 trades WR=43% the next, 25 trades WR=96% after
    that, all from the "same" command). A backtest that can't reproduce
    itself cannot be statistically validated at all. Fixed with a stable,
    deterministic hash (sum of char codes) that gives the same value in
    every process, every machine, every run.
    """
    return sum(ord(c) for c in symbol)


def generate_synthetic_data(symbol, bars=500, seed=42):
    """Generate realistic synthetic OHLC data with proper candle structure."""
    np.random.seed((seed + _stable_symbol_hash(symbol)) % (2**32 - 1))
    dates = pd.date_range("2023-01-01", periods=bars, freq="1h")

    # Build close prices with trend + volatility cycles
    trend = np.random.choice([-1, 1]) * 0.0001
    vol_cycle = np.sin(np.arange(bars) / 50) * 0.0003 + 0.0005  # volatility cycles
    noise = np.random.randn(bars) * vol_cycle
    close = 1.0850 + np.cumsum(noise + trend)

    # Add periodic shocks (news events)
    for i in range(20, bars, 25):
        close[i:] += np.random.randn() * 0.003

    # Build candles ensuring OHLC consistency
    opens = np.empty(bars)
    highs = np.empty(bars)
    lows = np.empty(bars)
    closes = close

    for i in range(bars):
        if i == 0:
            opens[i] = closes[i] + np.random.randn() * 0.0002
        else:
            opens[i] = closes[i-1] + np.random.randn() * 0.0001  # gap from prev close

        # Wick sizes (random but realistic)
        upper_wick = abs(np.random.randn()) * 0.0005
        lower_wick = abs(np.random.randn()) * 0.0005

        # Ensure: low <= min(open, close) and high >= max(open, close)
        highs[i] = max(opens[i], closes[i]) + upper_wick
        lows[i] = min(opens[i], closes[i]) - lower_wick

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.random.randint(100, 1000, bars),
    }, index=dates)

    # Add realistic indicators
    # RSI: bounded 10-90 (never negative, never > 100)
    price_changes = df["close"].diff()
    gains = price_changes.clip(lower=0).rolling(14, min_periods=1).mean()
    losses = (-price_changes.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gains / (losses + 1e-10)
    df["rsi"] = (100 - 100 / (1 + rs)).clip(5, 95)

    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["sma_50"] = df["close"].rolling(50, min_periods=1).mean()
    df["sma_200"] = df["close"].rolling(200, min_periods=1).mean()

    # ATR: from actual high-low ranges (realistic)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()

    # ADX: realistic range with occasional strong trends
    adx_base = np.random.uniform(15, 25, bars)
    # Add trend spikes
    for i in range(50, bars, 80):
        adx_base[i:min(i+30, bars)] += np.random.uniform(10, 25)
    df["adx"] = np.clip(adx_base, 10, 60)

    return df

def run_backtest(symbol, df, timeframe="H1", starting_balance=10000.0, risk_pct=0.02, warmup_bars=50, max_open_trades=3, max_hold_bars=100, spread_pips=None, commission_per_lot=7.0, slippage_pips=2.0, verbose=False, sim_seed=42):
    # CRITICAL FIX (reproducibility, part 2): BrokerSimulator draws slippage
    # from np.random.normal() and partial-fill behavior from Python's stdlib
    # `random` module -- a SEPARATE generator from numpy's, never seeded
    # anywhere in this file. Even after fixing the synthetic-data seed, repeat
    # runs of the identical backtest still drifted in P&L ($-531 / -527 /
    # -551 across 3 identical-trade-count runs) purely from unseeded
    # execution-cost simulation. Seed both here, explicitly, so the full
    # pipeline (data -> signals -> simulated execution) is one deterministic
    # function of (data, seed) -- a hard prerequisite for any of the
    # Monte Carlo/bootstrap/t-test results to mean anything.
    import random as _random
    _random.seed(sim_seed)
    np.random.seed(sim_seed)
    from backtest.broker_sim import BrokerSimulator, DEFAULT_SPREAD_PIPS
    from backtest.metrics import calculate_metrics
    from analysis._engine_utils import atr_value
    try:
        from analysis.unified_signal_engine import UnifiedSignalEngine
    except ImportError as e:
        return {"trades": [], "metrics": None, "error": str(e)}
    if spread_pips is None: spread_pips = DEFAULT_SPREAD_PIPS.get(symbol, 2.0)
    broker = BrokerSimulator(starting_balance=starting_balance, commission_per_lot=commission_per_lot, slippage_pips=slippage_pips)
    engine = UnifiedSignalEngine(timeframe=timeframe)
    open_trades, closed_trades, equity_curve = [], [], [starting_balance]
    entry_bar = {}  # trade_id -> bar index it was opened on (FIX: hold_bars must be
                     # measured from each trade's own entry, not from the warmup start)
    # FIX (duplicate-signal loop): the sub-engines (e.g. stop_hunt_signal_engine.py)
    # are STATELESS — each analyze() call only looks at a rolling window (e.g. "last
    # 20 candles") and has no memory of whether a setup was already traded. Because
    # this loop calls analyze() on every single bar with an expanding window, one
    # underlying stop-hunt/ICT/PA event stays "detected" for ~20 consecutive bars and
    # was previously re-opened as a brand-new trade on every one of those bars —
    # producing 5-8 near-identical, correlated trades off a single setup, each paying
    # fresh commission+slippage and usually all failing together (same SL cluster).
    # This was the dominant cause of the backtest's losses. Fix: remember the last
    # bar index + SL level a given (strategy, direction) fired a trade at; suppress
    # re-entry on the same symbol/strategy/direction while (a) we're still inside the
    # sub-engine's own lookback window AND (b) the new SL is essentially the same zone
    # (i.e. it's the same underlying event, not a genuinely new setup).
    SIGNAL_COOLDOWN_BARS = 20  # mirrors stop_hunt_signal_engine.py's 20-candle scan window
    last_signal = {}  # (strategy_name, action) -> {"bar": i, "sl": sl}
    total_bars = len(df)

    # ROBUSTNESS FIX #1 (confidence gate): the unified engine's consensus
    # already computes a Low/Medium/High confidence bucket, but this loop
    # used to take ANY BUY/SELL regardless of bucket. Low-confidence =
    # exactly one engine voted with no corroboration — that's the highest
    # false-positive-rate bucket. Requiring Medium+ cuts trade *count* but
    # raises trade *quality*, which is the stated goal (robustness > frequency).
    MIN_CONFIDENCE = {"Low": 0, "Medium": 1, "High": 2}
    MIN_CONFIDENCE_REQUIRED = 1  # Medium or better

    # ROBUSTNESS FIX #2 (correlation cap): two concurrent same-symbol,
    # same-direction trades are not two independent statistical draws --
    # they're ~one bet with double commission/slippage drag, and they
    # inflate the apparent trade count that Monte Carlo/t-test treat as
    # independent samples. Block opening a new trade in the same direction
    # on the same symbol while one is already open (max_open_trades still
    # caps total concurrent exposure across directions/strategies).
    open_directions = set()  # currently-open {action} for this symbol

    log.info(f"Starting: {symbol} {timeframe} | {total_bars} bars | balance=${starting_balance} | risk={risk_pct*100}%")
    rejection_stats = {"NO_TRADE": 0, "WAIT": 0, "engine_error": 0, "max_trades": 0, "total_bars": 0,
                        "low_confidence": 0, "correlated_direction": 0}
    for i in range(warmup_bars, total_bars):
        current_time = df.index[i]
        rejection_stats["total_bars"] += 1
        still_open = []
        for trade in open_trades:
            opened_at = entry_bar.get(trade.trade_id, i)
            result = broker.check_exit(trade, float(df.iloc[i]["high"]), float(df.iloc[i]["low"]), float(df.iloc[i]["close"]), current_time)
            if result:
                result.hold_bars = i - opened_at
                closed_trades.append(result)
                entry_bar.pop(trade.trade_id, None)
                # FIX (visibility gap): previously only OPEN was logged -- no
                # way to see from the log whether/how trades resolved.
                if verbose:
                    log.info(f"  [{current_time}] {result.exit_reason} {result.direction} {result.symbol} "
                             f"@ {result.exit_price:.5f} pnl=${result.pnl_usd:.2f} ({result.pnl_pips:+.1f}p) "
                             f"balance=${broker.get_balance():.2f}")
            else:
                trade.hold_bars = i - opened_at
                if trade.hold_bars > max_hold_bars:
                    closed = broker.close_trade(trade, float(df.iloc[i]["close"]), current_time, "timeout")
                    closed.hold_bars = trade.hold_bars
                    closed_trades.append(closed)
                    entry_bar.pop(trade.trade_id, None)
                    if verbose:
                        log.info(f"  [{current_time}] TIMEOUT {closed.direction} {closed.symbol} "
                                 f"@ {closed.exit_price:.5f} pnl=${closed.pnl_usd:.2f} ({closed.pnl_pips:+.1f}p) "
                                 f"balance=${broker.get_balance():.2f}")
                else: still_open.append(trade)
        open_trades = still_open
        # Rebuild the correlation set from what's ACTUALLY still open (not
        # incrementally), so it never drifts out of sync with real state.
        open_directions = {t.direction for t in open_trades}
        if len(open_trades) >= max_open_trades:
            rejection_stats["max_trades"] += 1
            equity_curve.append(broker.get_balance()); continue
        df_slice = df.iloc[:i+1].copy()
        try: result = engine.analyze(df_slice, symbol=symbol, lower_tf_df=None)
        except Exception as e:
            rejection_stats["engine_error"] += 1
            if verbose and rejection_stats["engine_error"] <= 3:
                log.info(f"  [{current_time}] Engine error: {str(e)[:80]}")
            equity_curve.append(broker.get_balance()); continue
        consensus = result.get("consensus", {}); action = consensus.get("action", "NO_TRADE")
        if action not in ("BUY", "SELL"):
            rejection_stats[action] = rejection_stats.get(action, 0) + 1
            if verbose and i % 50 == 0:  # Print rejection summary every 50 bars
                # Get rejection details from sub-engines
                sh_sig = result.get("stop_hunt", {}).get("signal", {}).get("action", "?")
                ict_sig = result.get("ict_amd", {}).get("signal", {}).get("action", "?")
                pa_sig = result.get("multi_strategy_pa", {}).get("signal", {}).get("action", "?")
                pat_count = len(result.get("detected_patterns", []))
                log.info(f"  [{current_time}] WAIT — SH={sh_sig} ICT={ict_sig} PA={pa_sig} patterns={pat_count} "
                         f"BUY={consensus.get('buy_score',0)} SELL={consensus.get('sell_score',0)}")
            equity_curve.append(broker.get_balance()); continue

        # ROBUSTNESS FIX #1: reject Low-confidence consensus outright. Low
        # means exactly one engine voted with no corroboration -- letting
        # these through was pure false-positive fuel with no offsetting
        # edge evidence.
        conf_bucket = consensus.get("confidence", "Low")
        if MIN_CONFIDENCE.get(conf_bucket, 0) < MIN_CONFIDENCE_REQUIRED:
            rejection_stats["low_confidence"] += 1
            equity_curve.append(broker.get_balance()); continue

        # ROBUSTNESS FIX #2: don't stack a second same-direction trade on
        # the same symbol on top of one already open. Two same-direction
        # positions are correlated risk, not two independent statistical
        # samples -- stacking them inflates trade count without adding real
        # diversification, which is what was destabilizing the Monte
        # Carlo/t-test/bootstrap results.
        if action in open_directions:
            rejection_stats["correlated_direction"] += 1
            equity_curve.append(broker.get_balance()); continue

        signal_data = None; strategy_name = "unified"
        for en, er in [("stop_hunt", result.get("stop_hunt", {}).get("signal", {})), ("ict_amd", result.get("ict_amd", {}).get("signal", {})), ("pa", result.get("multi_strategy_pa", {}).get("signal", {}))]:
            if er.get("action") == action: signal_data = er; strategy_name = en; break
        cp = float(df.iloc[i]["close"]); atr = atr_value(df_slice)
        if not signal_data:
            if action == "BUY": entry, sl, tp = cp, cp - atr*1.5, cp + atr*3
            else: entry, sl, tp = cp, cp + atr*1.5, cp - atr*3
            conf, conf_factor, grade = 50, 0, "C"
        else:
            entry = signal_data.get("entry_price") or cp
            sl = signal_data.get("stop_loss") or signal_data.get("sl")
            tp = signal_data.get("take_profit") or signal_data.get("take_profit_suggested") or signal_data.get("tp")
            if not sl or not tp:
                if not sl: sl = cp - atr*1.5 if action == "BUY" else cp + atr*1.5
                if not tp: tp = cp + atr*3 if action == "BUY" else cp - atr*3
            conf = signal_data.get("confidence", 50)
            if isinstance(conf, str): conf = {"Low": 30, "Medium": 60, "High": 85}.get(conf, 50)
            pats = result.get("detected_patterns", []); conf_factor = len(pats) if pats else 0; grade = "B"

        # FIX (duplicate-signal loop): skip if this is the same underlying
        # setup we already traded recently (same strategy+direction, SL
        # within ~10% of ATR of the last one, still inside the sub-engine's
        # own lookback window). See comment above `last_signal` init.
        key = (strategy_name, action)
        prev = last_signal.get(key)
        if prev is not None and (i - prev["bar"]) < SIGNAL_COOLDOWN_BARS and abs(sl - prev["sl"]) < atr * 0.1:
            rejection_stats["duplicate_signal"] = rejection_stats.get("duplicate_signal", 0) + 1
            equity_curve.append(broker.get_balance()); continue
        last_signal[key] = {"bar": i, "sl": sl}

        # ROBUSTNESS FIX #3: scale risk with conviction instead of firing
        # every trade at the same fixed 2% fraction. A grade-C fallback
        # trade (no sub-engine confirmed the consensus action; generic
        # 1.5x/3x ATR stops) and a High-confidence 2-engine ICT trade were
        # previously sized identically. Fractional sizing by grade/confidence
        # reduces the variance of the return stream -- which directly helps
        # Sharpe/t-stat stability -- without requiring any extra "edge".
        size_mult = {"High": 1.0, "Medium": 0.7}.get(conf_bucket, 0.5)
        if grade == "C":
            size_mult = min(size_mult, 0.5)  # unconfirmed fallback setup -> half size, floor
        risk_amount = broker.get_balance() * risk_pct * size_mult
        sl_dist = abs(entry - sl) / (0.01 if symbol.endswith("JPY") else 0.0001)
        pvl = 10.0
        lot = risk_amount / (sl_dist * pvl) if sl_dist > 0 else 0.01
        lot = max(0.01, min(round(lot, 2), 1.0))
        trade = broker.open_trade(symbol=symbol, direction=action, entry_price=entry, sl=sl, tp=tp, lot=lot, bar_time=current_time, confidence=int(conf), strategy=strategy_name, confluence_factors=conf_factor, quality_grade=grade)
        entry_bar[trade.trade_id] = i
        open_trades.append(trade)
        open_directions.add(action)
        if verbose: log.info(f"  [{current_time}] OPEN {action} {symbol} @ {entry:.5f} lot={lot} (mult={size_mult}x) ({strategy_name})")
        equity_curve.append(broker.get_balance())
    last_close = float(df.iloc[-1]["close"]); last_time = df.index[-1]
    for trade in open_trades: closed_trades.append(broker.close_trade(trade, last_close, last_time, "end_of_backtest"))
    metrics = calculate_metrics(trades=closed_trades, starting_balance=starting_balance, ending_balance=broker.get_balance())
    log.info(f"Done: {symbol} | {len(closed_trades)} trades | WR={metrics.win_rate:.1f}% | PF={metrics.profit_factor:.2f} | P&L=${metrics.total_pnl_usd:.2f} | DD={metrics.max_drawdown_pct:.1f}%")

    # Print rejection summary
    total_eval = rejection_stats["total_bars"]
    if total_eval > 0:
        log.info(f"  Rejection summary ({total_eval} bars evaluated):")
        for reason, count in sorted(rejection_stats.items(), key=lambda x: x[1], reverse=True):
            if reason != "total_bars" and count > 0:
                pct = count / total_eval * 100
                log.info(f"    {reason:15s}: {count:4d} ({pct:.1f}%)")
        signal_bars = total_eval - sum(v for k, v in rejection_stats.items() if k != "total_bars")
        log.info(f"    {'SIGNAL_GENERATED':15s}: {signal_bars:4d} ({signal_bars/total_eval*100:.1f}%)")

    return {"trades": closed_trades, "metrics": metrics, "equity_curve": equity_curve, "symbol": symbol, "timeframe": timeframe, "bars": total_bars, "rejection_stats": rejection_stats}

def _load_pair_df(pair: str, tf: str, bars: int, synthetic: bool):
    """Single data-loading path shared by run_pairs() and main() — MT5 or
    synthetic. Returns None (with a printed reason) if data can't be loaded.
    """
    if synthetic:
        return generate_synthetic_data(pair, bars=bars)
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MT5 not installed. Use --synthetic.")
        return None
    if not mt5.initialize():
        print("ERROR: MT5 not available. Use --synthetic.")
        return None
    tf_map = {"M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
              "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
    rates = mt5.copy_rates_from_pos(pair, tf_map[tf], 0, bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        print(f"No data for {pair}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    # NOTE: no indicator computation here — audit fix (§6.5, four divergent
    # RSI/ATR/ADX implementations). Indicators are now computed exactly once,
    # inside backtest.unified_engine._build_market_out(), via the same
    # canonical chain (data/indicator_registry -> data/indicators_ext ->
    # data/indicators) MarketAgent uses live. Raw OHLCV is all this loader
    # is responsible for.
    return df


def run_pairs(ns: "argparse.Namespace") -> list:
    """Run the SHARED-KERNEL backtest (backtest.unified_engine.run_unified_backtest,
    which drives the exact same AnalysisAgent -> DecisionAgent -> RiskEngine ->
    PositionSizer core.trader.AITrader.evaluate_decision_core() Demo/Real use)
    for one or more pairs, save CSVs, and return the list of
    UnifiedBacktestResult objects.

    This is the ONE backtest entry point used by both:
      - `python run_backtest.py` (this script's own CLI, see main())
      - `python main.py --mode backtest` (see main.py:_run_backtest())
    Previously those were two disconnected/broken paths (§6.4 of the
    execution-parity audit); now both call this function.
    """
    from backtest.unified_engine import run_unified_backtest

    pairs = [p.strip().upper() for p in ns.pairs.split(",")] if getattr(ns, "pairs", "") else [ns.pair.upper()]
    results = []
    for pair in pairs:
        print(f"\n{'='*60}\n  [Shared Kernel] Backtesting {pair} {ns.tf} | {ns.bars} bars\n{'='*60}\n")
        df = _load_pair_df(pair, ns.tf, ns.bars, getattr(ns, "synthetic", False))
        if df is None:
            continue
        result = run_unified_backtest(
            symbol=pair, df=df, timeframe=ns.tf,
            starting_balance=ns.balance,
            max_open_trades=getattr(ns, "max_trades", 3),
            max_hold_bars=getattr(ns, "max_hold", 100),
            spread_pips=getattr(ns, "spread", None),
            commission_per_lot=getattr(ns, "commission", 7.0),
            slippage_pips=getattr(ns, "slippage", 2.0),
            db_path=f"backtest/backtest_run_{pair}_{ns.tf}.db",
            verbose=getattr(ns, "verbose", False),
        )
        results.append(result)
        if result.error:
            print(f"  ERROR: {result.error}")
            continue
        if result.metrics:
            print(result.metrics.to_table())
        if result.trades:
            trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
            csv_path = f"backtest/results_{pair}_{ns.tf}.csv"
            trades_df.to_csv(csv_path, index=False)
            print(f"\n  Trades saved to: {csv_path}")
        if result.rejection_stats:
            total = result.rejection_stats.get("total_bars", 0)
            if total:
                print(f"\n  Rejection summary ({total} bars evaluated):")
                for reason, count in sorted(result.rejection_stats.items(), key=lambda x: x[1], reverse=True):
                    if reason != "total_bars" and count:
                        print(f"    {reason:20s}: {count:4d} ({count/total*100:.1f}%)")
    if len(results) > 1:
        print(f"\n{'='*60}\n  MULTI-PAIR SUMMARY (shared kernel)\n{'='*60}")
        for r in results:
            m = r.metrics
            if m:
                print(f"  {r.symbol:10s} : {m.total_trades:3d} trades | WR={m.win_rate:.1f}% | "
                      f"PF={m.profit_factor:.2f} | P&L=${m.total_pnl_usd:.2f}")
        print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="Event-Driven Backtest Runner")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--tf", default="H1", choices=["M15", "M30", "H1", "H4", "D1"])
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=0.02)
    parser.add_argument("--spread", type=float, default=None)
    parser.add_argument("--commission", type=float, default=7.0)
    parser.add_argument("--slippage", type=float, default=2.0)
    parser.add_argument("--max-trades", type=int, default=3)
    parser.add_argument("--max-hold", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    # FIX (2026-07-15): these two flags used to raise
    # "error: unrecognized arguments: --days 60 --report".
    parser.add_argument("--days", type=int, default=None,
                         help="Convenience alias for --bars, expressed in days instead of bar count. "
                              "Converted to bars using the selected --tf (e.g. H1 -> 24 bars/day).")
    parser.add_argument("--report", action="store_true",
                         help="No-op flag for compatibility — a full text report is always printed. "
                              "Kept so old commands using --report don't error out.")
    parser.add_argument("--legacy-engine", action="store_true",
                         help="Use the OLD UnifiedSignalEngine-only backtest loop instead of the "
                              "shared decision kernel (backtest.unified_engine). This engine has "
                              "no risk engine, no confidence/news/session gating, and validates "
                              "different logic than Demo/Real trade on — see the execution-parity "
                              "audit (Critical items 1-3). Kept only for A/B comparison against "
                              "old numbers; NOT representative of live behavior. Default is now "
                              "the shared kernel, which IS what Demo/Real run.")
    args = parser.parse_args()

    if args.days is not None:
        BARS_PER_DAY = {"M15": 96, "M30": 48, "H1": 24, "H4": 6, "D1": 1}
        args.bars = args.days * BARS_PER_DAY.get(args.tf, 24)

    if not args.legacy_engine:
        # Default path (execution-parity audit fix): run through the SAME
        # engine `main.py --mode backtest` uses, which is the same decision
        # core (AnalysisAgent -> DecisionAgent -> RiskEngine -> PositionSizer)
        # Demo/Real run. See run_pairs()'s docstring.
        run_pairs(args)
        return

    log.warning("[run_backtest] --legacy-engine requested: running the OLD "
                "UnifiedSignalEngine-only loop (no risk engine, no confidence/"
                "news/session gates). Results from this mode do NOT predict "
                "Demo/Real behavior — see execution-parity audit, Critical items 1-3.")
    pairs = [p.strip().upper() for p in args.pairs.split(",")] if args.pairs else [args.pair.upper()]
    all_results = []
    for pair in pairs:
        print(f"\n{'='*60}\n  Backtesting {pair} {args.tf} | {args.bars} bars\n{'='*60}\n")
        if args.synthetic: df = generate_synthetic_data(pair, bars=args.bars)
        else:
            try:
                import MetaTrader5 as mt5
                if not mt5.initialize(): print("ERROR: MT5 not available. Use --synthetic."); sys.exit(1)
                tf_map = {"M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
                rates = mt5.copy_rates_from_pos(pair, tf_map[args.tf], 0, args.bars)
                mt5.shutdown()
                if rates is None or len(rates) == 0: print(f"No data for {pair}"); continue
                df = pd.DataFrame(rates); df["time"] = pd.to_datetime(df["time"], unit="s"); df.set_index("time", inplace=True)
                df.rename(columns={"tick_volume": "volume"}, inplace=True)
                try:
                    from data.indicators import Indicators; ind = Indicators(); df = ind.add_all(df)
                except Exception:
                    import numpy as _np
                    _delta = df["close"].diff()
                    _gain = _delta.clip(lower=0)
                    _loss = (-_delta.clip(upper=0))
                    _avg_gain = _gain.ewm(alpha=1/14, adjust=False).mean()
                    _avg_loss = _loss.ewm(alpha=1/14, adjust=False).mean()
                    _rs = _avg_gain / _avg_loss
                    df["rsi"] = (100 - 100 / (1 + _rs)).fillna(50).clip(0, 100)
                    _tr = pd.concat([
                        df["high"] - df["low"],
                        (df["high"] - df["close"].shift(1)).abs(),
                        (df["low"]  - df["close"].shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    df["atr"] = _tr.ewm(alpha=1/14, adjust=False).mean()
                    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
                    df["ema_9"]  = df["close"].ewm(span=9,  adjust=False).mean()
                    df["sma_50"]  = df["close"].rolling(50,  min_periods=1).mean()
                    df["sma_200"] = df["close"].rolling(200, min_periods=1).mean()
                    _high, _low, _close = df["high"], df["low"], df["close"]
                    _dm_plus  = _high.diff()
                    _dm_minus = -_low.diff()
                    _dm_plus  = _dm_plus.where((_dm_plus > _dm_minus) & (_dm_plus > 0), 0)
                    _dm_minus = _dm_minus.where((_dm_minus > _dm_plus) & (_dm_minus > 0), 0)
                    _atr_s    = _tr.ewm(alpha=1/14, adjust=False).mean()
                    _dmp_s    = _dm_plus.ewm(alpha=1/14, adjust=False).mean()
                    _dmm_s    = _dm_minus.ewm(alpha=1/14, adjust=False).mean()
                    _di_plus  = 100 * _dmp_s / _atr_s.replace(0, _np.nan)
                    _di_minus = 100 * _dmm_s / _atr_s.replace(0, _np.nan)
                    _dx = 100 * (_di_plus - _di_minus).abs() / (_di_plus + _di_minus).replace(0, _np.nan)
                    df["adx"]      = _dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)
                    df["di_plus"]  = _di_plus.fillna(0)
                    df["di_minus"] = _di_minus.fillna(0)
                    log.warning("[run_backtest] data.indicators.Indicators unavailable — using inline Wilder's RSI/ATR/ADX fallback (correct formulas).")
            except ImportError: print("MT5 not installed. Use --synthetic."); sys.exit(1)
        result = run_backtest(symbol=pair, df=df, timeframe=args.tf, starting_balance=args.balance, risk_pct=args.risk, max_open_trades=args.max_trades, max_hold_bars=args.max_hold, spread_pips=args.spread, commission_per_lot=args.commission, slippage_pips=args.slippage, verbose=args.verbose)
        all_results.append(result)
        if result["metrics"]: print(result["metrics"].to_table())
        if result["trades"]:
            trades_df = pd.DataFrame([t.to_dict() for t in result["trades"]])
            csv_path = f"backtest/results_{pair}_{args.tf}.csv"; trades_df.to_csv(csv_path, index=False)
            print(f"\n  Trades saved to: {csv_path}")
            if len(result["trades"]) >= 10:
                print(f"\n{'='*55}")
                print(f"  STATISTICAL VALIDATION")
                print(f"{'='*55}")
                try:
                    from backtest.statistical_validation import run_full_validation
                    returns = [t.pnl_usd for t in result["trades"]]
                    validation = run_full_validation(returns)
                    print(validation.to_table())
                except ImportError:
                    print("  (statistical_validation module not available)")
                except Exception as e:
                    print(f"  Validation error: {e}")
                print(f"\n{'='*55}")
                print(f"  WALK-FORWARD ANALYSIS")
                print(f"{'='*55}")
                try:
                    from backtest.walk_forward import run_walk_forward, print_walk_forward_table
                    wf_result = run_walk_forward(result["trades"], n_windows=5)
                    print_walk_forward_table(wf_result)
                except ImportError:
                    print("  (walk_forward module not available)")
                except Exception as e:
                    print(f"  Walk-forward error: {e}")
    if len(all_results) > 1:
        print(f"\n{'='*60}\n  MULTI-PAIR SUMMARY\n{'='*60}")
        for r in all_results:
            m = r["metrics"]
            if m: print(f"  {r['symbol']:10s} : {m.total_trades:3d} trades | WR={m.win_rate:.1f}% | PF={m.profit_factor:.2f} | P&L=${m.total_pnl_usd:.2f}")
        print("="*60)
    if args.json:
        json_output = []
        for r in all_results:
            if r["metrics"]: json_output.append({"symbol": r["symbol"], "timeframe": r["timeframe"], "bars": r["bars"], "metrics": r["metrics"].to_dict(), "trades": [t.to_dict() for t in r["trades"]]})
        print(json.dumps(json_output, indent=2, default=str))

if __name__ == "__main__": main()