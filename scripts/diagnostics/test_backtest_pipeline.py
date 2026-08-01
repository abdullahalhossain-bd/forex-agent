#!/usr/bin/env python3
"""
test_backtest_pipeline.py — Backtest Pipeline Diagnostic
=========================================================
Read-only diagnostic that validates the backtest engine:
  historical data -> feature engineering -> optional ML model
  availability -> BacktestEngine.run_strategy() -> confidence
  values per trade/signal -> summary fields that reporting expects.

No trading side effects. Safe to run anytime.

Usage:
    cd /path/to/forex-agent
    python scripts/diagnostics/test_backtest_pipeline.py
"""

import os
import sys
import glob as _glob

# ── Ensure project root is on sys.path ──────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = ""):
    tag = "[PASS]" if passed else "[FAIL]"
    msg = f"  {tag} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, passed, detail))


def main():
    print("=" * 64)
    print("  BACKTEST PIPELINE DIAGNOSTIC")
    print("=" * 64)

    # ── Step 1: Historical data availability ────────────────────
    print("\n── Step 1: Historical Data ──")
    data_dir = os.path.join(_PROJECT_ROOT, "data", "historical")
    csv_files = _glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    if csv_files:
        step("Historical CSV files found", True,
             f"{len(csv_files)} file(s) in {data_dir}/")
        # Show first few
        for f in csv_files[:5]:
            size_kb = os.path.getsize(f) / 1024
            step(f"  {os.path.relpath(f, data_dir)}", True, f"{size_kb:.0f} KB")
    else:
        step("Historical CSV files found", False,
             f"no .csv files in {data_dir}/")

    # ── Step 2: HistoricalDataLoader ────────────────────────────
    print("\n── Step 2: HistoricalDataLoader ──")
    df = None
    loaded_file = None
    try:
        from backtest.data_loader import HistoricalDataLoader
        loader = HistoricalDataLoader()
        step("HistoricalDataLoader import", True)
    except Exception as e:
        step("HistoricalDataLoader import", False, str(e))

    if csv_files:
        # Try to load the first available CSV
        for csv_path in csv_files[:5]:
            try:
                pair_name = os.path.basename(os.path.dirname(csv_path))
                tf_name = os.path.splitext(os.path.basename(csv_path))[0]
                df = loader.load_csv(csv_path, pair=pair_name or "EURUSD",
                                     timeframe=tf_name or "15m")
                if df is not None and not df.empty:
                    loaded_file = csv_path
                    step("Load CSV via HistoricalDataLoader", True,
                         f"{os.path.basename(csv_path)}: {len(df)} rows, "
                         f"columns={list(df.columns)[:10]}...")
                    break
            except Exception as e:
                step(f"Load {os.path.basename(csv_path)}", False, str(e)[:120])
                continue

        if df is None:
            step("Load CSV via HistoricalDataLoader", False,
                 "none of the CSV files loaded successfully")
    else:
        step("Load CSV via HistoricalDataLoader", None,
             "no CSV files to load — skipping")

    # ── Step 3: Feature engineering on historical data ───────────
    print("\n── Step 3: Feature Engineering (Historical) ──")
    if df is not None and not df.empty:
        try:
            from ml.feature_engineer import get_feature_engineer
            fe = get_feature_engineer()

            # Build feature vector from the last row of historical data
            features = fe.build_feature_vector(df, pair="EURUSD", timeframe="15m")
            if features and len(features) > 0:
                step("Feature vector from historical data", True,
                     f"{len(features)} features")
            else:
                step("Feature vector from historical data", False, "empty dict")
        except Exception as e:
            step("Feature vector from historical data", False, str(e)[:120])
    else:
        step("Feature vector from historical data", None, "no data loaded")

    # ── Step 4: ML model availability for backtest ──────────────
    print("\n── Step 4: ML Model Availability (Backtest Context) ──")
    try:
        from ml.model_predictor import get_model_predictor
        predictor = get_model_predictor()
        is_ready = predictor.is_ready("EURUSD", "15m")
        step("ML model available for backtest", is_ready,
             f"is_ready={is_ready}" if is_ready else
             "no trained models — backtest will run rules-only")
    except Exception as e:
        step("ML model available for backtest", False, str(e)[:120])

    # ── Step 5: BacktestEngine instantiation ─────────────────────
    print("\n── Step 5: BacktestEngine ──")
    try:
        from backtest.engine import BacktestEngine
        engine = BacktestEngine(initial_balance=10000.0, risk_per_trade=0.01)
        step("BacktestEngine import + instantiation", True)
    except Exception as e:
        step("BacktestEngine import + instantiation", False, str(e))
        _summary()
        return

    # ── Step 6: Run a minimal backtest ───────────────────────────
    print("\n── Step 6: Minimal Backtest Run ──")
    bt_result = None
    if df is not None and len(df) >= 50:
        # Create a minimal strategy that just returns BUY/SELL/WAIT
        class MinimalStrategy:
            """Trivial strategy: BUY when RSI > 55, SELL when RSI < 45, else WAIT."""
            name = "diagnostic_minimal"

            def generate_signals(self, data):
                signals = []
                for i in range(len(data)):
                    rsi = data.iloc[i].get("rsi_14", 50)
                    if rsi > 55:
                        signals.append({"signal": "BUY", "confidence": 60, "reason": f"RSI={rsi:.1f}"})
                    elif rsi < 45:
                        signals.append({"signal": "SELL", "confidence": 60, "reason": f"RSI={rsi:.1f}"})
                    else:
                        signals.append({"signal": "WAIT", "confidence": 0, "reason": "no edge"})
                return signals

        try:
            # Use a small subset for the diagnostic
            subset = df.tail(min(200, len(df))).copy().reset_index(drop=True)
            bt_result = engine.run_strategy(
                strategy=MinimalStrategy(),
                df=subset,
                pair="EURUSD",
                timeframe="15m",
                save_report=False,      # don't write files
                save_to_memory=False,   # don't pollute DB
                enable_gating=False,    # run without confidence gating
            )
            if bt_result:
                step("BacktestEngine.run_strategy()", True,
                     f"returned result dict")
            else:
                step("BacktestEngine.run_strategy()", False, "returned None")
        except Exception as e:
            step("BacktestEngine.run_strategy()", False, str(e)[:200])
    else:
        step("Minimal backtest run", None,
             "no historical data loaded or too few rows — skipping")

    # ── Step 7: Check result structure and confidence values ─────
    print("\n── Step 7: Result Structure + Confidence Values ──")
    if bt_result:
        # Check summary fields
        summary = bt_result.get("summary", {})
        required_summary_keys = [
            "trades", "wins", "losses", "win_rate", "profit",
            "profit_factor", "max_drawdown", "average_rr",
        ]
        missing = [k for k in required_summary_keys if k not in summary]
        if not missing:
            step("Summary fields present", True,
                 f"trades={summary.get('trades', 0)}, "
                 f"win_rate={summary.get('win_rate', 0):.1%}, "
                 f"profit={summary.get('profit', 0):.2f}, "
                 f"max_drawdown={summary.get('max_drawdown', 0):.2f}, "
                 f"profit_factor={summary.get('profit_factor', 0):.2f}")
        else:
            step("Summary fields present", False,
                 f"missing: {missing}")

        # Check trades DataFrame
        trades_df = bt_result.get("trades")
        if trades_df is not None and hasattr(trades_df, 'shape') and len(trades_df) > 0:
            step("Trades DataFrame", True,
                 f"{len(trades_df)} trades, columns={list(trades_df.columns)[:10]}")

            # Print actual confidence values from trades
            if "confidence" in trades_df.columns:
                conf_values = trades_df["confidence"].dropna()
                if len(conf_values) > 0:
                    step("Confidence values in trades", True,
                         f"count={len(conf_values)}, "
                         f"mean={conf_values.mean():.1f}, "
                         f"min={conf_values.min():.1f}, "
                         f"max={conf_values.max():.1f}")
                    # Show first few
                    for idx, row in trades_df.head(5).iterrows():
                        conf = row.get("confidence", "N/A")
                        sig = row.get("signal", row.get("action", "N/A"))
                        print(f"         trade[{idx}]: signal={sig}, confidence={conf}")
                else:
                    step("Confidence values in trades", False, "all NaN/empty")
            else:
                step("Confidence column in trades", False,
                     f"available columns: {list(trades_df.columns)}")
        elif trades_df is not None and len(trades_df) == 0:
            step("Trades DataFrame", True, "0 trades (strategy produced no signals)")
        else:
            step("Trades DataFrame", False,
                 f"trades key type={type(trades_df).__name__}")

        # Check gating stats (if gating was enabled)
        gating = bt_result.get("gating_stats")
        if gating:
            step("Gating stats", True,
                 f"allowed={gating.get('allowed')}, "
                 f"blocked={gating.get('blocked')}, "
                 f"block_rate={gating.get('block_rate', 0):.1%}")
        elif not gating:
            step("Gating stats", None, "gating was disabled (enable_gating=False)")
    else:
        step("Result structure check", False, "no backtest result — skip")

    # ── Step 8: HonestBacktester (alternative engine) ────────────
    print("\n── Step 8: HonestBacktester (Look-Ahead Free) ──")
    try:
        from backtest.honest_backtest_engine import HonestBacktester
        step("HonestBacktester import", True)
    except ImportError:
        step("HonestBacktester import", False, "module not found")
    except Exception as e:
        step("HonestBacktester import", False, str(e))

    _summary()


def _summary():
    print("\n" + "=" * 64)
    passed = sum(1 for _, p, _ in results if p is True)
    failed = sum(1 for _, p, _ in results if p is False)
    skipped = sum(1 for _, p, _ in results if p is None)
    total = len(results)

    tag = "PASS" if failed == 0 and passed > 0 else "FAIL"
    print(f"  OVERALL: {tag}  ({passed} passed, {failed} failed, {skipped} skipped / {total} total)")

    if failed > 0:
        print("\n  Failed steps:")
        for name, p, detail in results:
            if p is False:
                print(f"    [FAIL] {name}: {detail}")

    print("=" * 64)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()