#!/usr/bin/env python3
# scripts/setup_market_dna.py
# ============================================================
# Market DNA — one-shot setup / readiness check.
#
# Run this BEFORE scripts/evaluate_market_dna_impact.py. It never
# touches live trading code paths; it only prepares the ground:
#
#   1. Verifies/installs the extra dependency (hdbscan).
#   2. Creates the model directory + DB tables.
#   3. Inventories what data actually exists (candles, closed
#      trades) so you know up front whether an impact evaluation
#      will be running on real history or a synthetic fallback.
#   4. If NO candle data exists anywhere, generates a synthetic
#      multi-year OHLCV dataset so the evaluation script always has
#      something to run the pipeline against — clearly labeled as
#      synthetic, never presented as a real-market result.
#
# Usage:
#   python -m scripts.setup_market_dna
#   python -m scripts.setup_market_dna --auto-install
#   python -m scripts.setup_market_dna --force-synthetic
# ============================================================

import argparse
import importlib
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODEL_DIR, DATA_DIR, DEFAULT_TIMEFRAME
from database.db import DB_PATH
from utils.logger import get_logger

log = get_logger(__name__)

SYNTHETIC_SYMBOL = "EURUSD"
SYNTHETIC_TIMEFRAME = "H1"
SYNTHETIC_BARS = 60_000          # ~7 years of H1 bars — enough for the eval script's fold-B/C trade-count floor
SYNTHETIC_OUT = DATA_DIR / "history" / SYNTHETIC_SYMBOL / f"{SYNTHETIC_SYMBOL}_{SYNTHETIC_TIMEFRAME}_synthetic.csv"

MIN_TRADES_FOR_ANY_JOURNAL = 100   # matches NO_STATISTICAL_EDGE floor in dna_journal.py


def _ok(msg):
    print(f"  [OK]   {msg}")


def _warn(msg):
    print(f"  [WARN] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")


# ── Step 1: dependencies ─────────────────────────────────────
def check_dependencies(auto_install: bool) -> bool:
    print("\n[1/4] Dependencies")
    required = {
        "hdbscan": "hdbscan>=0.8.33",
        "sklearn": "scikit-learn>=1.3.0",
        "scipy": "scipy>=1.10.0",
        "joblib": "joblib",
    }
    all_present = True
    for mod, pip_spec in required.items():
        try:
            importlib.import_module(mod)
            _ok(f"{mod} importable")
        except ImportError:
            all_present = False
            if auto_install:
                _warn(f"{mod} missing — installing `{pip_spec}` ...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", pip_spec],
                    check=False,
                )
                try:
                    importlib.import_module(mod)
                    _ok(f"{mod} installed successfully")
                except ImportError:
                    _fail(f"{mod} still not importable after install attempt")
            else:
                _fail(f"{mod} missing — run with --auto-install, or: pip install {pip_spec} --break-system-packages")
    return all_present


# ── Step 2: directories + DB schema ─────────────────────────
def setup_storage() -> None:
    print("\n[2/4] Storage (model dir + DB tables)")
    dna_dir = MODEL_DIR / "market_dna"
    dna_dir.mkdir(parents=True, exist_ok=True)
    _ok(f"model dir ready: {dna_dir}")

    from database.market_dna_schema import init_market_dna_tables
    init_market_dna_tables()
    with sqlite3.connect(DB_PATH) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'market_dna_%'"
        )}
    expected = {"market_dna_models", "market_dna_cluster_stats",
                "market_dna_drift_log", "market_dna_trade_assignments"}
    missing = expected - tables
    if missing:
        _fail(f"tables missing after init: {missing}")
    else:
        _ok(f"DB tables present: {sorted(expected)}")


# ── Step 3: data inventory ──────────────────────────────────
def inventory_data() -> dict:
    print("\n[3/4] Data inventory")
    report = {"candle_sources": [], "closed_trades": 0}

    # DB candles table
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT symbol, timeframe, COUNT(*), MIN(time), MAX(time) "
                "FROM candles GROUP BY symbol, timeframe"
            ).fetchall()
        for symbol, tf, n, tmin, tmax in rows:
            report["candle_sources"].append(
                {"source": "db", "symbol": symbol, "timeframe": tf, "rows": n, "range": (tmin, tmax)}
            )
            _ok(f"DB candles: {symbol} {tf} — {n} rows [{tmin} .. {tmax}]")
    except sqlite3.OperationalError as e:
        _warn(f"could not read candles table: {e}")

    # data/history/*/*.parquet (e.g. scripts/create_synthetic_data.py output)
    hist_dir = DATA_DIR / "history"
    if hist_dir.exists():
        for path in list(hist_dir.rglob("*.parquet")) + list(hist_dir.rglob("*.csv")):
            try:
                reader = pd.read_parquet if path.suffix == ".parquet" else pd.read_csv
                n = len(reader(path, columns=["close"]) if path.suffix == ".parquet" else reader(path, usecols=["close"]))
                report["candle_sources"].append({"source": path.suffix.lstrip("."), "path": str(path), "rows": n})
                _ok(f"{path.suffix[1:].upper()}: {path} — {n} rows")
            except Exception as e:
                _warn(f"could not read {path}: {e}")

    if not report["candle_sources"]:
        _warn("no candle data found anywhere (DB candles table empty, no parquet files)")

    # Closed trades
    try:
        with sqlite3.connect(DB_PATH) as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'").fetchone()
        report["closed_trades"] = n
        if n >= MIN_TRADES_FOR_ANY_JOURNAL:
            _ok(f"closed trades: {n} (>= {MIN_TRADES_FOR_ANY_JOURNAL} floor for a non-trivial journal)")
        elif n > 0:
            _warn(f"closed trades: {n} — below the {MIN_TRADES_FOR_ANY_JOURNAL} floor, "
                  f"journal stats will mostly read NO_STATISTICAL_EDGE")
        else:
            _warn("closed trades: 0 — evaluate_market_dna_impact.py will fall back to a "
                  "proxy reference strategy to generate trade-like outcomes")
    except sqlite3.OperationalError as e:
        _warn(f"could not read trades table: {e}")

    return report


# ── Step 4: synthetic fallback ──────────────────────────────
def generate_synthetic_candles(force: bool) -> Path:
    print("\n[4/4] Synthetic fallback data")
    if SYNTHETIC_OUT.exists() and not force:
        _ok(f"already exists: {SYNTHETIC_OUT} (use --force-synthetic to regenerate)")
        return SYNTHETIC_OUT

    SYNTHETIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Regime-switching random walk (not pure GBM) so HDBSCAN actually
    # has more than one density cluster to find — a few volatility/
    # trend regimes stitched together, purely for pipeline smoke
    # testing. NOT a real-market claim of any kind.
    n = SYNTHETIC_BARS
    regimes = rng.integers(0, 4, size=n // 500 + 1)
    price = 1.1000
    closes = []
    vols = []
    for i in range(n):
        regime = regimes[i // 500]
        drift, sigma = [(0.0, 0.0004), (0.00015, 0.0003), (-0.00012, 0.0009), (0.0, 0.00015)][regime]
        price += rng.normal(drift, sigma)
        closes.append(price)
        vols.append(rng.integers(500, 5000))

    closes = np.array(closes)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.0002, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.0002, n))

    end = pd.Timestamp.now(tz="UTC").floor("h")
    times = pd.date_range(end=end, periods=n, freq="h")

    df = pd.DataFrame({
        "time": times, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    })
    df.to_csv(SYNTHETIC_OUT, index=False)
    _ok(f"generated {n} synthetic H1 bars (4 stitched regimes) -> {SYNTHETIC_OUT}")
    return SYNTHETIC_OUT


def main():
    parser = argparse.ArgumentParser(description="Market DNA setup / readiness check")
    parser.add_argument("--auto-install", action="store_true", help="pip install missing deps")
    parser.add_argument("--force-synthetic", action="store_true", help="regenerate synthetic data even if present")
    parser.add_argument("--skip-synthetic", action="store_true", help="don't generate synthetic fallback data")
    args = parser.parse_args()

    print("=" * 60)
    print("MARKET DNA — SETUP")
    print("=" * 60)

    deps_ok = check_dependencies(args.auto_install)
    setup_storage()
    report = inventory_data()

    has_real_candles = any(
        s["rows"] >= 2000 and "synthetic" not in str(s.get("path", "")).lower()
        for s in report["candle_sources"]
    )
    if not has_real_candles and not args.skip_synthetic:
        generate_synthetic_candles(args.force_synthetic)
    elif not args.skip_synthetic:
        print("\n[4/4] Synthetic fallback data")
        _ok("skipped — sufficient real candle data already present")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dependencies ready : {'YES' if deps_ok else 'NO — see [FAIL] lines above'}")
    print(f"  Candle sources     : {len(report['candle_sources'])}")
    print(f"  Closed trades      : {report['closed_trades']}")
    print(f"  Real data usable   : {'YES' if has_real_candles else 'NO (synthetic fallback will be used)'}")
    print("\nNext: python -m scripts.evaluate_market_dna_impact")


if __name__ == "__main__":
    main()
