#!/usr/bin/env python3
"""
train_missing_pairs.py — Train baseline ML models for ALL missing currency pairs.

PROBLEM: Only 6 of 62 configured pairs have trained ML models. The other 56
pairs get NOT_READY from the predictor, meaning the ensemble has no ML
participation for them — only rules + LLM vote. This significantly reduces
ensemble confidence and agreement scores.

SOLUTION: This script trains baseline XGBoost + RandomForest models for
every pair that doesn't already have one, using REAL MT5 historical data
by default (via ml/mt5_data_loader.py). Synthetic OHLCV data is available
only as an explicit --synthetic opt-in for offline/no-MT5 testing — it is
never used automatically, including when this script is invoked by
core/runtime.py's auto_retrain repair path at boot.

The models are still "baseline" in the sense that they use default
hyperparameters/feature selection rather than a full tuning pass, but they
ARE trained on real market data unless --synthetic is passed explicitly.

USAGE:
    # Train all missing pairs (real MT5 data):
    python scripts/train_missing_pairs.py

    # Train specific missing pair:
    python scripts/train_missing_pairs.py --pair AUDJPY

    # Force retrain even if model exists:
    python scripts/train_missing_pairs.py --force

    # Offline/no-MT5 debug only — explicit opt-in, not for production:
    python scripts/train_missing_pairs.py --synthetic

MODELS SAVED TO:
    memory/ml_models/{PAIR}_15m/xgboost_vN.pkl
    memory/ml_models/{PAIR}_15m/random_forest_vN.pkl
    memory/ml_models/_registry.json (updated)
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from utils.logger import get_logger
log = get_logger("train_missing_pairs")

# ── Pair-specific price characteristics ──
# These make the synthetic data more realistic per-pair type.
# Format: (typical_price, pip_size, typical_daily_range_pips)
PAIR_PROFILES = {
    # Majors — tight spreads, moderate volatility
    "EURUSD": (1.0850, 0.0001, 70),
    "GBPUSD": (1.2650, 0.0001, 100),
    "USDJPY": (155.00, 0.01, 80),
    "USDCHF": (0.8850, 0.0001, 60),
    "USDCAD": (1.3650, 0.0001, 65),
    "AUDUSD": (0.6550, 0.0001, 70),
    "NZDUSD": (0.5950, 0.0001, 70),
    # EUR crosses — moderate-high volatility
    "EURGBP": (0.8550, 0.0001, 60),
    "EURJPY": (168.00, 0.01, 110),
    "EURCHF": (0.9600, 0.0001, 55),
    "EURAUD": (1.6600, 0.0001, 100),
    "EURCAD": (1.4800, 0.0001, 90),
    "EURNZD": (1.7700, 0.0001, 110),
    # GBP crosses — higher volatility
    "GBPJPY": (196.00, 0.01, 140),
    "GBPCHF": (1.1200, 0.0001, 90),
    "GBPAUD": (1.9300, 0.0001, 120),
    "GBPCAD": (1.7250, 0.0001, 100),
    "GBPNZD": (2.1200, 0.0001, 140),
    # AUD crosses
    "AUDJPY": (101.50, 0.01, 100),
    "AUDCHF": (0.5800, 0.0001, 70),
    "AUDCAD": (0.8950, 0.0001, 75),
    "AUDNZD": (1.0800, 0.0001, 80),
    # NZD crosses
    "NZDJPY": (92.50, 0.01, 85),
    "NZDCHF": (0.5300, 0.0001, 65),
    "NZDCAD": (0.8300, 0.0001, 65),
    # CAD/CHF/JPY crosses
    "CADJPY": (113.50, 0.01, 80),
    "CADCHF": (0.6500, 0.0001, 55),
    "CHFJPY": (175.00, 0.01, 100),
    # Metals — high volatility
    "XAUUSD": (2650.0, 0.01, 2500),
    "XAGUSD": (31.00, 0.001, 400),
    "XPTUSD": (950.0, 0.01, 1500),
    "XPDUSD": (1000.0, 0.01, 1200),
    # Energy
    "USOUSD": (72.00, 0.01, 150),
    "UKOUSD": (75.00, 0.01, 140),
    # Crypto — very high volatility
    "BTCUSD": (95000.0, 0.01, 3000),
    "ETHUSD": (3200.0, 0.01, 300),
    "LTCUSD": (85.00, 0.01, 15),
    "XRPUSD": (0.55, 0.0001, 10),
    # Indices
    "US30USD": (43000.0, 1.0, 500),
    "NAS100USD": (19500.0, 0.01, 300),
    "SPX500USD": (5800.0, 0.01, 80),
    "GER40USD": (18500.0, 0.01, 250),
    # Exotic
    "USDTRY": (32.50, 0.001, 300),
    "USDZAR": (18.00, 0.0001, 200),
    # Additional crosses
    "EURNOK": (11.50, 0.0001, 80),
    "EURSEK": (11.20, 0.0001, 80),
    "GBPSEK": (13.10, 0.0001, 100),
    "GBPNOK": (13.50, 0.0001, 90),
    "AUDSGD": (0.8800, 0.0001, 60),
    "NZDSGD": (0.8000, 0.0001, 55),
    "CADHKD": (5.75, 0.0001, 30),
    "SGDJPY": (118.00, 0.01, 70),
    "HKDJPY": (19.80, 0.001, 30),
    "MXNJPY": (7.80, 0.001, 60),
    # Asia Pacific
    "USDCNH": (7.25, 0.0001, 50),
    "USDHKD": (7.80, 0.0001, 10),
    "USDSGD": (1.35, 0.0001, 40),
    "USDMXN": (17.50, 0.0001, 150),
    "USDTHB": (35.00, 0.01, 40),
    "USDSAR": (3.75, 0.0001, 5),
    "USDAED": (3.67, 0.0001, 3),
}


def generate_realistic_ohlcv(symbol: str, bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate pair-specific synthetic OHLCV data.

    Uses PAIR_PROFILES to set realistic price levels and volatility
    for each instrument type (majors, crosses, metals, crypto, etc.).
    """
    profile = PAIR_PROFILES.get(symbol, (1.0, 0.0001, 70))
    base_price, pip, daily_range_pips = profile

    np.random.seed(seed + hash(symbol) % 10000)
    dates = pd.date_range("2023-06-01", periods=bars, freq="15min")

    # Simulate intraday volatility cycle (Asian/London/NY sessions)
    hour_of_day = dates.hour + dates.minute / 60.0
    # Higher volatility during London (8-16 UTC) and NY (13-21 UTC) overlap
    session_vol = 0.5 + 0.3 * np.exp(-((hour_of_day - 14) ** 2) / 18) + 0.2 * np.exp(-((hour_of_day - 2) ** 2) / 8)

    # Per-bar pip range scaled by session
    avg_bar_range_pips = daily_range_pips / 96  # 96 fifteen-minute bars per day
    bar_range = avg_bar_range_pips * session_vol * pip

    # Random walk with mean reversion (prevents price from drifting to 0 or infinity)
    close = np.full(bars, base_price)
    for i in range(1, bars):
        drift = np.random.randn() * bar_range[i]
        mean_revert = (base_price - close[i - 1]) * 0.001  # gentle pull toward base
        close[i] = close[i - 1] + drift + mean_revert

    # Build OHLC from close
    intrabar_range = np.abs(np.random.randn(bars)) * bar_range
    high = close + intrabar_range
    low = close - intrabar_range
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = np.random.randint(50, 2000, bars)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = "time"
    return df


def get_missing_pairs() -> list:
    """Find pairs from config.SYMBOLS that don't have trained models."""
    try:
        from ml.model_store import ModelStore, REGISTRY_PATH
    except ImportError:
        log.error("Cannot import ModelStore — check project structure")
        return []

    try:
        from config import SYMBOLS
        configured = [s.upper() for s in SYMBOLS]
    except ImportError:
        log.error("Cannot import config.SYMBOLS")
        return []

    # Check which pairs have at least one model registered
    store = ModelStore()
    missing = []

    if REGISTRY_PATH.exists():
        try:
            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            models_dict = registry.get("models", {})

            for pair in configured:
                has_model = False
                for key in models_dict:
                    # Check for any model type (xgboost, random_forest, lstm)
                    if key.startswith(f"{pair}_15m_"):
                        has_model = True
                        break
                if not has_model:
                    missing.append(pair)
        except Exception as e:
            log.warning(f"Error reading registry: {e}")
            missing = configured  # assume all missing
    else:
        missing = configured

    return missing


def train_one_pair(
    symbol: str,
    timeframe: str = "15m",
    bars: int = 100000,
    use_synthetic: bool = False,
) -> bool:
    """Train baseline XGBoost + RandomForest for one pair.

    REAL MT5 DATA BY DEFAULT (use_synthetic=False). This is what
    core/runtime.py calls automatically at boot when
    ML_MODEL_CONSISTENCY_ACTION=auto_retrain finds a registry entry whose
    .pkl file is missing on disk. Previously this unconditionally trained
    on synthetic OHLCV data (use_synthetic=True, hardcoded) — every
    auto-repair silently produced a fake-data model with no indication it
    wasn't trained on real market data. Synthetic is now an explicit
    opt-in for offline/no-MT5 testing only, never an automatic production
    fallback.
    """
    from scripts.train_models_quick import train_one_pair as _train

    if use_synthetic:
        log.warning(
            f"SYNTHETIC data explicitly requested for {symbol} {timeframe} "
            f"(--synthetic). Do not use this model in production."
        )
        return _train(symbol, timeframe, bars=bars, use_synthetic=True)

    log.info(f"Training {symbol} {timeframe} on REAL MT5 data...")
    ok = _train(symbol, timeframe, bars=bars, use_synthetic=False)
    if not ok:
        log.error(
            f"Real MT5 training FAILED for {symbol} {timeframe} (MT5 "
            f"unavailable or fetch error). Refusing to silently fall back "
            f"to synthetic data — fix the MT5 connection and retry, or "
            f"re-run with --synthetic if you deliberately want a "
            f"placeholder model."
        )
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline ML models for all missing currency pairs"
    )
    parser.add_argument("--pair", type=str, default=None,
                        help="Train only this pair (e.g. AUDJPY)")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if model already exists")
    parser.add_argument("--bars", type=int, default=100000,
                        help="Bars per pair to fetch/generate (default: 100000)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic OHLCV data instead of real MT5 "
                             "data (DEBUG/offline testing ONLY — not for "
                             "production, and never the automatic default)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  BASELINE MODEL TRAINING — Missing Pairs")
    log.info("=" * 60)
    log.info(f"  Mode : {'SYNTHETIC data (debug only)' if args.synthetic else 'REAL MT5 data'}")
    log.info(f"  Bars : {args.bars} per pair")

    if args.pair:
        pairs = [args.pair.upper()]
    elif args.force:
        # Force mode: train all configured pairs
        try:
            from config import SYMBOLS
            pairs = [s.upper() for s in SYMBOLS]
        except ImportError:
            pairs = list(PAIR_PROFILES.keys())
    else:
        # Default: only train pairs that are missing models
        pairs = get_missing_pairs()

    log.info(f"  Pairs to train: {len(pairs)}")
    if not args.pair and not args.force:
        log.info(f"  (only pairs WITHOUT existing models)")

    if not pairs:
        log.info("  All pairs already have models — nothing to do!")
        return

    log.info(f"\n  Pair list: {', '.join(pairs)}")
    log.info("")

    total_start = time.time()
    success = 0
    failed = 0
    errors = {}

    for i, pair in enumerate(pairs, 1):
        log.info(f"\n[{i}/{len(pairs)}] {pair}...")
        t0 = time.time()
        try:
            if train_one_pair(pair, bars=args.bars, use_synthetic=args.synthetic):
                success += 1
                log.info(f"  Done in {time.time() - t0:.1f}s")
            else:
                failed += 1
                errors[pair] = "train_one_pair returned False"
        except Exception as e:
            failed += 1
            errors[pair] = str(e)
            log.error(f"  FAILED: {e}")

    elapsed = time.time() - total_start

    # Summary
    log.info(f"\n{'=' * 60}")
    log.info(f"  BASELINE TRAINING COMPLETE")
    log.info(f"{'=' * 60}")
    log.info(f"  Total time  : {elapsed:.1f}s")
    log.info(f"  Success     : {success}/{len(pairs)} pairs")
    log.info(f"  Failed      : {failed}/{len(pairs)} pairs")

    if errors:
        log.info(f"\n  Failed pairs:")
        for pair, err in errors.items():
            log.info(f"    {pair}: {err[:80]}")

    log.info(f"\n  NEXT STEPS:")
    log.info(f"  1. Restart the bot — NOT_READY warnings should be gone")
    if args.synthetic:
        log.info(f"  2. These were SYNTHETIC placeholder models — retrain with")
        log.info(f"     real MT5 data before trading: python scripts/train_missing_pairs.py --force")
    else:
        log.info(f"  2. Models were trained on real MT5 data — ready to use.")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()