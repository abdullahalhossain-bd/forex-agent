#!/usr/bin/env python3
"""
train_missing_pairs_fast.py — Fast baseline ML models for ALL missing pairs.

Uses the 13-feature legacy pipeline (NOT FeatureEngineer's 161-feature per-bar
loop) for speed. Each pair takes ~0.5s instead of ~4s. The models serve as
BASELINE placeholders so the ensemble stops falling back to rules-only.

When real MT5 data becomes available, retrain with:
    python scripts/train_models_quick.py --pair <PAIR>

These baseline models will be auto-skipped by model_predictor.py's
auto-promote logic when newer 161-feature models are trained.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from utils.logger import get_logger
log = get_logger("train_missing_fast")

# ── Pair profiles: (base_price, pip_size, daily_range_pips) ──
PAIR_PROFILES = {
    "EURUSD": (1.085, 0.0001, 70), "GBPUSD": (1.265, 0.0001, 100),
    "USDJPY": (155.0, 0.01, 80), "USDCHF": (0.885, 0.0001, 60),
    "USDCAD": (1.365, 0.0001, 65), "AUDUSD": (0.655, 0.0001, 70),
    "NZDUSD": (0.595, 0.0001, 70), "EURGBP": (0.855, 0.0001, 60),
    "EURJPY": (168.0, 0.01, 110), "EURCHF": (0.960, 0.0001, 55),
    "EURAUD": (1.660, 0.0001, 100), "EURCAD": (1.480, 0.0001, 90),
    "EURNZD": (1.770, 0.0001, 110), "GBPJPY": (196.0, 0.01, 140),
    "GBPCHF": (1.120, 0.0001, 90), "GBPAUD": (1.930, 0.0001, 120),
    "GBPCAD": (1.725, 0.0001, 100), "GBPNZD": (2.120, 0.0001, 140),
    "AUDJPY": (101.5, 0.01, 100), "AUDCHF": (0.580, 0.0001, 70),
    "AUDCAD": (0.895, 0.0001, 75), "AUDNZD": (1.080, 0.0001, 80),
    "NZDJPY": (92.5, 0.01, 85), "NZDCHF": (0.530, 0.0001, 65),
    "NZDCAD": (0.830, 0.0001, 65), "CADJPY": (113.5, 0.01, 80),
    "CADCHF": (0.650, 0.0001, 55), "CHFJPY": (175.0, 0.01, 100),
    "XAUUSD": (2650.0, 0.01, 2500), "XAGUSD": (31.0, 0.001, 400),
    "XPTUSD": (950.0, 0.01, 1500), "XPDUSD": (1000.0, 0.01, 1200),
    "USOUSD": (72.0, 0.01, 150), "UKOUSD": (75.0, 0.01, 140),
    "BTCUSD": (95000.0, 0.01, 3000), "ETHUSD": (3200.0, 0.01, 300),
    "LTCUSD": (85.0, 0.01, 15), "XRPUSD": (0.55, 0.0001, 10),
    "US30USD": (43000.0, 1.0, 500), "NAS100USD": (19500.0, 0.01, 300),
    "SPX500USD": (5800.0, 0.01, 80), "GER40USD": (18500.0, 0.01, 250),
    "USDTRY": (32.5, 0.001, 300), "USDZAR": (18.0, 0.0001, 200),
    "EURNOK": (11.5, 0.0001, 80), "EURSEK": (11.2, 0.0001, 80),
    "GBPSEK": (13.1, 0.0001, 100), "GBPNOK": (13.5, 0.0001, 90),
    "AUDSGD": (0.880, 0.0001, 60), "NZDSGD": (0.800, 0.0001, 55),
    "CADHKD": (5.75, 0.0001, 30), "SGDJPY": (118.0, 0.01, 70),
    "HKDJPY": (19.80, 0.001, 30), "MXNJPY": (7.80, 0.001, 60),
    "USDCNH": (7.25, 0.0001, 50), "USDHKD": (7.80, 0.0001, 10),
    "USDSGD": (1.35, 0.0001, 40), "USDMXN": (17.50, 0.0001, 150),
    "USDTHB": (35.0, 0.01, 40), "USDSAR": (3.75, 0.0001, 5),
    "USDAED": (3.67, 0.0001, 3),
}


def generate_ohlcv(symbol: str, bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate pair-specific synthetic OHLCV data."""
    profile = PAIR_PROFILES.get(symbol, (1.0, 0.0001, 70))
    base_price, pip, daily_range_pips = profile
    np.random.seed(seed + hash(symbol) % 10000)
    dates = pd.date_range("2023-06-01", periods=bars, freq="15min")

    avg_bar_range = (daily_range_pips / 96) * pip
    bar_range = avg_bar_range * (0.5 + np.random.rand(bars))

    close = np.full(bars, base_price)
    for i in range(1, bars):
        drift = np.random.randn() * bar_range[i]
        mean_revert = (base_price - close[i - 1]) * 0.002
        close[i] = close[i - 1] + drift + mean_revert

    intrabar = np.abs(np.random.randn(bars)) * bar_range
    high = np.maximum(close, np.roll(close, 1)) + intrabar
    low = np.minimum(close, np.roll(close, 1)) - intrabar
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(50, 2000, bars)

    return pd.DataFrame({"open": open_, "high": high, "low": low,
                          "close": close, "volume": volume}, index=dates)


def build_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Fast 13-feature pipeline (vectorized, no per-bar loop)."""
    df = df.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_10"] = df["ret_1"].rolling(10).std()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    return df


def train_pair_fast(symbol: str, timeframe: str = "15m", bars: int = 2000) -> bool:
    """Train baseline XGBoost + RF using fast 13-feature pipeline."""
    from ml.model_store import ModelStore

    store = ModelStore()
    df = generate_ohlcv(symbol, bars=bars)

    # Label: 1 if close goes up in next 5 bars
    df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)
    df = build_features_fast(df)
    df = df.dropna(subset=["target"] + FEATURE_COLS)

    if len(df) < 100:
        log.warning(f"  {symbol}: only {len(df)} rows, skipping")
        return False

    feature_cols = FEATURE_COLS
    X = df[feature_cols].values
    y = df["target"].values
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results = {}

    # XGBoost
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                   random_state=42, use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        ver = store.save_model(model=model, pair=symbol, timeframe=timeframe,
                                model_type="xgboost",
                                metrics={"accuracy": float(acc), "training_bars": len(df),
                                         "baseline": True, "feature_pipeline": "fast_13"},
                                is_keras=False, feature_names=feature_cols)
        results["xgboost"] = f"{ver} acc={acc:.1%}"
    except Exception as e:
        log.warning(f"  {symbol} xgboost failed: {e}")

    # RandomForest
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))
        ver = store.save_model(model=rf, pair=symbol, timeframe=timeframe,
                                model_type="random_forest",
                                metrics={"accuracy": float(acc), "training_bars": len(df),
                                         "baseline": True, "feature_pipeline": "fast_13"},
                                is_keras=False, feature_names=feature_cols)
        results["random_forest"] = f"{ver} acc={acc:.1%}"
    except Exception as e:
        log.warning(f"  {symbol} rf failed: {e}")

    if results:
        log.info(f"  {symbol}: {', '.join(f'{k}={v}' for k, v in results.items())}")
        return True
    return False


# 13 feature column names (must match build_features_fast output)
FEATURE_COLS = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "vol_5", "vol_10", "rsi_14",
    "sma_10", "sma_20", "sma_50",
    "atr_14", "macd", "macd_signal",
]


def get_missing_pairs() -> list:
    """Find configured pairs without trained models."""
    from ml.model_store import REGISTRY_PATH
    try:
        from config import SYMBOLS
        configured = [s.upper() for s in SYMBOLS]
    except ImportError:
        return []

    if not REGISTRY_PATH.exists():
        return configured

    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        models_dict = registry.get("models", {})
        missing = []
        for pair in configured:
            has_model = any(k.startswith(f"{pair}_15m_") for k in models_dict)
            if not has_model:
                missing.append(pair)
        return missing
    except Exception:
        return configured


def main():
    parser = argparse.ArgumentParser(description="Fast baseline training for missing pairs")
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bars", type=int, default=2000)
    args = parser.parse_args()

    if args.pair:
        pairs = [args.pair.upper()]
    elif args.force:
        try:
            from config import SYMBOLS
            pairs = [s.upper() for s in SYMBOLS]
        except ImportError:
            pairs = list(PAIR_PROFILES.keys())
    else:
        pairs = get_missing_pairs()

    log.info(f"Training baseline models for {len(pairs)} pairs ({args.bars} bars each, fast 13-feature pipeline)")

    t0 = time.time()
    success = failed = 0

    for i, pair in enumerate(pairs, 1):
        try:
            if train_pair_fast(pair, bars=args.bars):
                success += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            log.error(f"  {pair} FAILED: {e}")

    log.info(f"\nDone in {time.time()-t0:.1f}s | Success: {success} | Failed: {failed}")
    log.info("NOTE: These are BASELINE models (13-feature, synthetic data).")
    log.info("Retrain with real MT5 data for production: python scripts/train_models_quick.py")


if __name__ == "__main__":
    main()