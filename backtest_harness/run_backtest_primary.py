"""
Backtest of the ACTUAL PRIMARY rule engine: strategy/signal_engine.py's
SignalEngine.generate() -- this is the module the code itself labels
"the production signal engine used by core/runtime.py". It is a scoring
system over indicator/regime/pattern/MTF context dicts, self-contained
(no imports), unmodified here.

What's REAL, unmodified production code in this harness:
  - strategy/signal_engine.py -> SignalEngine.generate()  (the decision logic)
  - analysis/market_regime.py -> MarketRegimeDetector       (regime ctx)
  - analysis/support_resistance.py -> SupportResistance     (sr ctx)
  - analysis/advanced_patterns.py -> AdvancedPatternDetector (advanced_pat_ctx)
  - risk/atr_risk_manager.py -> get_stop_loss/get_take_profit (SL/TP, 2x/3x ATR,
    the file's own defaults)

What's a NECESSARY APPROXIMATION (data.indicators.Indicators and
data.fetcher-backed modules were not uploaded -- they need a live broker
connection, not just OHLCV):
  - ind_ctx's raw indicator VALUES (EMA21/50/200, RSI14, MACD 12/26/9,
    Stoch 14/3/3, ADX14, ATR14) are computed here with standard, textbook
    formulas -- not proprietary logic, just the numbers SignalEngine
    expects as input. The categorical labels it reads (trend, rsi_signal,
    macd_cross) are bucketed from those values using conventional
    thresholds (documented inline).
  - mtf_bias is approximated from an H4 resample of the same data
    (EMA50 vs EMA200 alignment) since the real MTF fetch needs a live
    multi-timeframe data source.
  - pat_ctx={} and fib_ctx={} -- these match REAL production behavior:
    the code's own comments say both were already disabled after a
    win-rate audit (patterns <40%, fib 35.9%).
  - extended_ctx (17-module bonus layer) is NOT included -- skipped for
    scope, noted to the user.
  - TradePermission.check() (the full stateful risk gate, needs account/
    drawdown/streak state) is NOT run. A simple static confidence floor
    (>=40, matching the patched MIN_CONFIDENCE_PROD seen in
    core/constants.py) is applied instead as an approximation.
"""
import sys
import time
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from strategy.signal_engine import SignalEngine
from analysis.market_regime import MarketRegimeDetector
from analysis.support_resistance import SupportResistance
from analysis.advanced_patterns import AdvancedPatternDetector
from risk.atr_risk_manager import get_stop_loss, get_take_profit

WINDOW = 250          # rolling bars fed to each sub-module
TIMEOUT_BARS = 200
MIN_CONFIDENCE = 40   # approximates patched MIN_CONFIDENCE_PROD (core/constants.py)
SL_ATR_MULT = 2.0
TP_ATR_MULT = 3.0
POINT = 1e-5

USE_REGIME = True
USE_ADV_PATTERNS = True
USE_MTF = True


def load_h1(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime_utc"])
    df = df.set_index("datetime")
    h1 = (
        df.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last",
              "tick_volume": "sum", "spread": "mean"})
        .dropna().reset_index()
    )
    return h1


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema_21"] = ema(d["close"], 21)
    d["ema_50"] = ema(d["close"], 50)
    d["ema_200"] = ema(d["close"], 200)
    # CLAUDE FIX: MarketRegimeDetector._detect_direction() reads sma_50/
    # sma_200 (not ema_50/ema_200) -- these were missing from every prior
    # backtest in this review, silently forcing regime_result['direction']
    # to always be 'NEUTRAL' (the >=2-of-3 vote could only ever get 1 vote
    # from ema_21 alone). Adding them here fixes regime classification for
    # this regime-audit pass and any backtest going forward.
    d["sma_50"] = d["close"].rolling(50).mean()
    d["sma_200"] = d["close"].rolling(200).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))
    d["rsi"] = d["rsi"].fillna(50)

    ema12 = ema(d["close"], 12)
    ema26 = ema(d["close"], 26)
    d["macd"] = ema12 - ema26
    d["macd_signal"] = ema(d["macd"], 9)

    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["stoch_k"] = 100 * (d["close"] - low14) / (high14 - low14).replace(0, np.nan)
    d["stoch_k"] = d["stoch_k"].fillna(50)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean().fillna(50)

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    up_move = d["high"].diff()
    down_move = -d["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=d.index).rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["adx"] = dx.rolling(14).mean().fillna(0)

    d["spread_avg_20"] = d["spread"].rolling(20).mean()
    return d


def classify_trend(row) -> str:
    e21, e50, e200, atr = row["ema_21"], row["ema_50"], row["ema_200"], row["atr"]
    if pd.isna(e200) or pd.isna(atr) or atr == 0:
        return ""
    dist = (e21 - e200) / atr
    if e21 > e50 > e200:
        return "strong_bullish" if dist > 1.5 else "bullish"
    if e21 < e50 < e200:
        return "strong_bearish" if dist < -1.5 else "bearish"
    return "neutral"


def classify_rsi(rsi: float) -> str:
    if rsi >= 70:
        return "overbought"
    if rsi <= 30:
        return "oversold"
    if rsi > 50:
        return "bullish_zone"
    return "bearish_zone"


def classify_macd_cross(prev_row, row) -> str:
    if pd.isna(prev_row["macd"]) or pd.isna(prev_row["macd_signal"]):
        return ""
    if prev_row["macd"] <= prev_row["macd_signal"] and row["macd"] > row["macd_signal"]:
        return "bullish_cross"
    if prev_row["macd"] >= prev_row["macd_signal"] and row["macd"] < row["macd_signal"]:
        return "bearish_cross"
    return ""


def build_ind_ctx(sub: pd.DataFrame) -> dict:
    row = sub.iloc[-1]
    prev = sub.iloc[-2]
    return {
        "price": row["close"],
        "ema_21": row["ema_21"], "ema_50": row["ema_50"], "ema_200": row["ema_200"],
        "adx": row["adx"], "atr": row["atr"],
        "spread_pips": row["spread"] * POINT / POINT,  # already in "points"; treat as pips proxy
        "spread_avg_20": row["spread_avg_20"],
        "trend": classify_trend(row),
        "rsi": row["rsi"], "rsi_signal": classify_rsi(row["rsi"]),
        "macd": row["macd"], "macd_signal": row["macd_signal"],
        "macd_cross": classify_macd_cross(prev, row),
        "stoch_k": row["stoch_k"], "stoch_d": row["stoch_d"],
    }


def build_mtf_bias(h1_full: pd.DataFrame, up_to_idx: int) -> dict:
    """Approximate HTF bias from an H4 resample of data up to the current bar."""
    sub = h1_full.iloc[:up_to_idx].copy()
    if len(sub) < 220 * 4:
        return {"bias": "NEUTRAL", "confidence": "LOW"}
    sub = sub.set_index("datetime")
    h4 = sub.resample("4h").agg({"close": "last"}).dropna()
    if len(h4) < 220:
        return {"bias": "NEUTRAL", "confidence": "LOW"}
    e50 = ema(h4["close"], 50).iloc[-1]
    e200 = ema(h4["close"], 200).iloc[-1]
    price = h4["close"].iloc[-1]
    if price > e50 > e200:
        return {"bias": "BULLISH", "confidence": "HIGH"}
    if price < e50 < e200:
        return {"bias": "BEARISH", "confidence": "HIGH"}
    return {"bias": "NEUTRAL", "confidence": "LOW"}


def run_symbol(symbol, csv_path, out_path, time_budget_sec=220.0, ckpt_path=None, limit=None):
    h1_raw = load_h1(csv_path)
    if limit:
        h1_raw = h1_raw.iloc[:limit].reset_index(drop=True)
    h1 = compute_indicators(h1_raw)
    n = len(h1)

    sig_engine = SignalEngine()
    regime_det = MarketRegimeDetector()
    sr_engine = SupportResistance()
    adv_engine = AdvancedPatternDetector()

    ckpt_path = ckpt_path or (out_path + ".ckpt.pkl")
    if Path(ckpt_path).exists():
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        i = state["i"]; trades = state["trades"]
        open_trade = state["open_trade"]
        print(f"[{symbol}] RESUMED at bar {i-WINDOW}/{n-WINDOW}, {len(trades)} trades", flush=True)
    else:
        i = WINDOW
        trades = []
        open_trade = None

    t_start = time.time()
    log_every = 500

    while i < n - 1:
        if time.time() - t_start > time_budget_sec:
            break
        bar = h1.iloc[i]

        if open_trade is not None:
            side = open_trade["side"]; sl, tp = open_trade["sl"], open_trade["tp"]
            hit_sl = (bar["low"] <= sl) if side == "BUY" else (bar["high"] >= sl)
            hit_tp = (bar["high"] >= tp) if side == "BUY" else (bar["low"] <= tp)
            bars_held = i - open_trade["entry_idx"]
            outcome, exit_price = None, None
            if hit_sl:
                outcome, exit_price = "LOSS", sl
            elif hit_tp:
                outcome, exit_price = "WIN", tp
            elif bars_held >= TIMEOUT_BARS:
                outcome, exit_price = "TIMEOUT", bar["close"]
            if outcome is not None:
                r_risk = abs(open_trade["entry"] - sl)
                pnl = (exit_price - open_trade["entry"]) if side == "BUY" else (open_trade["entry"] - exit_price)
                r_mult = pnl / r_risk if r_risk > 0 else 0.0
                trades.append({
                    "symbol": symbol, "entry_time": str(open_trade["entry_time"]),
                    "exit_time": str(bar["datetime"]), "side": side,
                    "confidence": open_trade["confidence"], "signal_type": open_trade["signal_type"],
                    "entry": open_trade["entry"], "sl": sl, "tp": tp,
                    "exit_price": exit_price, "outcome": outcome,
                    "r_multiple": round(r_mult, 3), "bars_held": bars_held,
                })
                open_trade = None
            i += 1
            continue

        sub = h1.iloc[i - WINDOW:i].reset_index(drop=True)
        if sub["atr"].iloc[-1] != sub["atr"].iloc[-1] or sub["ema_200"].iloc[-1] != sub["ema_200"].iloc[-1]:
            i += 1
            continue  # NaN warmup

        try:
            ind_ctx = build_ind_ctx(sub)
            regime_result = regime_det.detect(sub) if USE_REGIME else None
            sr_result = sr_engine.analyze(sub, symbol=symbol)
            sr_ctx = sr_engine.get_ai_context(sr_result)
            advanced_pat_ctx = (
                adv_engine.get_ai_context(sub, ind_ctx=ind_ctx, sr_ctx=sr_ctx, regime_ctx=regime_result)
                if USE_ADV_PATTERNS else None
            )
            mtf_bias = build_mtf_bias(h1_raw, i) if USE_MTF else None

            result = sig_engine.generate(
                ind_ctx=ind_ctx, pat_ctx={}, sr_ctx=sr_ctx, regime=regime_result,
                mtf_bias=mtf_bias, advanced_pat_ctx=advanced_pat_ctx, fib_ctx={},
                extended_ctx=None,
            )
        except Exception:
            i += 1
            continue

        signal = result.get("signal", "WAIT")
        confidence = result.get("confidence", 0)
        if signal in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL") and confidence >= MIN_CONFIDENCE:
            side = "BUY" if "BUY" in signal else "SELL"
            fill_bar = h1.iloc[i + 1]
            spread_price = float(fill_bar["spread"]) * POINT if not pd.isna(fill_bar["spread"]) else 0.0
            raw_entry = fill_bar["open"]
            entry = raw_entry + spread_price if side == "BUY" else raw_entry - spread_price
            atr_val = float(bar["atr"])
            sl = get_stop_loss(side, entry, atr_val, SL_ATR_MULT)
            tp = get_take_profit(side, entry, atr_val, TP_ATR_MULT)
            valid = (side == "BUY" and sl < entry < tp) or (side == "SELL" and tp < entry < sl)
            if valid:
                open_trade = {
                    "side": side, "entry": entry, "sl": sl, "tp": tp,
                    "entry_idx": i + 1, "entry_time": fill_bar["datetime"],
                    "confidence": confidence, "signal_type": signal,
                }
        i += 1

        if (i - WINDOW) % log_every == 0:
            elapsed = time.time() - t_start
            print(f"[{symbol}] {i-WINDOW}/{n-WINDOW} bars | {len(trades)} trades | elapsed={elapsed:.0f}s", flush=True)

    if i >= n - 1:
        out_df = pd.DataFrame(trades)
        out_df.to_csv(out_path, index=False)
        if Path(ckpt_path).exists():
            Path(ckpt_path).unlink()
        print(f"[{symbol}] DONE. {len(trades)} trades -> {out_path}", flush=True)
        return out_df
    else:
        with open(ckpt_path, "wb") as f:
            pickle.dump({"i": i, "trades": trades, "open_trade": open_trade}, f)
        print(f"[{symbol}] PAUSED at {i-WINDOW}/{n-WINDOW} ({100*(i-WINDOW)/(n-WINDOW):.1f}%) | "
              f"{len(trades)} trades so far", flush=True)
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time_budget", type=float, default=220.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_regime", action="store_true")
    ap.add_argument("--no_adv_patterns", action="store_true")
    ap.add_argument("--no_mtf", action="store_true")
    args = ap.parse_args()
    globals()["USE_REGIME"] = not args.no_regime
    globals()["USE_ADV_PATTERNS"] = not args.no_adv_patterns
    globals()["USE_MTF"] = not args.no_mtf
    run_symbol(args.symbol, args.csv, args.out, args.time_budget, limit=args.limit)
