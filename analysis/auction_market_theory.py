"""
analysis/auction_market_theory.py — Auction Market Theory (AMT)
=================================================================
Deep Research Domain #2: Auction Market Theory

CONCEPT:
AMT views markets as continuous auctions where price advertises
opportunity and time regulates all opportunities. Key concepts:

- Initial Balance (IB): first hour's high-low range
- Value Area (VA): the range where 70% of trading occurred
- Value Area Rotation: price migrating from one VA to another
- Opening Auction: the initial price discovery phase
- Acceptance vs Rejection: price staying vs returning
- Single Prints: low-volume price levels (fast moves)
- Excess High/Low: tails beyond the value area

USAGE:
    from analysis.auction_market_theory import analyze_auction_market
    result = analyze_auction_market(df, session_start_hour=7)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict
from utils.logger import get_logger

log = get_logger("amt")


def analyze_auction_market(
    df: pd.DataFrame,
    session_start_hour: int = 7,  # London open UTC
    ib_duration_hours: int = 1,   # Initial Balance = first hour
    value_area_pct: float = 0.70, # 70% value area
) -> dict:
    """Analyze market using Auction Market Theory.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        session_start_hour: UTC hour when session opens (7=London, 12=NY).
        ib_duration_hours: Initial Balance duration in hours.
        value_area_pct: Percentage of volume for value area (default 70%).

    Returns:
        {
            "initial_balance": {"high": float, "low": float, "range": float},
            "value_area": {"high": float, "low": float, "poc": float},
            "price_location": str,  # ABOVE_VA / INSIDE_VA / BELOW_VA / IB_HIGH / IB_LOW
            "acceptance": bool,     # price accepted in current VA
            "single_prints": list,  # price levels with minimal volume
            "excess": {"high": float, "low": float},
            "signal": str,          # BUY / SELL / NEUTRAL
            "score": int,           # 0-100
            "reason": str,
        }
    """
    if df is None or len(df) < 20:
        return {"signal": "NEUTRAL", "score": 0, "reason": "Insufficient data"}

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Filter to today's session
    if len(df) > 0:
        today = df.index[-1].date()
        session_df = df[df.index.date == today]
    else:
        session_df = df

    if len(session_df) < 5:
        session_df = df.iloc[-50:]  # fallback

    # ── Initial Balance (IB) ──
    ib_bars = session_df.iloc[:max(ib_duration_hours * 4, 4)]  # 4 bars per hour (M15)
    ib_high = float(ib_bars["high"].max())
    ib_low = float(ib_bars["low"].min())
    ib_range = ib_high - ib_low

    # ── Volume Profile (simplified — price-based histogram) ──
    close_prices = session_df["close"].values
    volumes = session_df["volume"].values if "volume" in session_df.columns else np.ones(len(session_df))

    # Build price histogram
    if ib_range > 0:
        n_bins = max(20, int(ib_range / 0.0005))  # ~5 pip bins
        price_bins = np.linspace(ib_low * 0.998, ib_high * 1.002, n_bins)
        hist, bin_edges = np.histogram(close_prices, bins=price_bins, weights=volumes)

        # POC (Point of Control) — price level with highest volume
        poc_idx = np.argmax(hist)
        poc = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

        # Value Area — contain value_area_pct of total volume
        total_vol = hist.sum()
        target_vol = total_vol * value_area_pct

        # Expand from POC outward until we capture 70% of volume
        va_vol = hist[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx

        while va_vol < target_vol and (va_low_idx > 0 or va_high_idx < len(hist) - 1):
            # Expand in the direction with more volume
            expand_down = va_low_idx > 0
            expand_up = va_high_idx < len(hist) - 1

            if expand_down and expand_up:
                if hist[va_low_idx - 1] >= hist[va_high_idx + 1]:
                    va_low_idx -= 1
                    va_vol += hist[va_low_idx]
                else:
                    va_high_idx += 1
                    va_vol += hist[va_high_idx]
            elif expand_down:
                va_low_idx -= 1
                va_vol += hist[va_low_idx]
            elif expand_up:
                va_high_idx += 1
                va_vol += hist[va_high_idx]
            else:
                break

        va_high = float(bin_edges[va_high_idx + 1])
        va_low = float(bin_edges[va_low_idx])
    else:
        poc = float(session_df["close"].iloc[-1])
        va_high = ib_high
        va_low = ib_low

    # ── Current price location ──
    current_price = float(session_df["close"].iloc[-1])
    price_location = "INSIDE_VA"
    if current_price > va_high:
        price_location = "ABOVE_VA"
    elif current_price < va_low:
        price_location = "BELOW_VA"
    elif current_price > ib_high:
        price_location = "ABOVE_IB"
    elif current_price < ib_low:
        price_location = "BELOW_IB"

    # ── Acceptance vs Rejection ──
    # If price has been in current zone for ≥ 3 bars = acceptance
    recent_closes = session_df["close"].iloc[-3:].values
    if price_location == "ABOVE_VA":
        accepted = all(c > va_high for c in recent_closes)
    elif price_location == "BELOW_VA":
        accepted = all(c < va_low for c in recent_closes)
    else:
        accepted = True  # inside VA = always accepted

    # ── Single Prints (low-volume price levels) ──
    single_prints = []
    if ib_range > 0 and len(hist) > 0:
        median_vol = np.median(hist[hist > 0]) if np.any(hist > 0) else 0
        for i in range(len(hist)):
            if hist[i] > 0 and hist[i] < median_vol * 0.2:  # < 20% of median
                single_prints.append(float((bin_edges[i] + bin_edges[i + 1]) / 2))

    # ── Excess (tails beyond value area) ──
    excess_high = float(session_df["high"].max()) - va_high if float(session_df["high"].max()) > va_high else 0.0
    excess_low = va_low - float(session_df["low"].min()) if float(session_df["low"].min()) < va_low else 0.0

    # ── Signal generation ──
    signal = "NEUTRAL"
    score = 30
    reason = "Price inside value area — balanced market"

    if price_location == "ABOVE_VA" and not accepted:
        signal = "SELL"
        score = 65
        reason = "Price rejected above VA — distribution, sell opportunity"
    elif price_location == "BELOW_VA" and not accepted:
        signal = "BUY"
        score = 65
        reason = "Price rejected below VA — accumulation, buy opportunity"
    elif price_location == "ABOVE_VA" and accepted:
        signal = "BUY"
        score = 60
        reason = "Price accepted above VA — value migrating higher, buy"
    elif price_location == "BELOW_VA" and accepted:
        signal = "SELL"
        score = 60
        reason = "Price accepted below VA — value migrating lower, sell"

    log.info(
        f"[AMT] IB=[{ib_low:.5f}-{ib_high:.5f}] VA=[{va_low:.5f}-{va_high:.5f}] "
        f"POC={poc:.5f} price={current_price:.5f} loc={price_location} "
        f"accepted={accepted} → {signal} ({score}/100)"
    )

    return {
        "initial_balance": {"high": ib_high, "low": ib_low, "range": ib_range},
        "value_area": {"high": va_high, "low": va_low, "poc": poc},
        "price_location": price_location,
        "acceptance": accepted,
        "single_prints": single_prints[:5],  # top 5
        "excess": {"high": excess_high, "low": excess_low},
        "signal": signal,
        "score": score,
        "reason": reason,
    }


if __name__ == "__main__":
    np.random.seed(42)
    n = 96  # 1 day of M15
    dates = pd.date_range("2024-01-01 07:00", periods=n, freq="15min")
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.0002)
    df = pd.DataFrame({
        "open": close, "high": close + 0.0003, "low": close - 0.0003,
        "close": close, "volume": np.random.randint(100, 1000, n),
    }, index=dates)

    result = analyze_auction_market(df)
    print(f"AMT: {result['signal']} ({result['score']}/100)")
    print(f"  IB: {result['initial_balance']}")
    print(f"  VA: {result['value_area']}")
    print(f"  Location: {result['price_location']}, Accepted: {result['acceptance']}")
    print(f"  Reason: {result['reason']}")
    print("AMT smoke test passed.")
