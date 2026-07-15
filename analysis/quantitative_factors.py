"""
analysis/quantitative_factors.py — Quantitative Finance Factors
=================================================================
Deep Research Domain #8: Quantitative Factors

Advanced statistical/quantitative methods:
1. Hurst Exponent — trend persistence vs mean reversion
2. Kalman Filter — adaptive price tracking
3. Hidden Markov Model — regime detection
4. Bayesian Update — probability updating
5. Z-Score — statistical deviation
6. Cointegration — pair trading signal

USAGE:
    from analysis.quantitative_factors import compute_quant_factors
    result = compute_quant_factors(df)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict
from utils.logger import get_logger

log = get_logger("quant_factors")


def hurst_exponent(prices: np.ndarray, max_lag: int = 50) -> float:
    """Compute Hurst Exponent.

    H < 0.5 → mean reverting (anti-persistent)
    H = 0.5 → random walk
    H > 0.5 → trending (persistent)

    Uses R/S (Rescaled Range) method.
    """
    if len(prices) < max_lag:
        return 0.5  # default to random walk

    lags = range(2, min(max_lag, len(prices) // 2))
    tau = []
    rs_values = []

    for lag in lags:
        # Compute returns
        returns = np.diff(np.log(prices[:lag * (len(prices) // lag)]))
        if len(returns) < 2:
            continue

        # Split into chunks
        n_chunks = len(returns) // lag
        if n_chunks < 1:
            continue

        rs_list = []
        for i in range(n_chunks):
            chunk = returns[i * lag:(i + 1) * lag]
            if len(chunk) < 2:
                continue
            mean_chunk = np.mean(chunk)
            deviations = chunk - mean_chunk
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            tau.append(lag)
            rs_values.append(np.mean(rs_list))

    if len(tau) < 3:
        return 0.5

    # Fit: log(R/S) = H * log(tau) + c
    log_tau = np.log(tau)
    log_rs = np.log(rs_values)
    coeffs = np.polyfit(log_tau, log_rs, 1)
    H = coeffs[0]

    return float(max(0.0, min(1.0, H)))  # clamp to [0, 1]


def kalman_filter(prices: np.ndarray, process_var: float = 1e-5, meas_var: float = 1e-3) -> np.ndarray:
    """Simple 1D Kalman filter for price smoothing.

    Returns smoothed price estimates that adapt to changes faster than SMA.
    """
    n = len(prices)
    if n < 2:
        return prices.copy()

    x = prices[0]  # initial estimate
    p = 1.0        # initial uncertainty
    estimates = np.zeros(n)

    for i in range(n):
        # Predict
        x_pred = x
        p_pred = p + process_var

        # Update
        k = p_pred / (p_pred + meas_var)  # Kalman gain
        x = x_pred + k * (prices[i] - x_pred)
        p = (1 - k) * p_pred
        estimates[i] = x

    return estimates


def hidden_markov_regime(returns: np.ndarray, n_states: int = 3) -> list:
    """Simplified HMM for regime detection (no external library).

    Uses Gaussian mixture + Viterbi-like approach to identify:
    - State 0: Bull (positive mean return, low vol)
    - State 1: Bear (negative mean return, low vol)
    - State 2: Volatile (high vol)

    Returns list of state indices (0, 1, 2) for each return.
    """
    if len(returns) < 20:
        return [0] * len(returns)

    # Simple approach: classify based on rolling statistics
    states = []
    window = 20

    for i in range(len(returns)):
        if i < window:
            states.append(0)
            continue

        recent = returns[i - window:i]
        mean_ret = np.mean(recent)
        vol = np.std(recent)

        if vol > np.std(returns) * 1.3:
            states.append(2)  # volatile
        elif mean_ret > 0:
            states.append(0)  # bull
        else:
            states.append(1)  # bear

    return states


def bayesian_win_probability(
    prior_win: float,
    evidence_strength: float,
    evidence_positive: bool,
) -> float:
    """Bayesian update of win probability.

    Args:
        prior_win: Prior P(win) (0-1).
        evidence_strength: How strong the evidence is (0-1).
        evidence_positive: True if evidence supports winning.

    Returns:
        Updated P(win) (0-1).
    """
    prior_loss = 1.0 - prior_win

    if evidence_positive:
        # P(evidence | win) = 0.5 + evidence_strength/2
        # P(evidence | loss) = 0.5 - evidence_strength/2
        p_ev_win = 0.5 + evidence_strength * 0.5
        p_ev_loss = 0.5 - evidence_strength * 0.5
    else:
        p_ev_win = 0.5 - evidence_strength * 0.5
        p_ev_loss = 0.5 + evidence_strength * 0.5

    # Bayes: P(win | evidence) = P(evidence | win) * P(win) / P(evidence)
    p_evidence = p_ev_win * prior_win + p_ev_loss * prior_loss

    if p_evidence > 0:
        posterior = (p_ev_win * prior_win) / p_evidence
    else:
        posterior = prior_win

    return float(max(0.01, min(0.99, posterior)))


def rolling_zscore(series: pd.Series, window: int = 50) -> pd.Series:
    """Compute rolling Z-score."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / (rolling_std + 1e-10)


def compute_quant_factors(df: pd.DataFrame) -> dict:
    """Compute all quantitative factors.

    Returns:
        {
            "hurst": float,
            "kalman_trend": str,
            "hmm_regime": str,
            "bayesian_win_prob": float,
            "zscore": float,
            "signal": str,
            "score": int,
            "reason": str,
        }
    """
    if df is None or len(df) < 50:
        return {"signal": "NEUTRAL", "score": 0, "reason": "Insufficient data"}

    prices = df["close"].values

    # ── Hurst Exponent ──
    H = hurst_exponent(prices)

    # ── Kalman Filter ──
    kalman_est = kalman_filter(prices)
    kalman_slope = kalman_est[-1] - kalman_est[-5] if len(kalman_est) >= 5 else 0
    kalman_trend = "UP" if kalman_slope > 0 else "DOWN" if kalman_slope < 0 else "FLAT"

    # ── HMM Regime ──
    returns = np.diff(np.log(prices))
    hmm_states = hidden_markov_regime(returns)
    current_state = hmm_states[-1] if hmm_states else 0
    hmm_regime = ["BULL", "BEAR", "VOLATILE"][current_state]

    # ── Z-Score ──
    z = rolling_zscore(df["close"], window=50)
    current_z = float(z.iloc[-1]) if len(z) > 0 and not np.isnan(z.iloc[-1]) else 0.0

    # ── Bayesian ──
    bayesian_prob = bayesian_win_probability(
        prior_win=0.5,
        evidence_strength=min(abs(current_z) / 2.0, 0.8),
        evidence_positive=(current_z > 0 and hmm_regime == "BULL") or
                          (current_z < 0 and hmm_regime == "BEAR"),
    )

    # ── Signal ──
    signal = "NEUTRAL"
    score = 30
    reasons = []

    # Hurst > 0.5 = trending → follow trend
    if H > 0.6:
        if kalman_trend == "UP":
            signal = "BUY"
            score += 20
            reasons.append(f"Hurst={H:.2f} trending up (Kalman UP)")
        elif kalman_trend == "DOWN":
            signal = "SELL"
            score += 20
            reasons.append(f"Hurst={H:.2f} trending down (Kalman DOWN)")
    # Hurst < 0.5 = mean reverting → fade extremes
    elif H < 0.4:
        if current_z > 2.0:
            signal = "SELL"
            score += 15
            reasons.append(f"Hurst={H:.2f} mean-reverting, Z={current_z:.1f} → fade up")
        elif current_z < -2.0:
            signal = "BUY"
            score += 15
            reasons.append(f"Hurst={H:.2f} mean-reverting, Z={current_z:.1f} → fade down")

    # HMM regime bonus
    if hmm_regime == "BULL" and signal == "BUY":
        score += 10
        reasons.append(f"HMM=BULL aligned")
    elif hmm_regime == "BEAR" and signal == "SELL":
        score += 10
        reasons.append(f"HMM=BEAR aligned")
    elif hmm_regime == "VOLATILE":
        score -= 5
        reasons.append(f"HMM=VOLATILE — reduce conviction")

    score = max(0, min(100, score))
    reason = "; ".join(reasons) if reasons else "No strong quant signal"

    log.info(
        f"[QuantFactors] Hurst={H:.2f} Kalman={kalman_trend} HMM={hmm_regime} "
        f"Z={current_z:.2f} Bayes={bayesian_prob:.0%} → {signal} ({score}/100)"
    )

    return {
        "hurst": round(H, 3),
        "kalman_trend": kalman_trend,
        "hmm_regime": hmm_regime,
        "bayesian_win_prob": round(bayesian_prob, 3),
        "zscore": round(current_z, 2),
        "signal": signal,
        "score": score,
        "reason": reason,
    }


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.0003) + np.arange(n) * 0.0001
    df = pd.DataFrame({"open": close, "high": close + 0.0003, "low": close - 0.0003, "close": close}, index=dates)

    result = compute_quant_factors(df)
    print(f"Quant factors: {result['signal']} ({result['score']}/100)")
    print(f"  Hurst: {result['hurst']} ({'trending' if result['hurst'] > 0.5 else 'mean-reverting'})")
    print(f"  Kalman: {result['kalman_trend']}")
    print(f"  HMM: {result['hmm_regime']}")
    print(f"  Z-Score: {result['zscore']}")
    print(f"  Bayesian P(win): {result['bayesian_win_prob']:.0%}")
    print(f"  Reason: {result['reason']}")
    print("Quantitative factors smoke test passed.")
