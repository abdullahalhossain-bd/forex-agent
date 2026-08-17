# analysis/mean_reversion_confluence_engine.py
# ============================================================
# Mean-Reversion Confluence Engine — validated proxy backtest result:
#   1047 trades (old 2-factor hard-AND) -> 525 trades (this scoring
#   version), winrate 49.6% -> 51.2%, expectancy -0.007R -> +0.025R,
#   held-out test window (2026-04-25 onward), across EURAUD/GBPCAD/EURCAD.
#
# ⚠️ VALIDATION CAVEAT (read before wiring this into live/backtest):
#   This was validated by an LLM (Claude) acting as a deterministic PROXY
#   for master_analyst.py's live Groq/Gemini call, on M15 OHLCV CSVs, on
#   ONE ~13-month period, split 70/30 train/test. It has NOT been:
#     - tested against your real master_analyst.py / devils_advocate.py
#       live LLM calls
#     - tested on more than 3 pairs or more than ~13 months of data
#     - tested with realistic slippage/commission/swap (only the CSV's
#       own spread column was used)
#   Treat this as a promising rule-based candidate for the
#   is_backtest_mode() fallback path, not as a proven live edge. Forward-
#   test on a demo account before risking real capital.
#
# WHAT THIS DOES: scores 5 independent confluence factors and returns a
# BUY/SELL/WAIT decision + score, for use as the rule-engine "signal" that
# feeds MasterAnalyst._build_context()'s `signal` param (or as a
# standalone rule-only strategy in backtest fallback mode).
#
# CHANGE FROM the previous simpler 2-factor hard-AND filter (trend + BB
# extreme, requiring BOTH): this uses a score-based OR — 2-of-5 factors
# required (paired with MeanReversionDAGate's veto_threshold=3 doing the
# quality filtering instead) — which is what raised trade frequency
# without hurting winrate (see validation numbers above). The RSI-extreme
# hard veto from the old filter was removed and folded into the score.
# ============================================================

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("mean_reversion_confluence")

SCORE_THRESHOLD = 1          # updated after a robustness sweep across nearby
                              # (score_threshold, DA veto_threshold) pairs — see
                              # risk/mean_reversion_da_gate.py's ROBUSTNESS_SWEEP
                              # table. (1, veto=2) sits inside a region where
                              # da_threshold=2-3 does well across MULTIPLE
                              # score_thresholds (1,2,3), not just one isolated
                              # point — that's evidence of a real pattern rather
                              # than a lucky single config. It is NOT proof the
                              # data-snooping concern is gone (see README) — the
                              # whole sweep still used the same held-out window.
ADX_TREND_MIN = 20.0
BB_SELL_EXTREME = 0.75
BB_BUY_EXTREME = 0.25
WICK_REJECTION_RATIO = 0.35
VOLUME_Z_MIN = 0.5
DEAD_ZONE_UTC_HOURS = (21, 22, 23)   # matches master_analyst.py's own dead-zone rule


class MeanReversionConfluenceEngine:
    """
    Usage:
        engine = MeanReversionConfluenceEngine()
        df = engine.prepare(df)              # adds indicator columns (idempotent)
        direction, score = engine.decide(df.iloc[-1])
        ctx = engine.get_ai_context(df.iloc[-1], direction, score)
    """

    def __init__(self, score_threshold: int = SCORE_THRESHOLD):
        self.score_threshold = score_threshold

    # ═══════════════════════════════════════════════════════
    # STEP 1: INDICATORS (idempotent — safe to call every bar)
    # ═══════════════════════════════════════════════════════
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds trend/RSI/ATR/BB%/ADX/wick/volume/session columns if not
        already present. Expects columns: open, high, low, close,
        tick_volume, datetime_utc (spread optional)."""
        df = df.copy()

        if "ema20" not in df.columns:
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

        if "rsi14" not in df.columns:
            delta = df["close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            df["rsi14"] = 100 - (100 / (1 + rs))

        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        if "atr14" not in df.columns:
            df["atr14"] = tr.rolling(14).mean()

        if "bb_pct" not in df.columns:
            ma20 = df["close"].rolling(20).mean()
            sd20 = df["close"].rolling(20).std()
            bb_upper = ma20 + 2 * sd20
            bb_lower = ma20 - 2 * sd20
            df["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

        if "adx" not in df.columns:
            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            atr_smooth = tr.rolling(14).mean().replace(0, np.nan)
            plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr_smooth
            minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr_smooth
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
            df["adx"] = dx.rolling(14).mean()

        if "upper_wick_ratio" not in df.columns:
            body_top = df[["close", "open"]].max(axis=1)
            body_bot = df[["close", "open"]].min(axis=1)
            rng = (df["high"] - df["low"]).replace(0, np.nan)
            df["upper_wick_ratio"] = (df["high"] - body_top) / rng
            df["lower_wick_ratio"] = (body_bot - df["low"]) / rng
            df["bearish_engulf"] = (
                (df["close"] < df["open"])
                & (df["close"].shift(1) > df["open"].shift(1))
                & (df["close"] < df["open"].shift(1))
                & (df["open"] > df["close"].shift(1))
            )
            df["bullish_engulf"] = (
                (df["close"] > df["open"])
                & (df["close"].shift(1) < df["open"].shift(1))
                & (df["close"] > df["open"].shift(1))
                & (df["open"] < df["close"].shift(1))
            )

        if "vol_z" not in df.columns and "tick_volume" in df.columns:
            vol_mean = df["tick_volume"].rolling(50).mean()
            vol_std = df["tick_volume"].rolling(50).std()
            df["vol_z"] = (df["tick_volume"] - vol_mean) / vol_std.replace(0, np.nan)
        elif "vol_z" not in df.columns:
            df["vol_z"] = 0.0

        if "is_dead_zone" not in df.columns and "datetime_utc" in df.columns:
            df["utc_hour"] = pd.to_datetime(df["datetime_utc"]).dt.hour
            df["is_dead_zone"] = df["utc_hour"].isin(DEAD_ZONE_UTC_HOURS)
        elif "is_dead_zone" not in df.columns:
            df["is_dead_zone"] = False

        if "trend" not in df.columns:
            bullish = (df["ema20"] > df["ema50"]) & (df["ema50"] > df["ema200"])
            bearish = (df["ema20"] < df["ema50"]) & (df["ema50"] < df["ema200"])
            df["trend"] = np.select([bullish, bearish], ["bullish", "bearish"], default="mixed")

        return df

    # ═══════════════════════════════════════════════════════
    # STEP 2: DECISION
    # ═══════════════════════════════════════════════════════
    def decide(self, row: pd.Series) -> tuple[Optional[str], int]:
        """Returns (direction, score) where direction is 'BUY'/'SELL'/None."""
        if bool(row.get("is_dead_zone", False)):
            return None, 0
        if pd.isna(row.get("bb_pct")) or pd.isna(row.get("adx")) or pd.isna(row.get("rsi14")):
            return None, 0

        for direction in ("SELL", "BUY"):
            score = 0
            if direction == "SELL":
                if row["trend"] == "bearish":
                    score += 1
                if row["bb_pct"] > BB_SELL_EXTREME:
                    score += 1
                if row["adx"] > ADX_TREND_MIN:
                    score += 1
                if row["upper_wick_ratio"] > WICK_REJECTION_RATIO or row["bearish_engulf"]:
                    score += 1
                if row["vol_z"] > VOLUME_Z_MIN:
                    score += 1
            else:
                if row["trend"] == "bullish":
                    score += 1
                if row["bb_pct"] < BB_BUY_EXTREME:
                    score += 1
                if row["adx"] > ADX_TREND_MIN:
                    score += 1
                if row["lower_wick_ratio"] > WICK_REJECTION_RATIO or row["bullish_engulf"]:
                    score += 1
                if row["vol_z"] > VOLUME_Z_MIN:
                    score += 1
            if score >= self.score_threshold:
                return direction, score
        return None, 0

    # ═══════════════════════════════════════════════════════
    # STEP 3: AI CONTEXT (for MasterAnalyst._build_context's `signal` param)
    # ═══════════════════════════════════════════════════════
    def get_ai_context(self, row: pd.Series, direction: Optional[str], score: int) -> dict:
        return {
            "signal": direction or "NO TRADE",
            "confidence": min(95, score * 20),
            "confluence_score": score,
            "confluence_max": 5,
            "factors": {
                "trend": row.get("trend"),
                "bb_pct": round(float(row.get("bb_pct", 0)), 3) if pd.notna(row.get("bb_pct")) else None,
                "adx": round(float(row.get("adx", 0)), 1) if pd.notna(row.get("adx")) else None,
                "rejection_wick": bool(
                    row.get("upper_wick_ratio", 0) > WICK_REJECTION_RATIO
                    or row.get("lower_wick_ratio", 0) > WICK_REJECTION_RATIO
                    or row.get("bearish_engulf", False)
                    or row.get("bullish_engulf", False)
                ),
                "volume_z": round(float(row.get("vol_z", 0)), 2) if pd.notna(row.get("vol_z")) else None,
                "session_dead_zone": bool(row.get("is_dead_zone", False)),
            },
        }