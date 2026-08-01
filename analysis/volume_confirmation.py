"""
analysis/volume_confirmation.py — Day 97+ Volume Confirmation for Breakouts
=============================================================================
Book reference: "The Only Technical Analysis Book You Will Ever Need" (Brian Hale)
Page 9: "A stock breaking out of a long consolidation, with strong volume,
was treated as a high-probability long setup"
Page 6: "Volume mentioned as a data point worth tracking even on 'ignored' bars"

Problem this solves:
  - Breakout signals without volume confirmation are often false breakouts
  - A price breakout on LOW volume = institutions aren't participating = likely fake
  - A price breakout on HIGH volume = institutions are pushing it = more likely real

Solution:
  Check if current candle's volume is above average. If breakout + high volume =
  strong signal. If breakout + low volume = weaken confidence or reject.

Usage:
    from analysis.volume_confirmation import VolumeConfirmation
    vc = VolumeConfirmation()
    result = vc.check_breakout(df, direction="BUY", breakout_level=1.1050)
    # → {"confirmed": True, "volume_ratio": 1.8, "adjustment": +10}
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from utils.logger import get_logger

log = get_logger("volume_confirmation")


class VolumeConfirmation:
    """Validates breakouts with volume analysis.

    Book Page 9: breakout from consolidation WITH volume = high probability
    Book Page 6: don't ignore volume on any bar — it carries signal
    """

    # Config
    VOLUME_AVG_WINDOW = 20       # 20-bar average volume
    VOLUME_CONFIRM_MULT = 1.3    # volume must be 1.3× average to confirm breakout
    VOLUME_STRONG_MULT = 2.0     # volume > 2× average = strong confirmation
    VOLUME_WEAK_MULT = 0.7       # volume < 0.7× average = weak/no confirmation

    # Confidence adjustments
    CONFIRM_BOOST = 10           # +10% confidence when volume confirms
    STRONG_BOOST = 15            # +15% when volume is 2×+
    WEAK_PENALTY = -10           # -10% when volume is weak
    NO_VOLUME_PENALTY = -5       # -5% when volume data unavailable

    def check_breakout(
        self,
        df: pd.DataFrame,
        direction: str,
        breakout_level: float,
        volume_col: str = "volume",
    ) -> Dict[str, Any]:
        """Check if a breakout is confirmed by volume.

        Args:
            df: OHLCV DataFrame
            direction: "BUY" (broke above) or "SELL" (broke below)
            breakout_level: the price level that was broken
            volume_col: name of volume column

        Returns:
            {
                "confirmed": bool,
                "volume_ratio": float,  # current vol / avg vol
                "adjustment": int,      # confidence adjustment (+15 to -10)
                "reason": str,
            }
        """
        if volume_col not in df.columns or len(df) < self.VOLUME_AVG_WINDOW + 1:
            return {
                "confirmed": False,
                "volume_ratio": 0,
                "adjustment": self.NO_VOLUME_PENALTY,
                "reason": "insufficient volume data",
            }

        # Sanitize volume to numeric
        vol = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
        current_vol = float(vol.iloc[-1])
        avg_vol = float(vol.iloc[-self.VOLUME_AVG_WINDOW-1:-1].mean())

        if avg_vol <= 0:
            return {
                "confirmed": False,
                "volume_ratio": 0,
                "adjustment": self.NO_VOLUME_PENALTY,
                "reason": "average volume is zero",
            }

        volume_ratio = current_vol / avg_vol

        # Check if price actually broke the level
        close = float(df["close"].iloc[-1])
        if direction.upper() == "BUY":
            broke = close > breakout_level
        else:
            broke = close < breakout_level

        if not broke:
            return {
                "confirmed": False,
                "volume_ratio": round(volume_ratio, 2),
                "adjustment": 0,
                "reason": f"price {close} hasn't broken {breakout_level}",
            }

        # Volume confirmation logic
        if volume_ratio >= self.VOLUME_STRONG_MULT:
            return {
                "confirmed": True,
                "volume_ratio": round(volume_ratio, 2),
                "adjustment": self.STRONG_BOOST,
                "reason": f"strong volume: {volume_ratio:.1f}× average — institutions participating",
            }
        elif volume_ratio >= self.VOLUME_CONFIRM_MULT:
            return {
                "confirmed": True,
                "volume_ratio": round(volume_ratio, 2),
                "adjustment": self.CONFIRM_BOOST,
                "reason": f"volume confirmed: {volume_ratio:.1f}× average",
            }
        elif volume_ratio < self.VOLUME_WEAK_MULT:
            return {
                "confirmed": False,
                "volume_ratio": round(volume_ratio, 2),
                "adjustment": self.WEAK_PENALTY,
                "reason": f"weak volume: {volume_ratio:.1f}× average — likely false breakout",
            }
        else:
            return {
                "confirmed": False,
                "volume_ratio": round(volume_ratio, 2),
                "adjustment": 0,
                "reason": f"normal volume: {volume_ratio:.1f}× average — no confirmation",
            }

    def get_volume_context(self, df: pd.DataFrame, volume_col: str = "volume") -> Dict[str, Any]:
        """Get volume context for AI analysis (not just breakout validation).

        Book Page 6: "don't discard bars even if volume/price action seems anomalous"
        → anomalous volume IS signal. Report it.
        """
        if volume_col not in df.columns or len(df) < 2:
            return {"available": False, "reason": "no volume data"}

        vol = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
        current_vol = float(vol.iloc[-1])
        avg_vol = float(vol.iloc[-min(20, len(vol)-1):-1].mean()) if len(vol) > 1 else current_vol

        if avg_vol <= 0:
            return {"available": False, "reason": "zero average volume"}

        ratio = current_vol / avg_vol
        return {
            "available": True,
            "current_volume": round(current_vol, 0),
            "average_volume": round(avg_vol, 0),
            "volume_ratio": round(ratio, 2),
            "is_anomalous": ratio >= self.VOLUME_STRONG_MULT,
            "is_weak": ratio < self.VOLUME_WEAK_MULT,
            "description": (
                f"high volume ({ratio:.1f}× avg)" if ratio >= self.VOLUME_STRONG_MULT
                else f"above average ({ratio:.1f}× avg)" if ratio >= self.VOLUME_CONFIRM_MULT
                else f"normal ({ratio:.1f}× avg)" if ratio >= self.VOLUME_WEAK_MULT
                else f"low volume ({ratio:.1f}× avg)"
            ),
        }

    def check_trend_confirmation(
        self,
        df: pd.DataFrame,
        volume_col: str = "volume",
        lookback: int = 10,
    ) -> Dict[str, Any]:
        """Day 97+ Book Rule (Page 27): Volume/Price Trend Confirmation.

        Rising price + rising volume = confirmed strong uptrend (likely to continue)
        Falling price + falling volume = weak downtrend (possible reversal)
        Rising price + falling volume = divergence (uptrend weak, beware)
        Falling price + rising volume = strong downtrend (likely to continue)

        Returns:
            {
                "trend_confirmed": bool,
                "price_trend": "up" | "down" | "flat",
                "volume_trend": "up" | "down" | "flat",
                "divergence": bool,
                "adjustment": int,   # confidence adjustment
                "reason": str,
            }
        """
        if volume_col not in df.columns or len(df) < lookback + 1:
            return {
                "trend_confirmed": False,
                "price_trend": "unknown",
                "volume_trend": "unknown",
                "divergence": False,
                "adjustment": 0,
                "reason": "insufficient data",
            }

        # Sanitize
        close = pd.to_numeric(df["close"], errors="coerce").fillna(0)
        vol = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)

        # Calculate trends over lookback window
        recent_close = close.iloc[-lookback:]
        recent_vol = vol.iloc[-lookback:]

        # Simple linear trend: compare first half avg to second half avg
        mid = len(recent_close) // 2
        if mid < 1:
            return {
                "trend_confirmed": False,
                "price_trend": "unknown",
                "volume_trend": "unknown",
                "divergence": False,
                "adjustment": 0,
                "reason": "lookback too short",
            }

        price_first_half = float(recent_close.iloc[:mid].mean())
        price_second_half = float(recent_close.iloc[mid:].mean())
        vol_first_half = float(recent_vol.iloc[:mid].mean())
        vol_second_half = float(recent_vol.iloc[mid:].mean())

        # Determine trends
        price_change_pct = (price_second_half - price_first_half) / price_first_half * 100 if price_first_half > 0 else 0
        vol_change_pct = (vol_second_half - vol_first_half) / vol_first_half * 100 if vol_first_half > 0 else 0

        price_trend = "up" if price_change_pct > 0.05 else "down" if price_change_pct < -0.05 else "flat"
        volume_trend = "up" if vol_change_pct > 5 else "down" if vol_change_pct < -5 else "flat"

        # Book Page 27 rules
        divergence = False
        confirmed = False
        adjustment = 0
        reason = ""

        if price_trend == "up" and volume_trend == "up":
            confirmed = True
            adjustment = 10
            reason = f"confirmed strong uptrend (price +{price_change_pct:.2f}%, vol +{vol_change_pct:.1f}%)"
        elif price_trend == "down" and volume_trend == "down":
            confirmed = False
            adjustment = -5
            reason = f"weak downtrend, possible reversal (price {price_change_pct:.2f}%, vol {vol_change_pct:.1f}%)"
        elif price_trend == "up" and volume_trend == "down":
            divergence = True
            adjustment = -8
            reason = f"bearish divergence: price up but volume fading — uptrend weak"
        elif price_trend == "down" and volume_trend == "up":
            confirmed = True
            adjustment = -10  # negative = bearish confirmation
            reason = f"strong downtrend confirmed (price {price_change_pct:.2f}%, vol +{vol_change_pct:.1f}%)"
        else:
            reason = f"no clear trend confirmation (price {price_trend}, vol {volume_trend})"

        return {
            "trend_confirmed": confirmed,
            "price_trend": price_trend,
            "volume_trend": volume_trend,
            "divergence": divergence,
            "adjustment": adjustment,
            "reason": reason,
        }


# ── Singleton ─────────────────────────────────────────────────────

_VC: Optional[VolumeConfirmation] = None


def get_volume_confirmation() -> VolumeConfirmation:
    global _VC
    if _VC is None:
        _VC = VolumeConfirmation()
    return _VC
