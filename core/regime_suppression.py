"""
core/regime_suppression.py — Day 97+ False-Signal Regime Suppression
=====================================================================
Book reference: "The Only Technical Analysis Book You Will Ever Need" (Brian Hale)
Page 15: "False signals happen when conditions don't match the indicator's
historical performance. Avoid trading during known false-signal regimes."

Problem this solves:
  - Indicators like RSI, MACD, Bollinger Bands are calibrated for specific
    market conditions (trending or ranging)
  - When market regime changes (e.g., trend→chop, low-vol→high-vol spike),
    these indicators generate false signals
  - The system needs to know: "in THIS regime, THESE indicators are unreliable"

Solution:
  A regime-quality scorer that checks current market conditions and suppresses
  entry signals when conditions are known to produce false signals.

Known false-signal regimes (from book + empirical):
  1. Low-ADX chop (ADX < 15) — trend indicators (MACD, EMA cross) give false signals
  2. Post-news volatility spike — all indicators lag, price is noise-driven
  3. Session transitions (London→NY overlap end) — liquidity drain causes whipsaws
  4. Friday afternoon — weekend gap risk, thin liquidity
  5. Extreme volatility (ATR > 3× average) — stops get hit by noise
  6. Dead zone (Sydney session only) — no liquidity for breakouts

Usage:
    from core.regime_suppression import RegimeSuppressor
    rs = RegimeSuppressor()
    if rs.should_suppress(symbol="EURUSD", regime=regime_ctx, session=session_ctx):
        # don't enter new trades
        return "NO TRADE"
"""

from typing import Any, Dict, Optional
from utils.logger import get_logger

log = get_logger("regime_suppression")


class RegimeSuppressor:
    """Suppresses entry signals in known false-signal market conditions.

    Book Page 15: "indicators generate signals based on past data regardless
    of current context, which can produce false signals"
    """

    # ADX thresholds
    ADX_CHOP_THRESHOLD = 15    # below this = no trend = trend indicators unreliable
    ADX_MIN_FOR_TREND = 20     # below this = weak trend, be cautious

    # ATR spike threshold
    ATR_SPIKE_MULT = 3.0       # ATR > 3× average = extreme volatility

    # Session suppression
    SUPPRESSED_SESSIONS = {"DEAD_ZONE", "SYDNEY_ONLY"}

    # Friday close suppression (UTC)
    FRIDAY_SUPPRESS_HOUR_UTC = 20  # after 20:00 UTC Friday

    def should_suppress(
        self,
        symbol: str,
        regime: Optional[Dict[str, Any]] = None,
        session: Optional[Dict[str, Any]] = None,
        news_ctx: Optional[Dict[str, Any]] = None,
        ind_ctx: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        """Check if new entries should be suppressed for this symbol.

        Returns (should_suppress, reason).
        """
        from datetime import datetime, timezone

        # 1. Low-ADX chop — trend indicators give false signals
        if regime:
            adx = float(regime.get("adx", 0) or 0)
            regime_label = str(regime.get("regime", "")).upper()

            if adx < self.ADX_CHOP_THRESHOLD and regime_label in ("RANGING", "UNKNOWN"):
                return True, f"ADX={adx:.1f} < {self.ADX_CHOP_THRESHOLD} — chop regime, trend indicators unreliable"

            # 2. Extreme volatility — stops get hit by noise
            volatility = str(regime.get("volatility", "")).upper()
            if volatility == "EXTREME":
                return True, f"Extreme volatility — stops will be hit by noise"

        # 3. ATR spike check (if indicator data available)
        if ind_ctx:
            atr = float(ind_ctx.get("atr", 0) or 0)
            atr_avg = float(ind_ctx.get("atr_avg", 0) or 0)
            if atr > 0 and atr_avg > 0 and atr > atr_avg * self.ATR_SPIKE_MULT:
                return True, f"ATR spike: {atr:.5f} > {self.ATR_SPIKE_MULT}× avg ({atr_avg:.5f}) — extreme move"

        # 4. Session-based suppression
        if session:
            session_name = str(session.get("session", "")).upper()
            session_quality = str(session.get("quality", "")).upper()

            if session_name in self.SUPPRESSED_SESSIONS:
                return True, f"Session={session_name} — no liquidity for breakouts"

            if session_quality == "DEAD":
                return True, f"Session quality=DEAD — suppress entries"

        # 5. News-window suppression
        if news_ctx:
            news_blocked = news_ctx.get("trade_blocked", False)
            if news_blocked:
                return True, f"News window active — indicators lag during news"

            news_risk = str(news_ctx.get("risk_level", "")).upper()
            if news_risk == "HIGH":
                return True, f"High-impact news pending — suppress entries"

        # 6. Friday late session suppression
        now = datetime.now(timezone.utc)
        if now.weekday() == 4 and now.hour >= self.FRIDAY_SUPPRESS_HOUR_UTC:
            return True, f"Friday after {self.FRIDAY_SUPPRESS_HOUR_UTC}:00 UTC — weekend gap risk"

        return False, "OK"

    def get_regime_quality_score(
        self,
        regime: Optional[Dict[str, Any]] = None,
        session: Optional[Dict[str, Any]] = None,
        ind_ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Score the current regime quality from 0-100.

        100 = ideal conditions for signal reliability
        0 = worst possible conditions (suppress all entries)
        """
        score = 100.0

        if regime:
            adx = float(regime.get("adx", 0) or 0)
            volatility = str(regime.get("volatility", "")).upper()

            # ADX penalty
            if adx < self.ADX_CHOP_THRESHOLD:
                score -= 30  # chop = -30
            elif adx < self.ADX_MIN_FOR_TREND:
                score -= 15  # weak trend = -15

            # Volatility penalty
            if volatility == "EXTREME":
                score -= 25
            elif volatility == "HIGH":
                score -= 10

        if ind_ctx:
            atr = float(ind_ctx.get("atr", 0) or 0)
            atr_avg = float(ind_ctx.get("atr_avg", 0) or 0)
            if atr > 0 and atr_avg > 0:
                atr_ratio = atr / atr_avg
                if atr_ratio > self.ATR_SPIKE_MULT:
                    score -= 20
                elif atr_ratio > 2.0:
                    score -= 10

        if session:
            session_quality = str(session.get("quality", "")).upper()
            if session_quality == "DEAD":
                score -= 30
            elif session_quality == "LOW":
                score -= 15

        return max(0, min(100, score))


# ── Singleton ─────────────────────────────────────────────────────

_RS: Optional[RegimeSuppressor] = None


def get_regime_suppressor() -> RegimeSuppressor:
    global _RS
    if _RS is None:
        _RS = RegimeSuppressor()
    return _RS
