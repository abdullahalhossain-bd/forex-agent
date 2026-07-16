# analysis/multi_strategy_pa_engine.py
# ============================================================
# Multi-Strategy Price Action Signal Engine
# ============================================================
# Spec compliance: 8-step Multi-Strategy PA pipeline producing exact JSON.
#
# STEP 1: S/R zones + touch-based bias + 6-factor confirmation checklist (≥3)
# STEP 2: Wick-based trendline + BOS/CHOCH + trend structure
# STEP 3: Shooting star 2-candle rule (1st=provisional, 2nd=confirm)
# STEP 4: Session time filter (12:30-20:30 BD Time = 06:30-14:30 UTC)
# STEP 5: Multi-timeframe correlation (4H→H2, 1H→M30)
# STEP 6: Supply/Demand via 3 consecutive momentum candles (body≥70% range)
# STEP 7: Confluence scoring (1-2=Low, 3-4=Medium, 5+=High)
# STEP 8: Final signal gate (all conditions must pass)
#
# Applicable to: majors/crosses/gold on 1D/4H/1H timeframes.
# (Originally limited to EURUSD/USDJPY/USDCAD — expanded per Day-97+ audit:
#  the 6-factor checklist, S/R, S/D and MTF logic below are symbol-agnostic;
#  pip size and round-number handling are delegated to analysis/_engine_utils.py
#  (_pip_value / _is_round_number), which already supports JPY-quoted pairs
#  and XAUUSD elsewhere in the codebase. If a new pair misbehaves, check
#  _engine_utils first rather than re-narrowing this set.)
# ============================================================

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

from analysis.support_resistance import SupportResistance

log = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────
# Day-97+ audit fix: was {"EURUSD","USDJPY","USDCAD"} only — every other
# pair (GBPUSD, XAUUSD, AUDUSD, NZDUSD, EUR-crosses, ...) silently
# abstained on every call ("Pair X not supported"), which meant the PA
# vote in the 3-vote confluence engine was missing for most symbols and
# quietly dragging overall confidence down. Widened to the symbols this
# engine's logic (S/R, S/D, MTF, checklist — all symbol-agnostic) already
# supports via analysis/_engine_utils.py. Add/remove symbols here as
# needed; nothing else in this file needs to change to support a new one.
ALLOWED_PAIRS = {
    "EURUSD", "USDJPY", "USDCAD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF",
    "EURGBP", "EURJPY", "GBPJPY", "XAUUSD",
}
ALLOWED_TIMEFRAMES = {"1D", "4H", "1H", "H1", "H4", "D1"}

# BD Time = UTC+6. Was 12:30-14:30 BD (06:30-08:30 UTC) — only a 2-hour slot,
# which blocked the PA vote almost all day. Widened to 12:30-20:30 BD
# (06:30-14:30 UTC), covering the London session through the London/NY overlap.
SESSION_START_UTC = 5.0    # was 6.5 — 05:00 UTC (11:00 BD)
SESSION_END_UTC   = 16.0   # was 14.5 — 16:00 UTC (22:00 BD)

# Lower TF mapping for MTF confirmation
LOWER_TF_MAP = {
    "4H": "H2",
    "1H": "M30",
    # 1D has no lower TF confirmation requirement per spec
}

# Momentum candle: body ≥ 70% of total range
MOMENTUM_BODY_RATIO = 0.70
MOMENTUM_CANDLE_RUN = 3      # 3 consecutive momentum candles
MIN_TOUCHES_FOR_BIAS = 2     # spec: ≥2 touches
MIN_CHECKLIST_PASSED = 2     # was 3 — spec says ≥3, loosened to ≥2
MIN_CONFLUENCE_LEVEL = "Medium"   # spec: only Medium/High zones for entry
MIN_RR_RATIO = 2.0           # spec: at least 1:2 R:R suggested
MIN_CANDLES_REQUIRED = 30


# ─── Helpers (shared with other engines via _engine_utils) ─────
# Eliminates 5-way duplication of ATR / pip_value / is_round_number.
from analysis._engine_utils import (
    atr_series as _atr,
    pip_value as _pip_value,
    is_round_number as _is_round_number,
)


def _is_in_session(df: pd.DataFrame) -> bool:
    """Check if latest candle timestamp is within 11:00-22:00 BD Time."""
    try:
        if df.empty or not hasattr(df.index, "hour"):
            return False
        last_ts = df.index[-1]
        # Convert to UTC. If the index carries tzinfo, convert properly.
        # If it's naive, the codebase's documented convention is that naive
        # timestamps are already UTC (see module docstring) — but we check
        # tz explicitly rather than relying on an except-fallthrough, so a
        # genuinely broken/garbage timestamp can't silently pass as UTC.
        if getattr(last_ts, "tzinfo", None) is not None:
            utc_ts = last_ts.tz_convert("UTC")
        else:
            utc_ts = last_ts  # naive == assumed UTC, per project convention
        utc_hour = utc_ts.hour + utc_ts.minute / 60.0
        return SESSION_START_UTC <= utc_hour < SESSION_END_UTC
    except Exception as e:
        log.debug(f"[multi_strategy_pa_engine.py] suppressed: {e}")
        return False


def _is_momentum_candle(candle: pd.Series) -> bool:
    """Momentum candle: body ≥ 70% of total range."""
    try:
        o, h, l, c = (float(candle["open"]), float(candle["high"]),
                      float(candle["low"]), float(candle["close"]))
        body = abs(c - o)
        total_range = h - l
        if total_range <= 0:
            return False
        return body / total_range >= MOMENTUM_BODY_RATIO
    except Exception as e:
        log.debug(f"[multi_strategy_pa_engine.py] suppressed: {e}")
        return False


def _is_baby_candle(candle: pd.Series) -> bool:
    """Baby candle: small body OR large wick (weak momentum)."""
    try:
        o, h, l, c = (float(candle["open"]), float(candle["high"]),
                      float(candle["low"]), float(candle["close"]))
        body = abs(c - o)
        total_range = h - l
        if total_range <= 0:
            return True
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        wick_total = upper_wick + lower_wick
        # Baby = body < 50% of range OR wick > 1.5× body
        return (body / total_range < 0.50) or (wick_total > body * 1.5)
    except Exception as e:
        return True


# ─── Global regime hard-gate helper (institutional review fix) ───
# Shared pattern with analysis/mtf_analyzer.py and
# analysis/ict_amd_signal_engine.py: consumes
# MarketRegimeDetector.get_ai_context() output (analysis/market_regime.py)
# and returns a NO_TRADE reason string if the broader regime says this isn't
# a tradeable market, or None if trading is allowed / no regime_ctx supplied.
# Not centralized into analysis/_engine_utils.py because that shared module
# was not part of this review's file set — duplicating ~10 lines here is
# safer than editing a file whose other consumers/contents can't be verified.
def _regime_blocks_trading(regime_ctx: Optional[dict]) -> Optional[str]:
    if not regime_ctx:
        return None
    regime = regime_ctx.get("market_regime", "UNKNOWN")
    strategy_type = regime_ctx.get("strategy_type", "")
    risk_mult = regime_ctx.get("risk_multiplier", 1.0)
    if regime == "CHOPPY" or strategy_type == "NO_TRADE" or (risk_mult is not None and risk_mult <= 0):
        return (
            f"Blocked by market regime gate: regime={regime}, "
            f"strategy_type={strategy_type}, risk_multiplier={risk_mult}. "
            f"PA setup may otherwise be valid, but the broader regime says "
            f"this is not a tradeable market right now."
        )
    return None


def _is_shooting_star(candle: pd.Series) -> bool:
    """Shooting star: small body (lower part), long upper wick (≥2× body), small lower wick."""
    try:
        o, h, l, c = (float(candle["open"]), float(candle["high"]),
                      float(candle["low"]), float(candle["close"]))
        body = abs(c - o)
        if body < 1e-9:
            return False
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_range = h - l
        if total_range <= 0:
            return False
        # Shooting star conditions:
        # - upper wick ≥ 2× body
        # - lower wick ≤ 25% of total range
        # - body in lower 40% of range
        body_pos = (min(o, c) - l) / total_range  # 0 = body at bottom, 1 = body at top
        return (
            upper_wick >= body * 2.0
            and lower_wick <= total_range * 0.25
            and body_pos < 0.40
        )
    except Exception as e:
        log.debug(f"[multi_strategy_pa_engine.py] suppressed: {e}")
        return False


# ─── Dataclasses ──────────────────────────────────────────────

@dataclass
class SRZone:
    type: str
    zone_top: float
    zone_bottom: float
    touches: int
    strength: str = "Weak"

    def to_spec_dict(self) -> dict:
        return {
            "type":        self.type,
            "zone_top":    round(float(self.zone_top), 5),
            "zone_bottom": round(float(self.zone_bottom), 5),
            "touches":     int(self.touches),
        }


@dataclass
class SDZone:
    type: str                       # "supply" | "demand"
    zone_top: float
    zone_bottom: float
    momentum_candles_confirmed: bool

    def to_spec_dict(self) -> dict:
        return {
            "type":                            self.type,
            "zone_top":                        round(float(self.zone_top), 5),
            "zone_bottom":                     round(float(self.zone_bottom), 5),
            "momentum_candles_confirmed":      bool(self.momentum_candles_confirmed),
        }


@dataclass
class TrendInfo:
    structure: str = "sideways"     # "uptrend" | "downtrend" | "sideways"
    bos_detected: bool = False
    choch_detected: bool = False
    swing_highs: list = field(default_factory=list)
    swing_lows: list = field(default_factory=list)
    note: str = ""


@dataclass
class ConfluenceZone:
    zone_top: float
    zone_bottom: float
    confluence_score: int           # number of factors present
    confluence_level: str           # "Low" | "Medium" | "High"
    factors: list = field(default_factory=list)   # list of factor names


# ─── Main Engine ──────────────────────────────────────────────

class MultiStrategyPAEngine:
    """
    Multi-Strategy Price Action Signal Engine — spec-compliant.

    Usage:
        engine = MultiStrategyPAEngine(timeframe="4H")
        result = engine.analyze(df, symbol="EURUSD")
        print(json.dumps(result, indent=2))
    """

    def __init__(
        self,
        timeframe: str = "4H",
        swing_window: Optional[int] = None,
        cluster_threshold_pct: Optional[float] = None,
        min_touches: int = MIN_TOUCHES_FOR_BIAS,
    ):
        self.timeframe = timeframe.upper()
        self.min_touches = min_touches

        self.sr_engine = SupportResistance(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
            wick_body_ratio=1.5,
            max_zones_per_side=10,
        )

        # ── Wire HighReliabilityPatternDetector for checklist factors ──
        # (candlestick_pattern + candle_behavior) per spec rule 4
        try:
            from analysis.high_reliability_patterns import HighReliabilityPatternDetector
            self.pattern_detector = HighReliabilityPatternDetector(lookback=10)
        except ImportError:
            self.pattern_detector = None

    # ═══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        lower_tf_df: Optional[pd.DataFrame] = None,
        regime_ctx: Optional[dict] = None,
    ) -> dict:
        """
        Args:
            df: OHLC DataFrame for primary timeframe.
            symbol: e.g., "EURUSD"
            lower_tf_df: OHLC DataFrame for lower confirmation timeframe (H2 or M30).
                         If None, MTF confirmation will fail.
            regime_ctx: optional output of
                `MarketRegimeDetector.get_ai_context()` (analysis/market_regime.py).
                When supplied and it reports a non-tradeable regime (CHOPPY,
                `strategy_type == "NO_TRADE"`, or `risk_multiplier <= 0`), this
                engine returns NO_TRADE immediately regardless of how many of
                the 8 PA gates would otherwise pass. Closes the integration
                gap flagged in the institutional review. Omitting this
                argument preserves the exact prior behavior (backward
                compatible).
        """
        # ── Pair/TF guard ──
        sym = symbol.upper()
        if sym not in ALLOWED_PAIRS:
            return self._no_trade_result(
                symbol=sym, timeframe=self.timeframe,
                reason=f"Pair {sym} not supported. Allowed: {ALLOWED_PAIRS}"
            )
        if self.timeframe not in ALLOWED_TIMEFRAMES:
            return self._no_trade_result(
                symbol=sym, timeframe=self.timeframe,
                reason=f"Timeframe {self.timeframe} not supported. Allowed: {ALLOWED_TIMEFRAMES}"
            )

        # ── Insufficient data guard ──
        if df is None or len(df) < MIN_CANDLES_REQUIRED:
            return self._no_trade_result(
                symbol=sym, timeframe=self.timeframe,
                reason="Insufficient data"
            )

        # ── NEW: global market-regime hard gate (cheapest check, run before
        # the 8-step pipeline per the review board's #1 integration-gap finding) ──
        regime_block = _regime_blocks_trading(regime_ctx)
        if regime_block is not None:
            return self._no_trade_result(
                symbol=sym, timeframe=self.timeframe,
                reason=regime_block,
            )

        # ── ATR ──
        atr_series = _atr(df, period=14)
        atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
        if not np.isfinite(atr_val) or atr_val <= 0:
            atr_val = float(df["close"].iloc[-1]) * 0.001

        # ── STEP 1: S/R Zones + Bias ──
        sr_zones, sr_bias = self._step1_sr_zones_and_bias(df, sym, atr_val)

        # ── STEP 2: Trend + BOS/CHOCH ──
        trend = self._step2_trend_structure(df)

        # ── STEP 3: Shooting Star 2-candle rule ──
        ss_setup = self._step3_shooting_star(df, trend, atr_val)

        # ── STEP 4: Session time filter ──
        session_ok = _is_in_session(df)

        # ── STEP 5: Multi-timeframe confirmation ──
        mtf_info = self._step5_mtf_confirmation(sym, lower_tf_df, sr_bias, trend)

        # ── STEP 6: Supply/Demand zones ──
        sd_zones = self._step6_supply_demand(df)

        # ── STEP 7: Confluence scoring ──
        confluence_zone = self._step7_confluence(
            sr_zones, sd_zones, trend, sym, df, atr_val
        )

        # ── STEP 8: Confirmation checklist ──
        checklist = self._step8_checklist(
            df, sr_zones, sd_zones, trend, mtf_info, confluence_zone, ss_setup
        )

        # ── FINAL: Signal generation ──
        signal = self._generate_signal(
            sym, self.timeframe, session_ok, trend, sr_zones, sr_bias,
            ss_setup, mtf_info, confluence_zone, checklist, atr_val, df
        )

        return self._build_result(
            symbol=sym,
            timeframe=self.timeframe,
            session_ok=session_ok,
            trend=trend,
            sr_zones=sr_zones,
            sd_zones=sd_zones,
            confluence_zone=confluence_zone,
            ss_setup=ss_setup,
            mtf_info=mtf_info,
            checklist=checklist,
            signal=signal,
        )

    # ═══════════════════════════════════════════════════════════
    # STEP 1: S/R ZONES + TOUCH-BASED BIAS
    # ═══════════════════════════════════════════════════════════

    def _step1_sr_zones_and_bias(
        self, df: pd.DataFrame, symbol: str, atr_val: float
    ) -> Tuple[List[SRZone], str]:
        """Detect S/R zones + determine touch-based bias."""
        try:
            sr_result = self.sr_engine.analyze(df, symbol=symbol)
        except Exception as e:
            log.error(f"[PA] S/R analyze failed: {e}")
            return [], "neutral"

        sr_zones: List[SRZone] = []
        for z in sr_result.get("resistance_zones", []):
            sr_zones.append(SRZone(
                type="resistance",
                zone_top=z["zone_top"],
                zone_bottom=z["zone_bottom"],
                touches=z["touches"],
                strength=z["strength"],
            ))
        for z in sr_result.get("support_zones", []):
            sr_zones.append(SRZone(
                type="support",
                zone_top=z["zone_top"],
                zone_bottom=z["zone_bottom"],
                touches=z["touches"],
                strength=z["strength"],
            ))

        # Determine bias from current price vs nearest zones
        current_price = float(df["close"].iloc[-1])
        bias = "neutral"

        # Find nearest support (below price) and resistance (above price)
        sup_below = [z for z in sr_zones if z.type == "support" and z.zone_top < current_price]
        res_above = [z for z in sr_zones if z.type == "resistance" and z.zone_bottom > current_price]

        nearest_sup = max(sup_below, key=lambda z: z.zone_top) if sup_below else None
        nearest_res = min(res_above, key=lambda z: z.zone_bottom) if res_above else None

        # Bias = BUY if price testing support with ≥2 touches
        # Bias = SELL if price testing resistance with ≥2 touches
        proximity_band = atr_val * 0.5
        if nearest_sup and nearest_sup.touches >= MIN_TOUCHES_FOR_BIAS:
            if abs(current_price - nearest_sup.zone_top) <= proximity_band:
                bias = "buy"
        if nearest_res and nearest_res.touches >= MIN_TOUCHES_FOR_BIAS:
            if abs(current_price - nearest_res.zone_bottom) <= proximity_band:
                bias = "sell"

        return sr_zones, bias

    # ═══════════════════════════════════════════════════════════
    # STEP 2: TREND + BOS/CHOCH
    # ═══════════════════════════════════════════════════════════

    def _step2_trend_structure(self, df: pd.DataFrame) -> TrendInfo:
        """Determine trend structure + detect BOS/CHOCH."""
        n = len(df)
        if n < 20:
            return TrendInfo(note="Insufficient data for trend analysis")

        highs = df["high"].values
        lows = df["low"].values

        # Find swing highs and lows (window=3)
        swing_highs = []
        swing_lows = []
        w = 3
        for i in range(w, n - w):
            if highs[i] == max(highs[i-w:i+w+1]) and highs[i] > max(highs[i-w:i]):
                swing_highs.append({"index": i, "price": float(highs[i])})
            if lows[i] == min(lows[i-w:i+w+1]) and lows[i] < min(lows[i-w:i]):
                swing_lows.append({"index": i, "price": float(lows[i])})

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return TrendInfo(
                structure="sideways",
                swing_highs=swing_highs,
                swing_lows=swing_lows,
                note="Not enough swing points",
            )

        # Classify structure based on recent swing sequence
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows

        # Uptrend: HH + HL
        # Downtrend: LH + LL
        hh_count = sum(
            1 for i in range(1, len(recent_highs))
            if recent_highs[i]["price"] > recent_highs[i-1]["price"]
        )
        hl_count = sum(
            1 for i in range(1, len(recent_lows))
            if recent_lows[i]["price"] > recent_lows[i-1]["price"]
        )
        lh_count = sum(
            1 for i in range(1, len(recent_highs))
            if recent_highs[i]["price"] < recent_highs[i-1]["price"]
        )
        ll_count = sum(
            1 for i in range(1, len(recent_lows))
            if recent_lows[i]["price"] < recent_lows[i-1]["price"]
        )

        structure = "sideways"
        if hh_count >= 1 and hl_count >= 1 and hh_count >= lh_count:
            structure = "uptrend"
        elif lh_count >= 1 and ll_count >= 1 and lh_count >= hh_count:
            structure = "downtrend"

        # BOS detection: latest close broke most recent swing high (uptrend) or swing low (downtrend)
        last_close = float(df["close"].iloc[-1])
        bos_detected = False
        choch_detected = False

        last_swing_high = swing_highs[-1]["price"] if swing_highs else None
        last_swing_low = swing_lows[-1]["price"] if swing_lows else None

        if structure == "uptrend":
            # BOS = close above last swing high
            if last_swing_high and last_close > last_swing_high:
                bos_detected = True
            # CHOCH = close below last swing low
            if last_swing_low and last_close < last_swing_low:
                choch_detected = True
        elif structure == "downtrend":
            # BOS = close below last swing low
            if last_swing_low and last_close < last_swing_low:
                bos_detected = True
            # CHOCH = close above last swing high
            if last_swing_high and last_close > last_swing_high:
                choch_detected = True

        return TrendInfo(
            structure=structure,
            bos_detected=bos_detected,
            choch_detected=choch_detected,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            note=f"HH={hh_count}, HL={hl_count}, LH={lh_count}, LL={ll_count}",
        )

    # ═══════════════════════════════════════════════════════════
    # STEP 3: SHOOTING STAR 2-CANDLE RULE
    # ═══════════════════════════════════════════════════════════

    def _step3_shooting_star(
        self, df: pd.DataFrame, trend: TrendInfo, atr_val: float
    ) -> dict:
        """Detect 2-candle shooting star reversal pattern."""
        result = {
            "detected": False,
            "candle1_confirmed": False,
            "candle2_seller_pressure_confirmed": False,
        }

        if len(df) < 2:
            return result

        # Check last 2 candles
        c1 = df.iloc[-2]
        c2 = df.iloc[-1]

        # Step 3.1: 1st candle must be shooting star (bearish rejection)
        # AND trend must be downtrend OR price at resistance
        c1_is_ss = _is_shooting_star(c1)
        if c1_is_ss:
            result["candle1_confirmed"] = True

        # Step 3.2: 2nd candle must confirm seller pressure
        # = bearish candle (close < open) AND close in lower half of range
        o2, h2, l2, cl2 = (float(c2["open"]), float(c2["high"]),
                           float(c2["low"]), float(c2["close"]))
        c2_bearish = cl2 < o2
        range2 = h2 - l2
        c2_close_lower = (cl2 - l2) < (range2 / 2) if range2 > 0 else False
        c2_downward_momentum = (o2 - cl2) > atr_val * 0.3

        if c2_bearish and c2_close_lower and c2_downward_momentum:
            result["candle2_seller_pressure_confirmed"] = True

        # Setup detected only if BOTH candles confirm
        if result["candle1_confirmed"] and result["candle2_seller_pressure_confirmed"]:
            result["detected"] = True

        return result

    # ═══════════════════════════════════════════════════════════
    # STEP 5: MULTI-TIMEFRAME CONFIRMATION
    # ═══════════════════════════════════════════════════════════

    def _step5_mtf_confirmation(
        self,
        symbol: str,
        lower_tf_df: Optional[pd.DataFrame],
        primary_bias: str,
        primary_trend: TrendInfo,
    ) -> dict:
        """Lower timeframe confirmation."""
        lower_tf = LOWER_TF_MAP.get(self.timeframe)
        if lower_tf is None:
            # 1D has no lower TF confirmation required
            return {"lower_tf_used": "null", "aligned": True}

        if lower_tf_df is None or len(lower_tf_df) < MIN_CANDLES_REQUIRED:
            return {"lower_tf_used": lower_tf, "aligned": False}

        # Run trend analysis on lower TF
        lower_trend = self._step2_trend_structure(lower_tf_df)

        # Also run S/R bias on lower TF
        try:
            lower_sr_result = self.sr_engine.analyze(lower_tf_df, symbol=symbol)
            lower_atr_series = _atr(lower_tf_df, period=14)
            lower_atr = float(lower_atr_series.iloc[-1]) if not lower_atr_series.empty else None
            if lower_atr is None:
                lower_bias = "neutral"
            else:
                _, lower_bias = self._step1_sr_zones_and_bias(lower_tf_df, symbol, lower_atr)
        except Exception as e:
            log.debug(f"[multi_strategy_pa_engine.py] lower-TF ATR/bias suppressed: {e}")
            lower_bias = "neutral"

        # Aligned if: trend structure matches primary AND bias matches
        aligned = (
            lower_trend.structure == primary_trend.structure
            and (lower_bias == primary_bias or primary_bias == "neutral")
        )

        return {"lower_tf_used": lower_tf, "aligned": bool(aligned)}

    # ═══════════════════════════════════════════════════════════
    # STEP 6: SUPPLY/DEMAND ZONES (Momentum Candle-based)
    # ═══════════════════════════════════════════════════════════

    def _step6_supply_demand(self, df: pd.DataFrame) -> List[SDZone]:
        """
        Find 3 consecutive momentum candles (same direction) → mark base candle(s)
        before them as Supply (bearish momentum) or Demand (bullish momentum).

        If ANY of the 3 candles is a "baby" candle → cluster is INVALID.
        """
        sd_zones: List[SDZone] = []
        n = len(df)
        if n < MOMENTUM_CANDLE_RUN + 1:
            return sd_zones

        for i in range(MOMENTUM_CANDLE_RUN, n):
            # Look at 3 consecutive candles ending at i
            run = df.iloc[i - MOMENTUM_CANDLE_RUN + 1 : i + 1]

            # Check if all 3 are momentum candles AND same direction AND not baby
            all_momentum = True
            all_same_dir = True
            direction = None
            for idx, c in run.iterrows():
                if not _is_momentum_candle(c) or _is_baby_candle(c):
                    all_momentum = False
                    break
                c_dir = "bullish" if c["close"] > c["open"] else "bearish"
                if direction is None:
                    direction = c_dir
                elif c_dir != direction:
                    all_same_dir = False
                    break

            if not all_momentum or not all_same_dir:
                continue

            # Base candle = candle BEFORE the run
            base_idx = i - MOMENTUM_CANDLE_RUN
            if base_idx < 0:
                continue
            base = df.iloc[base_idx]
            base_high = float(base["high"])
            base_low = float(base["low"])

            if direction == "bullish":
                # 3 bullish momentum candles → demand zone at base
                sd_zones.append(SDZone(
                    type="demand",
                    zone_top=base_high,
                    zone_bottom=base_low,
                    momentum_candles_confirmed=True,
                ))
            else:
                # 3 bearish momentum candles → supply zone at base
                sd_zones.append(SDZone(
                    type="supply",
                    zone_top=base_high,
                    zone_bottom=base_low,
                    momentum_candles_confirmed=True,
                ))

        # Deduplicate (keep most recent)
        seen = set()
        unique_zones = []
        for z in reversed(sd_zones):
            key = (round(z.zone_top, 4), round(z.zone_bottom, 4), z.type)
            if key not in seen:
                seen.add(key)
                unique_zones.append(z)
        return unique_zones[:5]

    # ═══════════════════════════════════════════════════════════
    # STEP 7: CONFLUENCE SCORING
    # ═══════════════════════════════════════════════════════════

    def _step7_confluence(
        self,
        sr_zones: List[SRZone],
        sd_zones: List[SDZone],
        trend: TrendInfo,
        symbol: str,
        df: pd.DataFrame,
        atr_val: float,
    ) -> Optional[ConfluenceZone]:
        """
        Find the zone with highest confluence score.
        Factors:
          1. S/R zone
          2. Trendline (wick-based)
          3. Supply/Demand zone
          4. Higher timeframe zone (we use longer-lookback S/R as proxy)
          5. Round number
          6. Fibonacci level (we use 50% of recent swing as proxy)
        """
        if not sr_zones and not sd_zones:
            return None

        # Build candidate price levels from all zones
        candidates: List[float] = []
        for z in sr_zones:
            candidates.append((z.zone_top + z.zone_bottom) / 2)
        for z in sd_zones:
            candidates.append((z.zone_top + z.zone_bottom) / 2)

        # Add trendline approximation (use linear fit of swing lows/highs)
        # Simple proxy: latest swing low or high
        if trend.swing_lows:
            candidates.append(trend.swing_lows[-1]["price"])
        if trend.swing_highs:
            candidates.append(trend.swing_highs[-1]["price"])

        # Add round numbers near current price
        current_price = float(df["close"].iloc[-1])
        pip = _pip_value(symbol)
        for offset_mult in [-50, -25, 0, 25, 50]:
            rn = round(current_price / (50 * pip)) * (50 * pip) + offset_mult * pip
            candidates.append(rn)

        # Add Fibonacci 50% of recent swing
        if trend.swing_highs and trend.swing_lows:
            swing_high = trend.swing_highs[-1]["price"]
            swing_low = trend.swing_lows[-1]["price"]
            fib_50 = (swing_high + swing_low) / 2
            candidates.append(fib_50)

        # For each candidate, count how many factors are within ATR band
        best_zone = None
        best_score = 0
        for center in candidates:
            factors = []
            band = atr_val * 1.5  # confluence band

            # Factor 1: S/R zone
            for z in sr_zones:
                zc = (z.zone_top + z.zone_bottom) / 2
                if abs(zc - center) <= band:
                    factors.append(f"SR_{z.type}")
                    break

            # Factor 2: Trendline (use swing high/low as proxy)
            for sw in trend.swing_highs + trend.swing_lows:
                if abs(sw["price"] - center) <= band:
                    factors.append("Trendline")
                    break

            # Factor 3: Supply/Demand zone
            for z in sd_zones:
                zc = (z.zone_top + z.zone_bottom) / 2
                if abs(zc - center) <= band:
                    factors.append(f"SD_{z.type}")
                    break

            # Factor 4: Higher TF zone (use S/R with ≥4 touches as proxy)
            for z in sr_zones:
                if z.touches >= 4:
                    zc = (z.zone_top + z.zone_bottom) / 2
                    if abs(zc - center) <= band:
                        factors.append("HTF_zone")
                        break

            # Factor 5: Round number
            if _is_round_number(center, symbol):
                factors.append("Round_number")

            # Factor 6: Fibonacci
            if trend.swing_highs and trend.swing_lows:
                swing_high = trend.swing_highs[-1]["price"]
                swing_low = trend.swing_lows[-1]["price"]
                fib_50 = (swing_high + swing_low) / 2
                if abs(fib_50 - center) <= band:
                    factors.append("Fibonacci_50")

            score = len(factors)
            if score > best_score:
                best_score = score
                level = ("High" if score >= 5
                         else "Medium" if score >= 3
                         else "Low")
                best_zone = ConfluenceZone(
                    zone_top=center + atr_val * 0.3,
                    zone_bottom=center - atr_val * 0.3,
                    confluence_score=score,
                    confluence_level=level,
                    factors=factors,
                )

        return best_zone

    # ═══════════════════════════════════════════════════════════
    # STEP 8: CONFIRMATION CHECKLIST
    # ═══════════════════════════════════════════════════════════

    def _step8_checklist(
        self,
        df: pd.DataFrame,
        sr_zones: List[SRZone],
        sd_zones: List[SDZone],
        trend: TrendInfo,
        mtf_info: dict,
        confluence_zone: Optional[ConfluenceZone],
        ss_setup: dict,
    ) -> dict:
        """6-factor checklist — need ≥3 to pass.

        Per spec rule 4 (High-Reliability Pattern Library):
          - candlestick_pattern factor uses HighReliabilityPatternDetector
            (Hammer, Shooting Star, Engulfing, Morning/Evening Star, etc.)
          - candle_behavior factor uses pattern detector's Reversal-type
            patterns with High reliability (near zone)
        """
        # ── Build unified zone list for pattern confluence check ──
        unified_zones_for_patterns = []
        for z in sr_zones:
            unified_zones_for_patterns.append({
                "type": z.type.capitalize(),  # "Support" or "Resistance"
                "zone_top": z.zone_top,
                "zone_bottom": z.zone_bottom,
            })
        for z in sd_zones:
            unified_zones_for_patterns.append({
                "type": z.type.capitalize(),  # "Supply" or "Demand"
                "zone_top": z.zone_top,
                "zone_bottom": z.zone_bottom,
            })

        # ── Run HighReliabilityPatternDetector ──
        detected_patterns = []
        if self.pattern_detector is not None:
            try:
                atr_val = _atr(df)
                detected_patterns = self.pattern_detector.detect(
                    df, zones=unified_zones_for_patterns, atr_value=atr_val
                )
            except Exception as e:
                detected_patterns = []

        # 1. Candlestick pattern: ANY high-reliability pattern detected
        #    (Hammer, Shooting Star, Engulfing, Star patterns, etc.)
        #    Falls back to legacy engulfing/shooting-star detection if no patterns found.
        candlestick_pattern = (
            len(detected_patterns) > 0
            or ss_setup.get("detected", False)
            or self._detect_engulfing(df)
        )

        # 2. Chart pattern (double top/bottom, H&S — simple proxy: ≥2 touches on same side)
        chart_pattern = any(z.touches >= 2 for z in sr_zones)

        # 3. Candle behavior: Reversal-type pattern with High reliability (near zone)
        #    Per spec: "rejection wick, momentum shift" → patterns near zone
        candle_behavior = (
            any(p.type == "Reversal" and p.reliability == "High" for p in detected_patterns)
            or self._detect_rejection_wick(df)
        )

        # 4. Confluence level (Medium or High)
        confluence_level = (
            confluence_zone is not None
            and confluence_zone.confluence_level in ("Medium", "High")
        )

        # 5. Trendline confluence (trend is established + price near trendline)
        trendline_confluence = (
            trend.structure in ("uptrend", "downtrend")
            and len(trend.swing_highs) >= 2
            and len(trend.swing_lows) >= 2
        )

        # 6. Multi-timeframe alignment
        multi_tf_alignment = mtf_info.get("aligned", False)

        total = sum([
            candlestick_pattern, chart_pattern, candle_behavior,
            confluence_level, trendline_confluence, multi_tf_alignment,
        ])

        return {
            "candlestick_pattern":  bool(candlestick_pattern),
            "chart_pattern":        bool(chart_pattern),
            "candle_behavior":      bool(candle_behavior),
            "confluence_level":     bool(confluence_level),
            "trendline_confluence": bool(trendline_confluence),
            "multi_tf_alignment":   bool(multi_tf_alignment),
            "total_confirmed":      int(total),
        }

    def _detect_engulfing(self, df: pd.DataFrame) -> bool:
        """Detect bullish or bearish engulfing in last 3 candles."""
        if len(df) < 2:
            return False
        for i in range(-2, 0):
            c1 = df.iloc[i - 1]
            c2 = df.iloc[i]
            c1_body = abs(float(c1["close"]) - float(c1["open"]))
            c2_body = abs(float(c2["close"]) - float(c2["open"]))
            if c1_body < 1e-9 or c2_body < 1e-9:
                continue
            # Bullish engulfing
            if (c1["close"] < c1["open"] and c2["close"] > c2["open"]
                and c2["close"] >= c1["open"] and c2["open"] <= c1["close"]):
                return True
            # Bearish engulfing
            if (c1["close"] > c1["open"] and c2["close"] < c2["open"]
                and c2["open"] >= c1["close"] and c2["close"] <= c1["open"]):
                return True
        return False

    def _detect_rejection_wick(self, df: pd.DataFrame) -> bool:
        """Detect long rejection wick in last 3 candles."""
        if len(df) < 3:
            return False
        for i in range(-3, 0):
            c = df.iloc[i]
            o, h, l, cl = (float(c["open"]), float(c["high"]),
                           float(c["low"]), float(c["close"]))
            body = abs(cl - o)
            if body < 1e-9:
                continue
            upper_wick = h - max(o, cl)
            lower_wick = min(o, cl) - l
            if upper_wick >= body * 1.5 or lower_wick >= body * 1.5:
                return True
        return False

    # ═══════════════════════════════════════════════════════════
    # FINAL SIGNAL GENERATION
    # ═══════════════════════════════════════════════════════════

    def _generate_signal(
        self,
        symbol: str,
        timeframe: str,
        session_ok: bool,
        trend: TrendInfo,
        sr_zones: List[SRZone],
        sr_bias: str,
        ss_setup: dict,
        mtf_info: dict,
        confluence_zone: Optional[ConfluenceZone],
        checklist: dict,
        atr_val: float,
        df: pd.DataFrame,
    ) -> dict:
        """Final 8-gate signal generation."""
        # ── Gate 1: Session time ──
        if not session_ok:
            return self._no_trade_signal(
                "NO_TRADE (Outside trading window 11:00-22:00 BD Time)"
            )

        # ── Gate 2: Trend structure ──
        if trend.structure == "sideways":
            return self._wait_signal(
                f"Step 2: Sideways/consolidation trend — WAIT. ({trend.note})"
            )

        # ── Gate 3: Trend-bias alignment ──
        # Uptrend → BUY bias expected; Downtrend → SELL bias expected.
        # Was a hard NO_TRADE on any conflict. Softened per review: a
        # conflict here usually just means price is at an S/R zone that
        # opposes the higher-timeframe trend (e.g. a pullback to
        # resistance inside an uptrend) — informative, not necessarily
        # fatal. We still trade with the trend, but only when the rest of
        # the checklist is unusually strong, and confidence is capped so
        # this can never present as a High-confidence signal.
        bias_conflict = (
            (trend.structure == "uptrend" and sr_bias == "sell")
            or (trend.structure == "downtrend" and sr_bias == "buy")
        )
        if bias_conflict and checklist["total_confirmed"] < MIN_CHECKLIST_PASSED + 2:
            return self._no_trade_signal(
                f"Step 1+2: {trend.structure} but {sr_bias.upper()} bias — conflict, and "
                f"checklist ({checklist['total_confirmed']}/6) not strong enough to override"
            )

        # Determine action based on trend
        action = "BUY" if trend.structure == "uptrend" else "SELL"

        # ── Gate 4: Zone touch ≥2 ──
        # Already enforced in _step1 (min_touches=2). Verify zones exist near price.
        if not sr_zones:
            return self._no_trade_signal("Step 1: No S/R zones detected")

        # ── Gate 5: Confirmation checklist ≥3 ──
        if checklist["total_confirmed"] < MIN_CHECKLIST_PASSED:
            return self._no_trade_signal(
                f"Step 1 checklist: only {checklist['total_confirmed']}/6 factors confirmed "
                f"(need ≥{MIN_CHECKLIST_PASSED})"
            )

        # ── Gate 6: Confluence ≥ Medium ──
        if confluence_zone is None or confluence_zone.confluence_level == "Low":
            level = confluence_zone.confluence_level if confluence_zone else "None"
            return self._no_trade_signal(
                f"Step 7: Confluence level = {level} (need Medium/High)"
            )

        # ── Gate 7: MTF confirmation ──
        if not mtf_info.get("aligned", False):
            return self._no_trade_signal(
                f"Step 5: Lower timeframe ({mtf_info.get('lower_tf_used')}) confirmation failed"
            )

        # ── Compute entry, SL, TP ──
        current_price = float(df["close"].iloc[-1])
        sl_buffer = atr_val * 0.15

        if action == "BUY":
            # Entry at current price; SL below recent swing low or confluence zone bottom
            entry_price = current_price
            recent_lows = [s["price"] for s in trend.swing_lows[-3:]] if trend.swing_lows else []
            sl_anchor = min(recent_lows) if recent_lows else (confluence_zone.zone_bottom - sl_buffer)
            stop_loss = sl_anchor - sl_buffer
            # TP = nearest resistance zone OR confluence zone + min R:R
            res_above = [z for z in sr_zones if z.type == "resistance" and z.zone_bottom > entry_price]
            if res_above:
                take_profit = min(z.zone_bottom for z in res_above)
            else:
                # Fallback: 1:2 R:R
                risk = entry_price - stop_loss
                take_profit = entry_price + risk * MIN_RR_RATIO
        else:  # SELL
            entry_price = current_price
            # If shooting star setup detected, use 1st candle's upper wick for SL
            if ss_setup.get("detected"):
                # SL = shooting star upper wick + buffer
                c1 = df.iloc[-2]
                stop_loss = float(c1["high"]) + sl_buffer
            else:
                recent_highs = [s["price"] for s in trend.swing_highs[-3:]] if trend.swing_highs else []
                sl_anchor = max(recent_highs) if recent_highs else (confluence_zone.zone_top + sl_buffer)
                stop_loss = sl_anchor + sl_buffer
            # TP = nearest support zone
            sup_below = [z for z in sr_zones if z.type == "support" and z.zone_top < entry_price]
            if sup_below:
                take_profit = max(z.zone_top for z in sup_below)
            else:
                risk = stop_loss - entry_price
                take_profit = entry_price - risk * MIN_RR_RATIO

        # R:R check
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr = round(reward / risk, 2) if risk > 0 else None

        # Confidence
        score = checklist["total_confirmed"]
        if confluence_zone and confluence_zone.confluence_level == "High":
            score += 2
        elif confluence_zone and confluence_zone.confluence_level == "Medium":
            score += 1
        if trend.bos_detected:
            score += 1
        if rr and rr >= MIN_RR_RATIO * 2:
            score += 1
        confidence = "High" if score >= 7 else "Medium" if score >= 4 else "Low"
        if bias_conflict and confidence == "High":
            # Trend/bias conflict overrides only ever exit as Medium at best —
            # the checklist strength that let it through doesn't erase the
            # fact that price structure and S/R bias disagree.
            confidence = "Medium"

        reason = (
            f"{'Trend/bias conflict override — ' if bias_conflict else 'All gates passed: '}"
            f"trend={trend.structure}, "
            f"bias={sr_bias}, checklist={checklist['total_confirmed']}/6, "
            f"confluence={confluence_zone.confluence_level} ({confluence_zone.confluence_score} factors), "
            f"MTF={mtf_info.get('lower_tf_used')} aligned, "
            f"session=OK, R:R=1:{rr}. "
            f"{'Shooting star 2-candle confirmed.' if ss_setup.get('detected') else ''}"
        )

        return {
            "action":                action,
            "entry_price":           round(float(entry_price), 5),
            "stop_loss":             round(float(stop_loss), 5),
            "take_profit_suggested": round(float(take_profit), 5),
            "risk_reward":           rr,
            "reason":                reason,
            "confidence":            confidence,
        }

    # ═══════════════════════════════════════════════════════════
    # HELPERS — output assembly
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _no_trade_signal(reason: str) -> dict:
        return {
            "action":                "NO_TRADE",
            "entry_price":           None,
            "stop_loss":             None,
            "take_profit_suggested": None,
            "risk_reward":           None,
            "reason":                reason,
            "confidence":            "Low",
        }

    @staticmethod
    def _wait_signal(reason: str) -> dict:
        return {
            "action":                "WAIT",
            "entry_price":           None,
            "stop_loss":             None,
            "take_profit_suggested": None,
            "risk_reward":           None,
            "reason":                reason,
            "confidence":            "Low",
        }

    @staticmethod
    def _no_trade_result(symbol: str, timeframe: str, reason: str) -> dict:
        return {
            "pair": symbol,
            "timeframe": timeframe,
            "session_time_ok": False,
            "trend": {"structure": "sideways", "bos_detected": False, "choch_detected": False},
            "zones": {
                "support_resistance": [],
                "supply_demand": [],
                "strongest_confluence_zone": None,
            },
            "shooting_star_setup": {
                "detected": False,
                "candle1_confirmed": False,
                "candle2_seller_pressure_confirmed": False,
            },
            "multi_timeframe_confirmation": {"lower_tf_used": "null", "aligned": False},
            "confirmation_checklist": {
                "candlestick_pattern": False, "chart_pattern": False,
                "candle_behavior": False, "confluence_level": False,
                "trendline_confluence": False, "multi_tf_alignment": False,
                "total_confirmed": 0,
            },
            "signal": MultiStrategyPAEngine._no_trade_signal(reason),
        }

    @staticmethod
    def _build_result(
        symbol: str,
        timeframe: str,
        session_ok: bool,
        trend: TrendInfo,
        sr_zones: List[SRZone],
        sd_zones: List[SDZone],
        confluence_zone: Optional[ConfluenceZone],
        ss_setup: dict,
        mtf_info: dict,
        checklist: dict,
        signal: dict,
    ) -> dict:
        """Build spec-compliant JSON-ready dict."""
        return {
            "pair":            symbol,
            "timeframe":       timeframe,
            "session_time_ok": bool(session_ok),
            "trend": {
                "structure":      trend.structure,
                "bos_detected":   bool(trend.bos_detected),
                "choch_detected": bool(trend.choch_detected),
            },
            "zones": {
                "support_resistance": [z.to_spec_dict() for z in sr_zones[:5]],
                "supply_demand":      [z.to_spec_dict() for z in sd_zones[:5]],
                "strongest_confluence_zone": (
                    {
                        "zone_top":         round(float(confluence_zone.zone_top), 5),
                        "zone_bottom":      round(float(confluence_zone.zone_bottom), 5),
                        "confluence_score": int(confluence_zone.confluence_score),
                        "confluence_level": confluence_zone.confluence_level,
                    } if confluence_zone else None
                ),
            },
            "shooting_star_setup": {
                "detected":                                bool(ss_setup["detected"]),
                "candle1_confirmed":                       bool(ss_setup["candle1_confirmed"]),
                "candle2_seller_pressure_confirmed":       bool(ss_setup["candle2_seller_pressure_confirmed"]),
            },
            "multi_timeframe_confirmation": {
                "lower_tf_used": mtf_info.get("lower_tf_used", "null"),
                "aligned":       bool(mtf_info.get("aligned", False)),
            },
            "confirmation_checklist": {
                "candlestick_pattern":  bool(checklist["candlestick_pattern"]),
                "chart_pattern":        bool(checklist["chart_pattern"]),
                "candle_behavior":      bool(checklist["candle_behavior"]),
                "confluence_level":     bool(checklist["confluence_level"]),
                "trendline_confluence": bool(checklist["trendline_confluence"]),
                "multi_tf_alignment":   bool(checklist["multi_tf_alignment"]),
                "total_confirmed":      int(checklist["total_confirmed"]),
            },
            "signal": signal,
        }

    # ═══════════════════════════════════════════════════════════
    # LLM-FRIENDLY OUTPUT
    # ═══════════════════════════════════════════════════════════

    def analyze_to_json(
        self, df: pd.DataFrame, symbol: str, lower_tf_df: Optional[pd.DataFrame] = None
    ) -> str:
        return json.dumps(self.analyze(df, symbol, lower_tf_df), ensure_ascii=False, indent=2)

    def to_prompt_text(self, result: dict) -> str:
        lines = [
            f"=== MULTI-STRATEGY PA SIGNAL ({result['pair']} {result['timeframe']}) ===",
            f"Session OK: {result['session_time_ok']}",
            "",
            f"-- Trend --",
            f"  Structure: {result['trend']['structure']}",
            f"  BOS: {result['trend']['bos_detected']} | CHOCH: {result['trend']['choch_detected']}",
            "",
            "-- Zones --",
        ]
        for z in result["zones"]["support_resistance"][:3]:
            lines.append(f"  {z['type']}: [{z['zone_bottom']} - {z['zone_top']}] touches={z['touches']}")
        for z in result["zones"]["supply_demand"][:3]:
            lines.append(f"  {z['type']}: [{z['zone_bottom']} - {z['zone_top']}] momentum_confirmed={z['momentum_candles_confirmed']}")
        cz = result["zones"]["strongest_confluence_zone"]
        if cz:
            lines.append(f"  Confluence: [{cz['zone_bottom']} - {cz['zone_top']}] score={cz['confluence_score']} level={cz['confluence_level']}")

        ss = result["shooting_star_setup"]
        lines.append("")
        lines.append("-- Shooting Star --")
        lines.append(f"  Detected: {ss['detected']} | C1: {ss['candle1_confirmed']} | C2: {ss['candle2_seller_pressure_confirmed']}")

        mtf = result["multi_timeframe_confirmation"]
        lines.append("")
        lines.append(f"-- MTF ({mtf['lower_tf_used']}) --")
        lines.append(f"  Aligned: {mtf['aligned']}")

        chk = result["confirmation_checklist"]
        lines.append("")
        lines.append("-- Checklist --")
        lines.append(f"  Candlestick: {chk['candlestick_pattern']} | Chart: {chk['chart_pattern']}")
        lines.append(f"  Behavior: {chk['candle_behavior']} | Confluence: {chk['confluence_level']}")
        lines.append(f"  Trendline: {chk['trendline_confluence']} | MTF: {chk['multi_tf_alignment']}")
        lines.append(f"  Total: {chk['total_confirmed']}/6")

        sig = result["signal"]
        lines.append("")
        lines.append("-- Signal --")
        lines.append(f"  Action: {sig['action']} | Entry: {sig['entry_price']}")
        lines.append(f"  SL: {sig['stop_loss']} | TP: {sig['take_profit_suggested']} | R:R: 1:{sig['risk_reward']}")
        lines.append(f"  Confidence: {sig['confidence']}")
        lines.append(f"  Reason: {sig['reason']}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# Convenience: one-shot helper
# ============================================================

def detect_multi_strategy_pa_signal(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "4H",
    lower_tf_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> str:
    """
    One-shot helper — returns spec-compliant JSON.
    """
    engine = MultiStrategyPAEngine(timeframe=timeframe, **kwargs)
    return engine.analyze_to_json(df, symbol, lower_tf_df)


# ============================================================
# CLI entry
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-06-03 06:00", periods=n, freq="4h")  # 4H candles
    base = 1.0850
    close = base + np.cumsum(np.random.randn(n) * 0.0008)
    df = pd.DataFrame({
        "open":  close + np.random.randn(n) * 0.0003,
        "high":  close + abs(np.random.randn(n)) * 0.0012,
        "low":   close - abs(np.random.randn(n)) * 0.0012,
        "close": close,
    }, index=dates)

    engine = MultiStrategyPAEngine(timeframe="4H")
    result = engine.analyze(df, symbol="EURUSD")
    print(json.dumps(result, indent=2))