# analysis/ict_amd_signal_engine.py
# ============================================================
# ICT / Smart Money Concept — AMD + FVG + MSS Signal Engine
# ============================================================
# Spec compliance: 6-step ICT/SMC pipeline producing exact JSON.
#
# STEP 1: Accumulation range (Asian session) + zone strength
#   - Find Asian session high/low → Accumulation Range
#   - Validate range is "tight" (low volatility, ATR-relative)
#   - Detect all S/R zones (Strong/Medium/Weak)
#   - Identify strongest_zone + weakest_zone
#   - Trades ONLY from Strong/Medium zones (Weak = informational only)
#
# STEP 2: Manipulation / Stop Hunt (Judas Swing)
#   - London session sweep of Accumulation Range or Strong/Medium zone
#   - Wick pierces zone, body closes inside/near, wick ≥ 1.5× body
#   - 1–3 candles later price reverses
#   - Weak zone sweeps → manipulation_detected = false (not reliable)
#
# STEP 3: FVG (Fair Value Gap) within manipulation move
#   - 3-candle imbalance: c1.high < c3.low (bullish) OR c1.low > c3.high (bearish)
#   - Record top, bottom, 50% midpoint (Consequent Encroachment)
#   - FVG direction must be opposite of manipulation (i.e., in distribution direction)
#
# STEP 4: Market Structure Shift (MSS) confirmation
#   - After manipulation, price must break an opposite-direction swing point
#   - No MSS → no entry
#
# STEP 5: Take Profit + R:R filter
#   - TP = nearest Strong zone in distribution direction (Weak zones NOT used as TP)
#   - If nearest Strong zone fails R:R ≥ 1:6, try 2nd-nearest Strong zone
#   - If still fails → NO_TRADE / "R:R ratio 1:6 criteria পূরণ করেনি"
#
# STEP 6: Final entry signal (Distribution / NY session)
#   - All conditions must pass: accumulation_valid + manipulation_detected
#     + valid FVG + mss_confirmed + risk_reward ≥ 1:6
#   - Otherwise NO_TRADE with clear reason
#
# Output: exact spec JSON (no preamble)
# ============================================================

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from analysis.support_resistance import SupportResistance

log = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────
ASIAN_SESSION_START = 0     # 00:00 GMT
ASIAN_SESSION_END   = 6     # 06:00 GMT
LONDON_SESSION_START = 6    # 06:00 GMT
LONDON_SESSION_END   = 9    # 09:00 GMT
NY_SESSION_START    = 12    # 12:00 GMT
NY_SESSION_END      = 17    # 17:00 GMT

ACCUMULATION_MAX_ATR_MULT = 1.2   # Asian range ATR must be ≤ 1.2× overall median ATR
ACCUMULATION_MIN_CANDLES  = 4     # need ≥4 Asian session candles
WICK_BODY_RATIO_MIN       = 1.5   # spec: wick ≥ 1.5× body
MANIPULATION_REVERSAL_LOOKBACK = 3
FVG_MIN_GAP_ATR_MULT      = 0.10  # gap must be ≥ 0.10×ATR
MIN_RR_RATIO              = 6.0   # spec: minimum 1:6 R:R (strict!)
MIN_CANDLES_REQUIRED      = 30
MSS_LOOKBACK              = 10    # candles after manipulation to confirm MSS


# ─── Helpers (shared ATR imported from _engine_utils) ─────────
from analysis._engine_utils import atr_series as _atr


def _filter_by_session(df: pd.DataFrame, start_hr: int, end_hr: int) -> pd.DataFrame:
    """Filter DataFrame to candles whose hour is in [start_hr, end_hr)."""
    try:
        if df.empty or not hasattr(df.index, "hour"):
            return df
        hours = df.index.hour
        return df[(hours >= start_hr) & (hours < end_hr)]
    except Exception as e:
        return df.iloc[0:0]


def _strength_to_confidence(zone_strength: str, has_fvg: bool,
                             wick_body_ratio: float, rr_ratio: float) -> str:
    """Map confluence factors → confidence Low/Medium/High."""
    score = 0
    if zone_strength == "Strong":
        score += 2
    elif zone_strength == "Medium":
        score += 1
    if has_fvg:
        score += 1
    if wick_body_ratio >= 2.0:
        score += 1
    if rr_ratio >= 10.0:
        score += 1
    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


# ─── Dataclasses for structured results ───────────────────────

@dataclass
class ZoneInfo:
    """Single zone dict matching spec output schema."""
    type: str             # "support" | "resistance"
    zone_top: float
    zone_bottom: float
    touches: int
    strength: str = "Weak"  # kept for internal use; spec output omits this

    def to_spec_dict(self, include_strength: bool = False) -> dict:
        d = {
            "type":        self.type,
            "zone_top":    round(float(self.zone_top), 5),
            "zone_bottom": round(float(self.zone_bottom), 5),
            "touches":     int(self.touches),
        }
        if include_strength:
            d["strength"] = self.strength
        return d


@dataclass
class AccumulationResult:
    valid: bool = False
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    range_width_pct: Optional[float] = None
    asian_candle_count: int = 0
    note: str = ""

    def to_spec_dict(self) -> dict:
        return {
            "valid":      bool(self.valid),
            "range_high": round(float(self.range_high), 5) if self.range_high is not None else None,
            "range_low":  round(float(self.range_low), 5) if self.range_low is not None else None,
        }


@dataclass
class ManipulationResult:
    detected: bool = False
    direction: str = "null"        # "upside_sweep" | "downside_sweep" | "null"
    sweep_price: Optional[float] = None
    zone_strength_used: str = "null"   # "Strong" | "Medium" | "null"
    wick_body_ratio: float = 0.0
    break_index: int = -1
    confirm_index: int = -1
    wick_extreme: Optional[float] = None
    note: str = ""

    def to_spec_dict(self) -> dict:
        return {
            "detected":            bool(self.detected),
            "direction":           self.direction,
            "sweep_price":         round(float(self.sweep_price), 5) if self.sweep_price is not None else None,
            "zone_strength_used":  self.zone_strength_used if self.detected else "null",
        }


@dataclass
class FVGResult:
    found: bool = False
    type: str = "null"             # "bullish" | "bearish" | "null"
    top: Optional[float] = None
    bottom: Optional[float] = None
    midpoint: Optional[float] = None
    index: int = -1
    note: str = ""

    def to_spec_dict(self) -> dict:
        return {
            "found":     bool(self.found),
            "type":      self.type,
            "top":       round(float(self.top), 5) if self.top is not None else None,
            "bottom":    round(float(self.bottom), 5) if self.bottom is not None else None,
            "midpoint":  round(float(self.midpoint), 5) if self.midpoint is not None else None,
        }


# ─── Main Engine ──────────────────────────────────────────────

# ─── Global regime hard-gate helper (institutional review fix) ───
# Shared pattern with analysis/mtf_analyzer.py and
# analysis/multi_strategy_pa_engine.py: consumes
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
            f"ICT setup may otherwise be valid, but the broader regime says "
            f"this is not a tradeable market right now."
        )
    return None


class ICTAMDSignalEngine:
    """
    ICT/SMC AMD + FVG + MSS Signal Engine — spec-compliant.

    Usage:
        engine = ICTAMDSignalEngine(timeframe="H1")
        result = engine.analyze(df, symbol="EURUSD")
        print(json.dumps(result, indent=2))   # exact spec JSON
    """

    def __init__(
        self,
        timeframe: str = "H1",
        swing_window: Optional[int] = None,
        cluster_threshold_pct: Optional[float] = None,
        min_touches: int = 2,
        wick_body_ratio: float = WICK_BODY_RATIO_MIN,
        min_rr_ratio: float = MIN_RR_RATIO,
    ):
        self.timeframe = timeframe
        self.wick_body_ratio = wick_body_ratio
        self.min_rr_ratio = min_rr_ratio

        self.sr_engine = SupportResistance(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
            wick_body_ratio=wick_body_ratio,
            max_zones_per_side=10,  # get more zones for TP selection
        )

    # ═══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════

    def analyze(self, df: pd.DataFrame, symbol: str = "", regime_ctx: dict = None) -> dict:
        """Run full 6-step ICT/SMC pipeline. Returns spec-compliant dict.

        Args:
            df, symbol: as before.
            regime_ctx: optional output of
                `MarketRegimeDetector.get_ai_context()` (analysis/market_regime.py).
                When supplied and it reports a non-tradeable regime (CHOPPY,
                `strategy_type == "NO_TRADE"`, or `risk_multiplier <= 0`), the
                final signal is forced to NO_TRADE regardless of how many of
                the 6 ICT steps passed. This closes the integration gap
                flagged in the institutional review — a regime filter that
                isn't enforced as a hard gate on every signal engine isn't
                protecting capital. Omitting this argument preserves the
                exact prior behavior (backward compatible).
        """
        # ── Edge case: insufficient data ──
        if df is None or len(df) < MIN_CANDLES_REQUIRED:
            return self._build_result(
                zones={"strongest_zone": None, "weakest_zone": None},
                accumulation=AccumulationResult(note="Insufficient session/candle data"),
                manipulation=ManipulationResult(),
                fvg=FVGResult(),
                mss_confirmed=False,
                signal=self._no_trade_signal("Insufficient session/candle data"),
            )

        # ── NEW: global market-regime hard gate (checked first — cheapest,
        # and per the review board's #1 integration-gap finding, this should
        # short-circuit before spending effort on the 6-step pipeline) ──
        regime_block = _regime_blocks_trading(regime_ctx)
        if regime_block is not None:
            return self._build_result(
                zones={"strongest_zone": None, "weakest_zone": None},
                accumulation=AccumulationResult(note="Skipped: blocked by market regime gate"),
                manipulation=ManipulationResult(),
                fvg=FVGResult(),
                mss_confirmed=False,
                signal=self._no_trade_signal(regime_block),
            )

        # Compute ATR for various checks
        atr_series = _atr(df, period=14)
        atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
        if not np.isfinite(atr_val) or atr_val <= 0:
            atr_val = float(df["close"].iloc[-1]) * 0.001

        # ── STEP 1: Zones + Accumulation ──
        zones_all, strongest, weakest = self._step1_zones(df, symbol)
        accumulation = self._step1_accumulation(df, atr_series, atr_val)

        # ── STEP 2: Manipulation (only at Strong/Medium zones or Asian range) ──
        manipulation = self._step2_manipulation(
            df, zones_all, accumulation, atr_val, symbol
        )

        # ── STEP 3: FVG (within manipulation move, distribution direction) ──
        fvg = self._step3_fvg(df, manipulation, atr_val)

        # ── STEP 4: MSS confirmation ──
        mss_confirmed, mss_note = self._step4_mss(df, manipulation, atr_val)

        # ── STEP 5 + 6: Signal (R:R ≥ 1:6 from nearest Strong zone) ──
        signal = self._step5_6_signal(
            df, zones_all, accumulation, manipulation, fvg, mss_confirmed,
            mss_note, atr_val, symbol
        )

        return self._build_result(
            zones={"strongest_zone": strongest, "weakest_zone": weakest},
            accumulation=accumulation,
            manipulation=manipulation,
            fvg=fvg,
            mss_confirmed=mss_confirmed,
            signal=signal,
        )

    # ═══════════════════════════════════════════════════════════
    # STEP 1: ZONES + ACCUMULATION
    # ═══════════════════════════════════════════════════════════

    def _step1_zones(
        self, df: pd.DataFrame, symbol: str
    ) -> Tuple[List[ZoneInfo], Optional[ZoneInfo], Optional[ZoneInfo]]:
        """Detect all S/R zones. Identify strongest + weakest."""
        try:
            sr_result = self.sr_engine.analyze(df, symbol=symbol)
        except Exception as e:
            log.error(f"[ICT] S/R analyze failed: {e}")
            return [], None, None

        zones_all: List[ZoneInfo] = []
        for z in sr_result.get("resistance_zones", []):
            zones_all.append(ZoneInfo(
                type="resistance",
                zone_top=z["zone_top"],
                zone_bottom=z["zone_bottom"],
                touches=z["touches"],
                strength=z["strength"],
            ))
        for z in sr_result.get("support_zones", []):
            zones_all.append(ZoneInfo(
                type="support",
                zone_top=z["zone_top"],
                zone_bottom=z["zone_bottom"],
                touches=z["touches"],
                strength=z["strength"],
            ))

        if not zones_all:
            return [], None, None

        # Strongest = max touches (and Strong strength)
        # Weakest = min touches (typically Weak strength)
        strength_rank = {"Strong": 3, "Medium": 2, "Weak": 1}
        sorted_by_strength = sorted(
            zones_all,
            key=lambda z: (strength_rank.get(z.strength, 0), z.touches),
            reverse=True,
        )
        strongest = sorted_by_strength[0] if sorted_by_strength else None
        weakest = sorted_by_strength[-1] if sorted_by_strength else None

        return zones_all, strongest, weakest

    def _step1_accumulation(
        self, df: pd.DataFrame, atr_series: pd.Series, atr_val: float
    ) -> AccumulationResult:
        """Asian session accumulation range detection + tight-range validation."""
        asian_df = _filter_by_session(df, ASIAN_SESSION_START, ASIAN_SESSION_END)

        if len(asian_df) < ACCUMULATION_MIN_CANDLES:
            return AccumulationResult(
                valid=False,
                note=f"Insufficient Asian candles ({len(asian_df)} < {ACCUMULATION_MIN_CANDLES})",
            )

        range_high = float(asian_df["high"].max())
        range_low = float(asian_df["low"].min())
        range_width_pct = (range_high - range_low) / range_low if range_low > 0 else 0

        # Tight-range validation: ATR during Asian session must be ≤ 1.2× overall ATR
        asian_atr = atr_series.loc[asian_df.index].mean() if not asian_df.empty else np.nan
        if not np.isfinite(asian_atr) or asian_atr <= 0:
            asian_atr = atr_val

        # Also check that range width is reasonable (not > 1.5% of price for FX)
        max_range_pct = 0.015  # 1.5% max for "tight" range
        is_tight = (
            asian_atr <= ACCUMULATION_MAX_ATR_MULT * atr_val
            and range_width_pct <= max_range_pct
        )

        return AccumulationResult(
            valid=is_tight,
            range_high=range_high,
            range_low=range_low,
            range_width_pct=range_width_pct,
            asian_candle_count=len(asian_df),
            note=(
                f"Asian range [{range_low:.5f} – {range_high:.5f}] "
                f"width={range_width_pct*100:.2f}%, ATR_ratio={asian_atr/atr_val:.2f}"
                if is_tight
                else f"Asian range too wide (width={range_width_pct*100:.2f}% or ATR={asian_atr/atr_val:.2f}×)"
            ),
        )

    # ═══════════════════════════════════════════════════════════
    # STEP 2: MANIPULATION (Stop Hunt / Judas Swing)
    # ═══════════════════════════════════════════════════════════

    def _step2_manipulation(
        self,
        df: pd.DataFrame,
        zones_all: List[ZoneInfo],
        accumulation: AccumulationResult,
        atr_val: float,
        symbol: str,
    ) -> ManipulationResult:
        """
        Scan London session for stop-hunt on:
          (a) Accumulation range high/low (if accumulation valid), OR
          (b) Strong/Medium zones only

        Weak zone sweeps → manipulation_detected = False (not reliable).
        """
        london_df = _filter_by_session(df, LONDON_SESSION_START, LONDON_SESSION_END)

        # If no session info, fall back to last 20 candles
        scan_df = london_df if len(london_df) >= 3 else df.tail(20)
        if len(scan_df) < 3:
            return ManipulationResult(note="Insufficient London candles")

        # Build list of "valid sweep targets" = Strong/Medium zones + accumulation range
        sweep_targets: List[dict] = []
        for z in zones_all:
            if z.strength in ("Strong", "Medium"):
                sweep_targets.append({
                    "type": z.type,
                    "zone_top": z.zone_top,
                    "zone_bottom": z.zone_bottom,
                    "strength": z.strength,
                    "is_accumulation": False,
                })

        # Add accumulation range as sweep targets (if valid)
        if accumulation.valid:
            sweep_targets.append({
                "type": "resistance",
                "zone_top": accumulation.range_high,
                "zone_bottom": accumulation.range_high - atr_val * 0.3,  # treat as thin zone
                "strength": "Strong",  # accumulation range treated as strong
                "is_accumulation": True,
                "label": "accumulation_high",
            })
            sweep_targets.append({
                "type": "support",
                "zone_top": accumulation.range_low + atr_val * 0.3,
                "zone_bottom": accumulation.range_low,
                "strength": "Strong",
                "is_accumulation": True,
                "label": "accumulation_low",
            })

        if not sweep_targets:
            return ManipulationResult(note="No Strong/Medium zones or accumulation range")

        # Get scan_df indices in original df
        scan_indices = scan_df.index
        # Build a position lookup: position in scan_df → position in df
        df_index_to_pos = {ts: i for i, ts in enumerate(df.index)}

        best_event = None
        best_score = -1

        for target in sweep_targets:
            event = self._check_sweep_at_target(df, target, atr_val, scan_indices)
            if event.detected:
                # Score: Strong zone > Medium > accumulation
                strength_rank = {"Strong": 3, "Medium": 2, "Weak": 1}
                score = strength_rank.get(target["strength"], 0) * 10 + event.wick_body_ratio
                if score > best_score:
                    best_score = score
                    best_event = event

        return best_event if best_event else ManipulationResult(
            note="No sweep detected at Strong/Medium zones or accumulation range"
        )

    def _check_sweep_at_target(
        self,
        df: pd.DataFrame,
        target: dict,
        atr_val: float,
        scan_indices: pd.DatetimeIndex,
    ) -> ManipulationResult:
        """Check if any candle in scan window swept this target."""
        zone_top = float(target["zone_top"])
        zone_bottom = float(target["zone_bottom"])
        zone_width = max(zone_top - zone_bottom, 1e-9)
        # ATR-based "near" band (handles tight + wide zones)
        near_band = max(zone_width * 0.5, atr_val * 0.5)

        n = len(df)
        highs = df["high"].values
        lows = df["low"].values
        opens = df["open"].values
        closes = df["close"].values

        # Iterate over scan indices (positions in df)
        for ts in scan_indices:
            if ts not in df.index:
                continue
            i = df.index.get_loc(ts)
            if i >= n - 1:  # need room for confirmation candles
                continue

            o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
            body = abs(c - o)
            if body < 1e-9:
                continue

            # Check for UPSIDE sweep (price wicked above zone_top)
            if target["type"] == "resistance" and h > zone_top:
                upper_wick = h - max(o, c)
                wick_body_ratio = upper_wick / body
                # Body must close inside or near zone
                body_closes_near = (
                    zone_bottom - near_band <= c <= zone_top
                )
                if not body_closes_near:
                    continue
                if wick_body_ratio < self.wick_body_ratio:
                    continue
                # Look for reversal: 1–3 candles later, close below zone_bottom
                confirm_idx = self._find_reversal(
                    closes, i + 1, n, "below", zone_bottom
                )
                if confirm_idx is None:
                    continue
                # Direction: upside_sweep → distribution is DOWN
                return ManipulationResult(
                    detected=True,
                    direction="upside_sweep",
                    sweep_price=h,
                    zone_strength_used=target["strength"],
                    wick_body_ratio=wick_body_ratio,
                    break_index=i,
                    confirm_index=confirm_idx,
                    wick_extreme=h,
                    note=(
                        f"Upside sweep at {target['type']} zone "
                        f"[{zone_bottom:.5f}–{zone_top:.5f}] ({target['strength']}). "
                        f"Wick/body={wick_body_ratio:.2f}. Distribution direction: DOWN (SELL)."
                    ),
                )

            # Check for DOWNSIDE sweep (price wicked below zone_bottom)
            if target["type"] == "support" and l < zone_bottom:
                lower_wick = min(o, c) - l
                wick_body_ratio = lower_wick / body
                body_closes_near = (
                    zone_bottom <= c <= zone_top + near_band
                )
                if not body_closes_near:
                    continue
                if wick_body_ratio < self.wick_body_ratio:
                    continue
                # Look for reversal: 1–3 candles later, close above zone_top
                confirm_idx = self._find_reversal(
                    closes, i + 1, n, "above", zone_top
                )
                if confirm_idx is None:
                    continue
                # Direction: downside_sweep → distribution is UP
                return ManipulationResult(
                    detected=True,
                    direction="downside_sweep",
                    sweep_price=l,
                    zone_strength_used=target["strength"],
                    wick_body_ratio=wick_body_ratio,
                    break_index=i,
                    confirm_index=confirm_idx,
                    wick_extreme=l,
                    note=(
                        f"Downside sweep at {target['type']} zone "
                        f"[{zone_bottom:.5f}–{zone_top:.5f}] ({target['strength']}). "
                        f"Wick/body={wick_body_ratio:.2f}. Distribution direction: UP (BUY)."
                    ),
                )

        return ManipulationResult()

    def _find_reversal(
        self,
        closes: np.ndarray,
        start_idx: int,
        n: int,
        direction: str,
        threshold: float,
    ) -> Optional[int]:
        """Within MANIPULATION_REVERSAL_LOOKBACK candles, find close past threshold."""
        end_idx = min(n, start_idx + MANIPULATION_REVERSAL_LOOKBACK)
        for j in range(start_idx, end_idx):
            c = float(closes[j])
            if direction == "above" and c > threshold:
                return j
            if direction == "below" and c < threshold:
                return j
        return None

    # ═══════════════════════════════════════════════════════════
    # STEP 3: FVG DETECTION
    # ═══════════════════════════════════════════════════════════

    def _step3_fvg(
        self,
        df: pd.DataFrame,
        manipulation: ManipulationResult,
        atr_val: float,
    ) -> FVGResult:
        """
        Find FVG within manipulation move. FVG direction must match distribution
        direction (opposite of manipulation):
          upside_sweep → distribution is DOWN → look for BEARISH FVG
          downside_sweep → distribution is UP → look for BULLISH FVG
        """
        if not manipulation.detected:
            return FVGResult(note="No manipulation detected")

        # Expected FVG type
        if manipulation.direction == "upside_sweep":
            expected_fvg_type = "bearish"
        elif manipulation.direction == "downside_sweep":
            expected_fvg_type = "bullish"
        else:
            return FVGResult(note="Unknown manipulation direction")

        # Scan from manipulation break_index to confirm_index + 3 candles
        start_i = manipulation.break_index
        end_i = min(len(df), manipulation.confirm_index + 3)
        highs = df["high"].values
        lows = df["low"].values

        for i in range(max(2, start_i), end_i):
            c1_high, c1_low = float(highs[i - 2]), float(lows[i - 2])
            c3_high, c3_low = float(highs[i]), float(lows[i])

            if expected_fvg_type == "bullish":
                # Bullish FVG: c1.high < c3.low → gap [c1.high, c3.low]
                if c3_low > c1_high:
                    gap = c3_low - c1_high
                    if gap >= atr_val * FVG_MIN_GAP_ATR_MULT:
                        midpoint = (c1_high + c3_low) / 2
                        return FVGResult(
                            found=True,
                            type="bullish",
                            top=c3_low,
                            bottom=c1_high,
                            midpoint=midpoint,
                            index=i,
                            note=f"Bullish FVG [{c1_high:.5f} – {c3_low:.5f}] mid={midpoint:.5f}",
                        )
            else:  # bearish
                # Bearish FVG: c1.low > c3.high → gap [c3.high, c1.low]
                if c3_high < c1_low:
                    gap = c1_low - c3_high
                    if gap >= atr_val * FVG_MIN_GAP_ATR_MULT:
                        midpoint = (c1_low + c3_high) / 2
                        return FVGResult(
                            found=True,
                            type="bearish",
                            top=c1_low,
                            bottom=c3_high,
                            midpoint=midpoint,
                            index=i,
                            note=f"Bearish FVG [{c3_high:.5f} – {c1_low:.5f}] mid={midpoint:.5f}",
                        )

        return FVGResult(note=f"No {expected_fvg_type} FVG found in manipulation move")

    # ═══════════════════════════════════════════════════════════
    # STEP 4: MARKET STRUCTURE SHIFT (MSS)
    # ═══════════════════════════════════════════════════════════

    def _step4_mss(
        self,
        df: pd.DataFrame,
        manipulation: ManipulationResult,
        atr_val: float,
    ) -> Tuple[bool, str]:
        """
        After manipulation, check if price broke an opposite-direction
        swing point (local high/low) within MSS_LOOKBACK candles.

        upside_sweep → look for break of nearest swing LOW (price makes lower low)
        downside_sweep → look for break of nearest swing HIGH (price makes higher high)
        """
        if not manipulation.detected:
            return False, "No manipulation to confirm MSS"

        n = len(df)
        highs = df["high"].values
        lows = df["low"].values
        confirm_idx = manipulation.confirm_index

        # Look back from manipulation start to find the pre-manipulation swing
        # in the OPPOSITE direction of the sweep.
        # upside_sweep → look for the nearest swing LOW before manipulation (price went up, then came down)
        # → MSS = price breaks BELOW that swing low after manipulation
        # downside_sweep → look for nearest swing HIGH → MSS = price breaks ABOVE that swing high

        search_start = max(0, manipulation.break_index - MSS_LOOKBACK * 2)
        search_end = manipulation.break_index

        if manipulation.direction == "upside_sweep":
            # Pre-manipulation swing low → MSS = price goes below it
            swing_low = float(np.min(lows[search_start:search_end])) if search_end > search_start else None
            if swing_low is None:
                return False, "No pre-manipulation swing low to confirm MSS"
            # Check if any candle after confirm_idx closes below swing_low
            for j in range(confirm_idx, min(n, confirm_idx + MSS_LOOKBACK)):
                if float(df["close"].iloc[j]) < swing_low:
                    return True, f"MSS confirmed: price closed below swing low {swing_low:.5f}"
            return False, f"Price did not break swing low {swing_low:.5f} (no MSS)"

        elif manipulation.direction == "downside_sweep":
            swing_high = float(np.max(highs[search_start:search_end])) if search_end > search_start else None
            if swing_high is None:
                return False, "No pre-manipulation swing high to confirm MSS"
            for j in range(confirm_idx, min(n, confirm_idx + MSS_LOOKBACK)):
                if float(df["close"].iloc[j]) > swing_high:
                    return True, f"MSS confirmed: price closed above swing high {swing_high:.5f}"
            return False, f"Price did not break swing high {swing_high:.5f} (no MSS)"

        return False, "Unknown manipulation direction"

    # ═══════════════════════════════════════════════════════════
    # STEP 5 + 6: SIGNAL GENERATION
    # ═══════════════════════════════════════════════════════════

    def _step5_6_signal(
        self,
        df: pd.DataFrame,
        zones_all: List[ZoneInfo],
        accumulation: AccumulationResult,
        manipulation: ManipulationResult,
        fvg: FVGResult,
        mss_confirmed: bool,
        mss_note: str,
        atr_val: float,
        symbol: str,
    ) -> dict:
        """Generate final signal with all 6-step checks + R:R ≥ 1:6 filter."""
        # ── Sequential checks ──
        if not accumulation.valid:
            return self._no_trade_signal(
                f"Step 1 failed: {accumulation.note}"
            )

        if not manipulation.detected:
            return self._no_trade_signal(
                f"Step 2 failed: {manipulation.note or 'No manipulation detected at Strong/Medium zone'}"
            )

        if not fvg.found:
            return self._no_trade_signal(
                f"Step 3 failed: {fvg.note or 'No valid FVG in distribution direction'}"
            )

        if not mss_confirmed:
            return self._no_trade_signal(
                f"Step 4 failed: {mss_note or 'MSS not confirmed'}"
            )

        # ── Compute entry, SL, TP ──
        # Entry = FVG midpoint (Consequent Encroachment — classic ICT entry)
        # OR open of candle after MSS confirmation
        # We use FVG midpoint as primary entry (more precise)
        entry_price = fvg.midpoint

        # SL = wick extreme + buffer
        sl_buffer = atr_val * 0.15
        if manipulation.direction == "upside_sweep":
            # Distribution is DOWN → SELL
            # SL above wick extreme
            stop_loss = manipulation.wick_extreme + sl_buffer
            action = "SELL"
            # TP = nearest Strong SUPPORT zone below entry
            strong_supports = [
                z for z in zones_all
                if z.type == "support" and z.strength == "Strong" and z.zone_top < entry_price
            ]
            strong_supports.sort(key=lambda z: z.zone_top, reverse=True)  # nearest first
        else:  # downside_sweep → BUY
            stop_loss = manipulation.wick_extreme - sl_buffer
            action = "BUY"
            # TP = nearest Strong RESISTANCE zone above entry
            strong_resistances = [
                z for z in zones_all
                if z.type == "resistance" and z.strength == "Strong" and z.zone_bottom > entry_price
            ]
            strong_resistances.sort(key=lambda z: z.zone_bottom)  # nearest first

        # Geometry sanity
        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            return self._no_trade_signal("Invalid entry/SL geometry")

        # ── Try Strong zones as TP (Step 5 rule 4: try 2nd nearest if 1st fails R:R) ──
        candidate_tps = strong_supports if action == "SELL" else strong_resistances
        take_profit = None
        rr_ratio = 0.0

        for zone in candidate_tps:
            if action == "SELL":
                tp_candidate = zone.zone_top   # TP at top of support zone (conservative)
                if tp_candidate >= entry_price:
                    continue
                reward = entry_price - tp_candidate
            else:  # BUY
                tp_candidate = zone.zone_bottom  # TP at bottom of resistance zone
                if tp_candidate <= entry_price:
                    continue
                reward = tp_candidate - entry_price

            rr = reward / risk
            if rr >= self.min_rr_ratio:
                take_profit = tp_candidate
                rr_ratio = rr
                break

        # If no Strong zone TP meets R:R ≥ 1:6 → NO_TRADE
        if take_profit is None:
            return self._no_trade_signal(
                f"Step 5 failed: R:R ratio 1:6 criteria পূরণ করেনি "
                f"(no Strong zone TP ≥ 1:{self.min_rr_ratio:.0f}; "
                f"nearest candidates: {[z.zone_top if action=='SELL' else z.zone_bottom for z in candidate_tps[:3]]})"
            )

        # ── Final signal (Step 6) ──
        confidence = _strength_to_confidence(
            manipulation.zone_strength_used,
            fvg.found,
            manipulation.wick_body_ratio,
            rr_ratio,
        )

        reason = (
            f"All steps passed: "
            f"accumulation({'valid'}), "
            f"manipulation({manipulation.direction} @ {manipulation.zone_strength_used} zone, "
            f"wick/body={manipulation.wick_body_ratio:.2f}), "
            f"FVG({fvg.type} @ {fvg.midpoint:.5f}), "
            f"MSS(confirmed), "
            f"R:R=1:{rr_ratio:.1f} ≥ 1:{self.min_rr_ratio:.0f}. "
            f"Entry at FVG midpoint, SL beyond sweep wick, TP at nearest Strong zone."
        )

        return {
            "action":      action,
            "entry_price": round(float(entry_price), 5),
            "stop_loss":   round(float(stop_loss), 5),
            "take_profit": round(float(take_profit), 5),
            "risk_reward": round(float(rr_ratio), 2),
            "reason":      reason,
            "confidence":  confidence,
        }

    # ═══════════════════════════════════════════════════════════
    # HELPERS — output assembly
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _no_trade_signal(reason: str) -> dict:
        return {
            "action":      "NO_TRADE",
            "entry_price": None,
            "stop_loss":   None,
            "take_profit": None,
            "risk_reward": None,
            "reason":      reason,
            "confidence":  "Low",
        }

    @staticmethod
    def _build_result(
        zones: dict,
        accumulation: AccumulationResult,
        manipulation: ManipulationResult,
        fvg: FVGResult,
        mss_confirmed: bool,
        signal: dict,
    ) -> dict:
        """Build spec-compliant JSON-ready dict."""
        strongest = zones.get("strongest_zone")
        weakest = zones.get("weakest_zone")
        return {
            "zones": {
                "strongest_zone": strongest.to_spec_dict() if strongest else None,
                "weakest_zone":   weakest.to_spec_dict() if weakest else None,
            },
            "accumulation":  accumulation.to_spec_dict(),
            "manipulation":  manipulation.to_spec_dict(),
            "fvg":           fvg.to_spec_dict(),
            "mss_confirmed": bool(mss_confirmed),
            "signal":        signal,
        }

    # ═══════════════════════════════════════════════════════════
    # LLM-FRIENDLY OUTPUT
    # ═══════════════════════════════════════════════════════════

    def analyze_to_json(self, df: pd.DataFrame, symbol: str = "") -> str:
        """Run analyze() and return JSON string."""
        return json.dumps(self.analyze(df, symbol=symbol), ensure_ascii=False, indent=2)

    def to_prompt_text(self, result: dict) -> str:
        """Plain-text rendering for LLM prompts."""
        lines = ["=== ICT/SMC AMD SIGNAL ==="]

        zones = result.get("zones", {})
        s = zones.get("strongest_zone")
        w = zones.get("weakest_zone")
        lines.append("")
        lines.append("-- Zones --")
        if s:
            lines.append(f"  Strongest: {s['type']} [{s['zone_bottom']} - {s['zone_top']}] touches={s['touches']}")
        else:
            lines.append("  Strongest: (none)")
        if w:
            lines.append(f"  Weakest  : {w['type']} [{w['zone_bottom']} - {w['zone_top']}] touches={w['touches']}")
        else:
            lines.append("  Weakest  : (none)")

        acc = result.get("accumulation", {})
        lines.append("")
        lines.append("-- Accumulation (Asian session) --")
        lines.append(f"  Valid      : {acc.get('valid')}")
        lines.append(f"  Range High : {acc.get('range_high')}")
        lines.append(f"  Range Low  : {acc.get('range_low')}")

        man = result.get("manipulation", {})
        lines.append("")
        lines.append("-- Manipulation (London sweep) --")
        lines.append(f"  Detected        : {man.get('detected')}")
        lines.append(f"  Direction       : {man.get('direction')}")
        lines.append(f"  Sweep Price     : {man.get('sweep_price')}")
        lines.append(f"  Zone Strength   : {man.get('zone_strength_used')}")

        fvg = result.get("fvg", {})
        lines.append("")
        lines.append("-- FVG --")
        lines.append(f"  Found     : {fvg.get('found')}")
        lines.append(f"  Type      : {fvg.get('type')}")
        lines.append(f"  Top       : {fvg.get('top')}")
        lines.append(f"  Bottom    : {fvg.get('bottom')}")
        lines.append(f"  Midpoint  : {fvg.get('midpoint')}")

        lines.append("")
        lines.append(f"MSS Confirmed: {result.get('mss_confirmed')}")

        sig = result.get("signal", {})
        lines.append("")
        lines.append("-- Signal --")
        lines.append(f"  Action     : {sig.get('action')}")
        lines.append(f"  Entry      : {sig.get('entry_price')}")
        lines.append(f"  Stop Loss  : {sig.get('stop_loss')}")
        lines.append(f"  Take Profit: {sig.get('take_profit')}")
        lines.append(f"  R:R        : 1:{sig.get('risk_reward')}")
        lines.append(f"  Confidence : {sig.get('confidence')}")
        lines.append(f"  Reason     : {sig.get('reason')}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# Convenience: one-shot helper for LLM Agent integration
# ============================================================

def detect_ict_amd_signal(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "H1",
    **kwargs,
) -> str:
    """
    One-shot helper — pass OHLC df, get spec-compliant JSON back.

    Returns JSON string with the EXACT schema from the spec:
      {
        "zones": {strongest_zone, weakest_zone},
        "accumulation": {valid, range_high, range_low},
        "manipulation": {detected, direction, sweep_price, zone_strength_used},
        "fvg": {found, type, top, bottom, midpoint},
        "mss_confirmed": bool,
        "signal": {action, entry_price, stop_loss, take_profit, risk_reward, reason, confidence}
      }
    """
    engine = ICTAMDSignalEngine(timeframe=timeframe, **kwargs)
    return engine.analyze_to_json(df, symbol=symbol)


# ============================================================
# CLI entry — quick smoke test
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-06-03", periods=n, freq="h")  # Monday
    base = 1.0850
    close = base + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "open":  close + np.random.randn(n) * 0.0002,
        "high":  close + abs(np.random.randn(n)) * 0.0008,
        "low":   close - abs(np.random.randn(n)) * 0.0008,
        "close": close,
    }, index=dates)

    engine = ICTAMDSignalEngine(timeframe="H1")
    result = engine.analyze(df, symbol="EURUSD")
    print(json.dumps(result, indent=2))