"""
analysis/curve_mtf.py — Book 5 (Frank Miller S&D) Chapter 12 Multi-Frame "Curve"
=================================================================================

Pages 126-135 implement the book's most quantitatively rich methodology:
the **"curve"** — a top-down MTF filter that divides the price range
between the nearest demand and supply zones (on the higher timeframe)
into three sub-zones (High / Equilibrium / Low) and uses the current
price's position within the curve to set a directional bias.

  ── CORE DEFINITIONS (Book P130) ──────────────────────────────
    curve = price range bounded by:
      lower edge = proximal line of nearest DEMAND zone
      upper edge = proximal line of nearest SUPPLY zone

  ── CURVE SPLIT (Book P131, baseline) ─────────────────────────
    subzone_width = (upper_proximal - lower_proximal) / 3
    boundaries at:
      lower_proximal + 1 × subzone_width  (Low/Equilibrium border)
      lower_proximal + 2 × subzone_width  (Equilibrium/High border)

    Alternative (Book P132): set Fib tool to 33% and 66% between
    curve extremes — produces identical divisions. The standard
    Fibonacci retracement levels (23.6/38.2/61.8/78.6) are NOT used.

  ── ENGINE EXTENSIONS (institutional review, not in the book) ──
    The book gives a static, equal-thirds split. A production system
    trading the curve needs a few things the book doesn't specify;
    these are clearly separated from the book's rules below:
      • adaptive split ratios by market regime (§ADAPTIVE SPLIT)
      • an ATR-scaled boundary buffer to stop single-tick zone flips
      • a confidence score instead of a bare directional label
      • zone-freshness weighting (a 6-month-old zone != today's zone)
      • premium/discount weighting and a confluence score hook
      • an explicit staleness check so a caller (the orchestrator)
        knows when to rebuild the curve instead of silently reusing
        a stale one — the curve object itself stays a pure, stateless
        value type; "dynamic" behavior belongs to the caller.

  ── 5 ZONE AREAS (Book P131) ──────────────────────────────────
    Very Low  = inside the demand zone (below lower_proximal)
    Low       = [lower_proximal, low/eq boundary)
    Equilibrium = [low/eq boundary, eq/high boundary)
    High      = [eq/high boundary, upper_proximal)
    Very High = inside the supply zone (above upper_proximal)

  ── BIAS RULE (Book P133) ─────────────────────────────────────
    price in Very Low / Low  → BUY_ONLY
    price in Very High / High → SELL_ONLY
    price in Equilibrium     → TREND_FOLLOW_OR_NO_TRADE

  ── HTF OVERRIDE (Book P135) ──────────────────────────────────
    "The longer frame always wins."
    final_bias = higher_timeframe_bias
    Lower-timeframe signals are only actionable if they AGREE with
    the higher-timeframe bias. If they conflict, WAIT.

  ── TRADING STYLE → TIMEFRAME TRIPLET (Book P129) ─────────────
    Scalper  : 15m (long) / 5m (medium) / 1m (short)
    Day      : 1d  (long) / 4h (medium) / 1h (short)
    Swing    : 1w  (long) / 1d (medium) / 4h (short)
    Position : 1M  (long) / 1w (medium) / 1d (short)

Usage:
    from analysis.curve_mtf import CurveMTF, TradingStyle

    curve = CurveMTF.from_zones(
        nearest_demand={"proximal": 1.0800, "zone_low": 1.0790, "zone_high": 1.0810,
                         "state": "fresh", "candles_ago": 12, "quality_score": 82},
        nearest_supply={"proximal": 1.0900, "zone_low": 1.0890, "zone_high": 1.0910,
                         "state": "tested", "candles_ago": 40, "quality_score": 61},
        current_price=1.0820,
        timeframe="1d",
        atr=0.0015,               # optional — enables the boundary buffer
        regime_ctx={"regime": "TRENDING", "direction": "BULLISH"},  # optional — adaptive split
    )
    bias = curve.bias_for(1.0820)          # → DirectionalBias.BUY_ONLY
    conf = curve.confidence_for(1.0820)    # → CurveConfidence(score=..., reason=...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger("curve_mtf")


# ════════════════════════════════════════════════════════════════
#  TRADING STYLE → TIMEFRAME TRIPLET (Book P129)
# ════════════════════════════════════════════════════════════════

class TradingStyle(str, Enum):
    """Book P127-128: four trading styles by holding period."""
    SCALPER  = "scalper"
    DAY      = "day"
    SWING    = "swing"
    POSITION = "position"


# Book P129: recommended 3-timeframe combination per style
# Format: (long_tf, medium_tf, short_tf)
TIMEFRAME_TRIPLET: Dict[TradingStyle, Tuple[str, str, str]] = {
    TradingStyle.SCALPER:  ("15m", "5m", "1m"),
    TradingStyle.DAY:      ("1d",  "4h", "1h"),
    TradingStyle.SWING:    ("1w",  "1d", "4h"),
    TradingStyle.POSITION: ("1M",  "1w", "1d"),
}

# Approximate trade frequency per style (Book P127-128)
TRADE_FREQUENCY_PER_DAY: Dict[TradingStyle, Tuple[int, int]] = {
    TradingStyle.SCALPER:  (10, 30),    # 10-30 trades/day
    TradingStyle.DAY:      (1, 10),     # <10 trades/day
    TradingStyle.SWING:    (0, 1),      # days-weeks holding
    TradingStyle.POSITION: (0, 1),      # months-years holding
}


def get_timeframe_triplet(style: TradingStyle) -> Tuple[str, str, str]:
    """Return the (long, medium, short) timeframe triplet for a style."""
    return TIMEFRAME_TRIPLET[style]


# ════════════════════════════════════════════════════════════════
#  ZONE POSITION ENUM (Book P131 — 5 zone areas)
# ════════════════════════════════════════════════════════════════

class CurvePosition(str, Enum):
    """Where current price sits within the curve (Book P131)."""
    VERY_LOW     = "very_low"      # inside demand zone (below lower_proximal)
    LOW          = "low"           # lowest sub-zone of curve
    EQUILIBRIUM  = "equilibrium"   # middle sub-zone
    HIGH         = "high"          # highest sub-zone
    VERY_HIGH    = "very_high"     # inside supply zone (above upper_proximal)


class DirectionalBias(str, Enum):
    """Book P133: directional bias based on curve position."""
    BUY_ONLY                 = "BUY_ONLY"
    SELL_ONLY                = "SELL_ONLY"
    TREND_FOLLOW_OR_NO_TRADE = "TREND_FOLLOW_OR_NO_TRADE"


# Map curve position → directional bias (Book P133)
POSITION_TO_BIAS: Dict[CurvePosition, DirectionalBias] = {
    CurvePosition.VERY_LOW:    DirectionalBias.BUY_ONLY,
    CurvePosition.LOW:         DirectionalBias.BUY_ONLY,
    CurvePosition.EQUILIBRIUM: DirectionalBias.TREND_FOLLOW_OR_NO_TRADE,
    CurvePosition.HIGH:        DirectionalBias.SELL_ONLY,
    CurvePosition.VERY_HIGH:   DirectionalBias.SELL_ONLY,
}

# Zone states, as produced by analysis/order_block.py and friends.
# Used purely for freshness weighting here — this module does not
# reimplement zone-state detection.
_FRESHNESS_WEIGHT: Dict[str, float] = {
    "fresh":  1.00,
    "tested": 0.55,
    "broken": 0.10,   # a broken zone is a weak curve edge; still usable
    None:     0.70,   # unknown state (caller didn't provide it) — neutral-ish
}

# § ADAPTIVE SPLIT ------------------------------------------------
# Not in the book. The book's 33/33/33 split assumes price spends
# roughly equal time in each third, which only holds in a genuinely
# ranging market. In a trending market price spends most of its time
# pushing through the continuation side of the curve and comparatively
# little time near the origin zone, so a fixed equal-thirds split
# mislabels most of the range as "Equilibrium" (no-trade) when it
# should be actionable. Ratios are (low, equilibrium, high) and must
# sum to 1.0.
_DEFAULT_SPLIT: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
_TRENDING_SPLIT_CONTINUATION: Tuple[float, float, float] = (0.20, 0.20, 0.60)
_TRENDING_SPLIT_ORIGIN: Tuple[float, float, float] = (0.60, 0.20, 0.20)


def _split_ratios_for_regime(regime_ctx: Optional[Dict[str, Any]]) -> Tuple[float, float, float]:
    """
    Choose (low_ratio, eq_ratio, high_ratio) from a regime context.

    Engine extension, not book methodology — see module docstring.
    regime_ctx is whatever analysis/market_regime.py's
    MarketRegimeDetector.detect() / get_ai_context() returns; only
    'regime' and 'direction' are read, everything else is ignored so
    this stays decoupled from that module's exact schema.

    TRENDING + BULLISH  → expand the High sub-zone (continuation side
                            is above price in an uptrend nearing supply)
    TRENDING + BEARISH  → expand the Low sub-zone
    Anything else (RANGING / CHOPPY / BREAKOUT / unknown / missing)
    → the book's equal-thirds split.
    """
    if not regime_ctx:
        return _DEFAULT_SPLIT

    regime = str(regime_ctx.get("regime", "")).upper()
    direction = str(regime_ctx.get("direction", "")).upper()

    if regime != "TRENDING":
        return _DEFAULT_SPLIT
    if direction.startswith("BULL"):
        return _TRENDING_SPLIT_CONTINUATION
    if direction.startswith("BEAR"):
        return _TRENDING_SPLIT_ORIGIN
    return _DEFAULT_SPLIT


# ════════════════════════════════════════════════════════════════
#  CONFIDENCE / EXPLAINABILITY RESULT TYPES
# ════════════════════════════════════════════════════════════════

@dataclass
class CurveConfidence:
    """
    Explainable confidence for a bias call at a given price.

    score  : 0-100
    bias   : the DirectionalBias this confidence applies to
    factors: ordered list of (label, signed_contribution) pairs that
             summed (plus the 50 base) produce `score` — this is the
             audit trail, not just a black-box number.
    reason : one-line human-readable explanation (Bengali/English
             agnostic — caller decides language; this is composed
             from `factors`).
    """
    score: float
    bias: DirectionalBias
    factors: List[Tuple[str, float]] = field(default_factory=list)
    reason: str = ""


@dataclass
class CurveQuality:
    """Structural quality of the curve itself, independent of price."""
    curve_width: float
    curve_width_atr: Optional[float]   # width expressed in ATR multiples, if atr given
    demand_freshness: float            # 0-1
    supply_freshness: float            # 0-1
    is_narrow: bool                    # width below a usable ATR multiple
    quality_score: float               # 0-100


# ════════════════════════════════════════════════════════════════
#  CURVE DATA STRUCTURE
# ════════════════════════════════════════════════════════════════

@dataclass
class Curve:
    """
    Book P130-131: the price range between nearest demand and supply
    zone proximal lines, divided into sub-zones (equal thirds by
    default, adaptively re-weighted when a regime context is supplied
    at build time — see `_split_ratios_for_regime`).
    """
    # Source zone proximal lines (the curve's edges)
    demand_proximal: float      # lower edge (Book P130)
    supply_proximal: float      # upper edge (Book P130)

    # The 4 internal boundaries (Book P131, or adaptive — see above)
    very_low_top:    float      # = demand_proximal (top of demand zone)
    low_top:         float      # = demand_proximal + low_ratio * curve_width
    equilibrium_top: float      # = low_top + eq_ratio * curve_width
    high_top:        float      # = supply_proximal (top of high subzone)

    # Derived
    subzone_width:    float     # curve_width / 3 — kept for the book's
                                 # fixed-fib helper; NOT used for position
                                 # classification when ratios are adaptive
    curve_width:      float
    split_ratios: Tuple[float, float, float] = _DEFAULT_SPLIT

    # ATR-scaled boundary buffer (engine extension, issue: static
    # boundaries flip on a single tick). 0.0 disables it (book-exact
    # behavior). See `_BUFFER_ATR_MULT`.
    boundary_buffer: float = 0.0

    # Optional: full zone info for context (freshness, quality_score,
    # candles_ago, etc. — whatever order_block.py / fvg_detector.py
    # attached; this module doesn't require any particular schema, it
    # just reads a few optional keys defensively).
    demand_zone: Optional[Dict[str, Any]] = None
    supply_zone: Optional[Dict[str, Any]] = None
    timeframe: str = "unknown"

    # Optional liquidity map (engine extension — issue: no liquidity
    # map). Populated by the caller from analysis/liquidity.py; this
    # module treats it as opaque pass-through data plus one helper
    # (`nearest_liquidity`) for convenience.
    liquidity_above: List[float] = field(default_factory=list)
    liquidity_below: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.curve_width < 0:
            raise ValueError(f"curve_width cannot be negative: {self.curve_width}")
        if self.subzone_width < 0:
            raise ValueError(f"subzone_width cannot be negative: {self.subzone_width}")
        if self.boundary_buffer < 0:
            raise ValueError(f"boundary_buffer cannot be negative: {self.boundary_buffer}")
        ratio_sum = sum(self.split_ratios)
        if self.curve_width > 0 and abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {self.split_ratios} (sum={ratio_sum})")
        # A buffer wider than the equilibrium sub-zone would make the
        # boundaries cross — clamp defensively rather than produce a
        # nonsensical/inverted classification.
        eq_width = abs(self.equilibrium_top - self.low_top)
        max_buffer = eq_width / 2.0
        if self.boundary_buffer > max_buffer:
            log.warning(
                "[curve_mtf] boundary_buffer=%.6f exceeds half the equilibrium "
                "width (%.6f) — clamping to avoid crossed boundaries",
                self.boundary_buffer, max_buffer,
            )
            self.boundary_buffer = max_buffer

    @property
    def curve_low(self) -> float:
        return min(self.demand_proximal, self.supply_proximal)

    @property
    def curve_high(self) -> float:
        return max(self.demand_proximal, self.supply_proximal)

    # ══════════════════════════════════════════════════════════
    #  POSITION CLASSIFICATION (with ATR buffer / deadband)
    # ══════════════════════════════════════════════════════════

    def position_of(self, price: float) -> CurvePosition:
        """
        Determine which sub-zone a price falls into (Book P131),
        with an optional ATR-scaled deadband applied at every boundary
        (`self.boundary_buffer`, 0.0 = book-exact / no buffer).

        Deadband design (engine extension — see module docstring):
        each boundary requires price to clear it by `boundary_buffer`
        before the *outer* classification is granted; within the
        buffer band the price still reads as the *inner* (more
        conservative) zone. This kills single-tick flips at a zone
        edge without needing the caller to track state across calls.
        """
        lo = self.curve_low
        hi = self.curve_high
        if lo == hi:
            return CurvePosition.EQUILIBRIUM

        b = self.boundary_buffer
        normal = self.demand_proximal <= self.supply_proximal
        vlow_b  = self.very_low_top if normal else self.high_top
        low_b   = self.low_top
        eq_b    = self.equilibrium_top
        vhigh_b = self.high_top if normal else self.very_low_top

        if not normal:
            # Inverted edge case (supply below demand — rare, defensive only)
            vlow_b, vhigh_b = vhigh_b, vlow_b

        if price < vlow_b - b:
            return CurvePosition.VERY_LOW
        if price < low_b + b:
            return CurvePosition.LOW
        if price < eq_b - b:
            return CurvePosition.EQUILIBRIUM
        if price < vhigh_b + b:
            return CurvePosition.HIGH
        return CurvePosition.VERY_HIGH

    def bias_for(self, price: float) -> DirectionalBias:
        """Book P133: directional bias based on curve position."""
        pos = self.position_of(price)
        return POSITION_TO_BIAS[pos]

    # ══════════════════════════════════════════════════════════
    #  PREMIUM / DISCOUNT WEIGHTING (engine extension)
    # ══════════════════════════════════════════════════════════

    def discount_pct(self, price: float) -> float:
        """
        % of the way from demand (0%) to supply (100%). <50% = discount
        (cheap relative to the range, favors buys), >50% = premium
        (expensive, favors sells). Clipped to [0, 100]; prices inside
        the demand/supply zones themselves clip to 0/100 rather than
        going negative/over 100, since "how deep into the zone" is a
        separate freshness/quality question, not a range position one.
        """
        if self.curve_width <= 0:
            return 50.0
        pct = (price - self.curve_low) / self.curve_width * 100.0
        return max(0.0, min(100.0, pct))

    def premium_pct(self, price: float) -> float:
        return 100.0 - self.discount_pct(price)

    # ══════════════════════════════════════════════════════════
    #  ZONE FRESHNESS (engine extension — issue: no zone freshness)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _zone_freshness(zone: Optional[Dict[str, Any]]) -> float:
        """
        0-1 freshness weight for a zone dict. Reads `state` if present
        (as produced by order_block.py: 'fresh'/'tested'/'broken');
        falls back to a neutral weight if the caller didn't attach one.
        Does not recompute freshness itself — this module is a
        consumer of zone state, not a detector.
        """
        if not zone:
            return _FRESHNESS_WEIGHT[None]
        state = zone.get("state")
        return _FRESHNESS_WEIGHT.get(state, _FRESHNESS_WEIGHT[None])

    def quality(self, atr: Optional[float] = None,
                min_width_atr: float = 1.0) -> CurveQuality:
        """
        Structural quality of the curve, independent of where price is.
        A curve that's too narrow relative to ATR is noise — price will
        blow through all 5 zones on a single impulse candle and every
        classification above is meaningless. `min_width_atr` is the
        floor below which the curve is flagged `is_narrow`.
        """
        demand_fresh = self._zone_freshness(self.demand_zone)
        supply_fresh = self._zone_freshness(self.supply_zone)
        width_atr = (self.curve_width / atr) if atr and atr > 0 else None
        is_narrow = bool(width_atr is not None and width_atr < min_width_atr)

        score = 50.0
        score += 25.0 * demand_fresh
        score += 25.0 * supply_fresh
        if is_narrow:
            score -= 20.0
        score = max(0.0, min(100.0, score))

        return CurveQuality(
            curve_width=self.curve_width,
            curve_width_atr=width_atr,
            demand_freshness=demand_fresh,
            supply_freshness=supply_fresh,
            is_narrow=is_narrow,
            quality_score=round(score, 1),
        )

    # ══════════════════════════════════════════════════════════
    #  CONFIDENCE + EXPLAINABILITY (engine extension — issues:
    #  no confidence, no explainability)
    # ══════════════════════════════════════════════════════════

    def confidence_for(self, price: float, atr: Optional[float] = None,
                        regime_ctx: Optional[Dict[str, Any]] = None) -> CurveConfidence:
        """
        Explainable 0-100 confidence for the bias at `price`.

        Composition (base 50, each factor bounded so the sum can't
        blow past [0, 100]):
          + up to 20 — how deep price sits into its position
                       (closer to the far edge of its own sub-zone/zone
                       = more conviction, not just barely-crossed-the-line)
          + up to 20 — freshness of the *relevant* zone (the one on the
                       side the bias points toward: demand zone for
                       BUY_ONLY, supply zone for SELL_ONLY; both zones
                       for TREND_FOLLOW_OR_NO_TRADE, since neither is
                       "the" relevant one there)
          + up to 15 — regime alignment: bias direction agrees with
                       regime_ctx['direction'], when both are given
          + up to  5 — curve quality (see `quality()`), bonus only,
                       already partly captured by freshness above so
                       weighted lightly to avoid double-counting
        """
        pos = self.position_of(price)
        bias = POSITION_TO_BIAS[pos]
        factors: List[Tuple[str, float]] = []
        score = 50.0

        # ── depth-of-penetration ──
        depth_pts = self._depth_into_position(price, pos)
        depth_contribution = 20.0 * depth_pts
        score += depth_contribution
        factors.append((f"price is {depth_pts * 100:.0f}% through the {pos.value} zone", depth_contribution))

        # ── relevant-zone freshness ──
        if bias == DirectionalBias.BUY_ONLY:
            fresh = self._zone_freshness(self.demand_zone)
            fresh_contribution = 20.0 * fresh
            factors.append((f"demand zone freshness={fresh:.2f}", fresh_contribution))
        elif bias == DirectionalBias.SELL_ONLY:
            fresh = self._zone_freshness(self.supply_zone)
            fresh_contribution = 20.0 * fresh
            factors.append((f"supply zone freshness={fresh:.2f}", fresh_contribution))
        else:
            fresh = (self._zone_freshness(self.demand_zone) + self._zone_freshness(self.supply_zone)) / 2.0
            fresh_contribution = 10.0 * fresh  # equilibrium — half weight, no clear "relevant" side
            factors.append((f"avg zone freshness={fresh:.2f} (equilibrium)", fresh_contribution))
        score += fresh_contribution

        # ── regime alignment ──
        if regime_ctx:
            direction = str(regime_ctx.get("direction", "")).upper()
            aligned = (
                (bias == DirectionalBias.BUY_ONLY and direction.startswith("BULL")) or
                (bias == DirectionalBias.SELL_ONLY and direction.startswith("BEAR"))
            )
            opposed = (
                (bias == DirectionalBias.BUY_ONLY and direction.startswith("BEAR")) or
                (bias == DirectionalBias.SELL_ONLY and direction.startswith("BULL"))
            )
            if aligned:
                factors.append((f"regime direction={direction} agrees with curve bias", 15.0))
                score += 15.0
            elif opposed:
                factors.append((f"regime direction={direction} conflicts with curve bias", -15.0))
                score -= 15.0

        # ── curve quality bonus ──
        q = self.quality(atr=atr)
        quality_contribution = (q.quality_score / 100.0) * 5.0
        score += quality_contribution
        factors.append((f"curve quality={q.quality_score:.0f}/100", quality_contribution))
        if q.is_narrow:
            factors.append(("curve is narrow relative to ATR — low reliability", -10.0))
            score -= 10.0

        score = max(0.0, min(100.0, score))
        reason = f"{bias.value} ({score:.0f}% confidence): " + "; ".join(
            f"{label} ({'+' if val >= 0 else ''}{val:.1f})" for label, val in factors
        )
        return CurveConfidence(score=round(score, 1), bias=bias, factors=factors, reason=reason)

    def _depth_into_position(self, price: float, pos: CurvePosition) -> float:
        """0-1: how far price has pushed into its current sub-zone/zone."""
        if pos == CurvePosition.EQUILIBRIUM:
            width = max(self.equilibrium_top - self.low_top, 1e-12)
            return max(0.0, min(1.0, (price - self.low_top) / width))
        if pos == CurvePosition.LOW:
            width = max(self.low_top - self.very_low_top, 1e-12)
            return max(0.0, min(1.0, (price - self.very_low_top) / width))
        if pos == CurvePosition.HIGH:
            width = max(self.high_top - self.equilibrium_top, 1e-12)
            return max(0.0, min(1.0, (price - self.equilibrium_top) / width))
        if pos == CurvePosition.VERY_LOW:
            zone = self.demand_zone or {}
            zlo = float(zone.get("zone_low", self.very_low_top - self.curve_width * 0.1))
            width = max(self.very_low_top - zlo, 1e-12)
            return max(0.0, min(1.0, (self.very_low_top - price) / width))
        # VERY_HIGH
        zone = self.supply_zone or {}
        zhi = float(zone.get("zone_high", self.high_top + self.curve_width * 0.1))
        width = max(zhi - self.high_top, 1e-12)
        return max(0.0, min(1.0, (price - self.high_top) / width))

    # ══════════════════════════════════════════════════════════
    #  CONFLUENCE SCORE (engine extension — issue: no confluence)
    # ══════════════════════════════════════════════════════════

    def confluence_score(self, price: float, ob_score: Optional[float] = None,
                          fvg_score: Optional[float] = None,
                          liquidity_score: Optional[float] = None,
                          atr: Optional[float] = None,
                          regime_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combine this curve's confidence with other detectors' scores
        (order_block.py quality_score, fvg_detector.py score, a
        liquidity-proximity score — each pre-computed by the caller,
        this module doesn't own those calculations). Any subset may be
        omitted; only supplied scores are averaged in, curve confidence
        always participates.

        This is a simple, transparent weighted mean rather than a
        learned model on purpose — it's meant to be the auditable
        input a real probability model (or the orchestrator's fusion
        engine) consumes, not a replacement for one.
        """
        curve_conf = self.confidence_for(price, atr=atr, regime_ctx=regime_ctx)
        components: List[Tuple[str, float, float]] = [("curve", curve_conf.score, 1.0)]
        if ob_score is not None:
            components.append(("order_block", float(ob_score), 1.0))
        if fvg_score is not None:
            components.append(("fvg", float(fvg_score), 0.75))
        if liquidity_score is not None:
            components.append(("liquidity", float(liquidity_score), 0.75))

        weight_sum = sum(w for _, _, w in components)
        combined = sum(s * w for _, s, w in components) / weight_sum if weight_sum else curve_conf.score

        return {
            "bias": curve_conf.bias,
            "curve_confidence": curve_conf.score,
            "components": {name: score for name, score, _ in components},
            "confluence_score": round(combined, 1),
            "reason": curve_conf.reason,
        }

    # ══════════════════════════════════════════════════════════
    #  LIQUIDITY MAP HELPER (engine extension — issue: no liquidity map)
    # ══════════════════════════════════════════════════════════

    def nearest_liquidity(self, price: float) -> Dict[str, Optional[float]]:
        """Nearest liquidity level above/below price, from the optional
        liquidity_above/liquidity_below lists the caller attached at
        build time (see `CurveMTF.from_zones(..., liquidity_above=...)`).
        Pass-through convenience only — detection lives in liquidity.py."""
        above = sorted(l for l in self.liquidity_above if l >= price)
        below = sorted((l for l in self.liquidity_below if l <= price), reverse=True)
        return {
            "above": above[0] if above else None,
            "below": below[0] if below else None,
        }

    # ══════════════════════════════════════════════════════════
    #  MISC
    # ══════════════════════════════════════════════════════════

    def describe(self) -> str:
        """Human-readable description of the curve."""
        return (
            f"Curve [{self.curve_low:.5f}, {self.curve_high:.5f}] "
            f"width={self.curve_width:.5f} sub={self.subzone_width:.5f} "
            f"split={tuple(round(r, 2) for r in self.split_ratios)} "
            f"buffer={self.boundary_buffer:.5f} "
            f"boundaries: Low/Eq={self.low_top:.5f}, Eq/High={self.equilibrium_top:.5f}"
        )


# Engine default: buffer = 10% of ATR, per the review's proposed fix
# for single-tick boundary flips. 0.0 (disabled) if no ATR is given.
_BUFFER_ATR_MULT = 0.10


# ════════════════════════════════════════════════════════════════
#  CURVE BUILDER
# ════════════════════════════════════════════════════════════════

class CurveMTF:
    """
    Book 5 Chapter 12 — Multi-Frame "Curve" methodology.

    Provides:
      • from_zones()         — build a Curve from nearest demand/supply zones
      • get_bias()           — directional bias from curve position
      • check_alignment()    — verify LTF signal agrees with HTF bias
      • resolve_conflict()   — HTF-override hierarchy (Book P135)
      • is_stale()           — detect when the orchestrator should rebuild
                                the curve against fresh zones (engine
                                extension — see "no dynamic curve" below)
    """

    @staticmethod
    def from_zones(
        nearest_demand: Dict[str, Any],
        nearest_supply: Dict[str, Any],
        current_price: float,
        timeframe: str = "1d",
        atr: Optional[float] = None,
        regime_ctx: Optional[Dict[str, Any]] = None,
        liquidity_above: Optional[List[float]] = None,
        liquidity_below: Optional[List[float]] = None,
    ) -> Curve:
        """
        Build a Curve from the nearest demand and supply zones (Book P130).

        Args:
            nearest_demand : zone dict with at least "proximal" key.
                              Optionally "state"/"candles_ago"/"quality_score"
                              (order_block.py schema) for freshness weighting.
            nearest_supply : zone dict, same shape as nearest_demand.
            current_price  : validated against the built curve (logs a
                              warning, doesn't raise, if wildly outside it —
                              a caller passing a stale price shouldn't crash
                              curve construction, just get a loud signal).
            timeframe      : label of the timeframe this curve is built on.
            atr            : optional ATR on this timeframe. Enables the
                              boundary buffer (engine extension) and the
                              curve-width-in-ATR quality check. Omit to get
                              exact book behavior (buffer=0).
            regime_ctx     : optional market_regime.py output. Enables the
                              adaptive split ratio (engine extension). Omit
                              to get the book's fixed equal-thirds split.
            liquidity_above/liquidity_below : optional level lists from
                              analysis/liquidity.py for `nearest_liquidity()`.

        Returns:
            Curve
        """
        demand_proximal = float(nearest_demand.get(
            "proximal", nearest_demand.get("zone_high",
                                           nearest_demand.get("zone_low", 0))
        ))
        supply_proximal = float(nearest_supply.get(
            "proximal", nearest_supply.get("zone_low",
                                           nearest_supply.get("zone_high", 0))
        ))

        # Ensure demand_proximal is the LOWER edge for the Curve dataclass
        if demand_proximal > supply_proximal:
            demand_proximal, supply_proximal = supply_proximal, demand_proximal

        curve_width = abs(supply_proximal - demand_proximal)
        subzone_width = curve_width / 3.0

        low_ratio, eq_ratio, high_ratio = _split_ratios_for_regime(regime_ctx)
        low_top = demand_proximal + low_ratio * curve_width
        equilibrium_top = low_top + eq_ratio * curve_width

        buffer = (atr * _BUFFER_ATR_MULT) if atr and atr > 0 else 0.0

        curve = Curve(
            demand_proximal=demand_proximal,
            supply_proximal=supply_proximal,
            very_low_top=demand_proximal,
            low_top=low_top,
            equilibrium_top=equilibrium_top,
            high_top=supply_proximal,
            subzone_width=subzone_width,
            curve_width=curve_width,
            split_ratios=(low_ratio, eq_ratio, high_ratio),
            boundary_buffer=buffer,
            demand_zone=nearest_demand,
            supply_zone=nearest_supply,
            timeframe=timeframe,
            liquidity_above=list(liquidity_above or []),
            liquidity_below=list(liquidity_below or []),
        )

        # current_price is validated, not silently discarded (issue:
        # from_zones() received but ignored current_price).
        if current_price is not None and curve_width > 0:
            lo, hi = curve.curve_low, curve.curve_high
            slack = curve_width  # one curve-width of headroom either side
            if current_price < lo - slack or current_price > hi + slack:
                log.warning(
                    "[curve_mtf] current_price=%.5f is far outside curve "
                    "[%.5f, %.5f] on %s — nearest_demand/nearest_supply may "
                    "be stale or mismatched with the quoted price",
                    current_price, lo, hi, timeframe,
                )

        return curve

    # ══════════════════════════════════════════════════════════
    #  BIAS + ALIGNMENT (Book P133, P135)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def get_bias(curve: Curve, price: float) -> DirectionalBias:
        """Book P133: directional bias from curve position."""
        return curve.bias_for(price)

    @staticmethod
    def check_alignment(
        htf_bias: DirectionalBias,
        ltf_signal: str,
    ) -> bool:
        """
        Book P135: verify a lower-timeframe signal agrees with the
        higher-timeframe bias.

        Args:
            htf_bias    : bias from the higher timeframe curve
            ltf_signal  : "long" | "short" | "neutral" from the LTF

        Returns:
            True if LTF signal agrees with HTF bias (or bias is neutral)
        """
        ltf = ltf_signal.lower()
        if htf_bias == DirectionalBias.BUY_ONLY:
            return ltf in ("long", "neutral")
        if htf_bias == DirectionalBias.SELL_ONLY:
            return ltf in ("short", "neutral")
        # TREND_FOLLOW_OR_NO_TRADE — permit either direction
        return True

    @staticmethod
    def resolve_conflict(
        htf_bias: DirectionalBias,
        ltf_signals: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """
        Book P135: HTF override hierarchy — "the longer frame always wins".

        Args:
            htf_bias     : bias from the highest timeframe
            ltf_signals  : list of (timeframe_label, signal) tuples
                           e.g. [("4h", "long"), ("1h", "short")]

        Returns:
            {
                "final_bias": DirectionalBias,
                "actionable_signals": [...],   # LTF signals that agree with HTF
                "conflicting_signals": [...],  # LTF signals that conflict
                "decision": "trade" | "wait",
                "reason": str,
            }
        """
        actionable = []
        conflicting = []

        for tf, sig in ltf_signals:
            if CurveMTF.check_alignment(htf_bias, sig):
                actionable.append((tf, sig))
            else:
                conflicting.append((tf, sig))

        # If HTF bias is TREND_FOLLOW_OR_NO_TRADE, only trade if ALL
        # LTF signals agree with each other
        if htf_bias == DirectionalBias.TREND_FOLLOW_OR_NO_TRADE:
            if not conflicting and len(set(s for _, s in ltf_signals)) <= 1:
                decision = "trade"
                reason = "Equilibrium zone — LTF signals aligned, no HTF conflict"
            else:
                decision = "wait"
                reason = "Equilibrium zone — LTF signals mixed, wait for clarity"
        elif conflicting:
            decision = "wait"
            reason = (f"HTF bias={htf_bias.value} conflicts with LTF signals "
                      f"{[t for t, _ in conflicting]} — Book P135: longer frame wins, WAIT")
        elif actionable:
            decision = "trade"
            reason = f"HTF bias={htf_bias.value} aligns with LTF signals {[t for t, _ in actionable]}"
        else:
            decision = "wait"
            reason = "No actionable LTF signals"

        return {
            "final_bias": htf_bias,
            "actionable_signals": actionable,
            "conflicting_signals": conflicting,
            "decision": decision,
            "reason": reason,
        }

    # ══════════════════════════════════════════════════════════
    #  DYNAMIC CURVE — STALENESS CHECK (engine extension)
    # ══════════════════════════════════════════════════════════
    #  curve_mtf stays a pure/stateless builder on purpose: it has no
    #  event loop, no subscription to new BOS/OB/Supply detections, and
    #  shouldn't — that's the orchestrator's (mtf_analyzer / decision
    #  engine's) job. What this module CAN do is tell the caller when
    #  a curve it built is no longer backed by the current nearest
    #  zones, so the caller knows to call from_zones() again instead of
    #  silently trading a stale curve.

    @staticmethod
    def is_stale(curve: Curve, latest_demand: Dict[str, Any],
                 latest_supply: Dict[str, Any], price_tolerance: float = 1e-9) -> bool:
        """
        True if the demand/supply zones currently reported as "nearest"
        no longer match the ones `curve` was built from — i.e. a new
        BOS/OB/Supply has been detected and the orchestrator should call
        `CurveMTF.from_zones()` again before using this curve's bias.
        """
        new_demand_proximal = float(latest_demand.get(
            "proximal", latest_demand.get("zone_high", latest_demand.get("zone_low", 0))
        ))
        new_supply_proximal = float(latest_supply.get(
            "proximal", latest_supply.get("zone_low", latest_supply.get("zone_high", 0))
        ))
        lo, hi = sorted((new_demand_proximal, new_supply_proximal))
        return (
            abs(lo - curve.curve_low) > price_tolerance or
            abs(hi - curve.curve_high) > price_tolerance
        )

    # ══════════════════════════════════════════════════════════
    #  FIBONACCI ALTERNATIVE (Book P132)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def fib_levels_for_curve(curve: Curve) -> Dict[str, float]:
        """
        Book P132: alternative curve-splitting method using Fibonacci
        tool customized to 33% and 66% levels (NOT the standard
        23.6/38.2/61.8/78.6 retracement levels).

        Note: this always returns the book's fixed 33/66 levels
        (curve.demand_proximal + curve_width/3 and *2/3), even if the
        curve itself was built with adaptive split ratios — it's the
        book's alternative construction method, kept literal on purpose
        so it can be compared against curve.low_top/equilibrium_top.

        Returns:
            {"33%": float, "66%": float}
        """
        return {
            "33%": curve.demand_proximal + curve.subzone_width,
            "66%": curve.demand_proximal + 2 * curve.subzone_width,
        }


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 64)
    print("  CURVE MTF — Book 5 Chapter 12 (Pages 126-135)")
    print("=" * 64)

    # ── Book P131 worked example: proximal lines at $10 and $13 ──
    print("\n── Book P131 worked example ($10 / $13), book-exact (no atr/regime) ──")
    demand_zone = {"proximal": 10.0, "zone_low": 9.5, "zone_high": 10.0, "state": "fresh"}
    supply_zone = {"proximal": 13.0, "zone_low": 13.0, "zone_high": 13.5, "state": "tested"}
    curve = CurveMTF.from_zones(demand_zone, supply_zone, current_price=11.5, timeframe="1M")
    print(f"  {curve.describe()}")
    print(f"  Expected: sub-zone width = (13-10)/3 = 1.0")
    print(f"  Actual:   sub-zone width = {curve.subzone_width}")
    print(f"  Low/Eq boundary  = {curve.low_top}    (expected 11.0)")
    print(f"  Eq/High boundary = {curve.equilibrium_top}    (expected 12.0)")

    # ── Test all 5 positions ──
    print("\n── Position + Bias tests (Book P133) ──")
    test_prices = [9.0, 10.5, 11.5, 12.5, 14.0]
    for p in test_prices:
        pos = curve.position_of(p)
        bias = curve.bias_for(p)
        print(f"  price={p:>5.1f} → position={pos.value:<12} → bias={bias.value}")

    # ── ATR buffer / adaptive split (engine extensions) ──
    print("\n── With ATR buffer + TRENDING/BULLISH adaptive split ──")
    curve2 = CurveMTF.from_zones(
        demand_zone, supply_zone, current_price=11.5, timeframe="1M",
        atr=0.3, regime_ctx={"regime": "TRENDING", "direction": "BULLISH"},
    )
    print(f"  {curve2.describe()}")
    print(f"  (expect a 20/20/60 split favoring High — Low/Eq={curve2.low_top}, Eq/High={curve2.equilibrium_top})")
    print(f"  Boundary buffer = {curve2.boundary_buffer} (= 0.10 * ATR)")

    # ── Confidence + explainability ──
    print("\n── Confidence + explainability ──")
    for p in (10.3, 11.2, 12.8):
        conf = curve2.confidence_for(p, atr=0.3, regime_ctx={"regime": "TRENDING", "direction": "BULLISH"})
        print(f"  price={p}: {conf.reason}")

    # ── Curve quality ──
    print("\n── Curve quality ──")
    q = curve2.quality(atr=0.3)
    print(f"  width_atr={q.curve_width_atr:.2f} demand_fresh={q.demand_freshness} "
          f"supply_fresh={q.supply_freshness} narrow={q.is_narrow} score={q.quality_score}")

    # ── Confluence score ──
    print("\n── Confluence score (curve + mock OB/FVG scores) ──")
    conf_score = curve2.confluence_score(11.2, ob_score=78, fvg_score=65, atr=0.3)
    print(f"  {conf_score}")

    # ── Staleness check ──
    print("\n── Staleness check (dynamic curve hook) ──")
    same = CurveMTF.is_stale(curve2, demand_zone, supply_zone)
    moved_supply = {"proximal": 13.4, "zone_low": 13.4, "zone_high": 13.9}
    changed = CurveMTF.is_stale(curve2, demand_zone, moved_supply)
    print(f"  same zones  → is_stale={same} (expected False)")
    print(f"  moved supply → is_stale={changed} (expected True)")

    # ── Trading style → timeframe triplet (Book P129) ──
    print("\n── Trading Style → Timeframe Triplet (Book P129) ──")
    for style in TradingStyle:
        triplet = get_timeframe_triplet(style)
        freq = TRADE_FREQUENCY_PER_DAY[style]
        print(f"  {style.value:<9}: long={triplet[0]:<4} medium={triplet[1]:<4} "
              f"short={triplet[2]:<4}  (freq {freq[0]}-{freq[1]}/day)")

    # ── HTF override hierarchy (Book P135) ──
    print("\n── HTF Override Hierarchy (Book P135) ──")
    htf_bias = DirectionalBias.BUY_ONLY
    ltf_signals = [("1w", "short"), ("1d", "long")]
    result = CurveMTF.resolve_conflict(htf_bias, ltf_signals)
    print(f"  HTF bias: {htf_bias.value}")
    print(f"  LTF signals: {ltf_signals}")
    print(f"  Decision: {result['decision']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Actionable: {result['actionable_signals']}")
    print(f"  Conflicting: {result['conflicting_signals']}")

    # ── Fibonacci alternative (Book P132) ──
    print("\n── Fibonacci Alternative (Book P132) ──")
    fib = CurveMTF.fib_levels_for_curve(curve)
    print(f"  Fib 33% level = {fib['33%']}  (matches Low/Eq boundary = {curve.low_top})")
    print(f"  Fib 66% level = {fib['66%']}  (matches Eq/High boundary = {curve.equilibrium_top})")

    print("\n" + "=" * 64)
    print("  Curve MTF smoke test complete.")
    print("=" * 64)