# analysis/support_resistance.py
# ============================================================
# Support & Resistance Zone Engine (v2 — Zone-Based)
# ============================================================
# Upgrade (per spec):
#   1. Zones are RANGES (zone_top / zone_bottom), NOT single lines
#   2. Strength score: 2 touches = Weak, 3 = Medium, 4+ = Strong
#   3. Rejection candle validation: wick >= 1.5x body
#   4. Per-instrument volatility-adaptive cluster threshold (ATR-based)
#   5. Timeframe-adaptive swing_window (M5=3, M15=4, H1=4, H4=5, D1=5)
#   6. JSON-serializable output for LLM Agent integration
#   7. Only top 2-3 nearest/relevant zones returned
#   8. Backward compatible (keeps `center`, `nearest_support`, `nearest_res`)
# ============================================================

import json
from typing import Optional

import numpy as np
from utils.logger import get_logger
import pandas as pd

log = get_logger("support_resistance")
# NOTE (Day-NN fix): this used to be immediately overwritten by a bare
# `logging.getLogger(__name__)` two lines below, silently discarding
# whatever handlers/formatting utils.logger.get_logger sets up project-wide.
# Removed the second assignment — this is now the only logger for the module.


# ─── Timeframe → swing_window mapping ──────────────────────────
_TF_SWING_WINDOW = {
    "M1": 3, "M5": 3, "M15": 4, "M30": 4,
    "H1": 4, "H4": 5, "D1": 5, "W1": 5, "MN": 5,
}


def _classify_strength(touches: int) -> str:
    """2=Weak, 3=Medium, 4+=Strong"""
    if touches >= 4:
        return "Strong"
    if touches == 3:
        return "Medium"
    return "Weak"


def _strength_emoji(strength: str) -> str:
    return {"Weak": "🟡", "Medium": "🟠", "Strong": "🔴"}.get(strength, "⚪")


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as % of price — used for adaptive cluster threshold.

    BUG FIX (2026-08-05): production logs showed this hitting its except
    branch on nearly every candle with "The truth value of a Series is
    ambiguous" — that error only happens when `if ... or not np.isfinite(atr)`
    is evaluated on a Series instead of a scalar. Root cause: `df["high"]`
    / `df["low"]` / `df["close"]` return a DataFrame instead of a Series
    whenever the incoming df has duplicate-labeled columns (an upstream
    merge artifact), and duplicate index labels turn the aligned
    subtraction `(l - c.shift())` into a cartesian-style expansion — either
    of which makes `.iloc[-1]` return a Series/row instead of a scalar.

    BUG FIX (2026-08-11): the 2026-08-05 fix above defended against both
    causes but a 6-day live log (72 occurrences in a single session) still
    showed the exact same fallback firing on almost every call, confirming
    label-based alignment is still an active risk somewhere in this path
    (e.g. h/l/c retaining a duplicated index that survives the `.duplicated()`
    check because it's introduced *after* that check by an upstream reindex,
    or a third duplicate-column source the isinstance() guard doesn't catch).
    Rather than continue patching individual symptoms, this rewrite drops
    label-based alignment entirely: h/l/c are converted to raw numpy arrays
    up front, so every subsequent operation is purely positional and no
    pandas index — duplicated or not — can ever again cause a Series to leak
    into a boolean/float context. If the function still fails, the except
    branch now logs shape/dtype/duplicate-index diagnostics (once per
    process) instead of just the exception string, so a *new* root cause is
    identifiable from a single log line instead of blind guessing.
    """
    global _atr_pct_diag_logged
    try:
        if len(df) < period + 1:
            return 0.004  # default 0.4%
        h, l, c = df["high"], df["low"], df["close"]
        if isinstance(h, pd.DataFrame):
            h = h.iloc[:, 0]
        if isinstance(l, pd.DataFrame):
            l = l.iloc[:, 0]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        # Positional-only from here on — .to_numpy() strips the index so
        # duplicate/misaligned labels can no longer trigger label-based
        # alignment expansion anywhere downstream.
        h_arr = h.to_numpy(dtype=float, copy=False)
        l_arr = l.to_numpy(dtype=float, copy=False)
        c_arr = c.to_numpy(dtype=float, copy=False)
        if h_arr.ndim != 1 or l_arr.ndim != 1 or c_arr.ndim != 1:
            raise ValueError(
                f"non-1D array after extraction: h={h_arr.shape} "
                f"l={l_arr.shape} c={c_arr.shape}"
            )
        c_prev = np.roll(c_arr, 1)
        c_prev[0] = np.nan  # no prior close for the first bar
        tr = np.nanmax(
            np.vstack([
                h_arr - l_arr,
                np.abs(h_arr - c_prev),
                np.abs(l_arr - c_prev),
            ]),
            axis=0,
        )
        window = tr[-period:]
        atr = float(np.nanmean(window)) if window.size else float("nan")
        price = float(c_arr[-1])
        if price <= 0 or not np.isfinite(atr):
            return 0.004
        return float(atr / price)
    except Exception as e:
        if not _atr_pct_diag_logged:
            _atr_pct_diag_logged = True
            try:
                diag = (
                    f"shape={df.shape} dup_index={df.index.duplicated().any()} "
                    f"dup_columns={df.columns.duplicated().any()} "
                    f"dtypes={df.dtypes.to_dict()}"
                )
            except Exception:
                diag = "diagnostics unavailable"
            log.warning(
                f"[SR] _atr_pct fallback to 0.004 default (diagnostics logged "
                f"once per process): {e} | {diag}"
            )
        else:
            log.debug(f"[SR] _atr_pct fallback to 0.004 default: {e}")
        return 0.004


_atr_pct_diag_logged = False


class SupportResistance:
    """
    AI Trader-এর S/R Zone detection engine (v2 — Zone-Based).

    Output format per zone:
      {
        "zone_top":    <price>,
        "zone_bottom": <price>,
        "center":      <price>,         # backward-compat
        "touches":     <count>,
        "strength":    "Weak|Medium|Strong",
        "role":        "support|resistance",
        "last_touch_time": <ISO>,
        "distance_pips":   <float>      # from current price
      }
    """

    def __init__(
        self,
        window: int = 4,
        tolerance: float = 0.0015,
        # New v2 params (auto-tuned if not given)
        swing_window: Optional[int] = None,
        cluster_threshold_pct: Optional[float] = None,
        min_touches: int = 2,
        wick_body_ratio: float = 1.5,
        timeframe: str = "H1",
        max_zones_per_side: int = 3,
        # v3 params — trending-regime liquidity fallback (see
        # cluster_into_zones()/analyze() docstrings for the full
        # rationale). Multi-touch clustering is a ranging-market
        # assumption; these control the supplementary liquidity sources
        # used when it produces too few/no zones.
        trending_eqh_loose_multiplier: float = 2.0,
        max_raw_swing_levels: int = 3,
    ):
        # Backward compat
        self.window = window
        self.tolerance = tolerance

        # v2: auto-tune if not specified
        self.timeframe = (timeframe or "H1").upper()
        self.swing_window = swing_window or _TF_SWING_WINDOW.get(
            self.timeframe, 4
        )
        # cluster_threshold_pct: if not given, derive from ATR later
        self.cluster_threshold_pct = cluster_threshold_pct
        self.min_touches = max(2, min_touches)
        self.wick_body_ratio = wick_body_ratio
        self.max_zones_per_side = max_zones_per_side
        self.trending_eqh_loose_multiplier = trending_eqh_loose_multiplier
        self.max_raw_swing_levels = max_raw_swing_levels

    # ─────────────────────────────────────────────
    # STEP 1: Swing High & Low detection
    # ─────────────────────────────────────────────

    def find_swing_highs(self, df: pd.DataFrame) -> list:
        """Swing high = high > both left & right N candles (N = swing_window)."""
        swing_highs = []
        w = self.swing_window
        if len(df) < 2 * w + 1:
            return swing_highs

        highs = df["high"].values
        for i in range(w, len(df) - w):
            window_slice = highs[i - w : i + w + 1]
            if highs[i] == window_slice.max() and highs[i] > highs[i - w : i].max():
                swing_highs.append({
                    "index": i,
                    "time": df.index[i],
                    "price": float(highs[i]),
                })
        return swing_highs

    def find_swing_lows(self, df: pd.DataFrame) -> list:
        """Swing low = low < both left & right N candles (N = swing_window)."""
        swing_lows = []
        w = self.swing_window
        if len(df) < 2 * w + 1:
            return swing_lows

        lows = df["low"].values
        for i in range(w, len(df) - w):
            window_slice = lows[i - w : i + w + 1]
            if lows[i] == window_slice.min() and lows[i] < lows[i - w : i].min():
                swing_lows.append({
                    "index": i,
                    "time": df.index[i],
                    "price": float(lows[i]),
                })
        return swing_lows

    # ─────────────────────────────────────────────
    # STEP 2: Rejection-candle validation
    # ─────────────────────────────────────────────

    def _is_valid_rejection(
        self,
        candle: pd.Series,
        direction: str = "resistance",
    ) -> bool:
        """
        Rejection candle validation (per spec):
          For resistance: upper wick >= wick_body_ratio × body
          For support:    lower wick >= wick_body_ratio × body
        """
        try:
            o, h, l, c = (
                float(candle["open"]),
                float(candle["high"]),
                float(candle["low"]),
                float(candle["close"]),
            )
            body = abs(c - o)
            if body < 1e-9:
                # Doji — treat as rejection if any wick exists
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l
                wick = upper_wick if direction == "resistance" else lower_wick
                return wick > 0

            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            wick = upper_wick if direction == "resistance" else lower_wick
            return wick >= self.wick_body_ratio * body
        except Exception as e:
            log.debug(f"[SR] _is_valid_rejection failed on candle, treating as no-rejection: {e}")
            return False

    def _count_valid_rejections(
        self,
        df: pd.DataFrame,
        zone_top: float,
        zone_bottom: float,
        direction: str,
        proximity_pct: float = 0.0015,
    ) -> int:
        """
        Count candles within `proximity_pct` of the zone that show
        valid rejection wick. Used to enhance strength score.

        PERF (2026-08-06): previously looped candle-by-candle with
        `.iterrows()` (the slowest pandas row-access pattern) and called
        `_is_valid_rejection()` once per row. This is called once per zone
        built (potentially many times per analyze()), so vectorize with
        the same math `_is_valid_rejection()` uses — the boolean logic
        below is kept in lockstep with that method, just applied to whole
        Series instead of one candle at a time.
        """
        try:
            cp = float(df["close"].iloc[-1])
            if cp <= 0:
                return 0
            band = cp * proximity_pct
            # Instrumentation: log types and shapes to diagnose expensive comparisons
            try:
                low_col = df["low"]
                # `low_col` can be a Series or DataFrame when duplicate columns exist
                low_dtype = getattr(low_col, "dtype", type(low_col))
                low_shape = getattr(low_col, "shape", None)
            except Exception as _e:
                low_dtype = f"error:{_e}"
                low_shape = None
            try:
                log.debug(
                    "[SR-INSTR] _count_valid_rejections zone_top=%r (%s) band=%r (%s) df['low'].dtype=%r shape=%r",
                    zone_top,
                    type(zone_top).__name__,
                    band,
                    type(band).__name__,
                    low_dtype,
                    low_shape,
                )
            except Exception:
                # Best-effort instrumentation — do not raise
                log.debug("[SR-INSTR] _count_valid_rejections instrumentation skipped due to logging error")
            # We want candles that touched the zone (wick reached it)
            if direction == "resistance":
                touched = df[(df["high"] >= zone_bottom - band) &
                             (df["high"] <= zone_top + band)]
            else:
                touched = df[(df["low"] <= zone_top + band) &
                             (df["low"] >= zone_bottom - band)]
            if touched.empty:
                return 0

            o = touched["open"].astype(float)
            h = touched["high"].astype(float)
            l = touched["low"].astype(float)
            c = touched["close"].astype(float)
            body = (c - o).abs()
            upper_wick = h - np.maximum(o, c)
            lower_wick = np.minimum(o, c) - l
            wick = upper_wick if direction == "resistance" else lower_wick

            is_doji = body < 1e-9
            # Doji: rejection if any wick exists. Non-doji: wick >= ratio * body.
            valid = np.where(is_doji, wick > 0, wick >= self.wick_body_ratio * body)
            return int(np.sum(valid))
        except Exception as e:
            log.debug(f"[SR] _count_valid_rejections failed for zone [{zone_bottom}, {zone_top}]: {e}")
            return 0

    # ─────────────────────────────────────────────
    # STEP 3: Cluster swing points into ZONES (ranges)
    # ─────────────────────────────────────────────

    def _get_cluster_threshold(self, df: pd.DataFrame, atr_pct: Optional[float] = None) -> float:
        """
        Cluster threshold as % of price.
        Auto-tune from ATR if not specified.

        Per spec: ±0.3%–0.5% — but "instrument volatility অনুযায়ী adjust".
        For low-volatility FX majors (ATR ~0.07%), 0.3% would be ~4 ATRs,
        way too wide. We use ATR×1.5 with sensible caps:
          - Floor: 0.10% (10 pips on EURUSD) — prevents micro-clusters
          - Ceiling: 0.80% — caps on highly volatile instruments

        PERF (2026-08-06): `_atr_pct(df)` is an O(n) rolling computation
        over the whole df. Before this fix it was being recomputed from
        scratch by every caller in the pipeline (this method, once per
        zone in `_build_zone`, twice in `_raw_swing_levels`...) — 15-20+
        times per `analyze()` call on a live multi-symbol bot. Callers that
        already have `atr_pct` for this df (analyze() computes it once at
        the top) should pass it in here; this only falls back to computing
        it when no cached value is supplied, so direct/standalone callers
        keep working unchanged.
        """
        if self.cluster_threshold_pct is not None:
            return self.cluster_threshold_pct

        if atr_pct is None:
            atr_pct = _atr_pct(df, period=14)
        # ATR × 1.5 is a common "zone width" multiplier in S/R literature
        threshold = max(0.001, min(0.008, atr_pct * 1.5))
        return threshold

    def cluster_into_zones(
        self,
        swing_points: list,
        df: pd.DataFrame,
        direction: str,
        threshold_pct_override: Optional[float] = None,
        min_touches_override: Optional[int] = None,
        source: str = "cluster",
        atr_pct: Optional[float] = None,
    ) -> list:
        """
        Cluster nearby swing prices into ZONES (range/box).

        Spec rule:
          - Sort prices
          - Group consecutive prices within `cluster_threshold_pct`
          - Keep clusters with >= min_touches
          - zone_top = max swing price in cluster
          - zone_bottom = min swing price in cluster
          - strength: 2=Weak, 3=Medium, 4+=Strong

        threshold_pct_override / min_touches_override: let callers reuse
        this same clustering logic with looser parameters (e.g. the
        EQH/EQL trending-regime fallback in analyze()) without touching
        self.cluster_threshold_pct / self.min_touches, which stay the
        instance defaults used for the primary, ranging-market-tuned pass.
        source: tagged onto each built zone's "source" key so callers can
        tell a confirmed multi-touch cluster apart from a looser/fallback
        one (e.g. "cluster" vs "eqh_eql" vs "raw_swing").
        """
        if not swing_points:
            return []

        threshold_pct = (
            threshold_pct_override if threshold_pct_override is not None
            else self._get_cluster_threshold(df, atr_pct=atr_pct)
        )
        min_touches = (
            min_touches_override if min_touches_override is not None
            else self.min_touches
        )
        # Sort by price ascending
        sorted_pts = sorted(swing_points, key=lambda p: p["price"])
        current_cluster = [sorted_pts[0]]

        zones = []
        for p in sorted_pts[1:]:
            # Compare new price to cluster CENTER (mean), not just last price.
            # This prevents "drift chaining" where prices slowly drift apart
            # but each consecutive pair stays within threshold.
            cluster_prices = [pt["price"] for pt in current_cluster]
            cluster_center = float(np.mean(cluster_prices))
            cluster_min = min(cluster_prices)
            cluster_max = max(cluster_prices)
            # New point must be within threshold of BOTH center AND nearest boundary
            dist_to_center = abs(p["price"] - cluster_center) / cluster_center if cluster_center > 0 else 1.0
            # FIX: normalize each candidate distance by the boundary it's
            # being measured against, not unconditionally by cluster_min
            # (which was arbitrary when comparing against cluster_max).
            dist_to_min = abs(p["price"] - cluster_min) / cluster_min if cluster_min > 0 else 1.0
            dist_to_max = abs(p["price"] - cluster_max) / cluster_max if cluster_max > 0 else 1.0
            dist_to_nearest = min(dist_to_min, dist_to_max)
            if dist_to_center <= threshold_pct and dist_to_nearest <= threshold_pct:
                current_cluster.append(p)
            else:
                if len(current_cluster) >= min_touches:
                    zones.append(self._build_zone(current_cluster, df, direction, source=source, atr_pct=atr_pct))
                current_cluster = [p]

        # last cluster
        if len(current_cluster) >= min_touches:
            zones.append(self._build_zone(current_cluster, df, direction, source=source, atr_pct=atr_pct))

        return zones

    def _raw_swing_levels(
        self,
        swing_points: list,
        df: pd.DataFrame,
        direction: str,
        max_levels: Optional[int] = None,
        buffer_multiplier: float = 0.5,
        atr_pct: Optional[float] = None,
    ) -> list:
        """
        TRENDING-regime fallback liquidity source: turns the most recent
        INDIVIDUAL swing highs/lows into thin single-touch levels, without
        requiring a second point to cluster against.

        Why this exists: cluster_into_zones() needs >= 2 swing points to
        land within a tight price band before it will emit a zone. That's
        a ranging-market assumption — in a genuinely trending market,
        swing highs/lows structurally make higher-highs / lower-lows (or
        lower-lows / lower-highs) and rarely revisit the same price, so
        clustering produces few or no zones at all. StopHunt/ICT-AMD then
        have nothing to check a sweep against and both abstain, which is
        exactly the "0/3 consensus -> single-layer override becomes the
        default path" failure chain this fixes.

        A single untested swing high/low is still a legitimate liquidity
        target on its own in SMC/ICT terms — retail stops cluster just
        beyond it whether or not price has re-tested it. It's simply
        *unconfirmed* by a re-touch, so touches=1 -> _classify_strength()
        already buckets it as "Weak" -> no downstream change needed
        (StopHunt/ICT-AMD already accept "Weak" zones per the 07-17 fix).
        """
        if not swing_points:
            return []
        max_levels = max_levels if max_levels is not None else self.max_raw_swing_levels

        if atr_pct is None:
            atr_pct = _atr_pct(df, period=14)
        buffer_pct = max(atr_pct * buffer_multiplier, 0.0003)  # thin band, floor ~3 pips-equivalent

        # Most RECENT swing points first (by bar index/time, not by price) —
        # a trending market's oldest swing points are the least relevant
        # liquidity targets; the last few untested extremes are what
        # price is actually likely to sweep next.
        recent = sorted(swing_points, key=lambda p: p["index"], reverse=True)[:max_levels]

        price_ref = float(df["close"].iloc[-1]) if len(df) else 0.0
        atr_abs = atr_pct * price_ref

        levels = []
        for p in recent:
            price = float(p["price"])
            idx = p["index"]
            if price <= 0:
                continue
            band = price * buffer_pct
            zone_top = round(price + band, 5)
            zone_bottom = round(price - band, 5)
            valid_rej = self._count_valid_rejections(df, zone_top, zone_bottom, direction=direction)
            # FIX: mirror _build_zone()'s thin-zone check here too — this
            # method used to build its dict directly and skip it entirely,
            # so raw_swing zones always silently reported is_thin_zone=False
            # (via callers' .get(..., False)) regardless of actual width.
            # elevated_breakout_risk is left False explicitly: touches=1 here
            # always classifies as "Weak" strength, and that flag's formula
            # only fires on "Medium", so it can never be True for this tier —
            # but it's now set explicitly instead of silently missing, so
            # every zone dict (cluster / eqh_eql / raw_swing) has the same
            # shape.
            zone_width = zone_top - zone_bottom
            is_thin_zone = bool(atr_abs > 0 and zone_width < 0.5 * atr_abs)
            levels.append({
                "zone_top": zone_top,
                "zone_bottom": zone_bottom,
                "center": round(price, 5),
                "touches": 1,
                "valid_rejections": valid_rej,
                "strength": _classify_strength(1),
                "role": direction,
                "last_touch_time": str(df.index[idx]) if idx < len(df) else None,
                "last_touch_index": idx,
                "source": "raw_swing",
                "is_equal_level": False,
                "is_thin_zone": is_thin_zone,
                "elevated_breakout_risk": False,
            })
        return levels

    def _merge_zone_sources(self, *tiers: list, df: pd.DataFrame, atr_pct: Optional[float] = None) -> list:
        """
        Merge zone lists from different confidence tiers, highest-priority
        tier first (e.g. confirmed multi-touch cluster, then EQH/EQL loose
        cluster, then raw single-touch swing points). A candidate from a
        lower tier is dropped if it overlaps (within the standard cluster
        threshold) a zone already accepted from a higher tier — no need to
        report the same price level twice at different confidence tiers.
        """
        threshold_pct = self._get_cluster_threshold(df, atr_pct=atr_pct)
        merged: list = []
        for tier in tiers:
            for z in tier:
                ref = z["center"] if z["center"] > 0 else 1.0
                if any(abs(z["center"] - m["center"]) / ref <= threshold_pct for m in merged):
                    continue
                merged.append(z)
        return merged

    def _build_zone(
        self,
        cluster: list,
        df: pd.DataFrame,
        direction: str,
        source: str = "cluster",
        atr_pct: Optional[float] = None,
    ) -> dict:
        """Build a zone dict from a cluster of swing points."""
        prices = [p["price"] for p in cluster]
        zone_top = max(prices)
        zone_bottom = min(prices)
        center = float(np.mean(prices))
        touches = len(cluster)
        strength = _classify_strength(touches)
        last_idx = max(p["index"] for p in cluster)
        last_time = df.index[last_idx] if last_idx < len(df) else None

        # Enhance strength via rejection candle count
        valid_rej = self._count_valid_rejections(
            df, zone_top, zone_bottom, direction=direction
        )
        if valid_rej >= 4 and strength == "Medium":
            strength = "Strong"
        elif valid_rej >= 3 and strength == "Weak":
            strength = "Medium"

        # EVIDENCE-BASED FIX: walk-forward backtest, confirmed cross-pair
        # (EURUSD/AUDCAD/AUDCHF/AUDJPY, H1/H4/M15, no look-ahead) found
        # EQH/EQL-sourced zones bounce 71.2% of the time when touched
        # (n=80), vs 56.1% for standard multi-touch cluster zones (n=4292,
        # two-proportion z-test p=0.0066) and 53.5% for raw single-touch
        # swing levels (n=357, p=0.0038). EQH/EQL zones were previously
        # bucketed by raw touch count like any other source, undervaluing
        # them. Confidence: medium-high — effect replicated on a second,
        # larger, cross-pair sample after the original EURUSD-only n=40
        # finding; still the smallest-n source class of the three.
        if source == "eqh_eql" and strength == "Weak":
            strength = "Medium"

        # EVIDENCE-BASED FIX (thin-zone downgrade): same cross-pair backtest
        # (7,554 zone-instances) found zone width is by far the strongest
        # predictor of outcome — narrowest-quartile zones (~<=11 pips on
        # majors) broke out 61.8% of the time when touched, widest-quartile
        # zones (~>=29 pips) only 23.6% (z=18.76, p<1e-77, monotonic across
        # all 4 quartiles). Touch count alone doesn't capture this — a
        # 4-touch zone crammed into a band thinner than half the current
        # ATR is not "Strong" in practice. Confidence: high (largest sample
        # and effect size in the whole analysis).
        zone_width = zone_top - zone_bottom
        atr_pct_now = atr_pct if atr_pct is not None else _atr_pct(df, period=14)
        price_ref = float(df["close"].iloc[-1]) if len(df) else 0.0
        atr_abs = atr_pct_now * price_ref
        is_thin_zone = bool(atr_abs > 0 and zone_width < 0.5 * atr_abs)
        if is_thin_zone:
            if strength == "Strong":
                strength = "Medium"
            elif strength == "Medium":
                strength = "Weak"

        # EVIDENCE-BASED FLAG (Medium-tier retest risk): the same backtest
        # found "Medium" (3-touch) zones break out MORE than both "Weak"
        # (2-touch, 47.8%) and "Strong" (4+, 41.4%) — Medium broke out
        # 56.7% of the time (p=0.005 vs Weak, p<1e-11 vs Strong), and this
        # held up checked *within each individual timeframe separately*
        # (H1/H4/M15), so it isn't a timeframe artifact. Matches the
        # well-known "third touch often breaks" pattern: the first two
        # touches get defended, the third frequently clears the level. Not
        # renamed/reordered "Medium" itself — other code may assume a
        # Weak < Medium < Strong ordering on the string, and changing that
        # would be a silent breaking change — instead surfaced as an
        # explicit flag so the signal/decision layer can weight it.
        # Confidence: high (593-sample strength tier, effect holds
        # per-timeframe).
        elevated_breakout_risk = bool(strength == "Medium" and not is_thin_zone)

        return {
            "zone_top": round(zone_top, 5),
            "zone_bottom": round(zone_bottom, 5),
            "center": round(center, 5),  # backward-compat
            "touches": touches,
            "valid_rejections": valid_rej,
            "strength": strength,
            "role": direction,
            "last_touch_time": str(last_time) if last_time is not None else None,
            "last_touch_index": last_idx,
            "source": source,
            "is_equal_level": source == "eqh_eql",
            "is_thin_zone": is_thin_zone,
            "elevated_breakout_risk": elevated_breakout_risk,
        }

    # ─── Backward-compat: old API ─────────────────
    def create_price_zones(self, levels: list) -> list:
        """
        Backward-compat wrapper. Old callers passed a list of dicts
        with 'price' and expected 'center' + 'touches' back.
        """
        if not levels:
            return []
        # We need df for ATR; fall back to simple clustering with old tolerance
        zones = []
        for level in levels:
            price = level.get("price") if isinstance(level, dict) else level
            merged = False
            for zone in zones:
                if abs(price - zone["center"]) <= self.tolerance:
                    zone["prices"].append(price)
                    zone["center"] = round(float(np.mean(zone["prices"])), 5)
                    zone["touches"] += 1
                    merged = True
                    break
            if not merged:
                zones.append({
                    "center": round(float(price), 5),
                    "prices": [float(price)],
                    "touches": 1,
                })
        zones.sort(key=lambda z: z["touches"], reverse=True)
        return zones

    # ─────────────────────────────────────────────
    # STEP 4: Pivot Point Calculation (unchanged)
    # ─────────────────────────────────────────────

    def calculate_pivot(self, df: pd.DataFrame) -> dict:
        """Classic Pivot Point from previous complete candle."""
        if len(df) < 2:
            log.debug("[SR] calculate_pivot: need >=2 candles, got %d", len(df))
            return {}
        prev = df.iloc[-2]
        H, L, C = float(prev["high"]), float(prev["low"]), float(prev["close"])
        pivot = (H + L + C) / 3
        return {
            "pivot": round(pivot, 5),
            "R1":    round(2 * pivot - L, 5),
            "R2":    round(pivot + (H - L), 5),
            "S1":    round(2 * pivot - H, 5),
            "S2":    round(pivot - (H - L), 5),
        }

    # ─────────────────────────────────────────────
    # STEP 5: Nearest S/R from current price
    # ─────────────────────────────────────────────

    def find_nearest_levels(
        self,
        current_price: float,
        support_zones: list,
        resistance_zones: list,
    ) -> tuple:
        """
        Nearest support = zone whose center is at or below current price.
        Nearest resistance = zone whose center is at or above current price.
        Handles "price inside zone" (price testing the zone) by using center as reference.
        Returns dicts (or None) with 'center' key for backward-compat.
        """
        # Support: center <= current_price (includes price-inside-zone case)
        sup_below = [z for z in support_zones if z["center"] <= current_price]
        nearest_sup = max(sup_below, key=lambda z: z["center"]) if sup_below else None

        # Resistance: center >= current_price (includes price-inside-zone case)
        res_above = [z for z in resistance_zones if z["center"] >= current_price]
        nearest_res = min(res_above, key=lambda z: z["center"]) if res_above else None

        # Fallback: if price is inside a zone but center is on the wrong side,
        # try to find any zone that overlaps the price
        if nearest_sup is None and support_zones:
            overlapping = [z for z in support_zones
                          if z["zone_bottom"] <= current_price <= z["zone_top"]]
            if overlapping:
                nearest_sup = max(overlapping, key=lambda z: z["zone_top"])
        if nearest_res is None and resistance_zones:
            overlapping = [z for z in resistance_zones
                          if z["zone_bottom"] <= current_price <= z["zone_top"]]
            if overlapping:
                nearest_res = min(overlapping, key=lambda z: z["zone_bottom"])

        return nearest_sup, nearest_res

    # ─────────────────────────────────────────────
    # STEP 6: Filter to top N relevant zones (per spec rule 5)
    # ─────────────────────────────────────────────

    def _filter_relevant_zones(
        self,
        zones: list,
        current_price: float,
        max_zones: int = 3,
        side: str = "support",
    ) -> list:
        """
        Per spec rule 5: only return most recent / relevant zones.
        Sort by (closeness to price, then recency, then strength).
        """
        if not zones:
            return []

        # direction filter
        # FIX (2026-08-06): previously used a strict `zone_top < current_price`
        # / `zone_bottom > current_price` test, which drops a zone the moment
        # price is trading *inside* it (zone_bottom <= current_price <=
        # zone_top). That's exactly the "AT SUPPORT"/"AT RESISTANCE" case
        # get_ai_context() separately detects off the unfiltered
        # nearest_support/nearest_res — so the single most actionable zone
        # (the one price is standing on right now) was silently missing from
        # support_zones/resistance_zones (and therefore from to_json() /
        # to_prompt_text() / get_summary()) while get_ai_context() still
        # called it out by name, giving the LLM an inconsistent picture.
        # `<=`/`>=` include the "at zone" case; the sort key below already
        # clamps distance to a floor of 1e-9, so an at-zone entry (which can
        # have dist <= 0) naturally sorts first as the most relevant.
        if side == "support":
            relevant = [z for z in zones if z["zone_bottom"] <= current_price]
        else:
            relevant = [z for z in zones if z["zone_top"] >= current_price]

        # Strength weight for sort
        strength_weight = {"Strong": 3, "Medium": 2, "Weak": 1}

        def _sort_key(z):
            # distance from current price (smaller = more relevant)
            if side == "support":
                dist = current_price - z["center"]
            else:
                dist = z["center"] - current_price
            dist = max(dist, 1e-9)
            # Combined score: closeness * strength * recency
            recency = z.get("last_touch_index", 0) + 1
            score = strength_weight.get(z["strength"], 1) * (recency / 100) / dist
            return -score  # higher score first

        relevant.sort(key=_sort_key)
        return relevant[:max_zones]

    @staticmethod
    def _resolve_pip_value(symbol: str) -> float:
        """
        Instrument-aware pip size resolver.

        FIX (institutional review, Finding C-3): this logic previously lived
        inline only in analyze() (correct there) while get_ai_context()
        separately hardcoded 0.0001 for distance-to-support/resistance,
        which is ~100x wrong for JPY pairs and wrong for XAUUSD/indices too.
        Extracted here so both call sites share one source of truth instead
        of two definitions that can silently drift apart.

        JPY pairs use 0.01, XAUUSD uses 0.1, index CFDs use 1.0, all other
        FX majors/crosses default to 0.0001.
        """
        s = (symbol or "").upper()
        if s.endswith("JPY"):
            return 0.01
        if s == "XAUUSD":
            return 0.1
        if s in ("US30", "NAS100", "SPX500", "GER40"):
            return 1.0
        return 0.0001

    def _attach_distance_pips(self, zones: list, current_price: float, pip_value: float) -> list:
        """Add distance_pips to each zone for human-readable output."""
        for z in zones:
            z["distance_pips"] = round(abs(z["center"] - current_price) / pip_value, 1)
        return zones

    # ─────────────────────────────────────────────
    # STEP 7: FULL PIPELINE
    # ─────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> dict:
        """
        Full S/R Zone analysis pipeline.

        Returns dict with:
          - support_zones:    list of zone dicts (top N relevant)
          - resistance_zones: list of zone dicts (top N relevant)
          - all_support_zones:    full list (pre-filter)
          - all_resistance_zones: full list (pre-filter)
          - pivot:            pivot levels dict
          - nearest_support:  dict (or None) — backward compat
          - nearest_res:      dict (or None) — backward compat
          - current_price:    float
          - symbol:           str
          - timeframe:        str
          - cluster_threshold_pct: float (actual used)
          - swing_window:     int
        """
        # Ensure index is sorted
        if len(df) < 2 * self.swing_window + 5:
            log.warning(
                f"[SR] Insufficient candles ({len(df)}) for swing_window={self.swing_window}"
            )

        # 0. ATR% — computed ONCE per analyze() call and threaded through the
        # rest of the pipeline. PERF (2026-08-06): _atr_pct() is an O(n)
        # rolling calc over the whole df; previously every downstream caller
        # (cluster threshold x5, raw swing levels x2, _build_zone per zone
        # built) recomputed it independently — 15-20+ full recomputations
        # per analyze() call. A 10-zone result now does exactly one.
        atr_pct = _atr_pct(df, period=14)

        # 1. Swing points
        swing_highs = self.find_swing_highs(df)
        swing_lows = self.find_swing_lows(df)

        # 2. Cluster into zones (tier 1: confirmed multi-touch, tight threshold)
        all_resistance = self.cluster_into_zones(swing_highs, df, direction="resistance", atr_pct=atr_pct)
        all_support = self.cluster_into_zones(swing_lows, df, direction="support", atr_pct=atr_pct)

        # 2a. Trending-regime fallback tier 2: EQH/EQL loose cluster — same
        # clustering logic, threshold widened by trending_eqh_loose_multiplier,
        # tagged source="eqh_eql" (this is what _build_zone reads to set
        # is_equal_level=True).
        loose_threshold = self._get_cluster_threshold(df, atr_pct=atr_pct) * self.trending_eqh_loose_multiplier
        eqh_resistance = self.cluster_into_zones(
            swing_highs, df, direction="resistance",
            threshold_pct_override=loose_threshold, source="eqh_eql", atr_pct=atr_pct,
        )
        eql_support = self.cluster_into_zones(
            swing_lows, df, direction="support",
            threshold_pct_override=loose_threshold, source="eqh_eql", atr_pct=atr_pct,
        )

        # 2b. Trending-regime fallback tier 3: single-touch raw swing levels
        # for any price real estate the two cluster passes didn't cover.
        raw_resistance = self._raw_swing_levels(swing_highs, df, direction="resistance", atr_pct=atr_pct)
        raw_support = self._raw_swing_levels(swing_lows, df, direction="support", atr_pct=atr_pct)

        # 2c. Merge, highest-confidence tier first
        all_resistance = self._merge_zone_sources(all_resistance, eqh_resistance, raw_resistance, df=df, atr_pct=atr_pct)
        all_support = self._merge_zone_sources(all_support, eql_support, raw_support, df=df, atr_pct=atr_pct)

        # DIAGNOSTIC (temporary) - proves whether this analyze() call, and the
        # tier-2/tier-3 fallback, actually ran. Remove once confirmed.
        _src_counts_r = {}
        _src_counts_s = {}
        for z in all_resistance:
            _src_counts_r[z.get("source", "?")] = _src_counts_r.get(z.get("source", "?"), 0) + 1
        for z in all_support:
            _src_counts_s[z.get("source", "?")] = _src_counts_s.get(z.get("source", "?"), 0) + 1
        log.debug(f"[SR-DIAG] {symbol}: resistance={len(all_resistance)} {_src_counts_r} | support={len(all_support)} {_src_counts_s}")

        # 3. Pivot
        try:
            pivot = self.calculate_pivot(df)
        except Exception as e:
            pivot = {}

        # 4. Current price
        current_price = float(df["close"].iloc[-1])

        # 5. Filter to most relevant (per spec rule 5)
        relevant_support = self._filter_relevant_zones(
            all_support, current_price, max_zones=self.max_zones_per_side, side="support"
        )
        relevant_resistance = self._filter_relevant_zones(
            all_resistance, current_price, max_zones=self.max_zones_per_side, side="resistance"
        )

        # 6. Distance pips (instrument-aware)
        pip_value = self._resolve_pip_value(symbol)

        relevant_support = self._attach_distance_pips(relevant_support, current_price, pip_value)
        relevant_resistance = self._attach_distance_pips(relevant_resistance, current_price, pip_value)

        # 7. Nearest for backward compat
        # FIX: previously ran against relevant_support/relevant_resistance
        # (already filtered to top max_zones_per_side by a strength/recency/
        # distance score). That meant the true closest zone by price could
        # lose the relevance ranking to a stronger-but-farther zone and get
        # dropped before "nearest" was ever computed. Backtested on 5yr
        # EURUSD H1: this disagreed with the literal nearest zone in ~25%
        # of checkpoints. Downstream modules (market_bias.py etc., per the
        # get_ai_context docstring) rely on nearest_support/nearest_res
        # being the true nearest — so this must run against the full,
        # unfiltered zone list.
        nearest_sup, nearest_res = self.find_nearest_levels(
            current_price, all_support, all_resistance
        )
        # all_support/all_resistance aren't pip-annotated (only the
        # relevance-filtered lists get _attach_distance_pips'd above), so
        # backfill distance_pips on whichever single zone won out here to
        # preserve the old dict shape for any caller reading it directly.
        if nearest_sup is not None and "distance_pips" not in nearest_sup:
            nearest_sup = dict(nearest_sup)
            nearest_sup["distance_pips"] = round(abs(nearest_sup["center"] - current_price) / pip_value, 1)
        if nearest_res is not None and "distance_pips" not in nearest_res:
            nearest_res = dict(nearest_res)
            nearest_res["distance_pips"] = round(abs(nearest_res["center"] - current_price) / pip_value, 1)

        return {
            "support_zones":         relevant_support,
            "resistance_zones":      relevant_resistance,
            "all_support_zones":     all_support,
            "all_resistance_zones":  all_resistance,
            "pivot":                 pivot,
            "nearest_support":       nearest_sup,
            "nearest_res":           nearest_res,
            "current_price":         current_price,
            "symbol":                symbol,
            "timeframe":             self.timeframe,
            "swing_window":          self.swing_window,
            "cluster_threshold_pct": self._get_cluster_threshold(df, atr_pct=atr_pct),
            "min_touches":           self.min_touches,
            "wick_body_ratio":       self.wick_body_ratio,
        }

    # ─────────────────────────────────────────────
    # JSON OUTPUT — LLM Agent integration
    # ─────────────────────────────────────────────

    def to_json(self, result: dict) -> str:
        """
        Spec-compliant JSON output for LLM Agent consumption.

        Output schema:
          {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "current_price": 1.0850,
            "resistance_zones": [
              {"zone_top": ..., "zone_bottom": ..., "touches": N, "strength": "..."}
            ],
            "support_zones": [...]
          }
        """
        def _slim(zones):
            return [
                {
                    "zone_top":    z["zone_top"],
                    "zone_bottom": z["zone_bottom"],
                    "touches":     z["touches"],
                    "strength":    z["strength"],
                    "distance_pips": z.get("distance_pips"),
                    "last_touch_time": z.get("last_touch_time"),
                    # Backtest-derived risk flags (see IMPLEMENTATION_LOG):
                    "is_thin_zone": z.get("is_thin_zone", False),
                    "elevated_breakout_risk": z.get("elevated_breakout_risk", False),
                }
                for z in zones
            ]

        payload = {
            "symbol":           result.get("symbol", ""),
            "timeframe":        result.get("timeframe", ""),
            "current_price":    round(result.get("current_price", 0.0), 5),
            "resistance_zones": _slim(result.get("resistance_zones", [])),
            "support_zones":    _slim(result.get("support_zones", [])),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _risk_suffix(z: dict) -> str:
        """Short LLM-readable suffix for the backtest-derived risk flags."""
        tags = []
        if z.get("is_thin_zone"):
            tags.append("THIN-vs-ATR")
        if z.get("elevated_breakout_risk"):
            tags.append("3RD-TOUCH-RISK")
        return f" [{','.join(tags)}]" if tags else ""

    def to_prompt_text(self, result: dict) -> str:
        """
        LLM-friendly plain-text rendering for embedding into LLM prompts.
        Includes only relevant zones near current price.
        """
        cp = result.get("current_price", 0.0)
        sym = result.get("symbol", "")
        tf = result.get("timeframe", "")
        lines = [
            f"=== SUPPORT & RESISTANCE ZONES ({sym} {tf}) ===",
            f"Current Price: {cp:.5f}",
            "",
            "-- Resistance Zones (above price) --",
        ]
        if not result.get("resistance_zones"):
            lines.append("  (none)")
        else:
            for z in result["resistance_zones"]:
                lines.append(
                    f"  R: {z['zone_bottom']:.5f} → {z['zone_top']:.5f}  "
                    f"| touches={z['touches']} | strength={z['strength']} "
                    f"| dist={z.get('distance_pips','?')} pips"
                    f"{self._risk_suffix(z)}"
                )

        lines.append("")
        lines.append("-- Support Zones (below price) --")
        if not result.get("support_zones"):
            lines.append("  (none)")
        else:
            for z in result["support_zones"]:
                lines.append(
                    f"  S: {z['zone_bottom']:.5f} → {z['zone_top']:.5f}  "
                    f"| touches={z['touches']} | strength={z['strength']} "
                    f"| dist={z.get('distance_pips','?')} pips"
                    f"{self._risk_suffix(z)}"
                )
        lines.append("=" * 50)
        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # SUMMARY — Human readable
    # ─────────────────────────────────────────────

    def get_summary(self, result: dict) -> None:
        """Print human-readable zone summary."""
        cp = result.get("current_price", 0.0)
        sym = result.get("symbol", "")
        tf = result.get("timeframe", "")
        sw = result.get("swing_window", "?")
        cth = result.get("cluster_threshold_pct", 0)

        print("\n" + "═" * 56)
        print(f"  📐  S/R ZONES  ({sym} {tf})  swing_window={sw}  band={cth*100:.2f}%")
        print("═" * 56)
        print(f"  Current Price :  {cp:.5f}")
        print()

        print("  ── Resistance Zones ──")
        if not result.get("resistance_zones"):
            print("    (none)")
        else:
            for i, z in enumerate(result["resistance_zones"], 1):
                emoji = _strength_emoji(z["strength"])
                print(
                    f"    R{i} {emoji}  {z['zone_bottom']:.5f} → {z['zone_top']:.5f}"
                    f"  | touches={z['touches']}  rej={z.get('valid_rejections',0)}"
                    f"  | {z['strength']}"
                    f"  | +{z.get('distance_pips','?')} pips"
                )

        print()
        print("  ── Support Zones ──")
        if not result.get("support_zones"):
            print("    (none)")
        else:
            for i, z in enumerate(result["support_zones"], 1):
                emoji = _strength_emoji(z["strength"])
                print(
                    f"    S{i} {emoji}  {z['zone_bottom']:.5f} → {z['zone_top']:.5f}"
                    f"  | touches={z['touches']}  rej={z.get('valid_rejections',0)}"
                    f"  | {z['strength']}"
                    f"  | -{z.get('distance_pips','?')} pips"
                )

        piv = result.get("pivot", {})
        if piv:
            print()
            print("  ── Pivot Levels ──")
            print(f"    R2 : {piv.get('R2',0):.5f}   R1 : {piv.get('R1',0):.5f}")
            print(f"    PP : {piv.get('pivot',0):.5f}")
            print(f"    S1 : {piv.get('S1',0):.5f}   S2 : {piv.get('S2',0):.5f}")

        # Location
        sup = result.get("nearest_support")
        res = result.get("nearest_res")
        if sup and res:
            try:
                # Check if price is inside a zone first
                if sup["zone_bottom"] <= cp <= sup["zone_top"]:
                    loc = "🟢 AT SUPPORT — Price testing support zone"
                    print(f"\n  Location : {loc}")
                elif res["zone_bottom"] <= cp <= res["zone_top"]:
                    loc = "🔴 AT RESISTANCE — Price testing resistance zone"
                    print(f"\n  Location : {loc}")
                else:
                    total = res["center"] - sup["center"]
                    if total > 0:
                        pos = (cp - sup["center"]) / total
                        pos = max(0.0, min(1.0, pos))
                        if pos > 0.7:
                            loc = "🔴 Near Resistance — Sell pressure zone"
                        elif pos < 0.3:
                            loc = "🟢 Near Support — Buy pressure zone"
                        else:
                            loc = "🟡 Mid Range — Wait for direction"
                        print(f"\n  Location : {loc}  ({pos*100:.0f}% of range)")
            except Exception as e:
                log.debug(f"[support_resistance] suppressed: {e}")
                pass

        print("═" * 56 + "\n")

    # ─────────────────────────────────────────────
    # AI CONTEXT — for downstream modules
    # ─────────────────────────────────────────────

    def get_ai_context(self, result: dict) -> dict:
        """
        AI Brain / Fibonacci / Market Bias / Signal Engine এর জন্য S/R context.
        Keeps all old keys for backward-compat (nearest_support, nearest_resistance,
        support_strength, resistance_strength, dist_to_support_pips,
        dist_to_resistance_pips, price_location, pivot, R1, S1, role_reversal).

        Location calc uses zone BOUNDARIES (zone_top for support, zone_bottom for
        resistance) so that "price inside zone" is handled gracefully.
        """
        cp = result.get("current_price", 0.0)
        sup = result.get("nearest_support")
        res = result.get("nearest_res")

        # Backward-compat: downstream expects nearest_support/nearest_resistance
        # to be SCALAR price levels (used in Fibonacci, dat_framework, etc).
        # Round-21 audit fix: removed dead `sup_boundary` / `res_boundary`
        # variables. They were computed (closest zone boundary to price)
        # but never used — the actual distance calculations below use
        # zone CENTER (`nearest_sup_price` / `nearest_res_price`), not
        # the boundary. These were leftover from an incomplete boundary-
        # based distance feature. Removing to prevent confusion.
        if sup:
            nearest_sup_price = sup["center"]
        else:
            nearest_sup_price = None

        if res:
            nearest_res_price = res["center"]
        else:
            nearest_res_price = None

        sup_strength = sup["touches"] if sup else 0
        res_strength = res["touches"] if res else 0

        # Pip distances — use zone CENTER for backward compat with downstream
        # code (market_bias.py, etc.) that expects positive distance when price
        # is on the "expected" side of the level.
        # FIX (C-3): was hardcoded / 0.0001, ~100x wrong on JPY pairs and
        # wrong for XAUUSD/indices too. `result["symbol"]` is already
        # populated by analyze() for exactly this purpose.
        pip_value = self._resolve_pip_value(result.get("symbol", ""))
        if nearest_sup_price:
            dist_to_sup = round((cp - nearest_sup_price) / pip_value, 1)
        else:
            dist_to_sup = None
        if nearest_res_price:
            dist_to_res = round((nearest_res_price - cp) / pip_value, 1)
        else:
            dist_to_res = None

        # Location — handle "price inside zone" case explicitly
        location = "mid_range"
        inside_zone = False
        if sup and sup["zone_bottom"] <= cp <= sup["zone_top"]:
            location = "at_support"  # price is testing support
            inside_zone = True
        elif res and res["zone_bottom"] <= cp <= res["zone_top"]:
            location = "at_resistance"  # price is testing resistance
            inside_zone = True
        elif nearest_sup_price and nearest_res_price:
            # Standard range position calc
            total = nearest_res_price - nearest_sup_price
            if total > 0:
                pos = (cp - nearest_sup_price) / total
                pos = max(0.0, min(1.0, pos))  # clamp
                if pos > 0.7:
                    location = "near_resistance"
                elif pos < 0.3:
                    location = "near_support"

        pivot = result.get("pivot", {})

        return {
            # ── Backward-compat keys ──
            "nearest_support":      nearest_sup_price,
            "nearest_resistance":   nearest_res_price,
            "support_strength":     sup_strength,
            "resistance_strength":  res_strength,
            "dist_to_support_pips":    dist_to_sup,
            "dist_to_resistance_pips": dist_to_res,
            "price_location":       location,
            "inside_zone":          inside_zone,
            "pivot":                pivot.get("pivot"),
            "R1":                   pivot.get("R1"),
            "S1":                   pivot.get("S1"),
            "role_reversal":        self._detect_role_reversal(
                cp, nearest_sup_price, nearest_res_price, result
            ),
            # ── v2 Zone keys ──
            "support_zones":        result.get("support_zones", []),
            "resistance_zones":     result.get("resistance_zones", []),
            "all_support_zones":    result.get("all_support_zones", []),
            "all_resistance_zones": result.get("all_resistance_zones", []),
            "current_price":        cp,
            "timeframe":            result.get("timeframe", self.timeframe),
            "cluster_threshold_pct": result.get("cluster_threshold_pct"),
            "swing_window":         result.get("swing_window", self.swing_window),
            "nearest_support_zone": sup,  # full zone dict (v2)
            "nearest_resistance_zone": res,  # full zone dict (v2)
            # Zone summary text — for LLM prompt
            "zone_summary":         self.to_prompt_text(result),
            # JSON for LLM Agent
            "zones_json":           self.to_json(result),
        }

    def _detect_role_reversal(
        self,
        current_price: float,
        support: Optional[float],
        resistance: Optional[float],
        full_result: dict,
    ) -> dict:
        """Day 97+ Book Rule (Page 25): Role Reversal detection."""
        reversal = {
            "detected":      False,
            "type":          None,
            "broken_level":  None,
            "new_role":      None,
            "note":          "No role reversal detected",
        }

        if support and current_price < support:
            reversal.update({
                "detected":     True,
                "type":         "support_to_resistance",
                "broken_level": support,
                "new_role":     "resistance",
                "note":         f"Support {support:.5f} broken — now acts as resistance. Short bias on retest.",
            })

        if resistance and current_price > resistance:
            reversal.update({
                "detected":     True,
                "type":         "resistance_to_support",
                "broken_level": resistance,
                "new_role":     "support",
                "note":         f"Resistance {resistance:.5f} broken — now acts as support. Long bias on retest.",
            })

        return reversal


# ============================================================
# Convenience: detection with LLM-system-prompt output
# ============================================================

def detect_zones_for_llm(
    df: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "H1",
    swing_window: Optional[int] = None,
    cluster_threshold_pct: Optional[float] = None,
    min_touches: int = 2,
    wick_body_ratio: float = 1.5,
    max_zones_per_side: int = 3,
) -> str:
    """
    One-shot helper for LLM Agent integration.

    Pass OHLC df, get spec-compliant JSON back — ready to feed into an
    LLM S/R zone detection agent's context.

    Returns:
        JSON string:
          {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "current_price": 1.0850,
            "resistance_zones": [
              {"zone_top": ..., "zone_bottom": ..., "touches": N, "strength": "..."}
            ],
            "support_zones": [...]
          }
    """
    sr = SupportResistance(
        swing_window=swing_window,
        cluster_threshold_pct=cluster_threshold_pct,
        min_touches=min_touches,
        wick_body_ratio=wick_body_ratio,
        timeframe=timeframe,
        max_zones_per_side=max_zones_per_side,
    )
    result = sr.analyze(df, symbol=symbol)
    return sr.to_json(result)


# ============================================================
# CLI entry — quick test
# ============================================================
if __name__ == "__main__":
    # Synthetic OHLC for smoke test
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 1.0850
    noise = np.cumsum(np.random.randn(n) * 0.0005)
    close = base + noise
    df = pd.DataFrame({
        "open":  close + np.random.randn(n) * 0.0002,
        "high":  close + abs(np.random.randn(n)) * 0.0008,
        "low":   close - abs(np.random.randn(n)) * 0.0008,
        "close": close,
    }, index=dates)

    sr = SupportResistance(timeframe="H1")
    result = sr.analyze(df, symbol="EURUSD")
    sr.get_summary(result)
    print("\n--- JSON (LLM) ---\n")
    print(sr.to_json(result))
    print("\n--- Prompt text ---\n")
    print(sr.to_prompt_text(result))