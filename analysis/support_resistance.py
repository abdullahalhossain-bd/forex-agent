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
#
# Contract / assumptions (read before using in backtest):
#   - Callers should pass a DataFrame of *closed* bars (or freeze the
#     forming bar). Swing confirmation has a built-in lag of swing_window
#     bars by design; the last possible confirmed swing sits at
#     index len(df)-swing_window-1.
#   - Rejection counts and strength scores are computed on the *full*
#     supplied DataFrame (current live strength). For pure historical
#     walk-forward, slice the DataFrame up to the decision bar before
#     calling analyze().
# ============================================================

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("support_resistance")

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


def _safe_series(col: Any) -> pd.Series:
    """Return a clean 1-D float Series even when duplicate column labels exist."""
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col.astype(float)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as % of price — used for adaptive cluster threshold.

    Fully positional (numpy) after extraction so duplicate indexes or
    columns can never re-introduce Series-in-boolean errors.
    """
    global _atr_pct_diag_logged
    try:
        if len(df) < period + 1:
            return 0.004  # default 0.4%
        h = _safe_series(df["high"]).to_numpy(dtype=float, copy=False)
        l = _safe_series(df["low"]).to_numpy(dtype=float, copy=False)
        c = _safe_series(df["close"]).to_numpy(dtype=float, copy=False)
        if h.ndim != 1 or l.ndim != 1 or c.ndim != 1:
            raise ValueError(
                f"non-1D array after extraction: h={h.shape} l={l.shape} c={c.shape}"
            )
        c_prev = np.roll(c, 1)
        c_prev[0] = np.nan
        tr = np.nanmax(
            np.vstack([
                h - l,
                np.abs(h - c_prev),
                np.abs(l - c_prev),
            ]),
            axis=0,
        )
        window = tr[-period:]
        atr = float(np.nanmean(window)) if window.size else float("nan")
        price = float(c[-1])
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
    AI Trader S/R Zone detection engine (v2 — Zone-Based).

    Output format per zone:
      {
        "zone_top":    <price>,
        "zone_bottom": <price>,
        "center":      <price>,         # backward-compat
        "touches":     <count>,
        "strength":    "Weak|Medium|Strong",
        "role":        "support|resistance",
        "last_touch_time": <ISO>,
        "distance_pips":   <float>,     # from current price (relevant zones only)
        "source":      "cluster|eqh_eql|raw_swing",
        "is_thin_zone": bool,
        "elevated_breakout_risk": bool,
      }
    """

    def __init__(
        self,
        window: int = 4,
        tolerance: float = 0.0015,
        swing_window: Optional[int] = None,
        cluster_threshold_pct: Optional[float] = None,
        min_touches: int = 2,
        wick_body_ratio: float = 1.5,
        timeframe: str = "H1",
        max_zones_per_side: int = 3,
        trending_eqh_loose_multiplier: float = 2.0,
        max_raw_swing_levels: int = 3,
    ):
        self.window = window
        self.tolerance = tolerance

        self.timeframe = (timeframe or "H1").upper()
        self.swing_window = swing_window or _TF_SWING_WINDOW.get(self.timeframe, 4)
        self.cluster_threshold_pct = cluster_threshold_pct
        self.min_touches = max(2, min_touches)
        self.wick_body_ratio = wick_body_ratio
        self.max_zones_per_side = max_zones_per_side
        self.trending_eqh_loose_multiplier = trending_eqh_loose_multiplier
        self.max_raw_swing_levels = max_raw_swing_levels

    # ─────────────────────────────────────────────
    # STEP 1: Swing High & Low detection
    # ─────────────────────────────────────────────

    def find_swing_highs(self, df: pd.DataFrame) -> List[Dict]:
        """Strict local swing high: high[i] > max(left window) and > max(right window).

        Confirmation lag = swing_window bars (by design). Only emits swings
        that are fully confirmed by closed bars on both sides.
        """
        swing_highs: List[Dict] = []
        w = self.swing_window
        if len(df) < 2 * w + 1:
            return swing_highs

        highs = _safe_series(df["high"]).to_numpy(dtype=float, copy=False)
        for i in range(w, len(df) - w):
            left_max = highs[i - w : i].max()
            right_max = highs[i + 1 : i + w + 1].max()
            if highs[i] > left_max and highs[i] > right_max:
                swing_highs.append({
                    "index": i,
                    "time": df.index[i],
                    "price": float(highs[i]),
                })
        return swing_highs

    def find_swing_lows(self, df: pd.DataFrame) -> List[Dict]:
        """Strict local swing low: low[i] < min(left window) and < min(right window)."""
        swing_lows: List[Dict] = []
        w = self.swing_window
        if len(df) < 2 * w + 1:
            return swing_lows

        lows = _safe_series(df["low"]).to_numpy(dtype=float, copy=False)
        for i in range(w, len(df) - w):
            left_min = lows[i - w : i].min()
            right_min = lows[i + 1 : i + w + 1].min()
            if lows[i] < left_min and lows[i] < right_min:
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
        """Upper/lower wick >= wick_body_ratio × body (doji = any wick)."""
        try:
            o = float(candle["open"])
            h = float(candle["high"])
            l = float(candle["low"])
            c = float(candle["close"])
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            wick = upper_wick if direction == "resistance" else lower_wick
            if body < 1e-9:
                return wick > 0
            return wick >= self.wick_body_ratio * body
        except Exception as e:
            log.debug(f"[SR] _is_valid_rejection failed: {e}")
            return False

    def _count_valid_rejections(
        self,
        df: pd.DataFrame,
        zone_top: float,
        zone_bottom: float,
        direction: str,
        proximity_pct: float = 0.0015,
    ) -> int:
        """Vectorised count of valid rejection candles that touched the zone."""
        try:
            close = _safe_series(df["close"])
            cp = float(close.iloc[-1])
            if cp <= 0:
                return 0
            band = cp * proximity_pct

            high = _safe_series(df["high"])
            low = _safe_series(df["low"])
            open_ = _safe_series(df["open"])

            if direction == "resistance":
                mask = (high >= zone_bottom - band) & (high <= zone_top + band)
            else:
                mask = (low <= zone_top + band) & (low >= zone_bottom - band)

            if not mask.any():
                return 0

            o = open_[mask].to_numpy(dtype=float, copy=False)
            h = high[mask].to_numpy(dtype=float, copy=False)
            l = low[mask].to_numpy(dtype=float, copy=False)
            c = close[mask].to_numpy(dtype=float, copy=False)

            body = np.abs(c - o)
            upper_wick = h - np.maximum(o, c)
            lower_wick = np.minimum(o, c) - l
            wick = upper_wick if direction == "resistance" else lower_wick

            is_doji = body < 1e-9
            valid = np.where(is_doji, wick > 0, wick >= self.wick_body_ratio * body)
            return int(np.sum(valid))
        except Exception as e:
            log.debug(
                f"[SR] _count_valid_rejections failed for zone "
                f"[{zone_bottom}, {zone_top}]: {e}"
            )
            return 0

    # ─────────────────────────────────────────────
    # STEP 3: Cluster swing points into ZONES
    # ─────────────────────────────────────────────

    def _get_cluster_threshold(
        self, df: pd.DataFrame, atr_pct: Optional[float] = None
    ) -> float:
        """ATR-adaptive cluster threshold (% of price), floored/ceiled."""
        if self.cluster_threshold_pct is not None:
            return self.cluster_threshold_pct
        if atr_pct is None:
            atr_pct = _atr_pct(df, period=14)
        return max(0.001, min(0.008, atr_pct * 1.5))

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
        """Cluster nearby swing prices into zones (anti-drift centre+boundary test)."""
        if not swing_points:
            return []

        threshold_pct = (
            threshold_pct_override
            if threshold_pct_override is not None
            else self._get_cluster_threshold(df, atr_pct=atr_pct)
        )
        min_touches = (
            min_touches_override
            if min_touches_override is not None
            else self.min_touches
        )

        sorted_pts = sorted(swing_points, key=lambda p: p["price"])
        current_cluster = [sorted_pts[0]]
        zones = []

        for p in sorted_pts[1:]:
            cluster_prices = [pt["price"] for pt in current_cluster]
            cluster_center = float(np.mean(cluster_prices))
            cluster_min = min(cluster_prices)
            cluster_max = max(cluster_prices)

            dist_to_center = (
                abs(p["price"] - cluster_center) / cluster_center
                if cluster_center > 0 else 1.0
            )
            dist_to_min = (
                abs(p["price"] - cluster_min) / cluster_min
                if cluster_min > 0 else 1.0
            )
            dist_to_max = (
                abs(p["price"] - cluster_max) / cluster_max
                if cluster_max > 0 else 1.0
            )
            dist_to_nearest = min(dist_to_min, dist_to_max)

            if dist_to_center <= threshold_pct and dist_to_nearest <= threshold_pct:
                current_cluster.append(p)
            else:
                if len(current_cluster) >= min_touches:
                    zones.append(
                        self._build_zone(
                            current_cluster, df, direction,
                            source=source, atr_pct=atr_pct,
                        )
                    )
                current_cluster = [p]

        if len(current_cluster) >= min_touches:
            zones.append(
                self._build_zone(
                    current_cluster, df, direction,
                    source=source, atr_pct=atr_pct,
                )
            )
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
        """Trending-regime fallback: recent single-touch swing levels as thin zones."""
        if not swing_points:
            return []
        max_levels = max_levels if max_levels is not None else self.max_raw_swing_levels

        if atr_pct is None:
            atr_pct = _atr_pct(df, period=14)
        buffer_pct = max(atr_pct * buffer_multiplier, 0.0003)

        recent = sorted(swing_points, key=lambda p: p["index"], reverse=True)[:max_levels]
        price_ref = float(_safe_series(df["close"]).iloc[-1]) if len(df) else 0.0
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
            valid_rej = self._count_valid_rejections(
                df, zone_top, zone_bottom, direction=direction
            )
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

    def _merge_zone_sources(
        self, *tiers: list, df: pd.DataFrame, atr_pct: Optional[float] = None
    ) -> list:
        """Merge tiers highest-confidence first; drop lower-tier overlaps."""
        threshold_pct = self._get_cluster_threshold(df, atr_pct=atr_pct)
        merged: list = []
        for tier in tiers:
            for z in tier:
                ref = z["center"] if z["center"] > 0 else 1.0
                if any(
                    abs(z["center"] - m["center"]) / ref <= threshold_pct
                    for m in merged
                ):
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

        valid_rej = self._count_valid_rejections(
            df, zone_top, zone_bottom, direction=direction
        )
        if valid_rej >= 4 and strength == "Medium":
            strength = "Strong"
        elif valid_rej >= 3 and strength == "Weak":
            strength = "Medium"

        # EQH/EQL evidence-based upgrade (walk-forward confirmed)
        if source == "eqh_eql" and strength == "Weak":
            strength = "Medium"

        # Thin-zone downgrade (width vs ATR is the strongest outcome predictor)
        zone_width = zone_top - zone_bottom
        atr_pct_now = atr_pct if atr_pct is not None else _atr_pct(df, period=14)
        price_ref = float(_safe_series(df["close"]).iloc[-1]) if len(df) else 0.0
        atr_abs = atr_pct_now * price_ref
        is_thin_zone = bool(atr_abs > 0 and zone_width < 0.5 * atr_abs)
        if is_thin_zone:
            if strength == "Strong":
                strength = "Medium"
            elif strength == "Medium":
                strength = "Weak"

        elevated_breakout_risk = bool(strength == "Medium" and not is_thin_zone)

        return {
            "zone_top": round(zone_top, 5),
            "zone_bottom": round(zone_bottom, 5),
            "center": round(center, 5),
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
        """Simple absolute-tolerance clustering (legacy callers)."""
        if not levels:
            return []
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
    # STEP 4: Pivot Point Calculation
    # ─────────────────────────────────────────────

    def calculate_pivot(self, df: pd.DataFrame) -> dict:
        """Classic Pivot from the previous complete candle."""
        if len(df) < 2:
            log.debug("[SR] calculate_pivot: need >=2 candles, got %d", len(df))
            return {}
        prev = df.iloc[-2]
        H = float(prev["high"])
        L = float(prev["low"])
        C = float(prev["close"])
        pivot = (H + L + C) / 3
        return {
            "pivot": round(pivot, 5),
            "R1": round(2 * pivot - L, 5),
            "R2": round(pivot + (H - L), 5),
            "S1": round(2 * pivot - H, 5),
            "S2": round(pivot - (H - L), 5),
        }

    # ─────────────────────────────────────────────
    # STEP 5: Nearest S/R
    # ─────────────────────────────────────────────

    def find_nearest_levels(
        self,
        current_price: float,
        support_zones: list,
        resistance_zones: list,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """Nearest support/resistance by centre, with inside-zone fallback."""
        sup_below = [z for z in support_zones if z["center"] <= current_price]
        nearest_sup = max(sup_below, key=lambda z: z["center"]) if sup_below else None

        res_above = [z for z in resistance_zones if z["center"] >= current_price]
        nearest_res = min(res_above, key=lambda z: z["center"]) if res_above else None

        if nearest_sup is None and support_zones:
            overlapping = [
                z for z in support_zones
                if z["zone_bottom"] <= current_price <= z["zone_top"]
            ]
            if overlapping:
                nearest_sup = max(overlapping, key=lambda z: z["zone_top"])
        if nearest_res is None and resistance_zones:
            overlapping = [
                z for z in resistance_zones
                if z["zone_bottom"] <= current_price <= z["zone_top"]
            ]
            if overlapping:
                nearest_res = min(overlapping, key=lambda z: z["zone_bottom"])

        return nearest_sup, nearest_res

    # ─────────────────────────────────────────────
    # STEP 6: Filter to top N relevant zones
    # ─────────────────────────────────────────────

    def _filter_relevant_zones(
        self,
        zones: list,
        current_price: float,
        max_zones: int = 3,
        side: str = "support",
    ) -> list:
        """Return copies of the most relevant zones (includes zones that contain price)."""
        if not zones:
            return []

        if side == "support":
            relevant = [z for z in zones if z["zone_bottom"] <= current_price]
        else:
            relevant = [z for z in zones if z["zone_top"] >= current_price]

        strength_weight = {"Strong": 3, "Medium": 2, "Weak": 1}

        def _sort_key(z):
            if side == "support":
                dist = current_price - z["center"]
            else:
                dist = z["center"] - current_price
            dist = max(dist, 1e-9)
            recency = z.get("last_touch_index", 0) + 1
            score = strength_weight.get(z["strength"], 1) * (recency / 100) / dist
            return -score

        relevant.sort(key=_sort_key)
        # Return shallow copies so later distance_pips mutation cannot
        # pollute the all_* lists that share the original objects.
        return [dict(z) for z in relevant[:max_zones]]

    @staticmethod
    def _resolve_pip_value(symbol: str) -> float:
        """Instrument-aware pip size (shared by analyze + get_ai_context)."""
        s = (symbol or "").upper()
        if s.endswith("JPY"):
            return 0.01
        if s == "XAUUSD":
            return 0.1
        if s in ("US30", "NAS100", "SPX500", "GER40"):
            return 1.0
        return 0.0001

    def _attach_distance_pips(
        self, zones: list, current_price: float, pip_value: float
    ) -> list:
        """Add distance_pips (mutates the *copies* produced by the filter)."""
        for z in zones:
            z["distance_pips"] = round(
                abs(z["center"] - current_price) / pip_value, 1
            )
        return zones

    # ─────────────────────────────────────────────
    # STEP 7: FULL PIPELINE
    # ─────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> dict:
        """Full S/R Zone analysis pipeline."""
        if len(df) < 2 * self.swing_window + 5:
            log.warning(
                f"[SR] Insufficient candles ({len(df)}) for "
                f"swing_window={self.swing_window}"
            )

        atr_pct = _atr_pct(df, period=14)

        swing_highs = self.find_swing_highs(df)
        swing_lows = self.find_swing_lows(df)

        # Tier 1 — tight multi-touch clusters
        all_resistance = self.cluster_into_zones(
            swing_highs, df, direction="resistance", atr_pct=atr_pct
        )
        all_support = self.cluster_into_zones(
            swing_lows, df, direction="support", atr_pct=atr_pct
        )

        # Tier 2 — looser EQH/EQL
        loose_threshold = (
            self._get_cluster_threshold(df, atr_pct=atr_pct)
            * self.trending_eqh_loose_multiplier
        )
        eqh_resistance = self.cluster_into_zones(
            swing_highs, df, direction="resistance",
            threshold_pct_override=loose_threshold, source="eqh_eql", atr_pct=atr_pct,
        )
        eql_support = self.cluster_into_zones(
            swing_lows, df, direction="support",
            threshold_pct_override=loose_threshold, source="eqh_eql", atr_pct=atr_pct,
        )

        # Tier 3 — raw recent swings
        raw_resistance = self._raw_swing_levels(
            swing_highs, df, direction="resistance", atr_pct=atr_pct
        )
        raw_support = self._raw_swing_levels(
            swing_lows, df, direction="support", atr_pct=atr_pct
        )

        all_resistance = self._merge_zone_sources(
            all_resistance, eqh_resistance, raw_resistance, df=df, atr_pct=atr_pct
        )
        all_support = self._merge_zone_sources(
            all_support, eql_support, raw_support, df=df, atr_pct=atr_pct
        )

        # Diagnostic (debug only)
        _src_counts_r: Dict[str, int] = {}
        _src_counts_s: Dict[str, int] = {}
        for z in all_resistance:
            src = z.get("source", "?")
            _src_counts_r[src] = _src_counts_r.get(src, 0) + 1
        for z in all_support:
            src = z.get("source", "?")
            _src_counts_s[src] = _src_counts_s.get(src, 0) + 1
        log.debug(
            f"[SR-DIAG] {symbol}: resistance={len(all_resistance)} {_src_counts_r} | "
            f"support={len(all_support)} {_src_counts_s}"
        )

        try:
            pivot = self.calculate_pivot(df)
        except Exception:
            pivot = {}

        current_price = float(_safe_series(df["close"]).iloc[-1])

        relevant_support = self._filter_relevant_zones(
            all_support, current_price,
            max_zones=self.max_zones_per_side, side="support",
        )
        relevant_resistance = self._filter_relevant_zones(
            all_resistance, current_price,
            max_zones=self.max_zones_per_side, side="resistance",
        )

        pip_value = self._resolve_pip_value(symbol)
        relevant_support = self._attach_distance_pips(
            relevant_support, current_price, pip_value
        )
        relevant_resistance = self._attach_distance_pips(
            relevant_resistance, current_price, pip_value
        )

        # Nearest must be computed on the *unfiltered* lists
        nearest_sup, nearest_res = self.find_nearest_levels(
            current_price, all_support, all_resistance
        )
        if nearest_sup is not None and "distance_pips" not in nearest_sup:
            nearest_sup = dict(nearest_sup)
            nearest_sup["distance_pips"] = round(
                abs(nearest_sup["center"] - current_price) / pip_value, 1
            )
        if nearest_res is not None and "distance_pips" not in nearest_res:
            nearest_res = dict(nearest_res)
            nearest_res["distance_pips"] = round(
                abs(nearest_res["center"] - current_price) / pip_value, 1
            )

        return {
            "support_zones": relevant_support,
            "resistance_zones": relevant_resistance,
            "all_support_zones": all_support,
            "all_resistance_zones": all_resistance,
            "pivot": pivot,
            "nearest_support": nearest_sup,
            "nearest_res": nearest_res,
            "current_price": current_price,
            "symbol": symbol,
            "timeframe": self.timeframe,
            "swing_window": self.swing_window,
            "cluster_threshold_pct": self._get_cluster_threshold(df, atr_pct=atr_pct),
            "min_touches": self.min_touches,
            "wick_body_ratio": self.wick_body_ratio,
        }

    # ─────────────────────────────────────────────
    # JSON / Prompt / Summary / AI context
    # ─────────────────────────────────────────────

    def to_json(self, result: dict) -> str:
        def _slim(zones):
            return [
                {
                    "zone_top": z["zone_top"],
                    "zone_bottom": z["zone_bottom"],
                    "touches": z["touches"],
                    "strength": z["strength"],
                    "distance_pips": z.get("distance_pips"),
                    "last_touch_time": z.get("last_touch_time"),
                    "is_thin_zone": z.get("is_thin_zone", False),
                    "elevated_breakout_risk": z.get("elevated_breakout_risk", False),
                }
                for z in zones
            ]

        payload = {
            "symbol": result.get("symbol", ""),
            "timeframe": result.get("timeframe", ""),
            "current_price": round(result.get("current_price", 0.0), 5),
            "resistance_zones": _slim(result.get("resistance_zones", [])),
            "support_zones": _slim(result.get("support_zones", [])),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _risk_suffix(z: dict) -> str:
        tags = []
        if z.get("is_thin_zone"):
            tags.append("THIN-vs-ATR")
        if z.get("elevated_breakout_risk"):
            tags.append("3RD-TOUCH-RISK")
        return f" [{','.join(tags)}]" if tags else ""

    def to_prompt_text(self, result: dict) -> str:
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
                    f"| dist={z.get('distance_pips', '?')} pips"
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
                    f"| dist={z.get('distance_pips', '?')} pips"
                    f"{self._risk_suffix(z)}"
                )
        lines.append("=" * 50)
        return "\n".join(lines)

    def get_summary(self, result: dict) -> None:
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
                    f"  | touches={z['touches']}  rej={z.get('valid_rejections', 0)}"
                    f"  | {z['strength']}"
                    f"  | +{z.get('distance_pips', '?')} pips"
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
                    f"  | touches={z['touches']}  rej={z.get('valid_rejections', 0)}"
                    f"  | {z['strength']}"
                    f"  | -{z.get('distance_pips', '?')} pips"
                )

        piv = result.get("pivot", {})
        if piv:
            print()
            print("  ── Pivot Levels ──")
            print(f"    R2 : {piv.get('R2', 0):.5f}   R1 : {piv.get('R1', 0):.5f}")
            print(f"    PP : {piv.get('pivot', 0):.5f}")
            print(f"    S1 : {piv.get('S1', 0):.5f}   S2 : {piv.get('S2', 0):.5f}")

        sup = result.get("nearest_support")
        res = result.get("nearest_res")
        if sup and res:
            try:
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
                log.debug(f"[support_resistance] location calc suppressed: {e}")

        print("═" * 56 + "\n")

    def get_ai_context(self, result: dict) -> dict:
        """Compact context for downstream AI / bias / signal modules."""
        cp = result.get("current_price", 0.0)
        sup = result.get("nearest_support")
        res = result.get("nearest_res")

        nearest_sup_price = sup["center"] if sup else None
        nearest_res_price = res["center"] if res else None

        sup_strength = sup["touches"] if sup else 0
        res_strength = res["touches"] if res else 0

        pip_value = self._resolve_pip_value(result.get("symbol", ""))
        dist_to_sup = (
            round((cp - nearest_sup_price) / pip_value, 1)
            if nearest_sup_price is not None else None
        )
        dist_to_res = (
            round((nearest_res_price - cp) / pip_value, 1)
            if nearest_res_price is not None else None
        )

        location = "mid_range"
        inside_zone = False
        if sup and sup["zone_bottom"] <= cp <= sup["zone_top"]:
            location = "at_support"
            inside_zone = True
        elif res and res["zone_bottom"] <= cp <= res["zone_top"]:
            location = "at_resistance"
            inside_zone = True
        elif nearest_sup_price is not None and nearest_res_price is not None:
            total = nearest_res_price - nearest_sup_price
            if total > 0:
                pos = max(0.0, min(1.0, (cp - nearest_sup_price) / total))
                if pos > 0.7:
                    location = "near_resistance"
                elif pos < 0.3:
                    location = "near_support"

        pivot = result.get("pivot", {})

        return {
            "nearest_support": nearest_sup_price,
            "nearest_resistance": nearest_res_price,
            "support_strength": sup_strength,
            "resistance_strength": res_strength,
            "dist_to_support_pips": dist_to_sup,
            "dist_to_resistance_pips": dist_to_res,
            "price_location": location,
            "inside_zone": inside_zone,
            "pivot": pivot.get("pivot"),
            "R1": pivot.get("R1"),
            "S1": pivot.get("S1"),
            "role_reversal": self._detect_role_reversal(
                cp, nearest_sup_price, nearest_res_price, result
            ),
            "support_zones": result.get("support_zones", []),
            "resistance_zones": result.get("resistance_zones", []),
            "all_support_zones": result.get("all_support_zones", []),
            "all_resistance_zones": result.get("all_resistance_zones", []),
            "current_price": cp,
            "timeframe": result.get("timeframe", self.timeframe),
            "cluster_threshold_pct": result.get("cluster_threshold_pct"),
            "swing_window": result.get("swing_window", self.swing_window),
            "nearest_support_zone": sup,
            "nearest_resistance_zone": res,
            "zone_summary": self.to_prompt_text(result),
            "zones_json": self.to_json(result),
        }

    def _detect_role_reversal(
        self,
        current_price: float,
        support: Optional[float],
        resistance: Optional[float],
        full_result: dict,
    ) -> dict:
        """Simple break-of-nearest role-reversal flag."""
        reversal = {
            "detected": False,
            "type": None,
            "broken_level": None,
            "new_role": None,
            "note": "No role reversal detected",
        }
        if support is not None and current_price < support:
            reversal.update({
                "detected": True,
                "type": "support_to_resistance",
                "broken_level": support,
                "new_role": "resistance",
                "note": (
                    f"Support {support:.5f} broken — now acts as resistance. "
                    "Short bias on retest."
                ),
            })
        if resistance is not None and current_price > resistance:
            reversal.update({
                "detected": True,
                "type": "resistance_to_support",
                "broken_level": resistance,
                "new_role": "support",
                "note": (
                    f"Resistance {resistance:.5f} broken — now acts as support. "
                    "Long bias on retest."
                ),
            })
        return reversal


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
    """One-shot helper → spec-compliant JSON for LLM context."""
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


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 1.0850
    noise = np.cumsum(np.random.randn(n) * 0.0005)
    close = base + noise
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.0002,
        "high": close + np.abs(np.random.randn(n)) * 0.0008,
        "low": close - np.abs(np.random.randn(n)) * 0.0008,
        "close": close,
    }, index=dates)

    sr = SupportResistance(timeframe="H1")
    result = sr.analyze(df, symbol="EURUSD")
    sr.get_summary(result)
    print("\n--- JSON (LLM) ---\n")
    print(sr.to_json(result))
    print("\n--- Prompt text ---\n")
    print(sr.to_prompt_text(result))