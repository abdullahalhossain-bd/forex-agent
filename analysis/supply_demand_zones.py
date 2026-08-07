
"""
analysis/supply_demand_zones.py — Day 97+ Supply/Demand Zones
================================================================
Candlestick Bible: Supply/Demand zones are stronger than S/R because they
reflect institutional order flow, not just prior swing points.
 
Three criteria for a quality zone (from the book):
  1. Strength/speed of the move away from the zone (fast = institutional)
  2. Favorable risk/reward when traded
  3. Higher time frame zones (4H/daily) are most significant
 
Usage:
    from analysis.supply_demand_zones import SupplyDemandZones
    sd = SupplyDemandZones()
    result = sd.detect(df)
    # → {"demand_zones": [...], "supply_zones": [...], "nearest_demand": ..., "nearest_supply": ...}
"""
 
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from utils.logger import get_logger
 
log = get_logger("supply_demand")
 
# ── Pip-size helper (institutional review, Finding C-3) ─────────────────
# This module previously hardcoded pip = 0.0001 in _nearest() and
# calculate_entry_stop_tp(), which is wrong by ~100x for JPY-quoted pairs
# (1 pip = 0.01, not 0.0001). Prefer the project-wide shared helper if it's
# available (already used correctly in stop_hunt_signal_engine.py); fall
# back to a local, JPY-aware implementation otherwise so this module still
# works standalone.
try:
    from analysis._engine_utils import pip_value as _pip_value
except ImportError:  # pragma: no cover - fallback if _engine_utils is unavailable
    def _pip_value(symbol: str = "", price: Optional[float] = None) -> float:
        """Local fallback pip-size resolver.
 
        JPY-quoted pairs (e.g. USDJPY, EURJPY) use 0.01 as one pip; all other
        FX majors/crosses use 0.0001. If symbol isn't available, fall back to
        a price-magnitude heuristic (JPY-quoted pairs trade well above 20).
        """
        s = (symbol or "").upper()
        if "JPY" in s:
            return 0.01
        if price is not None and price > 20:
            return 0.01
        return 0.0001
 
 
class SupplyDemandZones:
    """Detects institutional supply/demand zones.
 
    Book: "Supply/demand zones are a stronger version of S/R, attributed
    to institutional order flow rather than just prior swing points."
 
    A demand zone = base of a strong bullish rally (institutions bought heavily).
    A supply zone = base of a strong bearish drop (institutions sold heavily).
    """
 
    # Config
    MIN_RALLY_CANDLES = 3       # minimum candles in the rally away from zone
    MIN_RALLY_PCT = 0.3         # rally must be at least 0.3% to qualify
    ZONE_TOLERANCE = 0.0005     # how close to zone = "at zone"
    MAX_ZONES = 5               # keep only top N strongest zones
    LONG_TAIL_WICK_RATIO = 0.5  # Book P42: majority-of-base long-tail disqualification threshold
    # Selection weighting used when trimming to MAX_ZONES (Finding #7): a pure
    # strength sort can push a recent, price-relevant zone out in favor of an
    # old, far-away-but-strong one. Recency and proximity are blended in
    # alongside strength so nearest_demand/nearest_supply stay meaningful.
    SELECTION_STRENGTH_WEIGHT = 0.5
    SELECTION_RECENCY_WEIGHT = 0.25
    SELECTION_PROXIMITY_WEIGHT = 0.25
 
    def detect(self, df: pd.DataFrame, symbol: str = "") -> Dict[str, Any]:
        """Detect supply and demand zones from OHLCV data.
 
        Args:
            df: OHLCV DataFrame.
            symbol: optional instrument symbol (e.g. "USDJPY"), used only to
                resolve the correct pip size for distance_pips on the nearest
                zone. Optional and defaults to "" to preserve backward
                compatibility with existing detect(df) callers — in that case
                pip size falls back to a price-magnitude heuristic.
 
        Zones are ERC-validated (>=2 extended-range candles in the departure
        move, Book P35) and disqualification-checked (Doji-only/staircase/
        long-tailed base, Book Ch.4) before being accepted — see
        has_valid_impulse() / is_zone_tradable().
 
        Returns:
            {
                "demand_zones": [{"zone_low": float, "zone_high": float,
                                   "zone_mid": float, "distal": float, "proximal": float,
                                   "base_idx": int, "base_idx_start": int, "base_idx_end": int,
                                   "strength": int, "rally_pct": float, "age_bars": int,
                                   "is_fresh": bool, "test_count": int, "is_original": bool,
                                   "quality_score": int, "quality_grade": str,
                                   "sd_pattern": str, "sd_pattern_type": str}],
                "supply_zones": [...],  # same shape, "drop_pct" instead of "rally_pct"
                "nearest_demand": {"price": float, "distance_pips": float} | None,
                "nearest_supply": {"price": float, "distance_pips": float} | None,
                "current_price": float,
                "balance_imbalance": dict,
            }
        """
        if len(df) < 10:
            return self._empty_result()
 
        # Sanitize
        df = df.copy()
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
 
        # FIX (minor finding): to_numeric(errors="coerce") turns unparsable
        # ticks into NaN silently. Without this check those NaNs propagate
        # into zone_low/zone_high/zone_mid (e.g. round(nan, 5) == nan), and
        # every downstream distance/quality calc against that zone silently
        # produces NaN or a bogus comparison rather than a clear failure.
        bad_rows = df[["open", "high", "low", "close"]].isna().any(axis=1).sum()
        if bad_rows:
            log.warning(f"[SupplyDemand] dropping {bad_rows} row(s) with non-numeric OHLC data")
            df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        if len(df) < 10:
            return self._empty_result()
 
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(df)
        current_price = float(close[-1])
 
        demand_zones = []
        supply_zones = []
 
        # Find demand zones: base of strong bullish rallies
        for i in range(self.MIN_RALLY_CANDLES, n - self.MIN_RALLY_CANDLES):
            rally_start = i
            rally_end = min(i + self.MIN_RALLY_CANDLES, n - 1)
 
            rally_pct = (close[rally_end] - close[rally_start]) / close[rally_start] * 100
            if rally_pct < self.MIN_RALLY_PCT:
                continue
 
            # The demand zone is the base candle(s) before the rally
            base_low = float(min(low[rally_start-1], low[rally_start]))
            base_high = float(max(high[rally_start-1], high[rally_start]))
 
            # Book Ch.3: Classify pattern type (Drop-Base-Rally or Rally-Base-Rally)
            # Check direction of move BEFORE the base (pre-base impulse)
            pre_base_idx = max(0, rally_start - 2)
            pre_base_close = float(close[pre_base_idx])
            base_close = float(close[rally_start])
            pre_move_is_drop = base_close < pre_base_close  # price fell into base
            pre_move_is_rally = base_close > pre_base_close  # price rose into base
 
            if pre_move_is_drop:
                sd_pattern = "Drop-Base-Rally"  # Reversal: demand zone
                sd_pattern_type = "Reversal"
            elif pre_move_is_rally:
                sd_pattern = "Rally-Base-Rally"  # Continuation: demand zone
                sd_pattern_type = "Continuation"
            else:
                sd_pattern = "Unknown-Rally"
                sd_pattern_type = "Unknown"
 
            # Strength = rally speed (faster = stronger institutional)
            strength = min(100, int(rally_pct * 20))
 
            # FIX (institutional review, Finding #2/#3): the book's actual
            # zone-drawing and validation methods existed on the class but
            # detect() never called them, so every returned zone was just a
            # raw 2-candle low/high midpoint — not ERC-validated, not
            # disqualification-checked, and with no distal/proximal lines
            # (which odd_enhancers.py's score_zone() requires — see its
            # docstring). Wired in here:
            base_idx_start = rally_start - 1
            base_idx_end = rally_start + 1  # exclusive, matches base slice below
            base_candles = df.iloc[base_idx_start:base_idx_end]
 
            tradable = self.is_zone_tradable(base_candles, "demand")
            if not tradable["tradable"]:
                continue  # Doji-only / staircase / long-tailed base — Book Ch.4
 
            if not self.has_valid_impulse(df, rally_start, rally_end):
                continue  # <2 ERCs in the departure move — not a genuine imbalance (Book P35)
 
            drawn = self.draw_zone_medium_risk(base_candles, "demand")
 
            demand_zones.append({
                "zone_low": round(base_low, 5),
                "zone_high": round(base_high, 5),
                "zone_mid": round((base_low + base_high) / 2, 5),
                "distal": drawn["distal"],      # FIX #3: schema match for odd_enhancers.py
                "proximal": drawn["proximal"],  # FIX #3: schema match for odd_enhancers.py
                "base_idx": base_idx_start,          # FIX #5: was never set (originality check was a no-op)
                "base_idx_start": base_idx_start,    # FIX #3: schema match for odd_enhancers.py
                "base_idx_end": base_idx_end,        # FIX #3: schema match for odd_enhancers.py
                "strength": strength,
                "rally_pct": round(rally_pct, 2),
                "age_bars": n - rally_start,
                "test_count": 0,  # placeholder — replaced with real check_zone_freshness() result below
                "is_fresh": True,  # placeholder — replaced with real check_zone_freshness() result below
                "sd_pattern": sd_pattern,  # Book Ch.3: Drop-Base-Rally or Rally-Base-Rally
                "sd_pattern_type": sd_pattern_type,  # Reversal or Continuation
            })
 
        # Find supply zones: base of strong bearish drops
        for i in range(self.MIN_RALLY_CANDLES, n - self.MIN_RALLY_CANDLES):
            drop_start = i
            drop_end = min(i + self.MIN_RALLY_CANDLES, n - 1)
 
            drop_pct = (close[drop_start] - close[drop_end]) / close[drop_start] * 100
            if drop_pct < self.MIN_RALLY_PCT:
                continue
 
            base_low = float(min(low[drop_start-1], low[drop_start]))
            base_high = float(max(high[drop_start-1], high[drop_start]))
 
            # Book Ch.3: Classify pattern type (Rally-Base-Drop or Drop-Base-Drop)
            pre_base_idx = max(0, drop_start - 2)
            pre_base_close = float(close[pre_base_idx])
            base_close = float(close[drop_start])
            pre_move_is_rally = base_close > pre_base_close  # price rose into base
            pre_move_is_drop = base_close < pre_base_close  # price fell into base
 
            if pre_move_is_rally:
                sd_pattern = "Rally-Base-Drop"  # Reversal: supply zone
                sd_pattern_type = "Reversal"
            elif pre_move_is_drop:
                sd_pattern = "Drop-Base-Drop"  # Continuation: supply zone
                sd_pattern_type = "Continuation"
            else:
                sd_pattern = "Unknown-Drop"
                sd_pattern_type = "Unknown"
 
            strength = min(100, int(drop_pct * 20))
 
            # FIX (institutional review, Finding #2/#3): see matching comment
            # in the demand-zone loop above — same wiring, mirrored for supply.
            base_idx_start = drop_start - 1
            base_idx_end = drop_start + 1
            base_candles = df.iloc[base_idx_start:base_idx_end]
 
            tradable = self.is_zone_tradable(base_candles, "supply")
            if not tradable["tradable"]:
                continue
 
            if not self.has_valid_impulse(df, drop_start, drop_end):
                continue
 
            drawn = self.draw_zone_medium_risk(base_candles, "supply")
 
            supply_zones.append({
                "zone_low": round(base_low, 5),
                "zone_high": round(base_high, 5),
                "zone_mid": round((base_low + base_high) / 2, 5),
                "distal": drawn["distal"],      # FIX #3: schema match for odd_enhancers.py
                "proximal": drawn["proximal"],  # FIX #3: schema match for odd_enhancers.py
                "base_idx": base_idx_start,          # FIX #5: was never set (originality check was a no-op)
                "base_idx_start": base_idx_start,    # FIX #3: schema match for odd_enhancers.py
                "base_idx_end": base_idx_end,        # FIX #3: schema match for odd_enhancers.py
                "strength": strength,
                "drop_pct": round(drop_pct, 2),
                "age_bars": n - drop_start,
                "test_count": 0,  # placeholder — replaced with real check_zone_freshness() result below
                "is_fresh": True,  # placeholder — replaced with real check_zone_freshness() result below
                "sd_pattern": sd_pattern,  # Book Ch.3: Rally-Base-Drop or Drop-Base-Drop
                "sd_pattern_type": sd_pattern_type,  # Reversal or Continuation
            })
 
        # Deduplicate, then select the top MAX_ZONES.
        # FIX (institutional review, Finding #7): selection used to be pure
        # strength-sort truncation, so an old (100+ bar ago) high-strength
        # zone could push out a recent, price-relevant zone — meaning
        # nearest_demand/nearest_supply were not reliably the nearest
        # *actionable* zone. Recency and proximity to current price are now
        # blended in alongside strength (see SELECTION_*_WEIGHT).
        demand_zones = self._select_top_zones(self._deduplicate(demand_zones), current_price, n)
        supply_zones = self._select_top_zones(self._deduplicate(supply_zones), current_price, n)
 
        # FIX (institutional review, Finding #2/#5): freshness/originality/
        # quality scoring existed on the class but were never invoked from
        # detect(), and check_zone_originality's base_idx lookup silently
        # defaulted to 0 for every zone (now fixed above), so this scoring
        # was previously either dead code or broken by a silent default.
        # Wired here against the FINAL zone lists (post-selection), against
        # each other zone of the same type for originality comparison.
        for zones_of_type in (demand_zones, supply_zones):
            for zone in zones_of_type:
                fresh = self.check_zone_freshness(zone, df)
                zone["is_fresh"] = fresh["is_fresh"]
                zone["test_count"] = fresh["test_count"]
                quality = self.score_zone_quality(zone, df, zones_of_type)
                zone["is_original"] = quality["is_original"]
                zone["quality_score"] = quality["quality_score"]
                zone["quality_grade"] = quality["quality_grade"]
 
        # Find nearest zones to current price
        nearest_demand = self._nearest(current_price, demand_zones, "demand", symbol=symbol)
        nearest_supply = self._nearest(current_price, supply_zones, "supply", symbol=symbol)
 
        # Book Ch.2: Balance/Imbalance cycle detection
        balance_imbalance = self._detect_balance_imbalance(df)
 
        result = {
            "demand_zones": demand_zones,
            "supply_zones": supply_zones,
            "nearest_demand": nearest_demand,
            "nearest_supply": nearest_supply,
            "current_price": round(current_price, 5),
            "balance_imbalance": balance_imbalance,
        }
 
        log.info(
            f"[SupplyDemand] {len(demand_zones)} demand zones, "
            f"{len(supply_zones)} supply zones detected"
        )
        return result
 
    def _deduplicate(self, zones: List[dict]) -> List[dict]:
        """Remove overlapping zones, keep the strongest."""
        if not zones:
            return []
        zones.sort(key=lambda z: z["strength"], reverse=True)
        deduped = []
        for z in zones:
            overlap = False
            for d in deduped:
                if abs(z["zone_mid"] - d["zone_mid"]) < self.ZONE_TOLERANCE:
                    overlap = True
                    break
            if not overlap:
                deduped.append(z)
        return deduped
 
    def _select_top_zones(
        self, zones: List[dict], current_price: float, n_bars: int
    ) -> List[dict]:
        """Select the top MAX_ZONES by a blended strength/recency/proximity score.
 
        FIX (institutional review, Finding #7): the previous implementation
        truncated to MAX_ZONES purely by `strength`, so an old, far-from-price
        but high-strength zone could crowd out a recent, near-price zone —
        silently degrading `nearest_demand`/`nearest_supply` since those are
        computed by scanning only whatever survived truncation. Blending in
        recency (age_bars) and proximity (distance from current_price) keeps
        the retained set actionable, not just historically strong.
        """
        if not zones:
            return []
        if len(zones) <= self.MAX_ZONES:
            return sorted(zones, key=lambda z: z["strength"], reverse=True)
 
        max_age = max(z["age_bars"] for z in zones) or 1
        max_dist = max(abs(z["zone_mid"] - current_price) for z in zones) or 1e-9
 
        def combined_score(z: dict) -> float:
            strength_n = z["strength"] / 100.0
            recency_n = 1.0 - (z["age_bars"] / max_age)  # newer (smaller age_bars) → closer to 1
            proximity_n = 1.0 - (abs(z["zone_mid"] - current_price) / max_dist)  # nearer price → closer to 1
            return (
                self.SELECTION_STRENGTH_WEIGHT * strength_n
                + self.SELECTION_RECENCY_WEIGHT * recency_n
                + self.SELECTION_PROXIMITY_WEIGHT * proximity_n
            )
 
        return sorted(zones, key=combined_score, reverse=True)[: self.MAX_ZONES]
 
    # ─────────────────────────────────────────────
    # Book Ch.3: ERC (Extended-Range Candlestick) Detection
    # ─────────────────────────────────────────────
 
    @staticmethod
    def is_erc(candle: pd.Series) -> bool:
        """
        Book Page 35 — ERC = Extended-Range Candlestick.
 
        Rule: body length > 50% of total candle range, with minimal wick.
        Formula: body / (high - low) > 0.5
 
        Args:
            candle: pd.Series with open, high, low, close
 
        Returns:
            True if candle is an ERC
        """
        try:
            o, h, l, c = float(candle["open"]), float(candle["high"]), float(candle["low"]), float(candle["close"])
            body = abs(c - o)
            total_range = h - l
            if total_range <= 0:
                return False
            return (body / total_range) > 0.50
        except Exception as e:
            log.debug(f"[supply_demand_zones.py] suppressed: {e}")
            return False
 
    def count_ercs(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> int:
        """
        Book Page 35 — count ERCs in a price move.
        Minimum 2 ERCs required to validate a strong move (zone candidate).
 
        Returns:
            Number of ERC candles in the range [start_idx, end_idx)
        """
        count = 0
        for i in range(start_idx, min(end_idx, len(df))):
            if self.is_erc(df.iloc[i]):
                count += 1
        return count
 
    def has_valid_impulse(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """
        Book Page 35 — validate that a move has ≥2 ERCs (genuine imbalance).
 
        Rule: moves dominated by indecision candles (not ERCs) are less reliable.
        """
        erc_count = self.count_ercs(df, start_idx, end_idx)
        return erc_count >= 2
 
    # ─────────────────────────────────────────────
    # Book Ch.4: Zone Drawing Methods (Distal/Proximal)
    # ─────────────────────────────────────────────
 
    @staticmethod
    def draw_zone_medium_risk(base_candles: pd.DataFrame, zone_type: str) -> dict:
        """
        Book Page 45 — Medium-risk zone drawing method.
 
        Supply zone:
          distal  = highest point (wick) of base
          proximal = lowest BODY price (open or close, whichever is lower) in base
 
        Demand zone:
          distal  = lowest point (wick) of base
          proximal = highest BODY price (open or close, whichever is higher) in base
 
        Returns:
            {"distal": float, "proximal": float, "method": "medium_risk"}
        """
        try:
            highs = base_candles["high"].values
            lows = base_candles["low"].values
            opens = base_candles["open"].values
            closes = base_candles["close"].values
 
            # Body lows/highs (min/max of open and close per candle)
            body_lows = np.minimum(opens, closes)
            body_highs = np.maximum(opens, closes)
 
            if zone_type == "supply":
                distal = float(np.max(highs))       # highest wick
                proximal = float(np.min(body_lows))  # lowest body
            else:  # demand
                distal = float(np.min(lows))         # lowest wick
                proximal = float(np.max(body_highs)) # highest body
 
            return {"distal": round(distal, 5), "proximal": round(proximal, 5),
                    "method": "medium_risk"}
        except Exception as e:
            return {"distal": 0, "proximal": 0, "method": "medium_risk", "error": str(e)}
 
    @staticmethod
    def draw_zone_high_risk(base_candles: pd.DataFrame, zone_type: str) -> dict:
        """
        Book Page 45 — High-risk zone drawing method.
 
        Supply zone:
          distal  = highest point (wick) of base
          proximal = lowest point (wick) of base (full range)
 
        Demand zone:
          distal  = lowest point (wick) of base
          proximal = highest point (wick) of base (full range)
 
        Trade-off: wider zone = earlier entry trigger, larger stop-loss.
        """
        try:
            highs = base_candles["high"].values
            lows = base_candles["low"].values
 
            if zone_type == "supply":
                distal = float(np.max(highs))
                proximal = float(np.min(lows))
            else:  # demand
                distal = float(np.min(lows))
                proximal = float(np.max(highs))
 
            return {"distal": round(distal, 5), "proximal": round(proximal, 5),
                    "method": "high_risk"}
        except Exception as e:
            return {"distal": 0, "proximal": 0, "method": "high_risk", "error": str(e)}
 
    @staticmethod
    def draw_zone_low_risk(base_candles: pd.DataFrame, zone_type: str) -> dict:
        """
        Book Page 46 (supply) / Page 48 (demand) — Low-risk zone drawing method.
 
        Supply zone:
          distal  = highest price (wick) of base
          proximal = highest BODY price (open or close, whichever is higher) in base
 
        Demand zone:
          distal  = lowest price (wick) of base
          proximal = lowest BODY price (open or close, whichever is lower) in base
 
        Trade-off: narrower zone = worse entry price, smaller stop-loss risk.
        """
        try:
            highs = base_candles["high"].values
            lows = base_candles["low"].values
            opens = base_candles["open"].values
            closes = base_candles["close"].values
 
            body_lows = np.minimum(opens, closes)
            body_highs = np.maximum(opens, closes)
 
            if zone_type == "supply":
                distal = float(np.max(highs))        # highest wick
                proximal = float(np.max(body_highs))  # highest body
            else:  # demand
                distal = float(np.min(lows))          # lowest wick
                proximal = float(np.min(body_lows))   # lowest body
 
            return {"distal": round(distal, 5), "proximal": round(proximal, 5),
                    "method": "low_risk"}
        except Exception as e:
            return {"distal": 0, "proximal": 0, "method": "low_risk", "error": str(e)}
 
    # ─────────────────────────────────────────────
    # Book Page 50: Core Entry/Stop/TP Formula
    # ─────────────────────────────────────────────
 
    @staticmethod
    def calculate_entry_stop_tp(
        distal: float, proximal: float, zone_type: str,
        next_opposing_zone: Optional[dict] = None, buffer_pips: float = 2.0,
        symbol: str = "",
    ) -> dict:
        """
        Book Page 50 — Core order-placement formula for S/D zone trades.
 
        Entry price = at the proximal line.
        Stop loss   = just beyond the distal line (+ buffer).
        Take profit  = at the next opposing zone.
 
        Args:
            distal: distal line price (outer boundary)
            proximal: proximal line price (inner boundary)
            zone_type: "supply" (short trade) or "demand" (long trade)
            next_opposing_zone: {"zone_low": float, "zone_high": float} or None
            buffer_pips: buffer beyond distal for stop loss (default 2 pips)
            symbol: optional instrument symbol, used to resolve correct pip
                size (e.g. "USDJPY"). Defaults to "" for backward
                compatibility; falls back to a price-magnitude heuristic.
 
        Returns:
            {"entry": float, "stop_loss": float, "take_profit": float, "risk": float, "reward": float, "rr": float}
        """
        # FIX (C-3): was hardcoded 0.0001, which is ~100x wrong on JPY pairs.
        pip = _pip_value(symbol, distal)
        buffer = buffer_pips * pip
 
        if zone_type == "supply":
            # Short trade
            entry = proximal
            stop_loss = distal + buffer  # above distal
            if next_opposing_zone:
                take_profit = float(next_opposing_zone.get("zone_high", next_opposing_zone.get("zone_low", entry)))
            else:
                take_profit = entry - abs(distal - proximal) * 2  # fallback 1:2 RR
        else:  # demand → long trade
            entry = proximal
            stop_loss = distal - buffer  # below distal
            if next_opposing_zone:
                take_profit = float(next_opposing_zone.get("zone_low", next_opposing_zone.get("zone_high", entry)))
            else:
                take_profit = entry + abs(distal - proximal) * 2  # fallback 1:2 RR
 
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0
 
        return {
            "entry": round(entry, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "risk": round(risk, 5),
            "reward": round(reward, 5),
            "rr": rr,
            "method": "proximal_entry_distal_stop",
        }
 
    # ─────────────────────────────────────────────
    # Book Page 48-49: Gap Detection
    # ─────────────────────────────────────────────
 
    @staticmethod
    def detect_gaps(df: pd.DataFrame, min_gap_pips: float = 5.0, symbol: str = "") -> list:
        """
        Book Page 48 — detect price gaps.
        Gap = open(current) ≠ close(previous).
 
        Book Page 49: zones at gaps are anchored at the ORIGIN of the gap.
 
        Args:
            symbol: optional instrument symbol (e.g. "USDJPY"), used to resolve
                the correct pip size. Defaults to "" for backward compatibility
                with existing detect_gaps(df) callers — falls back to a
                price-magnitude heuristic in that case.
 
        Returns list of gap dicts: {"index": int, "time": str, "type": "up"/"down",
                                     "gap_size_pips": float, "origin_price": float}
        """
        gaps = []
        try:
            for i in range(1, len(df)):
                prev_close = float(df.iloc[i-1]["close"])
                curr_open = float(df.iloc[i]["open"])
                # FIX (institutional review, Finding #4): was hardcoded
                # pip = 0.0001, which is ~100x wrong on JPY-quoted pairs
                # (1 pip = 0.01) — reported gap sizes on e.g. USDJPY were
                # off by two orders of magnitude, causing real gaps to be
                # filtered out (below min_gap_pips) or phantom gaps to pass.
                pip = _pip_value(symbol, curr_open)
                gap_pips = abs(curr_open - prev_close) / pip
 
                if gap_pips >= min_gap_pips:
                    gap_type = "up" if curr_open > prev_close else "down"
                    gaps.append({
                        "index": i,
                        "time": str(df.index[i]),
                        "type": gap_type,
                        "gap_size_pips": round(gap_pips, 1),
                        "origin_price": round(prev_close, 5),  # Book P49: anchor at origin
                        "destination_price": round(curr_open, 5),
                    })
        except Exception as e:
            log.debug(f"[supply_demand_zones] suppressed: {e}")
            pass
        return gaps
 
    # ─────────────────────────────────────────────
    # Book Page 57-60: Fresh + Original Zone Quality
    # ─────────────────────────────────────────────
 
    @staticmethod
    def check_zone_freshness(zone: dict, df: pd.DataFrame) -> dict:
        """
        Book Page 57-59 — check if a zone is fresh (untested).
 
        Rule: is_fresh = (number of times price touched proximal line == 0).
        Non-fresh zones have elevated risk of breaking through.
 
        Returns:
            {"is_fresh": bool, "test_count": int, "detail": str}
        """
        try:
            proximal = zone.get("proximal") or zone.get("zone_high") or zone.get("zone_low")
            if proximal is None:
                return {"is_fresh": True, "test_count": 0, "detail": "No proximal line"}
 
            zone_type = "supply" if zone.get("sd_pattern", "").endswith("Drop") else "demand"
            test_count = 0
 
            for i in range(len(df)):
                high = float(df.iloc[i]["high"])
                low = float(df.iloc[i]["low"])
 
                if zone_type == "supply":
                    # Price tested supply if high reached proximal
                    if high >= proximal:
                        test_count += 1
                else:  # demand
                    # Price tested demand if low reached proximal
                    if low <= proximal:
                        test_count += 1
 
            is_fresh = test_count == 0
            detail = (f"Fresh (untested)" if is_fresh
                      else f"Not fresh — tested {test_count} time(s)")
 
            return {"is_fresh": is_fresh, "test_count": test_count, "detail": detail}
        except Exception as e:
            return {"is_fresh": True, "test_count": 0, "detail": f"Error: {e}"}
 
    @staticmethod
    def check_zone_originality(zone: dict, all_zones: list, df: pd.DataFrame) -> dict:
        """
        Book Page 59-60 — check if a zone is original (not a reaction to another zone).
 
        Rule: scan LEFT from candidate zone. If the nearest prior structure is
        another zone's base → NOT original. If it's an impulse/imbalance candle
        (not a base) → IS original.
 
        Returns:
            {"is_original": bool, "detail": str}
        """
        try:
            zone_idx = zone.get("base_idx", 0)
 
            # Find nearest prior zone
            nearest_prior_zone = None
            for z in all_zones:
                z_idx = z.get("base_idx", 0)
                if z_idx < zone_idx and z_idx >= 0:
                    if nearest_prior_zone is None or z_idx > nearest_prior_zone.get("base_idx", -1):
                        nearest_prior_zone = z
 
            if nearest_prior_zone is None:
                # No prior zone found → original (formed out of nowhere)
                return {"is_original": True, "detail": "No prior zone found → original"}
 
            # Check if there's an impulse candle between prior zone and this zone
            prior_end = nearest_prior_zone.get("base_idx", 0) + 2
            if prior_end < zone_idx:
                # Check for ERC/impulse candles in between
                for i in range(prior_end, zone_idx):
                    if i < len(df):
                        if SupplyDemandZones.is_erc(df.iloc[i]):
                            return {"is_original": True,
                                    "detail": f"Impulse candle at idx {i} between zones → original"}
 
            # No impulse found between → likely a reaction zone
            return {"is_original": False,
                    "detail": f"Reaction to prior zone at idx {nearest_prior_zone.get('base_idx')}"}
        except Exception as e:
            return {"is_original": True, "detail": f"Error: {e}"}
 
    def score_zone_quality(self, zone: dict, df: pd.DataFrame, all_zones: list) -> dict:
        """
        Book Ch.5 — comprehensive zone quality scoring.
 
        Factors:
          1. Freshness (untested = higher quality)
          2. Originality (not a reaction = higher quality)
          3. ERC validation (≥2 ERCs in impulse = higher quality)
          4. No disqualifications (no Doji/staircase/long-tail)
 
        Returns:
            {"quality_score": int (0-100), "quality_grade": str (A-F),
             "is_fresh": bool, "is_original": bool, "factors": dict}
        """
        score = 0
        factors = {}
 
        # 1. Freshness (30 points)
        fresh = self.check_zone_freshness(zone, df)
        factors["freshness"] = fresh
        if fresh["is_fresh"]:
            score += 30
 
        # 2. Originality (25 points)
        original = self.check_zone_originality(zone, all_zones, df)
        factors["originality"] = original
        if original["is_original"]:
            score += 25
 
        # 3. Strength (20 points)
        strength = zone.get("strength", 0)
        factors["strength"] = strength
        if strength >= 60:
            score += 20
        elif strength >= 30:
            score += 10
 
        # 4. Pattern type (15 points — reversal > continuation per Book P32)
        pattern_type = zone.get("sd_pattern_type", "Unknown")
        factors["pattern_type"] = pattern_type
        if pattern_type == "Reversal":
            score += 15
        elif pattern_type == "Continuation":
            score += 8
 
        # 5. Fresh + original bonus (10 points)
        if fresh["is_fresh"] and original["is_original"]:
            score += 10  # Book P57: "ideal zone is both fresh AND original"
 
        grade = "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D" if score >= 20 else "F"
 
        return {
            "quality_score": min(100, score),
            "quality_grade": grade,
            "is_fresh": fresh["is_fresh"],
            "is_original": original["is_original"],
            "factors": factors,
        }
 
    # ─────────────────────────────────────────────
    # Book Ch.4: Zone Disqualification Checks
    # ─────────────────────────────────────────────
 
    @staticmethod
    def is_staircase_pattern(base_candles: pd.DataFrame, zone_type: str) -> bool:
        """
        Book Page 42-43 — Staircase pattern detection.
 
        Supply zone staircase: sequential candles each closing LOWER than prior close.
        Demand zone staircase: sequential candles each closing HIGHER than prior close.
 
        Returns True if staircase detected (zone should be disqualified).
        """
        try:
            closes = base_candles["close"].values
            if len(closes) < 3:
                return False
 
            if zone_type == "supply":
                # Check for monotonically decreasing closes
                for i in range(1, len(closes)):
                    if closes[i] >= closes[i-1]:
                        return False
                return True
            else:  # demand
                # Check for monotonically increasing closes
                for i in range(1, len(closes)):
                    if closes[i] <= closes[i-1]:
                        return False
                return True
        except Exception as e:
            log.debug(f"[supply_demand_zones.py] suppressed: {e}")
            return False
 
    @staticmethod
    def is_doji_only_base(base_candles: pd.DataFrame) -> bool:
        """
        Book Page 43 — Doji-only base disqualification.
 
        Rule: a zone formed by a single Doji candle should be skipped.
        Doji = open ≈ close (body ≤ 10% of range, per our Doji definition).
 
        Returns True if base is a single Doji (zone should be disqualified).
        """
        try:
            if len(base_candles) != 1:
                return False
            c = base_candles.iloc[0]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            body = abs(cl - o)
            total_range = h - l
            if total_range <= 0:
                return True  # zero-range = invalid
            return (body / total_range) <= 0.10  # Doji threshold
        except Exception as e:
            log.debug(f"[supply_demand_zones.py] suppressed: {e}")
            return False
 
    def has_long_tailed_candles(self, base_candles: pd.DataFrame, threshold: Optional[float] = None) -> bool:
        """
        Book Page 42 — Long-tailed candle disqualification.
 
        Rule: zones with several long-tailed (long-wick) candles are likely
        reaction/retest activity, not fresh institutional bases.
 
        Returns True if majority of base candles have long wicks (disqualify).
        """
        if threshold is None:
            threshold = self.LONG_TAIL_WICK_RATIO
        try:
            long_tail_count = 0
            for _, c in base_candles.iterrows():
                o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
                body = abs(cl - o)
                total_range = h - l
                if total_range <= 0:
                    continue
                upper_wick = h - max(o, cl)
                lower_wick = min(o, cl) - l
                max_wick = max(upper_wick, lower_wick)
                if max_wick > body * 2:  # wick ≥ 2× body = long-tailed
                    long_tail_count += 1
            return long_tail_count > len(base_candles) * threshold
        except Exception as e:
            log.debug(f"[supply_demand_zones.py] suppressed: {e}")
            return False
 
    def is_zone_tradable(self, base_candles: pd.DataFrame, zone_type: str) -> dict:
        """
        Book Ch.4 — check if a zone is tradable (not disqualified).
 
        Disqualifying conditions:
          1. Doji-only base (Page 43)
          2. Staircase pattern (Page 42-43)
          3. Several long-tailed candles (Page 42)
 
        Returns:
            {"tradable": bool, "disqualifications": [str]}
        """
        disqualifications = []
 
        if self.is_doji_only_base(base_candles):
            disqualifications.append("Doji-only base")
 
        if self.is_staircase_pattern(base_candles, zone_type):
            disqualifications.append("Staircase pattern")
 
        if self.has_long_tailed_candles(base_candles):
            disqualifications.append("Long-tailed candles majority")
 
        return {
            "tradable": len(disqualifications) == 0,
            "disqualifications": disqualifications,
        }
 
    def _detect_balance_imbalance(self, df: pd.DataFrame, lookback: int = 20) -> dict:
        """
        Book Ch.2 — Balance/Imbalance cycle detection.
 
        Balance area: buyers/sellers roughly equal → tight range, small candles.
        Imbalance area: one side dominates → large directional candle breakout.
 
        FIX (institutional review, Finding C-1): this method's body was
        previously orphaned *inside* is_zone_tradable(), sitting after that
        method's `return` statement — i.e. unreachable dead code — and the
        real `_detect_balance_imbalance` method did not exist anywhere in
        this class. Since detect() calls self._detect_balance_imbalance(df)
        unconditionally, every call to detect() raised AttributeError.
        Restored here as its own method; detection logic is unchanged from
        the original intent (nothing about the balance/imbalance algorithm
        itself was altered).
 
        Returns:
            {
                "current_phase": "balance" | "imbalance" | "transition",
                "range_compression": float,  # current_range / avg_range (< 1 = balance)
                "breakout_candle": bool,     # last candle is a breakout from balance
                "breakout_direction": "up" | "down" | "none",
                "detail": str,
            }
        """
        try:
            recent = df.tail(lookback)
            ranges = (recent["high"] - recent["low"]).values
            current_range = float(ranges[-1])
            avg_range = float(np.mean(ranges[:-1])) if len(ranges) > 1 else current_range
 
            if avg_range <= 0:
                return {"current_phase": "unknown", "range_compression": 1.0,
                        "breakout_candle": False, "breakout_direction": "none",
                        "detail": "Insufficient data"}
 
            compression = current_range / avg_range  # < 1 = balance, > 2 = imbalance
 
            # Check for breakout (imbalance): candle range > 2× average
            is_breakout = compression > 2.0
            last_candle = recent.iloc[-1]
            breakout_dir = "none"
            if is_breakout:
                if float(last_candle["close"]) > float(last_candle["open"]):
                    breakout_dir = "up"
                else:
                    breakout_dir = "down"
 
            # Determine phase
            if compression < 0.7:
                phase = "balance"
                detail = f"Range compression: {compression:.2f} (tight range = balance)"
            elif is_breakout:
                phase = "imbalance"
                detail = f"Breakout candle: {compression:.2f}× avg range ({breakout_dir})"
            else:
                phase = "transition"
                detail = f"Range ratio: {compression:.2f} (between balance and imbalance)"
 
            return {
                "current_phase": phase,
                "range_compression": round(compression, 3),
                "breakout_candle": is_breakout,
                "breakout_direction": breakout_dir,
                "detail": detail,
            }
        except Exception as e:
            return {"current_phase": "unknown", "range_compression": 1.0,
                    "breakout_candle": False, "breakout_direction": "none",
                    "detail": f"Error: {e}"}
 
    def _nearest(
        self, price: float, zones: List[dict], zone_type: str, symbol: str = ""
    ) -> Optional[dict]:
        if not zones:
            return None
        nearest = min(zones, key=lambda z: abs(z["zone_mid"] - price))
        distance = abs(price - nearest["zone_mid"])
        # FIX (C-3): was hardcoded 0.0001, which is ~100x wrong on JPY pairs.
        pip_size = _pip_value(symbol, price)
        return {
            "price": nearest["zone_mid"],
            "distance_pips": round(distance / pip_size, 1),
            "strength": nearest["strength"],
            "zone_low": nearest["zone_low"],
            "zone_high": nearest["zone_high"],
            "type": zone_type,
        }
 
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "demand_zones": [],
            "supply_zones": [],
            "nearest_demand": None,
            "nearest_supply": None,
        }
 
 
# ── Singleton (thread-safe) ────────────────────────────────────────
 
import threading as _threading
 
_SDZ: Optional[SupplyDemandZones] = None
_SDZ_LOCK = _threading.Lock()
 
 
def get_supply_demand_zones() -> SupplyDemandZones:
    """Thread-safe singleton accessor for the SupplyDemandZones instance.
 
    Note: SupplyDemandZones itself is stateless (no mutable attrs), so the
    singleton is safe to share across threads. But the singleton
    initialization itself must be guarded.
    """
    global _SDZ
    if _SDZ is None:
        with _SDZ_LOCK:
            # Re-check inside lock — another thread may have created it
            if _SDZ is None:
                _SDZ = SupplyDemandZones()
    return _SDZ
 
 
# ── Smoke test ───────────────────────────────────────────────────────────
# Added per institutional review (Finding C-1 / recommendation §7.8): this
# module previously had no smoke test at all, which is exactly why a call
# to a non-existent self._detect_balance_imbalance() shipped undetected.
# Running this file directly should never raise, and detect() must always
# return the documented schema, including a valid balance_imbalance dict.
if __name__ == "__main__":
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 1.0850
    close = base + np.cumsum(np.random.randn(n) * 0.0006)
    df_eurusd = pd.DataFrame({
        "open":  close + np.random.randn(n) * 0.0002,
        "high":  close + abs(np.random.randn(n)) * 0.0008,
        "low":   close - abs(np.random.randn(n)) * 0.0008,
        "close": close,
    }, index=dates)
 
    sd = SupplyDemandZones()
 
    # 1) detect() must not raise, and must return the documented schema
    #    (this is the exact call that used to crash with AttributeError).
    result = sd.detect(df_eurusd, symbol="EURUSD")
    for key in ("demand_zones", "supply_zones", "nearest_demand",
                "nearest_supply", "current_price", "balance_imbalance"):
        assert key in result, f"detect() result missing key: {key}"
    assert isinstance(result["balance_imbalance"], dict), \
        "balance_imbalance must be a dict (was previously unreachable/undefined)"
    assert result["balance_imbalance"].get("current_phase") in (
        "balance", "imbalance", "transition", "unknown"
    ), "balance_imbalance.current_phase has an unexpected value"
    print("detect() ran without raising — balance_imbalance schema OK")
 
    # 2) Pip-size fix: JPY pairs must use 0.01, not 0.0001.
    jpy_close = 155.0 + np.cumsum(np.random.randn(n) * 0.03)
    df_usdjpy = pd.DataFrame({
        "open":  jpy_close,
        "high":  jpy_close + 0.05,
        "low":   jpy_close - 0.05,
        "close": jpy_close,
    }, index=dates)
    result_jpy = sd.detect(df_usdjpy, symbol="USDJPY")
    assert _pip_value("USDJPY", jpy_close[-1]) == 0.01, "USDJPY pip size must be 0.01"
    assert _pip_value("EURUSD", close[-1]) == 0.0001, "EURUSD pip size must be 0.0001"
    print("Pip-size resolution OK (JPY=0.01, non-JPY=0.0001)")
 
    # 3) Empty-input path still returns the documented fallback schema.
    empty_result = sd.detect(pd.DataFrame({"open": [], "high": [], "low": [], "close": []}))
    assert empty_result == sd._empty_result()
    print("Empty-input fallback OK")
 
    # 4) FIX Finding #4: detect_gaps() must use the correct pip size on JPY pairs.
    gap_df = pd.DataFrame({
        "open":  [155.00, 155.20],
        "high":  [155.10, 155.30],
        "low":   [154.90, 155.10],
        "close": [155.05, 155.25],
    })
    gaps = sd.detect_gaps(gap_df, min_gap_pips=3.0, symbol="USDJPY")
    assert len(gaps) == 1, "expected exactly one detected gap"
    assert gaps[0]["gap_size_pips"] == 15.0, (
        f"USDJPY gap of 0.15 should be 15.0 pips (0.01/pip), got {gaps[0]['gap_size_pips']} "
        f"— indicates the old hardcoded pip=0.0001 bug"
    )
    print("detect_gaps() JPY pip-size fix OK (Finding #4)")
 
    # 5) FIX Findings #2/#3/#5: zones must be ERC-validated + disqualification-
    #    checked before being returned, and must carry distal/proximal/base_idx
    #    (the schema odd_enhancers.py's score_zone() expects) plus real
    #    quality/originality scoring — not the old raw midpoint-only zones.
    def _c(o, h, l, c):
        return {"open": o, "high": h, "low": l, "close": c}
 
    rows, price = [], 1.1000
    for _ in range(20):  # quiet lead-in
        rows.append(_c(price, price + 0.0002, price - 0.0002, price + 0.00005))
        price += 0.00005
    rows.append(_c(price, price + 0.0003, price - 0.0003, price + 0.00005))  # demand base
    rows.append(_c(price, price + 0.0003, price - 0.0003, price - 0.00002))
    for _ in range(4):  # strong bullish ERC rally (body > 50% of range)
        o, c = price, price + 0.0025
        rows.append(_c(o, c + 0.0001, o - 0.0001, c))
        price = c
    for _ in range(15):  # quiet mid-section
        rows.append(_c(price, price + 0.0002, price - 0.0002, price + 0.00003))
        price += 0.00003
    rows.append(_c(price, price + 0.0003, price - 0.0003, price - 0.00002))  # supply base
    rows.append(_c(price, price + 0.0003, price - 0.0003, price + 0.00002))
    for _ in range(4):  # strong bearish ERC drop
        o, c = price, price - 0.0025
        rows.append(_c(o, o + 0.0001, c - 0.0001, c))
        price = c
    for _ in range(15):  # quiet tail
        rows.append(_c(price, price + 0.0002, price - 0.0002, price - 0.00002))
        price -= 0.00002
 
    df_wired = pd.DataFrame(rows)
    result_wired = sd.detect(df_wired, symbol="EURUSD")
    assert len(result_wired["demand_zones"]) >= 1, "expected >=1 ERC-validated demand zone"
    assert len(result_wired["supply_zones"]) >= 1, "expected >=1 ERC-validated supply zone"
    for zone_list in (result_wired["demand_zones"], result_wired["supply_zones"]):
        z = zone_list[0]
        for key in ("distal", "proximal", "base_idx", "base_idx_start", "base_idx_end",
                    "quality_score", "quality_grade", "is_original", "is_fresh"):
            assert key in z, f"zone missing schema field: {key}"
        assert z["base_idx"] == z["base_idx_start"]
    print("Zone schema (distal/proximal/base_idx/quality) + ERC/disqualification filtering OK (Findings #2/#3/#5)")
 
    print("\nSupplyDemandZones smoke test passed.")