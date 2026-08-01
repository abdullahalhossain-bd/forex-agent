"""
analysis/odd_enhancers.py — Book 5 (Frank Miller S&D) Chapter 6 Scoring System
==============================================================================

Pages 61-75 implement the book's central quantitative framework:
the "Odd Enhancers" scoring system.

  zone_score = sum(enhancer_scores)   →   feed a tiered trade-decision

  ── 4 COMPULSORY enhancers (each 0-3 normalized) ──────────────
    1. Strength of Move      (Page 63)  — fast/ERC-heavy departure → high
    2. Time at the Zone      (Page 66)  — base ≤3 candles → high; ≥6 → 0
    3. Fresh Zone            (Page 68)  — 0 retests → 3; 1 → 1.5; ≥2 → 0
    4. Risk/Reward Ratio     (Page 69)  — R:R ≥3 → 2; 1.5-2 → 1; <1.5 → 0

  ── 2 OPTIONAL enhancers (confidence boosters, not summed) ────
    5. Original Zone         (Page 72)  — independent, not a reaction
    6. Overlapping Zones     (Page 73)  — multi-TF confluence (HTF dominates)

  ── Decision tiers (Page 74-75) ───────────────────────────────
    Total ≥ 10                 → full-conviction (limit order at proximal)
    Total 7-9 (no zero enhancer) → conditional; 2 entry tactics available:
        (a) Market order  : pierce zone, enter at close of piercing candle
        (b) Confirmation order : pierce + reversal momentum candle
                                  closing back past proximal line
    Total < 7                  → SKIP
    Any enhancer = 0           → SKIP (rule violation, hard gate)

SCORING WEIGHTS (resolved per Page 77 worked example):
  Page 77's end-to-end worked example (score = 10) reveals the ACTUAL weights:
    • Enhancer 1 (Strength):    0-2  (max 2)
    • Enhancer 2 (Time at Zone): 0-2  (max 2)
    • Enhancer 3 (Freshness):   0-3  (max 3)
    • Enhancer 4 (Risk/Reward): 0-3  (max 3)
    • TOTAL MAX = 10  (NOT 12)

  This matches the book's tier thresholds exactly:
    • Score 10       → Tier A (full conviction, max score)
    • Score 7-9      → Tier B (conditional, no zero enhancer)
    • Score < 7      → SKIP

  Page 78 confirms fractional/partial scoring is allowed
  (e.g., 1 point for partial strength, 1.5 for marginal R:R).

  Resolution of earlier source inconsistencies:
  • Page 62 "0-3 scale" = generic upper bound; actual per-enhancer maxes vary.
  • Page 68 "fresh=3" is correct (not the "2 points" in worked-example text).
  • Page 63 strength table 0-2 is correct (not rescaled).

Usage:
    from analysis.odd_enhancers import OddEnhancerScorer
    scorer = OddEnhancerScorer()
    result = scorer.score_zone(zone_dict, df, current_price, htf_zone=None)
    # → {
    #     "total_score": float,            # 0-12
    #     "tier": "A"|"B"|"SKIP",
    #     "compulsory": {1: score, 2: score, 3: score, 4: score},
    #     "optional":  {"original": bool, "overlapping": bool},
    #     "entry_method": "limit"|"market"|"confirmation",
    #     "reason": str
    #   }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("odd_enhancers")


# ════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class EnhancerScore:
    """Single enhancer evaluation result."""
    enhancer_id: int                  # 1-6
    name: str
    raw_value: Any                    # e.g. 2 (retests), 1.5 (rr), 4 (candles)
    score: float                      # 0-3 (normalized)
    max_score: float = 3.0
    note: str = ""

    @property
    def passed(self) -> bool:
        """Hard-gate check: enhancer must score > 0."""
        return self.score > 0.0


@dataclass
class ZoneScoreResult:
    """Final scoring result for a zone."""
    total_score: float                              # 0-10 (2+2+3+3 per Page 77)
    max_score: float = 10.0
    compulsory: Dict[int, EnhancerScore] = field(default_factory=dict)
    optional: Dict[str, bool] = field(default_factory=dict)
    tier: str = "SKIP"                              # "A" | "B" | "SKIP"
    entry_method: str = "none"                      # "limit"|"market"|"confirmation"|"none"
    reason: str = ""
    original_zone: bool = False
    overlapping_zone: bool = False
    htf_dominates: bool = False                     # Book P73: HTF zones stronger
    pa_confluence: bool = False                     # Book P88: PA pattern confirms zone
    pa_pattern: str = ""                            # which PA pattern confirmed

    @property
    def tradeable(self) -> bool:
        return self.tier != "SKIP"


# ════════════════════════════════════════════════════════════════
#  SCORER
# ════════════════════════════════════════════════════════════════

class OddEnhancerScorer:
    """
    Book 5 Chapter 6 — Odd Enhancers scoring system.

    4 compulsory + 2 optional enhancers, 0-3 normalized scale per compulsory.
    Tiered trade decision: A (full conviction) / B (conditional) / SKIP.
    """

    # ── Tier thresholds (Book P74-78) ──────────────────────────
    TIER_A_MIN = 10.0    # full conviction (max possible)
    TIER_B_MIN = 7.0     # conditional
    TOTAL_MAX  = 10.0    # 2 + 2 + 3 + 3 (per Page 77 worked example)

    # ── Per-enhancer maxima (Book P77 — actual weights) ────────
    ENH1_MAX = 2.0   # Strength of Move
    ENH2_MAX = 2.0   # Time at the Zone
    ENH3_MAX = 3.0   # Fresh Zone
    ENH4_MAX = 3.0   # Risk/Reward

    # ── Enhancer 2 thresholds (Book P65-66) ────────────────────
    BASE_CANDLES_HIGH = 3       # ≤3 → strong imbalance → top score
    BASE_CANDLES_SKIP = 6       # ≥6 → orders likely used up → skip

    # ── Enhancer 3 thresholds (Book P68) ───────────────────────
    FRESH_RETESTS_MAX = 0       # 0 retests → 3 pts
    ONCE_RETESTED_MAX = 1       # 1 retest  → 1.5 pts
    # ≥2 retests → 0 pts

    # ── Enhancer 4 thresholds (Book P69) ───────────────────────
    RR_RECOMMENDED  = 3.0       # R:R ≥ 1:3 → 3 pts
    RR_MARGINAL_MIN = 1.5       # 1:1.5 - 1:2 → 1.5 pts
    # R:R < 1:1.5 → 0 pts (skip)

    # ── Enhancer 1 (Strength) — ERC-based (Book P35, P63-65) ──
    ERC_BODY_RATIO = 0.50       # body/(high-low) > 0.5 = ERC
    STRONG_ERC_COUNT = 2        # ≥2 ERCs in departure = strong move

    # ── Optional Enhancer 6: HTF dominance (Book P73) ─────────
    HTF_RANK = {"M1": 1, "M5": 2, "M15": 3, "M30": 4, "H1": 5,
                "H4": 6, "D1": 7, "W1": 8, "MN": 9}

    # ── Book Chapter 7 (Pages 80-88): PA-pattern confluence ────
    # The book treats PA patterns at zones as an unofficial 7th enhancer
    # (confluence layer). We accept a list of DetectedPattern objects
    # from high_reliability_patterns.py and check whether any pattern
    # fires near the zone being scored.
    PA_PROXIMITY_ATR_MULT = 0.5   # pattern within 0.5×ATR of zone = confluence
    PA_LOOKBACK_CANDLES = 5       # only consider patterns in last 5 candles

    # ══════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════

    def score_zone(
        self,
        zone: Dict[str, Any],
        df: pd.DataFrame,
        current_price: float,
        htf_zone: Optional[Dict[str, Any]] = None,
        timeframe: str = "H1",
        pa_patterns: Optional[List[Any]] = None,
    ) -> ZoneScoreResult:
        """
        Score a supply/demand zone against the 4 compulsory + 2 optional
        odd enhancers, plus an optional Book-Chapter-7 PA confluence check.

        Args:
            zone       : zone dict from supply_demand_zones.py:
                         keys: zone_low, zone_high, distal, proximal,
                               test_count, sd_pattern, sd_pattern_type,
                               (optional) base_idx_start, base_idx_end
            df         : OHLCV DataFrame covering the zone formation
            current_price : latest close (for R:R computation)
            htf_zone   : optional higher-timeframe zone dict for
                         Enhancer 6 (overlapping zones) check
            timeframe  : str label ("M5","H1","H4","D1",...) for HTF check
            pa_patterns : optional list of DetectedPattern objects from
                          high_reliability_patterns.py — if any fires near
                          the zone (Book Chapter 7 confluence), set
                          result.pa_confluence = True + record pattern name

        Returns:
            ZoneScoreResult
        """
        result = ZoneScoreResult(total_score=0.0)

        # ── 4 compulsory enhancers ───────────────────────────
        e1 = self._score_strength_of_move(zone, df)
        e2 = self._score_time_at_zone(zone, df)
        e3 = self._score_freshness(zone)
        e4 = self._score_risk_reward(zone, current_price)

        result.compulsory = {1: e1, 2: e2, 3: e3, 4: e4}

        # ── 2 optional enhancers ─────────────────────────────
        result.original_zone = self._check_original_zone(zone, df)
        result.overlapping_zone, result.htf_dominates = self._check_overlapping(
            zone, htf_zone, timeframe
        )
        result.optional = {
            "original":   result.original_zone,
            "overlapping": result.overlapping_zone,
        }

        # ── Book Chapter 7: PA-pattern confluence (unofficial 7th enhancer) ──
        if pa_patterns:
            pa_hit = self._check_pa_confluence(zone, df, pa_patterns)
            result.pa_confluence = pa_hit is not None
            result.pa_pattern = pa_hit or ""
        else:
            result.pa_confluence = False
            result.pa_pattern = ""

        # ── Total + tier ──────────────────────────────────────
        result.total_score = round(sum(e.score for e in result.compulsory.values()), 2)

        # Hard gate: any compulsory enhancer = 0 → SKIP (Book P68: ≥2 retests,
        # P66: ≥6 candles, P69: R:R<1.5, P65: weak departure all = "stay out")
        any_zero = any(not e.passed for e in result.compulsory.values())

        if any_zero:
            result.tier = "SKIP"
            result.entry_method = "none"
            zeros = [k for k, v in result.compulsory.items() if not v.passed]
            result.reason = (
                f"SKIP — enhancer(s) {zeros} scored 0 (hard gate). "
                f"Total {result.total_score}/{self.TOTAL_MAX:.0f}."
            )
        elif result.total_score >= self.TIER_A_MIN:
            result.tier = "A"
            result.entry_method = "limit"
            result.reason = (
                f"TIER A — full conviction ({result.total_score}/{self.TOTAL_MAX:.0f}). "
                f"Enter limit order at proximal line."
            )
        elif result.total_score >= self.TIER_B_MIN:
            result.tier = "B"
            # Pick the more conservative entry by default
            result.entry_method = "confirmation"
            result.reason = (
                f"TIER B — conditional ({result.total_score}/{self.TOTAL_MAX:.0f}). "
                f"Use confirmation-order entry (lower risk)."
            )
        else:
            result.tier = "SKIP"
            result.entry_method = "none"
            result.reason = (
                f"SKIP — score {result.total_score}/{self.TOTAL_MAX:.0f} below tier B "
                f"threshold ({self.TIER_B_MIN})."
            )

        log.debug(
            f"[OddEnhancers] zone_score={result.total_score}/{self.TOTAL_MAX:.0f} "
            f"tier={result.tier} method={result.entry_method} "
            f"opt=[orig={result.original_zone} overlap={result.overlapping_zone} "
            f"pa={result.pa_confluence}({result.pa_pattern})]"
        )
        return result

    # ══════════════════════════════════════════════════════════
    #  ENHANCER 1: STRENGTH OF MOVE  (Book P63-65)
    # ══════════════════════════════════════════════════════════

    def _score_strength_of_move(
        self, zone: Dict[str, Any], df: pd.DataFrame
    ) -> EnhancerScore:
        """
        Book Page 63-65 + Page 77 worked example (confirms max = 2):
          Strong/fast departure (≥2 ERCs, large body candles) → high score.
          Weak/slow departure (indecision candles, doji-like) → low score.

        Score (0-2, per Page 77 actual weight):
          2.0 → ≥2 ERCs in departure, large imbalance
          1.0 → 1 ERC or moderate body candles (partial credit, Book P78)
          0.0 → 0 ERCs, dominated by indecision candles
        """
        # Determine departure candle range — use base_idx_end if available,
        # else fall back to scanning the next 3 candles after zone_high/low.
        start_idx = zone.get("base_idx_end")
        if start_idx is None or not isinstance(start_idx, int):
            start_idx = self._infer_departure_start(zone, df)

        # Look at the next 5 candles after the base for departure strength
        end_idx = min(start_idx + 5, len(df))

        if start_idx >= end_idx or start_idx < 0:
            return EnhancerScore(
                enhancer_id=1, name="Strength of Move",
                raw_value="unknown", score=0.0, max_score=self.ENH1_MAX,
                note="Could not locate departure candles",
            )

        erc_count = 0
        body_ratio_sum = 0.0
        n_candles = end_idx - start_idx
        for i in range(start_idx, end_idx):
            try:
                row = df.iloc[i]
                o, h, l, c = (float(row["open"]), float(row["high"]),
                              float(row["low"]),  float(row["close"]))
                body = abs(c - o)
                rng = h - l
                if rng > 0:
                    body_ratio = body / rng
                    body_ratio_sum += body_ratio
                    if body_ratio > self.ERC_BODY_RATIO:
                        erc_count += 1
            except Exception as e:
                log.debug(f"[odd_enhancers] suppressed: {e}")
                continue

        avg_body_ratio = (body_ratio_sum / n_candles) if n_candles > 0 else 0.0

        # Score (0-2 scale per Page 77)
        if erc_count >= self.STRONG_ERC_COUNT:
            score = 2.0
            note = f"Strong: {erc_count} ERCs in departure (avg body {avg_body_ratio:.2f})"
        elif erc_count == 1:
            score = 1.0
            note = f"Moderate: 1 ERC in departure (avg body {avg_body_ratio:.2f})"
        else:
            score = 0.0
            note = f"Weak: 0 ERCs, indecision-dominated (avg body {avg_body_ratio:.2f})"

        return EnhancerScore(
            enhancer_id=1, name="Strength of Move",
            raw_value={"erc_count": erc_count, "avg_body_ratio": round(avg_body_ratio, 3)},
            score=score, max_score=self.ENH1_MAX, note=note,
        )

    # ══════════════════════════════════════════════════════════
    #  ENHANCER 2: TIME AT THE ZONE  (Book P65-67)
    # ══════════════════════════════════════════════════════════

    def _score_time_at_zone(
        self, zone: Dict[str, Any], df: pd.DataFrame
    ) -> EnhancerScore:
        """
        Book Page 65-67 + Page 77 worked example (confirms max = 2):
          ≤3 candles in base  → strong imbalance → top score
          4-5 candles         → gray zone (partial credit)
          ≥6 candles          → too balanced, orders likely used up → SKIP

        Score (0-2, per Page 77 actual weight):
          2.0 → ≤3 candles
          1.0 → 4-5 candles (partial credit)
          0.0 → ≥6 candles (skip)
        """
        # Prefer explicit base_idx_start / base_idx_end if zone dict has them
        start = zone.get("base_idx_start")
        end = zone.get("base_idx_end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            candle_count = end - start
        else:
            # Fall back: infer from zone_low/zone_high — count candles whose
            # range overlaps the zone
            candle_count = self._infer_base_candle_count(zone, df)

        if candle_count <= self.BASE_CANDLES_HIGH:
            score = 2.0
            note = f"Strong: {candle_count} candle(s) in base (≤3)"
        elif candle_count < self.BASE_CANDLES_SKIP:
            score = 1.0
            note = f"Marginal: {candle_count} candles in base (4-5)"
        else:
            score = 0.0
            note = f"Skip: {candle_count} candles in base (≥6, too balanced)"

        return EnhancerScore(
            enhancer_id=2, name="Time at the Zone",
            raw_value=candle_count, score=score, max_score=self.ENH2_MAX, note=note,
        )

    # ══════════════════════════════════════════════════════════
    #  ENHANCER 3: FRESH ZONE  (Book P68)
    # ══════════════════════════════════════════════════════════

    def _score_freshness(self, zone: Dict[str, Any]) -> EnhancerScore:
        """
        Book Page 68 (explicit table, max = 3):
          0 retests (fresh) → 3.0
          1 retest          → 1.5
          ≥2 retests        → 0.0  ("hardly any pending orders remain")

        NOTE: source inconsistency — page 68 also says "fresh zone will receive
        2 points" in worked-example text. Page 77 confirms fresh = 3 in the
        end-to-end worked example (so the explicit rule wins).
        """
        retests = int(zone.get("test_count", 0))

        if retests <= self.FRESH_RETESTS_MAX:
            score = 3.0
            note = "Fresh zone (0 retests)"
        elif retests <= self.ONCE_RETESTED_MAX:
            score = 1.5
            note = f"Once retested ({retests} retest)"
        else:
            score = 0.0
            note = f"Stale zone ({retests} retests, ≥2 → skip)"

        return EnhancerScore(
            enhancer_id=3, name="Fresh Zone",
            raw_value=retests, score=score, max_score=self.ENH3_MAX, note=note,
        )

    # ══════════════════════════════════════════════════════════
    #  ENHANCER 4: RISK/REWARD RATIO  (Book P69)
    # ══════════════════════════════════════════════════════════

    def _score_risk_reward(
        self, zone: Dict[str, Any], current_price: float
    ) -> EnhancerScore:
        """
        Book Page 69 + Page 77 worked example (confirms max = 3):
          R:R ≥ 1:3       → 3.0 pts
          1:1.5 - 1:2     → 1.5 pts
          R:R < 1:1.5     → 0.0 pts (skip)

        Convention (Book P69): entry at proximal line, stop at distal line,
        TP at next opposing zone. We compute R:R from current_price if no
        explicit TP zone is provided.

        R:R is computed as reward (TP - entry) / risk (entry - stop).
        For supply zones: entry = proximal, stop = distal (above), TP = below.
        For demand zones: entry = proximal, stop = distal (below), TP = above.
        """
        proximal = float(zone.get("proximal", zone.get("zone_high", 0)))
        distal = float(zone.get("distal", zone.get("zone_low", 0)))
        is_supply = self._is_supply_zone(zone)

        # Determine entry / stop / TP
        entry = proximal
        if is_supply:
            # Supply: sell at proximal, stop above at distal
            stop = max(distal, proximal)
            # Default TP: 3× risk if no opposing zone provided
            risk = stop - entry
            if risk <= 0:
                return EnhancerScore(
                    enhancer_id=4, name="Risk/Reward",
                    raw_value=0.0, score=0.0, max_score=self.ENH4_MAX,
                    note=f"Invalid risk (entry={entry}, stop={stop})",
                )
            # Use provided TP zone if present
            tp = float(zone.get("opposing_zone_proximal", entry - 3 * risk))
            reward = entry - tp
        else:
            # Demand: buy at proximal, stop below at distal
            stop = min(distal, proximal)
            risk = entry - stop
            if risk <= 0:
                return EnhancerScore(
                    enhancer_id=4, name="Risk/Reward",
                    raw_value=0.0, score=0.0, max_score=self.ENH4_MAX,
                    note=f"Invalid risk (entry={entry}, stop={stop})",
                )
            tp = float(zone.get("opposing_zone_proximal", entry + 3 * risk))
            reward = tp - entry

        rr = reward / risk if risk > 0 else 0.0

        if rr >= self.RR_RECOMMENDED:
            score = 3.0
            note = f"R:R = 1:{rr:.2f} (≥1:3, recommended)"
        elif rr >= self.RR_MARGINAL_MIN:
            score = 1.5
            note = f"R:R = 1:{rr:.2f} (marginal, 1:1.5-1:2)"
        else:
            score = 0.0
            note = f"R:R = 1:{rr:.2f} (<1:1.5, skip)"

        return EnhancerScore(
            enhancer_id=4, name="Risk/Reward",
            raw_value=round(rr, 3), score=score, max_score=self.ENH4_MAX, note=note,
        )

    # ══════════════════════════════════════════════════════════
    #  OPTIONAL ENHANCER 5: ORIGINAL ZONE  (Book P72-73)
    # ══════════════════════════════════════════════════════════

    def _check_original_zone(
        self, zone: Dict[str, Any], df: pd.DataFrame
    ) -> bool:
        """
        Book Page 72-73: an "original" zone is formed independently — i.e.,
        the leftward scan from the base hits large impulse candles (not
        another zone's base). Reaction zones are weaker.

        Heuristic: scan 5 candles BEFORE the base; if ≥3 of them are ERCs
        in the same direction as the pre-base move, the zone is original.
        """
        base_start = zone.get("base_idx_start")
        if not isinstance(base_start, int) or base_start < 5:
            return False  # can't verify — assume not original

        erc_count = 0
        try:
            for i in range(max(0, base_start - 5), base_start):
                row = df.iloc[i]
                o, h, l, c = (float(row["open"]), float(row["high"]),
                              float(row["low"]),  float(row["close"]))
                body = abs(c - o)
                rng = h - l
                if rng > 0 and (body / rng) > self.ERC_BODY_RATIO:
                    erc_count += 1
        except Exception as e:
            log.debug(f"[odd_enhancers.py] suppressed: {e}")
            return False

        return erc_count >= 3

    # ══════════════════════════════════════════════════════════
    #  OPTIONAL ENHANCER 6: OVERLAPPING ZONES  (Book P73-74)
    # ══════════════════════════════════════════════════════════

    def _check_overlapping(
        self,
        zone: Dict[str, Any],
        htf_zone: Optional[Dict[str, Any]],
        timeframe: str,
    ) -> Tuple[bool, bool]:
        """
        Book Page 73-74: "Overlapping" / "nested" zones — zones from two
        different timeframes that align, creating confluence.
        Key rule: zones from a LONGER timeframe are inherently stronger
        than zones from a shorter timeframe.

        Returns (overlapping, htf_dominates):
          overlapping     — True if HTF zone overlaps current zone
          htf_dominates   — True if HTF timeframe > current timeframe
        """
        if not htf_zone:
            return False, False

        # Check overlap
        z_low = float(zone.get("zone_low", zone.get("distal", 0)))
        z_high = float(zone.get("zone_high", zone.get("proximal", 0)))
        h_low = float(htf_zone.get("zone_low", htf_zone.get("distal", 0)))
        h_high = float(htf_zone.get("zone_high", htf_zone.get("proximal", 0)))

        overlap = not (z_high < h_low or z_low > h_high)
        if not overlap:
            return False, False

        # Check HTF dominance
        htf_tf = htf_zone.get("timeframe", "")
        htf_rank = self.HTF_RANK.get(htf_tf, 0)
        cur_rank = self.HTF_RANK.get(timeframe, 0)
        htf_dominates = htf_rank > cur_rank

        return True, htf_dominates

    # ══════════════════════════════════════════════════════════
    #  BOOK CHAPTER 7: PA-PATTERN CONFLUENCE  (Pages 80-88)
    # ══════════════════════════════════════════════════════════

    def _check_pa_confluence(
        self,
        zone: Dict[str, Any],
        df: pd.DataFrame,
        pa_patterns: List[Any],
    ) -> Optional[str]:
        """
        Book Chapter 7 (Pages 80-88): price-action patterns layered on
        supply/demand zones act as an unofficial 7th enhancer (confluence).

        The book covers 5 PA patterns at zones:
          - Pin bar        (P80-81)
          - Inside bar     (P82-83)
          - Head & Shoulders (P83-84)
          - Double top/bottom (P85-86)
          - Engulfing      (P87-88)

        This method accepts a list of DetectedPattern objects (from
        high_reliability_patterns.py) and checks whether any pattern:
          (a) fired within the last PA_LOOKBACK_CANDLES candles, AND
          (b) is located within PA_PROXIMITY_ATR_MULT × ATR of the zone.

        Returns the pattern_name of the first match, or None.
        """
        if not pa_patterns or len(df) < 5:
            return None

        # Compute ATR for proximity check
        try:
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            close = df["close"].astype(float).values
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1]),
                ),
            )
            atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
        except Exception as e:
            atr = 0.0010  # fallback

        proximal = float(zone.get("proximal", zone.get("zone_high", 0)))
        distal = float(zone.get("distal", zone.get("zone_low", 0)))
        zone_min = min(proximal, distal)
        zone_max = max(proximal, distal)
        tolerance = self.PA_PROXIMITY_ATR_MULT * atr

        n = len(df)
        for pat in pa_patterns:
            try:
                # DetectedPattern has: pattern_name, candle_index, type, near_zone
                pat_idx = getattr(pat, "candle_index", -1)
                if pat_idx < 0:
                    continue
                # Recency check
                if pat_idx < n - self.PA_LOOKBACK_CANDLES:
                    continue

                # Proximity check — pattern candle's range overlaps zone ± tolerance
                pat_high = float(df.iloc[pat_idx]["high"])
                pat_low = float(df.iloc[pat_idx]["low"])
                if (pat_low <= zone_max + tolerance and
                        pat_high >= zone_min - tolerance):
                    return getattr(pat, "pattern_name", "unknown")
            except Exception as e:
                log.debug(f"[odd_enhancers] suppressed: {e}")
                continue

        return None

    # ══════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════

    def _infer_departure_start(
        self, zone: Dict[str, Any], df: pd.DataFrame
    ) -> int:
        """Find the first candle whose range starts after the zone."""
        z_high = float(zone.get("zone_high", zone.get("proximal", 0)))
        z_low = float(zone.get("zone_low", zone.get("distal", 0)))
        try:
            for i in range(len(df)):
                row = df.iloc[i]
                if float(row["low"]) > z_high or float(row["high"]) < z_low:
                    # candle is outside the zone
                    if i > 0 and (float(df.iloc[i-1]["low"]) <= z_high and
                                  float(df.iloc[i-1]["high"]) >= z_low):
                        return i  # first candle after exiting the zone
            return max(0, len(df) - 5)
        except Exception as e:
            return max(0, len(df) - 5)

    def _infer_base_candle_count(
        self, zone: Dict[str, Any], df: pd.DataFrame
    ) -> int:
        """Count candles whose range overlaps the zone."""
        z_low = float(zone.get("zone_low", zone.get("distal", 0)))
        z_high = float(zone.get("zone_high", zone.get("proximal", 0)))
        count = 0
        try:
            for i in range(len(df)):
                row = df.iloc[i]
                if (float(row["low"]) <= z_high and
                        float(row["high"]) >= z_low):
                    count += 1
        except Exception as e:
            count = 1
        return max(1, count)

    @staticmethod
    def _is_supply_zone(zone: Dict[str, Any]) -> bool:
        """
        Identify supply vs demand zone.

        Book convention (Chapter 3):
          Supply zones (institutional selling):
            - Rally-Base-Drop (RBD) — reversal
            - Drop-Base-Drop  (DBD) — continuation
          Demand zones (institutional buying):
            - Drop-Base-Rally  (DBR) — reversal
            - Rally-Base-Rally (RBR) — continuation

        Key: the LAST word after "Base-" determines the zone type.
             "Rally" at end = demand (zone is base of a rally up).
             "Drop"  at end = supply (zone is base of a drop down).
        """
        sd_pattern = str(zone.get("sd_pattern", "")).lower()
        zone_type = str(zone.get("type", "")).lower()

        # Explicit type field wins
        if "supply" in zone_type:
            return True
        if "demand" in zone_type:
            return False

        # Pattern-based: check the SUFFIX (what happens AFTER the base)
        if sd_pattern.endswith("drop"):
            return True
        if sd_pattern.endswith("rally"):
            return False

        # Fallback: presence-based (less reliable)
        # Use only the LAST segment to avoid false positives like "Drop-Base-Rally"
        return "supply" in sd_pattern


# ════════════════════════════════════════════════════════════════
#  TIER-B ENTRY STATE MACHINE  (Book P74-75)
# ════════════════════════════════════════════════════════════════

@dataclass
class ConfirmationEntrySignal:
    """Result of a Tier-B confirmation-order state-machine check."""
    triggered: bool
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    method: str = "none"           # "market" | "confirmation" | "none"
    piercing_idx: Optional[int] = None
    momentum_idx: Optional[int] = None
    reason: str = ""


class TierBEntryStateMachine:
    """
    Book Page 74-75 — Tier-B (score 7-9) entry tactics.

    Two-state-machine entry methods:
      (A) MARKET ORDER (P74):
          1. Wait for price to CLOSE INSIDE the zone (piercing candle).
          2. Enter at the close of that piercing candle.

      (B) CONFIRMATION ORDER (P75, lower risk):
          1. Wait for price to pierce into the zone.
          2. Wait for a reversal momentum candle that closes BACK beyond
             the proximal line in the trade direction.
          3. Enter at the close of that momentum candle.

    For supply zones: trade direction = DOWN, momentum candle = bearish.
    For demand zones: trade direction = UP, momentum candle = bullish.
    """

    PIERCE_LOOKBACK = 5       # how many recent candles to scan for piercing
    MOMENTUM_LOOKBACK = 3     # after pierce, how many candles to wait for confirmation
    MOMENTUM_BODY_RATIO = 0.55  # body/(high-low) threshold for "momentum candle"

    def __init__(self, scorer: Optional[OddEnhancerScorer] = None):
        self.scorer = scorer or OddEnhancerScorer()

    def check_market_order_entry(
        self,
        zone: Dict[str, Any],
        df: pd.DataFrame,
    ) -> ConfirmationEntrySignal:
        """
        Method A — Market Order entry (Book P74).

        Trigger: most recent CLOSED candle closes inside the zone.
        Entry price: that candle's close.
        Stop: distal line of the zone.
        """
        is_supply = self._is_supply_zone(zone)
        proximal = float(zone.get("proximal", zone.get("zone_high", 0)))
        distal = float(zone.get("distal", zone.get("zone_low", 0)))

        if len(df) < 2:
            return ConfirmationEntrySignal(
                triggered=False, method="market",
                reason="Insufficient data"
            )

        # Scan the last PIERCE_LOOKBACK closed candles
        for i in range(max(1, len(df) - self.PIERCE_LOOKBACK), len(df)):
            row = df.iloc[i]
            c = float(row["close"])
            # Piercing: close is inside the zone
            if (c <= proximal and c >= distal) or \
               (is_supply and c <= proximal and c >= distal) or \
               (not is_supply and c >= proximal and c <= distal):
                # For supply zone: proximal is upper edge, distal is upper-extreme
                # For demand zone: proximal is lower edge, distal is lower-extreme
                # "Inside zone" = between proximal and distal
                pass
            # Simpler: just check if close is within [min(p,d), max(p,d)]
            lo = min(proximal, distal)
            hi = max(proximal, distal)
            if lo <= c <= hi:
                return ConfirmationEntrySignal(
                    triggered=True,
                    entry_price=c,
                    stop_loss=distal,
                    method="market",
                    piercing_idx=i,
                    reason=f"Market-order entry: candle {i} closed inside zone "
                           f"({c:.5f} in [{lo:.5f}, {hi:.5f}])",
                )

        return ConfirmationEntrySignal(
            triggered=False, method="market",
            reason="No piercing candle in last "
                   f"{self.PIERCE_LOOKBACK} candles"
        )

    def check_confirmation_entry(
        self,
        zone: Dict[str, Any],
        df: pd.DataFrame,
    ) -> ConfirmationEntrySignal:
        """
        Method B — Confirmation Order entry (Book P75).

        Trigger sequence:
          1. Some candle in the last PIERCE_LOOKBACK closed inside the zone.
          2. AFTER that, a momentum candle closes back BEYOND the proximal
             line in the trade direction.
        Entry = close of momentum candle. Stop = distal.
        """
        is_supply = self._is_supply_zone(zone)
        proximal = float(zone.get("proximal", zone.get("zone_high", 0)))
        distal = float(zone.get("distal", zone.get("zone_low", 0)))

        if len(df) < 3:
            return ConfirmationEntrySignal(
                triggered=False, method="confirmation",
                reason="Insufficient data"
            )

        # Find the most recent piercing candle (closed inside zone)
        piercing_idx = None
        lo = min(proximal, distal)
        hi = max(proximal, distal)
        start_scan = max(1, len(df) - self.PIERCE_LOOKBACK - self.MOMENTUM_LOOKBACK)
        for i in range(len(df) - 1, start_scan - 1, -1):
            c = float(df.iloc[i]["close"])
            if lo <= c <= hi:
                piercing_idx = i
                break

        if piercing_idx is None:
            return ConfirmationEntrySignal(
                triggered=False, method="confirmation",
                reason="No piercing candle in recent history"
            )

        # After piercing_idx, look for a momentum candle
        # Supply zone: need bearish candle closing BELOW proximal
        # Demand zone: need bullish candle closing ABOVE proximal
        for i in range(piercing_idx + 1, min(len(df), piercing_idx + 1 + self.MOMENTUM_LOOKBACK)):
            row = df.iloc[i]
            o, h, l, c = (float(row["open"]), float(row["high"]),
                          float(row["low"]),  float(row["close"]))
            body = abs(c - o)
            rng = h - l
            if rng <= 0:
                continue
            body_ratio = body / rng

            if body_ratio < self.MOMENTUM_BODY_RATIO:
                continue  # not a momentum candle

            if is_supply:
                # Bearish momentum candle: close < open AND close < proximal
                if c < o and c < proximal:
                    return ConfirmationEntrySignal(
                        triggered=True,
                        entry_price=c,
                        stop_loss=distal,
                        method="confirmation",
                        piercing_idx=piercing_idx,
                        momentum_idx=i,
                        reason=f"Confirmation entry: bearish momentum candle {i} "
                               f"closed back below proximal ({c:.5f} < {proximal:.5f})",
                    )
            else:
                # Bullish momentum candle: close > open AND close > proximal
                if c > o and c > proximal:
                    return ConfirmationEntrySignal(
                        triggered=True,
                        entry_price=c,
                        stop_loss=distal,
                        method="confirmation",
                        piercing_idx=piercing_idx,
                        momentum_idx=i,
                        reason=f"Confirmation entry: bullish momentum candle {i} "
                               f"closed back above proximal ({c:.5f} > {proximal:.5f})",
                    )

        return ConfirmationEntrySignal(
            triggered=False, method="confirmation",
            piercing_idx=piercing_idx,
            reason="Pierce detected but no momentum reversal candle followed"
        )

    @staticmethod
    def _is_supply_zone(zone: Dict[str, Any]) -> bool:
        """Delegate to the shared implementation in OddEnhancerScorer."""
        return OddEnhancerScorer._is_supply_zone(zone)


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 64)
    print("  ODD ENHANCERS SCORING — Book 5 Chapter 6")
    print("=" * 64)

    # Build a tiny synthetic DataFrame
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 1.0850
    close = base + np.cumsum(np.random.randn(n) * 0.0005)
    high = close + np.abs(np.random.randn(n)) * 0.0008
    low = close - np.abs(np.random.randn(n)) * 0.0008
    opn = close + np.random.randn(n) * 0.0003
    df = pd.DataFrame({"open": opn, "high": high, "low": low, "close": close},
                      index=dates)

    # Build a fake demand zone (Drop-Base-Rally, fresh, 2-candle base, strong rally)
    zone = {
        "zone_low": float(low[5]),
        "zone_high": float(high[6]),
        "distal": float(low[5]),
        "proximal": float(high[6]),
        "sd_pattern": "Drop-Base-Rally",
        "sd_pattern_type": "Reversal",
        "test_count": 0,             # fresh
        "base_idx_start": 5,
        "base_idx_end": 7,           # 2-candle base
    }
    # Inject a strong rally (3 ERCs) after the base
    for i in range(7, 12):
        df.iloc[i, df.columns.get_loc("open")] = float(close[i-1])
        df.iloc[i, df.columns.get_loc("close")] = float(close[i-1]) + 0.0020
        df.iloc[i, df.columns.get_loc("high")] = float(close[i-1]) + 0.0025
        df.iloc[i, df.columns.get_loc("low")] = float(close[i-1]) + 0.0001

    scorer = OddEnhancerScorer()
    result = scorer.score_zone(zone, df, current_price=float(close[-1]))

    print(f"\nZone: {zone['sd_pattern']} (fresh={zone['test_count']==0})")
    print(f"  Total Score:  {result.total_score}/{result.max_score:.0f}")
    print(f"  Tier:         {result.tier}")
    print(f"  Entry Method: {result.entry_method}")
    print(f"  Reason:       {result.reason}")
    print(f"\n  Compulsory Enhancers:")
    for eid, e in result.compulsory.items():
        print(f"    #{eid} {e.name:<22} → {e.score:.1f}/{e.max_score:.0f}  ({e.note})")
    print(f"\n  Optional Enhancers:")
    print(f"    #5 Original Zone:    {result.original_zone}")
    print(f"    #6 Overlapping Zone: {result.overlapping_zone}  (HTF dominates: {result.htf_dominates})")
    print(f"    #7 PA Confluence:    {result.pa_confluence}  ({result.pa_pattern or 'none'})")

    # Test Tier-B confirmation state machine
    print("\n" + "─" * 64)
    print("  TIER-B ENTRY STATE MACHINE TEST")
    print("─" * 64)

    sm = TierBEntryStateMachine()

    # Case 1: market-order entry — make last candle close inside the zone
    df.iloc[len(df)-1, df.columns.get_loc("close")] = (zone["zone_low"] + zone["zone_high"]) / 2
    sig_m = sm.check_market_order_entry(zone, df)
    print(f"\n  Market-order signal: triggered={sig_m.triggered}")
    if sig_m.triggered:
        print(f"    Entry={sig_m.entry_price:.5f}  SL={sig_m.stop_loss:.5f}")
    print(f"    Reason: {sig_m.reason}")

    # Case 2: confirmation-order entry — pierce then bullish momentum candle
    # Make second-to-last candle pierce, last candle bullish-momentum back above proximal
    p_idx = len(df) - 2
    df.iloc[p_idx, df.columns.get_loc("open")] = zone["zone_high"] + 0.0010
    df.iloc[p_idx, df.columns.get_loc("close")] = (zone["zone_low"] + zone["zone_high"]) / 2  # inside
    df.iloc[p_idx, df.columns.get_loc("high")] = zone["zone_high"] + 0.0012
    df.iloc[p_idx, df.columns.get_loc("low")] = (zone["zone_low"] + zone["zone_high"]) / 2 - 0.0002

    m_idx = len(df) - 1
    df.iloc[m_idx, df.columns.get_loc("open")] = (zone["zone_low"] + zone["zone_high"]) / 2  # was inside
    df.iloc[m_idx, df.columns.get_loc("close")] = zone["zone_high"] + 0.0020  # bullish, above proximal
    df.iloc[m_idx, df.columns.get_loc("high")] = zone["zone_high"] + 0.0025
    df.iloc[m_idx, df.columns.get_loc("low")] = (zone["zone_low"] + zone["zone_high"]) / 2

    sig_c = sm.check_confirmation_entry(zone, df)
    print(f"\n  Confirmation-order signal: triggered={sig_c.triggered}")
    if sig_c.triggered:
        print(f"    Entry={sig_c.entry_price:.5f}  SL={sig_c.stop_loss:.5f}")
        print(f"    Pierce idx={sig_c.piercing_idx}, Momentum idx={sig_c.momentum_idx}")
    print(f"    Reason: {sig_c.reason}")

    print("\n" + "=" * 64)
    print("  All checks complete.")
    print("=" * 64)