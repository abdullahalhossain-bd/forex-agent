# analysis/unified_signal_engine.py
# ============================================================
# Unified Signal Engine — Connects All Strategy Engines
# ============================================================
# Orchestrates 5 engines into one coherent system:
#   1. SupportResistance (zone base — shared)
#   2. HighReliabilityPatternDetector (pattern library — shared)
#   3. StopHuntSignalEngine (stop hunt reversal)
#   4. ICTAMDSignalEngine (ICT/SMC AMD+FVG+MSS, 1:6 R:R)
#   5. MultiStrategyPAEngine (8-step PA, session filter, MTF)
#
# Architecture:
#   ┌─────────────────────────────────────────────────────────┐
#   │  SHARED LAYER (computed once, reused by all engines)    │
#   │  • OHLC DataFrame                                       │
#   │  • ATR                                                  │
#   │  • S/R Zones (SupportResistance engine)                 │
#   │  • All Zones list (S/R + S/D + Trendline) for confluence│
#   │  • Detected Patterns (HighReliabilityPatternDetector)   │
#   └─────────────────────────────────────────────────────────┘
#                              ↓
#   ┌─────────────────────────────────────────────────────────┐
#   │  STRATEGY ENGINES (each consumes shared layer)          │
#   │  • StopHunt (uses shared zones)                         │
#   │  • ICT/AMD (uses shared zones, runs own accumulation)   │
#   │  • Multi-Strategy PA (uses shared patterns via checklist│
#   │                       + own zones + own trend)          │
#   └─────────────────────────────────────────────────────────┘
#                              ↓
#   ┌─────────────────────────────────────────────────────────┐
#   │  UNIFIED OUTPUT                                         │
#   │  • Zones (merged + deduplicated)                        │
#   │  • Detected Patterns                                    │
#   │  • Per-engine signals (StopHunt, ICT/AMD, PA)           │
#   │  • Consensus signal (voting across engines)             │
#   │  • Final action: BUY/SELL/WAIT/NO_TRADE                 │
#   └─────────────────────────────────────────────────────────┘
# ============================================================

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from analysis.support_resistance import SupportResistance
from analysis.stop_hunt_signal_engine import StopHuntSignalEngine
from analysis.ict_amd_signal_engine import ICTAMDSignalEngine
from analysis.multi_strategy_pa_engine import MultiStrategyPAEngine
from analysis.high_reliability_patterns import (
    HighReliabilityPatternDetector,
    DetectedPattern,
)

log = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────
MIN_CANDLES_REQUIRED = 30


# ─── Helpers (shared with other engines via _engine_utils) ─────
from analysis._engine_utils import atr_value as _atr


def _zones_to_unified(sr_zones: list, sd_zones: list = None,
                       trendline_zones: list = None) -> list:
    """
    Merge zones from multiple engines into a unified list with consistent schema.
    Each zone: {"type": str, "zone_top": float, "zone_bottom": float, "touches": int, "strength": str, "source": str}
    """
    unified = []

    # S/R zones
    for z in sr_zones or []:
        unified.append({
            "type": z.get("type") or ("resistance" if z.get("role") == "resistance" else "support"),
            "zone_top": float(z.get("zone_top", 0)),
            "zone_bottom": float(z.get("zone_bottom", 0)),
            "touches": int(z.get("touches", 0)),
            "strength": z.get("strength", "Weak"),
            "source": "SR",
        })

    # S/D zones
    for z in sd_zones or []:
        unified.append({
            "type": z.get("type", "supply"),  # supply or demand
            "zone_top": float(z.get("zone_top", 0)),
            "zone_bottom": float(z.get("zone_bottom", 0)),
            "touches": 0,
            "strength": "Medium",  # S/D zones are institutional
            "source": "SD",
        })

    # Trendline zones (if provided)
    for z in trendline_zones or []:
        unified.append({
            "type": "Trendline",
            "zone_top": float(z.get("zone_top", 0)),
            "zone_bottom": float(z.get("zone_bottom", 0)),
            "touches": int(z.get("touches", 0)),
            "strength": z.get("strength", "Medium"),
            "source": "Trendline",
        })

    return unified


# ─── Main Unified Engine ──────────────────────────────────────

class UnifiedSignalEngine:
    """
    Master orchestrator — connects all 5 strategy engines into one system.

    Usage:
        engine = UnifiedSignalEngine(timeframe="4H")
        result = engine.analyze(df, symbol="EURUSD", lower_tf_df=lower_df)
        print(json.dumps(result, indent=2))
    """

    def __init__(
        self,
        timeframe: str = "4H",
        swing_window: Optional[int] = None,
        cluster_threshold_pct: Optional[float] = None,
        min_touches: int = 2,
        # Strategy-specific config
        enable_stop_hunt: bool = True,
        enable_ict_amd: bool = True,
        enable_pa: bool = True,
        enable_patterns: bool = True,
        # R:R thresholds
        ict_min_rr: float = 6.0,
        pa_min_rr: float = 2.0,
        # Pattern lookback
        pattern_lookback: int = 20,
    ):
        self.timeframe = timeframe.upper()
        self.enable_stop_hunt = enable_stop_hunt
        self.enable_ict_amd = enable_ict_amd
        self.enable_pa = enable_pa
        self.enable_patterns = enable_patterns

        # ── Shared S/R engine (base for all strategies) ──
        self.sr_engine = SupportResistance(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
            wick_body_ratio=1.5,
            max_zones_per_side=10,
        )

        # ── Strategy engines ──
        self.stop_hunt_engine = StopHuntSignalEngine(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
        )
        self.ict_engine = ICTAMDSignalEngine(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
            min_rr_ratio=ict_min_rr,
        )
        self.pa_engine = MultiStrategyPAEngine(
            timeframe=timeframe,
            swing_window=swing_window,
            cluster_threshold_pct=cluster_threshold_pct,
            min_touches=min_touches,
        )

        # ── Pattern detector (shared) ──
        self.pattern_detector = HighReliabilityPatternDetector(
            lookback=pattern_lookback,
        )

    # ═══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        lower_tf_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Run all enabled engines and produce unified output.

        Args:
            df: OHLC DataFrame (primary timeframe)
            symbol: e.g., "EURUSD"
            lower_tf_df: lower TF OHLC for MTF confirmation (H2 for 4H, M30 for 1H)

        Returns:
            Unified dict with all engine outputs + consensus signal.
        """
        # ── Edge case: insufficient data ──
        if df is None or len(df) < MIN_CANDLES_REQUIRED:
            return self._insufficient_data_result(symbol)

        sym = symbol.upper()

        # ── SHARED LAYER: compute once, reuse ──
        atr_val = _atr(df, period=14)

        # S/R Zones (shared)
        try:
            sr_result = self.sr_engine.analyze(df, symbol=sym)
            sr_zones_raw = sr_result.get("resistance_zones", []) + sr_result.get("support_zones", [])
        except Exception as e:
            log.error(f"[Unified] S/R analyze failed: {e}")
            sr_result = {}
            sr_zones_raw = []

        # Build unified zone list (for pattern confluence + signal sharing)
        # Tag each S/R zone with type
        sr_zones_tagged = []
        for z in sr_result.get("resistance_zones", []):
            sr_zones_tagged.append({**z, "type": "resistance"})
        for z in sr_result.get("support_zones", []):
            sr_zones_tagged.append({**z, "type": "support"})
        unified_zones = _zones_to_unified(sr_zones_tagged)

        # ── PATTERNS (shared) ──
        detected_patterns: List[DetectedPattern] = []
        pattern_dicts = []
        pattern_repetition = {"zone_strength_boosts": [], "momentum_sequence": None, "consolidation_detected": False}
        if self.enable_patterns:
            try:
                detected_patterns = self.pattern_detector.detect(
                    df, zones=unified_zones, atr_value=atr_val
                )
                pattern_dicts = [p.to_spec_dict() for p in detected_patterns]
                pattern_repetition = self.pattern_detector.analyze_repetition(detected_patterns)
            except Exception as e:
                log.error(f"[Unified] Pattern detection failed: {e}")

        # ── STRATEGY ENGINES ──
        # Each consumes the shared zones + patterns as needed

        # StopHunt engine (uses its own internal S/R, that's fine)
        if self.enable_stop_hunt:
            try:
                stop_hunt_result = self.stop_hunt_engine.analyze(df, symbol=sym)
            except Exception as e:
                log.error(f"[Unified] StopHunt engine failed: {e}")
                stop_hunt_result = self._fallback_stop_hunt(reason="StopHunt engine failed")
        else:
            stop_hunt_result = self._fallback_stop_hunt(reason="StopHunt engine disabled")

        # ICT/AMD engine (uses its own internal S/R + accumulation)
        if self.enable_ict_amd:
            try:
                ict_result = self.ict_engine.analyze(df, symbol=sym)
            except Exception as e:
                log.error(f"[Unified] ICT/AMD engine failed: {e}")
                ict_result = self._fallback_ict(reason="ICT/AMD engine failed")
        else:
            ict_result = self._fallback_ict(reason="ICT/AMD engine disabled")

        # Multi-Strategy PA engine (uses its own internal S/R + trend + checklist)
        # Pass lower_tf_df for MTF confirmation
        if self.enable_pa:
            try:
                pa_result = self.pa_engine.analyze(df, symbol=sym, lower_tf_df=lower_tf_df)
            except Exception as e:
                log.error(f"[Unified] PA engine failed: {e}")
                pa_result = self._fallback_pa(sym, reason="PA engine failed")
        else:
            pa_result = self._fallback_pa(sym, reason="PA engine disabled")

        # ── CONSENSUS SIGNAL ──
        consensus = self._compute_consensus(
            stop_hunt_result, ict_result, pa_result,
            detected_patterns, pattern_repetition
        )

        # ── BUILD UNIFIED OUTPUT ──
        return self._build_unified_result(
            symbol=sym,
            timeframe=self.timeframe,
            atr=atr_val,
            current_price=float(df["close"].iloc[-1]),
            sr_zones=sr_zones_tagged,
            unified_zones=unified_zones,
            detected_patterns=pattern_dicts,
            pattern_repetition=pattern_repetition,
            stop_hunt_result=stop_hunt_result,
            ict_result=ict_result,
            pa_result=pa_result,
            consensus=consensus,
        )

    # ═══════════════════════════════════════════════════════════
    # CONSENSUS SIGNAL (voting across engines)
    # ═══════════════════════════════════════════════════════════

    def _compute_consensus(
        self,
        stop_hunt_result: Optional[dict],
        ict_result: Optional[dict],
        pa_result: Optional[dict],
        detected_patterns: List[DetectedPattern],
        pattern_repetition: dict,
    ) -> dict:
        """
        Voting-based consensus across all strategy engines.

        Rules:
          - Each engine that produces BUY/SELL gets a weighted vote
          - NO_TRADE / WAIT does NOT vote (abstain)
          - If consolidation_detected (multi Doji) → bias toward WAIT
          - If 2+ engines agree on direction → consensus action = that direction
          - If only 1 engine votes → consensus = that engine's action but with lower confidence
          - If 0 engines vote → NO_TRADE
        """
        votes = []  # list of (action, weight, confidence, engine_name)

        if stop_hunt_result:
            sig = stop_hunt_result.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                # Stop hunt signals are high-conviction
                weight = 2.0
                votes.append((action, weight, sig.get("confidence", "Medium"), "StopHunt"))

        if ict_result:
            sig = ict_result.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                # ICT 1:6 R:R is highest conviction
                weight = 3.0
                votes.append((action, weight, sig.get("confidence", "Medium"), "ICT/AMD"))

        if pa_result:
            sig = pa_result.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                # PA engine is mid-conviction (depends on checklist)
                weight = 1.5
                votes.append((action, weight, sig.get("confidence", "Medium"), "PA"))
            elif action == "WAIT":
                # WAIT from PA = abstain but lean toward no-trade
                pass

        # Tally votes by direction
        buy_score = sum(w for a, w, c, e in votes if a == "BUY")
        sell_score = sum(w for a, w, c, e in votes if a == "SELL")
        total_score = buy_score + sell_score

        # Consolidation override
        if pattern_repetition.get("consolidation_detected", False):
            return {
                "action": "WAIT",
                "confidence": "Medium",
                "reason": "Consolidation detected (multiple Doji) — engines abstain, lean WAIT",
                "voting_engines": [],
                "buy_score": 0.0,
                "sell_score": 0.0,
            }

        # Determine consensus
        if total_score == 0:
            return {
                "action": "NO_TRADE",
                "confidence": "Low",
                "reason": "No engine produced BUY/SELL signal — all abstained",
                "voting_engines": [],
                "buy_score": 0.0,
                "sell_score": 0.0,
            }

        # Determine direction
        if buy_score > sell_score:
            consensus_action = "BUY"
            consensus_score = buy_score
            winning_engines = [e for a, w, c, e in votes if a == "BUY"]
        elif sell_score > buy_score:
            consensus_action = "SELL"
            consensus_score = sell_score
            winning_engines = [e for a, w, c, e in votes if a == "SELL"]
        else:
            # Tie — no consensus
            return {
                "action": "NO_TRADE",
                "confidence": "Low",
                "reason": f"Tie vote (BUY={buy_score}, SELL={sell_score}) — no consensus",
                "voting_engines": [{"engine": e, "action": a, "weight": w} for a, w, c, e in votes],
                "buy_score": buy_score,
                "sell_score": sell_score,
            }

        # Confidence based on vote count + score
        vote_count = len(winning_engines)
        if vote_count >= 2 and consensus_score >= 4.0:
            confidence = "High"
        elif vote_count >= 1 and consensus_score >= 2.0:
            confidence = "Medium"
        else:
            confidence = "Low"

        reason = (
            f"Consensus {consensus_action} from {vote_count} engine(s): "
            f"{', '.join(winning_engines)}. "
            f"Score: {consensus_score:.1f} (BUY={buy_score:.1f}, SELL={sell_score:.1f})."
        )

        return {
            "action": consensus_action,
            "confidence": confidence,
            "reason": reason,
            "voting_engines": [
                {"engine": e, "action": a, "weight": w, "confidence": c}
                for a, w, c, e in votes if a == consensus_action
            ],
            "buy_score": buy_score,
            "sell_score": sell_score,
        }

    # ═══════════════════════════════════════════════════════════
    # FALLBACK EMPTY RESULTS (for engine failures)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _fallback_stop_hunt(reason: str = "StopHunt engine failed") -> dict:
        return {
            "resistance_zones": [], "support_zones": [],
            "stop_hunt_detected": False, "stop_hunt_zone": "null",
            "signal": {
                "action": "NO_TRADE", "entry_price": None, "stop_loss": None,
                "take_profit": None, "reason": reason,
                "confidence": "Low",
            },
        }

    @staticmethod
    def _fallback_ict(reason: str = "ICT/AMD engine failed") -> dict:
        return {
            "zones": {"strongest_zone": None, "weakest_zone": None},
            "accumulation": {"valid": False, "range_high": None, "range_low": None},
            "manipulation": {"detected": False, "direction": "null",
                             "sweep_price": None, "zone_strength_used": "null"},
            "fvg": {"found": False, "type": "null", "top": None, "bottom": None, "midpoint": None},
            "mss_confirmed": False,
            "signal": {
                "action": "NO_TRADE", "entry_price": None, "stop_loss": None,
                "take_profit": None, "risk_reward": None,
                "reason": reason, "confidence": "Low",
            },
        }

    @staticmethod
    def _fallback_pa(symbol: str, reason: str = "PA engine failed") -> dict:
        return {
            "pair": symbol, "timeframe": "", "session_time_ok": False,
            "trend": {"structure": "sideways", "bos_detected": False, "choch_detected": False},
            "zones": {"support_resistance": [], "supply_demand": [], "strongest_confluence_zone": None},
            "shooting_star_setup": {"detected": False, "candle1_confirmed": False, "candle2_seller_pressure_confirmed": False},
            "multi_timeframe_confirmation": {"lower_tf_used": "null", "aligned": False},
            "confirmation_checklist": {
                "candlestick_pattern": False, "chart_pattern": False, "candle_behavior": False,
                "confluence_level": False, "trendline_confluence": False, "multi_tf_alignment": False,
                "total_confirmed": 0,
            },
            "signal": {
                "action": "NO_TRADE", "entry_price": None, "stop_loss": None,
                "take_profit_suggested": None, "risk_reward": None,
                "reason": reason, "confidence": "Low",
            },
        }

    @staticmethod
    def _insufficient_data_result(symbol: str) -> dict:
        return {
            "pair": symbol,
            "timeframe": "",
            "current_price": None,
            "atr": None,
            "zones": {"support_resistance": [], "unified_zones": []},
            "detected_patterns": [],
            "pattern_repetition": {"zone_strength_boosts": [], "momentum_sequence": None, "consolidation_detected": False},
            "stop_hunt": UnifiedSignalEngine._fallback_stop_hunt(),
            "ict_amd": UnifiedSignalEngine._fallback_ict(),
            "multi_strategy_pa": UnifiedSignalEngine._fallback_pa(symbol),
            "consensus": {
                "action": "NO_TRADE", "confidence": "Low",
                "reason": "Insufficient data",
                "voting_engines": [], "buy_score": 0.0, "sell_score": 0.0,
            },
        }

    # ═══════════════════════════════════════════════════════════
    # BUILD UNIFIED OUTPUT
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _build_unified_result(
        symbol: str,
        timeframe: str,
        atr: float,
        current_price: float,
        sr_zones: list,
        unified_zones: list,
        detected_patterns: list,
        pattern_repetition: dict,
        stop_hunt_result: Optional[dict],
        ict_result: Optional[dict],
        pa_result: Optional[dict],
        consensus: dict,
    ) -> dict:
        """Build the unified output dict."""
        # Extract S/R zones in spec format for output
        sr_zones_output = [
            {
                "type": z.get("type", "support"),
                "zone_top": round(float(z.get("zone_top", 0)), 5),
                "zone_bottom": round(float(z.get("zone_bottom", 0)), 5),
                "touches": int(z.get("touches", 0)),
                "strength": z.get("strength", "Weak"),
            }
            for z in sr_zones[:10]
        ]

        return {
            "pair": symbol,
            "timeframe": timeframe,
            "current_price": round(current_price, 5),
            "atr": round(atr, 6),
            # ── SHARED LAYER ──
            "zones": {
                "support_resistance": sr_zones_output,
                "unified_zones": [
                    {
                        "type": z["type"],
                        "zone_top": round(z["zone_top"], 5),
                        "zone_bottom": round(z["zone_bottom"], 5),
                        "touches": z["touches"],
                        "strength": z["strength"],
                        "source": z["source"],
                    }
                    for z in unified_zones[:15]
                ],
            },
            "detected_patterns": detected_patterns,
            "pattern_repetition": pattern_repetition,
            # ── PER-ENGINE RESULTS ──
            "stop_hunt": stop_hunt_result,
            "ict_amd": ict_result,
            "multi_strategy_pa": pa_result,
            # ── CONSENSUS ──
            "consensus": consensus,
        }

    # ═══════════════════════════════════════════════════════════
    # LLM-FRIENDLY OUTPUT
    # ═══════════════════════════════════════════════════════════

    def analyze_to_json(
        self, df: pd.DataFrame, symbol: str, lower_tf_df: Optional[pd.DataFrame] = None
    ) -> str:
        return json.dumps(self.analyze(df, symbol, lower_tf_df), ensure_ascii=False, indent=2)

    def to_prompt_text(self, result: dict) -> str:
        """Plain-text rendering for LLM prompts."""
        lines = [
            f"=== UNIFIED SIGNAL ({result['pair']} {result['timeframe']}) ===",
            f"Current Price: {result.get('current_price')}",
            f"ATR: {result.get('atr')}",
            "",
            "-- Zones (S/R) --",
        ]
        for z in result["zones"].get("support_resistance", [])[:5]:
            lines.append(f"  {z['type']}: [{z['zone_bottom']} - {z['zone_top']}] touches={z['touches']} ({z['strength']})")

        lines.append("")
        lines.append("-- Detected Patterns --")
        patterns = result.get("detected_patterns", [])
        if not patterns:
            lines.append("  (none)")
        else:
            for p in patterns[:10]:
                emoji = "🟢" if p["reliability"] == "High" else "⚪"
                lines.append(
                    f"  {emoji} {p['pattern_name']} ({p['type']}) @ {p['candle_index_or_time']} "
                    f"| near_zone={p['near_zone']} ({p['zone_type']}) | {p['reliability']}"
                )

        rep = result.get("pattern_repetition", {})
        if rep.get("consolidation_detected"):
            lines.append("  ⚠ Consolidation detected (multiple Doji) → lean WAIT")
        if rep.get("momentum_sequence"):
            ms = rep["momentum_sequence"]
            lines.append(f"  📈 Momentum sequence: {ms['direction']} x{ms['count']}")

        lines.append("")
        lines.append("-- Engine Signals --")

        # StopHunt
        sh = result.get("stop_hunt", {})
        if sh:
            sh_sig = sh.get("signal", {})
            lines.append(f"  StopHunt: {sh_sig.get('action')} (detected={sh.get('stop_hunt_detected')})")

        # ICT/AMD
        ict = result.get("ict_amd", {})
        if ict:
            ict_sig = ict.get("signal", {})
            lines.append(
                f"  ICT/AMD: {ict_sig.get('action')} "
                f"(acc={ict.get('accumulation', {}).get('valid')}, "
                f"manip={ict.get('manipulation', {}).get('detected')}, "
                f"fvg={ict.get('fvg', {}).get('found')}, "
                f"mss={ict.get('mss_confirmed')})"
            )

        # PA
        pa = result.get("multi_strategy_pa", {})
        if pa:
            pa_sig = pa.get("signal", {})
            lines.append(
                f"  PA: {pa_sig.get('action')} "
                f"(trend={pa.get('trend', {}).get('structure')}, "
                f"checklist={pa.get('confirmation_checklist', {}).get('total_confirmed')}/6, "
                f"session={pa.get('session_time_ok')})"
            )

        lines.append("")
        lines.append("-- Consensus --")
        con = result.get("consensus", {})
        lines.append(f"  Action: {con.get('action')}")
        lines.append(f"  Confidence: {con.get('confidence')}")
        lines.append(f"  Buy Score: {con.get('buy_score')} | Sell Score: {con.get('sell_score')}")
        lines.append(f"  Voting: {con.get('voting_engines')}")
        lines.append(f"  Reason: {con.get('reason')}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# Convenience: one-shot helper
# ============================================================

def detect_unified_signal(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "4H",
    lower_tf_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> str:
    """One-shot helper — returns unified JSON."""
    engine = UnifiedSignalEngine(timeframe=timeframe, **kwargs)
    return engine.analyze_to_json(df, symbol, lower_tf_df)


# ============================================================
# CLI entry
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-06-03 06:00", periods=n, freq="4h")
    base = 1.0850
    close = base + np.cumsum(np.random.randn(n) * 0.0008)
    df = pd.DataFrame({
        "open":  close + np.random.randn(n) * 0.0003,
        "high":  close + abs(np.random.randn(n)) * 0.0012,
        "low":   close - abs(np.random.randn(n)) * 0.0012,
        "close": close,
    }, index=dates)

    engine = UnifiedSignalEngine(timeframe="4H")
    result = engine.analyze(df, symbol="EURUSD")
    print(engine.to_prompt_text(result))
