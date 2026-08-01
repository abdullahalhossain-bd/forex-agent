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
# ADDED (live-wiring of the 40%+ WR audit): these two previously only ran
# inside backtest/per_strategy_tester.py, never in the live UnifiedSignalEngine.
from analysis.liquidity import LiquidityPoolAnalyzer
from analysis.cci_state_machine import CCIStateMachine

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
        enable_liquidity: bool = True,   # ADDED — 42.6% WR in audit
        enable_cci: bool = True,         # ADDED — 42.3% WR in audit
        # R:R thresholds
        ict_min_rr: float = 6.0,
        pa_min_rr: float = 2.0,
        liquidity_min_rr: float = 2.0,   # ADDED
        cci_min_rr: float = 1.5,         # ADDED
        # Pattern lookback
        pattern_lookback: int = 20,
    ):
        self.timeframe = timeframe.upper()
        self.enable_stop_hunt = enable_stop_hunt
        self.enable_ict_amd = enable_ict_amd
        self.enable_pa = enable_pa
        self.enable_patterns = enable_patterns
        self.enable_liquidity = enable_liquidity
        self.enable_cci = enable_cci
        self.liquidity_min_rr = liquidity_min_rr
        self.cci_min_rr = cci_min_rr

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

        # ── ADDED: Liquidity pool sweep-reversal engine (42.6% WR) ──
        self.liquidity_engine = LiquidityPoolAnalyzer()

        # ── ADDED: CCI state machine (42.3% WR) ──
        self.cci_engine = CCIStateMachine()

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

        # ── ADDED: Liquidity pool sweep-reversal (42.6% WR in 20-strategy audit) ──
        # Ported from backtest/per_strategy_tester.py::_test_liquidity — same
        # sweep-detection + ATR-based stop/TP logic, evaluated on the latest bar
        # only (backtest looped bar-by-bar over history; live only needs "now").
        if self.enable_liquidity:
            try:
                liquidity_result = self._analyze_liquidity(df, atr_val)
            except Exception as e:
                log.error(f"[Unified] Liquidity engine failed: {e}")
                liquidity_result = self._fallback_liquidity(reason="Liquidity engine failed")
        else:
            liquidity_result = self._fallback_liquidity(reason="Liquidity engine disabled")

        # ── ADDED: CCI state machine (42.3% WR in 20-strategy audit) ──
        # Ported from backtest/per_strategy_tester.py::_test_cci_state — uses
        # the already-computed sr_zones_tagged (shared layer) instead of
        # recomputing S/R per-bar as the backtest did (that recomputation
        # was for no-look-ahead-bias correctness across history; live only
        # has "now", so the shared zones from this call are already correct).
        if self.enable_cci:
            try:
                cci_result = self._analyze_cci(df, sr_zones_tagged, atr_val)
            except Exception as e:
                log.error(f"[Unified] CCI engine failed: {e}")
                cci_result = self._fallback_cci(reason="CCI engine failed")
        else:
            cci_result = self._fallback_cci(reason="CCI engine disabled")

        # ── CONSENSUS SIGNAL ──
        consensus = self._compute_consensus(
            stop_hunt_result, ict_result, pa_result,
            detected_patterns, pattern_repetition,
            liquidity_result=liquidity_result, cci_result=cci_result,
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
            liquidity_result=liquidity_result,
            cci_result=cci_result,
            consensus=consensus,
        )

    # ═══════════════════════════════════════════════════════════
    # ADDED: LIVE LIQUIDITY + CCI ANALYSIS (single-bar, "now" only)
    # ═══════════════════════════════════════════════════════════

    def _analyze_liquidity(self, df: pd.DataFrame, atr_val: float) -> dict:
        """Live liquidity-sweep-reversal signal for the current bar.

        Mirrors backtest/per_strategy_tester.py::_test_liquidity's direction
        and stop/TP logic exactly, so live behavior matches what was
        backtested (67.2%/42.6%-style WR numbers only mean something if the
        live signal generation matches the tested logic).
        """
        res = self.liquidity_engine.analyze(df)
        if not res.get("valid"):
            return self._fallback_liquidity(reason="No valid liquidity pools")

        sweep = res.get("recent_sweep")
        if not sweep:
            return self._fallback_liquidity(reason="No recent sweep detected")

        kind = sweep.get("kind")
        if kind == "high":
            action = "SELL"
        elif kind == "low":
            action = "BUY"
        else:
            return self._fallback_liquidity(reason=f"Unrecognized sweep kind: {kind}")

        if atr_val <= 0:
            return self._fallback_liquidity(reason="ATR unavailable")

        entry = float(df["close"].iloc[-1])
        pool_price = sweep.get("price", entry)
        if action == "BUY":
            stop = min(pool_price, entry) - 1.0 * atr_val
            tp = entry + abs(entry - stop) * self.liquidity_min_rr
        else:
            stop = max(pool_price, entry) + 1.0 * atr_val
            tp = entry - abs(entry - stop) * self.liquidity_min_rr

        return {
            "valid": True,
            "recent_sweep": sweep,
            "signal": {
                "action": action, "entry_price": round(entry, 5),
                "stop_loss": round(stop, 5), "take_profit": round(tp, 5),
                "r_rr": self.liquidity_min_rr,
                "reason": f"Liquidity sweep of {kind} at {pool_price}",
                "confidence": "Medium",
            },
        }

    def _analyze_cci(self, df: pd.DataFrame, sr_zones_tagged: list, atr_val: float) -> dict:
        """Live CCI-state-machine signal for the current bar.

        Mirrors backtest/per_strategy_tester.py::_test_cci_state's zone
        lookup + sm.evaluate() call, using the already-computed shared
        sr_zones_tagged instead of recomputing S/R (see note above analyze()
        call site for why that's safe in the live-only-latest-bar case).
        """
        cci = self._compute_cci(df, period=20)
        if cci is None or len(cci) == 0 or atr_val <= 0:
            return self._fallback_cci(reason="CCI unavailable")

        close = float(df["close"].iloc[-1])
        pip_approx = atr_val / 20  # rough pip proxy, consistent with the
                                    # 20*pip proximity check used in backtest
        zone_type = None
        for z in sr_zones_tagged:
            if z.get("type") == "support" and abs(close - z.get("zone_top", 0)) < 20 * pip_approx:
                zone_type = "demand"
                break
            if z.get("type") == "resistance" and abs(close - z.get("zone_bottom", 0)) < 20 * pip_approx:
                zone_type = "supply"
                break
        if zone_type is None:
            return self._fallback_cci(reason="Price not near a scored S/R zone")

        cci_val = float(cci[-1])
        sig = self.cci_engine.evaluate(
            cci_value=cci_val, zone_type=zone_type,
            position=None, trend_align=True, at_zone=True,
        )
        if sig.action != "ENTER":
            return self._fallback_cci(reason=f"CCI state machine: {sig.action}")

        action = "BUY" if sig.direction == "long" else "SELL"
        stop = close - 1.5 * atr_val if action == "BUY" else close + 1.5 * atr_val
        tp = close + 1.5 * atr_val * self.cci_min_rr if action == "BUY" \
            else close - 1.5 * atr_val * self.cci_min_rr

        return {
            "valid": True,
            "cci_value": round(cci_val, 2),
            "zone_type": zone_type,
            "signal": {
                "action": action, "entry_price": round(close, 5),
                "stop_loss": round(stop, 5), "take_profit": round(tp, 5),
                "r_rr": self.cci_min_rr,
                "reason": f"CCI={cci_val:.1f} at {zone_type} zone",
                "confidence": "High" if getattr(sig, "confluence_score", 0) == 3 else "Medium",
            },
        }

    @staticmethod
    def _compute_cci(df: pd.DataFrame, period: int = 20):
        """Compute CCI indicator. Ported from per_strategy_tester._compute_cci."""
        try:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            typical = (high + low + close) / 3
            sma = typical.rolling(period).mean()
            mad = typical.rolling(period).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            cci = (typical - sma) / (0.015 * mad)
            return cci.values
        except Exception as e:
            log.warning(f"[Unified] CCI computation failed: {e}")
            return None

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
        liquidity_result: Optional[dict] = None,   # ADDED
        cci_result: Optional[dict] = None,         # ADDED
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
        # Diagnostic trail — one entry per sub-engine regardless of whether
        # it voted, so a NO_TRADE/0.0 consensus can actually be explained
        # instead of being a silent dead-end in the logs (previously you
        # could see buy_score=0.0/sell_score=0.0 but never WHY each of the
        # 3 engines individually abstained).
        _vote_trail = []

        if stop_hunt_result:
            sig = stop_hunt_result.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                # Stop hunt signals are high-conviction
                weight = 2.0
                votes.append((action, weight, sig.get("confidence", "Medium"), "StopHunt"))
                _vote_trail.append(f"StopHunt={action}({sig.get('confidence', '?')}) voted")
            else:
                _vote_trail.append(
                    f"StopHunt={action} abstained: {str(sig.get('reason', 'no reason given'))[:80]}"
                )
        else:
            _vote_trail.append("StopHunt=no_result abstained: engine returned nothing")

        if ict_result:
            sig = ict_result.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                # ICT 1:6 R:R is highest conviction
                weight = 3.0
                votes.append((action, weight, sig.get("confidence", "Medium"), "ICT/AMD"))
                _vote_trail.append(f"ICT/AMD={action}({sig.get('confidence', '?')}) voted")
            else:
                _vote_trail.append(
                    f"ICT/AMD={action} abstained: {str(sig.get('reason', 'no reason given'))[:80]}"
                )
        else:
            _vote_trail.append("ICT/AMD=no_result abstained: engine returned nothing")

        if pa_result:
            sig = pa_result.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                # PA engine is mid-conviction (depends on checklist)
                weight = 1.5
                votes.append((action, weight, sig.get("confidence", "Medium"), "PA"))
                _vote_trail.append(f"PA={action}({sig.get('confidence', '?')}) voted")
            elif action == "WAIT":
                # WAIT from PA = abstain but lean toward no-trade
                _vote_trail.append(
                    f"PA=WAIT abstained: {str(sig.get('reason', 'no reason given'))[:80]}"
                )
            else:
                _vote_trail.append(
                    f"PA={action} abstained: {str(sig.get('reason', 'no reason given'))[:80]}"
                )
        else:
            _vote_trail.append("PA=no_result abstained: engine returned nothing")

        # ── ADDED: Liquidity sweep-reversal vote (42.6% WR) ──
        if liquidity_result:
            sig = liquidity_result.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                # Weaker backtested edge than StopHunt/ICT — moderate weight
                weight = 1.3
                votes.append((action, weight, sig.get("confidence", "Medium"), "Liquidity"))
                _vote_trail.append(f"Liquidity={action}({sig.get('confidence', '?')}) voted")
            else:
                _vote_trail.append(
                    f"Liquidity={action} abstained: {str(sig.get('reason', 'no reason given'))[:80]}"
                )
        else:
            _vote_trail.append("Liquidity=no_result abstained: engine returned nothing")

        # ── ADDED: CCI state machine vote (42.3% WR) ──
        if cci_result:
            sig = cci_result.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                weight = 1.0
                votes.append((action, weight, sig.get("confidence", "Medium"), "CCI"))
                _vote_trail.append(f"CCI={action}({sig.get('confidence', '?')}) voted")
            else:
                _vote_trail.append(
                    f"CCI={action} abstained: {str(sig.get('reason', 'no reason given'))[:80]}"
                )
        else:
            _vote_trail.append("CCI=no_result abstained: engine returned nothing")

        log.info("[Unified] Vote trail: " + " | ".join(_vote_trail))

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

        # === FINAL FLEXIBLE CONSENSUS LOGIC ===
        if buy_score >= 1.8 or (buy_score > 0 and sell_score == 0):
            consensus_action = "BUY"
            consensus_score = buy_score
        elif sell_score >= 1.8 or (sell_score > 0 and buy_score == 0):
            consensus_action = "SELL"
            consensus_score = sell_score
        else:
            return {
                "action": "NO_TRADE",
                "confidence": "Low",
                "reason": f"Insufficient consensus (BUY={buy_score:.1f}, SELL={sell_score:.1f})",
                "voting_engines": [],
                "buy_score": buy_score,
                "sell_score": sell_score,
            }

        vote_count = len([v for v in votes if v[0] == consensus_action])
        confidence = "High" if vote_count >= 2 or consensus_score >= 3.0 else "Medium" if consensus_score >= 1.5 else "Low"

        reason = f"Consensus {consensus_action} | {vote_count} engine(s) agreed | Score={consensus_score:.1f}"

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
    def _fallback_liquidity(reason: str = "Liquidity engine failed") -> dict:
        return {
            "valid": False, "recent_sweep": None,
            "signal": {
                "action": "NO_TRADE", "entry_price": None, "stop_loss": None,
                "take_profit": None, "r_rr": None,
                "reason": reason, "confidence": "Low",
            },
        }

    @staticmethod
    def _fallback_cci(reason: str = "CCI engine failed") -> dict:
        return {
            "valid": False, "cci_value": None, "zone_type": None,
            "signal": {
                "action": "NO_TRADE", "entry_price": None, "stop_loss": None,
                "take_profit": None, "r_rr": None,
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
            "liquidity": UnifiedSignalEngine._fallback_liquidity(reason="Insufficient data"),
            "cci_state": UnifiedSignalEngine._fallback_cci(reason="Insufficient data"),
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
        liquidity_result: Optional[dict] = None,   # ADDED
        cci_result: Optional[dict] = None,         # ADDED
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
            "liquidity": liquidity_result,   # ADDED
            "cci_state": cci_result,         # ADDED
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