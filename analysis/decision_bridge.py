"""
analysis/decision_bridge.py — Bridge: UnifiedSignalEngine → AdaptiveDecisionEngine
===================================================================================

Bridges the existing 5-engine UnifiedSignalEngine (which uses rigid
voting) with the new AdaptiveDecisionEngine (which uses calibrated
weights from backtest results).

This lets the live trading system benefit from:
  • All the existing engine implementations (no rewrite needed)
  • PLUS the adaptive learning from backtest results
  • PLUS the "single strategy can trade alone" mode (no more
    "all strategies mandatory = no trades" problem)

Two usage patterns:

Pattern 1 — Drop-in replacement (recommended):
    from analysis.decision_bridge import make_adaptive_decision
    unified_result = unified_engine.analyze(df, symbol=symbol)
    decision = make_adaptive_decision(unified_result, symbol=symbol)
    # decision.action / decision.confidence / decision.score

Pattern 2 — Explicit:
    from analysis.decision_bridge import UnifiedToAdaptiveBridge
    bridge = UnifiedToAdaptiveBridge(weights_path="calibrated_weights.json")
    signals = bridge.extract_signals(unified_result)
    decision = bridge.engine.decide(signals)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("decision_bridge")


# ════════════════════════════════════════════════════════════════
#  DEFAULT WEIGHTS PATH — portable across OS
# ════════════════════════════════════════════════════════════════

# Look for calibrated_weights.json in a few sensible locations:
#   1. Next to the forex_ai project (download/backtest_results/)
#   2. Inside forex_ai/ itself
#   3. Current working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # forex_ai/
_CANDIDATE_PATHS = [
    _PROJECT_ROOT.parent / "download" / "backtest_results" / "calibrated_weights.json",
    _PROJECT_ROOT / "download" / "backtest_results" / "calibrated_weights.json",
    _PROJECT_ROOT / "calibrated_weights.json",
    Path.cwd() / "calibrated_weights.json",
    Path.cwd() / "download" / "backtest_results" / "calibrated_weights.json",
]

DEFAULT_WEIGHTS_PATH = ""
for _p in _CANDIDATE_PATHS:
    if _p.exists():
        DEFAULT_WEIGHTS_PATH = str(_p)
        break

# If none found, set the default to the first candidate (will fail gracefully)
if not DEFAULT_WEIGHTS_PATH:
    DEFAULT_WEIGHTS_PATH = str(_CANDIDATE_PATHS[0])


# ════════════════════════════════════════════════════════════════
#  BRIDGE
# ════════════════════════════════════════════════════════════════

class UnifiedToAdaptiveBridge:
    """
    Converts UnifiedSignalEngine output → StrategySignal list →
    AdaptiveDecisionEngine decision.

    The bridge:
      1. Extracts individual engine signals from the unified result
      2. Maps them to StrategySignal objects (one per engine)
      3. Passes them to AdaptiveDecisionEngine for the final decision
    """

    # Map unified_engine keys → adaptive strategy names
    #
    # BUG FIX: these previously pointed at keys ("stop_hunt_result",
    # "ict_result", "pa_result") that UnifiedSignalEngine._build_unified_result()
    # never actually produces — its real output keys are "stop_hunt",
    # "ict_amd", "multi_strategy_pa". Because extract_signals() below reads
    # straight from unified_result.get(...) rather than through this map,
    # every one of those lookups was silently returning {} in production —
    # meaning stop_hunt, ict_amd, and pa never contributed a single vote to
    # adaptive_decision, no matter how strong their backtested win rate.
    # Only "candlestick_patterns" (via detected_patterns) was ever reaching
    # the adaptive engine. Corrected to match the real output keys, and
    # extract_signals() below now reads through ENGINE_KEY_MAP instead of
    # hardcoding a second, independently-drifting set of key names.
    ENGINE_KEY_MAP = {
        "stop_hunt":          "stop_hunt",
        "ict_amd":            "ict_amd",
        "multi_strategy_pa":  "multi_pa",
        "liquidity":          "liquidity",        # ADDED
        "cci_state":          "cci_state",         # ADDED
        "detected_patterns":  "candlestick_patterns",  # special handling
        "consensus":          "_consensus",  # we don't use consensus directly
    }

    def __init__(
        self,
        weights_path: Optional[str] = None,
        mode: str = "confluence",
        fallback_to_unified: bool = True,
    ):
        """
        Args:
            weights_path        : path to calibrated_weights.json (from backtest)
            mode                : "single" | "confluence" | "strict"
            fallback_to_unified : if adaptive engine fails, return unified consensus
        """
        from analysis.adaptive_decision_engine import (
            AdaptiveDecisionEngine, StrategySignal,
        )
        self.AdaptiveDecisionEngine = AdaptiveDecisionEngine
        self.StrategySignal = StrategySignal
        self.fallback_to_unified = fallback_to_unified

        self.engine = AdaptiveDecisionEngine(mode=mode)

        # Load calibrated weights if available
        wp = weights_path or DEFAULT_WEIGHTS_PATH
        if Path(wp).exists():
            try:
                self.engine.load_from_file(wp)
                log.info(f"[Bridge] Loaded calibrated weights from {wp}")
            except Exception as e:
                log.warning(f"[Bridge] Failed to load weights from {wp}: {e}")
        else:
            log.info(f"[Bridge] No weights file at {wp} — using defaults")

    # ══════════════════════════════════════════════════════════
    #  SIGNAL EXTRACTION
    # ══════════════════════════════════════════════════════════

    def extract_signals(self, unified_result: Dict[str, Any]) -> List[Any]:
        """
        Extract individual engine signals from a UnifiedSignalEngine result.

        Returns a list of StrategySignal objects.
        """
        signals = []

        # Stop Hunt
        sh = unified_result.get("stop_hunt", {})  # FIX: was "stop_hunt_result" (never existed)
        if sh:
            sig = sh.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                signals.append(self.StrategySignal(
                    strategy="stop_hunt",
                    action=action,
                    confidence=sig.get("confidence", "Medium"),
                    entry_price=sig.get("entry_price"),
                    stop_loss=sig.get("stop_loss"),
                    take_profit=sig.get("take_profit"),
                    r_multiple=sig.get("r_rr", 2.0),
                ))

        # ICT/AMD
        ict = unified_result.get("ict_amd", {})  # FIX: was "ict_result" (never existed)
        if ict:
            sig = ict.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                signals.append(self.StrategySignal(
                    strategy="ict_amd",
                    action=action,
                    confidence=sig.get("confidence", "Medium"),
                    entry_price=sig.get("entry_price"),
                    stop_loss=sig.get("stop_loss"),
                    take_profit=sig.get("take_profit"),
                    # BUG FIX: ict_amd_signal_engine.py's signal dict uses key
                    # "risk_reward" (see analysis/ict_amd_signal_engine.py L1004),
                    # never "r_rr" — this was silently always falling back to the
                    # 6.0 default regardless of the actual computed R:R.
                    r_multiple=sig.get("risk_reward", 6.0),  # ICT default 1:6
                ))

        # Multi-Strategy PA
        pa = unified_result.get("multi_strategy_pa", {})  # FIX: was "pa_result" (never existed)
        if pa:
            sig = pa.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                signals.append(self.StrategySignal(
                    strategy="multi_pa",
                    action=action,
                    confidence=sig.get("confidence", "Medium"),
                    entry_price=sig.get("entry_price"),
                    stop_loss=sig.get("stop_loss"),
                    # BUG FIX: multi_strategy_pa_engine.py's signal dict uses
                    # key "take_profit_suggested", not "take_profit" (see
                    # analysis/multi_strategy_pa_engine.py L1154) — this was
                    # silently always None for every PA-sourced trade level.
                    take_profit=sig.get("take_profit_suggested"),
                    # Also missing: PA provides its own computed R:R under
                    # "risk_reward" (L1155) — wasn't being read at all before.
                    r_multiple=sig.get("risk_reward", 2.0),
                ))

        # Candlestick patterns → unified signal
        patterns = unified_result.get("detected_patterns", [])
        if patterns:
            # Use the most recent + highest-reliability pattern
            best_pat = None
            best_score = -1
            for p in patterns:
                # Co-founder fix: p can be dict OR object — handle both
                if isinstance(p, dict):
                    _rel = p.get("reliability", "Medium")
                    _dir = p.get("direction", "")
                    _name = p.get("pattern_name", "unknown")
                else:
                    _rel = getattr(p, "reliability", "Medium")
                    _dir = getattr(p, "direction", "")
                    _name = getattr(p, "pattern_name", "unknown")
                score = 2 if _rel == "High" else 1
                if score > best_score:
                    best_score = score
                    best_pat = p
            if best_pat:
                # Normalize best_pat to dict-like access
                if isinstance(best_pat, dict):
                    _bp_rel = best_pat.get("reliability", "Medium")
                    _bp_dir = best_pat.get("direction", "")
                    _bp_name = best_pat.get("pattern_name", "unknown")
                else:
                    _bp_rel = getattr(best_pat, "reliability", "Medium")
                    _bp_dir = getattr(best_pat, "direction", "")
                    _bp_name = getattr(best_pat, "pattern_name", "unknown")
                # Direction: bullish patterns → BUY, bearish → SELL
                direction = _bp_dir.lower() if isinstance(_bp_dir, str) else ""
                if "bull" in direction:
                    action = "BUY"
                elif "bear" in direction:
                    action = "SELL"
                else:
                    action = "NO_TRADE"

                if action in ("BUY", "SELL"):
                    signals.append(self.StrategySignal(
                        strategy="candlestick_patterns",
                        action=action,
                        confidence=_bp_rel,
                        r_multiple=2.0,
                        metadata={"pattern": _bp_name},
                    ))

        # S/R zones — NOTE: left inactive on purpose. UnifiedSignalEngine
        # never produces a standalone scored "sr signal" (only raw zone
        # lists under result["zones"]["support_resistance"], with no
        # action/confidence). "sr_zones_result" never existed in the real
        # output, so this always no-op'd; kept here (rather than deleted)
        # as a placeholder in case S/R gets its own live signal engine later.
        sr = unified_result.get("sr_zones_result", {})
        if sr:
            sig = sr.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                signals.append(self.StrategySignal(
                    strategy="sr_zones",
                    action=action,
                    confidence=sig.get("confidence", "Medium"),
                    entry_price=sig.get("entry_price"),
                    stop_loss=sig.get("stop_loss"),
                    take_profit=sig.get("take_profit"),
                ))

        # SD zones — same situation as S/R above: UnifiedSignalEngine has
        # no live supply/demand-zone engine wired in (SupplyDemandZones is
        # only used inside backtest/per_strategy_tester.py). Left inactive.
        sd = unified_result.get("sd_zones_result", {})
        if sd:
            sig = sd.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                signals.append(self.StrategySignal(
                    strategy="sd_zones_scored",
                    action=action,
                    confidence=sig.get("confidence", "Medium"),
                    entry_price=sig.get("entry_price"),
                    stop_loss=sig.get("stop_loss"),
                    take_profit=sig.get("take_profit"),
                ))

        # ── ADDED: Liquidity sweep-reversal (42.6% WR) ──
        # Now live-wired in UnifiedSignalEngine._analyze_liquidity().
        liq = unified_result.get("liquidity", {})
        if liq:
            sig = liq.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                signals.append(self.StrategySignal(
                    strategy="liquidity",
                    action=action,
                    confidence=sig.get("confidence", "Medium"),
                    entry_price=sig.get("entry_price"),
                    stop_loss=sig.get("stop_loss"),
                    take_profit=sig.get("take_profit"),
                    r_multiple=sig.get("r_rr", 2.0),
                ))

        # ── ADDED: CCI state machine (42.3% WR) ──
        # Now live-wired in UnifiedSignalEngine._analyze_cci().
        cci = unified_result.get("cci_state", {})
        if cci:
            sig = cci.get("signal", {})
            action = sig.get("action", "NO_TRADE")
            if action in ("BUY", "SELL"):
                signals.append(self.StrategySignal(
                    strategy="cci_state",
                    action=action,
                    confidence=sig.get("confidence", "Medium"),
                    entry_price=sig.get("entry_price"),
                    stop_loss=sig.get("stop_loss"),
                    take_profit=sig.get("take_profit"),
                    r_multiple=sig.get("r_rr", 1.5),
                ))

        return signals

    # ══════════════════════════════════════════════════════════
    #  DECISION
    # ══════════════════════════════════════════════════════════

    def decide(
        self,
        unified_result: Dict[str, Any],
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make an adaptive decision from a unified result.

        Returns a dict that's compatible with the existing system:
        {
            "action": "BUY"|"SELL"|"NO_TRADE"|"WAIT",
            "confidence": "High"|"Medium"|"Low",
            "score": float,
            "reason": str,
            "agreeing_strategies": [...],
            "disagreeing_strategies": [...],
            "entry_price": float|None,
            "stop_loss": float|None,
            "take_profit": float|None,
            "source": "adaptive",
            "mode": "confluence",
            "legacy_consensus": {...},  # the original unified consensus, for reference
        }
        """
        # Extract signals
        signals = self.extract_signals(unified_result)

        # Make decision
        try:
            decision = self.engine.decide(signals, current_price=current_price)
            result = {
                "action": decision.action,
                "confidence": decision.confidence,
                "score": decision.score,
                "reason": decision.reason,
                "agreeing_strategies": decision.agreeing_strategies,
                "disagreeing_strategies": decision.disagreeing_strategies,
                "entry_price": decision.entry_price,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "source": "adaptive",
                "mode": decision.mode,
                "weighted_by": decision.weighted_by,
                "legacy_consensus": unified_result.get("consensus", {}),
                "n_signals_extracted": len(signals),
            }
        except Exception as e:
            log.error(f"[Bridge] Adaptive decision failed: {e}")
            if self.fallback_to_unified:
                consensus = unified_result.get("consensus", {})
                return {
                    "action": consensus.get("action", "NO_TRADE"),
                    "confidence": consensus.get("confidence", "Low"),
                    "score": 0.0,
                    "reason": f"Adaptive failed ({e}) — using legacy consensus",
                    "source": "legacy_fallback",
                    "legacy_consensus": consensus,
                }
            raise

        return result


# ════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTION
# ════════════════════════════════════════════════════════════════

# Singleton bridge cache, keyed by (mode, resolved_weights_path).
#
# FIX (institutional review, item #5): this used to be a single global
# instance keyed only on `mode`. If two calls used the same mode but
# different weights_path values, the second call's weights_path was
# silently ignored — the fast path returned the already-built bridge
# before weights_path was ever compared. That meant recalibrated weights
# could be dropped on the floor without any error until the process
# restarted. Keying the cache on (mode, weights_path) fixes that while
# still avoiding rebuilding the bridge on every call for the common case
# of repeated calls with the same mode + weights_path.
_bridge_cache: Dict[tuple, "UnifiedToAdaptiveBridge"] = {}
_bridge_lock = threading.Lock()


def make_adaptive_decision(
    unified_result: Dict[str, Any],
    current_price: Optional[float] = None,
    mode: str = "confluence",
    weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function: make an adaptive decision from a unified result.

    Uses a per-(mode, weights_path) singleton bridge for efficiency.
    Thread-safe: double-checked locking pattern prevents race condition
    where two threads both create new bridges simultaneously.

    Args:
        unified_result : output from UnifiedSignalEngine.analyze()
        current_price  : latest close price (optional)
        mode           : "single" | "confluence" | "strict"
        weights_path   : path to calibrated_weights.json. Different values
            (including switching between an explicit path and the default)
            now get their own cached bridge instead of silently reusing
            whichever bridge happened to be built first for this mode.

    Returns:
        Decision dict (see UnifiedToAdaptiveBridge.decide)
    """
    global _bridge_cache
    # Normalize so that repeated calls with weights_path=None consistently
    # hit the same cache entry (rather than treating None as a moving target).
    resolved_path = weights_path or DEFAULT_WEIGHTS_PATH
    cache_key = (mode, resolved_path)

    # Fast path: bridge for this exact (mode, weights_path) already exists
    bridge = _bridge_cache.get(cache_key)
    if bridge is not None:
        return bridge.decide(unified_result, current_price=current_price)

    # Slow path: acquire lock, re-check, then create
    with _bridge_lock:
        bridge = _bridge_cache.get(cache_key)
        if bridge is None:
            bridge = UnifiedToAdaptiveBridge(weights_path=weights_path, mode=mode)
            _bridge_cache[cache_key] = bridge
        return bridge.decide(unified_result, current_price=current_price)


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 64)
    print("  DECISION BRIDGE — Smoke Test")
    print("=" * 64)

    # Simulate a unified result with multiple engines agreeing
    fake_unified = {
        "stop_hunt": {
            "signal": {"action": "BUY", "confidence": "High",
                       "entry_price": 1.0850, "stop_loss": 1.0820,
                       "take_profit": 1.0910, "r_rr": 2.0}
        },
        "ict_amd": {
            "signal": {"action": "BUY", "confidence": "Medium",
                       "entry_price": 1.0850, "stop_loss": 1.0830,
                       "take_profit": 1.0970, "risk_reward": 6.0}
        },
        "multi_strategy_pa": {
            "signal": {"action": "NO_TRADE", "confidence": "Low"}
        },
        "detected_patterns": [],
        "consensus": {
            "action": "BUY", "confidence": "High",
            "buy_score": 5.0, "sell_score": 0.0,
            "reason": "Consensus BUY from 2 engines",
        },
    }

    bridge = UnifiedToAdaptiveBridge(mode="confluence")
    signals = bridge.extract_signals(fake_unified)
    print(f"\nExtracted {len(signals)} signals:")
    for s in signals:
        print(f"  {s.strategy:<20} {s.action:<5} conf={s.confidence}")

    decision = bridge.decide(fake_unified, current_price=1.0850)
    print(f"\n── Adaptive Decision ──")
    print(f"  Action:     {decision['action']}")
    print(f"  Confidence: {decision['confidence']}")
    print(f"  Score:      {decision['score']:.2f}")
    print(f"  Reason:     {decision['reason']}")
    print(f"  Source:     {decision['source']} ({decision['mode']})")
    print(f"  Agreeing:   {decision['agreeing_strategies']}")

    print(f"\n── Legacy Consensus (for comparison) ──")
    print(f"  {decision['legacy_consensus']}")

    # Test the convenience function
    print(f"\n── Convenience function (mode='single') ──")
    d2 = make_adaptive_decision(fake_unified, current_price=1.0850, mode="single")
    print(f"  Action: {d2['action']} ({d2['confidence']}, score={d2['score']:.2f})")

    print("\n" + "=" * 64)