# strategy/selector.py  —  Day 82 | Strategy Selector Router
# ============================================================
# আপনার architecture-এর সবচেয়ে দুর্বল লিংক ছিল:
#   Market Regime output dict হচ্ছে ঠিকই, কিন্তু কেউ সেটা consume
#   করে ঠিক কোন strategy module-কে activate করবে সেই router ছিল না।
#
# এই module সেই gap পূরণ করে। এটা:
#   1. MarketRegimeDetector এর result নেয়
#   2. দেখে regime + direction + strength + volatility
#   3. ঠিক করে কোন strategy family (TREND_FOLLOW / RANGE / SMC_PULLBACK
#      / BREAKOUT / SCALP / REVERSAL / WAIT) activate হবে
#   4. কোন analyzer modules সেই strategy-র জন্য প্রাসঙ্গিক তার list দেয়
#   5. Risk multiplier + position sizing hint দেয়
#   6. কোন setup-avoid করতে হবে তার avoid-list দেয়
#
# Flow:
#   MarketRegime.detect(df)
#        ↓
#   StrategySelector.select(regime_result, mtf_bias, structure_bias)
#        ↓
#   { strategy, active_modules, risk_mult, avoid, reason }
#        ↓
#   DecisionAgent বা MasterDecisionEngine consume করে
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("strategy_selector")


# ─────────────────────────────────────────────────────────────
# Strategy family constants — single source of truth
# ─────────────────────────────────────────────────────────────

STRATEGY_TREND_FOLLOW = "TREND_FOLLOW"
STRATEGY_RANGE        = "RANGE"
STRATEGY_SMC_PULLBACK = "SMC_PULLBACK"
STRATEGY_BREAKOUT     = "BREAKOUT"
STRATEGY_SCALP        = "SCALP"
STRATEGY_REVERSAL     = "REVERSAL"
STRATEGY_WAIT         = "WAIT"


# ─────────────────────────────────────────────────────────────
# Module library — কোন strategy-র জন্য কোন analyzer প্রাসঙ্গিক
# এই mapping এক জায়গায় রাখা আছে, যাতে পরে যোগ করা সহজ হয়।
# ─────────────────────────────────────────────────────────────

STRATEGY_MODULES: Dict[str, List[str]] = {
    STRATEGY_TREND_FOLLOW: [
        "analysis.structure",          # HH/HL trend structure
        "analysis.market_regime",      # regime re-confirm
        "analysis.ichimoku",           # cloud-based trend bias
        "analysis.divergence",         # trend continuation filter
        "analysis.support_resistance",
        "strategy.trend_follow",
    ],
    STRATEGY_RANGE: [
        "analysis.support_resistance", # range bounds
        "analysis.volatility",         # squeeze detect
        "analysis.divergence",         # mean-reversion signal
        "strategy.ema_rsi_combo",
    ],
    STRATEGY_SMC_PULLBACK: [
        "analysis.smc_engine",         # OB / FVG / premium-discount
        "analysis.smart_money",        # liquidity sweep
        "analysis.breaker_block",      # breaker confirmation
        "analysis.smc_advanced",       # mitigation / inducement
        "analysis.liquidity_engine",
        "analysis.structure",          # BOS / CHoCH
    ],
    STRATEGY_BREAKOUT: [
        "analysis.volatility",         # squeeze -> breakout trigger
        "analysis.structure",          # BOS confirmation
        "analysis.volume_profile",     # POC breakout
        "strategy.breakout",
    ],
    STRATEGY_SCALP: [
        "analysis.session_analysis",   # London/NY active?
        "analysis.support_resistance",
        "analysis.volatility",
        "strategy.scalping_strategy",
    ],
    STRATEGY_REVERSAL: [
        "analysis.divergence",         # primary reversal trigger
        "analysis.patterns",           # double top/bottom, H&S
        "analysis.advanced_patterns",
        "analysis.smart_money",        # liquidity sweep + CHoCH
        "strategy.reversal",
    ],
    STRATEGY_WAIT: [],
}


# ─────────────────────────────────────────────────────────────
# Strategy avoidance — কোন setup এই strategy-তে নিষেধ
# DecisionAgent-কে বলে দেয় কোন signal ignore করতে হবে।
# ─────────────────────────────────────────────────────────────

STRATEGY_AVOID: Dict[str, List[str]] = {
    STRATEGY_TREND_FOLLOW: ["counter_trend", "range_reversal", "scalp"],
    STRATEGY_RANGE:        ["breakout", "trend_follow", "displacement_continuation"],
    STRATEGY_SMC_PULLBACK: ["range_scalp", "counter_trend"],
    STRATEGY_BREAKOUT:     ["range_reversal", "early_entry"],
    STRATEGY_SCALP:        ["swing_hold", "news_hold"],
    STRATEGY_REVERSAL:     ["trend_continuation", "breakout_pullback"],
    STRATEGY_WAIT:         ["*"],  # avoid everything
}


class StrategySelector:
    """
    Market Regime result থেকে ঠিক কোন strategy activate হবে সেই router।

    Usage:
        selector = StrategySelector()
        choice   = selector.select(regime_result, mtf_bias, structure_bias)
        if choice["strategy"] == STRATEGY_WAIT:
            # stand aside — no trade
            return
        # else: activate choice["active_modules"] এবং choice["strategy"]
    """

    def __init__(self, conservative: bool = False):
        """
        conservative=True হলে stricter rules — ছোট ambiguity তে WAIT বেছে নেয়।
        Live trading-এ True রাখা ভালো, paper/backtest-এ False।
        """
        self.conservative = conservative

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════

    def select(
        self,
        regime:        Dict[str, Any],
        mtf_bias:      Optional[Dict[str, Any]] = None,
        structure:     Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        regime    : MarketRegimeDetector.detect(df) এর output
        mtf_bias  : mtf_analyzer.analyze() এর bias (optional, থাকলে ভালো)
        structure : structure engine এর ai_context (optional)

        Returns:
            {
              "strategy":        "TREND_FOLLOW" | "RANGE" | ... | "WAIT",
              "active_modules":  [...],
              "avoid":           [...],
              "risk_mult":       float,
              "position_mult":   float,   # position_sizer-কে hint
              "reason":          str,
              "confidence":      int,     # 0-100
              "regime_summary":  {...},   # ডিবাগ করার জন্য
            }
        """
        if not regime or not regime.get("regime"):
            return self._wait("No regime provided")

        r_regime     = regime.get("regime", "RANGING")
        r_direction  = regime.get("direction", "NEUTRAL")
        r_strength   = regime.get("strength", "WEAK")
        r_volatility = regime.get("volatility", "NORMAL")
        r_strategy   = regime.get("strategy", {}) or {}

        # Strategy suggestion dict-এ risk_mult আছে কিনা দেখো
        base_risk_mult = float(r_strategy.get("risk_mult", 1.0))

        # ── Step 1: Determine raw strategy family ──────────
        strategy = self._pick_strategy(
            r_regime, r_direction, r_strength, mtf_bias, structure, r_volatility
        )

        # ── Step 2: Conflict check — MTF বা structure এ কি conflict আছে ──
        conflict = self._detect_conflict(strategy, r_direction, mtf_bias, structure)
        if conflict and self.conservative:
            return self._wait(f"Conflict detected: {conflict}")

        # ── Round-5 audit fix: REVERSAL+conflict hard-WAIT ────────────
        # Previously: a non-conservative run would keep REVERSAL even
        # with a Structure-vs-regime conflict, applying only a −20
        # confidence penalty. Combined with WEAK strength (−15) and
        # HIGH volatility (−10), the base 70 − 5 (reversal) − 15 − 10
        # − 20 = 20% — but the user observed 30% conf REVERSAL trades
        # still firing, which is dangerous: reversal trading AGAINST
        # a confirmed structure bias is a recipe for account death.
        #
        # Now: REVERSAL is explicitly disallowed when ANY conflict is
        # detected, regardless of the `conservative` flag. Reversals
        # inherently bet against the prevailing structure; doubling
        # down by also betting against the regime bias is structurally
        # unsound. Downgrade to WAIT with a clear reason.
        if conflict and strategy == STRATEGY_REVERSAL:
            log.warning(
                f"[StrategySelector] REVERSAL blocked by conflict "
                f"({conflict}) — downgrading to WAIT. Reversal trading "
                f"against a confirmed conflict is structurally unsound."
            )
            return self._wait(
                f"REVERSAL conflict: {conflict} — reversal against "
                f"conflicting structure/regime bias is disallowed"
            )

        # ── Step 3: Volatility-based position sizing ───────
        position_mult = self._position_multiplier(r_volatility, r_strength)
        final_risk    = round(base_risk_mult * position_mult, 3)

        # ── Step 4: WAIT strategy হলে risk zero ────────────
        if strategy == STRATEGY_WAIT:
            final_risk = 0.0
            position_mult = 0.0

        # ── Step 5: Build result ──────────────────────────
        active = STRATEGY_MODULES.get(strategy, [])
        avoid  = STRATEGY_AVOID.get(strategy, [])

        confidence = self._confidence_level(
            strategy, r_strength, r_volatility, conflict
        )

        result = {
            "strategy":       strategy,
            "active_modules": active,
            "avoid":          avoid,
            "risk_mult":      final_risk,
            "position_mult":  round(position_mult, 3),
            "reason":         self._build_reason(
                strategy, r_regime, r_direction, r_strength, r_volatility, conflict
            ),
            "confidence":     confidence,
            "regime_summary": {
                "regime":        r_regime,
                "direction":     r_direction,
                "strength":      r_strength,
                "volatility":    r_volatility,
                "conflict":      conflict,
            },
        }

        log.info(
            f"[StrategySelector] → {strategy} | "
            f"risk={final_risk}x | conf={confidence}% | "
            f"modules={len(active)} | avoid={len(avoid)}"
            + (f" | conflict={conflict}" if conflict else "")
        )
        return result

    # ═══════════════════════════════════════════════════════
    # STRATEGY PICKER
    # ═══════════════════════════════════════════════════════

    def _pick_strategy(
        self,
        regime:       str,
        direction:    str,
        strength:     str,
        mtf_bias:     Optional[Dict[str, Any]],
        structure:    Optional[Dict[str, Any]],
        volatility:   str = "NORMAL",
    ) -> str:
        """
        মূল routing logic।
        Priority:
          1. BREAKOUT regime হলে strategy ও BREAKOUT
          2. RANGING regime হলে RANGE
          3. TRENDING + structure BULLISH/BEARISH আছে → SMC_PULLBACK
             (pullback to OB/FVG)
          4. TRENDING + strong → TREND_FOLLOW
          5. CHoCH detected → REVERSAL
          6. CHOPPY (ADX<15) + some volatility → SCALP (small, tactical)
          7. Fallback → WAIT

        ── Round-32 fix (log-driven, 2026-07-17): CHOPPY previously fell
        straight through to the "Unknown regime → WAIT" fallback, i.e.
        it was a hard-coded no-trade regime with STRATEGY_SCALP fully
        built (module list + avoid list already defined above) but never
        reachable. That's the right call when the market is ALSO dead
        quiet (LOW volatility — nothing to scalp), but a genuinely choppy
        market can still have small tradeable swings; the book's "not
        worth trading" note (market_regime.py) is about swing/trend
        strategies, not necessarily about tight, small-size scalping.
        So: CHOPPY + LOW volatility → still WAIT (truly dead market).
        CHOPPY + NORMAL/HIGH volatility → SCALP, with size already
        naturally capped by _position_multiplier's WEAK-strength 0.6x
        (CHOPPY almost always reports WEAK strength).
        """
        # CHoCH detected → reversal priority
        if structure:
            choch = structure.get("structure_choch", "NONE")
            if choch not in ("NONE", "") and choch is not None:
                return STRATEGY_REVERSAL

        if regime == "BREAKOUT":
            return STRATEGY_BREAKOUT

        if regime == "RANGING":
            # শুধু ranging না, MTF higher timeframe trending হলে
            # ছোট TF-তে range breakout trade করতে পারি, কিন্তু
            # conservative mode এ RANGE strategy ই বেস্ট।
            return STRATEGY_RANGE

        # TRENDING
        if regime == "TRENDING":
            # SMC context আছে কিনা দেখি — থাকলে pullback strategy
            # trend এ OB/FVG-তে pullback entry
            if structure and structure.get("structure_valid"):
                bias = structure.get("structure_bias", "NEUTRAL")
                if bias in ("BULLISH", "BEARISH") and direction != "NEUTRAL":
                    return STRATEGY_SMC_PULLBACK

            # Strong trend → simple trend follow
            if strength == "STRONG":
                return STRATEGY_TREND_FOLLOW

            # Moderate / weak trend → এখনো trend follow, কিন্তু
            # conservative mode এ WAIT
            if strength == "MODERATE":
                return STRATEGY_TREND_FOLLOW

            # Weak trend + conservative → wait
            if self.conservative:
                return STRATEGY_WAIT
            return STRATEGY_TREND_FOLLOW

        if regime == "CHOPPY":
            if volatility == "LOW":
                return STRATEGY_WAIT
            return STRATEGY_SCALP

        # Unknown regime
        return STRATEGY_WAIT

    # ═══════════════════════════════════════════════════════
    # CONFLICT DETECTOR
    # ═══════════════════════════════════════════════════════

    def _detect_conflict(
        self,
        strategy:    str,
        direction:   str,
        mtf_bias:    Optional[Dict[str, Any]],
        structure:   Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Multi-source conflict detect করে। থাকলে reason string ফেরৎ।
        না থাকলে None।

        Conflict cases:
          - MTF bias BEARISH কিন্তু strategy bullish (TREND_FOLLOW/SMC_PULLBACK)
          - Structure bias বিপরীত দিকে
          - Strategy RANGE কিন্তু strong BOS detected
        """
        if strategy == STRATEGY_WAIT:
            return None

        # MTF conflict
        if mtf_bias:
            mtf_dir = mtf_bias.get("bias") or mtf_bias.get("direction", "NEUTRAL")
            if mtf_dir in ("BULLISH", "BEARISH"):
                # strategy bullish কিন্তু MTF bearish → conflict
                bullish_strat = strategy in (
                    STRATEGY_TREND_FOLLOW, STRATEGY_SMC_PULLBACK, STRATEGY_BREAKOUT
                ) and direction == "BULLISH"
                bearish_strat = strategy in (
                    STRATEGY_TREND_FOLLOW, STRATEGY_SMC_PULLBACK, STRATEGY_BREAKOUT
                ) and direction == "BEARISH"

                if bullish_strat and mtf_dir == "BEARISH":
                    return "MTF bearish vs regime bullish"
                if bearish_strat and mtf_dir == "BULLISH":
                    return "MTF bullish vs regime bearish"

        # Structure conflict
        if structure and structure.get("structure_valid"):
            s_bias = structure.get("structure_bias", "NEUTRAL")
            if s_bias in ("BULLISH", "BEARISH") and direction != "NEUTRAL":
                if (s_bias == "BULLISH" and direction == "BEARISH") or \
                   (s_bias == "BEARISH" and direction == "BULLISH"):
                    return f"Structure {s_bias} vs regime {direction}"

        return None

    # ═══════════════════════════════════════════════════════
    # POSITION SIZING HINT
    # ═══════════════════════════════════════════════════════

    def _position_multiplier(self, volatility: str, strength: str) -> float:
        """
        Volatility + strength দেখে position size-র multiplier।

        High volatility + weak trend → 0.5x (half size)
        High volatility + strong trend → 0.7x
        Normal + strong → 1.0x
        Normal + moderate → 0.8x
        Low volatility + strong → 1.2x (tighter SL পাওয়া যায়)
        Low volatility + weak → 0.6x (no momentum)
        """
        vol_mult = {
            "HIGH":   0.6,
            "NORMAL": 1.0,
            "LOW":    0.9,
        }.get(volatility, 1.0)

        str_mult = {
            "STRONG":   1.1,
            "MODERATE": 0.9,
            "WEAK":     0.6,
        }.get(strength, 0.8)

        return round(vol_mult * str_mult, 3)

    # ═══════════════════════════════════════════════════════
    # CONFIDENCE LEVEL
    # ═══════════════════════════════════════════════════════

    def _confidence_level(
        self,
        strategy:    str,
        strength:    str,
        volatility:  str,
        conflict:    Optional[str],
    ) -> int:
        """
        0-100 scale। Conflict থাকলে -20, high vol -10, weak trend -10।
        Strong trend + normal vol → 75-85।
        """
        if strategy == STRATEGY_WAIT:
            return 0

        base = 70
        if strength == "STRONG":    base += 10
        elif strength == "WEAK":    base -= 15

        if volatility == "HIGH":    base -= 10
        elif volatility == "LOW":   base -= 5

        if conflict:
            # Round-5 audit fix: bumped from −20 to −30. A confirmed
            # Structure-vs-regime conflict is a serious warning sign —
            # the old −20 let too many low-confidence trades slip
            # through (the operator observed 30% conf REVERSAL firing
            # with conflict, which is now blocked upstream, but other
            # strategies like SMC_PULLBACK / BREAKOUT could still
            # reach 35-40% conf with conflict — too risky). −30 forces
            # them down to 15-25% where downstream RiskEngine will
            # reject (typical min_confidence=60).
            base -= 30

        # SMC pullback সবচেয়ে high-probability setup
        if strategy == STRATEGY_SMC_PULLBACK:    base += 5
        if strategy == STRATEGY_REVERSAL:        base -= 5  # reversal risky
        if strategy == STRATEGY_BREAKOUT:        base -= 5  # false breakout risk
        if strategy == STRATEGY_SCALP:           base -= 10  # CHOPPY-routed, extra caution

        return max(0, min(100, int(base)))

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _build_reason(
        self,
        strategy:   str,
        regime:     str,
        direction:  str,
        strength:   str,
        volatility: str,
        conflict:   Optional[str],
    ) -> str:
        parts = [
            f"regime={regime}",
            f"dir={direction}",
            f"str={strength}",
            f"vol={volatility}",
        ]
        if conflict:
            parts.append(f"conflict={conflict}")
        parts.append(f"→ {strategy}")
        return " | ".join(parts)

    def _wait(self, reason: str) -> Dict[str, Any]:
        return {
            "strategy":       STRATEGY_WAIT,
            "active_modules": [],
            "avoid":          ["*"],
            "risk_mult":      0.0,
            "position_mult":  0.0,
            "reason":         reason,
            "confidence":     0,
            "regime_summary": {"conflict": reason},
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, choice: Dict[str, Any]) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  🎯  STRATEGY SELECTOR  (Day 82)")
        log.info(bar)

        s = choice.get("strategy", "WAIT")
        icon = {
            STRATEGY_TREND_FOLLOW: "📈",
            STRATEGY_RANGE:        "🔄",
            STRATEGY_SMC_PULLBACK: "🎯",
            STRATEGY_BREAKOUT:     "💥",
            STRATEGY_SCALP:        "⚡",
            STRATEGY_REVERSAL:     "🔁",
            STRATEGY_WAIT:         "⏸️",
        }.get(s, "❓")

        log.info(f"  Strategy     : {icon}  {s}")
        log.info(f"  Confidence   : {choice.get('confidence', 0)}%")
        log.info(f"  Risk Mult    : {choice.get('risk_mult', 0)}x")
        log.info(f"  Position Mult: {choice.get('position_mult', 0)}x")

        active = choice.get("active_modules", [])
        if active:
            log.info(f"  Active Modules ({len(active)}):")
            for m in active:
                log.info(f"    • {m}")
        else:
            log.info("  Active Modules: (none — standing aside)")

        avoid = choice.get("avoid", [])
        if avoid and avoid != ["*"]:
            log.info(f"  Avoid        : {', '.join(avoid)}")
        elif avoid == ["*"]:
            log.info("  Avoid        : ALL (no trade)")

        log.info(f"  Reason       : {choice.get('reason', '')}")
        log.info(bar)


# ═══════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    sel = StrategySelector(conservative=False)

    # Mock regime result — trending bullish strong
    mock_trending = {
        "regime":     "TRENDING",
        "direction":  "BULLISH",
        "strength":   "STRONG",
        "volatility": "NORMAL",
        "strategy":   {"risk_mult": 1.0, "type": "TREND_FOLLOW"},
    }

    mock_struct = {
        "structure_valid": True,
        "structure_bias":  "BULLISH",
        "structure_choch": "NONE",
    }

    choice = sel.select(mock_trending, mtf_bias={"bias": "BULLISH"}, structure=mock_struct)
    sel.print_summary(choice)

    # Mock ranging
    mock_ranging = {
        "regime":     "RANGING",
        "direction":  "NEUTRAL",
        "strength":   "WEAK",
        "volatility": "LOW",
        "strategy":   {"risk_mult": 0.8, "type": "RANGE"},
    }
    choice2 = sel.select(mock_ranging)
    sel.print_summary(choice2)
