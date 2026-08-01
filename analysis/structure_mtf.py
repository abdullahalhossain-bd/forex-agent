# analysis/structure_mtf.py  —  Day 88 | Multi-Timeframe Structure Engine
# ============================================================
# আপনার বলা "Internal vs External Structure (H4/D1 vs M5/M15)"
# এটাই সেই module।
#
# SMC professional approach:
#   External Structure (HTF: H4 / D1)
#     - বড় timeframe-এ overall bias (bullish / bearish)
#     - Large swing points (HTF HH, HL, LH, LL)
#     - HTF BOS / CHoCH
#     - HTF Order Block ও FVG
#
#   Internal Structure (LTF: M5 / M15)
#     - ছোট timeframe-এ micro structure
#     - Entry-level refinement
#     - Internal BOS / CHoCH (refined entry timing)
#     - LTF OB / FVG (precision entry)
#
# Strategy:
#   1. HTF bias determine করো (external)
#   2. LTF-তে HTF direction-এ entry খোঁজো (internal)
#   3. HTF bearish কিন্তু LTF bullish → WAIT (conflict)
#   4. HTF ও LTF একমত → high-probability entry
#
# এই module MarketStructureEngine (Day 61) কে wrap করে এবং
# দুটা timeframe এর result merge করে combined context দেয়।
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional

from analysis.structure import MarketStructureEngine
from utils.logger import get_logger

log = get_logger("structure_mtf_engine")


# Standard MTF configuration — আপনার বলা "D1 ↓ H4 ↓ H1 ↓ M15 ↓ M5 Entry"
# কিন্তু implementation-এ আমরা শুধু ২টা tier ব্যবহার করি (external + internal)
# কারণ ৩-৫টা tier একসাথে দিলে signal latency বাড়ে।
DEFAULT_EXTERNAL_TF = "H4"
DEFAULT_INTERNAL_TF = "M15"


class MTFStructureEngine:
    """
    Internal vs External structure analyzer।

    Usage:
        engine = MTFStructureEngine(
            external_swing_window=8,   # HTF — larger window
            internal_swing_window=3,   # LTF — smaller window
        )

        # df_external: H4 OHLC, df_internal: M15 OHLC
        result = engine.analyze(df_external=df_h4, df_internal=df_m15)

        ctx = engine.get_ai_context(result)
    """

    def __init__(
        self,
        external_swing_window: int = 8,
        internal_swing_window: int = 3,
        external_tf:           str = DEFAULT_EXTERNAL_TF,
        internal_tf:           str = DEFAULT_INTERNAL_TF,
    ):
        """
        external_swing_window : HTF-র জন্য swing detection window (বড়)
        internal_swing_window : LTF-র জন্য swing detection window (ছোট)
        external_tf / internal_tf : শুধু label, যাতে log-এ বোঝা যায়
        """
        self.external_engine = MarketStructureEngine(swing_window=external_swing_window)
        self.internal_engine = MarketStructureEngine(swing_window=internal_swing_window)
        self.external_tf     = external_tf
        self.internal_tf     = internal_tf

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════

    def analyze(
        self,
        df_external: Optional[Any] = None,
        df_internal: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        df_external : HTF DataFrame (e.g. H4)
        df_internal : LTF DataFrame (e.g. M15)

        দুটোই দিলে best result। শুধু একটা দিলে সেটাই analyze করে
        এবং missing tier-কে "UNKNOWN" চিহ্নিত করে।
        """
        ext_result = None
        int_result = None

        if df_external is not None:
            ext_result = self.external_engine.analyze(df_external)
        if df_internal is not None:
            int_result = self.internal_engine.analyze(df_internal)

        if ext_result is None and int_result is None:
            return self._empty_result("Both dataframes are None")

        # Determine biases
        ext_bias = ext_result.get("structure", "NEUTRAL") if ext_result else "UNKNOWN"
        int_bias = int_result.get("structure", "NEUTRAL") if int_result else "UNKNOWN"

        # Conflict detection
        conflict = self._detect_conflict(ext_bias, int_bias)

        # Alignment
        alignment = self._alignment(ext_bias, int_bias)

        # Combined bias (external dominates)
        combined_bias = self._combined_bias(ext_bias, int_bias, conflict)

        # Trading permission
        permission = self._trading_permission(ext_bias, int_bias, conflict)

        # Build result
        result = {
            "valid":            True,
            "external_tf":      self.external_tf,
            "internal_tf":      self.internal_tf,
            "external":         ext_result,
            "internal":         int_result,
            "external_bias":    ext_bias,
            "internal_bias":    int_bias,
            "combined_bias":    combined_bias,
            "alignment":        alignment,
            "conflict":         conflict,
            "trade_permission": permission,
            "note":             self._note(ext_bias, int_bias, alignment, conflict),
        }

        log.info(
            f"[MTFStructure] Ext({self.external_tf})={ext_bias} | "
            f"Int({self.internal_tf})={int_bias} | "
            f"combined={combined_bias} | align={alignment} | "
            f"conflict={conflict} | permission={permission}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # CONFLICT DETECTION
    # ═══════════════════════════════════════════════════════

    def _detect_conflict(self, ext_bias: str, int_bias: str) -> bool:
        """
        Conflict = external ও internal বিপরীত দিকে।
        """
        if ext_bias == "UNKNOWN" or int_bias == "UNKNOWN":
            return False
        if ext_bias == "NEUTRAL" or int_bias == "NEUTRAL":
            return False
        if ext_bias == "RANGING" or int_bias == "RANGING":
            return False
        return ext_bias != int_bias

    def _alignment(self, ext_bias: str, int_bias: str) -> str:
        """
        ALIGNED    : দুটোই একই দিকে (BULLISH-BULLISH বা BEARISH-BEARISH)
        CONFLICT   : বিপরীত দিকে
        PARTIAL    : একটা NEUTRAL/RANGING, অন্যটা directional
        INCOMPLETE : একটা UNKNOWN
        """
        if ext_bias == "UNKNOWN" or int_bias == "UNKNOWN":
            return "INCOMPLETE"

        directional = {"BULLISH", "BEARISH"}
        if ext_bias in directional and int_bias in directional:
            if ext_bias == int_bias:
                return "ALIGNED"
            return "CONFLICT"

        # At least one is NEUTRAL/RANGING
        if ext_bias in directional or int_bias in directional:
            return "PARTIAL"
        return "PARTIAL"

    # ═══════════════════════════════════════════════════════
    # COMBINED BIAS
    # ═══════════════════════════════════════════════════════

    def _combined_bias(self, ext_bias: str, int_bias: str, conflict: bool) -> str:
        """
        External bias dominates। কিন্তু conflict থাকলে NEUTRAL।
        External UNKNOWN হলে internal fallback (weak)।
        """
        if conflict:
            return "NEUTRAL"
        if ext_bias in ("BULLISH", "BEARISH"):
            return ext_bias
        if int_bias in ("BULLISH", "BEARISH"):
            return int_bias   # weak signal, only LTF confirmation
        return "NEUTRAL"

    # ═══════════════════════════════════════════════════════
    # TRADING PERMISSION
    # ═══════════════════════════════════════════════════════

    def _trading_permission(self, ext_bias: str, int_bias: str, conflict: bool) -> str:
        """
        TRADE_ALLOWED  : aligned (best case)
        WAIT_CONFIRM   : partial / one tier missing
        NO_TRADE       : conflict or both neutral
        """
        if conflict:
            return "NO_TRADE"
        if ext_bias in ("BULLISH", "BEARISH") and int_bias == ext_bias:
            return "TRADE_ALLOWED"
        if ext_bias in ("BULLISH", "BEARISH") and int_bias in ("NEUTRAL", "RANGING"):
            return "WAIT_CONFIRM"   # HTF clear, LTF not yet confirming
        if int_bias in ("BULLISH", "BEARISH") and ext_bias in ("NEUTRAL", "RANGING", "UNKNOWN"):
            return "WAIT_CONFIRM"   # LTF signal but HTF unclear
        return "NO_TRADE"

    # ═══════════════════════════════════════════════════════
    # NOTE BUILDER
    # ═══════════════════════════════════════════════════════

    def _note(
        self, ext_bias: str, int_bias: str,
        alignment: str, conflict: bool,
    ) -> str:
        if alignment == "INCOMPLETE":
            return f"Missing tier data — ext={ext_bias}, int={int_bias}"
        if conflict:
            return f"Conflict: {self.external_tf}={ext_bias} vs {self.internal_tf}={int_bias} — avoid trades"
        if alignment == "ALIGNED":
            return f"Aligned {ext_bias} — high-probability {ext_bias.lower()} setups allowed"
        return f"Partial: ext={ext_bias}, int={int_bias} — wait for confirmation"

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not result.get("valid"):
            return {
                "mtf_structure_valid":     False,
                "mtf_combined_bias":       "NEUTRAL",
                "mtf_alignment":           "INCOMPLETE",
                "mtf_trade_permission":    "NO_TRADE",
                "mtf_external_bias":       "UNKNOWN",
                "mtf_internal_bias":       "UNKNOWN",
            }

        # Pull HTF BOS/CHoCH for context
        ext = result.get("external") or {}
        int_ = result.get("internal") or {}

        return {
            "mtf_structure_valid":     True,
            "mtf_combined_bias":       result.get("combined_bias"),
            "mtf_alignment":           result.get("alignment"),
            "mtf_conflict":            result.get("conflict", False),
            "mtf_trade_permission":    result.get("trade_permission"),
            "mtf_external_bias":       result.get("external_bias"),
            "mtf_internal_bias":       result.get("internal_bias"),
            "mtf_external_bos":        ext.get("bos", {}).get("event", "NONE") if ext else "NONE",
            "mtf_external_choch":      ext.get("choch", {}).get("event", "NONE") if ext else "NONE",
            "mtf_internal_bos":        int_.get("bos", {}).get("event", "NONE") if int_ else "NONE",
            "mtf_internal_choch":      int_.get("choch", {}).get("event", "NONE") if int_ else "NONE",
        }

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "valid":            False,
            "reason":           reason,
            "external_bias":    "UNKNOWN",
            "internal_bias":    "UNKNOWN",
            "combined_bias":    "NEUTRAL",
            "alignment":        "INCOMPLETE",
            "conflict":         False,
            "trade_permission": "NO_TRADE",
            "note":             reason,
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  🏛️  MTF STRUCTURE ENGINE  (Day 88)")
        log.info(bar)

        if not result.get("valid"):
            log.info(f"  ⚠️  {result.get('reason', 'No analysis')}")
            log.info(bar)
            return

        # External
        ext = result.get("external")
        if ext:
            icon = {"BULLISH": "🟢", "BEARISH": "🔴", "RANGING": "🟡"}.get(
                ext.get("structure", "NEUTRAL"), "⚪"
            )
            log.info(f"  External ({self.external_tf}): {icon}  {ext.get('structure', 'NEUTRAL')}")
            log.info(f"    BOS    : {ext.get('bos', {}).get('event', 'NONE')}")
            log.info(f"    CHoCH  : {ext.get('choch', {}).get('event', 'NONE')}")
        else:
            log.info(f"  External ({self.external_tf}): ⚪  no data")

        # Internal
        int_ = result.get("internal")
        if int_:
            icon = {"BULLISH": "🟢", "BEARISH": "🔴", "RANGING": "🟡"}.get(
                int_.get("structure", "NEUTRAL"), "⚪"
            )
            log.info(f"  Internal ({self.internal_tf}): {icon}  {int_.get('structure', 'NEUTRAL')}")
            log.info(f"    BOS    : {int_.get('bos', {}).get('event', 'NONE')}")
            log.info(f"    CHoCH  : {int_.get('choch', {}).get('event', 'NONE')}")
        else:
            log.info(f"  Internal ({self.internal_tf}): ⚪  no data")

        log.info("")
        log.info(f"  Combined Bias    : {result.get('combined_bias')}")
        log.info(f"  Alignment        : {result.get('alignment')}")
        log.info(f"  Conflict         : {result.get('conflict')}")
        log.info(f"  Trade Permission : {result.get('trade_permission')}")
        log.info(f"  Note             : {result.get('note')}")
        log.info(bar)


# ═══════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    # External (H4): strong uptrend
    n_ext = 200
    ext_prices = 1.1000 + np.cumsum(np.random.randn(n_ext) * 0.001 + 0.0005)
    df_h4 = pd.DataFrame({
        "open":  ext_prices,
        "high":  ext_prices + 0.001,
        "low":   ext_prices - 0.001,
        "close": ext_prices,
    })

    # Internal (M15): bullish (aligned)
    n_int = 300
    int_prices = 1.1000 + np.cumsum(np.random.randn(n_int) * 0.0005 + 0.0002)
    df_m15 = pd.DataFrame({
        "open":  int_prices,
        "high":  int_prices + 0.0005,
        "low":   int_prices - 0.0005,
        "close": int_prices,
    })

    engine = MTFStructureEngine(external_swing_window=8, internal_swing_window=3)
    result = engine.analyze(df_external=df_h4, df_internal=df_m15)
    engine.print_summary(result)

    ctx = engine.get_ai_context(result)
    print("\nAI Context:")
    for k, v in ctx.items():
        print(f"  {k:<28}: {v}")
