"""
analysis/institutional_flow.py — Day 96 Institutional Flow + COT Intelligence
=============================================================================
Tracks institutional positioning via Commitment of Traders (COT) data
from the CFTC (Commodity Futures Trading Commission).

COT data shows what LARGE traders (banks, hedge funds, corporations)
are doing in the futures markets — this is the closest free proxy for
"institutional flow" available to retail traders.

Data source: CFTC publishes COT reports weekly (Friday data, released
Saturday). We fetch from the CFTC's public website or Barchart's free
API.

Free alternatives when COT unavailable:
  - Synthetic institutional flow from price action (large-candle detection)
  - DXY trend as a USD institutional flow proxy

Output:
    {
      "source":          "cot_live" | "synthetic" | "fallback",
      "pair":            "EURUSD",
      "institutional_bias":  "LONG",      # what institutions are doing
      "net_position":    125000,          # contracts net long
      "position_change": 15000,           # vs last week
      "confidence":      75,              # 0-100
      "retail_vs_inst":  "DIVERGENT",     # retail long but inst short = divergence
    }

Usage:
    from analysis.institutional_flow import InstitutionalFlowEngine
    engine = InstitutionalFlowEngine()
    result = engine.analyze("EURUSD", retail_long_pct=72.3)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("institutional_flow")


# ── CFTC COT symbol mapping ──────────────────────────────────────
# Forex pair → CFTC futures symbol
COT_SYMBOL_MAP = {
    # Names exactly as published in the current CME futures-only COT report.
    "EURUSD": "EURO FX",
    "GBPUSD": "BRITISH POUND",
    "USDJPY": "JAPANESE YEN",
    "USDCHF": "SWISS FRANC",
    "AUDUSD": "AUSTRALIAN DOLLAR",
    "USDCAD": "CANADIAN DOLLAR",
    "NZDUSD": "NEW ZEALAND DOLLAR",
    "XAUUSD": "GOLD",
}


class InstitutionalFlowEngine:
    """Institutional flow tracker via COT data + synthetic fallback."""

    # HONESTY FLAG (institutional review fix): CFTC COT HTML parsing was
    # never implemented — `_fetch_cot_data` used to make a real HTTP request
    # to CFTC and then unconditionally discard the response and return None
    # (see the comment that was here: "COT parsing is notoriously difficult
    # without a dedicated library"). That meant every call to `analyze()`
    # always resolved to the synthetic large-candle proxy or the flat
    # fallback, never real institutional positioning data, while wasting a
    # real network round-trip (up to the 15s timeout) doing so. This flag
    # documents that reality explicitly instead of leaving it as a silent
    # dead code path, and a one-time warning is logged so this is visible in
    # production logs. Behavior is otherwise unchanged: `analyze()` still
    # always falls through to synthetic/fallback exactly as before.
    COT_PARSING_IMPLEMENTED = True
    _cot_warning_logged = False

    def __init__(self):
        # NOTE: currently unused — kept for forward-compatibility with a
        # future real COT parser (see COT_PARSING_IMPLEMENTED above).
        self._cache: Dict[str, tuple] = {}  # symbol -> (timestamp, data)
        self.CACHE_TTL = 3600 * 6  # 6 hours (COT is weekly anyway)

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def analyze(self, pair: str, retail_long_pct: float = 50.0, df: pd.DataFrame = None) -> Dict[str, Any]:
        """Get institutional flow data for a pair.

        Args:
            pair:            e.g. "EURUSD"
            retail_long_pct: retail trader long % (for divergence check)
            df:              OHLCV data (for synthetic fallback)

        Returns: dict with institutional_bias, net_position, confidence, etc.

        NOTE (Round-10): real CFTC COT parsing is NOW IMPLEMENTED — see
        `_fetch_cot_from_cftc()`. If the CFTC website is reachable and
        the symbol is mapped, this method returns real net-positioning
        data. On fetch/parse failure, it falls back to the synthetic
        large-candle proxy (if `df` is supplied) or flat NEUTRAL.
        """
        # Try COT data first
        cot_data = self._fetch_cot_data(pair)

        if cot_data:
            return self._build_cot_result(pair, cot_data, retail_long_pct)
        elif df is not None:
            # Synthetic: detect institutional moves from large candles
            return self._build_synthetic_result(pair, df, retail_long_pct)
        else:
            return self._fallback_result(pair, "No COT data + no df for synthetic")

    # ─────────────────────────────────────────────────────────
    # COT DATA FETCH
    # ─────────────────────────────────────────────────────────

    def _fetch_cot_data(self, pair: str) -> Optional[Dict]:
        """Round-10 audit fix: fetch CFTC Commitment of Traders (CoT) data.

        Previously: this method was a stub that always returned None with
        a "not implemented" warning. The operator's audit noted this made
        the institutional flow module a "placeholder" with conf=0%.

        Now: attempts to fetch the most recent CFTC CoT report from the
        public CFTC website (https://www.cftc.gov/dea/futures/). The CFTC
        publishes weekly CoT reports in text format — we parse the
        "Non-Commercial Positions" section (large speculators) to
        determine net positioning.

        If the fetch fails (network error, parse error, symbol not found),
        falls back to None — the caller then uses the synthetic proxy.

        Args:
            pair: e.g. "EURUSD" → mapped to CFTC symbol "EURO FX"

        Returns:
            dict with keys: net_long, net_short, net_pct, confidence, source
            None on failure
        """
        cot_symbol = COT_SYMBOL_MAP.get(pair.upper())
        if not cot_symbol:
            return None

        # Round-10: try to fetch real CoT data
        try:
            return self._fetch_cot_from_cftc(cot_symbol, pair)
        except Exception as e:
            if not InstitutionalFlowEngine._cot_warning_logged:
                log.info(
                    f"[InstFlow] COT fetch for {pair} ({cot_symbol}) failed — "
                    f"using synthetic proxy: {e}"
                )
                InstitutionalFlowEngine._cot_warning_logged = True
            return None

    def _fetch_cot_from_cftc(self, cot_symbol: str, pair: str) -> Optional[Dict]:
        """Fetch and parse CFTC CoT text report for a given symbol.

        The CFTC publishes the current CME futures-only report at:
            https://www.cftc.gov/dea/futures/deacmelf.htm

        We parse the "Non-Commercial Positions" section to extract
        long/short/open-interest for large speculators.

        This is a BEST-EFFORT parser — CFTC's text format can vary.
        On any parse error, returns None (caller falls back to synthetic).
        """
        import requests
        import re

        # The old per-symbol URL construction (e.g. /futures/eur.htm)
        # returns 404.  CFTC now publishes all CME instruments in one report.
        url = "https://www.cftc.gov/dea/futures/deacmelf.htm"
        log.debug(f"[InstFlow] Fetching COT: {url}")

        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (ForexAI/1.0)"
        })
        if resp.status_code != 200:
            log.debug(f"[InstFlow] CFTC HTTP {resp.status_code} for {cot_symbol}")
            return None

        text = resp.text

        # Extract this instrument's block, then its first COMMITMENTS row.
        # In the legacy futures-only report, columns 1 and 2 are the
        # non-commercial long/short positions used for the positioning bias.
        match = re.search(
            rf"{re.escape(cot_symbol)}\s+-\s+CHICAGO MERCANTILE EXCHANGE"
            r".*?\bAll\s*:\s*[\d,]+\s*:\s*([\d,]+)\s+([\d,]+)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if not match:
            log.debug(f"[InstFlow] Could not parse COT section for {cot_symbol}")
            return None

        long_pos = int(match.group(1).replace(",", ""))
        short_pos = int(match.group(2).replace(",", ""))
        total = long_pos + short_pos

        if total == 0:
            return None

        net_long = long_pos - short_pos
        net_pct = (net_long / total) * 100.0

        # Confidence: higher when positioning is more extreme (>30% net)
        confidence = min(100, abs(net_pct) * 2)

        log.info(
            f"[InstFlow] COT {pair} ({cot_symbol}): "
            f"long={long_pos:,} short={short_pos:,} "
            f"net={net_long:+,} ({net_pct:+.1f}%) conf={confidence:.0f}%"
        )

        return {
            "long": long_pos,
            "short": short_pos,
            "net_long": net_long,
            "net_pct": round(net_pct, 1),
            "confidence": round(confidence, 1),
            "source": "cftc_cot",
            "url": url,
        }


    def _build_cot_result(self, pair: str, cot: Dict, retail_long: float) -> Dict[str, Any]:
        """Build result from live COT data.

        Round-10: updated to use the new key names from _fetch_cot_from_cftc:
            net_long (was net_position)
            net_pct (was position_change)
            confidence (was calculated here)
        """
        net = cot.get("net_long", cot.get("net_position", 0))
        net_pct = cot.get("net_pct", 0)
        confidence = cot.get("confidence", 50.0)

        # Institutional bias: net positive = institutions long
        if net > 0:
            inst_bias = "LONG"
        elif net < 0:
            inst_bias = "SHORT"
        else:
            inst_bias = "NEUTRAL"

        # Divergence check: retail long but institutions short = SELL signal
        retail_bias = "LONG" if retail_long > 55 else "SHORT" if retail_long < 45 else "NEUTRAL"
        divergence = "DIVERGENT" if retail_bias != inst_bias and inst_bias != "NEUTRAL" else "ALIGNED"

        # Confidence based on position size + net_pct change
        # Round-19 audit fix: 'change' was undefined (leftover from old
        # variable name). The correct variable is 'net_pct' (renamed in
        # Round-10 but these two lines were missed).
        confidence = min(100, abs(net) / 1000 + abs(net_pct) / 500)

        result = {
            "source":              "cot_live",
            "pair":                pair,
            "institutional_bias":  inst_bias,
            "net_position":        net,
            "position_change":     net_pct,  # Round-19: was 'change' (undefined)
            "confidence":          int(confidence),
            "retail_vs_inst":      divergence,
            "retail_bias":         retail_bias,
            "fetched_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        log.info(
            f"[InstFlow] {pair} | inst={inst_bias} (net={net}) | "
            f"retail={retail_bias} | {divergence} | conf={confidence:.0f}%"
        )
        return result

    # ─────────────────────────────────────────────────────────
    # SYNTHETIC INSTITUTIONAL FLOW (from price action)
    # ─────────────────────────────────────────────────────────

    def _build_synthetic_result(self, pair: str, df: pd.DataFrame, retail_long: float) -> Dict[str, Any]:
        """Estimate institutional flow from large-candle (displacement) analysis.

        Institutional orders create large directional candles (displacement).
        By analyzing the ratio of large bullish vs bearish candles, we can
        estimate institutional direction.
        """
        if df is None or len(df) < 20:
            return self._fallback_result(pair, "insufficient data for synthetic")

        try:
            closes = df["close"].values
            opens = df["open"].values
            bodies = closes[-50:] - opens[-50:]  # last 50 candle bodies

            # Large candles = institutional activity (body > 1.5x average)
            avg_body = np.mean(np.abs(bodies))
            if avg_body == 0:
                return self._fallback_result(pair, "flat market")

            large_bullish = sum(1 for b in bodies if b > 0 and abs(b) > 1.5 * avg_body)
            large_bearish = sum(1 for b in bodies if b < 0 and abs(b) > 1.5 * avg_body)

            net_large = large_bullish - large_bearish

            if net_large > 3:
                inst_bias = "LONG"
            elif net_large < -3:
                inst_bias = "SHORT"
            else:
                inst_bias = "NEUTRAL"

            # Divergence
            retail_bias = "LONG" if retail_long > 55 else "SHORT" if retail_long < 45 else "NEUTRAL"
            divergence = "DIVERGENT" if retail_bias != inst_bias and inst_bias != "NEUTRAL" else "ALIGNED"

            confidence = min(100, abs(net_large) * 15)

            result = {
                "source":              "synthetic_displacement",
                "pair":                pair,
                "institutional_bias":  inst_bias,
                "net_position":        net_large,
                "position_change":     0,
                "confidence":          int(confidence),
                "retail_vs_inst":      divergence,
                "retail_bias":         retail_bias,
                "large_bullish":       large_bullish,
                "large_bearish":       large_bearish,
                "fetched_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            log.info(
                f"[InstFlow] {pair} | synthetic: inst={inst_bias} "
                f"(bull={large_bullish}/bear={large_bearish}) | "
                f"retail={retail_bias} | {divergence} | conf={confidence:.0f}%"
            )
            return result
        except Exception as e:
            return self._fallback_result(pair, f"synthetic failed: {e}")

    # ─────────────────────────────────────────────────────────
    # FALLBACK
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_result(pair: str, reason: str) -> Dict[str, Any]:
        return {
            "source":              "fallback",
            "pair":                pair,
            "institutional_bias":  "NEUTRAL",
            "net_position":        0,
            "position_change":     0,
            "confidence":          0,
            "retail_vs_inst":      "UNKNOWN",
            "retail_bias":         "NEUTRAL",
            "reason":              reason,
            "fetched_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ─────────────────────────────────────────────────────────
    # AI CONTEXT
    # ─────────────────────────────────────────────────────────

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "inst_source":          result.get("source", "fallback"),
            "inst_bias":            result.get("institutional_bias", "NEUTRAL"),
            "inst_confidence":      result.get("confidence", 0),
            "inst_retail_vs_inst":  result.get("retail_vs_inst", "UNKNOWN"),
            "inst_divergent":       result.get("retail_vs_inst") == "DIVERGENT",
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  🏦  INSTITUTIONAL FLOW  (Day 96)")
        log.info(bar)
        log.info(f"  Pair           : {result.get('pair','?')}")
        log.info(f"  Source         : {result.get('source','?')}")
        log.info(f"  Inst bias      : {result.get('institutional_bias','?')}")
        log.info(f"  Confidence     : {result.get('confidence',0)}%")
        log.info(f"  Retail vs Inst : {result.get('retail_vs_inst','?')}")
        if result.get("large_bullish") is not None:
            log.info(f"  Large bullish  : {result['large_bullish']}")
            log.info(f"  Large bearish  : {result['large_bearish']}")
        log.info(bar)
