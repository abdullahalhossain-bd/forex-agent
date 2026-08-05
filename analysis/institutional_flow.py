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
import time
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

    # FIX (audit H4/H5): how many days old a report can be before we treat
    # it as stale rather than current. CFTC's normal cadence is weekly
    # (Friday positioning, released the following week); this gives slack
    # for that lag without silently accepting month-old data as current.
    STALE_THRESHOLD_DAYS = 10

    def __init__(self):
        # FIX (audit H1/H2): this used to be declared but never read from
        # or written to anywhere — every analyze() call re-fetched from
        # CFTC over HTTP even though COT only updates weekly. Now actually
        # used by _fetch_cot_data() below.
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
        # P1 perf fix (2026-08-03, parity investigation): skip the CFTC
        # COT scrape in backtest mode. _fetch_cot_data() →
        # _fetch_cot_from_cftc() does an HTTPS GET to cftc.gov with a
        # 15s timeout per call. In backtest mode this returns TODAY's
        # COT report misapplied to a historical bar — a parity bug as
        # well as a perf bug. If `df` is available, fall through to the
        # synthetic large-candle proxy (which is purely computational,
        # no network) so the module still produces a meaningful signal
        # from the historical OHLCV. If no df, return flat NEUTRAL.
        from core.constants import is_backtest_mode
        if is_backtest_mode():
            if df is not None:
                return self._build_synthetic_result(pair, df, retail_long_pct)
            return self._fallback_result(pair, "backtest mode — COT fetch skipped, no df for synthetic")

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

        # FIX (audit H1/H2): serve from cache within TTL instead of making
        # a fresh HTTPS request to CFTC on every single analyze() call —
        # COT only updates weekly, so re-fetching every call was pure
        # waste (and a real network round-trip on the hot path).
        cache_key = pair.upper()
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached_at, cached_data = cached
            if (time.time() - cached_at) < self.CACHE_TTL:
                log.debug(f"[InstFlow] Using cached COT data for {pair} (age={time.time()-cached_at:.0f}s)")
                return cached_data

        # Round-10: try to fetch real CoT data
        try:
            data = self._fetch_cot_from_cftc(cot_symbol, pair)
        except Exception as e:
            if not InstitutionalFlowEngine._cot_warning_logged:
                log.info(
                    f"[InstFlow] COT fetch for {pair} ({cot_symbol}) failed — "
                    f"using synthetic proxy: {e}"
                )
                InstitutionalFlowEngine._cot_warning_logged = True
            return None

        # Only cache real, successful fetches — never cache a None/failure,
        # so a transient CFTC outage doesn't lock the engine out of real
        # data for the full 6-hour TTL once the site recovers.
        if data is not None:
            self._cache[cache_key] = (time.time(), data)
        return data

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

        # FIX (audit H4/H5): CFTC releases this report weekly (Friday
        # positioning data, published the following week — historically
        # Saturday, currently Friday afternoon). Without checking the
        # report's own as-of date, a stale/cached-by-CFTC page from weeks
        # ago would be used as if it were this week's positioning with no
        # indication anything was off. Parse the report date out of the
        # page header (best-effort — CFTC's text format isn't guaranteed),
        # and flag the data as stale if it's meaningfully older than one
        # normal release cycle should allow.
        report_date = self._extract_report_date(text)
        data_age_days = None
        stale = False
        if report_date is not None:
            data_age_days = (datetime.now(timezone.utc) - report_date).days
            # One release cycle is ~7 days; allow slack for the
            # Friday-data/next-week-release lag before calling it stale.
            stale = data_age_days > self.STALE_THRESHOLD_DAYS
        else:
            # Couldn't verify the date at all — treat conservatively as
            # unverified rather than silently assuming it's fresh.
            log.debug(f"[InstFlow] Could not parse report date for {cot_symbol}; freshness unverified")

        log.info(
            f"[InstFlow] COT {pair} ({cot_symbol}): "
            f"long={long_pos:,} short={short_pos:,} "
            f"net={net_long:+,} ({net_pct:+.1f}%) conf={confidence:.0f}% "
            f"report_date={report_date.date() if report_date else 'unknown'} "
            f"age_days={data_age_days} stale={stale}"
        )

        return {
            "long": long_pos,
            "short": short_pos,
            "net_long": net_long,
            "net_pct": round(net_pct, 1),
            "confidence": round(confidence, 1),
            "source": "cftc_cot",
            "url": url,
            "report_date": report_date.isoformat() if report_date else None,
            "data_age_days": data_age_days,
            "stale": stale,
        }

    @staticmethod
    def _extract_report_date(text: str) -> Optional[datetime]:
        """Best-effort extraction of the report's 'as of' date from the
        CFTC text header. Only looks near the top of the document, where
        the report date normally appears — searching the whole (large)
        report body risks matching an unrelated date deep in the data.
        Returns None (never guesses) if no recognizable date is found.
        """
        import re
        header = text[:3000]
        patterns_and_formats = [
            (r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", ("%m/%d/%y", "%m/%d/%Y")),
            (r"\b([A-Za-z]+ \d{1,2},\s*\d{4})\b", ("%B %d, %Y", "%b %d, %Y")),
        ]
        for pattern, formats in patterns_and_formats:
            m = re.search(pattern, header)
            if not m:
                continue
            for fmt in formats:
                try:
                    dt = datetime.strptime(m.group(1), fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        return None


    def _build_cot_result(self, pair: str, cot: Dict, retail_long: float) -> Dict[str, Any]:
        """Build result from live COT data.

        Round-10: updated to use the new key names from _fetch_cot_from_cftc:
            net_long (was net_position)
            net_pct (was position_change)
            confidence (was calculated here)

        FIX (audit C1): this used to recompute `confidence` a second time
        with `min(100, abs(net) / 1000 + abs(net_pct) / 500)`, silently
        discarding the confidence `_fetch_cot_from_cftc()` had already
        computed from how extreme the net positioning is. That second
        formula was leftover from an earlier design (see the Round-19 note
        that used to be here about an undefined `change` variable) and is
        never used now — the parser's confidence is the one number that
        actually reflects this specific report.

        FIX (audit C2): COT data is a WEEKLY snapshot (Friday positioning,
        released the following week) — treating it as `source: "cot_live"`
        misrepresents it as a fresh/intraday read. It's a strategic bias,
        not a live signal, and it can be several days old even when
        "fresh" by CFTC's own schedule. This now labels the source
        `"cot_weekly"`, carries the parsed report date + age through to
        the result, and marks the signal explicitly as STRATEGIC so
        downstream consumers (MasterAnalyst etc.) don't weight it like an
        intraday indicator. If the report is older than expected for the
        normal weekly release cycle, `stale` is set True and confidence is
        further discounted rather than presented at full strength.
        """
        net = cot.get("net_long", cot.get("net_position", 0))
        net_pct = cot.get("net_pct", 0)
        confidence = cot.get("confidence", 50.0)

        report_date = cot.get("report_date")
        data_age_days = cot.get("data_age_days")
        stale = bool(cot.get("stale", False))
        if stale:
            # Old report still usable (better than nothing) but shouldn't
            # be presented with the same confidence as a fresh one.
            confidence = round(confidence * 0.5, 1)

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

        result = {
            "source":              "cot_weekly",
            "signal_type":         "STRATEGIC",   # weekly positioning, NOT an intraday/live signal
            "pair":                pair,
            "institutional_bias":  inst_bias,
            "net_position":        net,
            "position_change":     net_pct,
            "confidence":          int(confidence),
            "retail_vs_inst":      divergence,
            "retail_bias":         retail_bias,
            "report_date":         report_date,
            "data_age_days":       data_age_days,
            "stale":               stale,
            "fetched_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        log.info(
            f"[InstFlow] {pair} | inst={inst_bias} (net={net}) | "
            f"retail={retail_bias} | {divergence} | conf={confidence:.0f}% | "
            f"report_date={report_date} age_days={data_age_days} stale={stale}"
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
                "signal_type":         "SYNTHETIC",  # price-action proxy, not real COT positioning
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
            "signal_type":         "UNKNOWN",
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
            "inst_signal_type":     result.get("signal_type", "SYNTHETIC"),  # STRATEGIC (weekly COT) vs SYNTHETIC vs UNKNOWN
            "inst_bias":            result.get("institutional_bias", "NEUTRAL"),
            "inst_confidence":      result.get("confidence", 0),
            "inst_retail_vs_inst":  result.get("retail_vs_inst", "UNKNOWN"),
            "inst_divergent":       result.get("retail_vs_inst") == "DIVERGENT",
            "inst_report_date":     result.get("report_date"),
            "inst_data_age_days":   result.get("data_age_days"),
            "inst_stale":           result.get("stale", False),
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  🏦  INSTITUTIONAL FLOW  (Day 96)")
        log.info(bar)
        log.info(f"  Pair           : {result.get('pair','?')}")
        log.info(f"  Source         : {result.get('source','?')} ({result.get('signal_type','?')})")
        log.info(f"  Inst bias      : {result.get('institutional_bias','?')}")
        log.info(f"  Confidence     : {result.get('confidence',0)}%")
        log.info(f"  Retail vs Inst : {result.get('retail_vs_inst','?')}")
        if result.get("report_date"):
            log.info(f"  Report date    : {result['report_date']} (age {result.get('data_age_days')}d, stale={result.get('stale', False)})")
        if result.get("large_bullish") is not None:
            log.info(f"  Large bullish  : {result['large_bullish']}")
            log.info(f"  Large bearish  : {result['large_bearish']}")
        log.info(bar)