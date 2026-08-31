"""
core/da_safety_net.py — Devil's Advocate 6-Point Deterministic Safety Net
=========================================================================
2026-08-27 hardening layer, born from a live-trade audit that identified
six systemic WR-killers the LLM reviewer could not reliably catch because
it either wasn't shown the data or was trusted to "notice" it:

    1. Session filter      — trades taken in Asian-session dead chop.
    2. Trend filter        — BUY signals taken against an active H4 downtrend
                             (counter-trend losses).
    3. Spread check        — trades executed while rollover/news spread spikes.
    4. ATR regime check    — entries into dead markets (ATR collapse) and
                             news spikes (single-candle blowouts).
    5. Structure-based SL  — SL placed by ATR alone, ignoring swing high/low,
                             so noise stops out trades before structure is
                             actually invalidated.
    6. Lot sizing check    — pip-value approximation errors on JPY/Gold/exotic
                             symbols silently pushing real risk above 1%.

Design rules:
    - DETERMINISTIC ONLY. No LLM call happens in this module. Every verdict
      is reproducible from the inputs alone so backtests can trust it.
    - VETO short-circuits the Devil's Advocate BEFORE the provider call:
      cheaper (no API spend), faster (no 25s timeout budget), and immune to
      model hallucination. The upstream trader.py already treats a DA
      decision of REJECT as a hard veto, so no executor changes are needed.
    - WARNs never block; they are surfaced into the DA evidence payload and
      preserved in the audit journal for later expectancy analysis.
    - Missing data produces SKIP, never a fabricated PASS. The module only
      vetoes on evidence it can actually see — same philosophy as the DA
      gate itself ("tell the reviewer what it does NOT know").

Toggles (env):
    DA_SAFETY_NET_ENABLED     default true   — master switch
    DA_SAFETY_NET_MODE        enforce | warn_only   (default enforce)
    DA_MIN_ATR_PIPS           default 4.0    — dead-market floor (in pips)
    DA_NEWS_SPIKE_ATR_MULT    default 2.5    — candle > N*ATR = spike entry
    DA_MAX_SPREAD_RATIO       default 0.35   — spread/ATR WARN threshold
    DA_MAX_SPREAD_RATIO_VETO  default 0.50   — spread/ATR hard VETO (was single 0.35)
    DA_STRUCT_SL_BUFFER_PIPS  default 2.0    — SL must clear structure by this
    DA_STRUCT_PROXIMITY_MULT  default 2.0    — ignore levels farther than
                                               N x sl_distance from entry
    DA_MIN_SL_PIPS            default 6.0    — absolute noise-floor for any SL
    DA_LOT_MISMATCH_TOL       default 0.25   — relative lot deviation allowed

The session check deliberately requires an explicit wall-clock opt-in
(``da_allow_wallclock_session`` on trade_context) when no session data was
supplied: guessing "now" during a research/backtest replay would veto on the
developer's local clock instead of the bar's time. core/trader.py sets that
flag on the live pipeline only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("da_safety_net")

VERDICT_VETO = "VETO"
VERDICT_WARN = "WARN"
VERDICT_PASS = "PASS"
VERDICT_SKIP = "SKIP"

_BULL_KW = ("bull", "buy", "long", "up")
_BEAR_KW = ("bear", "sell", "short", "down")

# Sessions counted as "real liquidity" for the chop filter.
_MAJOR_SESSIONS = {"london", "new_york"}
# Currencies with native liquidity during the Tokyo/Sydney window.
_ASIAN_NATIVE_CURRENCIES = ("JPY", "AUD", "NZD")

# 2026-08-27 robustness fix: the legacy "session" key sometimes carries a
# QUALITY string (e.g. SessionAnalyzer.trade_quality = "🟢 BEST — highest
# liquidity") instead of a session name. Only tokens from this vocabulary
# count as session data; anything else means "no reliable session info" and
# must SKIP rather than risk a false asian_dead_chop veto.
_KNOWN_SESSIONS = {"sydney", "tokyo", "london", "new_york", "newyork"}

_UNSET = object()


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.getenv(name, "").strip() or default)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


@dataclass
class CheckOutcome:
    """Result of one deterministic safety-net check."""
    name: str
    verdict: str          # VETO / WARN / PASS / SKIP
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.name,
            "verdict": self.verdict,
            "reason": self.reason,
            **({"details": self.details} if self.details else {}),
        }


class DASafetyNet:
    """Six deterministic checks run inside DevilsAdvocateGate.review()
    before any LLM involvement."""

    def __init__(self) -> None:
        self.enabled = self._flag("DA_SAFETY_NET_ENABLED", True)
        mode = (os.getenv("DA_SAFETY_NET_MODE", "enforce") or "enforce").strip().lower()
        self.mode = mode if mode in {"enforce", "warn_only"} else "enforce"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _flag(name: str, default: bool) -> bool:
        v = os.getenv(name, "").strip().lower()
        if not v:
            return default
        return v in {"1", "true", "yes", "on"}

    @staticmethod
    def _f(value: Any) -> Optional[float]:
        try:
            f = float(value)
            return f
        except (TypeError, ValueError):
            return None

    @classmethod
    def _trend_opposes(cls, trend: Any, signal: str) -> bool:
        t = str(trend or "").lower()
        if t in ("", "unknown", "sideways", "neutral", "ranging", "none", "range"):
            return False
        if signal == "BUY":
            return any(k in t for k in _BEAR_KW)
        if signal == "SELL":
            return any(k in t for k in _BULL_KW)
        return False

    @staticmethod
    def _clean_symbol(symbol: Any) -> str:
        s = str(symbol or "").upper().replace("/", "").replace("=X", "")
        # strip broker suffix like 'm' (EURUSDm) conservatively: only when
        # the result still looks like a valid pair length
        if len(s) > 6 and s[:6] in {
            "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "US30", "NAS100",
        } :
            return s[:6]
        if len(s) >= 6 and not s[:3].isalpha():
            return s[:6]
        return s[:6]

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        trade_context: Dict[str, Any],
        signal: str,
        risk_out: Dict[str, Any],
        decision_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all six checks. Returns a structured summary dict."""
        checks: List[CheckOutcome] = []
        mc = trade_context.get("market_context") or {}
        ind_ctx = (decision_out.get("ind_ctx") or {}) if isinstance(decision_out, dict) else {}
        sr_ctx = (decision_out.get("sr_ctx") or {}) if isinstance(decision_out, dict) else {}
        regime = (decision_out.get("regime") or {}) if isinstance(decision_out, dict) else {}
        analysis_out = trade_context.get("analysis_out") or {}

        symbol = str(trade_context.get("symbol") or trade_context.get("pair") or "")

        sig = str(signal or "").upper()

        checks.append(self._check_session(trade_context, mc, symbol))
        checks.append(self._check_h4_trend(mc, decision_out, sig))
        news_active = self._news_is_active(analysis_out)
        checks.append(self._check_spread(symbol, ind_ctx, analysis_out, news_active))
        checks.append(self._check_atr_regime(ind_ctx, regime, mc))
        checks.append(self._check_structure_sl(sig, risk_out, mc, sr_ctx, ind_ctx))
        checks.append(self._check_lot_sizing(symbol, risk_out))

        vetoes = [c for c in checks if c.verdict == VERDICT_VETO]
        warnings = [c for c in checks if c.verdict == VERDICT_WARN]

        effective_vetoes: List[CheckOutcome] = []
        if vetoes and self.mode == "warn_only":
            # operator chose observability-first: downgrade to warnings
            warnings = warnings + [
                CheckOutcome(v.name, VERDICT_WARN,
                             f"[warn_only] {v.reason}", v.details)
                for v in vetoes
            ]
        else:
            effective_vetoes = vetoes

        summary = {
            "enabled": self.enabled,
            "mode": self.mode,
            "vetoes": [v.to_dict() for v in effective_vetoes],
            "warnings": [w.to_dict() for w in warnings],
            "checks": [c.to_dict() for c in checks],
            "pass_count": sum(1 for c in checks if c.verdict == VERDICT_PASS),
            "skip_count": sum(1 for c in checks if c.verdict == VERDICT_SKIP),
        }
        if effective_vetoes:
            log.warning(
                f"[DASafetyNet] {symbol}: {len(effective_vetoes)} veto(s): "
                f"{[f'{v.name}({v.reason[:60]})' for v in effective_vetoes]}"
            )
        elif warnings:
            log.info(
                f"[DASafetyNet] {symbol}: {len(warnings)} warning(s), no veto"
            )
        return summary

    # ------------------------------------------------------------------
    # 1. Session filter
    # ------------------------------------------------------------------
    def _check_session(self, trade_context: Dict[str, Any], mc: Dict[str, Any], symbol: str) -> CheckOutcome:
        name = "session_filter"
        sessions: Any = _UNSET

        raw_list = mc.get("sessions_active")
        if isinstance(raw_list, list):
            parsed = [str(s).strip().lower() for s in raw_list if s]
            sessions = [s for s in parsed if s in _KNOWN_SESSIONS]
            if parsed and not sessions:
                # caller supplied data but NONE of it is a recognizable
                # session name — do not misread as "market closed"
                return CheckOutcome(
                    name, VERDICT_SKIP,
                    f"sessions_active held only unrecognized names "
                    f"({', '.join(parsed[:3])}) — cannot evaluate",
                )
        else:
            raw_str = mc.get("session")
            if isinstance(raw_str, str) and raw_str.strip():
                parts = [
                    p.strip().lower().replace("-", "_").replace(" ", "_")
                    for p in re.split(r"[,/+&]", raw_str)
                    if p.strip()
                ]
                sessions = [p for p in parts if p in _KNOWN_SESSIONS]
                if not sessions:
                    # e.g. a trade-quality string — NOT session data. Skip,
                    # never fabricate a veto from unparseable input.
                    return CheckOutcome(
                        name, VERDICT_SKIP,
                        f"session key held unrecognized value ({raw_str[:40]!r}) "
                        "— not a session name; cannot evaluate",
                    )

        if sessions is _UNSET:
            if trade_context.get("da_allow_wallclock_session"):
                try:
                    from utils.session import SessionAnalyzer
                    ctx = SessionAnalyzer().get_current_session()
                    sessions = [str(s).lower() for s in ctx.get("active_sessions") or []]
                except Exception as exc:  # pragma: no cover
                    return CheckOutcome(name, VERDICT_SKIP,
                                        f"live session lookup failed: {exc}")
            else:
                return CheckOutcome(
                    name, VERDICT_SKIP,
                    "no session data supplied and wall-clock use not authorized",
                )

        if not sessions:
            return CheckOutcome(name, VERDICT_VETO, "market_closed",
                                {"sessions": sessions})

        majors = [s for s in sessions if s in _MAJOR_SESSIONS]
        if majors:
            return CheckOutcome(name, VERDICT_PASS,
                                f"major session active ({', '.join(majors)})",
                                {"sessions": sessions})

        asian_native = any(cur in symbol.upper() for cur in _ASIAN_NATIVE_CURRENCIES)
        if asian_native:
            return CheckOutcome(
                name, VERDICT_WARN,
                "Asian-only session; pair has native Tokyo liquidity — proceed cautiously",
                {"sessions": sessions},
            )
        return CheckOutcome(
            name, VERDICT_VETO,
            "asian_dead_chop: no London/NY session and pair has no native "
            "Tokyo liquidity — historical WR killer",
            {"sessions": sessions},
        )

    # ------------------------------------------------------------------
    # 2. H4 counter-trend filter
    # ------------------------------------------------------------------
    def _check_h4_trend(self, mc: Dict[str, Any], decision_out: Dict[str, Any], signal: str) -> CheckOutcome:
        name = "h4_trend_filter"
        mtf = (decision_out.get("mtf_trends") or {}) if isinstance(decision_out, dict) else {}
        h4 = None
        for src in (mtf, mc):
            if not isinstance(src, dict):
                continue
            for k in ("4h", "H4", "h4_trend", "h4"):
                v = src.get(k)
                if v not in (None, "", "unknown"):
                    h4 = v
                    break
            if h4 is not None:
                break
        if h4 in (None, "", "unknown"):
            return CheckOutcome(name, VERDICT_SKIP, "no H4 trend data available")
        if self._trend_opposes(h4, signal):
            return CheckOutcome(
                name, VERDICT_VETO,
                f"H4 trend '{h4}' actively opposes {signal} — counter-trend loss pattern",
                {"h4_trend": str(h4)},
            )
        return CheckOutcome(name, VERDICT_PASS, f"H4 trend '{h4}' does not oppose signal",
                            {"h4_trend": str(h4)})

    # ------------------------------------------------------------------
    # 3. Spread check
    # ------------------------------------------------------------------
    def _news_is_active(self, analysis_out: Dict[str, Any]) -> bool:
        news_ctx = analysis_out.get("news_ctx") if isinstance(analysis_out, dict) else None
        if not isinstance(news_ctx, dict):
            return False
        lvl = str(news_ctx.get("risk_level", "")).lower()
        return lvl in {"high", "critical", "severe"}

    def _check_spread(self, symbol: str, ind_ctx: Dict[str, Any],
                      analysis_out: Dict[str, Any], news_active: bool) -> CheckOutcome:
        name = "spread_check"
        spread_pips = self._f(ind_ctx.get("spread_pips")) if isinstance(ind_ctx, dict) else None
        if spread_pips is None or spread_pips <= 0:
            return CheckOutcome(name, VERDICT_SKIP, "no live spread data (spread_pips missing/zero)")

        # Instrument-aware limits via core.spread_policy (XAUUSD live ~260
        # must not be vetoed by legacy MAX_SPREAD_PIPS DEFAULT=3 / hard 5).
        try:
            from core.spread_policy import get_max_spread_pips, clean_symbol as _cs
            max_allowed = float(get_max_spread_pips(symbol))
            clean = _cs(symbol)
        except Exception:
            try:
                from broker.spread_monitor import MAX_SPREAD_PIPS
                clean = self._clean_symbol(symbol)
                max_allowed = float(MAX_SPREAD_PIPS.get(clean, MAX_SPREAD_PIPS.get("DEFAULT", 25.0)))
            except Exception:
                clean = self._clean_symbol(symbol)
                max_allowed = 25.0
        if news_active:
            max_allowed *= 0.5  # mirror SpreadMonitor.NEWS_WINDOW_MULTIPLIER

        atr = self._f(ind_ctx.get("atr")) if isinstance(ind_ctx, dict) else None
        details: Dict[str, Any] = {
            "spread_pips": spread_pips,
            "max_allowed_pips": round(max_allowed, 2),
            "news_window": news_active,
            "symbol_clean": clean if "clean" in dir() else self._clean_symbol(symbol),
        }

        # ★ FIX (2026-09-01): Two-tier spread/ATR ratio.
        # Old single threshold 0.35 caused hard VETO at 35.1% (noise-level
        # overshoot) — e.g. USDCHF "spread eats 35.1% of ATR (> 35%)".
        # Now: WARN at DA_MAX_SPREAD_RATIO (default 0.35), hard VETO only
        # above DA_MAX_SPREAD_RATIO_VETO (default 0.50). Absolute pip cap
        # (max_allowed) still hard-vetoes as before.
        ratio_warn = _env_float("DA_MAX_SPREAD_RATIO", 0.35)
        ratio_veto = _env_float("DA_MAX_SPREAD_RATIO_VETO", 0.50)
        if ratio_veto < ratio_warn:
            ratio_veto = ratio_warn
        spread_ratio: Optional[float] = None
        if atr and atr > 0:
            try:
                from core.constants import get_pip_size
                pip_size = get_pip_size(self._clean_symbol(symbol)) or 0.0001
                atr_pips = float(atr) / pip_size
                # Floor atr_pips so tiny M15 ATR during quiet NY doesn't
                # inflate the ratio and false-veto on normal spreads.
                atr_pips = max(atr_pips, _env_float("DA_MIN_ATR_PIPS", 4.0))
                if atr_pips > 0:
                    spread_ratio = spread_pips / atr_pips
                    details["spread_to_atr_ratio"] = round(spread_ratio, 4)
                    details["atr_pips_used"] = round(atr_pips, 2)
            except Exception:
                pass

        if spread_pips > max_allowed:
            return CheckOutcome(
                name, VERDICT_VETO,
                f"spread {spread_pips} pips > max {max_allowed:.2f}"
                + (" (news window)" if news_active else ""),
                details,
            )
        if spread_ratio is not None and spread_ratio > ratio_veto:
            return CheckOutcome(
                name, VERDICT_VETO,
                f"spread eats {round(spread_ratio * 100, 1)}% of ATR "
                f"(> {ratio_veto:.0%} hard cap) — edge destroyed by cost",
                details,
            )
        if spread_ratio is not None and spread_ratio > ratio_warn:
            return CheckOutcome(
                name, VERDICT_WARN,
                f"spread eats {round(spread_ratio * 100, 1)}% of ATR "
                f"(> {ratio_warn:.0%} warn) — elevated cost, not veto",
                details,
            )
        if spread_pips > max_allowed * 0.7:
            return CheckOutcome(
                name, VERDICT_WARN,
                f"spread {spread_pips} pips approaching limit {max_allowed:.2f}",
                details,
            )
        return CheckOutcome(name, VERDICT_PASS, f"spread OK ({spread_pips} pips)", details)

    # ------------------------------------------------------------------
    # 4. ATR regime check
    # ------------------------------------------------------------------
    def _check_atr_regime(self, ind_ctx: Dict[str, Any], regime: Dict[str, Any],
                          mc: Dict[str, Any]) -> CheckOutcome:
        name = "atr_regime_check"
        atr = self._f(ind_ctx.get("atr")) if isinstance(ind_ctx, dict) else None
        candle_range = self._f(ind_ctx.get("last_candle_range")) if isinstance(ind_ctx, dict) else None
        vol_strings = []
        for src in (regime if isinstance(regime, dict) else {},
                    regime.get("volatility") if isinstance(regime, dict) else None,
                    mc.get("volatility")):
            if isinstance(src, str) and src.strip():
                vol_strings.append(src)
            elif isinstance(src, dict):
                for k in ("volatility", "regime"):
                    v = src.get(k)
                    if isinstance(v, str) and v.strip():
                        vol_strings.append(v)
        vol_l = " ".join(vol_strings).lower()

        if atr is None or atr <= 0:
            return CheckOutcome(name, VERDICT_SKIP, "no ATR data available")

        details: Dict[str, Any] = {"atr": atr}
        if candle_range is not None and candle_range > 0:
            details["last_candle_range"] = candle_range
            spike_mult = _env_float("DA_NEWS_SPIKE_ATR_MULT", 2.5)
            if candle_range > spike_mult * atr:
                return CheckOutcome(
                    name, VERDICT_VETO,
                    f"news_spike_entry: last candle {candle_range} > {spike_mult}x ATR "
                    f"({round(candle_range / atr, 2)}x) — entering into a blowout",
                    details,
                )

        min_atr_pips_default = _env_float("DA_MIN_ATR_PIPS", 4.0)
        # atr here is a raw price delta; convert to pips when possible using
        # the pipeline-provided spread conversion heuristic is unreliable
        # without the symbol, so callers pass ind_ctx["atr_pips"] when they
        # can. Fall back to relative regime text.
        atr_pips = self._f(ind_ctx.get("atr_pips")) if isinstance(ind_ctx, dict) else None
        if atr_pips is None and candle_range:
            pass
        low_regime = bool(vol_l) and ("low" in vol_l or "sleep" in vol_l)
        if atr_pips is not None and atr_pips > 0:
            details["atr_pips"] = atr_pips
            if atr_pips < min_atr_pips_default:
                return CheckOutcome(
                    name, VERDICT_VETO,
                    f"atr_collapse: {atr_pips} pips < floor {min_atr_pips_default} — dead market",
                    details,
                )
            return CheckOutcome(name, VERDICT_PASS, f"ATR healthy ({atr_pips} pips)", details)
        if low_regime:
            return CheckOutcome(
                name, VERDICT_WARN,
                f"regime flagged low volatility ('{vol_l.strip()}') but ATR-floor "
                "unverifiable (no atr_pips)",
                details,
            )
        return CheckOutcome(name, VERDICT_PASS, "ATR present, no collapse/spike signals", details)

    # ------------------------------------------------------------------
    # 5. Structure-based SL sanity
    # ------------------------------------------------------------------
    def _check_structure_sl(self, signal: str, risk_out: Dict[str, Any],
                            mc: Dict[str, Any], sr_ctx: Dict[str, Any],
                            ind_ctx: Dict[str, Any]) -> CheckOutcome:
        name = "structure_sl_check"
        if signal not in {"BUY", "SELL"}:
            return CheckOutcome(name, VERDICT_SKIP, "not a directional trade")
        entry = self._f(risk_out.get("entry"))
        sl = self._f(risk_out.get("sl_price"))
        if entry is None or sl is None or entry <= 0 or sl <= 0:
            return CheckOutcome(name, VERDICT_SKIP, "entry/SL prices unavailable")

        sl_distance = abs(entry - sl)
        sl_distance_pips_hint = self._f(risk_out.get("sl_pips"))

        min_sl_pips = _env_float("DA_MIN_SL_PIPS", 6.0)
        buffer_pips = _env_float("DA_STRUCT_SL_BUFFER_PIPS", 2.0)
        proximity_mult = _env_float("DA_STRUCT_PROXIMITY_MULT", 2.0)

        protective = None
        side = ""
        if signal == "BUY":
            level = self._first_price(sr_ctx, mc, key="nearest_support")
            if level is not None and level < entry:
                protective = level
                side = "support"
        else:
            level = self._first_price(sr_ctx, mc, key="nearest_resistance")
            if level is not None and level > entry:
                protective = level
                side = "resistance"

        # need pip size to translate buffers — derive from SL pips hint if present
        pip_size: Optional[float] = None
        if sl_distance_pips_hint and sl_distance_pips_hint > 0:
            pip_size = sl_distance / sl_distance_pips_hint
        else:
            sym = str(mc.get("symbol") or "") or ""
            try:
                from core.constants import get_pip_size
                pip_size = get_pip_size(sym)
            except Exception:
                pip_size = None

        details: Dict[str, Any] = {
            "entry": entry, "sl": sl,
            "sl_distance_pips": round(sl_distance / pip_size, 1) if pip_size else None,
        }

        # 5a. absolute noise floor
        if pip_size and sl_distance_pips_hint and sl_distance_pips_hint < min_sl_pips:
            return CheckOutcome(
                name, VERDICT_VETO,
                f"sl_inside_noise: {sl_distance_pips_hint} pips < absolute floor {min_sl_pips} "
                "— pure-noise stop",
                details,
            )

        if protective is None:
            return CheckOutcome(name, VERDICT_SKIP, "no nearby protective structure known",
                                details)

        dist_to_struct = abs(entry - protective)
        if pip_size and sl_distance and round(dist_to_struct / sl_distance, 3) > proximity_mult:
            return CheckOutcome(
                name, VERDICT_SKIP,
                f"nearest {side} is {round(dist_to_struct / sl_distance, 2)}x SL distance away — "
                "higher-timeframe level irrelevant at this stop horizon",
                details,
            )

        buffer_px = (buffer_pips * pip_size) if pip_size else 0.0
        if signal == "BUY":
            inside_noise = sl > protective - buffer_px
            proper = protective - buffer_px
        else:
            inside_noise = sl < protective + buffer_px
            proper = protective + buffer_px

        details[f"nearest_{side}"] = protective
        details["structure_aware_sl"] = round(proper, 5)

        if inside_noise:
            return CheckOutcome(
                name, VERDICT_VETO,
                f"SL sits INSIDE structure noise: {side} {protective} with only "
                f"{buffer_pips} pip clearance required — price routinely wicks there; "
                "stop will be hunted before structure invalidates the thesis",
                details,
            )
        return CheckOutcome(
            name, VERDICT_PASS,
            f"SL respects {side} {protective} (+{buffer_pips} pip clearance)",
            details,
        )

    @staticmethod
    def _first_price(*sources: Dict[str, Any], key: str) -> Optional[float]:
        f = DASafetyNet._f
        for src in sources:
            if not isinstance(src, dict):
                continue
            v = f(src.get(key))
            if v is not None:
                return v
        return None

    # ------------------------------------------------------------------
    # 6. Lot sizing verification
    # ------------------------------------------------------------------
    def _check_lot_sizing(self, symbol: str, risk_out: Dict[str, Any]) -> CheckOutcome:
        name = "lot_sizing_check"
        balance = self._f(risk_out.get("balance"))
        risk_pct = self._f(risk_out.get("risk_pc"))
        if risk_pct is None:
            risk_pct = self._f(risk_out.get("risk_percent"))
        lot = self._f(risk_out.get("lot"))
        if lot is None:
            lot = self._f(risk_out.get("lot_size"))
        sl_pips = self._f(risk_out.get("sl_pips"))
        if None in (balance, risk_pct, lot, sl_pips) or balance <= 0 or sl_pips <= 0:
            return CheckOutcome(name, VERDICT_SKIP,
                                "balance/risk_pc/lot/sl_pips incomplete — cannot verify sizing")

        clean = self._clean_symbol(symbol)
        details: Dict[str, Any] = {"symbol_used": clean}

        unmapped = False
        try:
            from core.constants import PIP_VALUE_USD, get_pip_value_usd
            pip_val = get_pip_value_usd(clean)
            unmapped = clean not in PIP_VALUE_USD
            details["pip_value_per_lot"] = pip_val
        except Exception as exc:
            return CheckOutcome(name, VERDICT_SKIP, f"pip value lookup failed: {exc}")

        expected_lot = (balance * risk_pct / 100.0) / (sl_pips * pip_val)
        actual_risk_pct = (lot * sl_pips * pip_val) / balance * 100.0
        details["expected_lot"] = round(expected_lot, 4)
        details["actual_lot"] = round(lot, 4)
        details["actual_risk_pct"] = round(actual_risk_pct, 4)
        details["intended_risk_pct"] = risk_pct

        tol = _env_float("DA_LOT_MISMATCH_TOL", 0.25)
        rel_dev = abs(lot - expected_lot) / expected_lot if expected_lot > 0 else 0.0

        if rel_dev > tol:
            return CheckOutcome(
                name, VERDICT_VETO,
                f"lot mismatch: booked {round(lot, 2)} vs correct {round(expected_lot, 4)} "
                f"({round(rel_dev * 100, 0)}% off) — real risk ≠ intended risk",
                details,
            )
        if actual_risk_pct > risk_pct * 1.5:
            return CheckOutcome(
                name, VERDICT_VETO,
                f"risk overflow: actual {actual_risk_pct:.2f}% vs intended {risk_pct}% "
                "(min-lot or rounding pushed risk past tolerance)",
                details,
            )
        # un-mapped symbol fallback becomes its own warning even if numbers matched
        if unmapped:
            return CheckOutcome(name, VERDICT_WARN,
                                f"pip value for '{clean}' uses DEFAULT fallback table",
                                details)
        if lot < 0.01 and expected_lot < 0.01:
            return CheckOutcome(
                name, VERDICT_WARN,
                "intended position below broker minimum — broker will bump you to 0.01 "
                "which may exceed intended risk",
                details,
            )
        return CheckOutcome(
            name, VERDICT_PASS,
            f"sizing consistent (lot {round(lot, 2)}, real risk {actual_risk_pct:.2f}%)",
            details,
        )
