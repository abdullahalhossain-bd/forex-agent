"""
intelligence/sentiment_model.py — Financial news sentiment analyzer
====================================================================

A FinBERT-style sentiment analyzer that understands forex-specific
language. Outputs: sentiment, tone (HAWKISH/DOVISH), currency, impact_score, etc.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
log = get_logger("sentiment_model")


# ── Configuration ─────────────────────────────────────────────────────
LLM_AVAILABLE = False
_groq_client = None
_gemini_client = None
_key_manager = None

# Separate models for each provider (Fixed)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Throttling
_FALLBACK_THROTTLE_LOCK = threading.Lock()
_fallback_last_call_ts: float = 0.0
_fallback_call_count_this_cycle: int = 0
FALLBACK_LLM_CALL_INTERVAL_SEC = float(os.getenv("SENTIMENT_FALLBACK_LLM_INTERVAL_SEC", "2.0"))
FALLBACK_MAX_LLM_CALLS_PER_CYCLE = int(os.getenv("SENTIMENT_FALLBACK_MAX_CALLS_PER_CYCLE", "20"))

_SENTIMENT_CYCLE_LOCK = threading.Lock()
_sentiment_call_count_this_cycle: int = 0
SENTIMENT_MAX_LLM_CALLS_PER_CYCLE = int(os.getenv("SENTIMENT_MAX_LLM_CALLS_PER_CYCLE", "5"))

# Bug fix: with 60+ pairs/news-items per cycle and a 5-call cap, every item
# past the 5th independently hit the cap check and emitted its own
# "LLM skipped — sentiment cap reached" INFO log line — 30+ duplicate lines
# per cycle, plus a lock acquisition for every single item even after the
# cap was already known to be exhausted. Track whether we've already logged
# the cap-reached event this cycle so repeats stay silent (still returns
# False so callers correctly fall back to rule-based analysis), and expose
# a cheap lock-free peek so callers can skip straight to the rule-based
# path instead of calling into the LLM path at all once the budget is gone.
_sentiment_cap_logged_this_cycle: bool = False


def sentiment_cap_reached() -> bool:
    """Cheap, lock-free peek so callers can skip straight to the rule-based
    path (no log spam, no redundant lock acquisition) once the per-cycle
    LLM budget is exhausted."""
    return _sentiment_call_count_this_cycle >= SENTIMENT_MAX_LLM_CALLS_PER_CYCLE


def _check_sentiment_cycle_cap() -> tuple[bool, str]:
    global _sentiment_call_count_this_cycle, _sentiment_cap_logged_this_cycle
    with _SENTIMENT_CYCLE_LOCK:
        if _sentiment_call_count_this_cycle >= SENTIMENT_MAX_LLM_CALLS_PER_CYCLE:
            first_time = not _sentiment_cap_logged_this_cycle
            _sentiment_cap_logged_this_cycle = True
            reason = f"sentiment cap: {SENTIMENT_MAX_LLM_CALLS_PER_CYCLE} calls/cycle reached"
            if not first_time:
                reason += "|_repeat_"  # tells analyze() not to log this one
            return False, reason
        _sentiment_call_count_this_cycle += 1
        return True, "ok"


def _check_fallback_throttle() -> tuple[bool, str]:
    import time as _time
    global _fallback_last_call_ts, _fallback_call_count_this_cycle
    with _FALLBACK_THROTTLE_LOCK:
        now = _time.monotonic()
        if _fallback_call_count_this_cycle >= FALLBACK_MAX_LLM_CALLS_PER_CYCLE:
            return False, f"fallback throttle: {FALLBACK_MAX_LLM_CALLS_PER_CYCLE} calls/cycle reached"
        elapsed = now - _fallback_last_call_ts
        if elapsed < FALLBACK_LLM_CALL_INTERVAL_SEC:
            return False, f"fallback throttle: {elapsed:.2f}s < {FALLBACK_LLM_CALL_INTERVAL_SEC}s min interval"
        _fallback_last_call_ts = now
        _fallback_call_count_this_cycle += 1
        return True, "ok"


# ── Client Initialization ─────────────────────────────────────────────
try:
    from core.llm_key_manager import get_llm_key_manager
    _key_manager = get_llm_key_manager()

    _groq_client = _key_manager.get_groq_client()
    if _groq_client is not None:
        LLM_AVAILABLE = True
        log.info(f"[SentimentModel] Groq client initialized | model={GROQ_MODEL}")

    if not LLM_AVAILABLE:
        _gemini_client = _key_manager.get_gemini_client()
        if _gemini_client is not None:
            LLM_AVAILABLE = True
            log.info(f"[SentimentModel] Gemini client initialized (key manager) | model={GEMINI_MODEL}")

except Exception as e:
    log.warning(f"[SentimentModel] LLMKeyManager init failed: {e} — trying single-key")

    # Groq single key
    groq_key = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=groq_key)
            LLM_AVAILABLE = True
            log.info(f"[SentimentModel] Groq single-key initialized | model={GROQ_MODEL}")
        except Exception as e2:
            log.warning(f"[SentimentModel] Groq init failed: {e2}")

    # Gemini single key
    if not LLM_AVAILABLE:
        gemini_key = os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                from google import genai as google_genai
                _gemini_client = google_genai.Client(api_key=gemini_key)
                LLM_AVAILABLE = True
                log.info(f"[SentimentModel] Gemini single-key initialized | model={GEMINI_MODEL}")
            except Exception as e2:
                log.warning(f"[SentimentModel] Gemini init failed: {e2}")

if not LLM_AVAILABLE:
    log.warning("[SentimentModel] No LLM available — using rule-based fallback")


# ── Data Class ────────────────────────────────────────────────────────
@dataclass
class SentimentResult:
    sentiment: str           # positive / negative / neutral
    tone: str                # HAWKISH / DOVISH / NEUTRAL
    currency: str            # USD / EUR / GBP / JPY / ALL
    impact_score: float      # 0.0 - 1.0
    keywords: List[str]
    summary: str
    confidence: float        # 0-100
    source: str              # llm / rule_based

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Rule-based Keywords ───────────────────────────────────────────────
HAWKISH_KEYWORDS = [
    "higher rates", "rate hike", "rate increase", "tightening", "fight inflation",
    "inflation concern", "strong economy", "robust growth", "hot economy",
    "aggressive", "hawkish", "higher for longer", "inflation fight",
    "monetary tightening", "reduce balance sheet", "quantitative tightening",
    "rate normalization", "overheating", "wage growth strong",
]

DOVISH_KEYWORDS = [
    "rate cut", "rate reduction", "cut rates", "lower rates", "easing",
    "dovish", "accommodative", "support economy", "economic support",
    "quantitative easing", "qe", "stimulus", "soft landing",
    "inflation cooling", "inflation slowing", "weak economy",
    "recession risk", "employment concern", "growth concern",
    "pause rate hikes", "pause hikes", "hold rates steady",
]

CURRENCY_KEYWORDS = {
    "USD": ["fed ", "fomc", "federal reserve", "powell", "dollar", "usd", "us economy", "us inflation", "us rates"],
    "EUR": ["ecb", "lagarde", "eurozone", "euro area", "euro ", "eur ", "eu inflation", "eu rates"],
    "GBP": ["boe", "bailey", "bank of england", "uk economy", "pound", "gbp", "uk inflation", "brexit"],
    "JPY": ["boj", "ueda", "bank of japan", "japan economy", "yen", "jpy", "japan inflation"],
}


# ── LLM Prompt ────────────────────────────────────────────────────────
_SENTIMENT_PROMPT = """You are a financial sentiment analyzer specialized in forex news.

Analyze the following news headline/snippet and return ONLY valid JSON (no markdown, no extra text).

JSON schema:
{
  "sentiment": "positive" | "negative" | "neutral",
  "tone": "HAWKISH" | "DOVISH" | "NEUTRAL",
  "currency": "USD" | "EUR" | "GBP" | "JPY" | "ALL",
  "impact_score": 0.0-1.0,
  "keywords": ["most important 2-4 financial terms"],
  "summary": "1-sentence forex-impact summary",
  "confidence": 0-100
}

Rules:
- HAWKISH = higher rates / tightening / inflation fighting → bullish for that currency
- DOVISH  = rate cuts / easing / economic support → bearish for that currency
- If the news is about Fed/USD, currency = "USD"; ECB/EUR → "EUR" etc.
- impact_score: 1.0 = extreme (FOMC), 0.7 = high (CPI/NFP), 0.4 = medium, 0.2 = low

News text:
"""


class SentimentModel:
    def __init__(self):
        self._lock = threading.RLock()

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(
                sentiment="neutral", tone="NEUTRAL", currency="ALL",
                impact_score=0.0, keywords=[], summary="empty input",
                confidence=0.0, source="rule_based",
            )

        # Bug fix: once the per-cycle LLM budget is exhausted, skip the LLM
        # path entirely instead of calling into it (and its "skipped" log)
        # for every remaining item this cycle.
        if LLM_AVAILABLE and not sentiment_cap_reached():
            try:
                result = self._analyze_with_llm(text)
                if result is not None:
                    return result
            except Exception as e:
                log.warning(f"[SentimentModel] LLM analysis failed: {e} — using rule-based fallback")

        return self._analyze_with_rules(text)

    def _analyze_with_llm(self, text: str) -> Optional[SentimentResult]:
        # Cycle protection
        allowed, reason = _check_sentiment_cycle_cap()
        if not allowed:
            if reason.endswith("|_repeat_"):
                log.debug(f"[SentimentModel] LLM skipped — {reason[:-len('|_repeat_')]}")
            else:
                log.info(f"[SentimentModel] LLM skipped — {reason}")
            return None

        # Throttle
        if _key_manager is not None:
            try:
                allowed, reason = _key_manager.check_cycle_throttle()
                if not allowed:
                    log.info(f"[SentimentModel] LLM skipped — {reason}")
                    return None
            except Exception:
                pass
        else:
            allowed, reason = _check_fallback_throttle()
            if not allowed:
                log.info(f"[SentimentModel] LLM skipped — {reason}")
                return None

        prompt = _SENTIMENT_PROMPT + f'"""\n{text[:1500]}\n"""'
        raw = None

        # Groq
        groq_client = _key_manager.get_groq_client() if _key_manager is not None else _groq_client
        if groq_client is not None:
            try:
                resp = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=400,
                )
                raw = resp.choices[0].message.content

                if _key_manager is not None:
                    tokens_used = getattr(getattr(resp, 'usage', None), 'total_tokens', 0)
                    _key_manager.mark_groq_success(tokens_used=tokens_used, client=groq_client)
                log.debug(f"[SentimentModel] Groq success | {GROQ_MODEL}")
            except Exception as e:
                if _key_manager is not None:
                    is_rate_limited = "429" in str(e) or "rate limit" in str(e).lower()
                    _key_manager.mark_groq_failure(str(e), is_rate_limited, groq_client)
                log.debug(f"[SentimentModel] Groq failed: {e}")

        # Gemini Fallback
        if raw is None:
            gemini_client = _key_manager.get_gemini_client() if _key_manager is not None else _gemini_client
            if gemini_client is not None:
                try:
                    resp = gemini_client.models.generate_content(
                        model=GEMINI_MODEL,      # ← Fixed here
                        contents=prompt
                    )
                    raw = resp.text

                    if _key_manager is not None:
                        tokens_used = getattr(getattr(resp, 'usage_metadata', None), 'total_token_count', 0)
                        _key_manager.mark_gemini_success(tokens_used=tokens_used, client=gemini_client)
                    log.debug(f"[SentimentModel] Gemini fallback success | {GEMINI_MODEL}")
                except Exception as e:
                    if _key_manager is not None:
                        is_rate_limited = "429" in str(e) or "rate limit" in str(e).lower()
                        _key_manager.mark_gemini_failure(str(e), is_rate_limited, gemini_client)
                    log.warning(f"[SentimentModel] Gemini failed: {e}")

        if raw is None:
            return None

        # Parse JSON
        try:
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw).strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group() if match else raw)

            return SentimentResult(
                sentiment=str(data.get("sentiment", "neutral")).lower(),
                tone=str(data.get("tone", "NEUTRAL")).upper(),
                currency=str(data.get("currency", "ALL")).upper(),
                impact_score=float(data.get("impact_score", 0.0)),
                keywords=list(data.get("keywords", []))[:5],
                summary=str(data.get("summary", ""))[:300],
                confidence=float(data.get("confidence", 50)),
                source="llm",
            )
        except Exception as e:
            log.warning(f"[SentimentModel] JSON parse failed: {e}")
            return None

    def _analyze_with_rules(self, text: str) -> SentimentResult:
        """Rule-based fallback"""
        text_lower = text.lower()

        hawkish_hits = sum(1 for kw in HAWKISH_KEYWORDS if kw in text_lower)
        dovish_hits = sum(1 for kw in DOVISH_KEYWORDS if kw in text_lower)

        tone = "HAWKISH" if hawkish_hits > dovish_hits else "DOVISH" if dovish_hits > hawkish_hits else "NEUTRAL"

        # Currency detection
        currency = "ALL"
        currency_hits = {}
        for cur, keywords in CURRENCY_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > 0:
                currency_hits[cur] = hits
        if currency_hits:
            currency = max(currency_hits, key=currency_hits.get)

        sentiment = "positive" if tone == "HAWKISH" else "negative" if tone == "DOVISH" else "neutral"

        total_hits = hawkish_hits + dovish_hits
        impact_score = min(1.0, total_hits * 0.2)

        keywords = []
        for kw in HAWKISH_KEYWORDS + DOVISH_KEYWORDS:
            if kw in text_lower and kw not in keywords:
                keywords.append(kw)
            if len(keywords) >= 4:
                break

        return SentimentResult(
            sentiment=sentiment,
            tone=tone,
            currency=currency,
            impact_score=impact_score,
            keywords=keywords,
            summary=f"Rule-based: {tone} signal for {currency}" if tone != "NEUTRAL" else "No strong signal",
            confidence=min(80.0, 40.0 + total_hits * 10),
            source="rule_based",
        )


# ── Singleton & Reset ─────────────────────────────────────────────────
_SENTIMENT_MODEL: Optional[SentimentModel] = None


def reset_fallback_throttle_cycle() -> None:
    global _fallback_call_count_this_cycle, _sentiment_call_count_this_cycle
    global _sentiment_cap_logged_this_cycle
    with _FALLBACK_THROTTLE_LOCK:
        _fallback_call_count_this_cycle = 0
    with _SENTIMENT_CYCLE_LOCK:
        _sentiment_call_count_this_cycle = 0
        _sentiment_cap_logged_this_cycle = False


def get_sentiment_model() -> SentimentModel:
    global _SENTIMENT_MODEL
    if _SENTIMENT_MODEL is None:
        _SENTIMENT_MODEL = SentimentModel()
    return _SENTIMENT_MODEL