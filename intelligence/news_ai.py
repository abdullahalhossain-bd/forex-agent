"""
intelligence/news_ai.py — NewsIntelligence main orchestrator
==============================================================

The brain of the Day 66 News Intelligence Engine. Orchestrates:

  1. news_sources      → fetch news/events from 4 sources
  2. event_classifier  → classify each event (FOMC/NFP/CPI/CB_SPEECH/...)
  3. sentiment_model   → analyze tone (HAWKISH/DOVISH) per news item
  4. currency_impact   → map tone → per-pair directional bias
  5. Output: news_bias for the AnalysisAgent + Telegram alerts for HIGH events

Public API:
    NewsIntelligence().analyze(hours_ahead=24, pairs=["EURUSD", ...]) -> NewsBiasReport
    NewsIntelligence().should_block_trade(pair) -> {"blocked": bool, "reason": str}
    NewsIntelligence().adjust_confidence(pair, base_confidence) -> adjusted_confidence
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

from intelligence.news_sources import NewsSources, NewsItem
from intelligence.sentiment_model import SentimentModel, SentimentResult, get_sentiment_model
from intelligence.currency_impact import (
    CurrencyImpact, CurrencyImpactEngine, get_currency_impact_engine,
)
from intelligence.event_classifier import (
    EventClassification, EventClassifier, get_event_classifier,
)

log = get_logger("news_ai")

MEMORY_PATH = Path("memory/news_analysis_memory.jsonl")


@dataclass
class NewsBiasReport:
    """Top-level output of NewsIntelligence.analyze()."""
    generated_at: str
    next_high_impact_event: Optional[Dict[str, Any]] = None
    pair_biases: Dict[str, str] = field(default_factory=dict)         # pair → BULLISH/BEARISH/NEUTRAL
    pair_confidence_adjustments: Dict[str, float] = field(default_factory=dict)  # pair → +/- delta
    blocked_pairs: Dict[str, str] = field(default_factory=dict)       # pair → block reason
    sentiment_summary: str = ""
    total_events_analyzed: int = 0
    high_impact_count: int = 0
    sources_used: List[str] = field(default_factory=list)
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NewsIntelligence:
    """Main orchestrator — wires all 4 sub-modules together."""

    def __init__(self, pairs: Optional[List[str]] = None):
        self.sources = NewsSources()
        self.sentiment = get_sentiment_model()
        self.classifier = get_event_classifier()
        self.pairs = [p.upper() for p in (pairs or [])]
        self.impact_engine = get_currency_impact_engine(self.pairs)
        self._lock = threading.RLock()
        self._last_report: Optional[NewsBiasReport] = None
        self._last_analysis_at: Optional[datetime] = None
        # Cache for 5 minutes — avoid re-fetching news every cycle
        self._cache_ttl_sec = 300
        # Bug fix: format_telegram_alert() used to return a message for
        # whatever next_high_impact_event happened to be current on every
        # call, with no memory of what was already sent. trader.py's
        # auto-restart loop calls this once at the top of every run(), so
        # every crash/reconnect restart re-sent the SAME alert — and thanks
        # to the RSS "unknown pubDate → now()" bug in news_sources.py, a
        # stale headline could keep re-qualifying as brand new each time.
        # Track the last event key we actually alerted on (and when) so a
        # repeat call within the alert's own relevance window is suppressed.
        self._last_alerted_key: Optional[str] = None
        self._last_alerted_at: Optional[datetime] = None

    def set_pairs(self, pairs: List[str]) -> None:
        """Update the pair universe (called when AutonomousTraderSystem boots)."""
        with self._lock:
            self.pairs = [p.upper() for p in pairs]
            self.impact_engine = get_currency_impact_engine(self.pairs)

    # ── Main entry: full analysis ──────────────────────────────────

    def analyze(self, hours_ahead: int = 24) -> NewsBiasReport:
        """Run the full pipeline and return a NewsBiasReport.

        1. Fetch all news from 4 sources
        2. Classify each event
        3. Run sentiment analysis on each
        4. Calculate currency impact → pair bias
        5. Identify blocked pairs (in event window)
        6. Identify next high-impact event for dashboard
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            # Use cache if fresh
            if (self._last_report is not None and self._last_analysis_at is not None
                    and (now - self._last_analysis_at).total_seconds() < self._cache_ttl_sec):
                return self._last_report

        report = NewsBiasReport(
            generated_at=now.isoformat(timespec="seconds"),
            sources_used=[],
        )

        # Step 1: fetch all news
        try:
            all_news = self.sources.fetch_all_flat(hours_ahead=hours_ahead)
            report.total_events_analyzed = len(all_news)
            sources_seen = set()
            for item in all_news:
                sources_seen.add(item.source)
            report.sources_used = sorted(sources_seen)
        except Exception as e:
            log.error(f"[NewsAI] News fetch failed: {e}")
            return report

        if not all_news:
            log.info("[NewsAI] No news items found")
            with self._lock:
                self._last_report = report
                self._last_analysis_at = now
            return report

        # Step 2 + 3: classify + sentiment for each item
        currency_impacts: List[CurrencyImpact] = []
        next_high_impact: Optional[Dict[str, Any]] = None
        next_high_impact_dt: Optional[datetime] = None

        for item in all_news:
            # Step 2: classify
            classification = self.classifier.classify(item.event, item.impact)

            # Step 3: check block window
            # Day 101 hotfix: this used to pass `item.time_iso or now.isoformat()`
            # into is_in_block_window(). When an item had no time_iso (common for
            # headline-only sources that don't carry an exact release timestamp),
            # that silently substituted the CURRENT time as the event's time.
            # is_in_block_window() then computed delta = event_time - now = 0
            # minutes → "in block window, 0min" — i.e. every untimed high-impact
            # headline was treated as happening RIGHT NOW, regardless of when
            # (or whether) it actually occurs. This is exactly what produced the
            # "CPI in 0min" block while economic_calendar_api.py — which has a
            # real timestamp for the same CPI release from the FairEconomy feed —
            # correctly reported it as ~78 minutes away. Two pipelines, one with
            # real timing data and one silently faking "now" for missing data,
            # disagreed about the same event.
            #
            # Fix: an event with no known time is a "we don't know when" case,
            # not a "happening now" case — it must not be able to trigger the
            # block window at all. Skip the block check entirely when time_iso
            # is missing/empty instead of fabricating a timestamp.
            if item.time_iso:
                block_check = self.classifier.is_in_block_window(
                    classification, item.time_iso, now=now,
                )
            else:
                block_check = {
                    "in_block_window": False,
                    "block_reason": "event time unknown — cannot evaluate block window",
                }

            # If this is a high-impact event in the future, track the nearest one
            if classification.is_high_impact and item.time_iso:
                try:
                    ev_dt = datetime.fromisoformat(item.time_iso.replace("Z", "+00:00"))
                    if ev_dt.tzinfo is None:
                        ev_dt = ev_dt.replace(tzinfo=timezone.utc)
                    if ev_dt > now and (next_high_impact_dt is None or ev_dt < next_high_impact_dt):
                        next_high_impact_dt = ev_dt
                        next_high_impact = {
                            "event": item.event,
                            "currency": item.currency,
                            "category": classification.category,
                            "risk_level": classification.risk_level,
                            "time_iso": item.time_iso,
                            "minutes_until": int((ev_dt - now).total_seconds() / 60),
                            "action": classification.action,
                        }
                        report.high_impact_count += 1
                except Exception:
                    pass

            # Step 3: sentiment analysis (only for items with a headline/event text)
            text_to_analyze = item.headline or item.event
            try:
                sentiment = self.sentiment.analyze(text_to_analyze)
            except Exception as e:
                log.debug(f"[NewsAI] Sentiment failed for '{text_to_analyze[:50]}': {e}")
                continue

            # Step 4: currency impact (only if non-neutral tone + meaningful impact)
            if sentiment.tone != "NEUTRAL" and sentiment.impact_score >= 0.2:
                impact = self.impact_engine.calculate(
                    currency=sentiment.currency,
                    tone=sentiment.tone,
                    impact_score=sentiment.impact_score,
                    confidence=sentiment.confidence,
                    source_event=item.event,
                )
                currency_impacts.append(impact)

            # Block pair if this event is in its block window AND affects that pair's currency
            if block_check["in_block_window"] and classification.action == "BLOCK":
                affected_pairs = self._pairs_affected_by_currency(item.currency)
                for pair in affected_pairs:
                    if pair not in report.blocked_pairs:
                        report.blocked_pairs[pair] = block_check["block_reason"]

            # Append to details
            report.details.append({
                "source": item.source,
                "event": item.event,
                "currency": item.currency,
                "impact": item.impact,
                "time_iso": item.time_iso,
                "category": classification.category,
                "risk_level": classification.risk_level,
                "tone": sentiment.tone,
                "sentiment_confidence": sentiment.confidence,
                "impact_score": sentiment.impact_score,
                "in_block_window": block_check["in_block_window"],
                "block_reason": block_check["block_reason"],
            })

        # Step 5: merge all currency impacts → per-pair biases
        if currency_impacts:
            merged = self.impact_engine.merge_impacts(currency_impacts)
            report.pair_biases = merged
            # Confidence adjustment: +10 if bias aligns with BUY/SELL, -10 if opposes
            # (We don't know the technical signal here — the AnalysisAgent will use pair_biases
            # to adjust its own confidence.)
            for pair, bias in merged.items():
                if bias != "NEUTRAL":
                    report.pair_confidence_adjustments[pair] = 10.0  # placeholder; refined downstream

        # Step 6: build sentiment summary
        if currency_impacts:
            tones = [imp.tone for imp in currency_impacts]
            currencies = list({imp.currency for imp in currency_impacts})
            bullish_currencies = [imp.currency for imp in currency_impacts if imp.bias == "BULLISH"]
            bearish_currencies = [imp.currency for imp in currency_impacts if imp.bias == "BEARISH"]
            parts = []
            if bullish_currencies:
                parts.append(f"Bullish: {', '.join(bullish_currencies)}")
            if bearish_currencies:
                parts.append(f"Bearish: {', '.join(bearish_currencies)}")
            report.sentiment_summary = f"News sentiment — {' | '.join(parts)}" if parts else "Neutral news sentiment"

        report.next_high_impact_event = next_high_impact

        with self._lock:
            self._last_report = report
            self._last_analysis_at = now
        return report

    def _pairs_affected_by_currency(self, currency: str) -> List[str]:
        """Return all pairs in our universe containing this currency."""
        currency = currency.upper()
        if currency == "ALL":
            return list(self.pairs)
        return [p for p in self.pairs if currency in (p[:3], p[3:])]

    # ── Convenience: should this pair be blocked? ───────────────────

    def should_block_trade(self, pair: str) -> Dict[str, Any]:
        """Return {'blocked': bool, 'reason': str} for a given pair."""
        pair = pair.upper()
        report = self.analyze()
        if pair in report.blocked_pairs:
            return {"blocked": True, "reason": report.blocked_pairs[pair]}
        return {"blocked": False, "reason": ""}

    # ── Convenience: confidence adjustment for a pair ───────────────

    def adjust_confidence(
        self, pair: str, base_confidence: float, technical_signal: str = "",
    ) -> Dict[str, Any]:
        """Adjust confidence based on news bias alignment.

        If news bias ALIGNS with technical signal → +10 confidence
        If news bias OPPOSES technical signal → -15 confidence
        If neutral → no change

        Returns: {
            "adjusted_confidence": float,
            "original_confidence": float,
            "change": float,
            "reason": str,
            "news_bias": str,
        }
        """
        pair = pair.upper()
        report = self.analyze()
        news_bias = report.pair_biases.get(pair, "NEUTRAL")

        if news_bias == "NEUTRAL" or not technical_signal:
            return {
                "adjusted_confidence": base_confidence,
                "original_confidence": base_confidence,
                "change": 0.0,
                "reason": "no news bias",
                "news_bias": news_bias,
            }

        tech = technical_signal.upper()
        if tech in ("BUY", "BULLISH") and news_bias == "BULLISH":
            change = 10.0
            reason = f"News {news_bias} aligns with BUY"
        elif tech in ("SELL", "BEARISH") and news_bias == "BEARISH":
            change = 10.0
            reason = f"News {news_bias} aligns with SELL"
        elif tech in ("BUY", "BULLISH") and news_bias == "BEARISH":
            change = -15.0
            reason = f"News {news_bias} opposes BUY"
        elif tech in ("SELL", "BEARISH") and news_bias == "BULLISH":
            change = -15.0
            reason = f"News {news_bias} opposes SELL"
        else:
            change = 0.0
            reason = "neutral alignment"

        adjusted = max(0.0, min(100.0, base_confidence + change))
        return {
            "adjusted_confidence": adjusted,
            "original_confidence": base_confidence,
            "change": change,
            "reason": reason,
            "news_bias": news_bias,
        }

    # ── Telegram alert helper ──────────────────────────────────────

    def format_telegram_alert(self) -> Optional[str]:
        """Format a Telegram alert if there's a high-impact event in the next 60min."""
        report = self.analyze()
        nxt = report.next_high_impact_event
        if not nxt:
            return None
        minutes_until = nxt.get("minutes_until", 999)
        if minutes_until > 60 or minutes_until < -30:
            return None

        # De-dup: same (event, currency, time_iso) already alerted recently
        # → skip. This is what actually stops "AUTO-RESTART" loops (or any
        # repeat call) from re-blasting the identical alert every few minutes.
        key = f"{nxt.get('event')}|{nxt.get('currency')}|{nxt.get('time_iso')}"
        now = datetime.now(timezone.utc)
        with self._lock:
            if (self._last_alerted_key == key and self._last_alerted_at is not None
                    and (now - self._last_alerted_at).total_seconds() < self._cache_ttl_sec * 2):
                return None
            self._last_alerted_key = key
            self._last_alerted_at = now

        emoji = "⚠️" if minutes_until > 0 else "🔴"
        return (
            f"{emoji} HIGH IMPACT NEWS\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Event: {nxt.get('event', 'Unknown')}\n"
            f"Currency: {nxt.get('currency', 'ALL')}\n"
            f"Category: {nxt.get('category', 'OTHER')}\n"
            f"Risk: {nxt.get('risk_level', 'HIGH')}\n"
            f"Time: {nxt.get('minutes_until', 0)} min {'until' if minutes_until > 0 else 'ago'}\n"
            f"━━━━━━━━━━━━━━━━━━━━━"
        )

    # ── Learning memory (record past predictions for accuracy tracking) ──

    def record_prediction(
        self,
        event: str,
        currency: str,
        predicted_bias: str,
        predicted_confidence: float,
    ) -> None:
        """Record a news-impact prediction for later accuracy analysis."""
        try:
            MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "type": "prediction",
                "event": event,
                "currency": currency,
                "predicted_bias": predicted_bias,
                "predicted_confidence": predicted_confidence,
            }
            with MEMORY_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            log.debug(f"[NewsAI] prediction record failed: {e}")

    def record_outcome(
        self,
        event: str,
        currency: str,
        actual_direction: str,  # BULLISH / BEARISH / NEUTRAL (actual price move)
        predicted_bias: str,
    ) -> None:
        """Record the actual outcome of a news event (for accuracy tracking)."""
        try:
            MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            correct = (actual_direction.upper() == predicted_bias.upper())
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "type": "outcome",
                "event": event,
                "currency": currency,
                "actual_direction": actual_direction,
                "predicted_bias": predicted_bias,
                "correct": correct,
            }
            with MEMORY_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            log.debug(f"[NewsAI] outcome record failed: {e}")

    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Return prediction accuracy stats from memory."""
        if not MEMORY_PATH.exists():
            return {"total_predictions": 0, "correct": 0, "accuracy_pct": 0.0}
        try:
            predictions = {}
            outcomes = []
            for line in MEMORY_PATH.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "prediction":
                        predictions[entry["event"]] = entry
                    elif entry.get("type") == "outcome":
                        outcomes.append(entry)
                except json.JSONDecodeError:
                    continue
            total = len(outcomes)
            correct = sum(1 for o in outcomes if o.get("correct"))
            return {
                "total_predictions": len(predictions),
                "outcomes_recorded": total,
                "correct": correct,
                "accuracy_pct": round((correct / total * 100) if total else 0.0, 1),
            }
        except Exception as e:
            log.warning(f"[NewsAI] accuracy stats failed: {e}")
            return {"error": str(e)}

    # ── Status / latest report ─────────────────────────────────────

    def latest_report(self) -> Optional[NewsBiasReport]:
        """Return the most recent analysis (or run one if stale)."""
        with self._lock:
            if (self._last_report is None
                or self._last_analysis_at is None
                or (datetime.now(timezone.utc) - self._last_analysis_at).total_seconds() > self._cache_ttl_sec):
                pass
            else:
                return self._last_report
        return self.analyze()


# ── singleton ───────────────────────────────────────────────────────
_NEWS_AI: Optional[NewsIntelligence] = None


def get_news_intelligence(pairs: Optional[List[str]] = None) -> NewsIntelligence:
    global _NEWS_AI
    if _NEWS_AI is None:
        _NEWS_AI = NewsIntelligence(pairs=pairs)
    elif pairs and _NEWS_AI.pairs != [p.upper() for p in pairs]:
        _NEWS_AI.set_pairs(pairs)
    return _NEWS_AI