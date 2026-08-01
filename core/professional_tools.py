"""
core/professional_tools.py — Professional trading enhancements
==============================================================

Three production-grade utilities that make the agent behave like a
professional human trader:

1. **SessionAwarePairSelector** — picks the TOP-N pairs to analyze per cycle
   based on the current trading session. London/NY overlap gets USD/EUR/GBP
   pairs; Tokyo gets JPY/AUD/NZD pairs; etc. This makes the agent analyze
   the RIGHT pairs at the RIGHT time instead of blindly scanning all 28.

2. **DynamicPositionSizer** — adjusts lot size based on:
   * Confidence (higher conf → larger lot, capped by Kelly)
   * Recent win rate (winning streak → scale up, losing streak → scale down)
   * Volatility regime (high vol → smaller lot)
   * Session quality (overlap → normal, dead zone → blocked anyway)

3. **TradeJournal** — auto-logs every trade decision (entry, exit, win/loss,
   PnL, R:R, lessons learned) to CSV + JSON. Professional traders keep
   journals; this agent does too. Reused by the daily Telegram report.

All three are self-contained, thread-safe, and have no external deps.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("professional_tools")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DIR = PROJECT_ROOT / "memory" / "trade_journal"
JOURNAL_CSV = JOURNAL_DIR / "journal.csv"
JOURNAL_JSONL = JOURNAL_DIR / "journal.jsonl"


# ─────────────────────────────────────────────────────────────────────
# 1. SESSION-AWARE PAIR SELECTOR
# ─────────────────────────────────────────────────────────────────────

# Maps each session to the pairs that historically perform best in it.
# These are well-known institutional pair-session preferences.
# Updated for 30-pair universe (includes metals).
SESSION_PAIR_PRIORITY = {
    "TOKYO":         ["USDJPY", "AUDJPY", "NZDJPY", "AUDUSD", "NZDUSD", "CADJPY", "CHFJPY", "EURJPY", "GBPJPY", "AUDNZD", "XAUUSD"],
    "LONDON":        ["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "GBPCHF", "EURAUD", "GBPAUD", "EURCAD", "GBPCAD", "XAUUSD"],
    "NEW_YORK":      ["EURUSD", "GBPUSD", "USDCAD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD"],
    "LONDON_NY_OVERLAP": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "EURJPY", "GBPJPY", "AUDUSD", "USDCHF", "XAUUSD", "XAGUSD"],
    "SYDNEY":        ["AUDUSD", "NZDUSD", "AUDNZD", "AUDJPY", "NZDJPY", "EURAUD", "GBPAUD"],
    "BETWEEN_SESSIONS": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"],
    "DEAD_ZONE":     [],  # no trades
    "ASIAN":         ["USDJPY", "AUDJPY", "NZDJPY", "AUDUSD", "NZDUSD", "AUDNZD", "XAUUSD"],
}


class SessionAwarePairSelector:
    """Picks the most relevant pairs for the current trading session.

    Instead of analyzing all 28 pairs every cycle (slow + low-quality signals
    outside their prime session), this selector returns only the pairs that
    are actively traded in the current session — typically 8-12 pairs.
    This makes each cycle faster AND each analysis more focused.
    """

    def __init__(self, all_pairs: List[str]):
        self.all_pairs = [p.upper() for p in all_pairs]
        self._lock = threading.RLock()

    def select(self, session: str, top_n: int = 12) -> List[str]:
        """Return up to `top_n` pairs prioritized for this session.

        If the session is DEAD_ZONE, returns empty list (no trading).
        If fewer than top_n pairs are mapped, falls back to majors.
        """
        session = (session or "BETWEEN_SESSIONS").upper()
        if session == "DEAD_ZONE":
            return []

        priority = SESSION_PAIR_PRIORITY.get(session, SESSION_PAIR_PRIORITY["BETWEEN_SESSIONS"])

        # Keep only pairs in our actual trading universe
        selected = [p for p in priority if p in self.all_pairs]

        # If we have fewer than top_n, fill with majors not yet included
        majors = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
        for m in majors:
            if len(selected) >= top_n:
                break
            if m in self.all_pairs and m not in selected:
                selected.append(m)

        # Final fill: any remaining pair from the universe
        for p in self.all_pairs:
            if len(selected) >= top_n:
                break
            if p not in selected:
                selected.append(p)

        return selected[:top_n]

    def select_with_session(self, top_n: int = 12) -> tuple[List[str], str]:
        """Auto-detect current session and return (pairs, session_name)."""
        from utils.session import SessionAnalyzer
        sa = SessionAnalyzer()
        ctx = sa.get_current_session()
        active = ctx.get("active_sessions") or []
        overlap = ctx.get("overlap")

        if overlap:
            session = overlap
        elif "LONDON" in active and "NEW_YORK" in active:
            session = "LONDON_NY_OVERLAP"
        elif "LONDON" in active:
            session = "LONDON"
        elif "NEW_YORK" in active:
            session = "NEW_YORK"
        elif "TOKYO" in active or "ASIAN" in active:
            session = "TOKYO" if "TOKYO" in active else "ASIAN"
        elif "SYDNEY" in active:
            session = "SYDNEY"
        else:
            session = "BETWEEN_SESSIONS"

        pairs = self.select(session, top_n=top_n)
        return pairs, session


# ─────────────────────────────────────────────────────────────────────
# 2. DYNAMIC POSITION SIZER
# ─────────────────────────────────────────────────────────────────────


class DynamicPositionSizer:
    """Adjusts lot size based on confidence, recent performance, and volatility.

    ⚠️  STATUS (institutional review, see audit report): NOT CURRENTLY
        WIRED IN. get_position_sizer() in this file is not called from
        trader.py. The position sizer actually used in the live pipeline
        (AITrader._apply_advanced_sizing) is risk.position_sizer
        .PositionSizer — a separate, Kelly-criterion-based implementation
        registered via the service registry in runtime.py's boot_risk().
        Keep this class only if you want a lighter-weight sizing option;
        otherwise it's redundant with the active sizer and safe to remove.

    Formula:
        base_risk = RISK_PER_TRADE (1%) of balance
        confidence_mult = 0.5 + (confidence - 55) / 90   # 0.5x at 55%, 1.0x at ~78%, 1.5x at 100%
        streak_mult = 1.0 + (recent_win_rate - 50) / 200  # 0.75x at 0% WR, 1.25x at 100% WR
        vol_mult = clamp(1.0 / vol_factor, 0.5, 1.5)      # high vol → 0.5x, low vol → 1.5x

        final_risk = base_risk × confidence_mult × streak_mult × vol_mult
        final_risk = clamp(final_risk, 0.005, MAX_RISK_PER_PAIR)  # never below 0.5% or above 2%

    Returns a lot size that respects the final risk % given SL distance.
    """

    def __init__(self):
        self._lock = threading.RLock()

    def calculate(
        self,
        balance: float,
        confidence: float,                     # 0-100
        sl_pips: float,
        pip_value_usd: float,                  # per standard lot
        recent_win_rate: float = 50.0,         # 0-100
        volatility_factor: float = 1.0,        # 1.0 = normal, >1 = high vol, <1 = low vol
        risk_per_trade: float = 0.01,          # 1%
        max_risk_per_pair: float = 0.02,       # 2%
        min_risk: float = 0.005,               # 0.5%
    ) -> Dict[str, Any]:
        """Returns dict with: lot, risk_usd, risk_pct, multipliers."""
        if confidence < 55 or sl_pips <= 0 or pip_value_usd <= 0:
            return {
                "lot": 0.0, "risk_usd": 0.0, "risk_pct": 0.0,
                "confidence_mult": 0.0, "streak_mult": 0.0, "vol_mult": 0.0,
                "reason": "below min confidence or invalid SL",
            }

        # Confidence multiplier: 55% → 0.5x, 78% → 1.0x, 100% → 1.5x
        confidence_mult = 0.5 + (confidence - 55) / 90
        confidence_mult = max(0.3, min(1.5, confidence_mult))

        # Streak multiplier: 0% WR → 0.75x, 50% WR → 1.0x, 100% WR → 1.25x
        streak_mult = 1.0 + (recent_win_rate - 50) / 200
        streak_mult = max(0.5, min(1.25, streak_mult))

        # Volatility multiplier: high vol → smaller size
        vol_mult = 1.0 / max(0.5, volatility_factor)
        vol_mult = max(0.5, min(1.5, vol_mult))

        # Final risk %
        final_risk_pct = risk_per_trade * confidence_mult * streak_mult * vol_mult
        final_risk_pct = max(min_risk, min(max_risk_per_pair, final_risk_pct))

        # Convert to lot size
        risk_usd = balance * final_risk_pct
        lot = risk_usd / (sl_pips * pip_value_usd) if sl_pips > 0 else 0
        lot = max(0.01, min(round(lot, 2), 100.0))

        return {
            "lot": lot,
            "risk_usd": round(risk_usd, 2),
            "risk_pct": round(final_risk_pct * 100, 3),
            "confidence_mult": round(confidence_mult, 3),
            "streak_mult": round(streak_mult, 3),
            "vol_mult": round(vol_mult, 3),
            "reason": "ok",
        }


# ─────────────────────────────────────────────────────────────────────
# 3. TRADE JOURNAL
# ─────────────────────────────────────────────────────────────────────


@dataclass
class JournalEntry:
    """One trade decision in the journal."""
    timestamp: str
    cycle: int
    symbol: str
    timeframe: str
    session: str
    decision: str                    # BUY / SELL / WAIT / NO TRADE
    confidence: float
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    lot: float = 0.0
    rr_ratio: float = 0.0
    risk_usd: float = 0.0
    reason: str = ""
    llm_analysis: str = ""
    master_analysis: str = ""
    # Filled in after the trade closes:
    close_time: Optional[str] = None
    close_price: Optional[float] = None
    result: Optional[str] = None     # WIN / LOSS / BE / TIMEOUT
    pnl_usd: float = 0.0
    pnl_pips: float = 0.0
    lesson: str = ""

    def to_csv_row(self) -> List[str]:
        return [
            self.timestamp, str(self.cycle), self.symbol, self.timeframe,
            self.session, self.decision, str(self.confidence),
            str(self.entry or ""), str(self.sl or ""),
            str(self.tp1 or ""), str(self.tp2 or ""),
            str(self.lot), str(self.rr_ratio), str(self.risk_usd),
            self.reason[:200],
            self.close_time or "", str(self.close_price or ""),
            self.result or "", str(self.pnl_usd), str(self.pnl_pips),
            self.lesson[:200],
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CSV_HEADER = [
    "timestamp", "cycle", "symbol", "timeframe", "session",
    "decision", "confidence", "entry", "sl", "tp1", "tp2",
    "lot", "rr_ratio", "risk_usd", "reason",
    "close_time", "close_price", "result", "pnl_usd", "pnl_pips", "lesson",
]


class TradeJournal:
    """Append-only trade journal with CSV + JSONL persistence."""

    def __init__(self):
        self._lock = threading.RLock()
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        if not JOURNAL_CSV.exists():
            with JOURNAL_CSV.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)
        self._cycle = 0

    def next_cycle(self) -> int:
        with self._lock:
            self._cycle += 1
            return self._cycle

    def log_decision(self, entry: JournalEntry) -> None:
        """Append a trade decision (open or no-trade) to the journal."""
        with self._lock:
            try:
                with JOURNAL_CSV.open("a", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerow(entry.to_csv_row())
            except Exception as e:
                log.warning(f"[Journal] CSV write failed: {e}")
            try:
                with JOURNAL_JSONL.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.to_dict(), default=str) + "\n")
            except Exception as e:
                log.warning(f"[Journal] JSONL write failed: {e}")

    def log_close(
        self,
        symbol: str,
        cycle: int,
        close_price: float,
        result: str,
        pnl_usd: float,
        pnl_pips: float,
        lesson: str = "",
    ) -> None:
        """Record the close of an open trade."""
        with self._lock:
            try:
                with JOURNAL_JSONL.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type": "close",
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "symbol": symbol, "cycle": cycle,
                        "close_price": close_price, "result": result,
                        "pnl_usd": pnl_usd, "pnl_pips": pnl_pips,
                        "lesson": lesson,
                    }, default=str) + "\n")
            except Exception as e:
                log.warning(f"[Journal] Close write failed: {e}")

    def recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the last N decisions from the JSONL log."""
        if not JOURNAL_JSONL.exists():
            return []
        with self._lock:
            try:
                lines = JOURNAL_JSONL.read_text(encoding="utf-8").strip().split("\n")
                entries = []
                for line in lines[-limit:]:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                return entries
            except Exception as e:
                log.warning(f"[Journal] Read failed: {e}")
                return []

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics from the journal."""
        decisions = self.recent_decisions(limit=1000)
        total = len(decisions)
        trades = [d for d in decisions if d.get("decision") in ("BUY", "SELL")]
        waits = [d for d in decisions if d.get("decision") in ("WAIT", "NO TRADE")]
        wins = sum(1 for d in decisions if d.get("result") == "WIN")
        losses = sum(1 for d in decisions if d.get("result") == "LOSS")
        total_pnl = sum(float(d.get("pnl_usd", 0) or 0) for d in decisions if d.get("result"))
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) else 0.0
        return {
            "total_decisions": total,
            "trade_signals": len(trades),
            "wait_signals": len(waits),
            "closed_wins": wins,
            "closed_losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl_usd": round(total_pnl, 2),
        }


# ── singletons ──────────────────────────────────────────────────────

_PAIR_SELECTOR: Optional[SessionAwarePairSelector] = None
_POSITION_SIZER: Optional[DynamicPositionSizer] = None
_TRADE_JOURNAL: Optional[TradeJournal] = None


def get_pair_selector(all_pairs: Optional[List[str]] = None) -> SessionAwarePairSelector:
    global _PAIR_SELECTOR
    if _PAIR_SELECTOR is None or (all_pairs and _PAIR_SELECTOR.all_pairs != [p.upper() for p in all_pairs]):
        pairs = all_pairs or ["EURUSD", "GBPUSD", "USDJPY"]
        _PAIR_SELECTOR = SessionAwarePairSelector(pairs)
    return _PAIR_SELECTOR


def get_position_sizer() -> DynamicPositionSizer:
    global _POSITION_SIZER
    if _POSITION_SIZER is None:
        _POSITION_SIZER = DynamicPositionSizer()
    return _POSITION_SIZER


def get_trade_journal() -> TradeJournal:
    global _TRADE_JOURNAL
    if _TRADE_JOURNAL is None:
        _TRADE_JOURNAL = TradeJournal()
    return _TRADE_JOURNAL