# alerts/telegram_bot.py
# ============================================================
# Telegram Alert & Command System — Full Upgrade (Fixed)
# ============================================================
# FIXES APPLIED:
#   1. send_message() — correct fallback order:
#      Markdown first → plain text fallback (not reversed).
#   2. Command handlers (cmd_status, cmd_calendar, cmd_daily, etc.)
#      now route through a shared _reply() helper that also falls back
#      to plain text, so DB/dynamic content can never break a reply.
#   3. cmd_daily — no longer creates a new TelegramNotifier() on every
#      call; reuses a module-level shared instance instead.
#   4. IS_TRADING_PAUSED protected by asyncio.Lock to prevent race
#      conditions in concurrent async environments.
#   5. notify_weekly_calendar / cmd_calendar — long messages are
#      automatically chunked into ≤4096-char pieces so Telegram never
#      silently drops an oversized message.
# ============================================================

import os
import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Optional

from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
from database.db import TraderDB
from utils.logger import get_logger

log = get_logger("telegram_bot")

# ── Global trading-pause state + callback mechanism ───────────

IS_TRADING_PAUSED: bool = False
_pause_lock = asyncio.Lock()
_on_pause_changed: Optional[Callable[[bool], None]] = None

TELEGRAM_MSG_LIMIT = 4096  # Telegram hard limit per message

# ── CRITICAL FIX: Telegram security — command authorization ─────
# Without this, ANYONE who knows the bot username can send /pause, /resume, /close
# and control your trading system. This is a severe security risk.
#
# Set ALLOWED_USER_IDS in .env as comma-separated Telegram user IDs:
#   ALLOWED_USER_IDS=123456789,987654321
# To find your user ID: message @userinfobot on Telegram.
# If not set, ALL users are rejected (fail-closed).
_ALLOWED_USER_IDS: set[int] = set()
_ALLOWED_CHAT_IDS: set[int] = set()

def _load_allowed_ids():
    """Load allowed user/chat IDs from environment."""
    global _ALLOWED_USER_IDS, _ALLOWED_CHAT_IDS
    users_str = os.getenv("ALLOWED_USER_IDS", "")
    chats_str = os.getenv("ALLOWED_CHAT_IDS", "")
    if users_str:
        _ALLOWED_USER_IDS = {int(x.strip()) for x in users_str.split(",") if x.strip()}
    if chats_str:
        _ALLOWED_CHAT_IDS = {int(x.strip()) for x in chats_str.split(",") if x.strip()}

_load_allowed_ids()

def _is_authorized(update) -> bool:
    """Check if the message sender is authorized to run commands.

    Fail-closed: if no IDs are configured, ALL users are rejected.
    This is safer than fail-open — you must explicitly configure access.
    """
    if not _ALLOWED_USER_IDS and not _ALLOWED_CHAT_IDS:
        # No IDs configured — reject everyone (fail-closed)
        log.warning(f"[Telegram] Command rejected — no ALLOWED_USER_IDS configured. "
                    f"Set in .env to enable commands. User: {update.effective_user}")
        return False
    user_id = update.effective_user.id if update.effective_user else 0
    chat_id = update.effective_chat.id if update.effective_chat else 0
    if user_id in _ALLOWED_USER_IDS:
        return True
    if chat_id in _ALLOWED_CHAT_IDS:
        return True
    log.warning(f"[Telegram] Unauthorized command from user_id={user_id}, chat_id={chat_id}")
    return False

async def _unauthorized_reply(update):
    """Reply to unauthorized users."""
    await _reply(update, "⛔ Unauthorized. This bot is private.")

# ── Day 81+ hotfix: per-channel rate limiter ──────────────────
# Telegram floods when the bot sends dozens of messages per minute
# (trade-open alerts, news alerts, confluence alerts, restart alerts).
# This sliding-window limiter drops messages above TELEGRAM_MAX_MSG_PER_MIN
# (default 10) so the bot doesn't get muted by users or rate-limited by Telegram.

class _RateLimiter:
    """Sliding-window per-channel rate limiter."""
    def __init__(self, max_per_min: int = 10):
        self.max_per_min = max_per_min
        self._timestamps: deque = deque()  # monotonic timestamps of sent msgs
        self._dropped_count = 0

    def allow(self) -> bool:
        now = time.monotonic()
        # Evict timestamps older than 60 seconds
        cutoff = now - 60.0
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_min:
            self._dropped_count += 1
            if self._dropped_count % 10 == 1:  # log every 10th drop
                log.warning(
                    f"[Telegram] rate limit: dropped {self._dropped_count} messages "
                    f"({len(self._timestamps)}/{self.max_per_min} in last 60s)"
                )
            return False
        self._timestamps.append(now)
        return True

# Singleton rate limiter — loaded from config on first use
_RATE_LIMITER: Optional[_RateLimiter] = None

def _get_rate_limiter() -> _RateLimiter:
    global _RATE_LIMITER
    if _RATE_LIMITER is None:
        try:
            from config import TELEGRAM_MAX_MSG_PER_MIN
            limit = TELEGRAM_MAX_MSG_PER_MIN
        except Exception:
            limit = 10
        _RATE_LIMITER = _RateLimiter(max_per_min=limit)
    return _RATE_LIMITER


def register_pause_callback(callback: Callable[[bool], None]) -> None:
    """
    Register a callback that fires the moment IS_TRADING_PAUSED changes.

        from alerts.telegram_bot import register_pause_callback
        register_pause_callback(my_engine.on_pause_changed)

    The callback receives the *new* value of IS_TRADING_PAUSED.
    """
    global _on_pause_changed
    _on_pause_changed = callback
    log.info("📞 Pause-state callback registered")


async def _set_trading_paused(value: bool) -> None:
    """Internal async helper — updates flag AND invokes callback."""
    global IS_TRADING_PAUSED
    async with _pause_lock:
        IS_TRADING_PAUSED = value
    if _on_pause_changed is not None:
        try:
            _on_pause_changed(value)
        except Exception as exc:
            log.error(f"❌ Pause callback raised: {exc}")


def _escape_markdown(text) -> str:
    """
    Strip characters that break Telegram's legacy Markdown (V1) entity
    parser when they appear in dynamic/unsanitized strings.

    A single unmatched '*', '_', '`', or '[' causes the ENTIRE send to
    fail with "Can't parse entities". Removing them from dynamic content
    before interpolation is the simplest robust fix.
    """
    if text is None:
        return "—"
    if not isinstance(text, str):
        text = str(text)
    for ch in ("*", "_", "`", "["):
        text = text.replace(ch, "")
    return text


def _scrub_token_from_error(error_str: str) -> str:
    """Round-12 audit fix: remove Telegram bot tokens from exception messages.

    python-telegram-bot v20+ usually sanitizes URLs in exceptions, but
    urllib3/httpx low-level errors can include the full request URL:
        https://api.telegram.org/bot<TOKEN>/sendMessage

    This function strips any token-like substring from the error message
    before it reaches the logger. Safe to call on any string — if no
    token pattern is found, the original string is returned unchanged.
    """
    import re as _re
    # Match: bot<digits>:<35-40 alnum> (the token portion in a URL)
    # OR: <digits>:<35-40 alnum> (bare token)
    scrubbed = _re.sub(
        r"(bot\d{8,12}:[A-Za-z0-9_-]{30,45}|\b\d{8,12}:[A-Za-z0-9_-]{30,45}\b)",
        "<REDACTED:TELEGRAM_TOKEN>",
        error_str,
    )
    return scrubbed


def _sanitize_for_markdown(text: str) -> str:
    """
    Defensive pre-check run right before a Markdown send attempt.

    Legacy Telegram Markdown (V1) rejects the WHOLE message if any of
    '*', '_', '`' appear an odd number of times, or if a '[' appears
    without a matching '](url)' link pattern. Upstream callers (e.g.
    reject_reason strings like "Circuit breaker [COOLDOWN]: ...") often
    pass raw dynamic text straight into send_message() without calling
    _escape_markdown() themselves. Rather than relying on every call site
    to remember to escape, neutralize unbalanced entities here so a
    single stray character never nukes the whole message.
    """
    import re
    # Strip '[' / ']' unless they form a real [text](url) link.
    if not re.search(r"\[[^\]\n]*\]\([^)\n]*\)", text):
        text = text.replace("[", "").replace("]", "")
    # If any formatting character appears an odd number of times, it's
    # unbalanced — drop all instances of that character rather than
    # letting the send fail outright.
    for ch in ("*", "_", "`"):
        if text.count(ch) % 2 != 0:
            text = text.replace(ch, "")
    return text


def _chunk_message(text: str, limit: int = TELEGRAM_MSG_LIMIT) -> list[str]:
    """
    Split a long message into chunks of at most `limit` characters,
    splitting on newlines where possible to avoid cutting mid-line.
    """
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


# ══════════════════════════════════════════════════════════════
#  TelegramNotifier — outbound notification templates
# ══════════════════════════════════════════════════════════════

class TelegramNotifier:
    """Handles every outgoing notification for the trading bot."""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not self.token or not self.chat_id:
            log.critical("[TelegramNotifier] TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set — all notifications disabled!")
            self.bot = None
        else:
            self.bot = Bot(token=self.token)

    # ── core sender ─────────────────────────────────────────────

    async def send_message(self, text: str, priority: bool = False):
        """
        Send a Markdown-formatted message. If Telegram rejects the
        Markdown (e.g. unmatched entity), falls back to plain text so
        alerts are never silently dropped.

        Long messages are chunked automatically to stay within Telegram's
        4096-character limit.

        Day 81+ hotfix: per-channel rate limiter drops messages above
        TELEGRAM_MAX_MSG_PER_MIN (default 10) to prevent Telegram floods.

        Day 102+ CRITICAL hotfix: PRIORITY BYPASS for risk-critical alerts.
        Previously, all messages went through the same rate limiter — so
        after 10 status messages in a minute, the 11th (which might be
        "DAILY LOSS LIMIT REACHED" or "CIRCUIT BREAKER TRIGGERED") was
        silently dropped. Now callers can pass `priority=True` to bypass
        the rate limiter. Use sparingly — only for alerts where dropping
        the message would cause real financial harm (loss limit, drawdown,
        circuit breaker, kill switch).
        """
        if not self.bot:
            return

        # Day 81+ rate limit check — bypassed for priority messages
        if not priority and not _get_rate_limiter().allow():
            return  # silently drop — already logged in _RateLimiter.allow()

        for chunk in _chunk_message(text):
            # FIX #1: Try Markdown first, fall back to plain text.
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=_sanitize_for_markdown(chunk),
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception as e:
                # Round-12 audit fix: scrub potential token/URL from
                # exception message before logging. python-telegram-bot
                # v20+ generally sanitizes URLs, but urllib3/httpx
                # low-level errors can include the full request URL
                # (which contains the bot token). The operator's audit
                # found tokens leaking via this exact path.
                _scrubbed = _scrub_token_from_error(str(e))
                log.warning(f"⚠️ Markdown send failed ({_scrubbed}), retrying as plain text…")
                try:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=chunk,
                    )
                except Exception as e2:
                    _scrubbed2 = _scrub_token_from_error(str(e2))
                    log.error(f"❌ Failed to send Telegram alert (all attempts): {_scrubbed2}")

    # ── 1. TRADE OPENED ────────────────────────────────────────

    async def notify_trade_open(
        self,
        trade_data: dict,
        confidence: int,
        reasons: list,
        confidence_breakdown_lines: list = None,
    ):
        """
        trade_data keys: pair, signal, entry, sl, tp, lot
        confidence: 0-100
        reasons: list of AI reasoning strings (top 3 shown)
        confidence_breakdown_lines: itemized scorecard lines, e.g.
            ["Rule: +18", "Trend: +15", "Momentum: +10", "Sentiment: +8",
             "LLM: -12", "Liquidity: -6", "Resistance: -5", "Total = 68%"]
            Produced by core/confidence_breakdown.py — Liquidity and
            Resistance are always included here, even when those
            EntrySafetyFilters checks passed, so the risk they carried
            stays visible instead of disappearing once filtered.
        """
        pair   = _escape_markdown(trade_data.get("pair", "—"))
        signal = _escape_markdown(trade_data.get("signal", "—"))
        entry  = trade_data.get("entry", "—")
        sl     = trade_data.get("sl", "—")
        tp     = trade_data.get("tp", "—")
        lot    = trade_data.get("lot", "—")

        if confidence >= 80:
            conf_icon = "🟢"
        elif confidence >= 60:
            conf_icon = "🟡"
        else:
            conf_icon = "🔴"

        msg = (
            f"🟢 *TRADE OPENED* 🟢\n\n"
            f"📊 *Pair:* {pair}\n"
            f"📍 *Action:* {signal}\n"
            f"💰 *Entry:* `{entry}`\n"
            f"🛡 *Stop Loss:* `{sl}`\n"
            f"🎯 *Take Profit:* `{tp}`\n"
            f"📦 *Lot Size:* {lot}\n"
            f"{conf_icon} *Confidence:* {confidence}%\n\n"
        )

        if confidence_breakdown_lines:
            msg += "📐 *Confidence Breakdown:*\n"
            for line in confidence_breakdown_lines:
                msg += f"  {_escape_markdown(line)}\n"
            msg += "\n"

        msg += f"🧠 *AI Reasoning:*\n"
        for r in reasons[:3]:
            msg += f"  ✅ {_escape_markdown(r)}\n"

        await self.send_message(msg)

    # ── 2. TRADE CLOSED ────────────────────────────────────────

    async def notify_trade_close(self, trade_data: dict):
        """
        trade_data keys: pair, result, pnl, pips, rr_ratio
        """
        result = trade_data.get("result", "CLOSED")
        pnl    = trade_data.get("pnl", 0)
        pips   = trade_data.get("pips", 0)
        rr     = trade_data.get("rr_ratio", 0)

        if result == "WIN":
            icon    = "🏆"
            pnl_str = f"+${round(pnl, 2)}"
        else:
            icon    = "🔴"
            pnl_str = f"-${abs(round(pnl, 2))}"

        pips_str = f"+{round(pips, 1)}" if pips >= 0 else f"{round(pips, 1)}"

        msg = (
            f"{icon} *TRADE CLOSED* {icon}\n\n"
            f"📊 *Pair:* {_escape_markdown(trade_data.get('pair', '—'))}\n"
            f"📋 *Result:* {_escape_markdown(result)}\n"
            f"💵 *Profit/Loss:* {pnl_str}\n"
            f"📏 *Pips:* {pips_str} pips\n"
            f"📈 *R:R Ratio:* 1:{rr}"
        )
        await self.send_message(msg)

    # ── 3. RISK WARNINGS ───────────────────────────────────────

    async def notify_daily_loss_limit(self, used: float, limit: float):
        """Fired when daily loss limit is reached or close to it."""
        pct = (used / limit * 100) if limit else 0
        if pct >= 100:
            msg = (
                f"🚨 *DAILY LOSS LIMIT REACHED* 🚨\n\n"
                f"💀 *Used:* ${used:,.2f} / ${limit:,.2f} ({pct:.0f}%)\n"
                f"🛑 *Action:* Trading has been automatically paused for the day.\n"
                f"⏳ Resume manually with /resume tomorrow."
            )
        else:
            msg = (
                f"⚠️ *DAILY LOSS WARNING* ⚠️\n\n"
                f"📊 *Used:* ${used:,.2f} / ${limit:,.2f} ({pct:.0f}%)\n"
                f"💡 Consider reducing position sizes or pausing trading."
            )
        # Day 102+ hotfix: priority=True — this MUST NOT be dropped by the
        # rate limiter. A dropped daily-loss alert means the operator
        # doesn't know trading has been paused.
        await self.send_message(msg, priority=True)

    async def notify_drawdown_alert(self, drawdown_pct: float, max_allowed: float):
        """Fired when account drawdown exceeds safe thresholds."""
        if drawdown_pct >= max_allowed:
            msg = (
                f"🔴 *DRAWDOWN ALERT* 🔴\n\n"
                f"📉 *Current Drawdown:* {drawdown_pct:.1f}%\n"
                f"🛡 *Max Allowed:* {max_allowed:.1f}%\n"
                f"🚨 *Action:* Circuit breaker triggered! Trading paused.\n"
                f"⏳ Review your positions and resume with /resume when ready."
            )
        else:
            msg = (
                f"⚠️ *DRAWDOWN WARNING* ⚠️\n\n"
                f"📉 *Current Drawdown:* {drawdown_pct:.1f}%\n"
                f"🛡 *Max Allowed:* {max_allowed:.1f}%\n"
                f"💡 Drawdown is approaching the safety limit. Trade with caution."
            )
        # Day 102+ hotfix: priority=True — drawdown alerts must not be
        # dropped by the rate limiter.
        await self.send_message(msg, priority=True)

    # ── 4. DAILY REPORT ────────────────────────────────────────

    async def notify_daily_report(self, report: dict):
        """
        report keys: total_trades, wins, losses, pnl_pct, pnl_abs,
                     best_trade (dict), worst_trade (dict), win_rate
        """
        total   = report.get("total_trades", 0)
        wins    = report.get("wins", 0)
        losses  = report.get("losses", 0)
        wr      = report.get("win_rate", 0)
        pnl_pct = report.get("pnl_pct", 0)
        pnl_abs = report.get("pnl_abs", 0)

        pnl_icon = "📈" if pnl_abs >= 0 else "📉"
        pnl_sign = "+" if pnl_abs >= 0 else ""

        msg = (
            f"📊 *DAILY TRADING REPORT* 📊\n"
            f"🗓 *Date:* {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"
            f"🔢 *Total Trades:* {total}\n"
            f"✅ *Wins:* {wins}  |  ❌ *Losses:* {losses}\n"
            f"🎯 *Win Rate:* {wr:.1f}%\n"
            f"{pnl_icon} *P/L:* {pnl_sign}${round(pnl_abs, 2)} ({pnl_sign}{pnl_pct:.2f}%)\n\n"
        )

        best = report.get("best_trade")
        if best:
            msg += (
                f"🏆 *Best Trade:*\n"
                f"  📊 {_escape_markdown(best.get('pair', '—'))} → "
                f"+${round(best.get('pnl', 0), 2)} "
                f"({best.get('pips', 0)} pips)\n\n"
            )

        worst = report.get("worst_trade")
        if worst:
            msg += (
                f"💀 *Worst Trade:*\n"
                f"  📊 {_escape_markdown(worst.get('pair', '—'))} → "
                f"-${abs(round(worst.get('pnl', 0), 2))} "
                f"({worst.get('pips', 0)} pips)\n\n"
            )

        msg += "🤖 _AI Trader — keeping you informed_"
        await self.send_message(msg)

    # ── 5. NEWS WARNING ────────────────────────────────────────

    async def notify_news_warning(self, event_name: str, time_remaining: str):
        safe_event = _escape_markdown(event_name)
        safe_time  = _escape_markdown(time_remaining)
        msg = (
            f"⚠️ *HIGH IMPACT NEWS WARNING* ⚠️\n\n"
            f"📰 *Event:* {safe_event}\n"
            f"⏰ *Time:* Happening in {safe_time}\n"
            f"🛑 *Action:* Trading paused automatically."
        )
        await self.send_message(msg)

    # ── 5b. SYSTEM WARNING ──────────────────────────────────────
    # BUG FIX: internal recovery pauses (e.g. after repeated cycle errors)
    # used to be sent through notify_news_warning(), which is meant for
    # real economic-calendar events. That produced nonsensical alerts like
    # "📰 Event: System warning: trading paused for recovery" with
    # "⏰ Time: Happening in 5 minutes" for something that was already
    # happening. This is a separate template for non-news system pauses.
    async def notify_system_warning(self, reason: str, pause_duration: str):
        safe_reason = _escape_markdown(reason)
        safe_duration = _escape_markdown(pause_duration)
        msg = (
            f"⚠️ *SYSTEM WARNING* ⚠️\n\n"
            f"🔧 *Reason:* {safe_reason}\n"
            f"⏸️ *Pause duration:* {safe_duration}\n"
            f"🛑 *Action:* Trading paused automatically for recovery."
        )
        await self.send_message(msg)

    # ── 6. WEEKLY CALENDAR ─────────────────────────────────────

    async def notify_weekly_calendar(self, weekly_calendar: dict):
        """
        weekly_calendar — NewsFilter.get_weekly_calendar() output:
            {"2026-06-22": [{"time":..,"currency":..,"event":..,"volatility":{...}}, ...], ...}

        FIX #5: Long calendars are auto-chunked so Telegram never drops them.
        """
        if not weekly_calendar:
            await self.send_message(
                "📅 *FOREX WEEKLY CALENDAR*\n\n✅ No major high-impact events this week."
            )
            return

        msg = "📅 *FOREX WEEKLY CALENDAR* 📅\n\n"
        for day, events in weekly_calendar.items():
            msg += f"🗓 *{_escape_markdown(day)}*\n"
            if not events:
                msg += "  ✅ No major events\n\n"
                continue
            for e in events:
                vol_level = e.get("volatility", {}).get("level", "")
                tag = "⚠️ " if vol_level in ("HIGH", "EXTREME") else "🔸 "
                msg += (
                    f"  {tag}{_escape_markdown(e.get('time'))}  "
                    f"{_escape_markdown(e.get('currency'))}  "
                    f"{_escape_markdown(e.get('event'))}\n"
                )
            msg += "\n"

        # send_message() handles chunking internally
        await self.send_message(msg)

    # ── 7. MORNING BRIEFING ────────────────────────────────────

    async def notify_morning_briefing(
        self,
        date_str: str,
        high_impact_today: list,
        fundamental_scores: dict | None = None,
        session_schedule: dict | None = None,
    ):
        """
        Enhanced morning briefing with market overview + session schedule.

        session_schedule — optional dict of session windows:
            {
                "Asian":    {"open": "00:00 UTC", "close": "08:00 UTC", "active": True},
                "London":   {"open": "07:00 UTC", "close": "16:00 UTC", "active": True},
                "New York": {"open": "12:00 UTC", "close": "21:00 UTC", "active": True},
            }
        """
        msg = (
            f"🌅 *AI TRADER — MORNING BRIEFING* 🌅\n\n"
            f"🗓 *Date:* {_escape_markdown(date_str)}\n\n"
        )

        if session_schedule:
            msg += "🕐 *Trading Sessions Today:*\n"
            for session, info in session_schedule.items():
                icon = "🟢" if info.get("active") else "🔴"
                msg += (
                    f"  {icon} *{_escape_markdown(session)}:* "
                    f"{info.get('open', '—')} → {info.get('close', '—')}\n"
                )
            msg += "\n"

        if high_impact_today:
            msg += "⚠️ *High Impact Events Today:*\n"
            pause_windows = []
            for e in high_impact_today:
                vol       = e.get("volatility", {})
                vol_level = vol.get("level", "?")
                tag = "🔴" if vol_level in ("HIGH", "EXTREME") else "🔸"
                msg += (
                    f"  {tag} {_escape_markdown(e.get('time'))} — "
                    f"{_escape_markdown(e.get('currency'))} "
                    f"{_escape_markdown(e.get('event'))} [{vol_level}]\n"
                )
                pause_windows.append(
                    f"{_escape_markdown(e.get('currency'))} pairs: "
                    f"±30 min around {_escape_markdown(e.get('time'))}"
                )

            msg += "\n⏸ *Trading Pause Windows:*\n"
            for w in pause_windows:
                msg += f"  🛑 {w}\n"
        else:
            msg += "✅ No major high-impact events today — normal trading conditions.\n"

        if fundamental_scores:
            msg += "\n🌐 *Fundamental Bias:*\n"
            for cur, score in fundamental_scores.items():
                if score > 10:
                    icon = "🟢"
                elif score < -10:
                    icon = "🔴"
                else:
                    icon = "🟡"
                msg += f"  {icon} {_escape_markdown(cur)}: {score:+d}\n"

        msg += "\n🤖 _Have a profitable day!_"
        await self.send_message(msg)


# ── Module-level shared notifier (used by command handlers) ───
# FIX #3: cmd_daily no longer instantiates a new TelegramNotifier()
# on every call — they all share this singleton instead.
_shared_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Return (or lazily create) the shared TelegramNotifier instance."""
    global _shared_notifier
    if _shared_notifier is None:
        _shared_notifier = TelegramNotifier()
    return _shared_notifier


# ── Shared reply helper for command handlers ──────────────────
# FIX #2: All command handlers use this instead of reply_text()
# with hardcoded ParseMode.MARKDOWN, so dynamic DB content is safe.

async def _reply(update, text: str):
    """
    Reply to a Telegram update with Markdown, falling back to plain
    text if parsing fails. Chunks long messages automatically.
    """
    for chunk in _chunk_message(text):
        try:
            await update.message.reply_text(_sanitize_for_markdown(chunk), parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            log.warning(f"⚠️ Markdown reply failed ({e}), retrying as plain text…")
            try:
                await update.message.reply_text(chunk)
            except Exception as e2:
                log.error(f"❌ Failed to reply to Telegram command: {e2}")


# ══════════════════════════════════════════════════════════════
#  INCOMING COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════

async def cmd_start(update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message with available commands."""
    msg = (
        "🤖 *AI Forex Trader Bot*\n\n"
        "📡 *Available Commands:*\n\n"
        "📊 /status — System status & portfolio snapshot\n"
        "🛑 /pause — Pause all trading\n"
        "▶️ /resume — Resume trading\n"
        "📅 /calendar — Weekly economic calendar\n"
        "📈 /daily — Today's trading report\n"
        "ℹ️ /help — Show this message\n\n"
        "🧠 _Powered by AI Trading Engine_"
    )
    await _reply(update, msg)


async def cmd_help(update, context: ContextTypes.DEFAULT_TYPE):
    """Alias for /start."""
    await cmd_start(update, context)


async def cmd_status(update, context: ContextTypes.DEFAULT_TYPE):
    """Full system status with portfolio snapshot."""
    try:
        db    = TraderDB()
        stats = db.get_overall_stats()
    except Exception:
        stats = {}

    status_str  = "⏸️ PAUSED" if IS_TRADING_PAUSED else "🚀 RUNNING"
    status_icon = "🟡" if IS_TRADING_PAUSED else "🟢"

    balance = stats.get("balance", 0)
    total   = stats.get("total", 0)
    wins    = stats.get("wins", 0)
    losses  = stats.get("losses", 0)
    wr      = stats.get("win_rate", 0)
    pnl     = stats.get("total_pnl", 0)
    open_t  = stats.get("open_trades", 0)

    pnl_sign = "+" if pnl >= 0 else ""
    pnl_icon = "📈" if pnl >= 0 else "📉"

    msg = (
        f"📊 *AI TRADER — SYSTEM STATUS* 📊\n\n"
        f"{status_icon} *System State:* {status_str}\n\n"
        f"💰 *Balance:* ${balance:,.2f}\n"
        f"{pnl_icon} *Total P/L:* {pnl_sign}${round(pnl, 2)}\n\n"
        f"🔢 *Total Trades:* {total}\n"
        f"✅ *Wins:* {wins}  |  ❌ *Losses:* {losses}\n"
        f"🎯 *Win Rate:* {wr}%\n"
        f"📂 *Open Positions:* {open_t}\n\n"
        f"🕐 *Last Check:* {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
    )
    await _reply(update, msg)


async def cmd_pause(update, context: ContextTypes.DEFAULT_TYPE):
    """Pause trading — sets IS_TRADING_PAUSED and invokes callback."""
    if not _is_authorized(update):
        await _unauthorized_reply(update)
        return
    if IS_TRADING_PAUSED:
        await _reply(update, "⏸️ Trading is *already paused*.")
        return

    await _set_trading_paused(True)
    log.info("🛑 Trading paused via Telegram /pause command")
    await _reply(
        update,
        "🛑 *TRADING PAUSED* 🛑\n\n"
        "No new trades will be executed.\n"
        "▶️ Use /resume to restart trading.",
    )


async def cmd_resume(update, context: ContextTypes.DEFAULT_TYPE):
    """Resume trading — clears IS_TRADING_PAUSED and invokes callback."""
    if not _is_authorized(update):
        await _unauthorized_reply(update)
        return
    if not IS_TRADING_PAUSED:
        await _reply(update, "🚀 Trading is *already running*.")
        return

    await _set_trading_paused(False)
    log.info("▶️ Trading resumed via Telegram /resume command")
    await _reply(
        update,
        "▶️ *TRADING RESUMED* ▶️\n\n"
        "🤖 Scanning market for setups…\n"
        "🛑 Use /pause to stop at any time.",
    )


async def cmd_calendar(update, context: ContextTypes.DEFAULT_TYPE):
    """Show this week's high-impact economic events."""
    try:
        from fundamental.news_filter import NewsFilter
        nf       = NewsFilter()
        calendar = nf.get_weekly_calendar()
    except Exception:
        calendar = None

    if not calendar:
        await _reply(update, "📅 No major high-impact events found for this week.")
        return

    msg = "📅 *FOREX WEEKLY CALENDAR* 📅\n\n"
    for day, events in calendar.items():
        msg += f"🗓 *{_escape_markdown(day)}*\n"
        for e in events:
            vol_level = e.get("volatility", {}).get("level", "")
            tag = "🔴 " if vol_level in ("HIGH", "EXTREME") else "🔸 "
            msg += (
                f"  {tag}{_escape_markdown(e.get('time'))}  "
                f"{_escape_markdown(e.get('currency'))}  "
                f"{_escape_markdown(e.get('event'))}\n"
            )
        msg += "\n"

    # _reply() handles chunking for long calendars
    await _reply(update, msg)


async def cmd_daily(update, context: ContextTypes.DEFAULT_TYPE):
    """Generate today's trading report on demand."""
    try:
        db      = TraderDB()
        stats   = db.get_overall_stats()
        pnl     = stats.get("total_pnl", 0)
        balance = stats.get("balance", 10000)
        pnl_pct = (pnl / 10000) * 100 if balance else 0

        report = {
            "total_trades": stats.get("total", 0),
            "wins":         stats.get("wins", 0),
            "losses":       stats.get("losses", 0),
            "win_rate":     stats.get("win_rate", 0),
            "pnl_abs":      pnl,
            "pnl_pct":      pnl_pct,
        }

        # FIX #3: Use shared notifier, not a new instance
        await get_notifier().notify_daily_report(report)
        await _reply(update, "📊 Daily report sent above ☝️")

    except Exception as e:
        await _reply(update, f"❌ Could not generate daily report: {_escape_markdown(str(e))}")


# ══════════════════════════════════════════════════════════════
#  BOT STARTUP
# ══════════════════════════════════════════════════════════════
def start_telegram_bot_polling():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        log.warning("⚠️ TELEGRAM_TOKEN not set — skipping bot polling.")
        return

    # ✅ নতুন event loop এ চালাও — main loop এর সাথে conflict হবে না
    import threading

    def _run():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        app = (
            Application.builder()
            .token(token)
            # P3 audit fix: timeouts must be set on the BUILDER, not on
            # run_polling(). In newer python-telegram-bot versions
            # (≥20.7), run_polling() no longer accepts read_timeout /
            # write_timeout / connect_timeout / pool_timeout kwargs —
            # passing them crashes with "unexpected keyword argument
            # 'read_timeout'" and the polling thread dies. The correct
            # place is ApplicationBuilder, which propagates them to the
            # underlying HTTPXRequest used by getUpdates long-polling.
            .read_timeout(60)
            .write_timeout(30)
            .connect_timeout(30)
            .pool_timeout(30)
            .build()
        )
        app.add_handler(CommandHandler("start",    cmd_start))
        app.add_handler(CommandHandler("help",     cmd_help))
        app.add_handler(CommandHandler("status",   cmd_status))
        app.add_handler(CommandHandler("pause",    cmd_pause))
        app.add_handler(CommandHandler("resume",   cmd_resume))
        app.add_handler(CommandHandler("calendar", cmd_calendar))
        app.add_handler(CommandHandler("daily",    cmd_daily))

        # ── Day 93 — Register extension commands (positions, close,
        # symbols, indicators, source, account) ──────────────────
        try:
            from alerts.telegram_ext import register_extension_commands
            register_extension_commands(app)
        except Exception as e:
            log.warning(f"[Telegram] extension commands failed to load: {e}")

        # ── Network-resilient error handler ────────────────────────
        # When the network is down (DNS, getaddrinfo failed, proxy error,
        # etc.), python-telegram-bot logs a full traceback every 5 seconds
        # and floods the log.  Catch these specific errors and log a
        # single compact line instead — the polling loop will auto-retry.
        async def _on_error(update, context):
            err = context.error
            err_str = str(err)
            is_network = any(s in err_str.lower() for s in (
                "getaddrinfo", "connection", "timeout", "timed out",
                "network", "dns", "unreachable", "refused", "reset",
                "11001", "etimedout", "ehostunreach",
            ))
            if is_network:
                # Compact one-line warning — no traceback spam.
                log.warning(f"⚠️ Telegram network error (auto-retry): {err_str[:80]}")
            else:
                # Real error — log normally with traceback.
                log.error(f"❌ Telegram error: {err}", exc_info=context.error)
        app.add_error_handler(_on_error)

        log.info("🤖 Telegram Bot Polling Started…")
        
        # Day 82 CRITICAL FIX: Delete any existing webhook BEFORE starting polling.
        # If a webhook is set, starting polling causes 409 Conflict errors.
        # This can happen if: a previous run crashed mid-webhook-setup, or a
        # competing process set a webhook on this token, or Telegram has stale
        # state. Deleting the webhook now ensures clean polling-only mode.
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # delete_webhook() is a coroutine that must run in an event loop.
            # Use synchronous loop.run_until_complete() so we don't need to
            # await here (we're in a sync context before run_polling).
            loop.run_until_complete(app.bot.delete_webhook(drop_pending_updates=False))
            log.info("[Telegram] Pre-polling webhook cleanup complete")
        except Exception as e:
            # Not fatal — if there was no webhook, delete returns gracefully.
            # If it fails for other reasons, polling might still work anyway.
            log.debug(f"[Telegram] Webhook deletion (non-fatal): {e}")
        
        # Round-5 audit fix: wrap run_polling in a retry loop so that
        # if the polling thread crashes (network glitch, transient
        # library error, etc.), it auto-restarts instead of dying
        # silently and leaving the bot unresponsive to commands.
        #
        # The previous code did `try: app.run_polling(...) except: log`
        # which logged the crash and then the thread exited. The
        # operator's bot would silently stop responding to /status
        # /pause /resume commands until the next process restart.
        #
        # Now: up to 5 restart attempts with exponential backoff
        # (10s, 20s, 40s, 80s, 160s). After 5 failures, give up and
        # log CRITICAL — by then something is structurally wrong
        # (bad token, network partition) that needs human attention.
        MAX_POLLING_RESTARTS = 5
        for restart_attempt in range(1, MAX_POLLING_RESTARTS + 1):
            try:
                # P3 audit fix: enable long polling. Timeouts (read/write/
                # connect/pool) are set on the ApplicationBuilder above —
                # run_polling() in newer python-telegram-bot (≥20.7) no
                # longer accepts them as kwargs.
                app.run_polling(
                    poll_interval=2.0,       # seconds between getUpdates when idle
                    timeout=30,              # long-poll: hold connection up to 30s
                    allowed_updates=[
                        "message",
                        "edited_message",
                        "callback_query",
                    ],
                    drop_pending_updates=False,  # don't skip queued updates on restart
                )
                # If run_polling() returns normally (it shouldn't — it's
                # a blocking call until stop()), break the retry loop.
                break
            except Exception as e:
                log.error(
                    f"❌ Telegram polling crashed (attempt {restart_attempt}/"
                    f"{MAX_POLLING_RESTARTS}): {e}"
                )
                if restart_attempt >= MAX_POLLING_RESTARTS:
                    log.critical(
                        f"❌ Telegram polling giving up after "
                        f"{MAX_POLLING_RESTARTS} failed attempts. Bot "
                        f"will not respond to commands until process "
                        f"restart. Outbound alerts still work via "
                        f"TelegramNotifier.send_message()."
                    )
                    break
                # Exponential backoff: 10s, 20s, 40s, 80s, 160s
                backoff = 10 * (2 ** (restart_attempt - 1))
                log.warning(
                    f"⚠️ Telegram polling will retry in {backoff}s..."
                )
                import time as _t
                _t.sleep(backoff)
                # Rebuild the Application for the next attempt — once
                # run_polling() has crashed, the app object may be in
                # an inconsistent state. Building fresh is safer than
                # trying to re-enter run_polling on the same instance.
                try:
                    app = (
                        Application.builder()
                        .token(token)
                        .read_timeout(60)
                        .write_timeout(30)
                        .connect_timeout(30)
                        .pool_timeout(30)
                        .build()
                    )
                    log.info(
                        f"🔄 Telegram Application rebuilt for attempt "
                        f"{restart_attempt + 1}/{MAX_POLLING_RESTARTS}"
                    )
                except Exception as rebuild_e:
                    log.error(
                        f"❌ Failed to rebuild Telegram Application: "
                        f"{rebuild_e}"
                    )

    t = threading.Thread(target=_run, daemon=True)
    t.start()