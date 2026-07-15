"""
alerts/telegram_ext.py — Day 93 Telegram command extensions
============================================================
Adds /positions, /close, /symbols, /indicators, /source commands
to the existing Telegram bot. These let you monitor + control the
bot remotely from your phone.

This module is INTENDED to be imported + registered by the existing
alerts/telegram_bot.py during bot startup. It doesn't replace the
existing bot — it extends it.

Usage (called from telegram_bot.py boot sequence):
    from alerts.telegram_ext import register_extension_commands
    register_extension_commands(application, trader_system)

Commands added:
    /positions   — List all open MT5 positions (ticket, symbol, PnL)
    /close <id>  — Close position with given ticket (MT5 only)
    /symbols     — List configured trading pairs + their spread
    /indicators  — Show latest indicator snapshot for a pair
                   (e.g. /indicators EURUSD)
    /source      — Show which data source MT5/API is active
    /account     — Show account balance/equity/margin (MT5 only)
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("telegram_ext")


def _escape_md(text: str) -> str:
    """Escape Markdown special chars for Telegram."""
    if not text:
        return ""
    for ch in ("_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", ".", "!"):
        text = text.replace(ch, f"\\{ch}")
    return text


def _fmt_pnl(pnl: float) -> str:
    """Format PnL with color icon."""
    if pnl is None:
        return "N/A"
    if pnl > 0:
        return f"🟢 +${pnl:.2f}"
    if pnl < 0:
        return f"🔴 ${pnl:.2f}"
    return f"⚪ ${pnl:.2f}"


def _fmt_position(p: Dict[str, Any]) -> str:
    """Format one position as a Telegram line."""
    ticket = p.get("ticket", "?")
    symbol = p.get("symbol", "?")
    direction = p.get("type", "?").upper()
    volume = p.get("volume", 0)
    pnl = p.get("pnl", 0)
    sl = p.get("sl", 0)
    tp = p.get("tp", 0)
    open_price = p.get("price_open", 0)
    cur_price = p.get("price_current", 0)

    dir_icon = "🟢" if direction == "BUY" else "🔴"
    return (
        f"{dir_icon} #{ticket} {symbol} {direction} {volume:.2f}lot\n"
        f"   Open: {open_price:.5f} | Cur: {cur_price:.5f}\n"
        f"   SL: {sl:.5f} | TP: {tp:.5f}\n"
        f"   PnL: {_fmt_pnl(pnl)}"
    )


async def cmd_positions(update, context):
    """/positions — List all open MT5 positions."""
    # Day 102+ CRITICAL hotfix: authorization required.
    # Open positions expose live P&L, SL/TP levels, and lot sizes —
    # sensitive financial data. Without this check, anyone who knows
    # the bot username could read all open positions.
    from alerts.telegram_bot import _is_authorized, _unauthorized_reply
    if not _is_authorized(update):
        await _unauthorized_reply(update)
        return
    from data.data_orchestrator import get_data_orchestrator
    orch = get_data_orchestrator()

    positions = orch.get_open_positions()

    if not positions:
        # Check if MT5 was even available
        status = orch.status()
        if not status["mt5_available"]:
            msg = (
                "📊 *Open Positions*\n\n"
                f"⚠️ MT5 unavailable (running on API fallback)\n"
                f"Data source: `{status['api_source']}`\n\n"
                "Position monitoring requires MT5 connection.\n"
                "Run on Windows with MT5 terminal to enable."
            )
        else:
            msg = "📊 *Open Positions*\n\n✅ No open positions."
    else:
        total_pnl = sum(p.get("pnl", 0) for p in positions)
        lines = [f"📊 *Open Positions ({len(positions)})*\n"]
        for p in positions[:10]:  # cap at 10 to avoid message-too-long
            lines.append(_fmt_position(p))
            lines.append("")
        if len(positions) > 10:
            lines.append(f"_...and {len(positions) - 10} more_")
        lines.append(f"\n💰 Total PnL: {_fmt_pnl(total_pnl)}")
        msg = "\n".join(lines)

    await _reply_md(update, msg)


async def cmd_close(update, context):
    """/close <ticket> — Close an open position by ticket."""
    # CRITICAL FIX: authorization check — without this, anyone can close positions
    from alerts.telegram_bot import _is_authorized, _unauthorized_reply
    if not _is_authorized(update):
        await _unauthorized_reply(update)
        return
    if not context.args:
        await _reply_md(update, "Usage: `/close <ticket>`\n\nExample: `/close 12345678`")
        return

    try:
        ticket = int(context.args[0])
    except ValueError:
        await _reply_md(update, "❌ Ticket must be a number.\nExample: `/close 12345678`")
        return

    from data.data_orchestrator import get_data_orchestrator
    orch = get_data_orchestrator()

    if not orch.status()["mt5_available"]:
        await _reply_md(update, "❌ MT5 unavailable — cannot close positions on API fallback mode.")
        return

    msg = f"⏳ Closing position #{ticket}..."
    await _reply_md(update, msg)

    # Run the close in a thread (MT5 calls are blocking)
    # Fix: this code already runs inside a coroutine, so there IS a running
    # loop — asyncio.get_running_loop() is the correct, non-deprecated way
    # to fetch it (asyncio.get_event_loop() warns/will error when called
    # outside of a running loop context in newer Python versions).
    success = await asyncio.get_running_loop().run_in_executor(
        None, orch.close_position, ticket
    )

    if success:
        await _reply_md(update, f"✅ Position #{ticket} closed successfully.")
    else:
        await _reply_md(update, f"❌ Failed to close position #{ticket}. Check logs.")


async def cmd_symbols(update, context):
    """/symbols — List configured trading pairs + spread."""
    # Day 102+ CRITICAL hotfix: authorization required.
    # Exposes the bot's pair list and spread data — operational intel
    # that an attacker could use to front-run or manipulate.
    from alerts.telegram_bot import _is_authorized, _unauthorized_reply
    if not _is_authorized(update):
        await _unauthorized_reply(update)
        return
    from config import SYMBOLS
    from data.data_orchestrator import get_data_orchestrator
    orch = get_data_orchestrator()

    lines = [f"💱 *Configured Pairs ({len(SYMBOLS)})*\n"]
    for symbol in SYMBOLS:
        info = orch.get_symbol_info(symbol)
        spread = info.get("spread", "?") if info else "?"
        digits = info.get("digits", "?") if info else "?"
        src = info.get("source", "?") if info else "?"
        lines.append(f"• `{symbol}`  spread={spread}  digits={digits}  [{src}]")

    lines.append(f"\n_Data source: `{orch.status()['last_source']}`_")
    await _reply_md(update, "\n".join(lines))


async def cmd_indicators(update, context):
    """/indicators [symbol] — Show latest indicator snapshot."""
    # Day 102+ CRITICAL hotfix: authorization required.
    # Exposes the bot's indicator stack — reveals which signals the
    # bot is currently seeing, which an attacker could exploit.
    from alerts.telegram_bot import _is_authorized, _unauthorized_reply
    if not _is_authorized(update):
        await _unauthorized_reply(update)
        return
    if not context.args:
        await _reply_md(update, "Usage: `/indicators EURUSD`\n\nShows latest indicator values.")
        return

    symbol = context.args[0].upper()
    from data.data_orchestrator import get_data_orchestrator
    from data.indicators_ext import ExtendedIndicators

    orch = get_data_orchestrator()
    df = orch.get_candles(symbol, "M15", limit=200)
    if df is None or len(df) < 30:
        await _reply_md(update, f"❌ Could not fetch data for {symbol}.")
        return

    ind = ExtendedIndicators()
    df = ind.add_all(df, include_patterns=False)
    ctx = ind.get_ai_context(df)

    lines = [f"📈 *{symbol} M15 Indicators*\n"]
    lines.append(f"Price: `{ctx.get('price','?')}`  Trend: *{ctx.get('trend','?')}*")
    lines.append(f"RSI: `{ctx.get('rsi','?')}` ({ctx.get('rsi_signal','?')})")
    lines.append(f"MACD: `{ctx.get('macd','?')}`  cross: {ctx.get('macd_cross','?')}")
    lines.append(f"ADX: `{ctx.get('adx','?')}`  ATR: `{ctx.get('atr','?')}`")
    lines.append(f"Stoch K/D: `{ctx.get('stoch_k','?')}`/`{ctx.get('stoch_d','?')}`")
    lines.append(f"BB%: `{ctx.get('bb_pct','?')}`  width: `{ctx.get('bb_width','?')}`")
    lines.append(f"EMA9/21: `{ctx.get('ema_9','?')}` / `{ctx.get('ema_21','?')}`")
    lines.append(f"SMA50/200: `{ctx.get('sma_50','?')}` / `{ctx.get('sma_200','?')}`")
    lines.append(f"CCI: `{ctx.get('cci','?')}`")
    lines.append(f"Pivot P/R1/S1: `{ctx.get('pivot_p','?')}` / `{ctx.get('pivot_r1','?')}` / `{ctx.get('pivot_s1','?')}`")
    lines.append(f"\n_Source: `{orch.status()['last_source']}`_")

    await _reply_md(update, "\n".join(lines))


async def cmd_source(update, context):
    """/source — Show which data sources are active."""
    # Day 102+ CRITICAL hotfix: authorization required.
    # Reveals whether MT5 is connected (operational state an attacker
    # could use to time exploits when the bot is on API fallback).
    from alerts.telegram_bot import _is_authorized, _unauthorized_reply
    if not _is_authorized(update):
        await _unauthorized_reply(update)
        return
    from data.data_orchestrator import get_data_orchestrator
    orch = get_data_orchestrator()
    status = orch.status()

    lines = ["🔌 *Data Source Status*\n"]
    lines.append(f"MT5 available: {'✅' if status['mt5_available'] else '❌'}")
    lines.append(f"MT5 initialized: {'✅' if status['mt5_initialized'] else '❌'}")
    lines.append(f"API fallback: `{status['api_source']}`")
    lines.append(f"Last served by: `{status['last_source']}`")
    lines.append(f"Preferred (env): `{status['preferred_source'] or 'auto'}`")

    lines.append("\n_Principle: MT5 first, API fallback only when MT5 unavailable._")
    await _reply_md(update, "\n".join(lines))


async def cmd_account(update, context):
    """/account — Show account balance/equity/margin (MT5 only)."""
    # Day 102+ CRITICAL hotfix: authorization required.
    # Account balance, equity, margin, and floating P&L are the most
    # sensitive financial data the bot holds. Must be gated.
    from alerts.telegram_bot import _is_authorized, _unauthorized_reply
    if not _is_authorized(update):
        await _unauthorized_reply(update)
        return
    from data.data_orchestrator import get_data_orchestrator
    orch = get_data_orchestrator()

    if not orch.status()["mt5_available"]:
        await _reply_md(update, "❌ MT5 unavailable — account info requires MT5 connection.")
        return

    info = orch.get_account_info()
    if not info:
        await _reply_md(update, "❌ Could not fetch account info.")
        return

    lines = ["💼 *MT5 Account*\n"]
    lines.append(f"Balance: `${info.get('balance', 0):.2f}`")
    lines.append(f"Equity: `${info.get('equity', 0):.2f}`")
    lines.append(f"Margin: `${info.get('margin', 0):.2f}`")
    lines.append(f"Free margin: `${info.get('margin_free', 0):.2f}`")
    lines.append(f"Floating PnL: {_fmt_pnl(info.get('profit', 0))}")
    lines.append(f"Margin level: `{info.get('margin_level', 0):.1f}%`")

    await _reply_md(update, "\n".join(lines))


# ── Helper ────────────────────────────────────────────────────────

async def _reply_md(update, text: str):
    """Reply with Markdown formatting (Telegram parse_mode=MarkdownV2)."""
    try:
        await update.message.reply_text(text, parse_mode="MarkdownV2")
    except Exception:
        # Fallback: send without markdown if formatting fails
        try:
            await update.message.reply_text(text.replace("*", "").replace("`", ""))
        except Exception as e:
            log.error(f"telegram_ext reply failed: {e}")


# ── Registration ──────────────────────────────────────────────────

def register_extension_commands(application):
    """Register all Day 93 extension commands with the bot application.

    Call this from alerts/telegram_bot.py during startup:

        from alerts.telegram_ext import register_extension_commands
        register_extension_commands(application)
    """
    from telegram.ext import CommandHandler

    commands = [
        ("positions",   cmd_positions),
        ("close",       cmd_close),
        ("symbols",     cmd_symbols),
        ("indicators",  cmd_indicators),
        ("source",      cmd_source),
        ("account",     cmd_account),
    ]

    for name, handler in commands:
        try:
            application.add_handler(CommandHandler(name, handler))
            log.info(f"[TelegramExt] registered /{name}")
        except Exception as e:
            log.warning(f"[TelegramExt] failed to register /{name}: {e}")

    log.info(f"[TelegramExt] {len(commands)} extension commands registered")


# ── Rich signal alert (for trade notifications) ───────────────────

async def notify_rich_signal(
    bot,
    chat_id: str,
    signal_data: Dict[str, Any],
):
    """Send a richly-formatted trade signal alert to Telegram.

    Args:
        bot:       telegram.Bot instance
        chat_id:   target Telegram chat ID
        signal_data: dict with keys:
            pair, direction, confidence, entry, sl, tp, lot,
            strategy, regime, reasons (list), source (mt5/api)

    Example output:
        🟢 EURUSD BUY signal
        Confidence: 85% | Strategy: SMC_PULLBACK
        Entry: 1.0850 | SL: 1.0820 | TP: 1.0910
        Lot: 0.10 | Risk: 1%
        Regime: TRENDING BULLISH STRONG

        Reasons:
        • Strong BOS + bullish CHoCH
        • RSI bullish zone (62)
        • Price at OB support
    """
    direction = signal_data.get("direction", "").upper()
    pair = signal_data.get("pair", "?")
    conf = signal_data.get("confidence", 0)
    entry = signal_data.get("entry", 0)
    sl = signal_data.get("sl", 0)
    tp = signal_data.get("tp", 0)
    lot = signal_data.get("lot", 0)
    strategy = signal_data.get("strategy", "?")
    regime = signal_data.get("regime", "?")
    reasons = signal_data.get("reasons", [])
    source = signal_data.get("source", "?")

    icon = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⏸️"

    lines = [
        f"{icon} *{pair} {direction}* signal",
        f"Confidence: *{conf}%* | Strategy: `{strategy}`",
        f"Entry: `{entry}` | SL: `{sl}` | TP: `{tp}`",
        f"Lot: `{lot}` | Regime: {regime}",
    ]
    if reasons:
        lines.append("\n_Reasons:_")
        for r in reasons[:5]:
            lines.append(f"• {r}")
    lines.append(f"\n_Source: `{source}`_")

    text = "\n".join(lines)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown",
        )
    except Exception as e:
        log.warning(f"notify_rich_signal failed: {e}")
        # Fallback plain text
        try:
            await bot.send_message(chat_id=chat_id, text=text.replace("*","").replace("`",""))
        except Exception:
            pass