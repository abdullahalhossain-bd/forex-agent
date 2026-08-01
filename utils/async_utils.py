"""utils/async_utils.py — safe helpers for running coroutines from sync code.

Fixes the repeated `DeprecationWarning: There is no current event loop`
pattern that used to be copy-pasted around the codebase as:

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(coro)
        else:
            loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(coro)
        loop.close()

`asyncio.get_event_loop()` is deprecated outside of coroutines/callbacks
since Python 3.10 and is scheduled for removal — calling it with no
running loop and no loop set on the current thread will eventually raise
instead of silently creating one.

This module gives a single, non-deprecated replacement:

    from utils.async_utils import run_coro_sync
    run_coro_sync(notifier.send_message(msg))

Behavior:
  - If called from inside a running event loop (e.g. from an `async def`
    context), the coroutine is scheduled on that loop with
    `asyncio.ensure_future` (fire-and-forget) since we can't block a
    loop that's already running.
  - If called from plain sync code with no running loop, we reuse a
    single persistent background loop (created lazily, never closed
    mid-run) instead of repeatedly creating + closing a new loop each
    call. This also avoids the "Event loop is closed" errors seen when
    long-lived objects like a python-telegram-bot `Bot` cache a loop
    reference internally.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Coroutine, Optional

from utils.logger import get_logger

log = get_logger("async_utils")

_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_lock = threading.Lock()


def _get_persistent_loop() -> asyncio.AbstractEventLoop:
    """Get (or lazily create) a single persistent background event loop.

    Reused across calls so long-lived clients (e.g. Telegram Bot) that
    cache a reference to "the" event loop keep working, instead of the
    old pattern of creating + closing a fresh loop on every call.
    """
    global _bg_loop
    with _bg_lock:
        if _bg_loop is None or _bg_loop.is_closed():
            _bg_loop = asyncio.new_event_loop()
        return _bg_loop


def run_coro_sync(coro: Coroutine) -> None:
    """Run an async coroutine safely from synchronous code.

    Replacement for the deprecated `asyncio.get_event_loop()` idiom.
    Fire-and-forget: does not return the coroutine's result. Intended
    for side-effecting calls like sending a Telegram notification.
    """
    try:
        # Are we already inside a running loop (e.g. called from async
        # code)? asyncio.get_running_loop() is the modern, non-deprecated
        # way to check this — it raises RuntimeError if there isn't one.
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None:
        # Can't block a loop that's already running — schedule instead.
        running_loop.create_task(coro)
        return

    # No running loop on this thread — use our persistent background loop.
    loop = _get_persistent_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coro)
    except Exception as e:
        log.warning(f"[async_utils] run_coro_sync failed: {e}")