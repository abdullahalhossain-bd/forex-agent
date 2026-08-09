"""
backtest/persistent_runner.py — Phase 3 backtest runner with checkpoint,
persistence, resume, and live progress dashboard.

This is the high-level entry point for Phase 3 backtests. It wraps the
existing run_unified_backtest decision core but adds:

  - Per-trade JSONL persistence (trades/<symbol>.jsonl, losses/<symbol>.jsonl)
  - Atomic checkpoint.json (per-bar cursor + open_trades + broker state)
  - Atomic summary.json (continuously updated stats)
  - Resume from interrupted run (--resume flag)
  - Graceful Ctrl+C / SIGTERM shutdown
  - Live progress dashboard (terminal)
  - Optional async LLM loss-analysis queue (non-blocking)

ARCHITECTURE
------------
The runner calls run_unified_backtest's INTERNAL loop bar-by-bar so it
can checkpoint after each bar (or every K bars). This is achieved by
extracting the loop body into _run_bar() — see P3-1 audit for the
extraction rationale.

PERSISTENCE CONTRACT
--------------------
For every closed trade:
  1. append to trades/<symbol>.jsonl  (durable immediately, fsync)
  2. if pnl_usd < 0, also append to losses/<symbol>.jsonl
  3. update summary.json with new stats (atomic)
  4. update checkpoint.json with new cursor + open_trades (atomic, every K bars)

RESUME CONTRACT
---------------
On --resume:
  1. Read checkpoint.json → get last cursor i, open_trades, broker.balance, broker._tc
  2. Read trades/<symbol>.jsonl → count already-closed trades, skip them
  3. Reconstruct broker state (balance, _tc counter)
  4. Reconstruct open_trades list
  5. Resume loop from i+1

NO DUPLICATE TRADES: trade_id counter (broker._tc) is restored from
checkpoint, so new trades get trade_id > max(existing). Plus a dedup
safety check: before appending any trade, verify trade_id not already
in trades/<symbol>.jsonl.

LIVE PROGRESS
-------------
Every K bars (default 100), the runner prints a compact dashboard:

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    BACKTEST RUN: 2026-08-08_153022
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Overall: 47.2% | Elapsed 00:28:14 | ETA 00:31:40
    Trades: 8,421 | Wins 4,392 | Losses 4,029 | WR 52.15%
    P&L: Gross +$842.40 | Loss -$701.20 | Net +$141.20
    EURUSD 100% | GBPUSD 100% | USDJPY 72% | USDCHF 0%
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USAGE
-----
    # Start a new run
    python -m backtest.persistent_runner \\
        --symbols EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD \\
        --timeframe H1 \\
        --balance 10000 \\
        --workers 4

    # Resume an interrupted run
    python -m backtest.persistent_runner --run-id 2026-08-08_153022 --resume

    # Run without LLM analysis (faster)
    python -m backtest.persistent_runner --symbols EURUSD --no-llm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.broker_sim import BrokerSimulator, SimulatedTrade, DEFAULT_SPREAD_PIPS
# HistoricalExecutionAdapter is imported from core.execution_adapter below.
# The backtest package does not define backtest.execution_adapter_patch.
from backtest.persistence import (
    RunDir, atomic_write_json, read_json, append_jsonl,
    serialize_trade, deserialize_trade, read_jsonl_count,
)
from core.constants import (
    set_backtest_mode, reset_backtest_memory, is_backtest_mode,
    COMMISSION_USD_PER_LOT, BROKER_SLIPPAGE_PIPS,
)
from core.csv_data_provider import HistoricalCSVDataProvider

log = logging.getLogger("persistent_runner")


# ── Try to import the existing HistoricalExecutionAdapter ────────────────
# It lives in core/execution_adapter.py. We import it here so we don't
# duplicate the wrapper logic.
try:
    from core.execution_adapter import HistoricalExecutionAdapter
except Exception:
    HistoricalExecutionAdapter = None  # will create a minimal fallback below


# ── Graceful shutdown handler ─────────────────────────────────────────────

class ShutdownSignal:
    """Set to True when SIGINT/SIGTERM received. Polled by the runner loop."""
    def __init__(self):
        self.requested = False
        self.reason = ""

    def trigger(self, reason: str = "user requested"):
        self.requested = True
        self.reason = reason


_shutdown = ShutdownSignal()


def _install_signal_handlers():
    """Install SIGINT/SIGTERM handlers that set the shutdown flag."""
    def handler(signum, frame):
        sig_name = signal.Signals(signum).name
        log.warning(f"\n[persistent_runner] {sig_name} received — finishing current bar, then shutting down gracefully...")
        _shutdown.trigger(sig_name)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ── State snapshot / restore ──────────────────────────────────────────────

def snapshot_state(
    cursor: int,
    open_trades: list[SimulatedTrade],
    closed_count: int,
    entry_bar: dict,
    broker_balance: float,
    broker_tc: int,
    rejection_stats: dict,
    per_symbol_stats: dict,
    symbol: str,
) -> dict:
    """Capture all state needed to resume after interrupt."""
    return {
        "cursor": int(cursor),
        "symbol": symbol,
        "open_trades": [serialize_trade(t) for t in open_trades],
        "closed_trades_count": int(closed_count),
        "entry_bar": {str(k): int(v) for k, v in entry_bar.items()},
        "broker_balance": float(broker_balance),
        "broker_tc": int(broker_tc),
        "rejection_stats": dict(rejection_stats),
        "per_symbol_stats": dict(per_symbol_stats),
        "snapshot_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def restore_state(checkpoint: dict) -> dict:
    """Reconstruct state dict from a checkpoint."""
    return {
        "cursor": checkpoint.get("cursor", 0),
        "symbol": checkpoint.get("symbol"),
        "open_trades": [deserialize_trade(d) for d in checkpoint.get("open_trades", [])],
        "closed_trades_count": checkpoint.get("closed_trades_count", 0),
        "entry_bar": {int(k): int(v) for k, v in checkpoint.get("entry_bar", {}).items()},
        "broker_balance": checkpoint.get("broker_balance", 10000.0),
        "broker_tc": checkpoint.get("broker_tc", 0),
        "rejection_stats": checkpoint.get("rejection_stats", {}),
        "per_symbol_stats": checkpoint.get("per_symbol_stats", {}),
    }


# ── Per-symbol worker ─────────────────────────────────────────────────────

class SymbolWorker:
    """Runs ONE symbol's backtest with persistence + resume.

    Designed to be called from a worker process in multi-symbol mode,
    or directly in-process for single-symbol runs.

    Lifecycle:
      1. __init__: load CSV provider, build broker+adapter, set up RunDir
      2. run(): per-bar loop with periodic checkpoint + dashboard updates
      3. On shutdown: write final checkpoint, flush trades, exit
    """

    def __init__(
        self,
        run_dir: RunDir,
        symbol: str,
        timeframe: str,
        starting_balance: float = 10000.0,
        warmup_bars: int = 300,
        max_open_trades: int = 10,
        max_hold_bars: int = 100,
        spread_pips: float | None = None,
        commission_per_lot: float | None = None,
        slippage_pips: float | None = None,
        checkpoint_every: int = 100,
        enable_llm_queue: bool = True,
        verbose: bool = False,
        seed: int = 42,
    ):
        self.run_dir = run_dir
        self.symbol = symbol
        self.timeframe = timeframe
        self.starting_balance = starting_balance
        self.warmup_bars = warmup_bars
        self.max_open_trades = max_open_trades
        self.max_hold_bars = max_hold_bars
        self.checkpoint_every = checkpoint_every
        self.enable_llm_queue = enable_llm_queue
        self.verbose = verbose
        self.seed = seed

        # Cost defaults from shared constants
        self._commission = commission_per_lot if commission_per_lot is not None else COMMISSION_USD_PER_LOT
        self._slippage = slippage_pips if slippage_pips is not None else BROKER_SLIPPAGE_PIPS
        self._spread_pips = spread_pips

        # Set up per-process state (must run before AITrader construction)
        set_backtest_mode(True)
        # Seed RNG for reproducibility (per-process, so each worker has its own stream)
        random.seed(seed)
        np.random.seed(seed)

        # Per-symbol memory isolation: clear backtest memory before each symbol
        # (each worker gets a fresh memory/_backtest/)
        reset_backtest_memory()

        # State — initialized in run() or restore()
        self.provider: HistoricalCSVDataProvider | None = None
        self.broker: BrokerSimulator | None = None
        self.adapter = None
        self.primary_df: pd.DataFrame | None = None
        self.total_bars: int = 0

        # Loop state
        self.cursor: int = warmup_bars
        self.open_trades: list[SimulatedTrade] = []
        self.closed_count: int = 0
        self.entry_bar: dict[int, int] = {}
        self.rejection_stats: dict = {
            "WAIT": 0, "NO_TRADE_ANALYSIS": 0, "risk_rejected": 0,
            "permission_blocked": 0, "engine_error": 0, "max_trades": 0,
            "total_bars": 0,
        }
        self.per_symbol_stats: dict = {
            "trades": 0, "wins": 0, "losses": 0,
            "gross_profit": 0.0, "gross_loss": 0.0, "net_pnl": 0.0,
        }

        # Performance tracking
        self._start_time: float | None = None
        self._bars_processed_since_start: int = 0

        # ── AUDIT ADDITION (gate-ablation harness) ─────────────────────
        # Read comma-separated gate names from env var FOREX_BYPASS_CHECKS.
        # When unset/empty, behavior is byte-identical to baseline (no bypass).
        # When set, the named gates are bypassed in TradePermission.check()
        # for THIS run only — never persisted, never affects the live path.
        # Valid names: see risk/trade_permission._BYPASS_CHECK_ALIASES
        # (e.g. "Min confidence", "Session quality", "Confluence quality",
        #  "Risk approved", "S/R zone alignment", "Valid signal",
        #  "Trend alignment (regime)", or aliases like "min_confidence").
        self._bypass_checks: set[str] = self._read_bypass_checks_from_env()
        # Per-gate stats aggregated from perm_out["checks"] on EVERY bar that
        # reached the permission stage. Records how often each gate passed
        # vs failed, regardless of early-exit ordering — this is what the
        # operator needs to judge "is this gate usefully filtering or just
        # redundantly blocking?". NOT used by any trading decision.
        self._gate_stats: dict[str, dict[str, int]] = {}
        # Per-bar record of which gates blocked the trade (for the JSON
        # result's "remaining_blockers" field). Reset each bar.
        self._gate_first_blocker: str = ""

    # ── Setup ────────────────────────────────────────────────────────────

    @staticmethod
    def _read_bypass_checks_from_env() -> set[str]:
        """Read FOREX_BYPASS_CHECKS env var (comma-separated gate names).

        Returns an empty set when the env var is unset or empty — in that
        case `bypass_checks` passed to `evaluate_decision_core()` is None-
        equivalent and TradePermission runs every gate normally (baseline
        behavior, byte-identical to pre-audit).

        The env var is the ONLY way bypasses are injected — there is no
        CLI flag, no config-file persistence, nothing written to disk.
        This guarantees a bypass can ONLY be enabled by an explicit,
        ephemeral env var on a single process invocation, and CANNOT
        leak into the live trading path (which doesn't read this env var).
        """
        raw = os.environ.get("FOREX_BYPASS_CHECKS", "").strip()
        if not raw:
            return set()
        names = {s.strip() for s in raw.split(",") if s.strip()}
        log.warning(
            f"[AUDIT] FOREX_BYPASS_CHECKS is set — bypassing {len(names)} gate(s): "
            f"{sorted(names)}  (THIS RUN ONLY — does NOT affect live path)"
        )
        return names

    def _record_gate_stats(self, perm_out: dict) -> None:
        """Aggregate per-gate pass/fail counts from perm_out['checks'].

        Called on every bar that reaches the permission stage (i.e. analysis
        did not error, decision was BUY/SELL, risk approved). For each gate
        listed in perm_out['checks'], increments:
            self._gate_stats[gate_name]['passed'] += 1   (passed=True)
            self._gate_stats[gate_name]['failed'] += 1   (passed=False)

        Also records the FIRST failing gate (closest to top of the checks
        list) as `self._gate_first_blocker` for this bar — useful for the
        "remaining_blockers" field in the per-experiment JSON result.

        IMPORTANT: this method is OBSERVATION-ONLY. It does not influence
        any trading decision. The bypass itself happens inside
        TradePermission.check() via the `bypass_checks` argument — this
        method just records what the gate *would have* decided, so the
        operator can compare bypass vs baseline.
        """
        checks = perm_out.get("checks") if isinstance(perm_out, dict) else None
        if not isinstance(checks, list):
            return
        first_blocker = ""
        for chk in checks:
            if not isinstance(chk, dict):
                continue
            name = chk.get("check", "?")
            passed = bool(chk.get("passed", False))
            slot = self._gate_stats.setdefault(name, {"passed": 0, "failed": 0})
            if passed:
                slot["passed"] += 1
            else:
                slot["failed"] += 1
                if not first_blocker:
                    first_blocker = name
        self._gate_first_blocker = first_blocker

    def _setup_provider_and_broker(self):
        """Build provider, broker, adapter. Called once at start or resume."""
        # Provider: use HistoricalCSVDataProvider (CSV is primary)
        self.provider = HistoricalCSVDataProvider(self.symbol, self.timeframe)
        self.primary_df = self.provider.primary_df
        self.total_bars = len(self.primary_df)

        # Broker: per-symbol instance (own balance, own _tc counter)
        spread_dict = None
        if self._spread_pips is not None:
            spread_dict = {self.symbol: float(self._spread_pips)}
        self.broker = BrokerSimulator(
            spread_pips=spread_dict,
            commission_per_lot=self._commission,
            slippage_pips=self._slippage,
            starting_balance=self.starting_balance,
            enforce_spread_limit=True,
        )
        # Adapter: thin wrapper around broker
        if HistoricalExecutionAdapter is not None:
            self.adapter = HistoricalExecutionAdapter(self.broker)
        else:
            # Minimal fallback (shouldn't normally happen — adapter exists)
            self.adapter = self.broker

    def _try_make_trader(self):
        """Try to construct the live AITrader. Returns None if it fails
        (e.g. missing `memory` module in audit env)."""
        try:
            from backtest.unified_engine import _make_backtest_trader
            db_path = str(self.run_dir.root / f"trader_{self.symbol}.db")
            trader = _make_backtest_trader(
                symbol=self.symbol,
                timeframe=self.timeframe,
                starting_balance=self.starting_balance,
                db_path=db_path,
            )
            return trader
        except Exception as e:
            log.warning(f"[{self.symbol}] Could not construct AITrader: {e}")
            log.warning(f"[{self.symbol}] Falling back to SKIP-ANALYSIS mode — "
                        f"trades will be synthetic for testing checkpoint/resume only")
            return None

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self, resume: bool = False) -> dict:
        """Run the backtest for this symbol. Returns final per-symbol stats."""
        self._setup_provider_and_broker()
        trader = self._try_make_trader()

        # Restore state if resuming
        if resume:
            self._restore_from_checkpoint()

        # Skip if already complete
        if self.cursor >= self.total_bars:
            log.info(f"[{self.symbol}] Already complete (cursor {self.cursor} >= {self.total_bars})")
            return self._build_final_stats()

        log.info(f"[{self.symbol}] Starting backtest: "
                 f"{self.total_bars} bars, starting from cursor {self.cursor}, "
                 f"warmup {self.warmup_bars}")
        self._start_time = time.perf_counter()
        self._bars_processed_since_start = 0

        # Main per-bar loop
        try:
            for i in range(self.cursor, self.total_bars):
                if _shutdown.requested:
                    log.info(f"[{self.symbol}] Shutdown requested at bar {i} — saving checkpoint and exiting")
                    self._save_checkpoint(i)
                    break

                self._process_bar(i, trader)

                # Periodic checkpoint + dashboard update
                if (i - self.warmup_bars + 1) % self.checkpoint_every == 0:
                    self._save_checkpoint(i + 1)  # next bar to process
                    self._update_summary()
                    self._log_progress(i)

            else:
                # Loop completed without break — process end-of-backtest closes
                if not _shutdown.requested:
                    self._close_open_trades_at_end()
                    self._save_checkpoint(self.total_bars)
                    self._update_summary()
                    elapsed = time.perf_counter() - self._start_time
                    log.info(f"[{self.symbol}] COMPLETE: "
                             f"{self.closed_count} trades, "
                             f"${self.per_symbol_stats['net_pnl']:+.2f} P&L, "
                             f"{elapsed:.1f}s")
        except Exception as e:
            log.error(f"[{self.symbol}] FATAL error at bar {self.cursor}: {e}", exc_info=True)
            self._save_checkpoint(self.cursor)
            raise

        return self._build_final_stats()

    def _process_bar(self, i: int, trader):
        """Process one bar: exits → market_out → decision → trade open."""
        current_time = self.primary_df.index[i]
        self.rejection_stats["total_bars"] += 1

        # 1. Exits first — bar high/low sweep against open trades
        still_open = []
        for trade in self.open_trades:
            opened_at = self.entry_bar.get(trade.trade_id, i)
            result = self.broker.check_exit(
                trade,
                float(self.primary_df.iloc[i]["high"]),
                float(self.primary_df.iloc[i]["low"]),
                float(self.primary_df.iloc[i]["close"]),
                current_time,
            )
            if result:
                result.hold_bars = i - opened_at
                self._on_trade_closed(result, opened_at, i, current_time, "natural_exit")
            else:
                trade.hold_bars = i - opened_at
                if trade.hold_bars > self.max_hold_bars:
                    closed = self.broker.close_trade(
                        trade, float(self.primary_df.iloc[i]["close"]),
                        current_time, "timeout"
                    )
                    closed.hold_bars = trade.hold_bars
                    self._on_trade_closed(closed, opened_at, i, current_time, "timeout")
                else:
                    still_open.append(trade)
        self.open_trades = still_open

        # 2. Max-open-trades gate
        if len(self.open_trades) >= self.max_open_trades:
            self.rejection_stats["max_trades"] += 1
            return

        # 3. Build market_out via provider
        self.provider.advance_to(i)
        try:
            market_out = self.provider.get_market_out(self.symbol, self.timeframe)
        except Exception as e:
            self.rejection_stats["engine_error"] += 1
            if self.verbose:
                log.debug(f"[{self.symbol}] bar {i} market_out error: {e}")
            return

        # 4. Decision core — only if trader is available
        if trader is None:
            # SKIP-ANALYSIS mode (audit env without `memory` module)
            # Skip the bar — no trade will be opened. Useful for testing
            # checkpoint/resume/dedup without the full decision pipeline.
            return

        try:
            session_ctx = {
                "current_session": "BACKTEST",
                "gmt_time": str(current_time),
                "session_strategy": "n/a",
            }
            # AUDIT: forward env-var-driven bypass_checks so TradePermission
            # can skip the named gate(s) for THIS run only. When the env
            # var is unset, self._bypass_checks is empty and the call is
            # equivalent to the original `evaluate_decision_core(market_out,
            # session_ctx)` — baseline behavior is preserved exactly.
            bypass = self._bypass_checks or None
            core = trader.evaluate_decision_core(
                market_out, session_ctx, bypass_checks=bypass,
            )
        except Exception as e:
            self.rejection_stats["engine_error"] += 1
            if self.verbose:
                log.debug(f"[{self.symbol}] bar {i} decision core error: {e}")
            return

        analysis_out = core.get("analysis_out", {})
        dec_out = core.get("dec_out", {})
        risk_out = core.get("risk_out", {})
        perm_out = core.get("perm_out", {})

        # 5. Rejection gates
        if "error" in analysis_out:
            self.rejection_stats["NO_TRADE_ANALYSIS"] += 1
            return

        action = dec_out.get("decision", "WAIT")
        if action not in ("BUY", "SELL"):
            self.rejection_stats["WAIT"] += 1
            return

        if not risk_out.get("approved"):
            self.rejection_stats["risk_rejected"] += 1
            return

        # AUDIT: record per-gate pass/fail BEFORE the early-exit on
        # perm_out['allowed'] — this captures the FULL picture of which
        # gates would have blocked the trade, even when the bypass
        # flipped the final 'allowed' to True. Without this, an
        # ablation run can't tell us which OTHER gates still failed.
        self._record_gate_stats(perm_out)

        if not perm_out.get("allowed"):
            self.rejection_stats["permission_blocked"] += 1
            return

        # 6. Extract trade params + open
        entry = dec_out.get("entry") or float(self.primary_df.iloc[i]["close"])
        sl = risk_out.get("sl_price")
        tp = risk_out.get("tp_price")
        lot = risk_out.get("lot") or 0.01
        confidence = dec_out.get("confidence", 0)

        if not sl or not tp:
            self.rejection_stats["engine_error"] += 1
            return

        spread_pips = market_out.get("ind_ctx", {}).get("spread_pips")
        trade = self.adapter.open_trade(
            symbol=self.symbol, direction=action, entry_price=entry,
            sl=sl, tp=tp, lot=lot, bar_time=current_time,
            spread_pips=spread_pips,
            confidence=int(confidence) if confidence else 0,
            strategy="persistent_runner",
            confluence_factors=0, quality_grade="B",
        )
        # Spread rejection
        if trade is None:
            self.rejection_stats["permission_blocked"] += 1
            return

        self.entry_bar[trade.trade_id] = i
        self.open_trades.append(trade)
        if self.verbose:
            log.info(f"[{self.symbol}] bar {i} OPEN {action} @ {entry:.5f} lot={lot} conf={confidence}")

    def _on_trade_closed(self, trade: SimulatedTrade, opened_at: int, exit_idx: int,
                          exit_time, exit_type: str):
        """Called when a trade closes. Persists it + updates stats."""
        self.closed_count += 1

        # Build the persisted trade record
        record = serialize_trade(trade)
        record.update({
            "run_id": self.run_dir.run_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "opened_at_bar": opened_at,
            "closed_at_bar": exit_idx,
            "exit_type": exit_type,
            "result": "WIN" if trade.pnl_usd >= 0 else "LOSS",
            "closed_at": str(exit_time),
            "llm_analysis_status": "pending" if trade.pnl_usd < 0 and self.enable_llm_queue else "n/a",
        })

        # Persist immediately (atomic, fsync)
        self.run_dir.append_trade(self.symbol, record)

        # Losses go to a separate file too
        if trade.pnl_usd < 0:
            self.run_dir.append_loss(self.symbol, record)
            # Enqueue for LLM analysis (async, non-blocking)
            if self.enable_llm_queue:
                self._enqueue_llm_analysis(trade, record)

        # Update per-symbol stats
        self.per_symbol_stats["trades"] += 1
        if trade.pnl_usd >= 0:
            self.per_symbol_stats["wins"] += 1
            self.per_symbol_stats["gross_profit"] += float(trade.pnl_usd)
        else:
            self.per_symbol_stats["losses"] += 1
            self.per_symbol_stats["gross_loss"] += float(trade.pnl_usd)
        self.per_symbol_stats["net_pnl"] += float(trade.pnl_usd)

        if self.verbose:
            log.info(f"[{self.symbol}] CLOSE {trade.direction} {trade.exit_reason} "
                     f"pnl=${trade.pnl_usd:+.2f} ({trade.pnl_pips:+.1f}p) "
                     f"hold={trade.hold_bars}bars")

    def _enqueue_llm_analysis(self, trade: SimulatedTrade, record: dict):
        """Enqueue a loss for async LLM analysis. Non-blocking."""
        # Just append to the queue file — LLM worker (separate process/thread)
        # will pick it up later. We only capture enough context for the LLM
        # to analyze; the full forensic record stays in trades/<symbol>.jsonl.
        queue_record = {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "run_id": self.run_dir.run_id,
            "direction": trade.direction,
            "entry": trade.entry_price,
            "exit": trade.exit_price,
            "sl": trade.stop_loss,
            "tp": trade.take_profit,
            "lot": trade.lot_size,
            "pnl_usd": float(trade.pnl_usd),
            "pnl_pips": float(trade.pnl_pips),
            "exit_reason": trade.exit_reason,
            "hold_bars": trade.hold_bars,
            "confidence": trade.confidence,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "enqueued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "pending",
        }
        try:
            self.run_dir.append_llm_queue(queue_record)
        except Exception as e:
            log.warning(f"[{self.symbol}] Failed to enqueue LLM analysis for trade {trade.trade_id}: {e}")

    def _close_open_trades_at_end(self):
        """Force-close any still-open trades at end of backtest."""
        if not self.open_trades:
            return
        last_close = float(self.primary_df.iloc[-1]["close"])
        last_time = self.primary_df.index[-1]
        last_idx = len(self.primary_df) - 1
        for trade in self.open_trades:
            opened_at = self.entry_bar.get(trade.trade_id, last_idx)
            closed = self.broker.close_trade(trade, last_close, last_time, "end_of_backtest")
            closed.hold_bars = last_idx - opened_at
            self._on_trade_closed(closed, opened_at, last_idx, last_time, "end_of_backtest")
        self.open_trades = []

    # ── Checkpoint / resume ──────────────────────────────────────────────

    def _save_checkpoint(self, next_cursor: int):
        """Save current state so we can resume from next_cursor."""
        snap = snapshot_state(
            cursor=next_cursor,
            open_trades=self.open_trades,
            closed_count=self.closed_count,
            entry_bar=self.entry_bar,
            broker_balance=self.broker.get_balance() if self.broker else self.starting_balance,
            broker_tc=self.broker._tc if self.broker else 0,
            rejection_stats=self.rejection_stats,
            per_symbol_stats={self.symbol: self.per_symbol_stats},
            symbol=self.symbol,
        )
        try:
            self.run_dir.write_checkpoint(snap)
        except Exception as e:
            log.warning(f"[{self.symbol}] Failed to write checkpoint: {e}")

    def _restore_from_checkpoint(self):
        """Restore state from a previous checkpoint."""
        cp = self.run_dir.read_checkpoint()
        if cp is None:
            log.info(f"[{self.symbol}] No checkpoint found — starting fresh")
            return

        if cp.get("symbol") != self.symbol:
            log.warning(f"[{self.symbol}] Checkpoint symbol mismatch "
                        f"(checkpoint={cp.get('symbol')}, this={self.symbol}) — starting fresh")
            return

        state = restore_state(cp)
        self.cursor = state["cursor"]
        self.open_trades = state["open_trades"]
        self.closed_count = state["closed_trades_count"]
        self.entry_bar = state["entry_bar"]
        self.rejection_stats = state["rejection_stats"]
        # Restore broker state
        if self.broker is not None:
            self.broker.balance = state["broker_balance"]
            self.broker._tc = state["broker_tc"]
        # Restore per-symbol stats
        saved_stats = state["per_symbol_stats"].get(self.symbol, {})
        if saved_stats:
            self.per_symbol_stats.update(saved_stats)

        # Sanity check: count trades in JSONL, compare to closed_count
        jsonl_count = self.run_dir.count_trades(self.symbol)
        if jsonl_count != self.closed_count:
            log.warning(f"[{self.symbol}] Checkpoint closed_count ({self.closed_count}) "
                        f"!= JSONL trade count ({jsonl_count}) — using JSONL count")
            self.closed_count = jsonl_count

        log.info(f"[{self.symbol}] Resumed from cursor {self.cursor} "
                 f"({self.closed_count} trades already closed, "
                 f"{len(self.open_trades)} open trades restored)")

    # ── Progress / summary ───────────────────────────────────────────────

    def _update_summary(self):
        """Update the persistent summary.json with current per-symbol stats."""
        # Read existing summary (may have other symbols' stats)
        existing = self.run_dir.read_summary() or {}
        existing.setdefault("symbols", {})
        existing["symbols"][self.symbol] = self.per_symbol_stats

        # Recompute aggregate stats across all symbols
        total_trades = sum(s.get("trades", 0) for s in existing["symbols"].values())
        total_wins = sum(s.get("wins", 0) for s in existing["symbols"].values())
        total_losses = sum(s.get("losses", 0) for s in existing["symbols"].values())
        gross_profit = sum(s.get("gross_profit", 0.0) for s in existing["symbols"].values())
        gross_loss = sum(s.get("gross_loss", 0.0) for s in existing["symbols"].values())
        net_pnl = gross_profit + gross_loss

        existing.update({
            "total_trades": total_trades,
            "wins": total_wins,
            "losses": total_losses,
            "win_rate": round(100.0 * total_wins / max(total_trades, 1), 2),
            "gross_profit_usd": round(gross_profit, 2),
            "gross_loss_usd": round(gross_loss, 2),
            "net_pnl_usd": round(net_pnl, 2),
            "current_equity_usd": round(self.starting_balance + net_pnl, 2),
        })

        try:
            self.run_dir.write_summary(existing)
        except Exception as e:
            log.warning(f"[{self.symbol}] Failed to write summary: {e}")

    def _log_progress(self, current_bar: int):
        """Print a compact progress line."""
        if self._start_time is None:
            return
        elapsed = time.perf_counter() - self._start_time
        bars_done = current_bar - self.warmup_bars
        bars_total = self.total_bars - self.warmup_bars
        bars_remaining = max(0, bars_total - bars_done)
        bars_per_sec = bars_done / max(elapsed, 0.001)
        eta_sec = bars_remaining / max(bars_per_sec, 0.001)
        pct = 100.0 * bars_done / max(bars_total, 1)
        log.info(f"[{self.symbol}] {pct:5.1f}% | bar {current_bar}/{self.total_bars} | "
                 f"{bars_per_sec:.1f} bars/s | ETA {eta_sec/60:.1f}min | "
                 f"trades {self.closed_count} | "
                 f"W/L {self.per_symbol_stats['wins']}/{self.per_symbol_stats['losses']} | "
                 f"P&L ${self.per_symbol_stats['net_pnl']:+.2f}")

    def _build_final_stats(self) -> dict:
        """Build the final stats dict for this symbol."""
        elapsed = (time.perf_counter() - self._start_time) if self._start_time else 0
        bars_per_sec = self._bars_processed_since_start / max(elapsed, 0.001)
        trades = self.per_symbol_stats["trades"]
        wins = self.per_symbol_stats["wins"]
        losses = self.per_symbol_stats["losses"]
        gross_profit = self.per_symbol_stats["gross_profit"]
        gross_loss = self.per_symbol_stats["gross_loss"]
        net_pnl = self.per_symbol_stats["net_pnl"]
        # AUDIT: compute profit_factor + max_drawdown + average_trade here
        # so the per-experiment JSON has the full metrics the operator
        # asked for, without needing a separate metrics module.
        profit_factor = (
            round(gross_profit / abs(gross_loss), 4)
            if gross_loss < 0 else 0.0
        )
        avg_trade = round(net_pnl / trades, 2) if trades > 0 else 0.0
        # Max drawdown from the closed-trade equity curve (per-trade, not
        # per-bar — simpler, conservative vs. intrabar drawdown)
        max_dd = self._compute_max_drawdown()
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "total_bars": self.total_bars,
            "bars_processed": self.cursor - self.warmup_bars,
            "elapsed_sec": round(elapsed, 2),
            "bars_per_sec": round(bars_per_sec, 2),
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(100.0 * wins / max(trades, 1), 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_pnl": round(net_pnl, 2),
            "profit_factor": profit_factor,
            "max_drawdown": round(max_dd, 2),
            "average_trade": avg_trade,
            "final_balance": round(self.broker.get_balance() if self.broker else self.starting_balance, 2),
            "rejection_stats": dict(self.rejection_stats),
            # AUDIT: per-gate pass/fail counts (only for bars that reached
            # the permission stage). Lets the operator see which gates
            # blocked trades even when a bypass was active.
            "gate_stats": dict(self._gate_stats),
            # AUDIT: which gates were bypassed in this run (empty for baseline)
            "bypassed_gates": sorted(self._bypass_checks),
            "blocked_trades": (
                self.rejection_stats.get("permission_blocked", 0)
                + self.rejection_stats.get("risk_rejected", 0)
                + self.rejection_stats.get("WAIT", 0)
                + self.rejection_stats.get("NO_TRADE_ANALYSIS", 0)
            ),
            "completed": self.cursor >= self.total_bars,
        }

    def _compute_max_drawdown(self) -> float:
        """Compute max drawdown from the closed-trade equity curve.

        Walks the persisted trades JSONL for this symbol, reconstructs the
        equity curve (starting_balance + cumulative pnl_usd per closed
        trade), and returns the largest peak-to-trough drop as a positive
        number. Returns 0.0 if no trades or no JSONL (e.g. SKIP-ANALYSIS
        mode where no trades were opened).
        """
        try:
            trades = list(self.run_dir.read_trades(self.symbol))
        except Exception:
            return 0.0
        if not trades:
            return 0.0
        equity = self.starting_balance
        peak = equity
        max_dd = 0.0
        for t in trades:
            pnl = float(t.get("pnl_usd", 0.0))
            equity += pnl
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        return max_dd


# ── Multi-symbol runner ──────────────────────────────────────────────────

def run_multi_symbol(
    symbols: list[str],
    timeframe: str = "H1",
    starting_balance: float = 10000.0,
    warmup_bars: int = 300,
    max_open_trades: int = 10,
    max_hold_bars: int = 100,
    workers: int = 1,
    checkpoint_every: int = 100,
    enable_llm: bool = True,
    verbose: bool = False,
    run_id: str | None = None,
    resume: bool = False,
) -> dict:
    """Run backtest across multiple symbols, optionally in parallel.

    Args:
        symbols: list of symbol names (e.g. ["EURUSD", "GBPUSD"])
        timeframe: H1 / M15 / H4 (default H1)
        starting_balance: per-symbol starting balance (each symbol is independent)
        workers: number of parallel worker processes (1 = serial)
        resume: if True, resume from existing run_id's checkpoint
        run_id: explicit run_id (default: timestamp-based)

    Returns:
        Final aggregated stats dict
    """
    # Generate run_id
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    run_dir = RunDir(run_id)
    run_dir.mkdirs()

    # Write or read config
    if not resume:
        config = {
            "run_id": run_id,
            "symbols": symbols,
            "timeframe": timeframe,
            "starting_balance": starting_balance,
            "warmup_bars": warmup_bars,
            "max_open_trades": max_open_trades,
            "max_hold_bars": max_hold_bars,
            "workers": workers,
            "checkpoint_every": checkpoint_every,
            "enable_llm": enable_llm,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        run_dir.write_config(config)
    else:
        config = run_dir.read_config() or {}
        log.info(f"[run {run_id}] Resuming — config: {config.get('symbols', symbols)} symbols")
        # Use config's symbols if available
        if config.get("symbols"):
            symbols = config["symbols"]

    # Install signal handlers (graceful shutdown)
    _install_signal_handlers()

    log.info(f"[run {run_id}] Starting {len(symbols)} symbols, {workers} workers, "
             f"timeframe={timeframe}, balance=${starting_balance}")
    log.info(f"[run {run_id}] Output dir: {run_dir.root}")

    # Try to import config.MAX_OPEN_TRADES if not specified
    if max_open_trades is None:
        try:
            from config import MAX_OPEN_TRADES as _MOT
            max_open_trades = int(_MOT)
        except Exception:
            max_open_trades = 10

    if workers <= 1:
        # Serial mode
        results = {}
        for symbol in symbols:
            if _shutdown.requested:
                log.warning(f"[run {run_id}] Shutdown requested — skipping remaining symbols")
                break
            worker = SymbolWorker(
                run_dir=run_dir, symbol=symbol, timeframe=timeframe,
                starting_balance=starting_balance, warmup_bars=warmup_bars,
                max_open_trades=max_open_trades, max_hold_bars=max_hold_bars,
                checkpoint_every=checkpoint_every, enable_llm_queue=enable_llm,
                verbose=verbose,
            )
            try:
                result = worker.run(resume=resume)
                results[symbol] = result
            except Exception as e:
                log.error(f"[run {run_id}] {symbol} failed: {e}", exc_info=True)
                results[symbol] = {"symbol": symbol, "error": str(e)}
    else:
        # Parallel mode — spawn one process per symbol (max `workers` at a time)
        results = _run_parallel(
            run_dir=run_dir, symbols=symbols, timeframe=timeframe,
            starting_balance=starting_balance, warmup_bars=warmup_bars,
            max_open_trades=max_open_trades, max_hold_bars=max_hold_bars,
            workers=workers, checkpoint_every=checkpoint_every,
            enable_llm=enable_llm, verbose=verbose, resume=resume,
        )

    # Build final aggregated stats
    final_summary = run_dir.read_summary() or {}
    final_summary["results_per_symbol"] = results
    final_summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    final_summary["shutdown_reason"] = _shutdown.reason if _shutdown.requested else "completed"

    # Write final metrics
    try:
        atomic_write_json(run_dir.metrics_path, final_summary, indent=2)
        log.info(f"[run {run_id}] Final metrics saved to {run_dir.metrics_path}")
    except Exception as e:
        log.error(f"[run {run_id}] Failed to save final metrics: {e}")

    # Print final summary
    _print_final_summary(run_id, run_dir, results)

    return final_summary


def _run_parallel(
    run_dir: RunDir, symbols: list[str], timeframe: str,
    starting_balance: float, warmup_bars: int, max_open_trades: int,
    max_hold_bars: int, workers: int, checkpoint_every: int,
    enable_llm: bool, verbose: bool, resume: bool,
) -> dict:
    """Run symbols in parallel using multiprocessing."""
    from multiprocessing import Process, Queue

    # Each worker is a separate process (not thread) — Python GIL would
    # serialize threads, but pandas/numpy release the GIL for many ops
    # so threads could work too. Processes are safer for state isolation.
    results_queue: Queue = Queue()
    procs = []

    # Spawn workers in waves of `workers` at a time
    pending = list(symbols)
    running: list[Process] = []

    def spawn_worker(symbol: str):
        p = Process(
            target=_worker_main,
            args=(run_dir.run_id, symbol, timeframe, starting_balance,
                   warmup_bars, max_open_trades, max_hold_bars,
                   checkpoint_every, enable_llm, verbose, resume, results_queue),
            name=f"bt-{symbol}",
        )
        p.start()
        running.append(p)
        log.info(f"[parallel] Spawned worker for {symbol} (pid={p.pid})")

    while pending or running:
        # Spawn new workers up to capacity
        while pending and len(running) < workers and not _shutdown.requested:
            symbol = pending.pop(0)
            spawn_worker(symbol)

        if not running:
            break

        # Wait for any worker to finish
        time.sleep(0.5)
        for p in running[:]:
            if not p.is_alive():
                p.join()
                running.remove(p)
                log.info(f"[parallel] Worker {p.name} exited")

        # Check for results
        while not results_queue.empty():
            try:
                result = results_queue.get_nowait()
                log.info(f"[parallel] {result.get('symbol')} done: "
                         f"{result.get('trades', 0)} trades, "
                         f"${result.get('net_pnl', 0):+.2f} P&L")
            except Exception:
                break

        if _shutdown.requested and not running:
            break

    # Drain remaining results
    final_results = {}
    while not results_queue.empty():
        try:
            result = results_queue.get_nowait()
            final_results[result.get("symbol")] = result
        except Exception:
            break

    return final_results


def _worker_main(run_id, symbol, timeframe, starting_balance, warmup_bars,
                  max_open_trades, max_hold_bars, checkpoint_every,
                  enable_llm, verbose, resume, results_queue):
    """Entry point for parallel worker process."""
    # Each process must set up its own state
    run_dir = RunDir(run_id)
    worker = SymbolWorker(
        run_dir=run_dir, symbol=symbol, timeframe=timeframe,
        starting_balance=starting_balance, warmup_bars=warmup_bars,
        max_open_trades=max_open_trades, max_hold_bars=max_hold_bars,
        checkpoint_every=checkpoint_every, enable_llm_queue=enable_llm,
        verbose=verbose,
    )
    try:
        result = worker.run(resume=resume)
        results_queue.put(result)
    except Exception as e:
        results_queue.put({"symbol": symbol, "error": str(e)})


def _print_final_summary(run_id: str, run_dir: RunDir, results: dict):
    """Print the final summary to stdout."""
    summary = run_dir.read_summary() or {}

    print()
    print("=" * 70)
    print(f"BACKTEST RUN: {run_id}")
    print("=" * 70)

    if _shutdown.requested:
        print(f"Status: INTERRUPTED ({_shutdown.reason})")
    else:
        print("Status: COMPLETED")
    print()

    print("Trades:")
    print(f"  Total:        {summary.get('total_trades', 0):,}")
    print(f"  Wins:         {summary.get('wins', 0):,}")
    print(f"  Losses:       {summary.get('losses', 0):,}")
    win_rate = summary.get('win_rate', 0)
    print(f"  Win Rate:     {win_rate:.2f}%")
    print()

    print("P&L:")
    print(f"  Gross Profit: ${summary.get('gross_profit_usd', 0):+.2f}")
    print(f"  Gross Loss:   ${summary.get('gross_loss_usd', 0):+.2f}")
    print(f"  Net P&L:      ${summary.get('net_pnl_usd', 0):+.2f}")
    print(f"  Equity:       ${summary.get('current_equity_usd', 0):.2f}")
    print()

    print("Per-Symbol:")
    symbols_dict = summary.get("symbols", {})
    if symbols_dict:
        print(f"  {'Symbol':<10} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'WR':>7} {'Net P&L':>12}")
        print(f"  {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*12}")
        for sym, s in sorted(symbols_dict.items()):
            wr = 100.0 * s.get("wins", 0) / max(s.get("trades", 1), 1)
            print(f"  {sym:<10} {s.get('trades',0):>7} {s.get('wins',0):>6} "
                  f"{s.get('losses',0):>7} {wr:>6.2f}% ${s.get('net_pnl',0):>+11.2f}")
    print()

    # LLM queue stats
    llm_queue_path = run_dir.llm_queue_path
    if llm_queue_path.exists():
        from backtest.persistence import read_jsonl_count
        queued = read_jsonl_count(llm_queue_path)
        analyzed = 0
        for sym_file in run_dir.llm_dir.glob("*.jsonl"):
            if sym_file.name != "queue.jsonl":
                analyzed += read_jsonl_count(sym_file)
        print("LLM Analysis:")
        print(f"  Losses queued: {queued}")
        print(f"  Analyzed:      {analyzed}")
        print(f"  Pending:       {queued - analyzed}")
        print()

    print(f"Output dir: {run_dir.root}")
    print(f"  Checkpoint: {run_dir.checkpoint_path.name}")
    print(f"  Summary:    {run_dir.summary_path.name}")
    print(f"  Metrics:    {run_dir.metrics_path.name}")
    print(f"  Trades:     {run_dir.trades_dir}/")
    print(f"  Losses:     {run_dir.losses_dir}/")
    print(f"  LLM queue:  {run_dir.llm_dir}/")
    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 persistent backtest runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbols", nargs="+", default=["EURUSD"],
                        help="Symbols to backtest (default: EURUSD)")
    parser.add_argument("--timeframe", default="H1",
                        help="Primary decision timeframe (default: H1)")
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="Per-symbol starting balance (default: 10000)")
    parser.add_argument("--warmup", type=int, default=300,
                        help="Warmup bars (default: 300, matches live)")
    parser.add_argument("--max-open-trades", type=int, default=None,
                        help="Max concurrent trades per symbol (default: config.MAX_OPEN_TRADES)")
    parser.add_argument("--max-hold-bars", type=int, default=100,
                        help="Max bars to hold a trade before timeout (default: 100)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1 = serial)")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save checkpoint every N bars (default: 100)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM loss-analysis queue (faster)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose per-bar logging")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (default: timestamp-based)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing run-id checkpoint")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy libraries
    for name in ("urllib3", "matplotlib", "PIL", "pandas_ta"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Resolve max_open_trades from config if not specified
    max_open = args.max_open_trades
    if max_open is None:
        try:
            from config import MAX_OPEN_TRADES
            max_open = int(MAX_OPEN_TRADES)
        except Exception:
            max_open = 10

    run_multi_symbol(
        symbols=args.symbols,
        timeframe=args.timeframe,
        starting_balance=args.balance,
        warmup_bars=args.warmup,
        max_open_trades=max_open,
        max_hold_bars=args.max_hold_bars,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        enable_llm=not args.no_llm,
        verbose=args.verbose,
        run_id=args.run_id,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()