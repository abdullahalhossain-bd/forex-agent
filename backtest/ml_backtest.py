"""
backtest/ml_backtest.py — ML Backtest Engine (Day 74)
======================================================

Runs historical backtests comparing 3 system versions:
  A) Rule Only (indicators + patterns + SMC + risk rules)
  B) Rule + Intelligence (sentiment + news + liquidity + session)
  C) Full AI (ML ensemble + RL + LLM + Master Decision)

Each system is simulated on the same historical data, and performance
metrics are collected: win rate, profit factor, max drawdown, Sharpe
ratio, average R:R, trade count.

⚠️ Round-20 AUDIT FIX — CRITICAL DISCLAIMER:
─────────────────────────────────────────────
The "full_ai" system mode in this file is a SIMULATED approximation —
it does NOT call any real AI/ML/LLM modules. It simply uses looser
thresholds and additional indicator checks to approximate what a
"better" system might achieve. This was confirmed by the operator's
institutional audit: "এই ফাইল Rule Only vs Rule+Intelligence vs Full
AI compare করার কথা বলে, কিন্তু _generate_signal() মেথড শুধু
RSI/MACD/EMA-এর উপর হার্ডকোডেড threshold বাড়ায়".

The comparison results (win rate, profit factor) for "full_ai" are
therefore FABRICATED — they do not represent what the actual trading
pipeline would achieve. Do NOT use these numbers to make deployment
decisions.

To get REAL AI backtest results, use one of:
  1. `backtest/engine.py` with `enable_gating=True` (Round-8 fix) —
     this runs the actual ConfidenceEngine + LiveRiskManager gates
  2. `backtest/honest_backtest_engine.py` — honest single-pass engine
  3. `backtest/per_strategy_tester.py` — per-strategy detailed analysis

Usage:
    from backtest.ml_backtest import MLBacktest
    bt = MLBacktest()
    result = bt.run_backtest(pair="EURUSD", timeframe="15m", limit=2000)

    # IMPORTANT: result["systems"]["full_ai"] is SIMULATED, not real AI.
    # Use backtest/engine.py with enable_gating=True for real AI testing.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger
from core.constants import MEMORY_DIR

log = get_logger("ml_backtest")

DB_PATH = MEMORY_DIR / "ml_backtest_results.db"


@dataclass
class TradeRecord:
    """One simulated trade."""
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    direction: str        # BUY / SELL
    pnl_pips: float
    pnl_usd: float
    result: str           # WIN / LOSS
    rr_ratio: float
    system: str           # rule_only / rule_intel / full_ai


@dataclass
class BacktestMetrics:
    """Performance metrics for one system version."""
    system: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_rr: float = 0.0
    total_pnl_usd: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def calculate(self) -> None:
        """Calculate derived metrics from trade list."""
        if not self.trades:
            return
        self.total_trades = len(self.trades)
        wins = [t for t in self.trades if t["result"] == "WIN"]
        losses = [t for t in self.trades if t["result"] == "LOSS"]
        self.wins = len(wins)
        self.losses = len(losses)
        self.win_rate = (self.wins / self.total_trades * 100) if self.total_trades else 0

        gross_profit = sum(t["pnl_usd"] for t in wins)
        gross_loss = abs(sum(t["pnl_usd"] for t in losses))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)
        self.profit_factor = min(self.profit_factor, 99.0)  # cap for sanity

        self.total_pnl_usd = sum(t["pnl_usd"] for t in self.trades)

        # Max drawdown
        equity = [0.0]
        for t in self.trades:
            equity.append(equity[-1] + t["pnl_usd"])
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            if v > peak: peak = v
            dd = peak - v
            if dd > max_dd: max_dd = dd
        self.max_drawdown_pct = (max_dd / 10000) * 100  # as % of $10k

        # Sharpe ratio (simplified)
        returns = [t["pnl_usd"] for t in self.trades]
        if returns and np.std(returns) > 0:
            self.sharpe_ratio = float(np.mean(returns) / np.std(returns))

        # Average R:R
        rrs = [t["rr_ratio"] for t in self.trades if t["rr_ratio"] > 0]
        self.avg_rr = float(np.mean(rrs)) if rrs else 0

        win_pips = [t["pnl_pips"] for t in wins]
        loss_pips = [t["pnl_pips"] for t in losses]
        self.avg_win_pips = float(np.mean(win_pips)) if win_pips else 0
        self.avg_loss_pips = float(np.mean(loss_pips)) if loss_pips else 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "trades"}

    def summary_line(self) -> str:
        return (
            f"{self.system:<20s} | WR={self.win_rate:.1f}% | PF={self.profit_factor:.2f} | "
            f"DD={self.max_drawdown_pct:.1f}% | Sharpe={self.sharpe_ratio:.2f} | "
            f"Trades={self.total_trades} | PnL=${self.total_pnl_usd:.0f}"
        )


class MLBacktest:
    """Backtest engine comparing 3 system versions."""

    def __init__(self, initial_balance: float = 10000.0, risk_per_trade: float = 0.01):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self._init_db()

    def _init_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(DB_PATH)) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS ml_backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT, timeframe TEXT,
                    system TEXT, win_rate REAL, profit_factor REAL,
                    max_drawdown REAL, sharpe REAL, avg_rr REAL,
                    total_trades INTEGER, total_pnl REAL,
                    approved INTEGER, timestamp TEXT
                )
            """)
            c.commit()

    def run_backtest(
        self,
        pair: str = "EURUSD",
        timeframe: str = "15m",
        limit: int = 2000,
    ) -> Dict[str, BacktestMetrics]:
        """Run backtests for all 3 system versions.

        Returns dict: {"rule_only": metrics, "rule_intel": metrics, "full_ai": metrics}
        """
        log.info(f"[MLBacktest] Starting backtest: {pair} {timeframe} | {limit} candles")

        # Load historical data
        df = self._load_data(pair, timeframe, limit)
        if df is None or len(df) < 100:
            log.error(f"[MLBacktest] Insufficient data: {len(df) if df is not None else 0} rows")
            return {}

        # Run each system
        results = {}
        for system_name in ("rule_only", "rule_intel", "full_ai"):
            log.info(f"[MLBacktest] Running system: {system_name}")
            trades = self._simulate_system(df, pair, system_name)
            metrics = BacktestMetrics(system=system_name, trades=trades)
            metrics.calculate()
            results[system_name] = metrics
            log.info(f"[MLBacktest] {metrics.summary_line()}")
            self._save_result(pair, timeframe, metrics)

        return results

    def _load_data(self, pair: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Load historical OHLCV data with indicators."""
        try:
            from data.fetcher import DataFetcher
            from data.indicators import Indicators
            fetcher = DataFetcher()
            df = fetcher.fetch_ohlcv(pair, timeframe, limit=limit)
            if df is None or df.empty:
                return None
            df = Indicators().add_all(df)
            return df
        except Exception as e:
            log.error(f"[MLBacktest] Data load failed: {e}")
            return None

    def _simulate_system(self, df: pd.DataFrame, pair: str, system: str) -> List[Dict[str, Any]]:
        """Simulate trades for a given system version on historical data.

        Each system uses different signal generation logic:
        - rule_only: Basic indicator signals (RSI, MACD, EMA)
        - rule_intel: + SMC + session + liquidity (simulated)
        - full_ai: + ML probability boost (simulated improvement)
        """
        trades: List[Dict[str, Any]] = []
        pip_size = 0.01 if pair.endswith("JPY") else 0.0001
        position = None

        for i in range(50, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]

            # Generate signal based on system version
            signal = self._generate_signal(df, i, system)

            # Close existing position on SL/TP or opposite signal
            if position:
                close_result = self._check_exit(position, next_row, pip_size)
                if close_result:
                    trades.append(close_result)
                    position = None

            # Open new position
            if not position and signal in ("BUY", "SELL"):
                position = self._open_position(signal, row, next_row, pip_size, pair, system)

        # Close any remaining position
        if position:
            last_row = df.iloc[-1]
            close_result = self._close_position(position, last_row, "end_of_data", pip_size)
            if close_result:
                trades.append(close_result)

        return trades

    def _generate_signal(self, df: pd.DataFrame, i: int, system: str) -> str:
        """Generate a trading signal based on the system version."""
        row = df.iloc[i]
        rsi = float(row.get("rsi_14", 50))
        macd = float(row.get("macd", 0))
        macd_signal = float(row.get("macd_signal", 0))
        ema_20 = float(row.get("ema_20", 0))
        ema_50 = float(row.get("ema_50", 0))
        close = float(row.get("close", 0))

        if system == "rule_only":
            # Basic indicator-only system
            buy_score = 0
            sell_score = 0
            if rsi < 35: buy_score += 1
            if rsi > 65: sell_score += 1
            if macd > macd_signal: buy_score += 1
            if macd < macd_signal: sell_score += 1
            if ema_20 > ema_50: buy_score += 1
            if ema_20 < ema_50: sell_score += 1
            if buy_score >= 2 and buy_score > sell_score: return "BUY"
            if sell_score >= 2 and sell_score > buy_score: return "SELL"
            return "WAIT"

        elif system == "rule_intel":
            # + Session + SMC + liquidity (simulated improvements)
            buy_score = 0
            sell_score = 0
            if rsi < 40 and macd > macd_signal: buy_score += 2
            if rsi > 60 and macd < macd_signal: sell_score += 2
            if ema_20 > ema_50: buy_score += 1
            if ema_20 < ema_50: sell_score += 1
            # Simulated intelligence boost
            atr = float(row.get("atr", 0.001))
            if atr > 0 and close > ema_20 > ema_50:
                buy_score += 1
            if atr > 0 and close < ema_20 < ema_50:
                sell_score += 1
            if buy_score >= 3 and buy_score > sell_score: return "BUY"
            if sell_score >= 3 and sell_score > buy_score: return "SELL"
            return "WAIT"

        elif system == "full_ai":
            # ⚠️ Round-20 AUDIT FIX: This is SIMULATED, NOT real AI.
            # This mode does NOT call any agents/ai/ml/llm modules — it
            # simply uses looser thresholds to approximate "better"
            # performance. The results are FABRICATED and should not
            # be used to make deployment decisions.
            #
            # For REAL AI backtest results, use:
            #   backtest/engine.py with enable_gating=True
            if not hasattr(self, '_full_ai_warning_logged'):
                log.warning(
                    "[MLBacktest] 'full_ai' system is SIMULATED (hardcoded "
                    "thresholds, no real AI/ML/LLM calls). Results are "
                    "FABRICATED. Use backtest/engine.py with "
                    "enable_gating=True for real AI backtest results."
                )
                self._full_ai_warning_logged = True
            # Full AI system — best signals (simulated ML improvement)
            buy_score = 0
            sell_score = 0
            if rsi < 40: buy_score += 2
            if rsi > 60: sell_score += 2
            if macd > macd_signal: buy_score += 1
            if macd < macd_signal: sell_score += 1
            if ema_20 > ema_50: buy_score += 1
            if ema_20 < ema_50: sell_score += 1
            # ML boost — better entry timing
            bb_position = float(row.get("bb_position", 0.5))
            if bb_position < 0.2: buy_score += 2  # near lower band
            if bb_position > 0.8: sell_score += 2  # near upper band
            if buy_score >= 4 and buy_score > sell_score: return "BUY"
            if sell_score >= 4 and sell_score > buy_score: return "SELL"
            return "WAIT"

        return "WAIT"

    def _open_position(self, signal: str, row, next_row, pip_size, pair, system) -> Dict:
        """Open a simulated position."""
        entry = float(next_row.get("open", row.get("close", 0)))
        atr = float(row.get("atr", 0.001))
        sl_distance = atr * 1.5
        tp_distance = atr * 3.0

        if signal == "BUY":
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            sl = entry + sl_distance
            tp = entry - tp_distance

        risk_usd = self.initial_balance * self.risk_per_trade
        lot = risk_usd / (sl_distance / pip_size * 10) if sl_distance > 0 else 0.01
        lot = max(0.01, min(round(lot, 2), 10.0))

        return {
            "direction": signal,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "lot": lot,
            "entry_time": str(next_row.name),
            "system": system,
            "pip_size": pip_size,
        }

    def _check_exit(self, position: Dict, row, pip_size: float) -> Optional[Dict]:
        """Check if SL or TP was hit. Returns trade record if closed."""
        high = float(row.get("high", 0))
        low = float(row.get("low", 0))
        close = float(row.get("close", 0))

        if position["direction"] == "BUY":
            if low <= position["sl"]:
                return self._close_position(position, {"close": position["sl"], "high": high, "low": low}, "SL", pip_size)
            if high >= position["tp"]:
                return self._close_position(position, {"close": position["tp"], "high": high, "low": low}, "TP", pip_size)
        else:
            if high >= position["sl"]:
                return self._close_position(position, {"close": position["sl"], "high": high, "low": low}, "SL", pip_size)
            if low <= position["tp"]:
                return self._close_position(position, {"close": position["tp"], "high": high, "low": low}, "TP", pip_size)
        return None

    def _close_position(self, position: Dict, row, reason: str, pip_size: float) -> Dict:
        """Close a position and calculate PnL."""
        exit_price = float(row.get("close", 0))
        entry = position["entry"]

        if position["direction"] == "BUY":
            pnl_pips = (exit_price - entry) / pip_size
        else:
            pnl_pips = (entry - exit_price) / pip_size

        pnl_usd = pnl_pips * 10 * position["lot"]  # simplified pip value
        result = "WIN" if pnl_usd > 0 else "LOSS"

        risk = abs(entry - position["sl"]) / pip_size
        reward = abs(exit_price - entry) / pip_size
        rr_ratio = reward / risk if risk > 0 else 0

        return {
            "entry_time": position["entry_time"],
            "entry_price": entry,
            "exit_time": str(row.name) if hasattr(row, "name") else "",
            "exit_price": exit_price,
            "direction": position["direction"],
            "pnl_pips": round(pnl_pips, 1),
            "pnl_usd": round(pnl_usd, 2),
            "result": result,
            "rr_ratio": round(rr_ratio, 2),
            "system": position["system"],
        }

    def _save_result(self, pair: str, timeframe: str, metrics: BacktestMetrics) -> None:
        """Save backtest result to DB."""
        try:
            with sqlite3.connect(str(DB_PATH)) as c:
                c.execute("""
                    INSERT INTO ml_backtest_results
                    (pair, timeframe, system, win_rate, profit_factor, max_drawdown,
                     sharpe, avg_rr, total_trades, total_pnl, approved, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pair, timeframe, metrics.system,
                    metrics.win_rate, metrics.profit_factor,
                    metrics.max_drawdown_pct, metrics.sharpe_ratio,
                    metrics.avg_rr, metrics.total_trades,
                    metrics.total_pnl_usd,
                    1 if metrics.profit_factor > 1.3 else 0,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ))
                c.commit()
        except Exception as e:
            log.warning(f"[MLBacktest] DB save failed: {e}")

    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get recent backtest results."""
        try:
            with sqlite3.connect(str(DB_PATH)) as c:
                rows = c.execute(
                    "SELECT * FROM ml_backtest_results ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            cols = ["id", "pair", "timeframe", "system", "win_rate", "profit_factor",
                    "max_drawdown", "sharpe", "avg_rr", "total_trades", "total_pnl",
                    "approved", "timestamp"]
            return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            log.warning(f"Suppressed exception at line 407: {e}")
            return []
