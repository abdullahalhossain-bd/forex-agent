#!/usr/bin/env python3
"""Read-only MT5 provenance audit for the Market/Analysis/Decision boundaries.

This script never imports execution or calls an order API. It independently
fetches closed MT5 candles, runs the analysis agents, and compares only fields
actually consumed by those agents.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()

try:
    import MetaTrader5 as mt5
except ImportError as exc:  # pragma: no cover - environment dependent
    print(f"FAIL MT5 package: unavailable ({exc})")
    raise SystemExit(1)

from agents.analysis_agent import AnalysisAgent
from agents.decision_agent import DecisionAgent
from agents.market_agent import MarketAgent
from agents.risk_agent import RiskAgent
from data.fetcher import DataFetcher
from data.fetcher import (
    TIMEFRAME_MAP,
    _to_mt5_symbol,
)
from data.indicator_registry import add_canonical_indicators, get_ai_context


# These are the fields read directly by AnalysisAgent/DecisionAgent, not an
# invented market schema. Context objects are reported separately because
# their values may come from APIs, caches, config, or LLMs rather than MT5.
INDICATOR_FIELDS = (
    "price", "trend", "rsi", "rsi_signal", "macd", "macd_cross", "atr",
    "adx", "bb_upper", "bb_lower", "bb_pct", "sma_20", "sma_50",
    "sma_200", "ema_9", "ema_21", "stoch_k", "stoch_d", "cci", "vwap",
    "spread_pips", "spread_avg_20",
)


def _result(label: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {label}: {detail}")
    return ok


def _same(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(right)))
    except (TypeError, ValueError):
        return left == right


def _closed_rates(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], 1, limit)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
    frame = pd.DataFrame(rates)
    if "time" not in frame.columns:
        raise RuntimeError("MT5 returned rates without a time field")
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.set_index("time").sort_index()
    frame = frame.rename(columns={"tick_volume": "volume"})
    return frame


def _compare_dict(prefix: str, observed: dict, expected: dict, fields: tuple[str, ...]) -> int:
    failures = 0
    for field in fields:
        if field not in observed:
            continue
        actual = observed.get(field)
        reference = expected.get(field)
        ok = _same(actual, reference)
        if not ok:
            failures += 1
        _result(
            f"{prefix}.{field}", ok,
            f"agent={actual!r} independent={reference!r}",
        )
    return failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", default=os.getenv("AUDIT_SYMBOL", "EURUSD"))
    parser.add_argument("timeframe", nargs="?", default=os.getenv("AUDIT_TIMEFRAME", "M15"))
    parser.add_argument("--bars", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    normalizer = DataFetcher.__new__(DataFetcher)
    canonical_symbol = normalizer._normalize_symbol(args.symbol)
    canonical_timeframe = normalizer._normalize_timeframe(args.timeframe)
    if canonical_timeframe is None:
        print(f"FAIL timeframe: unsupported {args.timeframe!r}")
        return 1
    broker_symbol = _to_mt5_symbol(canonical_symbol)

    print("MT5 AGENT DATA-INTEGRITY AUDIT (READ ONLY)")
    print(f"UTC now: {datetime.now(timezone.utc).isoformat()}")
    print(f"Requested symbol/timeframe: {args.symbol}/{args.timeframe}")
    print(f"Canonical symbol/timeframe: {canonical_symbol}/{canonical_timeframe}")
    print(f"Broker symbol: {broker_symbol}")

    login = int(os.getenv("MT5_LOGIN", "0") or 0)
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")
    path = os.getenv("MT5_PATH", "")
    if not mt5.initialize(path=path or None, login=login, password=password, server=server):
        print(f"FAIL MT5 connection: initialize/login failed: {mt5.last_error()}")
        return 1

    failures = 0
    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        failures += not _result(
            "MT5 connection", bool(terminal and getattr(terminal, "connected", False)),
            f"terminal={getattr(terminal, 'name', None)!r} connected={getattr(terminal, 'connected', None)!r}",
        )
        failures += not _result(
            "MT5 broker/server", bool(account and getattr(account, "server", None)),
            f"configured={server!r} connected={getattr(account, 'server', None)!r} login={getattr(account, 'login', None)!r}",
        )
        try:
            from config import MT5_ONLY_MODE, TEST_MODE
            config_ok = bool(MT5_ONLY_MODE) and not bool(TEST_MODE)
            config_detail = f"MT5_ONLY_MODE={MT5_ONLY_MODE!r} TEST_MODE={TEST_MODE!r}"
        except Exception as exc:
            config_ok = False
            config_detail = f"config import failed: {exc}"
        failures += not _result("live mode configuration", config_ok, config_detail)

        info = mt5.symbol_info(broker_symbol)
        selected = bool(info and (info.visible or mt5.symbol_select(broker_symbol, True)))
        failures += not _result(
            "MT5 symbol", selected,
            f"symbol={broker_symbol!r} visible={getattr(info, 'visible', None)!r}",
        )
        tick = mt5.symbol_info_tick(broker_symbol)
        tick_time = datetime.fromtimestamp(tick.time, timezone.utc) if tick and tick.time else None
        failures += not _result(
            "latest tick", tick_time is not None,
            f"timestamp={tick_time.isoformat() if tick_time else None} bid={getattr(tick, 'bid', None)!r} ask={getattr(tick, 'ask', None)!r}",
        )

        independent_df = _closed_rates(broker_symbol, canonical_timeframe, args.bars)
        latest = independent_df.iloc[-1]
        latest_time = independent_df.index[-1]
        failures += not _result(
            "closed candle", latest_time <= pd.Timestamp.now(tz="UTC"),
            f"timestamp={latest_time.isoformat()} open={latest.get('open')!r} high={latest.get('high')!r} low={latest.get('low')!r} close={latest.get('close')!r} volume={latest.get('volume')!r}",
        )
        failures += not _result(
            "timestamp/index", independent_df.index.is_monotonic_increasing and independent_df.index.tz is not None,
            f"timezone={independent_df.index.tz} monotonic={independent_df.index.is_monotonic_increasing}",
        )

        # Run the real market and analysis boundary. No risk/execution object
        # is constructed and no DecisionAgent is invoked by this diagnostic.
        market_out = MarketAgent(canonical_symbol, canonical_timeframe, verbose=False).run()
        failures += not _result(
            "MarketAgent result", "error" not in market_out,
            f"error={market_out.get('error')!r} source={market_out.get('data_source')!r}",
        )
        failures += not _result(
            "MarketAgent MT5 source", market_out.get("data_source") == "mt5",
            f"data_source={market_out.get('data_source')!r}",
        )
        if "error" in market_out:
            return 1

        agent_df = market_out["df"]
        agent_latest = agent_df.iloc[-1]
        failures += not _result(
            "MarketAgent symbol/timeframe", market_out.get("symbol") == canonical_symbol and market_out.get("timeframe") == canonical_timeframe,
            f"agent={market_out.get('symbol')!r}/{market_out.get('timeframe')!r}",
        )
        agent_latest_time = pd.Timestamp(agent_df.index[-1])
        failures += not _result(
            "MarketAgent latest timestamp", agent_latest_time == latest_time,
            f"agent={agent_latest_time.isoformat()} independent={latest_time.isoformat()}",
        )
        for field in ("open", "high", "low", "close", "volume"):
            failures += not _same(agent_latest.get(field), latest.get(field))
            _result(f"MarketAgent.df.{field}", _same(agent_latest.get(field), latest.get(field)), f"agent={agent_latest.get(field)!r} independent={latest.get(field)!r}")

        independent_with_indicators = get_ai_context(
            add_canonical_indicators(independent_df.copy(), include_patterns=True)
        )
        failures += _compare_dict("AnalysisAgent.ind_ctx", market_out.get("ind_ctx", {}), independent_with_indicators, INDICATOR_FIELDS)

        analysis_out = AnalysisAgent().run(market_out)
        failures += not _result(
            "AnalysisAgent boundary", isinstance(analysis_out, dict) and "error" not in analysis_out,
            f"error={analysis_out.get('error') if isinstance(analysis_out, dict) else analysis_out!r}",
        )
        if isinstance(analysis_out, dict):
            print("DecisionAgent consumed fields (provenance):")
            for path in (
                "final_signal", "signal", "llm", "master_ctx", "sentiment_ctx", "conflict",
                "news", "session_ctx", "unified_signal", "ensemble", "rl_agent", "momentum_ctx",
                "confluence", "advanced_pat_ctx", "pat_ctx", "signal_timestamp", "generated_at",
            ):
                value = analysis_out.get(path)
                print(f"INFO analysis_out.{path}: {value!r} source=analysis/external/cache/config (not MT5 proof)")

            # Reproduce the controller's risk -> decision boundary using the
            # connected account balance. RiskAgent only calculates values;
            # no execution object or order API is imported or invoked.
            signal = (
                analysis_out.get("master_ctx", {}).get("master_signal")
                or analysis_out.get("signal", {}).get("signal", "NO TRADE")
            )
            entry = (
                analysis_out.get("master_ctx", {}).get("master_entry")
                or market_out.get("ind_ctx", {}).get("price")
                or market_out.get("ind_ctx", {}).get("close")
            )
            balance = float(getattr(account, "balance", 0.0) or 0.0)
            risk_out = RiskAgent(account_balance=balance).calculate(
                signal=signal,
                entry=entry,
                ind_ctx=market_out.get("ind_ctx", {}),
                regime=market_out.get("regime", {}),
                symbol=canonical_symbol,
            )
            print(f"INFO risk_out: {risk_out!r} source=RiskAgent calculated from agent inputs/account balance")
            decision_out = DecisionAgent().decide(market_out, analysis_out, risk_out)
            failures += not _result(
                "DecisionAgent boundary", isinstance(decision_out, dict),
                f"decision={decision_out.get('decision') if isinstance(decision_out, dict) else decision_out!r} source=DecisionAgent",
            )
            if isinstance(decision_out, dict):
                print(f"INFO decision_out: {decision_out!r} source=DecisionAgent (no execution performed)")

    except Exception as exc:
        print(f"FAIL diagnostic execution: {type(exc).__name__}: {exc}")
        failures += 1
    finally:
        mt5.shutdown()

    verdict = "PASS" if failures == 0 else "FAIL"
    print(f"FINAL {verdict}: {failures} failed checks")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
