from itertools import product

import pandas as pd

from analytics.analytics import PerformanceAnalyzer
from backtest.data_loader import HistoricalDataLoader
from backtest.report import BacktestReport
from backtest.simulator import ForexSimulator
from utils.logger import get_logger

# Round-8 audit fix: bridge live gating pipeline into backtest.
# Without this, backtest results don't reflect the confidence/tier
# gates that block most signals in live trading.
try:
    from backtest.gating_bridge import BacktestGate
    _GATE_AVAILABLE = True
except Exception as _gate_e:
    BacktestGate = None  # type: ignore
    _GATE_AVAILABLE = False
    log = get_logger("backtest_engine")
    log.warning(
        f"[BacktestEngine] gating_bridge unavailable — backtest will "
        f"run in UNGATED mode (every signal trades). Live gating "
        f"behavior will NOT be reflected. Error: {_gate_e}"
    )
else:
    log = get_logger("backtest_engine")

try:
    from memory.knowledge_store import KnowledgeStore
except Exception as e:
    KnowledgeStore = None


class BacktestEngine:
    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,
        report_writer: BacktestReport | None = None,
    ):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.loader = HistoricalDataLoader()
        self.analyzer = PerformanceAnalyzer()
        self.reporter = report_writer or BacktestReport()

    def load_dataset(self, file_path: str, pair: str = "EURUSD", timeframe: str = "15m") -> pd.DataFrame:
        return self.loader.load_csv(file_path=file_path, pair=pair, timeframe=timeframe)

    def run_strategy(
        self,
        strategy,
        df: pd.DataFrame | None = None,
        file_path: str | None = None,
        pair: str = "EURUSD",
        timeframe: str = "15m",
        save_report: bool = True,
        save_to_memory: bool = False,
        report_name: str | None = None,
        enable_gating: bool = True,
    ) -> dict:
        if df is None:
            if not file_path:
                raise ValueError("Either df or file_path is required")
            df = self.load_dataset(file_path=file_path, pair=pair, timeframe=timeframe)

        symbol = self._clean_pair(pair or df.attrs.get("pair", "EURUSD"))
        simulator = ForexSimulator(default_timeout_candles=self._timeout_candles(timeframe))
        balance = self.initial_balance
        open_position = None
        trades = []
        equity_curve = []
        warmup = getattr(strategy, "warmup", 50)

        # ── Round-8 audit fix: initialize live gating bridge ──────────
        # This makes the backtest reflect what would ACTUALLY happen
        # in live trading: signals below the tier's min_confidence get
        # blocked, ConfidenceEngine sample_size grows as trades close,
        # tier auto-promotes from 1 → 2 → 3 as win count grows.
        gate = None
        gating_stats = None
        if enable_gating and _GATE_AVAILABLE:
            try:
                gate = BacktestGate(
                    symbol=symbol,
                    timeframe=timeframe,
                    initial_balance=self.initial_balance,
                )
                log.info(
                    f"[BacktestEngine] Gating ENABLED for {symbol} {timeframe} "
                    f"(tier={gate._lrm.current_tier.tier if gate._lrm else '?'})"
                )
            except Exception as gate_e:
                log.warning(
                    f"[BacktestEngine] Gating init failed — running UNGATED: {gate_e}"
                )
                gate = None
        else:
            log.info(
                f"[BacktestEngine] Gating DISABLED (enable_gating={enable_gating}, "
                f"bridge_available={_GATE_AVAILABLE})"
            )

        for i in range(warmup, len(df)):
            candle = df.iloc[i]

            if open_position:
                closed = simulator.evaluate_exit(open_position, candle, i)
                if closed:
                    balance = round(balance + closed["pnl"], 2)
                    closed["balance_after"] = balance
                    trades.append(closed)
                    open_position = None

                    # ── Round-8: record outcome to build sample_size ──
                    # This is what simulates the real-world bootstrap
                    # period: the first few trades have no historical
                    # data (flat prior), but as trades close, the
                    # ConfidenceEngine starts adjusting confidence.
                    if gate is not None:
                        try:
                            gate.record_outcome(
                                pattern=closed.get("pattern", "unknown"),
                                pair=symbol,
                                timeframe=timeframe,
                                regime=closed.get("regime", "UNKNOWN"),
                                won=(closed.get("result") == "WIN"),
                            )
                            # Maybe promote tier
                            total_closed = len(trades)
                            wins = sum(1 for t in trades if t.get("result") == "WIN")
                            wr = (wins / total_closed * 100.0) if total_closed > 0 else 0.0
                            gate.maybe_promote_tier(total_closed, wr)
                        except Exception as _e:
                            log.debug(f"[BacktestEngine] gate.record_outcome skipped: {_e}")

            if open_position is None:
                history = df.iloc[:i]
                signal = strategy.generate(history)
                if str(signal.get("signal", "HOLD")).upper() in {"BUY", "SELL"}:
                    # ── Round-8: run live gating pipeline before opening ──
                    if gate is not None:
                        try:
                            gate_result = gate.check(
                                signal=signal,
                                confidence_base=float(signal.get("confidence", 70)),
                                pattern=signal.get("pattern", "unknown"),
                                regime=signal.get("regime", "UNKNOWN"),
                                balance=balance,
                                atr=float(signal.get("atr", 0.001)),
                                sl_pips=float(signal.get("stop_pips", 15)),
                                tp_pips=float(signal.get("stop_pips", 15)) * float(signal.get("rr_ratio", 2.0)),
                                spread_pips=float(signal.get("spread_pips", 1.5)),
                            )
                            if not gate_result.allowed:
                                # Signal BLOCKED by live gating — log and skip.
                                # This is the metric that was completely missing
                                # from backtest before this fix.
                                log.debug(
                                    f"[BacktestEngine] SIGNAL BLOCKED at bar {i} "
                                    f"{symbol} {signal.get('signal')} | "
                                    f"stage={gate_result.reject_stage} | "
                                    f"reason={gate_result.reject_reason} | "
                                    f"final_conf={gate_result.final_confidence:.0f}%"
                                )
                                # Store blocked signal info for the report
                                if not hasattr(self, "_blocked_signals"):
                                    self._blocked_signals = []
                                self._blocked_signals.append({
                                    "bar": i,
                                    "signal": signal.get("signal"),
                                    "stage": gate_result.reject_stage,
                                    "reason": gate_result.reject_reason,
                                    "final_confidence": gate_result.final_confidence,
                                    "tier": gate_result.tier,
                                })
                                # Skip opening — do NOT call simulator.open_position
                                equity_curve.append({"time": str(df.index[i]), "equity": balance})
                                continue
                            # Gate passed — overwrite confidence with gated value
                            signal["confidence"] = int(round(gate_result.final_confidence))
                            signal["_gated"] = True
                            signal["_tier"] = gate_result.tier
                        except Exception as gate_e:
                            log.debug(
                                f"[BacktestEngine] gate.check failed (non-fatal, "
                                f"opening trade anyway): {gate_e}"
                            )

                    open_position = simulator.open_position(
                        candle=candle,
                        signal=signal,
                        pair=symbol,
                        balance=balance,
                        risk_per_trade=self.risk_per_trade,
                        candle_index=i,
                    )

            equity_curve.append({"time": str(df.index[i]), "equity": balance})

        if open_position is not None:
            last_candle = df.iloc[-1]
            closed = simulator.force_close(open_position, last_candle, len(df) - 1)
            balance = round(balance + closed["pnl"], 2)
            closed["balance_after"] = balance
            trades.append(closed)
            # Record final outcome
            if gate is not None:
                try:
                    gate.record_outcome(
                        pattern=closed.get("pattern", "unknown"),
                        pair=symbol,
                        timeframe=timeframe,
                        regime=closed.get("regime", "UNKNOWN"),
                        won=(closed.get("result") == "WIN"),
                    )
                except Exception:
                    pass

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        period_label = self._period_label(df)
        summary = self.analyzer.summarize(
            trades_df=trades_df,
            equity_curve=equity_df,
            strategy_name=getattr(strategy, "name", strategy.__class__.__name__),
            pair=symbol,
            period_label=period_label,
        )
        summary["strategy_version"] = getattr(strategy, "version", "v1")
        monte_carlo = self.analyzer.monte_carlo(trades_df, runs=1000, initial_balance=self.initial_balance)

        walk_forward_hint = None
        report_files = {}
        if save_report:
            report_files = self.reporter.save(
                summary=summary,
                trades_df=trades_df,
                ranking=None,
                walk_forward=walk_forward_hint,
                report_name=report_name or f"{getattr(strategy, 'name', 'strategy').lower().replace(' ', '_')}_{symbol.lower()}",
            )

        result = {
            "summary": summary,
            "trades": trades_df,
            "equity_curve": equity_df,
            "monte_carlo": monte_carlo,
            "report_files": report_files,
        }

        # ── Round-8: include gating stats in the result ──────────────
        # This lets the operator see what % of signals were blocked by
        # live gating — the key metric that was completely invisible
        # before this fix.
        if gate is not None:
            gating_stats = gate.stats()
            result["gating_stats"] = gating_stats
            log.info(
                f"[BacktestEngine] Gating summary for {symbol} {timeframe}: "
                f"allowed={gating_stats['allowed']}/{gating_stats['total_signals']} "
                f"({1 - gating_stats['block_rate']:.1%}) | "
                f"blocked={gating_stats['blocked']} ({gating_stats['block_rate']:.1%}) | "
                f"reasons={gating_stats['block_reasons']} | "
                f"final_tier={gating_stats.get('current_tier')}"
            )
        elif hasattr(self, "_blocked_signals") and self._blocked_signals:
            result["gating_stats"] = {"degraded_mode": True}
        else:
            result["gating_stats"] = {"enabled": False}

        if save_to_memory:
            self.save_backtest_memory(summary=summary, monte_carlo=monte_carlo)

        log.info(
            f"[BacktestEngine] Completed | strategy={summary.get('strategy')} | "
            f"trades={summary.get('trades')} | win_rate={summary.get('win_rate')}%"
        )
        return result

    def run_walk_forward(
        self,
        strategy,
        df: pd.DataFrame | None = None,
        file_path: str | None = None,
        pair: str = "EURUSD",
        timeframe: str = "15m",
        save_report: bool = True,
    ) -> dict:
        if df is None:
            if not file_path:
                raise ValueError("Either df or file_path is required")
            df = self.load_dataset(file_path=file_path, pair=pair, timeframe=timeframe)

        train_df, validation_df, test_df = self._walk_forward_split(df)
        train_result = self.run_strategy(strategy, df=train_df, pair=pair, timeframe=timeframe, save_report=False)
        validation_result = self.run_strategy(strategy, df=validation_df, pair=pair, timeframe=timeframe, save_report=False)
        test_result = self.run_strategy(strategy, df=test_df, pair=pair, timeframe=timeframe, save_report=False)

        wf_summary = {
            "train": train_result["summary"],
            "validation": validation_result["summary"],
            "test": test_result["summary"],
            "overfitting_risk": self._overfitting_risk(
                train_result["summary"].get("win_rate", 0),
                validation_result["summary"].get("win_rate", 0),
                test_result["summary"].get("win_rate", 0),
            ),
        }

        report_files = {}
        if save_report:
            report_files = self.reporter.save(
                summary=test_result["summary"],
                trades_df=test_result["trades"],
                walk_forward=wf_summary,
                report_name=f"walk_forward_{getattr(strategy, 'name', 'strategy').lower().replace(' ', '_')}",
            )

        return {
            "train": train_result,
            "validation": validation_result,
            "test": test_result,
            "summary": wf_summary,
            "report_files": report_files,
        }

    def compare_strategies(
        self,
        strategies: list,
        df: pd.DataFrame | None = None,
        file_path: str | None = None,
        pair: str = "EURUSD",
        timeframe: str = "15m",
        save_report: bool = True,
    ) -> dict:
        if df is None:
            if not file_path:
                raise ValueError("Either df or file_path is required")
            df = self.load_dataset(file_path=file_path, pair=pair, timeframe=timeframe)

        strategy_results = [
            self.run_strategy(strategy=s, df=df, pair=pair, timeframe=timeframe, save_report=False)
            for s in strategies
        ]
        ranking = self.analyzer.rank_strategies(strategy_results)

        report_files = {}
        if save_report and strategy_results:
            top_name = ranking[0]["strategy"] if ranking else strategy_results[0]["summary"]["strategy"]
            top_result = next(
                (item for item in strategy_results if item["summary"]["strategy"] == top_name),
                strategy_results[0],
            )
            report_files = self.reporter.save(
                summary=top_result["summary"],
                trades_df=top_result["trades"],
                ranking=ranking,
                report_name=f"strategy_ranking_{self._clean_pair(pair).lower()}",
            )

        return {
            "results": strategy_results,
            "ranking": ranking,
            "report_files": report_files,
        }

    def optimize_strategy(
        self,
        strategy_class,
        param_grid: dict,
        df: pd.DataFrame | None = None,
        file_path: str | None = None,
        pair: str = "EURUSD",
        timeframe: str = "15m",
    ) -> dict:
        if df is None:
            if not file_path:
                raise ValueError("Either df or file_path is required")
            df = self.load_dataset(file_path=file_path, pair=pair, timeframe=timeframe)

        _, validation_df, test_df = self._walk_forward_split(df)
        keys = list(param_grid.keys())
        values = [param_grid[key] for key in keys]
        candidates = []

        for combo in product(*values):
            params = dict(zip(keys, combo))
            strategy = strategy_class(**params)
            result = self.run_strategy(strategy=strategy, df=validation_df, pair=pair, timeframe=timeframe, save_report=False)
            score = self.analyzer.rank_strategies([result])[0]["score"]
            candidates.append({"params": params, "score": score, "summary": result["summary"]})

        candidates.sort(key=lambda item: item["score"], reverse=True)
        best = candidates[0]
        best_strategy = strategy_class(**best["params"])
        out_of_sample = self.run_strategy(strategy=best_strategy, df=test_df, pair=pair, timeframe=timeframe, save_report=False)

        return {
            "best_params": best["params"],
            "best_validation": best["summary"],
            "test_summary": out_of_sample["summary"],
            "candidates": candidates,
        }

    def save_backtest_memory(self, summary: dict, monte_carlo: dict | None = None) -> None:
        if not KnowledgeStore:
            return
        try:
            store = KnowledgeStore()
            memory_text = (
                f"Backtest completed for strategy {summary.get('strategy')} on {summary.get('pair')}. "
                f"Period: {summary.get('period')}. Trades: {summary.get('trades')}. "
                f"Win rate: {summary.get('win_rate')}%. Average RR: 1:{summary.get('average_rr')}. "
                f"Profit factor: {summary.get('profit_factor')}. Max drawdown: {summary.get('max_drawdown')}%. "
                f"Best setup: {summary.get('best_setup')}. Biggest mistake: {summary.get('biggest_mistake')}."
            )
            if monte_carlo and monte_carlo.get("runs"):
                memory_text += (
                    f" Monte Carlo median final balance: {monte_carlo.get('median_final_balance')}. "
                    f"Worst drawdown seen: {monte_carlo.get('worst_drawdown')}%."
                )
            store.add_memory(
                memory_text,
                metadata={
                    "type": "backtest",
                    "pair": summary.get("pair", ""),
                    "strategy": summary.get("strategy", ""),
                    "version": summary.get("strategy_version", "v1"),
                    "win_rate": str(summary.get("win_rate", 0)),
                },
            )
        except Exception as e:
            log.warning(f"[BacktestEngine] Memory save skipped: {e}")

    def _walk_forward_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 60:
            months = list(df.index.to_period("M").unique())
            if len(months) >= 6:
                train_months = months[:-2]
                validation_month = months[-2]
                test_month = months[-1]
                return (
                    df[df.index.to_period("M").isin(train_months)].copy(),
                    df[df.index.to_period("M") == validation_month].copy(),
                    df[df.index.to_period("M") == test_month].copy(),
                )

        train_end = int(len(df) * 0.67)
        validation_end = int(len(df) * 0.84)
        return (
            df.iloc[:train_end].copy(),
            df.iloc[train_end:validation_end].copy(),
            df.iloc[validation_end:].copy(),
        )

    def _overfitting_risk(self, train_wr: float, validation_wr: float, test_wr: float) -> str:
        val_drop = train_wr - validation_wr
        test_drop = validation_wr - test_wr
        if val_drop > 12 or test_drop > 10:
            return "HIGH"
        if val_drop > 6 or test_drop > 6:
            return "MEDIUM"
        return "LOW"

    def _timeout_candles(self, timeframe: str) -> int:
        mapping = {
            "1m": 2880,
            "5m": 576,
            "15m": 192,
            "30m": 96,
            "1h": 48,
            "4h": 12,
            "1d": 3,
        }
        return mapping.get(timeframe, 192)

    def _period_label(self, df: pd.DataFrame) -> str:
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
            return f"{df.index.min().strftime('%b %Y')} - {df.index.max().strftime('%b %Y')}"
        return f"{len(df)} candles"

    def _clean_pair(self, pair: str) -> str:
        # Round-14 fix: see backtest/simulator.py — blanket "USDT"->"USD"
        # replace corrupted USDTRY/USDTHB. Only strip a trailing Tether suffix.
        cleaned = str(pair).upper().replace("/", "").replace("=X", "").strip()
        if cleaned.endswith("USDT"):
            cleaned = cleaned[:-1]
        return cleaned