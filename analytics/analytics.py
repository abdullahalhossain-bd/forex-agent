from itertools import product

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("performance_analytics")


class PerformanceAnalyzer:
    def summarize(
        self,
        trades_df: pd.DataFrame,
        equity_curve: pd.DataFrame,
        strategy_name: str,
        pair: str,
        period_label: str,
    ) -> dict:
        if trades_df.empty:
            return {
                "strategy": strategy_name,
                "pair": pair,
                "period": period_label,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "profit": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "average_rr": 0.0,
                "sharpe": 0.0,
            }

        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        total_trades = len(trades_df)
        total_pnl = round(float(trades_df["pnl"].sum()), 2)
        total_win = float(wins["pnl"].sum()) if not wins.empty else 0.0
        total_loss = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
        avg_rr = float(trades_df["rr_ratio"].mean()) if "rr_ratio" in trades_df.columns else 0.0
        win_rate = round((len(wins) / total_trades) * 100, 1) if total_trades else 0.0
        profit_factor = round(min(total_win / total_loss, 999.0), 2) if total_loss else 999.0
        sharpe = self._sharpe(trades_df["pnl"], equity_curve)
        max_drawdown = self._max_drawdown(equity_curve)
        expectancy = round(total_pnl / total_trades, 2) if total_trades else 0.0

        pair_stats = self._win_rate_by(trades_df, "pair")
        session_stats = self._win_rate_by(trades_df, "session")
        setup_stats = self._win_rate_by(trades_df, "pattern")

        best_pair = max(pair_stats, key=lambda k: pair_stats[k].get("win_rate", 0))
        worst_pair = min(pair_stats, key=lambda k: pair_stats[k].get("win_rate", 100))

        summary = {
            "strategy": strategy_name,
            "pair": pair,
            "period": period_label,
            "trades": total_trades,
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "win_rate": win_rate,
            "profit": total_pnl,
            "profit_factor": profit_factor,
            "max_drawdown": round(max_drawdown, 2),
            "average_rr": round(avg_rr, 2),
            "sharpe": round(sharpe, 2),
            "expectancy": expectancy,
            "best_pair": best_pair,
            "worst_pair": worst_pair,
            "best_setup": next(iter(setup_stats.keys()), "N/A"),
            "biggest_mistake": next(reversed(setup_stats.keys()), "N/A") if setup_stats else "N/A",
            "session_stats": session_stats,
            "pair_stats": pair_stats,
            "setup_stats": setup_stats,
        }
        return summary

    def rank_strategies(self, strategy_results: list[dict]) -> list[dict]:
        rankings = []
        for result in strategy_results:
            summary = result["summary"]
            score = (
                summary.get("profit_factor", 0) * 20
                + summary.get("win_rate", 0) * 0.5
                + summary.get("average_rr", 0) * 10
                + summary.get("sharpe", 0) * 8
                - summary.get("max_drawdown", 0) * 1.2
            )
            rankings.append(
                {
                    "strategy": summary.get("strategy"),
                    "win_rate": summary.get("win_rate"),
                    "average_rr": summary.get("average_rr"),
                    "profit_factor": summary.get("profit_factor"),
                    "max_drawdown": summary.get("max_drawdown"),
                    "profit": summary.get("profit"),
                    "score": round(score, 2),
                }
            )
        rankings.sort(key=lambda item: item["score"], reverse=True)
        return rankings

    def monte_carlo(self, trades_df: pd.DataFrame, runs: int = 1000, initial_balance: float = 10000.0) -> dict:
        if trades_df.empty:
            return {"runs": runs, "status": "no_trades"}

        pnl_values = trades_df["pnl"].tolist()
        final_balances = []
        drawdowns = []
        rng = np.random.default_rng(42)

        for _ in range(runs):
            sampled = rng.permutation(pnl_values)
            equity = initial_balance
            peak = initial_balance
            max_dd = 0.0
            for pnl in sampled:
                equity += pnl
                if equity > peak:
                    peak = equity
                max_dd = max(max_dd, ((peak - equity) / peak) * 100 if peak else 0.0)
            final_balances.append(equity)
            drawdowns.append(max_dd)

        return {
            "runs": runs,
            "best_final_balance": round(max(final_balances), 2),
            "worst_final_balance": round(min(final_balances), 2),
            "median_final_balance": round(float(np.median(final_balances)), 2),
            "mean_final_balance": round(float(np.mean(final_balances)), 2),
            "worst_drawdown": round(float(max(drawdowns)), 2),
            "median_drawdown": round(float(np.median(drawdowns)), 2),
        }

    def parameter_grid(self, param_options: dict) -> list[dict]:
        keys = list(param_options.keys())
        values = [param_options[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in product(*values)]

    def _sharpe(self, pnl_series: pd.Series, equity_curve: pd.DataFrame = None) -> float:
        """
        P1-10 (Audit Fix): Sharpe ratio should use equity-curve returns,
        not trade-level PnL. Trade-level PnL overstates Sharpe because it
        ignores the time between trades (zero-return periods).

        If equity_curve is provided, compute returns from it.
        Otherwise, fall back to trade-level PnL with a warning.
        """
        # Preferred: use equity curve returns
        if equity_curve is not None and not equity_curve.empty and "equity" in equity_curve.columns:
            equity = equity_curve["equity"].astype(float).dropna()
            if len(equity) < 2:
                return 0.0
            returns = equity.pct_change().dropna()
            if len(returns) < 2 or returns.std() == 0:
                return 0.0
            # Annualize: sqrt(bars_per_year)
            # Assume daily equity snapshots → 252 trading days/year
            bars_per_year = 252
            return float((returns.mean() / returns.std()) * np.sqrt(bars_per_year))

        # Fallback: trade-level PnL (less accurate)
        pnl_series = pnl_series.dropna()
        if len(pnl_series) < 2 or pnl_series.std() == 0:
            return 0.0
        # No annualization factor for trade-level — just raw ratio
        log.warning("[Analytics] Sharpe computed from trade-level PnL (no equity curve) — "
                    "result is NOT annualized and may be overstated")
        return float(pnl_series.mean() / pnl_series.std())

    def _max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        if equity_curve.empty or "equity" not in equity_curve.columns:
            return 0.0
        equity = equity_curve["equity"].astype(float)
        peak = equity.cummax()
        dd = ((peak - equity) / peak.replace(0, np.nan)) * 100
        return float(dd.fillna(0).max())

    def _win_rate_by(self, trades_df: pd.DataFrame, column: str) -> dict:
        if column not in trades_df.columns:
            return {}
        grouped = {}
        for key, group in trades_df.groupby(column):
            if pd.isna(key):
                continue
            wins = int((group["pnl"] > 0).sum())
            grouped[str(key)] = {
                "trades": int(len(group)),
                "wins": wins,
                "win_rate": round((wins / len(group)) * 100, 1) if len(group) else 0.0,
                "pnl": round(float(group["pnl"].sum()), 2),
            }
        return dict(sorted(grouped.items(), key=lambda item: item[1]["win_rate"], reverse=True))
