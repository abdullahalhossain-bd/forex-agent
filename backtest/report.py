import json
from pathlib import Path

import pandas as pd


class BacktestReport:
    def to_text(self, summary: dict, ranking: list[dict] | None = None, walk_forward: dict | None = None) -> str:
        lines = [
            "==============================",
            "",
            "AI STRATEGY REPORT",
            "",
            f"Strategy: {summary.get('strategy', 'N/A')}",
            f"Period: {summary.get('period', 'N/A')}",
            f"Trades: {summary.get('trades', 0)}",
            f"Wins: {summary.get('wins', 0)}",
            f"Losses: {summary.get('losses', 0)}",
            f"Win Rate: {summary.get('win_rate', 0)}%",
            f"Average RR: 1:{summary.get('average_rr', 0)}",
            f"Profit Factor: {summary.get('profit_factor', 0)}",
            f"Max Drawdown: {summary.get('max_drawdown', 0)}%",
            f"Sharpe: {summary.get('sharpe', 0)}",
            f"Best Pair: {summary.get('best_pair', 'N/A')}",
            f"Worst Pair: {summary.get('worst_pair', 'N/A')}",
            f"Best Setup: {summary.get('best_setup', 'N/A')}",
            f"Biggest Mistake: {summary.get('biggest_mistake', 'N/A')}",
        ]

        if walk_forward:
            lines.extend(
                [
                    "",
                    "Walk Forward:",
                    f"Train Win Rate: {walk_forward.get('train', {}).get('win_rate', 0)}%",
                    f"Validation Win Rate: {walk_forward.get('validation', {}).get('win_rate', 0)}%",
                    f"Test Win Rate: {walk_forward.get('test', {}).get('win_rate', 0)}%",
                    f"Overfitting Risk: {walk_forward.get('overfitting_risk', 'UNKNOWN')}",
                ]
            )

        if ranking:
            lines.append("")
            lines.append("Ranking:")
            for idx, item in enumerate(ranking, start=1):
                lines.append(
                    f"{idx}. {item['strategy']} | Win {item['win_rate']}% | "
                    f"RR 1:{item['average_rr']} | Score {item['score']}"
                )

        lines.extend(["", "=============================="])
        return "\n".join(lines)

    def save(
        self,
        summary: dict,
        trades_df: pd.DataFrame | None = None,
        ranking: list[dict] | None = None,
        walk_forward: dict | None = None,
        report_name: str | None = None,
    ) -> dict:
        report_dir = Path("reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        base_name = report_name or self._slug(summary.get("strategy", "backtest_report"))

        text_path = report_dir / f"{base_name}.txt"
        json_path = report_dir / f"{base_name}.json"
        csv_path = report_dir / f"{base_name}_trades.csv"

        text_path.write_text(self.to_text(summary, ranking=ranking, walk_forward=walk_forward), encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "ranking": ranking or [],
                    "walk_forward": walk_forward or {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if trades_df is not None and not trades_df.empty:
            trades_df.to_csv(csv_path, index=False)

        return {
            "text": str(text_path),
            "json": str(json_path),
            "trades_csv": str(csv_path) if trades_df is not None and not trades_df.empty else None,
        }

    def _slug(self, text: str) -> str:
        return (
            str(text)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
