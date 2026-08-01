"""initial schema — candles, indicators, patterns, analysis, trades, economic_history

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-25 00:00:00 UTC

Captures the schema that database/db.py has been creating via
CREATE TABLE IF NOT EXISTS since day 1. Existing databases will be
marked as at this revision without any DDL running (the tables already
exist). Fresh databases will get all tables created in one shot.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the initial schema using op.create_table (canonical Alembic API)."""

    # ── candles ──────────────────────────────────────────────
    op.create_table(
        "candles",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol",    sa.Text, nullable=False),
        sa.Column("timeframe", sa.Text, nullable=False),
        sa.Column("time",      sa.Text, nullable=False),
        sa.Column("open",      sa.REAL),
        sa.Column("high",      sa.REAL),
        sa.Column("low",       sa.REAL),
        sa.Column("close",     sa.REAL),
        sa.Column("volume",    sa.REAL),
        sa.UniqueConstraint("symbol", "timeframe", "time", name="uq_candles_symbol_tf_time"),
    )

    # ── indicators ───────────────────────────────────────────
    op.create_table(
        "indicators",
        sa.Column("id",        sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol",    sa.Text, nullable=False),
        sa.Column("timeframe", sa.Text, nullable=False),
        sa.Column("time",      sa.Text, nullable=False),
        sa.Column("rsi",       sa.REAL),
        sa.Column("macd",      sa.REAL),
        sa.Column("macd_sig",  sa.REAL),
        sa.Column("sma_20",    sa.REAL),
        sa.Column("sma_50",    sa.REAL),
        sa.Column("sma_200",   sa.REAL),
        sa.Column("atr",       sa.REAL),
        sa.Column("bb_upper",  sa.REAL),
        sa.Column("bb_lower",  sa.REAL),
        sa.Column("trend",     sa.Text),
        sa.UniqueConstraint("symbol", "timeframe", "time", name="uq_indicators_symbol_tf_time"),
    )

    # ── patterns ─────────────────────────────────────────────
    op.create_table(
        "patterns",
        sa.Column("id",        sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol",    sa.Text, nullable=False),
        sa.Column("timeframe", sa.Text, nullable=False),
        sa.Column("time",      sa.Text, nullable=False),
        sa.Column("pattern",   sa.Text),
        sa.Column("engulfing", sa.Text),
        sa.Column("star",      sa.Text),
        sa.Column("signal",    sa.Text),
    )

    # ── analysis ─────────────────────────────────────────────
    op.create_table(
        "analysis",
        sa.Column("id",           sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_time",     sa.Text, nullable=False),
        sa.Column("symbol",       sa.Text, nullable=False),
        sa.Column("timeframe",    sa.Text, nullable=False),
        sa.Column("bias_score",   sa.Integer),
        sa.Column("bias_label",   sa.Text),
        sa.Column("context_json", sa.Text),
    )

    # ── trades ───────────────────────────────────────────────
    # Includes the swap/error_message/mt5_ticket columns that
    # database/db.py added via _migrate_trades_table() over time —
    # fresh databases get them all up front.
    op.create_table(
        "trades",
        sa.Column("id",            sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("pair",          sa.Text, nullable=False),
        sa.Column("timeframe",     sa.Text),
        sa.Column("type",          sa.Text, nullable=False),
        sa.Column("entry",         sa.REAL),
        sa.Column("sl",            sa.REAL),
        sa.Column("tp",            sa.REAL),
        sa.Column("lot",           sa.REAL),
        sa.Column("confidence",    sa.Integer),
        sa.Column("open_time",     sa.Text, nullable=False),
        sa.Column("close_time",    sa.Text),
        sa.Column("exit_price",    sa.REAL),
        sa.Column("result",        sa.Text),
        sa.Column("pnl",           sa.REAL),
        sa.Column("pnl_pips",      sa.REAL),
        sa.Column("spread_cost",   sa.REAL),
        sa.Column("commission",    sa.REAL),
        sa.Column("slippage",      sa.REAL),
        sa.Column("swap",          sa.REAL, server_default="0"),
        sa.Column("error_message", sa.Text),
        sa.Column("mt5_ticket",    sa.Text),
        sa.Column("pattern",       sa.Text),
        sa.Column("regime",        sa.Text),
        sa.Column("trend",         sa.Text),
        sa.Column("rsi",           sa.REAL),
        sa.Column("session",       sa.Text),
        sa.Column("status",        sa.Text, server_default="OPEN"),
        sa.Column("context_json",  sa.Text),
    )

    # ── economic_history ─────────────────────────────────────
    op.create_table(
        "economic_history",
        sa.Column("id",               sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("event",            sa.Text, nullable=False),
        sa.Column("currency",         sa.Text, nullable=False),
        sa.Column("impact",           sa.Text),
        sa.Column("event_time",       sa.Text, nullable=False),
        sa.Column("expected",         sa.Text),
        sa.Column("actual",           sa.Text),
        sa.Column("market_reaction",  sa.Text),
        sa.Column("pips_moved",       sa.REAL),
        sa.Column("lesson",           sa.Text),
        sa.Column("created_at",       sa.Text, nullable=False),
    )

    # ── Indexes for query performance ────────────────────────
    op.create_index(
        "ix_candles_symbol_tf_time",
        "candles",
        ["symbol", "timeframe", "time"],
        unique=True,
    )
    op.create_index(
        "ix_trades_status_open_time",
        "trades",
        ["status", "open_time"],
    )
    op.create_index(
        "ix_analysis_run_time",
        "analysis",
        ["run_time"],
    )


def downgrade() -> None:
    """Drop all tables. Use with extreme caution — loses all data."""
    op.drop_index("ix_analysis_run_time", table_name="analysis")
    op.drop_index("ix_trades_status_open_time", table_name="trades")
    op.drop_index("ix_candles_symbol_tf_time", table_name="candles")
    for table in (
        "economic_history",
        "trades",
        "analysis",
        "patterns",
        "indicators",
        "candles",
    ):
        op.drop_table(table)
