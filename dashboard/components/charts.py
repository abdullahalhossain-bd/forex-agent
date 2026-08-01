# dashboard/components/charts.py  —  Day 56 | Reusable Chart Components
# ============================================================

import pandas as pd
import streamlit as st


def equity_curve_chart(curve: list, title: str = "Account Equity") -> None:
    if not curve:
        st.info("No equity data yet.")
        return
    df = pd.DataFrame(curve).set_index("step")
    st.markdown(f"**{title}**")
    st.area_chart(df["equity"])


def drawdown_chart(curve: list) -> None:
    """Equity curve থেকে running drawdown % derive করে plot করে।"""
    if not curve:
        st.info("No equity data yet.")
        return
    df = pd.DataFrame(curve)
    df["peak"] = df["equity"].cummax()
    df["drawdown_pct"] = (df["peak"] - df["equity"]) / df["peak"].replace(0, 1) * 100
    st.markdown("**Drawdown (%)**")
    st.area_chart(df.set_index("step")["drawdown_pct"])
    st.caption(f"Max drawdown so far: {df['drawdown_pct'].max():.1f}%")


def pattern_performance_chart(rows: list) -> None:
    if not rows:
        st.info("No pattern performance data yet.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(
        df.rename(columns={"key": "Pattern / Pair / TF / Regime", "win_rate": "Win Rate %", "total": "Trades"}),
        use_container_width=True, hide_index=True,
    )
    st.bar_chart(df.set_index("key")["win_rate"])


def strategy_comparison_chart(rows: list) -> None:
    if not rows:
        st.info("No backtest results yet.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(
        df.rename(columns={
            "strategy": "Strategy", "win_rate": "Win Rate %",
            "profit_factor": "Profit Factor", "max_dd": "Max DD %",
        }),
        use_container_width=True, hide_index=True,
    )
    st.bar_chart(df.set_index("strategy")[["win_rate", "profit_factor"]])


def decision_timeline_view(events: list) -> None:
    """⭐ AI Decision Timeline — chronological event list।"""
    if not events:
        st.info("No timeline events recorded yet.")
        return
    for e in events:
        st.markdown(f"**{e.get('time')}** — {e.get('event')}")
        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;│", unsafe_allow_html=True)


def profit_distribution_chart(trades: list) -> None:
    if not trades:
        st.info("No trades yet.")
        return
    df = pd.DataFrame(trades)
    if "pnl" not in df.columns:
        st.info("No P/L data available.")
        return
    st.bar_chart(df["pnl"])


def open_positions_table(positions: list) -> None:
    if not positions:
        st.info("No open positions.")
        return
    df = pd.DataFrame(positions)
    df = df.rename(columns={
        "pair": "Pair", "direction": "Direction", "entry": "Entry",
        "current": "Current", "pnl": "P/L", "lots": "Lots",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


def trade_replay_chart(trade: dict) -> None:
    """⭐ Trade Replay — single trade-এর before/after snapshot view।"""
    if not trade:
        st.info("Select a trade to replay.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before Entry**")
        st.write({
            "Pair":       trade.get("pair"),
            "Timeframe":  trade.get("timeframe"),
            "Pattern":    trade.get("pattern"),
            "Regime":     trade.get("regime"),
            "Confidence": trade.get("confidence"),
        })
    with c2:
        st.markdown("**Result**")
        outcome = "WIN" if trade.get("win") else "LOSS"
        icon = "✅" if trade.get("win") else "❌"
        st.write({
            "Outcome": f"{icon} {outcome}",
            "PnL":     trade.get("pnl"),
            "RR":      trade.get("rr"),
            "Time":    trade.get("timestamp"),
        })