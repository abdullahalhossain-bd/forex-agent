# dashboard/pages/1_live_room.py  —  Day 56 | Live Trading Room
# ============================================================
# Purpose: বর্তমান market এবং trade দেখতে — Active Signals,
# Open Positions, Real-Time P&L।
# ============================================================

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from components import data_loader, metrics, charts

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

st.set_page_config(page_title="Live Trading Room", page_icon="📡", layout="wide")

with st.sidebar:
    refresh_on = st.toggle("Auto-refresh", value=True, key="live_refresh")
    interval = st.slider("Refresh interval (sec)", 5, 60, 10, key="live_interval")

if refresh_on and AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=interval * 1000, key="live_room_autorefresh")

st.title("📡 Live Trading Room")

# ── Real-Time P&L ────────────────────────────────────────────
pnl = data_loader.get_todays_pnl()
c1, c2, c3 = st.columns(3)
c1.metric("Today's P/L", f"${pnl['pnl']:,.2f}")
c2.metric("Win Rate", f"{pnl['win_rate']}%" if pnl["win_rate"] is not None else "—")
c3.metric("Trades Today", pnl["trades"])

st.divider()

# ── Active Signals ───────────────────────────────────────────
st.subheader("🎯 Active Signals")
signals = data_loader.get_live_signals()
for sig in signals:
    metrics.signal_card(sig)

st.divider()

# ── Open Positions ───────────────────────────────────────────
st.subheader("📂 Open Positions")
positions = data_loader.get_open_positions()
charts.open_positions_table(positions)

if positions:
    total_pnl = sum(p.get("pnl", 0) for p in positions)
    st.metric("Floating P/L", f"${total_pnl:,.2f}")

st.divider()

# ── Market overview placeholder ─────────────────────────────
st.subheader("🔵 Live Market Overview")
st.caption("Replace with real tick/candle feed once connected to MT5.")
charts.equity_curve_chart(data_loader.get_equity_curve(), title="Recent Equity Movement")