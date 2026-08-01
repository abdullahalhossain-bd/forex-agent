# dashboard/pages/5_risk_monitor.py  —  Day 56 | Risk Monitor
# ============================================================
# সবচেয়ে গুরুত্বপূর্ণ safety page — Daily Risk Meter, Drawdown Chart,
# Risk Per Trade status।
# ============================================================

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from components import data_loader, metrics, charts, alerts

st.set_page_config(page_title="Risk Monitor", page_icon="🛡", layout="wide")

st.title("🛡 Risk Monitor")
st.caption("The safety net — daily loss limits, drawdown, and per-trade risk discipline.")

risk = data_loader.get_risk_status()

# ── Daily Risk Meter ──────────────────────────────────────────
st.subheader("💰 Daily Risk Meter")
metrics.risk_meter(used=risk["daily_used"], limit=risk["daily_limit"])

if risk["daily_used"] >= risk["daily_limit"] * 0.9:
    alerts.alert_banner("ERROR", "⛔ Approaching daily loss limit — consider stopping trading for today.")
elif risk["daily_used"] >= risk["daily_limit"] * 0.6:
    alerts.alert_banner("WARNING", "⚠️ More than 60% of today's loss limit used — trade carefully.")

st.divider()

# ── Drawdown Chart ────────────────────────────────────────────
st.subheader("📉 Drawdown Chart")
curve = data_loader.get_equity_curve()
charts.equity_curve_chart(curve, title="Account Equity")
charts.drawdown_chart(curve)

st.divider()

# ── Risk Per Trade ────────────────────────────────────────────
st.subheader("⚖️ Risk Per Trade")
metrics.risk_per_trade_gauge(
    current_pct=risk["current_risk_pct"],
    max_pct=risk["max_allowed_pct"],
)

st.caption(
    "Risk % is read live from `memory/strategy_config.json`, which Day 55's "
    "AutoOptimizer adjusts automatically based on volatility and drawdown."
)