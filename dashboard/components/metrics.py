# dashboard/components/metrics.py  —  Day 56 | KPI / Gauge / Risk-Meter Components
# ============================================================
# পুরো dashboard জুড়ে reuse হওয়া metric widgets।
# ============================================================

import streamlit as st


def kpi_row(metrics: dict) -> None:
    """
    Example:
        kpi_row({"Win Rate": "62%", "Total Trades": 120, "Profit Factor": 1.8, "Drawdown": "9.4%"})
    """
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def status_badge(status: str) -> str:
    status = (status or "").upper()
    if status in ("CONNECTED", "OK", "SAFE", "ENABLED", "ACTIVE"):
        return f"🟢 {status}"
    if status in ("WARNING", "CAUTION", "DEGRADED"):
        return f"🟡 {status}"
    return f"🔴 {status}"


def risk_meter(used: float, limit: float, label: str = "Daily Loss Limit") -> None:
    """
    Daily risk usage progress bar — green→yellow→red অনুযায়ী।
    """
    remaining = max(0.0, limit - used)
    pct_used = min(1.0, used / limit) if limit else 0.0

    st.markdown(f"**{label}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Limit", f"${limit:,.0f}")
    c2.metric("Used", f"${used:,.0f}")
    c3.metric("Remaining", f"${remaining:,.0f}")

    color = "normal"
    if pct_used >= 0.9:
        color = "🔴"
    elif pct_used >= 0.6:
        color = "🟡"
    else:
        color = "🟢"
    st.progress(pct_used, text=f"{color} {pct_used*100:.0f}% of daily limit used")


def risk_per_trade_gauge(current_pct: float, max_pct: float) -> None:
    """Single trade risk % vs max allowed — SAFE/WARNING/BLOCKED status।"""
    if current_pct <= max_pct:
        status, icon = "SAFE", "✅"
    elif current_pct <= max_pct * 1.25:
        status, icon = "WARNING", "⚠️"
    else:
        status, icon = "BLOCKED", "⛔"

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Risk", f"{current_pct:.2f}%")
    c2.metric("Max Allowed", f"{max_pct:.2f}%")
    c3.metric("Status", f"{icon} {status}")


def confidence_breakdown(scores: dict) -> None:
    """
    Example:
        confidence_breakdown({"Technical": 85, "SMC": 90, "Sentiment": 70,
                               "News Filter": 100, "Risk": 95})
    """
    st.markdown("**Confidence Breakdown**")
    for label, value in scores.items():
        st.progress(value / 100, text=f"{label} — {value}%")


def big_number(label: str, value, delta=None, help_text: str = None) -> None:
    st.metric(label, value, delta=delta, help=help_text)


def signal_card(signal: dict) -> None:
    """একটা single live-signal card (pair/signal/confidence/entry/sl/tp)।"""
    sig = signal.get("signal", "WAIT")
    icon = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⚪"}.get(sig, "⚪")
    with st.container(border=True):
        st.markdown(f"### {icon} {signal.get('pair')} — {sig}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confidence", f"{signal.get('confidence', 0)}%")
        c2.metric("Entry", signal.get("entry", "-"))
        c3.metric("SL", signal.get("sl", "-"))
        c4.metric("TP", signal.get("tp", "-"))