# dashboard/app.py  —  AI Trading Command Center (Full Upgrade)
# ============================================================
# Single-page Streamlit app with sidebar navigation between:
#   🏠 Home          — KPIs, system health, emergency controls
#   📡 Live Room     — current positions, entry/SL/TP, floating P/L, time open
#   🧠 AI Brain      — current analysis, confidence, market regime, active strategy
#   🛡 Risk Monitor  — balance, equity, drawdown, daily loss, circuit breaker
#   📚 Learning      — recent mistakes, lessons, strategy performance
#
# Features:
#   - Auto-refresh every 30 seconds (configurable)
#   - Emergency controls (pause/resume trading, circuit breaker status)
#   - Navigation sidebar with page links
#   - Backward-compatible with existing components imports
# ============================================================

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from datetime import datetime, timezone

from components import data_loader, metrics, alerts, charts

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Trading Command Center",
    page_icon="🧠",
)


# ══════════════════════════════════════════════════════════════
#  CRITICAL FIX: Dashboard Authentication
#  Without this, anyone with network access to the dashboard port
#  can pause/resume trading or view positions.
#  Set DASHBOARD_PASSWORD in .env to enable auth.
# ══════════════════════════════════════════════════════════════

import os as _os
import socket as _socket

_DASHBOARD_PASSWORD = _os.getenv("DASHBOARD_PASSWORD", "")
# Day 102+ CRITICAL hotfix: detect non-localhost deployment.
# When the dashboard is reachable beyond localhost, an unset
# DASHBOARD_PASSWORD is an existential risk — anyone with the URL can
# pause/resume trading, view positions, and read account balance.
# We refuse to render the dashboard in that case.
_DASHBOARD_BIND_ADDR = _os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
# Day 102+ CRITICAL hotfix: detect non-localhost deployment.
# `0.0.0.0` means "bind to ALL interfaces" — remotely reachable, NOT localhost.
# Only `localhost` / `127.0.0.1` / `::1` are safe without a password.
_IS_LOCALHOST_BIND = _DASHBOARD_BIND_ADDR in ("localhost", "127.0.0.1", "::1")

# Determine the request host's loopback status from Streamlit's request
# headers (best-effort — fall back to bind address).
try:
    _request_addr = st.context.headers.get("host", "") if hasattr(st, "context") else ""
except Exception:
    _request_addr = ""

if _DASHBOARD_PASSWORD:
    # Auth enabled — require password
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.markdown("## 🔒 Dashboard Login")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == _DASHBOARD_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Wrong password")
        st.stop()
elif not _IS_LOCALHOST_BIND:
    # Day 102+ CRITICAL hotfix: refuse to render when bound to a
    # non-localhost address without a password. Previously this branch
    # was `else: pass` — the dashboard ran fully unauthenticated.
    st.error("🚫 DASHBOARD SECURITY CONFIGURATION ERROR")
    st.markdown(
        "The dashboard is bound to a non-localhost address "
        f"(`{_DASHBOARD_BIND_ADDR}`) but `DASHBOARD_PASSWORD` is not set.\n\n"
        "**Refusing to start.** Set `DASHBOARD_PASSWORD` in your `.env` file "
        "before exposing the dashboard beyond localhost."
    )
    st.stop()
# else: localhost bind + no password → allow (dev convenience)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR — Navigation + Auto-Refresh + Emergency Controls
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("## 🧠 AI Trader")
    st.caption("Command Center v2.0")

    st.divider()

    # ── Navigation ──────────────────────────────────────────
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to page",
        [
            "🏠 Home",
            "📡 Live Trading Room",
            "🧠 AI Brain",
            "🛡 Risk Monitor",
            "📚 Learning Center",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Auto-Refresh ────────────────────────────────────────
    st.markdown("### 🔄 Live Updates")
    refresh_on = st.toggle("Auto-refresh", value=True)
    interval = st.slider("Refresh interval (sec)", 5, 120, 30)

    if refresh_on:
        if AUTOREFRESH_AVAILABLE:
            st_autorefresh(interval=interval * 1000, key="main_autorefresh")
        else:
            st.caption("Install `streamlit-autorefresh` for live polling.")

    st.divider()

    # ── Emergency Controls in Sidebar ───────────────────────
    st.markdown("### 🛑 Emergency")
    ctrl = data_loader.get_system_control()
    trading_enabled = ctrl.get("trading_enabled", True)
    status_icon = "🟢" if trading_enabled else "🔴"
    status_text = "RUNNING" if trading_enabled else "STOPPED"
    st.markdown(f"**Status:** {status_icon} {status_text}")

    c1, c2 = st.columns(2)
    if c1.button("⏸ Pause", use_container_width=True, type="primary" if trading_enabled else "secondary"):
        data_loader.set_system_control(trading_enabled=False, last_changed_by="dashboard_user")
        st.rerun()
    if c2.button("▶ Resume", use_container_width=True, type="secondary" if trading_enabled else "primary"):
        data_loader.set_system_control(trading_enabled=True, last_changed_by="dashboard_user")
        st.rerun()

    # ── Circuit Breaker Status ──────────────────────────────
    risk = data_loader.get_risk_status()
    daily_used = risk.get("daily_used", 0)
    daily_limit = risk.get("daily_limit", 300)
    cb_pct = (daily_used / daily_limit * 100) if daily_limit else 0

    if cb_pct >= 100:
        cb_status = "🔴 TRIPPED"
    elif cb_pct >= 80:
        cb_status = "🟡 WARNING"
    else:
        cb_status = "🟢 NORMAL"

    st.markdown(f"**Circuit Breaker:** {cb_status}")
    st.progress(min(1.0, cb_pct / 100), text=f"Daily loss: {cb_pct:.0f}%")

    st.divider()
    st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")


# ══════════════════════════════════════════════════════════════
#  PAGE: 🏠 HOME
# ══════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.title("🧠 AI Trading Command Center")
    st.caption("One place to see everything your AI Trader sees, learns, and decides — now with global macro context.")

    ctrl = data_loader.get_system_control()
    status_icon = "🟢" if ctrl.get("trading_enabled") else "🔴"
    st.markdown(
        f"**Status:** {status_icon} {'Trading Enabled' if ctrl.get('trading_enabled') else 'Trading STOPPED'}  "
        f"&nbsp;|&nbsp; **Mode:** `{ctrl.get('mode', 'DEMO')}`",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Today's Quick KPIs ──────────────────────────────────
    pnl = data_loader.get_todays_pnl()
    risk = data_loader.get_risk_status()

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Today's P/L", f"${pnl['pnl']:,.2f}")
    kpi_cols[1].metric("Win Rate Today", f"{pnl['win_rate']}%" if pnl["win_rate"] is not None else "—")
    kpi_cols[2].metric("Trades Today", pnl["trades"])
    kpi_cols[3].metric("Current Risk / Trade", f"{risk['current_risk_pct']}%")

    st.divider()

    # ── Global Market State ─────────────────────────────────
    global_state = data_loader.get_global_market_state()
    alerts.global_market_state_panel(global_state)

    st.divider()

    # ── System Health ───────────────────────────────────────
    health = data_loader.get_system_health()
    alerts.system_health_panel(health)

    st.divider()

    # ── Emergency Control Panel (full) ──────────────────────
    alerts.emergency_control_panel()

    st.divider()

    # ── Pending Approvals ───────────────────────────────────
    pending = data_loader.get_pending_optimizer_suggestions()
    alerts.pending_approvals_alert(pending)

    st.divider()

    # ── Equity Snapshot ─────────────────────────────────────
    st.markdown("### 📈 Equity Snapshot")
    charts.equity_curve_chart(data_loader.get_equity_curve())


# ══════════════════════════════════════════════════════════════
#  PAGE: 📡 LIVE TRADING ROOM
# ══════════════════════════════════════════════════════════════

elif page == "📡 Live Trading Room":
    st.title("📡 Live Trading Room")
    st.caption("Real-time view of active positions, signals, and floating P/L.")

    # ── Session Status Bar ──────────────────────────────────
    ctrl = data_loader.get_system_control()
    trading_icon = "🟢" if ctrl.get("trading_enabled") else "🔴"
    st.markdown(
        f"**Engine:** {trading_icon} {'ACTIVE' if ctrl.get('trading_enabled') else 'PAUSED'}  "
        f"&nbsp;|&nbsp; **Mode:** `{ctrl.get('mode', 'DEMO')}`  "
        f"&nbsp;|&nbsp; **Time:** `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Real-Time P/L Summary ───────────────────────────────
    pnl = data_loader.get_todays_pnl()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Today's P/L", f"${pnl['pnl']:,.2f}",
              delta=f"{pnl['pnl']:,.2f}" if pnl["pnl"] else None)
    c2.metric("Win Rate", f"{pnl['win_rate']}%" if pnl["win_rate"] is not None else "—")
    c3.metric("Trades Today", pnl["trades"])

    positions = data_loader.get_open_positions()
    total_floating = sum(p.get("pnl", 0) for p in positions)
    floating_icon = "📈" if total_floating >= 0 else "📉"
    c4.metric("Floating P/L", f"${total_floating:,.2f}")

    st.divider()

    # ── Current Positions (enhanced) ────────────────────────
    st.subheader("📂 Current Positions")

    if not positions:
        st.info("📭 No open positions right now.")
    else:
        for pos in positions:
            direction = pos.get("direction", "—")
            dir_icon = "🟢" if direction == "BUY" else "🔴"
            pair = pos.get("pair", "—")
            entry = pos.get("entry", "—")
            sl = pos.get("sl", "—")
            tp = pos.get("tp", "—")
            current = pos.get("current", "—")
            pos_pnl = pos.get("pnl", 0)
            lots = pos.get("lots", "—")
            time_open = pos.get("time_open", pos.get("opened_at", "—"))

            pnl_sign = "+" if pos_pnl >= 0 else ""
            pnl_color = "🟢" if pos_pnl >= 0 else "🔴"

            with st.container(border=True):
                pos_cols = st.columns([2, 1, 1, 1, 1, 1, 1])
                pos_cols[0].markdown(f"### {dir_icon} {pair}")
                pos_cols[1].metric("Entry", f"`{entry}`")
                pos_cols[2].metric("SL", f"`{sl}`")
                pos_cols[3].metric("TP", f"`{tp}`")
                pos_cols[4].metric("Current", f"`{current}`")
                pos_cols[5].metric("P/L", f"{pnl_color} {pnl_sign}${pos_pnl:,.2f}")
                pos_cols[6].metric("Time Open", time_open)

                # Mini detail row
                detail_cols = st.columns([1, 1, 1])
                detail_cols[0].caption(f"📦 Lot Size: {lots}")
                if sl != "—" and entry != "—":
                    try:
                        sl_dist = abs(float(entry) - float(sl))
                        detail_cols[1].caption(f"🛡 SL Distance: {sl_dist:.4f}")
                    except (ValueError, TypeError):
                        detail_cols[1].caption(f"🛡 SL Distance: —")
                if tp != "—" and entry != "—":
                    try:
                        tp_dist = abs(float(tp) - float(entry))
                        detail_cols[2].caption(f"🎯 TP Distance: {tp_dist:.4f}")
                    except (ValueError, TypeError):
                        detail_cols[2].caption(f"🎯 TP Distance: —")

        # ── Positions Summary Table ─────────────────────────
        st.markdown("#### 📊 Positions Summary")
        charts.open_positions_table(positions)

    st.divider()

    # ── Active Signals ──────────────────────────────────────
    st.subheader("🎯 Active Signals")
    signals = data_loader.get_live_signals()
    for sig in signals:
        metrics.signal_card(sig)

    st.divider()

    # ── Equity Movement ─────────────────────────────────────
    st.subheader("🔵 Equity Movement")
    charts.equity_curve_chart(data_loader.get_equity_curve(), title="Recent Equity Movement")


# ══════════════════════════════════════════════════════════════
#  PAGE: 🧠 AI BRAIN
# ══════════════════════════════════════════════════════════════

elif page == "🧠 AI Brain":
    st.title("🧠 AI Brain Monitor")
    st.caption("A window into what the AI is currently thinking, and why.")

    brain = data_loader.get_ai_brain_state()

    # ── Current Analysis ────────────────────────────────────
    st.subheader("🌍 Current Market Analysis")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Pair / Timeframe:** {brain.get('pair')} {brain.get('timeframe')}")
        st.markdown(f"**Market Regime:** `{brain.get('market_regime')}`")
    with c2:
        st.markdown(f"**Structure:** {brain.get('structure')}")
        st.markdown(f"**Liquidity:** {brain.get('liquidity')}")
    with c3:
        st.markdown(f"**SMC:** {brain.get('smc')}")

    # ── Active Strategy ─────────────────────────────────────
    strategy_cfg = data_loader.get_strategy_config()
    active_pairs = strategy_cfg.get("active_pairs", [])
    active_str = ", ".join(active_pairs) if active_pairs else "None"

    st.divider()
    st.subheader("⚙️ Active Strategy Configuration")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Risk Per Trade", f"{strategy_cfg.get('risk_percent', 1.0)}%")
    sc2.metric("Active Pairs", active_str)
    sc3.metric("Strategy Version", strategy_cfg.get("version", "1.0"))

    st.divider()

    # ── AI Decision ─────────────────────────────────────────
    st.subheader("💭 AI Reasoning & Decision")
    st.info(f"💬 *\"{brain.get('reasoning')}\"*")

    decision = brain.get("decision", "WAIT")
    decision_icon = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⚪", "NO TRADE": "⛔"}.get(decision, "⚪")
    confidence = brain.get("confidence", 0)

    if confidence >= 80:
        conf_icon = "🟢 HIGH"
    elif confidence >= 60:
        conf_icon = "🟡 MEDIUM"
    else:
        conf_icon = "🔴 LOW"

    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", f"{decision_icon} {decision}")
    c2.metric("Confidence", f"{confidence}%")
    c3.metric("Confidence Level", conf_icon)

    st.divider()

    # ── Confidence Breakdown ────────────────────────────────
    st.subheader("📊 Confidence Breakdown")
    metrics.confidence_breakdown(brain.get("confidence_breakdown", {}))

    st.divider()

    # ── Decision Timeline ───────────────────────────────────
    st.subheader("🕒 AI Decision Timeline")
    st.caption("Step-by-step trace of how the AI arrived at its latest decision.")
    events = data_loader.get_decision_timeline()
    charts.decision_timeline_view(events)

    st.divider()

    # ── Memory Context ──────────────────────────────────────
    st.subheader("🧩 Memory Context Used in This Decision")
    st.caption(
        "If MemoryIntegration / DeepMistakeAnalyzer flagged a relevant past lesson, "
        "it would be shown here before the trade is taken."
    )
    recent_mistakes = data_loader.get_recent_mistakes(limit=3)
    if recent_mistakes:
        for m in recent_mistakes:
            st.markdown(f"- **{m['market']}** — {m['reason']} → _{m['lesson']}_")
    else:
        st.write("No relevant memory flags for the current setup.")


# ══════════════════════════════════════════════════════════════
#  PAGE: 🛡 RISK MONITOR
# ══════════════════════════════════════════════════════════════

elif page == "🛡 Risk Monitor":
    st.title("🛡 Risk Monitor")
    st.caption("The safety net — balance, equity, drawdown, daily loss limits, and circuit breaker status.")

    risk = data_loader.get_risk_status()

    # ── Account Overview ────────────────────────────────────
    st.subheader("💰 Account Overview")
    curve = data_loader.get_equity_curve()
    if curve:
        current_equity = curve[-1].get("equity", 10000)
        peak_equity = max(e.get("equity", 0) for e in curve)
        current_dd = ((peak_equity - current_equity) / peak_equity * 100) if peak_equity else 0
        balance = current_equity
    else:
        balance = 10000.0
        current_equity = 10000.0
        peak_equity = 10000.0
        current_dd = 0.0

    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Balance", f"${balance:,.2f}")
    ac2.metric("Equity", f"${current_equity:,.2f}")
    ac3.metric("Peak Equity", f"${peak_equity:,.2f}")
    ac4.metric("Current Drawdown", f"{current_dd:.2f}%")

    st.divider()

    # ── Circuit Breaker Status ──────────────────────────────
    st.subheader("⚡ Circuit Breaker Status")

    daily_used = risk.get("daily_used", 0)
    daily_limit = risk.get("daily_limit", 300)
    dd_pct = current_dd
    max_dd = risk.get("max_allowed_pct", 1.0) * 10  # approximate

    # Determine circuit breaker state
    cb_daily_tripped = daily_used >= daily_limit
    cb_dd_tripped = dd_pct >= max_dd
    cb_any_tripped = cb_daily_tripped or cb_dd_tripped

    if cb_any_tripped:
        cb_state = "🔴 TRIPPED — TRADING HALTED"
    elif daily_used >= daily_limit * 0.8 or dd_pct >= max_dd * 0.8:
        cb_state = "🟡 WARNING — APPROACHING LIMIT"
    else:
        cb_state = "🟢 NORMAL — ALL CLEAR"

    st.markdown(f"### {cb_state}")

    cb1, cb2 = st.columns(2)
    with cb1:
        st.markdown("#### 📉 Daily Loss Circuit")
        daily_pct = (daily_used / daily_limit * 100) if daily_limit else 0
        st.progress(min(1.0, daily_pct / 100), text=f"{daily_pct:.0f}% of daily limit")
        st.caption(f"Used: ${daily_used:,.2f} / Limit: ${daily_limit:,.2f}")
        if cb_daily_tripped:
            st.error("⛔ Daily loss limit REACHED — trading should be paused.")

    with cb2:
        st.markdown("#### 📉 Drawdown Circuit")
        dd_ratio = (dd_pct / max_dd) if max_dd else 0
        st.progress(min(1.0, dd_ratio), text=f"{dd_pct:.2f}% / {max_dd:.2f}% max")
        st.caption(f"Current DD: {dd_pct:.2f}% / Max Allowed: {max_dd:.2f}%")
        if cb_dd_tripped:
            st.error("⛔ Max drawdown EXCEEDED — trading should be paused.")

    st.divider()

    # ── Daily Risk Meter ────────────────────────────────────
    st.subheader("💰 Daily Risk Meter")
    metrics.risk_meter(used=daily_used, limit=daily_limit)

    if daily_used >= daily_limit * 0.9:
        alerts.alert_banner("ERROR", "⛔ Approaching daily loss limit — consider stopping trading for today.")
    elif daily_used >= daily_limit * 0.6:
        alerts.alert_banner("WARNING", "⚠️ More than 60% of today's loss limit used — trade carefully.")

    st.divider()

    # ── Drawdown Chart ──────────────────────────────────────
    st.subheader("📉 Drawdown Chart")
    charts.equity_curve_chart(curve, title="Account Equity")
    charts.drawdown_chart(curve)

    st.divider()

    # ── Risk Per Trade ──────────────────────────────────────
    st.subheader("⚖️ Risk Per Trade")
    metrics.risk_per_trade_gauge(
        current_pct=risk["current_risk_pct"],
        max_pct=risk["max_allowed_pct"],
    )

    st.caption(
        "Risk % is read live from `memory/strategy_config.json`, which the "
        "AutoOptimizer adjusts automatically based on volatility and drawdown."
    )


# ══════════════════════════════════════════════════════════════
#  PAGE: 📚 LEARNING CENTER
# ══════════════════════════════════════════════════════════════

elif page == "📚 Learning Center":
    st.title("📚 Learning Center")
    st.caption("Everything the AI has learned from its own trading history — mistakes, lessons, and strategy performance.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Recent Mistakes", "📊 Strategy Performance", "✅ Learned Rules", "🎬 Trade Replay"]
    )

    # ── Recent Mistakes ─────────────────────────────────────
    with tab1:
        st.subheader("Recent Mistakes & Lessons")
        mistakes = data_loader.get_recent_mistakes(limit=10)
        if not mistakes:
            st.info("No recorded losses yet — nothing to review. 🎉")
        for m in mistakes:
            with st.container(border=True):
                mistake_cols = st.columns([2, 1, 2])
                mistake_cols[0].markdown(f"**❌ Reason:** {m['reason']}")
                mistake_cols[1].markdown(f"**Market:** `{m['market']}`")
                mistake_cols[2].markdown(f"**💡 Lesson:** {m['lesson']}")

    # ── Strategy / Pattern Performance ──────────────────────
    with tab2:
        st.subheader("Strategy & Pattern Performance")

        # Pattern performance
        st.markdown("#### 🎯 Pattern Performance")
        rows = data_loader.get_pattern_performance()
        charts.pattern_performance_chart(rows)

        st.divider()

        # Strategy comparison
        st.markdown("#### ⚖️ Strategy Comparison")
        charts.strategy_comparison_chart(data_loader.get_backtest_results())

    # ── AI Learned Rules ────────────────────────────────────
    with tab3:
        st.subheader("AI Learned Rules")
        rules = data_loader.get_learned_rules()
        if not rules:
            st.info("No active rules yet.")
        for r in rules:
            with st.container(border=True):
                st.markdown(f"✅ {r['summary']}")
                if r.get("lesson"):
                    st.caption(f"💡 {r['lesson']}")

    # ── Trade Replay ────────────────────────────────────────
    with tab4:
        st.subheader("🎬 Trade Replay")
        st.caption("Step back through a past trade exactly as the AI saw it.")

        trades = data_loader.get_trade_replay_list(limit=20)
        if not trades:
            st.info("No trade history yet.")
        else:
            labels = [
                f"{t.get('timestamp', '')[:19]} — {t.get('pair')} "
                f"({'WIN' if t.get('win') else 'LOSS'}, {t.get('pnl')})"
                for t in trades
            ]
            idx = st.selectbox(
                "Select a trade to replay",
                range(len(trades)),
                format_func=lambda i: labels[i],
            )
            charts.trade_replay_chart(trades[idx])
