# dashboard/pages/4_strategy_lab.py  —  Day 56 | Strategy Lab ⭐⭐⭐⭐⭐
# ============================================================
# AI-এর research room: Backtest Results, Strategy Comparison,
# Parameter Optimization (Day 55 AutoOptimizer থেকে), এবং A/B Testing।
# ============================================================

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from components import data_loader, charts

st.set_page_config(page_title="Strategy Lab", page_icon="🧪", layout="wide")

st.title("🧪 Strategy Lab")
st.caption("Where the AI's strategies get tested, compared, and tuned.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Backtest Results", "⚖️ Strategy Comparison", "🎛 Parameter Optimization", "🧬 A/B Testing & Approvals"]
)

# ── Backtest Results ───────────────────────────────────────────
with tab1:
    st.subheader("Backtest Results")
    results = data_loader.get_backtest_results()
    for r in results:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Strategy", r["strategy"])
            c2.metric("Win Rate", f"{r['win_rate']}%")
            c3.metric("Profit Factor", r["profit_factor"])
            c4.metric("Max DD", f"{r['max_dd']}%")

# ── Strategy Comparison ────────────────────────────────────────
with tab2:
    st.subheader("Strategy Comparison")
    charts.strategy_comparison_chart(data_loader.get_backtest_results())

# ── Parameter Optimization (Day 55 bridge) ────────────────────
with tab3:
    st.subheader("Parameter Optimization")
    cfg = data_loader.get_strategy_config()

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk", f"{cfg.get('risk_percent', 1.0)}%")

    session_pref = cfg.get("session_preference", {})
    best_session = next(iter(session_pref.values()), {}).get("preferred_session", "—") if session_pref else "—"
    c2.metric("Best Session", best_session)
    c3.metric("Active Pairs", len(cfg.get("active_pairs", [])))

    st.markdown("**Active pairs:** " + ", ".join(cfg.get("active_pairs", [])) or "None")

    if cfg.get("disabled_pairs"):
        st.markdown("**Disabled pairs:**")
        for p, d in cfg["disabled_pairs"].items():
            st.markdown(f"- ⛔ {p}: {d.get('reason')}")

    st.divider()
    st.markdown("**Strategy Version History**")
    versions = data_loader.get_strategy_versions()
    if not versions:
        st.info("No version history yet — AutoOptimizer hasn't saved a version.")
    else:
        for v in versions[::-1]:
            with st.expander(f"v{v.get('label')} — {v.get('notes', '')}"):
                snap = v.get("performance_snapshot", {})
                st.write(snap if snap else "No performance snapshot recorded.")

# ── A/B Testing + Pending Approvals (Day 55 bridge) ───────────
with tab4:
    st.subheader("Pending Optimizer Suggestions")
    pending = data_loader.get_pending_optimizer_suggestions()
    if not pending:
        st.success("✅ No pending suggestions — nothing awaiting your approval.")
    else:
        for s in pending:
            with st.container(border=True):
                st.markdown(f"**[{s.get('type')}] → {s.get('target')}**")
                st.markdown(f"_Why:_ {s.get('reason')}")
                c1, c2 = st.columns(2)
                c1.button(f"✅ Approve {s.get('id')}", key=f"approve_{s.get('id')}")
                c2.button(f"❌ Reject {s.get('id')}", key=f"reject_{s.get('id')}")
                st.caption(
                    "Wire these buttons to `AutoOptimizer.approve()` / `.reject()` "
                    "from learning/auto_optimizer.py in your engine process."
                )

    st.divider()
    st.subheader("A/B Strategy Test")
    st.caption("Compares 'no filter' vs 'filtered' performance for a pattern+regime combo (Day 52 engine).")
    c1, c2 = st.columns(2)
    pattern = c1.text_input("Pattern", value="Bullish Engulfing")
    regime = c2.text_input("Regime to exclude", value="RANGING")
    if st.button("Run A/B Test"):
        try:
            from learning.deep_analyzer import DeepMistakeAnalyzer
            result = DeepMistakeAnalyzer().run_ab_test(pattern, regime)
            st.json(result)
        except Exception as e:
            st.error(f"Could not run live A/B test ({e}). Showing demo result instead.")
            st.json({
                "pattern": pattern, "filter": f"Exclude {regime}",
                "strategy_a": {"win_rate": 48, "trades": 80},
                "strategy_b": {"win_rate": 64, "trades": 55},
                "verdict": "✅ Filter HELPS — win rate improves by +16%",
            })