# dashboard/pages/3_learning_center.py  —  Day 56 | Learning Center
# ============================================================
# Day 52-55 intelligence visualize করে: Recent Mistakes, Pattern
# Performance, AI Learned Rules, এবং (⭐ Bonus 2) Trade Replay System।
# ============================================================

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from components import data_loader, charts

st.set_page_config(page_title="Learning Center", page_icon="📚", layout="wide")

st.title("📚 Learning Center")
st.caption("Everything the AI has learned from its own trading history (Day 52–55 intelligence).")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Recent Mistakes", "📊 Pattern Performance", "✅ Learned Rules", "🎬 Trade Replay"]
)

# ── Recent Mistakes ───────────────────────────────────────────
with tab1:
    st.subheader("Recent Mistakes")
    mistakes = data_loader.get_recent_mistakes(limit=10)
    if not mistakes:
        st.info("No recorded losses yet — nothing to review.")
    for m in mistakes:
        with st.container(border=True):
            st.markdown(f"**Reason:** {m['reason']}")
            st.markdown(f"**Market:** `{m['market']}`")
            st.markdown(f"**Lesson:** {m['lesson']}")

# ── Pattern Performance ───────────────────────────────────────
with tab2:
    st.subheader("Pattern Performance")
    rows = data_loader.get_pattern_performance()
    charts.pattern_performance_chart(rows)

# ── AI Learned Rules ──────────────────────────────────────────
with tab3:
    st.subheader("AI Learned Rules")
    rules = data_loader.get_learned_rules()
    if not rules:
        st.info("No active rules yet.")
    for r in rules:
        st.markdown(f"✓ {r['summary']}")

# ── Trade Replay System  (⭐ Bonus 2) ──────────────────────────
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
        idx = st.selectbox("Select a trade to replay", range(len(trades)), format_func=lambda i: labels[i])
        charts.trade_replay_chart(trades[idx])