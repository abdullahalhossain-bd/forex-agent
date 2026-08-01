# dashboard/pages/2_ai_brain.py  —  Day 56 | AI Brain Monitor ⭐⭐⭐⭐⭐
# ============================================================
# এখানে দেখা যায় AI কী চিন্তা করছে: market understanding, reasoning,
# decision, confidence breakdown, এবং (⭐ Bonus 1) Decision Timeline।
# ============================================================

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from components import data_loader, metrics, charts

st.set_page_config(page_title="AI Brain Monitor", page_icon="🧠", layout="wide")

st.title("🧠 AI Brain Monitor")
st.caption("A window into what the AI is currently thinking, and why.")

brain = data_loader.get_ai_brain_state()

# ── Current Market Understanding ─────────────────────────────
st.subheader("🌍 Current Market Understanding")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Pair / Timeframe:** {brain.get('pair')} {brain.get('timeframe')}")
    st.markdown(f"**Market Regime:** `{brain.get('market_regime')}`")
    st.markdown(f"**Structure:** {brain.get('structure')}")
with c2:
    st.markdown(f"**Liquidity:** {brain.get('liquidity')}")
    st.markdown(f"**SMC:** {brain.get('smc')}")

st.divider()

# ── AI Reasoning ──────────────────────────────────────────────
st.subheader("💭 AI Reasoning")
st.info(f"\"{brain.get('reasoning')}\"")

decision = brain.get("decision", "WAIT")
decision_icon = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⚪", "NO TRADE": "⛔"}.get(decision, "⚪")
c1, c2 = st.columns(2)
c1.metric("Decision", f"{decision_icon} {decision}")
c2.metric("Confidence", f"{brain.get('confidence', 0)}%")

st.divider()

# ── Confidence Breakdown ─────────────────────────────────────
st.subheader("📊 Confidence Breakdown")
metrics.confidence_breakdown(brain.get("confidence_breakdown", {}))

st.divider()

# ── Decision Timeline  (⭐ Bonus 1) ───────────────────────────
st.subheader("🕒 AI Decision Timeline")
st.caption("Step-by-step trace of how the AI arrived at its latest decision — useful for debugging.")
events = data_loader.get_decision_timeline()
charts.decision_timeline_view(events)

st.divider()

# ── Memory context injection (Day 52 bridge) ─────────────────
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