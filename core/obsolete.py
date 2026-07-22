"""
core/obsolete.py — Explicit registry of obsolete / orphan modules
==================================================================

The repository audit identified ~30 modules that exist on disk but are not
wired into the live runtime. Some are dead-by-design (superseded by a
newer implementation), some are dead-by-omission (the orchestrator stack
references sub-modules that were never created), and some are dead-by-
duplication (parallel implementations of the same class).

Per the project's "no silent orphan modules" rule, every such module is
listed here with:
  * the module path
  * a category (DEAD / SUPERSEDED / BROKEN / DUPLICATE / SMOKE_ONLY)
  * a one-line justification
  * the recommended action

Runtime code is expected to import `OBSOLETE_MODULES` and surface it in
health reports so operators can see exactly what is intentionally not
wired in.

CORRECTION (execution-parity audit §6.6, 2026-07-19): several entries
below describe `ict_amd_signal_engine.py` / `unified_signal_engine.py` as
having "superseded" the deleted Day-61/62 SMC/liquidity modules. That is
true only in the narrow sense that they replaced those specific modules'
*functionality* — it does NOT mean either module is the live decision
pipeline. `agents/analysis_agent.py` never imports them at module scope;
`UnifiedSignalEngine` is wired in as a single, lazily-imported,
try/except-wrapped, informational vote (weight 1.0) among ~29 other
analysis modules that feed `strategy.signal_engine.SignalEngine` /
`core.master_decision`, which is what actually sets `final_signal` live.
Do not read "superseded by X" below as "X is now the live pipeline."
The live pipeline is `core.trader.AITrader.evaluate_decision_core()`
(AnalysisAgent -> DecisionAgent -> RiskEngine -> PositionSizer), which as
of the same audit fix is also what `backtest/unified_engine.py` replays
bar-by-bar — see that module's docstring for the shared-kernel design.

This file is the single source of truth — do not duplicate this list
elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class ObsoleteCategory(str, Enum):
    DEAD = "dead"                  # zero importers, no replacement in flight
    SUPERSEDED = "superseded"      # replaced by a newer module
    BROKEN = "broken"              # imports something that does not exist
    DUPLICATE = "duplicate"        # byte-identical or near-identical twin
    SMOKE_ONLY = "smoke_only"      # only imported by tests/test_pipeline.py
    LEGACY_STUB = "legacy_stub"    # placeholder / 0-byte file
    WIRED = "wired"                # was dead, now wired into live pipeline (audit fix)


@dataclass(frozen=True)
class ObsoleteEntry:
    path: str
    category: ObsoleteCategory
    reason: str
    action: str


OBSOLETE_MODULES: List[ObsoleteEntry] = [
    # ── agents/ ──────────────────────────────────────────────────────
    ObsoleteEntry(
        "agents/chart_agent.py",
        ObsoleteCategory.SUPERSEDED,
        "Standalone Playwright S/R drawer; superseded by computer_use/chart_drawer.py + "
        "coordinate_mapper.py. CORRECTED (2026-07-22): a duplicate entry further down "
        "in this file claimed '220-line chart agent...Archived to .dead_code_archived "
        "in Round-22' — VERIFIED FALSE via `ls agents/chart_agent.py`, file is still "
        "present at its original path. Verified via grep: zero external importers.",
        "Delete or move to legacy/. Wired runtime uses computer_use/ stack. "
        "File is NOT archived — still on disk.",
    ),

    # ── ai/ ──────────────────────────────────────────────────────────
    ObsoleteEntry(
        "ai/automated_retraining.py",
        ObsoleteCategory.DEAD,
        "Zero importers; no CLI entry point. Depends on ai/model_versioning which has an mlflow init bug.",
        "Kept on disk; wired via core/runtime.py boot_ai() with safe try/except. Marked legacy.",
    ),
    ObsoleteEntry(
        "ai/model_versioning.py",
        ObsoleteCategory.SMOKE_ONLY,
        "MLflow init runs at import time even when mlflow is absent → NameError. Only consumed by automated_retraining + core/monitoring_system.",
        "Patched: mlflow calls now guarded behind MLFLOW_AVAILABLE flag.",
    ),

    # ── analysis/ — the Day 61–64 SMC/Liquidity/Currency cluster ─────
    # UPDATE 2026-07-02: 7 of these modules were DELETED in the duplicate-cleanup pass.
    # They are kept in this registry as historical record (category=DELETED).
    ObsoleteEntry(
        "analysis/smart_money.py",
        ObsoleteCategory.DEAD,
        "Day-61 master SMC orchestrator never wired into agents/analysis_agent.py. Superseded by ict_amd_signal_engine.py + unified_signal_engine.py stack. DELETED 2026-07-02.",
        "DELETED — superseded by ict_amd_signal_engine.py.",
    ),
    ObsoleteEntry(
        "analysis/structure.py",
        ObsoleteCategory.SUPERSEDED,
        "CORRECTED 2026-07-02: NOT dead. Live — imported by agents/analysis_agent.py:41 (MarketStructureEngine) and analysis/structure_mtf.py. Was incorrectly listed as DEAD.",
        "LIVE module — keep. Used by analysis_agent + structure_mtf.",
    ),
    ObsoleteEntry(
        "analysis/liquidity.py",
        ObsoleteCategory.WIRED,
        "CORRECTED (2026-07-22, verified via grep + reachability check): the "
        "'DELETED 2026-07-02 / only consumer was dead smart_money.py' claim "
        "below was false on two counts — the file was never deleted (still "
        "on disk at this path), and it has a live, current consumer: "
        "agents/analysis_agent.py:337 does `from analysis.liquidity import "
        "LiquidityEngine`. agents/analysis_agent.py is reachable from main.py "
        "(part of the live AnalysisAgent -> DecisionAgent -> RiskEngine "
        "pipeline), so this import executes on every live decision cycle.",
        "ACTIVE — do not delete. Live consumer: agents/analysis_agent.py:337.",
    ),
    ObsoleteEntry(
        "analysis/liquidity_engine.py",
        ObsoleteCategory.DEAD,
        "Day-62 liquidity orchestrator never wired into AnalysisAgent. DELETED 2026-07-02.",
        "DELETED — superseded by unified_signal_engine.py.",
    ),
    ObsoleteEntry(
        "analysis/liquidity_zones.py",
        ObsoleteCategory.DEAD,
        "Only consumer was dead liquidity_engine.py. DELETED 2026-07-02.",
        "DELETED — superseded by stop_hunt_signal_engine.py.",
    ),
    ObsoleteEntry(
        "analysis/stop_hunt_detector.py",
        ObsoleteCategory.DEAD,
        "Only consumer was dead liquidity_engine.py. DELETED 2026-07-02.",
        "DELETED — superseded by stop_hunt_signal_engine.py.",
    ),
    ObsoleteEntry(
        "analysis/session_analysis.py",
        ObsoleteCategory.DEAD,
        "London-manipulation detector; only consumer was dead liquidity_engine.py. DELETED 2026-07-02.",
        "DELETED — superseded by session_analyzer.py (live) + ict_amd_signal_engine.py.",
    ),
    ObsoleteEntry(
        "analysis/amd_strategy.py",
        ObsoleteCategory.DEAD,
        "Day-36/37 AMD strategy. Superseded by ict_amd_signal_engine.py (stricter spec: 6-step pipeline, 1:6 R:R). DELETED 2026-07-02.",
        "DELETED — superseded by ict_amd_signal_engine.py.",
    ),
    ObsoleteEntry(
        "analysis/currency_strength.py",
        ObsoleteCategory.DEAD,
        "Day-64 currency-strength orchestrator never wired into AnalysisAgent.",
        "Marked legacy. Wire via AnalysisAgent when adopting Day-64 pipeline.",
    ),
    ObsoleteEntry(
        "analysis/currency_ranker.py",
        ObsoleteCategory.DEAD,
        "Only consumer is dead currency_strength.py.",
        "Marked legacy (transitively dead).",
    ),
    ObsoleteEntry(
        "analysis/strength_calculator.py",
        ObsoleteCategory.DEAD,
        "Only consumer is dead currency_strength.py.",
        "Marked legacy (transitively dead).",
    ),
    ObsoleteEntry(
        "analysis/mtf_analyzer.py",
        ObsoleteCategory.LEGACY_STUB,
        "Only _detect_bos/_detect_choch/_detect_liquidity_sweep helpers are used by smc_engine. Public analyze() pipeline (~700 LOC) never invoked.",
        "Kept (helpers are live). Public analyze() marked dormant.",
    ),
    ObsoleteEntry(
        "analysis/database/__init__.py",
        ObsoleteCategory.LEGACY_STUB,
        "Empty placeholder subpackage; no concrete DB modules.",
        "Kept as marker; will populate if analysis ever needs its own DB.",
    ),

    # ── broker/ — the dead MT5 data cluster + a few smoke-only guards
    ObsoleteEntry(
        "broker/market_data_manager.py",
        ObsoleteCategory.DEAD,
        "Zero importers. Was meant to be the single MT5 data entry point. "
        "CORRECTED (2026-07-22): a duplicate entry further down in this file "
        "claimed '114 lines...Archived to .dead_code_archived in Round-28' — "
        "VERIFIED FALSE via `ls broker/market_data_manager.py`, file is still "
        "present at its original path. Verified via grep: only comment/docstring "
        "mentions in broker/__init__.py and core/trader.py, no real import.",
        "Marked legacy. If reviving MT5 data path, wire into server/signal_pipeline.py. "
        "File is NOT archived — still on disk.",
    ),
    ObsoleteEntry(
        "broker/mt5_data.py",
        ObsoleteCategory.DEAD,
        "Only consumers are dead broker/market_data_manager.py + broker/symbol_manager.py.",
        "Marked legacy (transitively dead).",
    ),
    ObsoleteEntry(
        "broker/symbol_manager.py",
        ObsoleteCategory.DEAD,
        "Only consumer is dead broker/market_data_manager.py.",
        "Marked legacy (transitively dead).",
    ),
    ObsoleteEntry(
        "broker/data_validator.py",
        ObsoleteCategory.DEAD,
        "Broker-side validator; only consumer is dead broker/market_data_manager.py. (data/validator.py is the live one.)",
        "Marked legacy (transitively dead).",
    ),
    # NOTE (2026-07-22): a duplicate entry for broker/position_manager.py
    # used to sit here claiming DEAD/"zero importers". VERIFIED FALSE and
    # REMOVED — core/trader.py:261 does `from broker.position_manager
    # import PositionManager` and actively calls .register_open() /
    # .poll_once() for live MT5 trade management (trailing stop, breakeven,
    # partial close). This was a real-money-risk mislabeling: the stale
    # DEAD entry sat earlier in this file than the correct WIRED entry
    # below (Round-22 fix), so a reader scanning top-to-bottom could act
    # on the wrong one. See the WIRED entry for broker/position_manager.py
    # further down — that one is accurate and is the sole entry now.
    ObsoleteEntry(
        "broker/safety_guard.py",
        ObsoleteCategory.SMOKE_ONLY,
        "Only imported by tests/test_pipeline.py. Production safety gate is in core/trader.py (TradePermission + CorrelationFilter).",
        "Kept for tests. Marked smoke-only.",
    ),
    ObsoleteEntry(
        "broker/spread_monitor.py",
        ObsoleteCategory.SMOKE_ONLY,
        "Only imported by tests/test_pipeline.py.",
        "Kept for tests. Marked smoke-only.",
    ),
    ObsoleteEntry(
        "broker/health_monitor.py",
        ObsoleteCategory.SMOKE_ONLY,
        "Constructed by execution_router but check_once()/run_loop() never called. Canonical replacement is core/health_monitor.py.",
        "Kept; MT5-mode callbacks still wired. Marked smoke-only.",
    ),

    # ── risk/ ────────────────────────────────────────────────────────
    # NOTE (2026-07-22): duplicate risk/risk_simulator.py entry removed from
    # here — merged into the corrected entry in the Round-27 section below
    # (same DEAD conclusion; that entry also documents the false "archived"
    # claim found there and adds the overlap note with risk/monte_carlo.py).
    ObsoleteEntry(
        "risk/portfolio_manager.py",
        ObsoleteCategory.SUPERSEDED,
        "Pre-Day-58 portfolio prototype; superseded by risk/capital_manager.py + "
        "risk/exposure_manager.py. Module-level singleton runs at import time. "
        "CORRECTED (2026-07-22): a duplicate entry further down in this file "
        "claimed this was 'Archived to .dead_code_archived in Round-27' — "
        "VERIFIED FALSE via `ls risk/portfolio_manager.py`, file is still "
        "present at its original path. Verified via grep: zero external "
        "importers (only the module-level `portfolio_manager = "
        "PortfolioManager()` self-instantiation).",
        "Marked legacy. Do not import. File is NOT archived — still on disk.",
    ),

    # ── scanner/ ─────────────────────────────────────────────────────
    ObsoleteEntry(
        "scanner/scanner.py",
        ObsoleteCategory.DUPLICATE,
        "Byte-identical duplicate of scanner/config.py minus header. Zero importers.",
        "Marked legacy. Delete on next cleanup pass.",
    ),

    # ── fundamental/ ─────────────────────────────────────────────────
    ObsoleteEntry(
        "fundamental/fundamental_sentiment.py",
        ObsoleteCategory.DEAD,
        "FundamentalSentimentScore never imported. DB methods (get_currency_fundamental_bias) exist for its benefit only.",
        "Marked legacy. Wire via AnalysisAgent if reviving fundamental scoring.",
    ),

    # ── memory/ ──────────────────────────────────────────────────────
    ObsoleteEntry(
        "memory/trade_context.py",
        ObsoleteCategory.LEGACY_STUB,
        "0-byte placeholder.",
        "Kept as marker; populate if memory needs a typed trade-context dataclass.",
    ),
    ObsoleteEntry(
        "memory/confidence_calibrator.py",
        ObsoleteCategory.SUPERSEDED,
        "Superseded by hybrid/confidence_calibrator.py (also currently dead). Class-name collision.",
        "Marked legacy. Pick one canonical calibrator on next cleanup.",
    ),

    # ── learning/ ────────────────────────────────────────────────────
    ObsoleteEntry(
        "learning/weekly_review.py",
        ObsoleteCategory.DEAD,
        "run_weekly_review() never invoked.",
        "Wired via core/runtime.py boot_learning() — invoked on Sundays by DailyRoutineManager.",
    ),
    ObsoleteEntry(
        "learning/memory_integration.py",
        ObsoleteCategory.DEAD,
        "MemoryIntegration never instantiated.",
        "Wired via core/runtime.py boot_learning() and exposed to AITrader through registry.",
    ),

    # ── automation/ — entire folder dead, now wired via runtime ─────
    ObsoleteEntry(
        "automation/error_handler.py",
        ObsoleteCategory.DEAD,
        "ErrorHandler never instantiated. Canonical replacement: core/event_bus + core/runtime_metrics.record_error.",
        "Wired via core/runtime.py boot_automation().",
    ),
    ObsoleteEntry(
        "automation/runtime_metrics.py",
        ObsoleteCategory.SUPERSEDED,
        "Superseded by core/runtime_metrics.py (canonical).",
        "Kept for backward compat; core/runtime_metrics is the live one.",
    ),
    ObsoleteEntry(
        "automation/daily_review.py",
        ObsoleteCategory.DEAD,
        "DailyReview never invoked.",
        "Wired via core/runtime.py boot_automation() + DailyRoutineManager.",
    ),
    ObsoleteEntry(
        "automation/system_health.py",
        ObsoleteCategory.SUPERSEDED,
        "Superseded by core/health_monitor.py (canonical).",
        "Kept for backward compat; core/health_monitor is the live one.",
    ),

    # ── orchestrator/ — broken + dead cluster ────────────────────────
    ObsoleteEntry(
        "orchestrator/trading_orchestrator.py",
        ObsoleteCategory.BROKEN,
        "Imports 4 missing sub-modules: safety_controller, self_healing, mode_manager, decision_journal.",
        "Patched: 4 stub modules created. TradingOrchestrator now importable and wired via core/runtime.py boot_orchestrator().",
    ),
    ObsoleteEntry(
        "orchestrator/safety_controller.py",
        ObsoleteCategory.LEGACY_STUB,
        "Created as minimal stub to unblock trading_orchestrator import.",
        "Live stub — extends SafetyController if logic is added.",
    ),
    ObsoleteEntry(
        "orchestrator/self_healing.py",
        ObsoleteCategory.LEGACY_STUB,
        "Created as minimal stub to unblock trading_orchestrator import.",
        "Live stub — extends SelfHealingSystem if logic is added.",
    ),
    ObsoleteEntry(
        "orchestrator/mode_manager.py",
        ObsoleteCategory.LEGACY_STUB,
        "Created as minimal stub to unblock trading_orchestrator import.",
        "Live stub — extends ModeManager if logic is added.",
    ),
    ObsoleteEntry(
        "orchestrator/decision_journal.py",
        ObsoleteCategory.LEGACY_STUB,
        "Created as minimal stub to unblock trading_orchestrator import.",
        "Live stub — extends DecisionJournal if logic is added.",
    ),

    # ── hybrid/ — entire folder dead ─────────────────────────────────
    ObsoleteEntry(
        "hybrid/flow_controller.py",
        ObsoleteCategory.DEAD,
        "FlowController never instantiated. Day-49 quant+vision pipeline.",
        "Wired via core/runtime.py boot_hybrid() (constructed, not actively driven).",
    ),
    ObsoleteEntry(
        "hybrid/decision_validator.py",
        ObsoleteCategory.DEAD,
        "Only consumer is dead flow_controller.py.",
        "Marked legacy (transitively dead via flow_controller).",
    ),
    ObsoleteEntry(
        "hybrid/execution_router.py",
        ObsoleteCategory.DUPLICATE,
        "Parallel reimplementation of execution/execution_router.py. Class-name collision.",
        "Marked legacy. Canonical router is execution/execution_router.py.",
    ),
    ObsoleteEntry(
        "hybrid/confidence_calibrator.py",
        ObsoleteCategory.DUPLICATE,
        "Parallel reimplementation of memory/confidence_calibrator.py. Class-name collision.",
        "Marked legacy. Canonical calibrator TBD on next cleanup.",
    ),

    # ── analytics/ ───────────────────────────────────────────────────
    ObsoleteEntry(
        "analytics/performance_report.py",
        ObsoleteCategory.DEAD,
        "PerformanceReport class never instantiated. OptimizationSuggester not even re-exported.",
        "Wired via core/runtime.py boot_analytics(). Invoked by weekly review.",
    ),

    # ── core/ ────────────────────────────────────────────────────────
    ObsoleteEntry(
        "core/monitoring_system.py",
        ObsoleteCategory.SUPERSEDED,
        "Never imported. Canonical replacement: core/health_monitor.py. "
        "CORRECTED (2026-07-22): a duplicate entry further down in this file "
        "claimed '631 lines...Archived to .dead_code_archived in Round-29' — "
        "VERIFIED FALSE via `ls core/monitoring_system.py`, file is still "
        "present at its original path. Verified via grep: zero external "
        "importers (only internal self-references and the module-level "
        "`monitoring_system = MonitoringSystem()` singleton).",
        "Kept for backward compat; core/health_monitor is the live one. "
        "File is NOT archived — still on disk.",
    ),

    # ── computer_use/ — broken imports ───────────────────────────────
    ObsoleteEntry(
        "computer_use/browser_controller.py",
        ObsoleteCategory.BROKEN,
        "Referenced by tradingview_agent.py and computer_agent.py but file does not exist (the actual class BrowserController lives in browser_control.py).",
        "Documented as broken. computer_agent + tradingview_agent + run_day46_demo are excluded from runtime wiring.",
    ),
    ObsoleteEntry(
        "computer_use/tradingview_agent.py",
        ObsoleteCategory.BROKEN,
        "Imports missing browser_controller module.",
        "Excluded from runtime. Fix on dedicated computer_use revival pass.",
    ),
    ObsoleteEntry(
        "computer_use/computer_agent.py",
        ObsoleteCategory.BROKEN,
        "Imports missing browser_controller module AND missing BrowserAgent class.",
        "Excluded from runtime. Fix on dedicated computer_use revival pass.",
    ),
    ObsoleteEntry(
        "computer_use/run_day46_demo.py",
        ObsoleteCategory.BROKEN,
        "Transitively broken via tradingview_agent.",
        "Excluded from runtime. Demo script only.",
    ),

    # ── data/ ────────────────────────────────────────────────────────
    ObsoleteEntry(
        "data/verify_data_coverage.py",
        ObsoleteCategory.SMOKE_ONLY,
        "CLI-only verification script with stale hardcoded 2026-06-21 target date.",
        "Kept as a CLI utility. Not part of runtime.",
    ),

    # ── duplicate folder ─────────────────────────────────────────────
    ObsoleteEntry(
        "risk - Copy/",
        ObsoleteCategory.DUPLICATE,
        "Byte-for-byte duplicate of risk/ folder. Its autonomous_risk.py even imports from the real risk/ package, confirming it's stale shadow code.",
        "Do not import. Delete on next cleanup pass.",
    ),

    # ════════════════════════════════════════════════════════════════
    # Round-22+ audit additions — discovered during institutional audit
    # of analysis/, ml/, agents/, orchestrator/, strategies/, risk/,
    # broker/, core/ folders. These were NOT in the original registry.
    # ════════════════════════════════════════════════════════════════

    # ── Root duplicate (Round-19) ────────────────────────────────────
    ObsoleteEntry(
        "trader.py",
        ObsoleteCategory.DUPLICATE,
        "1,975-line stale copy of core/trader.py. Missing _reject(), _sync_balance(), "
        "_get_live_open_pairs() and other safety methods present in the live version. "
        "Renamed to trader.py.dead_duplicate_removed in Round-19.",
        "Archived. Do not import — use core/trader.py.",
    ),

    # ── analysis/ dead code (Round-22) ───────────────────────────────
    ObsoleteEntry(
        "analysis/candlestick_patterns_mw.py",
        ObsoleteCategory.WIRED,
        "CORRECTED (post-Round-22 audit): the 'superseded by candlestick_"
        "patterns_ml.py' claim below was wrong — a direct read of both files "
        "shows _mw.py is an independent 33-pattern scanner (1/2/3-bar, own "
        "trend filter) while _ml.py is a separate 8-pattern boolean detector. "
        "Neither duplicates the other, and the file was never actually moved "
        "to .dead_code_archived as this entry previously claimed. Now wired "
        "into analysis/extended_modules_adapter.py as a directional vote.",
        "WIRED — see analysis/extended_modules_adapter.py:_vote_candlestick_patterns_mw.",
    ),
    ObsoleteEntry(
        "analysis/candlestick_patterns_br.py",
        ObsoleteCategory.WIRED,
        "CORRECTED (post-Round-22 audit): 584-line Brazilian-book scanner "
        "(11 patterns) with its own trend/volume/next-bar confirmation "
        "filtering that candlestick_patterns_ml.py doesn't have — not a "
        "duplicate. File was never actually archived as this entry "
        "previously claimed. Now wired into analysis/extended_modules_"
        "adapter.py as a bullish-only directional vote (module has no "
        "bearish pattern set).",
        "WIRED — see analysis/extended_modules_adapter.py:_vote_candlestick_patterns_br.",
    ),
    ObsoleteEntry(
        "analysis/supermao_ichimoku.py",
        ObsoleteCategory.WIRED,
        "CORRECTED (post-Round-22 audit): 200-line alternate Ichimoku "
        "implementation, distinct from the live analysis/ichimoku.py (own "
        "TK-cross + cloud-position entry rule, ported from a different MQL4 "
        "source). File was never actually archived as this entry previously "
        "claimed. Now wired into analysis/extended_modules_adapter.py as a "
        "directional vote (requires >=90 bars of history for the 52-period "
        "Senkou Span B + 26-bar displacement).",
        "WIRED — see analysis/extended_modules_adapter.py:_vote_supermao_ichimoku.",
    ),

    # ── agents/ dead code (Round-22) ─────────────────────────────────
    # NOTE (2026-07-22): duplicate agents/chart_agent.py entry removed from
    # here — merged into the corrected entry in the agents/ section above
    # (it falsely claimed "Archived...Round-22"; file is still on disk).

    # ── orchestrator/ dead code (Round-25) ───────────────────────────
    ObsoleteEntry(
        "orchestrator/trading_sessions.py",
        ObsoleteCategory.DEAD,
        "296-line trading sessions module. 0 importers.",
        "Archived to .dead_code_archived in Round-25.",
    ),

    # ── strategies/ dead code (Round-26) ─────────────────────────────
    ObsoleteEntry(
        "strategy/multi_strategy_set.py",
        ObsoleteCategory.DEAD,
        "410-line multi-strategy set with df.eval() rule engine. 0 importers.",
        "Archived to .dead_code_archived in Round-26.",
    ),
    ObsoleteEntry(
        "strategies/",
        ObsoleteCategory.DEAD,
        "Entire strategies/ (plural) folder — 11 files, ~1,753 lines. All 0 importers. "
        "Includes: pattern_strategies_ml, scalping_strategy, ema_rsi_combo, reversal, "
        "trend_follow, breakout, momentum, retest, pullback, mean_reversion, range_trading. "
        "Pullback.py had a confirmed copy-paste bug (bc→bear_c) fixed before archiving.",
        "All archived to .dead_code_archived in Round-26.",
    ),

    # ── risk/ dead code (Round-27) ───────────────────────────────────
    ObsoleteEntry(
        "risk/book_guardrails.py",
        ObsoleteCategory.DEAD,
        "539 lines. 0 importers.",
        "Archived to .dead_code_archived in Round-27.",
    ),
    # NOTE (2026-07-22): duplicate risk/portfolio_manager.py entry removed
    # from here — merged into the corrected entry in the risk/ section
    # above (it falsely claimed "Archived...Round-27"; file is still on disk).
    ObsoleteEntry(
        "risk/risk_simulator.py",
        ObsoleteCategory.DEAD,
        "448 lines. 0 importers. CORRECTED (2026-07-22): a near-duplicate "
        "entry above (risk/ section) already covers this file with the same "
        "DEAD conclusion; this entry's 'Archived to .dead_code_archived in "
        "Round-27' claim is FALSE — `ls risk/risk_simulator.py` confirms the "
        "file is still present at its original path. Category (DEAD) is "
        "correct; only the archival claim was wrong.",
        "Marked legacy. Do not import. File is NOT archived — still on disk.",
    ),
    ObsoleteEntry(
        "risk/order_split_manager.py",
        ObsoleteCategory.DEAD,
        "400 lines. 0 importers.",
        "Archived to .dead_code_archived in Round-27.",
    ),
    ObsoleteEntry(
        "risk/controlled_grid_scaler.py",
        ObsoleteCategory.DEAD,
        "343 lines. 0 importers.",
        "Archived to .dead_code_archived in Round-27.",
    ),
    ObsoleteEntry(
        "risk/basket_exit.py",
        ObsoleteCategory.DEAD,
        "342 lines. 0 importers.",
        "Archived to .dead_code_archived in Round-27.",
    ),
    ObsoleteEntry(
        "risk/atr_risk_manager.py",
        ObsoleteCategory.DEAD,
        "298 lines. 0 importers. Note: had unguarded division (M6) fixed in Round-19 before archiving.",
        "Archived to .dead_code_archived in Round-27.",
    ),
    ObsoleteEntry(
        "risk/probability_distribution.py",
        ObsoleteCategory.DEAD,
        "256 lines. 0 importers.",
        "Archived to .dead_code_archived in Round-27.",
    ),
    ObsoleteEntry(
        "risk/compounding.py",
        ObsoleteCategory.DEAD,
        "206 lines. 0 importers.",
        "Archived to .dead_code_archived in Round-27.",
    ),
    ObsoleteEntry(
        "risk/entry_score.py, risk/institutional_entry_framework.py, "
        "risk/revenge_trading_detector.py, risk/structure_stop.py, "
        "risk/confirmation_bias_defense.py",
        ObsoleteCategory.DEAD,
        "5 additional dead risk modules, ~573 lines combined. All 0 importers.",
        "All archived to .dead_code_archived in Round-27.",
    ),
    ObsoleteEntry(
        "risk/entry_quality_guardrails.py",
        ObsoleteCategory.WIRED,
        "1,716 lines. Was DEAD (0 importers) — built from real-trade post-mortem "
        "(GBPUSD M5, 2026-07-02). 12 entry-quality checks (chasing filter, SL swing "
        "anchor, TP structure validation, indecision candles, etc.). "
        "Round-22 fix: NOW WIRED into trade_permission.py as final gate.",
        "Live — wired in Round-22. No action needed.",
    ),

    # ── broker/ dead code (Round-28) ─────────────────────────────────
    ObsoleteEntry(
        "broker/broker_factory.py",
        ObsoleteCategory.DEAD,
        "243 lines. 0 importers.",
        "Archived to .dead_code_archived in Round-28.",
    ),
    ObsoleteEntry(
        "broker/mt5_historical_fetcher.py",
        ObsoleteCategory.DEAD,
        "182 lines. 0 importers.",
        "Archived to .dead_code_archived in Round-28.",
    ),
    ObsoleteEntry(
        "broker/magic_number.py",
        ObsoleteCategory.DEAD,
        "161 lines. 0 importers.",
        "Archived to .dead_code_archived in Round-28.",
    ),
    # NOTE (2026-07-22): duplicate broker/market_data_manager.py entry
    # removed from here — merged into the corrected entry in the broker/
    # section above (it falsely claimed "Archived...Round-28"; file is
    # still on disk).
    ObsoleteEntry(
        "broker/position_manager.py",
        ObsoleteCategory.WIRED,
        "747 lines. Was DEAD (0 importers) — trailing stop, breakeven, partial close, "
        "Friday close. Paper trades got active management but live MT5 trades did not. "
        "Round-22 fix: NOW WIRED into core/trader.py for mt5_demo/mt5_live modes.",
        "Live — wired in Round-22. No action needed.",
    ),

    # ── core/ dead code (Round-29) ───────────────────────────────────
    ObsoleteEntry(
        "core/production_trading_system.py",
        ObsoleteCategory.DEAD,
        "658 lines. 0 importers. Third 'production-ready' attempt (alongside "
        "production_hardening.py and production_excellence.py). Only production_hardening.py is live.",
        "FIX (execution-parity audit item 8, 2026-07-19): this entry claimed "
        "'Archived to .dead_code_archived in Round-29' but the file was still "
        "present under its live .py name with 0 importers — an alternate, "
        "self-contained trading engine sitting as a landmine for anyone who "
        "ran it directly expecting it to be 'the' engine. Actually renamed "
        "to core/production_trading_system.py.dead_code_archived now.",
    ),
    ObsoleteEntry(
        "core/production_excellence.py",
        ObsoleteCategory.DEAD,
        "517 lines. 0 importers. Another abandoned 'production-ready' attempt.",
        "FIX (execution-parity audit item 8, 2026-07-19): same registry/reality "
        "mismatch as production_trading_system.py above — actually renamed to "
        "core/production_excellence.py.dead_code_archived now.",
    ),
    # NOTE (2026-07-22): duplicate core/monitoring_system.py entry removed
    # from here — merged into the corrected entry in the core/ section
    # above (it falsely claimed "Archived...Round-29"; file is still on disk).
    ObsoleteEntry(
        "core/signal_scorer.py",
        ObsoleteCategory.DEAD,
        "291 lines. 0 importers.",
        "Archived to .dead_code_archived in Round-29.",
    ),
]


def obsolete_index() -> Dict[str, ObsoleteEntry]:
    """Return a {path: entry} map for quick lookup."""
    return {e.path: e for e in OBSOLETE_MODULES}


def obsolete_summary() -> Dict[str, int]:
    """Counts per category — useful for the final report."""
    counts: Dict[str, int] = {}
    for entry in OBSOLETE_MODULES:
        counts[entry.category.value] = counts.get(entry.category.value, 0) + 1
    counts["total"] = len(OBSOLETE_MODULES)
    return counts


def is_obsolete(path: str) -> bool:
    return path in obsolete_index()