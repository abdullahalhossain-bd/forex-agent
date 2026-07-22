"""
core/obsolete.py — Evidence-backed registry of obsolete / orphan / support modules
====================================================================================

CONSOLIDATED (Registry Finalization pass, 2026-07-22): this file was
rewritten from a historical notes file into an evidence-backed registry.
Every entry now carries a `status`, an `archive_state` (was the file
actually moved off its live import path, or just copied?), a
`confidence` level, and — where verified this pass — an `evidence`
string naming the exact importer/consumer.

`confidence=HIGH` means the importer graph for that path was directly
grepped and the file's on-disk state directly checked during the Phase
A / B1 / B2 / B3 audit (2026-07-19 to 2026-07-22). `confidence=MEDIUM`
means the entry's current status is inherited from an earlier corrected
audit pass (Round-19 through Round-29 / execution-parity audit) but was
not independently re-verified during this consolidation. `confidence=LOW`
means the entry is unresolved/contradictory and needs a dedicated look.

Known unresolved items after this pass (see bottom of file, `UNKNOWN_OR_REVIEW`):
  * risk/revenge_trading_detector.py — archived copy and live original
    differ (a bug fix landed on the "dead" file after it was archived).
    Do not delete without a human decision.

This file is the single source of truth — do not duplicate this list
elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ObsoleteStatus(str, Enum):
    ACTIVE = "active"                              # live, on the main decision/execution path
    ACTIVE_SUPPORT = "active_support"               # live, registered/constructed but not decision-critical
    ACTIVE_DYNAMIC = "active_dynamic"                # live via lazy/dynamic import only
    ORPHAN_READY = "orphan_ready"                    # complete implementation, zero importers, no broken deps
    LEGACY = "legacy"                                # kept intentionally for back-compat, superseded module still registered
    SUPERSEDED = "superseded"                        # replaced by a newer module, no live importer
    MANUAL_REVIEW_REQUIRED = "manual_review_required"  # conflicting signals, needs a human decision
    UTILITY_TEST = "utility_test"                    # only imported by tests
    UTILITY_BACKTEST = "utility_backtest"            # only used by backtest tooling
    UTILITY_DOC = "utility_doc"                      # docs/CLI tooling only
    STALE_ENTRY = "stale_entry"                      # registry entry describes files that no longer exist on disk
    DEAD = "dead"                                    # zero importers, no replacement in flight, safe-ish to remove
    UNKNOWN = "unknown"                              # not yet verified


class ArchiveState(str, Enum):
    MOVED = "moved"                # original file no longer exists at its live path
    COPIED_ONLY = "copied_only"    # a .dead_code_archived copy exists AND the original is still live-importable
    NO_ARCHIVE = "no_archive"      # registry/comments claim archival but no .dead_code_archived copy exists
    NOT_APPLICABLE = "n/a"         # entry was never claimed to be archived
    UNKNOWN = "unknown"            # not checked this pass


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ObsoleteEntry:
    path: str
    status: ObsoleteStatus
    archive_state: ArchiveState
    reason: str
    action: str
    confidence: Confidence
    evidence: str = ""


OBSOLETE_MODULES: List[ObsoleteEntry] = [

    ObsoleteEntry(
        'agents/chart_agent.py',
        ObsoleteStatus.SUPERSEDED,
        ArchiveState.NO_ARCHIVE,
        'Standalone Playwright S/R drawer; superseded by computer_use/chart_drawer.py stack (that stack itself was later deleted entirely — see computer_use/ STALE_ENTRY below).',
        'Zero importers confirmed. File is on disk, un-archived. Delete or move to legacy/ on next pass.',
        Confidence.HIGH,
        evidence='grep: zero importers of agents.chart_agent anywhere in repo (2026-07-22).',
    ),
    ObsoleteEntry(
        'ai/automated_retraining.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NOT_APPLICABLE,
        'Zero importers; no CLI entry point.',
        "CORRECTED: previous action text claimed 'wired via core/runtime.py boot_ai() with safe try/except' — FALSE. boot_ai() body contains no reference to automated_retraining. Genuinely dead.",
        Confidence.HIGH,
        evidence="grep core/runtime.py boot_ai() body: zero matches for 'retraining' (2026-07-22).",
    ),
    ObsoleteEntry(
        'ai/model_versioning.py',
        ObsoleteStatus.UTILITY_TEST,
        ArchiveState.NOT_APPLICABLE,
        'MLflow init guarded behind MLFLOW_AVAILABLE flag. Only consumed by dead automated_retraining.py + core/monitoring_system.py (itself superseded/unimported).',
        'Kept; not on any live path.',
        Confidence.MEDIUM,
        evidence='Inherited from prior audit, not re-grepped this pass.',
    ),
    ObsoleteEntry(
        'analysis/smart_money.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Day-61 master SMC orchestrator never wired into agents/analysis_agent.py.',
        "CORRECTED: registry claimed 'DELETED 2026-07-02' — FALSE, file still present on disk, never actually deleted. Category (DEAD/zero-importers) still correct.",
        Confidence.HIGH,
        evidence='`test -f analysis/smart_money.py` → exists (2026-07-22).',
    ),
    ObsoleteEntry(
        'analysis/structure.py',
        ObsoleteStatus.ACTIVE,
        ArchiveState.NOT_APPLICABLE,
        'Live — imported by agents/analysis_agent.py (MarketStructureEngine) and analysis/structure_mtf.py.',
        'Keep. Do not touch.',
        Confidence.MEDIUM,
        evidence='Inherited from prior corrected audit entry; not independently re-grepped this pass.',
    ),
    ObsoleteEntry(
        'analysis/liquidity.py',
        ObsoleteStatus.ACTIVE,
        ArchiveState.NOT_APPLICABLE,
        'Verified live consumer: agents/analysis_agent.py does `from analysis.liquidity import LiquidityEngine`, and analysis_agent.py is reachable from main.py as part of the live AnalysisAgent -> DecisionAgent -> RiskEngine pipeline.',
        'ACTIVE — do not delete.',
        Confidence.HIGH,
        evidence='Explicit verified finding, applied per registry-finalization instructions (2026-07-22).',
    ),
    ObsoleteEntry(
        'analysis/liquidity_engine.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Day-62 liquidity orchestrator never wired into AnalysisAgent.',
        "CORRECTED: 'DELETED 2026-07-02' claim is FALSE — file still on disk. Category correct.",
        Confidence.HIGH,
        evidence='`test -f` → exists (2026-07-22).',
    ),
    ObsoleteEntry(
        'analysis/liquidity_zones.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Only consumer was dead liquidity_engine.py.',
        "CORRECTED: 'DELETED 2026-07-02' claim is FALSE — file still on disk. Category correct.",
        Confidence.HIGH,
        evidence='`test -f` → exists (2026-07-22).',
    ),
    ObsoleteEntry(
        'analysis/stop_hunt_detector.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Only consumer was dead liquidity_engine.py.',
        "CORRECTED: 'DELETED 2026-07-02' claim is FALSE — file still on disk. Category correct.",
        Confidence.HIGH,
        evidence='`test -f` → exists (2026-07-22).',
    ),
    ObsoleteEntry(
        'analysis/session_analysis.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'London-manipulation detector; only consumer was dead liquidity_engine.py.',
        "CORRECTED: 'DELETED 2026-07-02' claim is FALSE — file still on disk. Live replacement is session_analyzer.py (different file, still present) + ict_amd_signal_engine.py.",
        Confidence.HIGH,
        evidence='`test -f` → exists (2026-07-22).',
    ),
    ObsoleteEntry(
        'analysis/amd_strategy.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Day-36/37 AMD strategy, superseded by ict_amd_signal_engine.py (stricter spec).',
        "CORRECTED: 'DELETED 2026-07-02' claim is FALSE — file still on disk. Category correct.",
        Confidence.HIGH,
        evidence='`test -f` → exists (2026-07-22).',
    ),
    ObsoleteEntry(
        'analysis/currency_strength.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NOT_APPLICABLE,
        'Day-64 currency-strength orchestrator never wired into AnalysisAgent.',
        'Marked legacy. Wire via AnalysisAgent if adopting Day-64 pipeline.',
        Confidence.MEDIUM,
        evidence='Not re-grepped this pass; no archival claim to check.',
    ),
    ObsoleteEntry(
        'analysis/currency_ranker.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NOT_APPLICABLE,
        'Only consumer is dead currency_strength.py.',
        'Marked legacy (transitively dead).',
        Confidence.MEDIUM,
        evidence='Not re-grepped this pass.',
    ),
    ObsoleteEntry(
        'analysis/strength_calculator.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NOT_APPLICABLE,
        'Only consumer is dead currency_strength.py.',
        'Marked legacy (transitively dead).',
        Confidence.MEDIUM,
        evidence='Not re-grepped this pass.',
    ),
    ObsoleteEntry(
        'analysis/mtf_analyzer.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'Only _detect_bos/_detect_choch/_detect_liquidity_sweep helpers are used by smc_engine. Public analyze() pipeline (~700 LOC) never invoked.',
        'Keep (helpers live). Public analyze() dormant.',
        Confidence.MEDIUM,
        evidence='Inherited from prior audit.',
    ),
    ObsoleteEntry(
        'analysis/database/__init__.py',
        ObsoleteStatus.LEGACY,
        ArchiveState.NOT_APPLICABLE,
        'Empty placeholder subpackage; no concrete DB modules.',
        'Kept as marker.',
        Confidence.MEDIUM,
        evidence='Inherited from prior audit.',
    ),
    ObsoleteEntry(
        'analysis/candlestick_patterns_mw.py',
        ObsoleteStatus.ACTIVE,
        ArchiveState.NO_ARCHIVE,
        'Independent 33-pattern scanner, wired into analysis/extended_modules_adapter.py as a directional vote.',
        "ACTIVE — see extended_modules_adapter.py:_vote_candlestick_patterns_mw. File was never actually archived despite an earlier duplicate entry's claim.",
        Confidence.MEDIUM,
        evidence='Inherited from Round-22 corrected audit; not independently re-grepped this pass.',
    ),
    ObsoleteEntry(
        'analysis/candlestick_patterns_br.py',
        ObsoleteStatus.ACTIVE,
        ArchiveState.NO_ARCHIVE,
        '584-line Brazilian-book scanner (11 patterns), wired into extended_modules_adapter.py as a bullish-only directional vote.',
        'ACTIVE — see extended_modules_adapter.py:_vote_candlestick_patterns_br.',
        Confidence.MEDIUM,
        evidence='Inherited from Round-22 corrected audit; not independently re-grepped this pass.',
    ),
    ObsoleteEntry(
        'analysis/supermao_ichimoku.py',
        ObsoleteStatus.ACTIVE,
        ArchiveState.NO_ARCHIVE,
        '200-line alternate Ichimoku implementation, distinct from live analysis/ichimoku.py, wired into extended_modules_adapter.py as a directional vote.',
        'ACTIVE — see extended_modules_adapter.py:_vote_supermao_ichimoku.',
        Confidence.MEDIUM,
        evidence='Inherited from Round-22 corrected audit; not independently re-grepped this pass.',
    ),
    ObsoleteEntry(
        'broker/market_data_manager.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        'Zero real importers; only comment/docstring mentions in broker/__init__.py and core/trader.py, no actual import statement.',
        'Marked legacy. If reviving MT5 data path, wire into server/signal_pipeline.py.',
        Confidence.HIGH,
        evidence='grep: 0 real importers; `.dead_code_archived` copy is byte-identical to original (copy-not-move) (2026-07-22).',
    ),
    ObsoleteEntry(
        'broker/mt5_data.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Only consumers are dead broker/market_data_manager.py + broker/symbol_manager.py (both themselves zero-importer).',
        'Marked legacy (transitively dead).',
        Confidence.HIGH,
        evidence='grep: only referenced by market_data_manager.py and symbol_manager.py, both themselves dead; no .dead_code_archived copy exists (2026-07-22).',
    ),
    ObsoleteEntry(
        'broker/symbol_manager.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Only consumer is dead broker/market_data_manager.py.',
        'Marked legacy (transitively dead).',
        Confidence.HIGH,
        evidence='grep: 1 importer (market_data_manager.py, itself dead); no archive copy (2026-07-22).',
    ),
    ObsoleteEntry(
        'broker/data_validator.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Broker-side validator; only consumer is dead broker/market_data_manager.py. (data/validator.py is the live one.)',
        'Marked legacy (transitively dead).',
        Confidence.HIGH,
        evidence='grep: 1 importer (market_data_manager.py, itself dead); no archive copy (2026-07-22).',
    ),
    ObsoleteEntry(
        'broker/safety_guard.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'Lazily imported by core/runtime.py, not just tests. Production safety gate is in core/trader.py (TradePermission + CorrelationFilter), so this is a secondary guard, not decision-critical.',
        'CORRECTED: previous category SMOKE_ONLY undersold it — core/runtime.py:841 does `from broker.safety_guard import SafetyGuard` at boot.',
        Confidence.HIGH,
        evidence='grep core/runtime.py:841 (2026-07-22).',
    ),
    ObsoleteEntry(
        'broker/spread_monitor.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'Lazily imported by core/runtime.py at boot, not just tests.',
        'CORRECTED: previous category SMOKE_ONLY undersold it — core/runtime.py:850 imports it.',
        Confidence.HIGH,
        evidence='grep core/runtime.py:850 (2026-07-22).',
    ),
    ObsoleteEntry(
        'broker/health_monitor.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'Constructed by core/runtime.py and execution/execution_router.py; check_once()/run_loop() are not scheduled, so it does no periodic work, but it is instantiated on the live path. Canonical replacement for actual health checks is core/health_monitor.py.',
        'CORRECTED: previous category SMOKE_ONLY undersold it — imported in 2 live files, not just tests.',
        Confidence.HIGH,
        evidence='grep core/runtime.py:943, execution/execution_router.py:235 (2026-07-22).',
    ),
    ObsoleteEntry(
        'broker/broker_factory.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '243 lines. 0 importers.',
        "Marked legacy. NOTE: 'Archived to .dead_code_archived' claim was misleading — the original file is still present and importable, only a copy sits at the archived name.",
        Confidence.HIGH,
        evidence='grep: 0 importers; checksum-identical .dead_code_archived copy exists, original not removed (2026-07-22).',
    ),
    ObsoleteEntry(
        'broker/mt5_historical_fetcher.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '182 lines. 0 importers.',
        'Marked legacy. Archive claim corrected as above.',
        Confidence.HIGH,
        evidence='Same verification method as broker_factory.py (2026-07-22).',
    ),
    ObsoleteEntry(
        'broker/magic_number.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '161 lines. 0 importers.',
        'Marked legacy. Archive claim corrected as above.',
        Confidence.HIGH,
        evidence='Same verification method as broker_factory.py (2026-07-22).',
    ),
    ObsoleteEntry(
        'broker/position_manager.py',
        ObsoleteStatus.ACTIVE,
        ArchiveState.NOT_APPLICABLE,
        '747 lines. Trailing stop, breakeven, partial close, Friday close.',
        'ACTIVE — core/trader.py imports PositionManager and calls .register_open() / .poll_once() for live MT5 trade management (trailing stop, breakeven, partial close).',
        Confidence.HIGH,
        evidence='Explicit verified finding, applied per registry-finalization instructions (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/portfolio_manager.py',
        ObsoleteStatus.SUPERSEDED,
        ArchiveState.COPIED_ONLY,
        'Pre-Day-58 portfolio prototype; superseded by risk/capital_manager.py + risk/exposure_manager.py. Module-level singleton runs at import time but has zero external importers.',
        'Marked legacy. Do not import.',
        Confidence.HIGH,
        evidence='grep: 0 external importers; checksum-identical .dead_code_archived copy, original not removed (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/book_guardrails.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '539 lines. 0 importers.',
        'Marked legacy. Archive claim corrected — copy only, original still live-importable.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/risk_simulator.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '448 lines. 0 importers.',
        'Marked legacy. Archive claim corrected — copy only, original still live-importable.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/order_split_manager.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '400 lines. 0 importers.',
        'Marked legacy. Archive claim corrected — copy only.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/controlled_grid_scaler.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '343 lines. 0 importers.',
        'Marked legacy. Archive claim corrected — copy only.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/basket_exit.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '342 lines. 0 importers.',
        'Marked legacy. Archive claim corrected — copy only.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/atr_risk_manager.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '298 lines. 0 importers. Had unguarded division (M6) fixed in Round-19 before archiving.',
        'Marked legacy. Archive claim corrected — copy only.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/probability_distribution.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '256 lines. 0 importers.',
        'Marked legacy. Archive claim corrected — copy only.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/compounding.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '206 lines. 0 importers.',
        'Marked legacy. Archive claim corrected — copy only.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/entry_score.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '0 importers. Split out of a combined 5-file registry entry for clarity (registry finalization, 2026-07-22).',
        'Marked legacy. Archive claim corrected — copy only.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/institutional_entry_framework.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '0 importers. Split out of a combined 5-file registry entry for clarity.',
        'Marked legacy. Archive claim corrected — copy only.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/revenge_trading_detector.py',
        ObsoleteStatus.MANUAL_REVIEW_REQUIRED,
        ArchiveState.COPIED_ONLY,
        '0 importers at grep time. Split out of a combined 5-file registry entry. UNLIKE its siblings, the live original and the .dead_code_archived copy are NOT checksum-identical — line 49 was edited on the live file after archiving (bare `except:` tightened to `except (ValueError, TypeError):`).',
        'DO NOT auto-delete or auto-move. Someone touched this file after it was marked dead — either a maintenance pass or a sign of pending revival. Needs a human decision before any cleanup action.',
        Confidence.HIGH,
        evidence='diff risk/revenge_trading_detector.py vs .dead_code_archived copy: 1 line differs (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/structure_stop.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '0 importers. Split out of a combined 5-file registry entry.',
        'Marked legacy. Archive claim corrected — copy only.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/confirmation_bias_defense.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '0 importers. Split out of a combined 5-file registry entry.',
        'Marked legacy. Archive claim corrected — copy only.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'risk/entry_quality_guardrails.py',
        ObsoleteStatus.ACTIVE,
        ArchiveState.NOT_APPLICABLE,
        '1,716 lines. 12 entry-quality checks (chasing filter, SL swing anchor, TP structure validation, indecision candles, etc.), built from a real-trade post-mortem.',
        'ACTIVE — wired into risk/trade_permission.py as the final gate.',
        Confidence.HIGH,
        evidence='grep: risk/trade_permission.py:193 `from risk.entry_quality_guardrails import run_all_entry_quality_checks` (2026-07-22).',
    ),
    ObsoleteEntry(
        'scanner/scanner.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NOT_APPLICABLE,
        'Byte-identical duplicate of scanner/config.py minus header. Zero importers.',
        'Marked legacy. Delete on next cleanup pass.',
        Confidence.MEDIUM,
        evidence='Inherited from prior audit; not re-grepped this pass.',
    ),
    ObsoleteEntry(
        'fundamental/fundamental_sentiment.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'FundamentalSentimentScore is constructed and registered at boot.',
        "CORRECTED: previous claim 'never imported' is FALSE — core/runtime.py:375-376 does `from fundamental.fundamental_sentiment import FundamentalSentimentScore` inside a try/except and registers it as a service. Not decision-critical (not on the AnalysisAgent path), but genuinely constructed at runtime, not dead.",
        Confidence.HIGH,
        evidence='grep core/runtime.py:375-376 (2026-07-22).',
    ),
    ObsoleteEntry(
        'memory/trade_context.py',
        ObsoleteStatus.LEGACY,
        ArchiveState.NOT_APPLICABLE,
        '0-byte placeholder.',
        'Kept as marker.',
        Confidence.MEDIUM,
        evidence='Inherited from prior audit.',
    ),
    ObsoleteEntry(
        'memory/confidence_calibrator.py',
        ObsoleteStatus.SUPERSEDED,
        ArchiveState.NOT_APPLICABLE,
        'Class-name collision with hybrid/confidence_calibrator.py (also dead) AND with the actual live intelligence/confidence_calibrator.py (see new entry below). Neither memory/ nor hybrid/ version has any importer.',
        'Marked legacy. The canonical, live calibrator is intelligence/confidence_calibrator.py — do not confuse the three.',
        Confidence.HIGH,
        evidence='grep confirms zero importers of memory.confidence_calibrator; the live one is intelligence/confidence_calibrator.py per Phase B2 audit (2026-07-22).',
    ),
    ObsoleteEntry(
        'learning/weekly_review.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'run_weekly_review() is registered as a live callable.',
        "CORRECTED: registered, not just 'wired via boot_learning()' in the abstract — core/runtime.py:1099-1100 does `from learning.weekly_review import run_weekly_review` and `registry.register_instance('weekly_review_fn', run_weekly_review)`.",
        Confidence.HIGH,
        evidence='grep core/runtime.py:1099-1100 (2026-07-22).',
    ),
    ObsoleteEntry(
        'learning/memory_integration.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'MemoryIntegration is registered as a live service.',
        "CORRECTED: core/runtime.py:1085-1086 does `from learning.memory_integration import MemoryIntegration` and `registry.register('memory_integration', ...)`.",
        Confidence.HIGH,
        evidence='grep core/runtime.py:1085-1086 (2026-07-22).',
    ),
    ObsoleteEntry(
        'automation/error_handler.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'ErrorHandler is registered as a live service at boot.',
        "CORRECTED: core/runtime.py:1198-1199 does `from automation.error_handler import ErrorHandler` and `registry.register_instance('error_handler', ErrorHandler())`.",
        Confidence.HIGH,
        evidence='grep core/runtime.py:1198-1199 (2026-07-22).',
    ),
    ObsoleteEntry(
        'automation/runtime_metrics.py',
        ObsoleteStatus.SUPERSEDED,
        ArchiveState.NOT_APPLICABLE,
        'Superseded by core/runtime_metrics.py (canonical).',
        'Kept for backward compat; core/runtime_metrics is the live one.',
        Confidence.HIGH,
        evidence='grep core/runtime.py:46 imports only core.runtime_metrics, never automation.runtime_metrics (2026-07-22).',
    ),
    ObsoleteEntry(
        'automation/daily_review.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'DailyReview is registered as a live service at boot.',
        "CORRECTED: core/runtime.py:1205-1206 does `from automation.daily_review import DailyReview` and `registry.register('daily_review', ...)`.",
        Confidence.HIGH,
        evidence='grep core/runtime.py:1205-1206 (2026-07-22).',
    ),
    ObsoleteEntry(
        'automation/system_health.py',
        ObsoleteStatus.LEGACY,
        ArchiveState.NOT_APPLICABLE,
        "Registered explicitly as 'system_health_legacy' — code itself labels it legacy. Canonical replacement is core/health_monitor.py.",
        'Kept for backward compat; core/health_monitor is the live one.',
        Confidence.HIGH,
        evidence="grep core/runtime.py:1212-1214, service name literally 'system_health_legacy' (2026-07-22).",
    ),
    ObsoleteEntry(
        'orchestrator/trading_orchestrator.py',
        ObsoleteStatus.ACTIVE,
        ArchiveState.NOT_APPLICABLE,
        'Previously imported 4 missing sub-modules (safety_controller, self_healing, mode_manager, decision_journal); those stubs now exist.',
        'ACTIVE — imported by orchestrator/daily_routine.py and core/runtime.py:1251.',
        Confidence.HIGH,
        evidence='grep: 2 live importers found (2026-07-22).',
    ),
    ObsoleteEntry(
        'orchestrator/safety_controller.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'Minimal stub created to unblock trading_orchestrator import.',
        'Live stub — imported by orchestrator/trading_orchestrator.py (itself ACTIVE).',
        Confidence.HIGH,
        evidence='grep: 1 importer, trading_orchestrator.py:32 (2026-07-22).',
    ),
    ObsoleteEntry(
        'orchestrator/self_healing.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'Minimal stub created to unblock trading_orchestrator import.',
        'Live stub — imported by orchestrator/trading_orchestrator.py.',
        Confidence.HIGH,
        evidence='grep: 1 importer, trading_orchestrator.py:33 (2026-07-22).',
    ),
    ObsoleteEntry(
        'orchestrator/mode_manager.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'Minimal stub created to unblock trading_orchestrator import.',
        'Live stub — imported by orchestrator/trading_orchestrator.py.',
        Confidence.HIGH,
        evidence='grep: 1 importer, trading_orchestrator.py:35 (2026-07-22).',
    ),
    ObsoleteEntry(
        'orchestrator/decision_journal.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'Minimal stub created to unblock trading_orchestrator import.',
        'Live stub — imported by orchestrator/trading_orchestrator.py.',
        Confidence.HIGH,
        evidence='grep: 1 importer, trading_orchestrator.py:36 (2026-07-22).',
    ),
    ObsoleteEntry(
        'orchestrator/trading_sessions.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '296-line trading sessions module. 0 importers.',
        'Marked legacy. Archive claim corrected — copy only, original still live-importable.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'hybrid/flow_controller.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NOT_APPLICABLE,
        'FlowController never instantiated. Day-49 quant+vision pipeline.',
        "CORRECTED: previous claim 'wired via core/runtime.py boot_hybrid() (constructed, not actively driven)' is FALSE — boot_hybrid()'s only line is `log.info('boot_hybrid: skipped (hybrid/ is legacy — system uses core/trader.py)')`. Nothing is constructed.",
        Confidence.HIGH,
        evidence='grep core/runtime.py boot_hybrid() body — explicit skip, zero references to flow_controller (2026-07-22).',
    ),
    ObsoleteEntry(
        'hybrid/decision_validator.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NOT_APPLICABLE,
        'Only consumer is dead flow_controller.py.',
        'Marked legacy (transitively dead via flow_controller, which is confirmed skipped).',
        Confidence.HIGH,
        evidence='Same verification as flow_controller.py (2026-07-22).',
    ),
    ObsoleteEntry(
        'hybrid/execution_router.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NOT_APPLICABLE,
        'Parallel reimplementation of execution/execution_router.py. Class-name collision.',
        'Marked legacy. Canonical router is execution/execution_router.py.',
        Confidence.MEDIUM,
        evidence='Inherited from prior audit; not re-grepped this pass.',
    ),
    ObsoleteEntry(
        'hybrid/confidence_calibrator.py',
        ObsoleteStatus.SUPERSEDED,
        ArchiveState.NOT_APPLICABLE,
        'Parallel reimplementation of memory/confidence_calibrator.py (also dead). Class-name collision with the actual live intelligence/confidence_calibrator.py.',
        'Marked legacy. See intelligence/confidence_calibrator.py for the live one.',
        Confidence.HIGH,
        evidence='grep confirms zero importers; live one identified in Phase B2 (2026-07-22).',
    ),
    ObsoleteEntry(
        'analytics/performance_report.py',
        ObsoleteStatus.ACTIVE_SUPPORT,
        ArchiveState.NOT_APPLICABLE,
        'PerformanceReport is constructed and registered at boot.',
        'CORRECTED: core/runtime.py:1017-1019 does `from analytics.performance_report import PerformanceReport` and registers it as a service.',
        Confidence.HIGH,
        evidence='grep core/runtime.py:1017-1019 (2026-07-22).',
    ),
    ObsoleteEntry(
        'core/monitoring_system.py',
        ObsoleteStatus.SUPERSEDED,
        ArchiveState.NO_ARCHIVE,
        'Never imported outside itself. Canonical replacement: core/health_monitor.py.',
        "Kept for backward compat. No .dead_code_archived copy exists at all — the earlier 'Archived to .dead_code_archived in Round-29' claim was entirely false, not even a copy was made.",
        Confidence.HIGH,
        evidence='`test -f core/monitoring_system.py.dead_code_archived` → does not exist (2026-07-22).',
    ),
    ObsoleteEntry(
        'core/production_trading_system.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        "658 lines. 0 importers. Third 'production-ready' attempt (alongside production_hardening.py and production_excellence.py). Only production_hardening.py is live.",
        "CORRECTED: previous claim 'actually renamed to core/production_trading_system.py.dead_code_archived' is FALSE — both the original and the archived-name copy exist on disk, byte-identical. Nothing was renamed; it was copied.",
        Confidence.HIGH,
        evidence='`ls` shows both files present, checksum-identical (2026-07-22).',
    ),
    ObsoleteEntry(
        'core/production_excellence.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        "517 lines. 0 importers. Another abandoned 'production-ready' attempt.",
        "CORRECTED: same false 'renamed' claim as production_trading_system.py — copy only, original still present.",
        Confidence.HIGH,
        evidence='`ls` shows both files present, checksum-identical (2026-07-22).',
    ),
    ObsoleteEntry(
        'core/signal_scorer.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        '291 lines. 0 importers.',
        "Marked legacy. No .dead_code_archived copy exists — 'Archived to .dead_code_archived in Round-29' claim was entirely false.",
        Confidence.HIGH,
        evidence='`test -f core/signal_scorer.py.dead_code_archived` → does not exist (2026-07-22).',
    ),
    ObsoleteEntry(
        'computer_use/browser_controller.py',
        ObsoleteStatus.STALE_ENTRY,
        ArchiveState.MOVED,
        'Historically: referenced by tradingview_agent.py and computer_agent.py but the file never existed (class lived in browser_control.py instead).',
        "CORRECTED: the entire computer_use/ directory no longer exists on disk at all — not just this file. 'Fix on dedicated computer_use revival pass' action text is stale; there is nothing left to revive without recreating the folder from scratch.",
        Confidence.HIGH,
        evidence="`find . -iname '*computer_use*'` → zero results anywhere in the repo (2026-07-22).",
    ),
    ObsoleteEntry(
        'computer_use/tradingview_agent.py',
        ObsoleteStatus.STALE_ENTRY,
        ArchiveState.MOVED,
        'Historically imported the missing browser_controller module.',
        'CORRECTED: folder no longer exists. See browser_controller.py entry above.',
        Confidence.HIGH,
        evidence='Same verification as browser_controller.py (2026-07-22).',
    ),
    ObsoleteEntry(
        'computer_use/computer_agent.py',
        ObsoleteStatus.STALE_ENTRY,
        ArchiveState.MOVED,
        'Historically imported the missing browser_controller module and BrowserAgent class.',
        'CORRECTED: folder no longer exists. See browser_controller.py entry above.',
        Confidence.HIGH,
        evidence='Same verification as browser_controller.py (2026-07-22).',
    ),
    ObsoleteEntry(
        'computer_use/run_day46_demo.py',
        ObsoleteStatus.STALE_ENTRY,
        ArchiveState.MOVED,
        'Historically transitively broken via tradingview_agent.',
        'CORRECTED: folder no longer exists. See browser_controller.py entry above.',
        Confidence.HIGH,
        evidence='Same verification as browser_controller.py (2026-07-22).',
    ),
    ObsoleteEntry(
        'data/verify_data_coverage.py',
        ObsoleteStatus.UTILITY_DOC,
        ArchiveState.NOT_APPLICABLE,
        'CLI-only verification script with stale hardcoded 2026-06-21 target date.',
        'Kept as a CLI utility. Not part of runtime.',
        Confidence.MEDIUM,
        evidence='Inherited from prior audit.',
    ),
    ObsoleteEntry(
        'risk - Copy/',
        ObsoleteStatus.DEAD,
        ArchiveState.NOT_APPLICABLE,
        "Byte-for-byte duplicate of risk/ folder. Its autonomous_risk.py even imports from the real risk/ package, confirming it's stale shadow code.",
        'Do not import. Delete on next cleanup pass.',
        Confidence.MEDIUM,
        evidence='Inherited from prior audit; not re-diffed this pass.',
    ),
    ObsoleteEntry(
        'trader.py',
        ObsoleteStatus.DEAD,
        ArchiveState.MOVED,
        '1,975-line stale copy of core/trader.py. Missing _reject(), _sync_balance(), _get_live_open_pairs() and other safety methods present in the live version.',
        'Archived — renamed to trader.py.dead_duplicate_removed in Round-19 (this one actually was a rename, not a copy).',
        Confidence.MEDIUM,
        evidence='Inherited from prior audit; not re-checked this pass whether original trader.py at repo root still exists.',
    ),
    ObsoleteEntry(
        'strategy/multi_strategy_set.py',
        ObsoleteStatus.DEAD,
        ArchiveState.COPIED_ONLY,
        '410-line multi-strategy set with df.eval() rule engine. 0 importers.',
        'Marked legacy. Archive claim corrected — copy only, original still live-importable.',
        Confidence.HIGH,
        evidence='grep + checksum verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/pattern_strategies_ml.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of the strategies/ (plural) folder — 11 files, ~1,753 lines, all 0 importers.',
        "CORRECTED: 'All archived to .dead_code_archived in Round-26' is FALSE for the entire folder — not one of the 11 files has an archive copy. Split from the folder-level entry for per-file tracking.",
        Confidence.HIGH,
        evidence='`ls strategies/*.dead_code_archived` → no matches (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/scalping_strategy.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of strategies/ folder, 0 importers.',
        'See pattern_strategies_ml.py entry — same folder-wide false archive claim.',
        Confidence.HIGH,
        evidence='Same verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/ema_rsi_combo.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of strategies/ folder, 0 importers.',
        'See pattern_strategies_ml.py entry.',
        Confidence.HIGH,
        evidence='Same verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/reversal.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of strategies/ folder, 0 importers.',
        'See pattern_strategies_ml.py entry.',
        Confidence.HIGH,
        evidence='Same verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/trend_follow.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of strategies/ folder, 0 importers.',
        'See pattern_strategies_ml.py entry.',
        Confidence.HIGH,
        evidence='Same verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/breakout.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of strategies/ folder, 0 importers.',
        'See pattern_strategies_ml.py entry.',
        Confidence.HIGH,
        evidence='Same verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/momentum.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of strategies/ folder, 0 importers.',
        'See pattern_strategies_ml.py entry.',
        Confidence.HIGH,
        evidence='Same verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/retest.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of strategies/ folder, 0 importers.',
        'See pattern_strategies_ml.py entry.',
        Confidence.HIGH,
        evidence='Same verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/pullback.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of strategies/ folder, 0 importers. Had a confirmed copy-paste bug (bc→bear_c), fixed but never actually archived.',
        'See pattern_strategies_ml.py entry.',
        Confidence.HIGH,
        evidence='Same verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/mean_reversion.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of strategies/ folder, 0 importers.',
        'See pattern_strategies_ml.py entry.',
        Confidence.HIGH,
        evidence='Same verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'strategies/range_trading.py',
        ObsoleteStatus.DEAD,
        ArchiveState.NO_ARCHIVE,
        'Part of strategies/ folder, 0 importers.',
        'See pattern_strategies_ml.py entry.',
        Confidence.HIGH,
        evidence='Same verification (2026-07-22).',
    ),
    ObsoleteEntry(
        'intelligence/confidence_calibrator.py',
        ObsoleteStatus.ACTIVE,
        ArchiveState.NOT_APPLICABLE,
        'NEW ENTRY (Phase B2 discovery): this is the real, live confidence calibrator — unrelated to the dead memory/ and hybrid/ versions of the same class name.',
        'ACTIVE. Importer chain: agents/analysis_agent.py -> intelligence/confluence_engine.py -> intelligence/confidence_calibrator.py. Do not delete when cleaning up the memory/ or hybrid/ duplicates.',
        Confidence.HIGH,
        evidence='grep intelligence/confluence_engine.py:61 `from intelligence.confidence_calibrator import get_calibrator`; confluence_engine itself is imported by agents/analysis_agent.py:1477, core/runtime.py:414, core/trader.py:1799 (2026-07-22).',
    ),
    ObsoleteEntry(
        'intelligence/ported_indicators_registry.py',
        ObsoleteStatus.ORPHAN_READY,
        ArchiveState.NOT_APPLICABLE,
        'NEW ENTRY (Phase B2 discovery): complete implementation, zero runtime importers, no broken dependencies.',
        "Not currently wired into any live path. Candidate for wiring into AnalysisAgent if the ported-indicator set is wanted, or for deletion if it's abandoned — needs a product decision, not a bug fix.",
        Confidence.HIGH,
        evidence='grep: 0 importers found anywhere in repo (2026-07-22).',
    ),

]


def obsolete_index() -> Dict[str, ObsoleteEntry]:
    """Return a {path: entry} map for quick lookup."""
    return {e.path: e for e in OBSOLETE_MODULES}


def obsolete_summary() -> Dict[str, int]:
    """Counts per status — useful for the final report."""
    counts: Dict[str, int] = {}
    for entry in OBSOLETE_MODULES:
        counts[entry.status.value] = counts.get(entry.status.value, 0) + 1
    counts["total"] = len(OBSOLETE_MODULES)
    return counts


def archive_consistency_report() -> Dict[str, int]:
    """Counts per archive_state — flags how many entries still have a
    misleading 'archived' claim vs. their real filesystem state."""
    counts: Dict[str, int] = {}
    for entry in OBSOLETE_MODULES:
        counts[entry.archive_state.value] = counts.get(entry.archive_state.value, 0) + 1
    return counts


def is_obsolete(path: str) -> bool:
    return path in obsolete_index()