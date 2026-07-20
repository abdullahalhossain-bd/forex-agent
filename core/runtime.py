"""
core/runtime.py — Central runtime wiring (composition root)
============================================================

This is the single composition root that knows how to instantiate every
runtime module and register it with the ServiceRegistry / LifecycleManager.
It replaces the ad-hoc initialization in main.py's ForexAISystem class
(which only wired ~9 of the ~24 required services).

Each `boot_<phase>()` function:
  1. Imports its modules lazily so a broken import in one phase can't kill
     another.
  2. Registers services into the ServiceRegistry.
  3. Returns a PhaseResult.
  4. Publishes startup events on the EventBus.
  5. Records metrics via RuntimeMetrics.

The phases are wired in `register_default_phases()` and driven by
`LifecycleManager.boot()`.

Public API:
  * `Runtime` — top-level facade exposing registry, lifecycle, bus, metrics,
    health_monitor.
  * `boot_runtime()` — convenience: register all phases, boot, return Runtime.
  * `get_runtime()` — singleton accessor.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from core.event_bus import EventBus, get_bus
from core.health_monitor import HealthMonitor, get_health_monitor
from core.lifecycle import (
    LifecycleManager,
    Phase,
    PhaseResult,
    get_lifecycle,
)
from core.runtime_metrics import RuntimeMetrics, get_metrics
from core.service_registry import ServiceRegistry, ServiceStatus, get_registry

# ── Telegram polling guard — একবারের বেশি start হবে না ──────
_telegram_polling_started = False
_telegram_polling_lock = threading.Lock()

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Runtime:
    """Facade that bundles every runtime infrastructure service."""

    def __init__(self):
        self.registry: ServiceRegistry = get_registry()
        self.bus: EventBus = get_bus()
        self.lifecycle: LifecycleManager = get_lifecycle()
        self.metrics: RuntimeMetrics = get_metrics()
        self.health: HealthMonitor = get_health_monitor()
        self._booted = False

    def boot(self, until: Optional[Phase] = None) -> None:
        if self._booted:
            log.warning(
                "Runtime.boot() called on an already-booted Runtime — ignoring. "
                "Call shutdown() first if a re-boot is genuinely intended."
            )
            return
        register_default_phases(self.lifecycle)
        self.lifecycle.boot(until=until)
        self._booted = True
        try:
            self.health.start()
        except Exception as e:
            log.warning("Health monitor failed to start: %s", e)
        self._start_metrics_publisher()

    def _start_metrics_publisher(self) -> None:
        def _push():
            while True:
                try:
                    self.metrics.snapshot_to_bus()
                except Exception as e:
                    log.debug("metrics publisher error: %s", e)
                time.sleep(30)
        t = threading.Thread(target=_push, name="metrics-publisher", daemon=True)
        t.start()

    def shutdown(self) -> None:
        try:
            self.health.stop()
        except Exception as e:
            log.warning("Suppressed exception while stopping health monitor: %s", e)
            pass
        self.bus.publish("system.shutdown", {"ts": time.time()}, source="runtime")
        self.lifecycle.shutdown()
        self._booted = False

    def is_booted(self) -> bool:
        return self._booted

    def status(self) -> dict:
        return {
            "booted": self._booted,
            "phases": self.lifecycle.report(),
            "health": self.health.status(),
            "metrics": self.metrics.build_report(),
            "bus": self.bus.stats(),
        }


_RUNTIME: Optional[Runtime] = None


def get_runtime() -> Runtime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = Runtime()
    return _RUNTIME


# ─────────────────────────────────────────────────────────────────────
# Phase boot functions
# ─────────────────────────────────────────────────────────────────────


def boot_bootstrap(registry: ServiceRegistry) -> PhaseResult:
    """Phase 1 — paths, config, logging, event bus, metrics, registry."""
    from config import Config, EXECUTION_MODE, APPROVAL_MODE, USE_SCANNER, SYMBOLS, DEFAULT_TIMEFRAME
    from core.constants import LOGS_DIR, MEMORY_DIR, DATABASE_DIR, BACKUPS_DIR, REPORTS_DIR, DATA_DIR

    for d in (LOGS_DIR, MEMORY_DIR, DATABASE_DIR, BACKUPS_DIR, REPORTS_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    registry.register_instance("config", Config())
    registry.register_instance("paths", {
        "logs": LOGS_DIR, "memory": MEMORY_DIR, "db": DATABASE_DIR,
        "backups": BACKUPS_DIR, "reports": REPORTS_DIR, "data": DATA_DIR,
    })
    if not registry.has("execution_mode"):
        registry.register_instance("execution_mode", EXECUTION_MODE)
    if not registry.has("approval_mode"):
        registry.register_instance("approval_mode", APPROVAL_MODE)
    if not registry.has("use_scanner"):
        registry.register_instance("use_scanner", USE_SCANNER)
    if not registry.has("symbols"):
        registry.register_instance("symbols", list(SYMBOLS))
    if not registry.has("timeframe"):
        registry.register_instance("timeframe", DEFAULT_TIMEFRAME)
    registry.register_instance("event_bus", get_bus())
    registry.register_instance("metrics", get_metrics())
    registry.register_instance("health_monitor", get_health_monitor())

    try:
        from core.llm_key_manager import LLMKeyManager, get_llm_key_manager
        manager = get_llm_key_manager()
        registry.register_instance("llm_key_manager", manager)
        status = manager.status()
        log.info("LLM KeyManager wired: Groq=%d keys (%d active), Gemini=%d keys (%d active)",
                 status["groq"]["total"], status["groq"]["available"],
                 status["gemini"]["total"], status["gemini"]["available"])
    except Exception as e:
        log.warning("LLMKeyManager init failed: %s", e)

    get_bus().publish("system.startup", {"phase": "bootstrap"}, source="runtime")
    return PhaseResult(
        phase=Phase.BOOTSTRAP, ok=True, duration_sec=0.0,
        services_registered=["config", "paths", "event_bus", "metrics", "health_monitor"],
    )


def boot_persistence(registry: ServiceRegistry) -> PhaseResult:
    """Phase 2 — TraderDB + memory stores + KnowledgeStore.

    CRITICAL FIX: Previously always returned ok=True even if DB failed.
    Now returns ok=False if critical services (db) fail to initialize.
    Non-critical services (trade_memory, learning_engine) don't block boot
    but are logged as warnings.
    """
    services = []
    failed_critical = []

    # ── CRITICAL: TraderDB — required for trade journaling, duplicate
    # protection, and orphan cleanup. If this fails, DO NOT proceed.
    try:
        from database.db import TraderDB
        db = TraderDB()
        registry.register_instance("db", db)
        services.append("db")
    except Exception as e:
        log.error("CRITICAL: TraderDB init failed: %s", e)
        registry.mark("db", ServiceStatus.FAILED, str(e))
        failed_critical.append("db")

    # ── NON-CRITICAL: memory/learning services — degraded mode OK
    try:
        from memory.trade_memory import TradeMemory
        tm = TradeMemory(seed_rules=True)
        registry.register_instance("trade_memory", tm)
        services.append("trade_memory")
    except Exception as e:
        log.warning("TradeMemory init failed (non-critical): %s", e)

    try:
        from memory.learning import LearningEngine
        registry.register("learning_engine", lambda r: LearningEngine())
        services.append("learning_engine")
    except Exception as e:
        log.warning("LearningEngine init failed (non-critical): %s", e)

    try:
        from memory.history import AnalysisHistory
        registry.register("analysis_history", lambda r: AnalysisHistory())
        services.append("analysis_history")
    except Exception as e:
        log.warning("AnalysisHistory init failed (non-critical): %s", e)

    try:
        from memory.knowledge_store import KnowledgeStore
        registry.register("knowledge_store", lambda r: KnowledgeStore())
        services.append("knowledge_store")
    except Exception as e:
        log.warning("KnowledgeStore init failed (non-critical): %s", e)

    get_bus().publish("system.startup", {"phase": "persistence", "services": services}, source="runtime")

    # FIX: return ok=False if critical services failed
    if failed_critical:
        log.error("boot_persistence FAILED — critical services down: %s", failed_critical)
        return PhaseResult(phase=Phase.PERSISTENCE, ok=False, duration_sec=0.0,
                           services_registered=services,
                           error=f"Critical services failed: {failed_critical}")
    return PhaseResult(phase=Phase.PERSISTENCE, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_data(registry: ServiceRegistry) -> PhaseResult:
    """Phase 3 — DataFetcher, DataValidator, Indicators, AutomatedUpdater."""
    services = []
    try:
        from data.fetcher import get_data_fetcher
        registry.register_instance("data_fetcher", get_data_fetcher())
        services.append("data_fetcher")
    except Exception as e:
        log.error("DataFetcher init failed: %s", e)

    try:
        from data.validator import DataValidator
        registry.register_instance("data_validator", DataValidator())
        services.append("data_validator")
    except Exception as e:
        log.error("DataValidator init failed: %s", e)

    try:
        from data.indicators import Indicators
        registry.register_instance("indicators", Indicators())
        services.append("indicators")
    except Exception as e:
        log.error("Indicators init failed: %s", e)

    try:
        from data.automated_updater import data_updater
        registry.register_instance("data_updater", data_updater)
        services.append("data_updater")
    except Exception as e:
        log.error("data_updater init failed: %s", e)

    get_bus().publish("system.startup", {"phase": "data", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.DATA, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_market(registry: ServiceRegistry) -> PhaseResult:
    """Phase 4 — MarketScanner + CorrelationFilter + OpportunityRanker + MT5Connection."""
    services = []
    try:
        from scanner.market_scanner import MarketScanner
        from scanner.correlation_filter import CorrelationFilter
        from scanner.opportunity_ranker import OpportunityRanker
        registry.register("market_scanner", lambda r: MarketScanner(risk_engine=r.try_resolve("risk_engine")))
        registry.register_instance("correlation_filter", CorrelationFilter())
        registry.register_instance("opportunity_ranker", OpportunityRanker())
        services.extend(["market_scanner", "correlation_filter", "opportunity_ranker"])
    except Exception as e:
        log.error("Scanner init failed: %s", e)

    # Day 131 fix: the previous fallback default here was the literal
    # string "paper" — but config.py explicitly documents that
    # ExecutionRouter (and therefore the rest of the system) only
    # supports "mt5_demo"; "paper" is a removed legacy mode. If
    # "execution_mode" was ever somehow missing from the registry (e.g. a
    # future refactor reorders phases, or boot_bootstrap partially fails),
    # this fallback would silently skip MT5Connection creation even though
    # the real, intended mode is mt5_demo — which is exactly how a
    # "No shared MT5Connection in registry" report could occur with no
    # obvious error at boot. Falling back to config's own EXECUTION_MODE
    # keeps this phase self-consistent with the rest of the app instead of
    # guessing an unsupported value.
    try:
        from config import EXECUTION_MODE as _CONFIG_EXECUTION_MODE
    except Exception:
        _CONFIG_EXECUTION_MODE = "mt5_demo"
    mode = registry.get("execution_mode", _CONFIG_EXECUTION_MODE)

    if mode == "mt5_demo":
        # Day 131 fix: MT5Connection creation used to be a single
        # try/once — any transient failure at boot (terminal still
        # starting up, brief network hiccup, etc.) permanently left the
        # registry without "mt5_connection" for the rest of the process,
        # with no retry and no distinguishable health signal beyond a log
        # line. This is one of the root causes behind persistent
        # "No shared MT5Connection in registry" symptoms even though MT5
        # was reachable moments later. Now retries with backoff and
        # records an explicit FAILED/DEGRADED status in the registry so
        # health() and monitoring can see it, rather than only a log line.
        import time as _time
        from broker.mt5_connection import get_mt5_connection
        from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH

        max_attempts = 3
        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                # Bug fix: was `MT5Connection(...)` — a brand-new instance
                # every boot/retry that independently re-ran mt5.initialize()
                # + mt5.login(), even though data/fetcher.py,
                # data/data_orchestrator.py, and execution_router.py may
                # build their own MT5Connection too if not given this one.
                # get_mt5_connection() ensures all of them converge on the
                # SAME already-connected instance for this (login, server).
                conn = get_mt5_connection(login=MT5_LOGIN, password=MT5_PASSWORD,
                                          server=MT5_SERVER, path=MT5_PATH or None,
                                          auto_connect=False)
                registry.register_instance("mt5_connection", conn)
                services.append("mt5_connection")
                last_err = None
                break
            except Exception as e:
                last_err = e
                log.warning(
                    "MT5 connection init attempt %d/%d failed: %s",
                    attempt, max_attempts, e,
                )
                if attempt < max_attempts:
                    _time.sleep(min(2 ** attempt, 10))
        if last_err is not None:
            log.error("MT5 connection init failed after %d attempts: %s", max_attempts, last_err)
            registry.register_failed(
                "mt5_connection",
                error=f"MT5 connection unavailable after {max_attempts} attempts: {last_err}",
            )
            get_bus().publish(
                "system.degraded",
                {"phase": "market", "service": "mt5_connection", "error": str(last_err)},
                source="runtime",
            )

    get_bus().publish("system.startup", {"phase": "market", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.MARKET, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_research(registry: ServiceRegistry) -> PhaseResult:
    """Phase 5 — ResearchAgent + HypothesisEngine + ExperimentRunner + Reports."""
    services = []
    try:
        from research.research_agent import ResearchAgent
        registry.register("research_agent", lambda r: ResearchAgent())
        services.append("research_agent")
    except Exception as e:
        log.warning("ResearchAgent not available: %s", e)

    try:
        from research.hypothesis_engine import HypothesisEngine
        registry.register("hypothesis_engine", lambda r: HypothesisEngine())
        services.append("hypothesis_engine")
    except Exception as e:
        log.warning("HypothesisEngine not available: %s", e)

    try:
        from research.experiment_runner import ExperimentRunner
        registry.register("experiment_runner", lambda r: ExperimentRunner())
        services.append("experiment_runner")
    except Exception as e:
        log.warning("ExperimentRunner not available: %s", e)

    try:
        from research.research_report import ResearchReportGenerator
        registry.register("research_report", lambda r: ResearchReportGenerator())
        services.append("research_report")
    except Exception as e:
        log.warning("ResearchReportGenerator not available: %s", e)

    get_bus().publish("system.startup", {"phase": "research", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.RESEARCH, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_fundamental(registry: ServiceRegistry) -> PhaseResult:
    """Phase 6 — NewsFilter + Day 66 NewsIntelligence engine."""
    services = []
    try:
        from fundamental.news_filter import NewsFilter
        registry.register_instance("news_filter", NewsFilter())
        services.append("news_filter")
    except Exception as e:
        log.error("NewsFilter init failed: %s", e)

    try:
        from fundamental.fundamental_sentiment import FundamentalSentimentScore
        registry.register("fundamental_sentiment", lambda r: FundamentalSentimentScore())
        services.append("fundamental_sentiment")
    except Exception as e:
        log.warning("FundamentalSentimentScore not available: %s", e)

    try:
        from intelligence.news_ai import NewsIntelligence, get_news_intelligence
        from config import SYMBOLS
        news_ai = get_news_intelligence(pairs=list(SYMBOLS))
        registry.register_instance("news_intelligence", news_ai)
        services.append("news_intelligence")
        log.info("Day 66 NewsIntelligence engine wired")
    except Exception as e:
        log.warning("NewsIntelligence init failed: %s", e)

    get_bus().publish("system.startup", {"phase": "fundamental", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.FUNDAMENTAL, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_analysis(registry: ServiceRegistry) -> PhaseResult:
    """Phase 7 — Analysis engines + ML pipeline (Day 67-72)."""
    services = []
    try:
        from analysis.intermarket import IntermarketEngine
        registry.register_instance("intermarket_engine", IntermarketEngine())
        services.append("intermarket_engine")
    except Exception as e:
        log.warning("IntermarketEngine init failed: %s", e)

    try:
        from analysis.session_analyzer import SessionAnalyzer
        registry.register_instance("session_analyzer_analytics", SessionAnalyzer())
        services.append("session_analyzer_analytics")
    except Exception as e:
        log.warning("SessionAnalyzer init failed: %s", e)

    try:
        from intelligence.confluence_engine import ConfluenceEngine, get_confluence_engine
        engine = get_confluence_engine()
        registry.register_instance("confluence_engine", engine)
        services.append("confluence_engine")
        log.info("Day 67 ConfluenceEngine wired")
    except Exception as e:
        log.warning("ConfluenceEngine init failed: %s", e)

    try:
        from ml.feature_engineer import FeatureEngineer, get_feature_engineer
        registry.register_instance("feature_engineer", get_feature_engineer())
        services.append("feature_engineer")
    except Exception as e:
        log.warning("FeatureEngineer init failed: %s", e)

    try:
        from ml.feature_store import FeatureStore, get_feature_store
        store = get_feature_store()
        registry.register_instance("feature_store", store)
        services.append("feature_store")
        stats = store.stats()
        log.info("Day 68 FeatureStore wired (%d feature rows, %d labels)",
                 stats.get("total_feature_rows", 0), stats.get("total_labels", 0))
    except Exception as e:
        log.warning("FeatureStore init failed: %s", e)

    try:
        from ml.label_generator import LabelGenerator, get_label_generator
        registry.register_instance("label_generator", get_label_generator())
        services.append("label_generator")
    except Exception as e:
        log.warning("LabelGenerator init failed: %s", e)

    try:
        from ml.data_preprocessor import DataPreprocessor, get_preprocessor
        registry.register_instance("data_preprocessor", get_preprocessor())
        services.append("data_preprocessor")
    except Exception as e:
        log.warning("DataPreprocessor init failed: %s", e)

    try:
        from ml.feature_selector import FeatureSelector, get_feature_selector
        registry.register_instance("feature_selector", get_feature_selector())
        services.append("feature_selector")
    except Exception as e:
        log.warning("FeatureSelector init failed: %s", e)

    try:
        from ml.dataset_builder import DatasetBuilder, get_dataset_builder
        registry.register_instance("dataset_builder", get_dataset_builder())
        services.append("dataset_builder")
    except Exception as e:
        log.warning("DatasetBuilder init failed: %s", e)

    try:
        from ml.model_evaluator import ModelEvaluator, get_evaluator
        registry.register_instance("model_evaluator", get_evaluator())
        services.append("model_evaluator")
    except Exception as e:
        log.warning("ModelEvaluator init failed: %s", e)

    try:
        from ml.model_store import ModelStore, get_model_store
        store = get_model_store()
        registry.register_instance("model_store", store)
        services.append("model_store")
        models = store.list_models()
        log.info("Day 69 ModelStore wired (%d models on disk)", len(models))
    except Exception as e:
        log.warning("ModelStore init failed: %s", e)

    try:
        from ml.model_trainer import ModelTrainer, get_model_trainer
        registry.register_instance("model_trainer", get_model_trainer())
        services.append("model_trainer")
    except Exception as e:
        log.warning("ModelTrainer init failed: %s", e)

    try:
        from ml.model_predictor import ModelPredictor, get_model_predictor
        predictor = get_model_predictor()
        registry.register_instance("model_predictor", predictor)
        services.append("model_predictor")
        log.info("Day 69 ModelPredictor wired (ensemble prediction ready)")
    except Exception as e:
        log.warning("ModelPredictor init failed: %s", e)

    try:
        from ml.voting_engine import VotingEngine, get_voting_engine
        registry.register_instance("voting_engine", get_voting_engine())
        services.append("voting_engine")
    except Exception as e:
        log.warning("VotingEngine init failed: %s", e)

    try:
        from ml.confidence_fusion import ConfidenceFusion, get_confidence_fusion
        registry.register_instance("confidence_fusion", get_confidence_fusion())
        services.append("confidence_fusion")
    except Exception as e:
        log.warning("ConfidenceFusion init failed: %s", e)

    try:
        from ml.ensemble_store import EnsembleStore, get_ensemble_store
        store = get_ensemble_store()
        registry.register_instance("ensemble_store", store)
        services.append("ensemble_store")
    except Exception as e:
        log.warning("EnsembleStore init failed: %s", e)

    try:
        from ml.ensemble import EnsembleEngine, get_ensemble_engine
        engine = get_ensemble_engine()
        registry.register_instance("ensemble_engine", engine)
        services.append("ensemble_engine")
        log.info("Day 70 EnsembleEngine wired (brain fusion layer ready)")
    except Exception as e:
        log.warning("EnsembleEngine init failed: %s", e)

    try:
        from ml.reward_engine import RewardEngine, get_reward_engine
        registry.register_instance("reward_engine", get_reward_engine())
        services.append("reward_engine")
    except Exception as e:
        log.warning("RewardEngine init failed: %s", e)

    try:
        from ml.rl_agent import RLAgent, get_rl_agent
        agent = get_rl_agent()
        registry.register_instance("rl_agent", agent)
        services.append("rl_agent")
        log.info("Day 71 RL Agent wired (source=%s, model_loaded=%s)",
                 agent.status().get("source", "heuristic"),
                 agent.status().get("model_loaded", False))
    except Exception as e:
        log.warning("RLAgent init failed: %s", e)

    try:
        from ml.rl_policy_store import RLPolicyStore, get_rl_policy_store
        store = get_rl_policy_store()
        registry.register_instance("rl_policy_store", store)
        services.append("rl_policy_store")
    except Exception as e:
        log.warning("RLPolicyStore init failed: %s", e)

    try:
        from ml.cv_splitter import PurgedEmbargoedSplitter, get_cv_splitter
        registry.register_instance("cv_splitter", get_cv_splitter())
        services.append("cv_splitter")
    except Exception as e:
        log.warning("PurgedEmbargoedSplitter init failed: %s", e)

    try:
        from ml.walk_forward import WalkForwardValidator, get_walk_forward_validator
        registry.register_instance("walk_forward_validator", get_walk_forward_validator())
        services.append("walk_forward_validator")
    except Exception as e:
        log.warning("WalkForwardValidator init failed: %s", e)

    try:
        from ml.monte_carlo import MonteCarloSimulator, get_monte_carlo_simulator
        registry.register_instance("monte_carlo_simulator", get_monte_carlo_simulator())
        services.append("monte_carlo_simulator")
    except Exception as e:
        log.warning("MonteCarloSimulator init failed: %s", e)

    try:
        from ml.regime_test import RegimeTester, get_regime_tester
        registry.register_instance("regime_tester", get_regime_tester())
        services.append("regime_tester")
    except Exception as e:
        log.warning("RegimeTester init failed: %s", e)

    try:
        from ml.sensitivity_test import SensitivityTester, get_sensitivity_tester
        registry.register_instance("sensitivity_tester", get_sensitivity_tester())
        services.append("sensitivity_tester")
    except Exception as e:
        log.warning("SensitivityTester init failed: %s", e)

    try:
        from ml.validation import ValidationEngine, get_validation_engine
        engine = get_validation_engine()
        registry.register_instance("validation_engine", engine)
        services.append("validation_engine")
        stats = engine.stats()
        log.info("Day 72 ValidationEngine wired (validated=%d approved=%d champion=%s)",
                 stats.get("total_validated", 0), stats.get("approved", 0),
                 stats.get("champion", {}).get("model_name", "none") if stats.get("champion") else "none")
    except Exception as e:
        log.warning("ValidationEngine init failed: %s", e)

    get_bus().publish("system.startup", {"phase": "analysis", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.ANALYSIS, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_ai(registry: ServiceRegistry) -> PhaseResult:
    """Phase 8 — AIAnalyst + MasterAnalyst + ModelVersionManager."""
    services = []
    try:
        from ai.ai_analyst import AIAnalyst
        registry.register_instance("ai_analyst", AIAnalyst())
        services.append("ai_analyst")
    except Exception as e:
        log.error("AIAnalyst init failed: %s", e)

    try:
        from agents.master_analyst import MasterAnalyst
        registry.register_instance("master_analyst", MasterAnalyst())
        services.append("master_analyst")
    except Exception as e:
        log.warning("MasterAnalyst init failed: %s", e)

    try:
        from ai.model_versioning import model_manager
        registry.register_instance("model_manager", model_manager)
        services.append("model_manager")
    except Exception as e:
        log.warning("ModelVersionManager not available (optional): %s", e)

    get_bus().publish("system.startup", {"phase": "ai", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.AI, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_agents(registry: ServiceRegistry) -> PhaseResult:
    """Phase 9 — Market/Analysis/Decision/Learning/Risk agents."""
    services = []
    try:
        from agents.market_agent import MarketAgent
        from agents.analysis_agent import AnalysisAgent
        from agents.decision_agent import DecisionAgent
        from agents.learning_agent import LearningAgent
        from agents.risk_agent import RiskAgent
        registry.register("market_agent_class", lambda r: MarketAgent)
        registry.register("analysis_agent_class", lambda r: AnalysisAgent)
        registry.register("decision_agent_class", lambda r: DecisionAgent)
        registry.register("learning_agent_class", lambda r: LearningAgent)
        registry.register("risk_agent_class", lambda r: RiskAgent)
        services.extend(["market_agent_class", "analysis_agent_class",
                         "decision_agent_class", "learning_agent_class",
                         "risk_agent_class"])
    except Exception as e:
        log.error("Agent class registration failed: %s", e)

    get_bus().publish("system.startup", {"phase": "agents", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.AGENTS, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_strategy(registry: ServiceRegistry) -> PhaseResult:
    """Phase 10 — SignalEngine + strategy package."""
    services = []
    try:
        from strategy.signal_engine import SignalEngine
        registry.register_instance("signal_engine", SignalEngine())
        services.append("signal_engine")
    except Exception as e:
        log.error("SignalEngine init failed: %s", e)

    try:
        import strategies
        registry.register_instance("strategies_package", strategies)
        services.append("strategies_package")
    except Exception as e:
        log.warning("strategies package not available: %s", e)

    get_bus().publish("system.startup", {"phase": "strategy", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.STRATEGY, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_hybrid(registry: ServiceRegistry) -> PhaseResult:
    """Phase 11 — hybrid FlowController.

    FIX: FlowController requires 5+ constructor args (market_agent_factory,
    analysis_agent, risk_agent, decision_agent, execution_router). Calling
    FlowController() with zero args always crashes. The entire hybrid/
    folder is legacy/dead code — the system uses core/trader.py and
    agents/* instead.

    This phase is now a no-op. The hybrid/ folder should be moved to legacy/.
    """
    services = []
    # Log that hybrid/ is intentionally skipped — it's dead code.
    log.info("boot_hybrid: skipped (hybrid/ is legacy — system uses core/trader.py)")
    get_bus().publish("system.startup", {"phase": "hybrid", "services": services,
                                          "note": "skipped — legacy module"}, source="runtime")
    return PhaseResult(phase=Phase.HYBRID, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_risk(registry: ServiceRegistry) -> PhaseResult:
    """Phase 12 — RiskEngine + CircuitBreaker + TradePermission + DrawdownController.

    Day 75+76 additions: Kill Switch (3-level emergency brake), Live Risk
    Manager (central pre-trade gate), Exposure Manager (correlation groups),
    Drawdown Monitor (capital preservation), Risk Reporter (event log),
    plus the Day 76 Smart Capital Allocation Engine — Kelly Calculator,
    Volatility Adjuster, Confidence Scaler, Correlation Manager, and the
    master PositionSizer that fuses them all.
    """
    services = []
    core_risk_failed = False
    core_risk_error = None
    try:
        from risk.risk_engine import RiskEngine
        from risk.circuit_breaker import CircuitBreaker
        from risk.trade_permission import TradePermission
        from risk.drawdown_controller import DrawdownController

        symbols = registry.get("symbols", ["EURUSD", "GBPUSD", "USDJPY"])
        # Audit fix: this was hardcoded to 10000 regardless of the account's
        # real configured balance, while LiveRiskManager (registered later
        # in this same phase) correctly reads INITIAL_BALANCE from config.
        # That mismatch meant CircuitBreaker/DrawdownController/RiskEngine
        # loss and drawdown thresholds were silently computed against a
        # fictitious $10,000 balance on any account that isn't exactly
        # that size. Prefer an explicit registry override (if the caller
        # set one), then config.INITIAL_BALANCE, then fall back to the
        # previous 10000 default only if config is unavailable.
        try:
            from config import INITIAL_BALANCE as _CONFIG_INITIAL_BALANCE
        except Exception as e:
            log.warning("Risk phase: could not import INITIAL_BALANCE from config (%s) — "
                        "using 10000 fallback", e)
            _CONFIG_INITIAL_BALANCE = 10000
        balance = registry.get("balance", _CONFIG_INITIAL_BALANCE)
        cb = CircuitBreaker(balance=balance)
        # Bug fix: this was still calling DrawdownController() with no
        # argument, so it silently fell back to its own default
        # (initial_balance=10000.0) regardless of the real account balance
        # computed just above — while CircuitBreaker right next to it
        # correctly received `balance`. On any account that isn't exactly
        # $10,000 (e.g. a $99,170.98 live/demo balance), every drawdown %
        # DrawdownController computed was measured against a fictitious
        # $10,000 peak, so ordinary balance fluctuations could trigger
        # false GREEN/YELLOW/ORANGE/RED transitions and false daily/weekly
        # limit trips. Pass the same resolved `balance` here too.
        dd = DrawdownController(initial_balance=balance)
        registry.register_instance("circuit_breaker", cb)
        registry.register_instance("drawdown_controller", dd)
        registry.register_instance("trade_permission", TradePermission())
        registry.register("risk_engine_factory",
                          lambda r: (lambda symbol, balance=balance:
                                     RiskEngine(balance=balance, symbol=symbol)))
        services.extend(["circuit_breaker", "drawdown_controller",
                         "trade_permission", "risk_engine_factory"])
    except Exception as e:
        # CRITICAL: RiskEngine/CircuitBreaker/TradePermission/DrawdownController
        # are the pre-trade risk gates. If they fail to load, the system must
        # NOT report a successful boot and proceed to trade ungated — that
        # would mean live orders can execute with no circuit breaker and no
        # drawdown control. Same failure-reporting pattern as boot_persistence.
        log.critical("CRITICAL: Risk core init failed — trading must not proceed "
                     "ungated: %s", e, exc_info=True)
        core_risk_failed = True
        core_risk_error = str(e)

    try:
        from risk.autonomous_risk import AutonomousRiskManager
        # Bug fix: same $10,000-fallback issue as DrawdownController above —
        # AutonomousRiskManager() with no args defaults to balance=10000.0,
        # which also seeds its own internal DrawdownController with the
        # wrong initial balance. Reuse the resolved balance from the risk
        # phase above; `balance` may not exist if that block raised before
        # reaching its assignment, so resolve it again defensively here.
        try:
            _arm_balance = balance
        except NameError:
            try:
                from config import INITIAL_BALANCE as _CONFIG_INITIAL_BALANCE
            except Exception:
                _CONFIG_INITIAL_BALANCE = 10000
            _arm_balance = registry.get("balance", _CONFIG_INITIAL_BALANCE)
        registry.register("autonomous_risk_manager",
                          lambda r, _b=_arm_balance: AutonomousRiskManager(balance=_b))
        services.append("autonomous_risk_manager")
    except Exception as e:
        log.warning("AutonomousRiskManager not available: %s", e)

    # ── Day 75 — Live Risk Framework ──────────────────────────────────
    # 3-level kill switch, exposure/correlation manager, drawdown monitor
    # (capital preservation mode), risk event reporter (DB-backed), and
    # the central LiveRiskManager that runs all 6 checks pre-trade.
    try:
        from risk.kill_switch import KillSwitch, get_kill_switch
        ks = get_kill_switch()
        registry.register_instance("kill_switch", ks)
        services.append("kill_switch")
    except Exception as e:
        log.warning("KillSwitch init failed: %s", e)

    try:
        from risk.exposure_manager import ExposureManager, get_exposure_manager
        registry.register_instance("exposure_manager", get_exposure_manager())
        services.append("exposure_manager")
    except Exception as e:
        log.warning("ExposureManager init failed: %s", e)

    try:
        from risk.drawdown_monitor import DrawdownMonitor, get_drawdown_monitor
        registry.register_instance("drawdown_monitor", get_drawdown_monitor())
        services.append("drawdown_monitor")
    except Exception as e:
        log.warning("DrawdownMonitor init failed: %s", e)

    try:
        from risk.risk_reporter import RiskReporter, get_risk_reporter
        registry.register_instance("risk_reporter", get_risk_reporter())
        services.append("risk_reporter")
    except Exception as e:
        log.warning("RiskReporter init failed: %s", e)

    try:
        from risk.live_risk_manager import LiveRiskManager, get_live_risk_manager
        mgr = get_live_risk_manager()
        # Pick up the live config balance if available, otherwise default.
        try:
            from config import INITIAL_BALANCE
            mgr.initial_balance = float(INITIAL_BALANCE)
        except Exception as e:
            log.warning("Suppressed exception while setting LiveRiskManager initial_balance: %s", e)
            pass
        registry.register_instance("live_risk_manager", mgr)
        services.append("live_risk_manager")
        log.info("Day 75 LiveRiskManager wired (tier=%d, mode=%s)",
                 mgr.current_tier.tier, mgr.drawdown_monitor.status().get("mode", "NORMAL"))
    except Exception as e:
        log.warning("LiveRiskManager init failed: %s", e)

    # ── Day 76 — Smart Capital Allocation Engine ──────────────────────
    # Kelly Criterion + Volatility adjustment + Confidence scaling +
    # Correlation management, fused inside the master PositionSizer.
    try:
        from risk.kelly_calculator import KellyCalculator, get_kelly_calculator
        registry.register_instance("kelly_calculator", get_kelly_calculator())
        services.append("kelly_calculator")
    except Exception as e:
        log.warning("KellyCalculator init failed: %s", e)

    try:
        from risk.volatility_adjuster import VolatilityAdjuster, get_volatility_adjuster
        registry.register_instance("volatility_adjuster", get_volatility_adjuster())
        services.append("volatility_adjuster")
    except Exception as e:
        log.warning("VolatilityAdjuster init failed: %s", e)

    try:
        from risk.confidence_scaler import ConfidenceScaler, get_confidence_scaler
        registry.register_instance("confidence_scaler", get_confidence_scaler())
        services.append("confidence_scaler")
    except Exception as e:
        log.warning("ConfidenceScaler init failed: %s", e)

    try:
        from risk.correlation_manager import CorrelationManager, get_correlation_manager
        registry.register_instance("correlation_manager", get_correlation_manager())
        services.append("correlation_manager")
    except Exception as e:
        log.warning("CorrelationManager init failed: %s", e)

    try:
        from risk.position_sizer import PositionSizer, get_position_sizer
        registry.register_instance("position_sizer", get_position_sizer())
        services.append("position_sizer")
        log.info("Day 76 PositionSizer wired (Kelly×Vol×Conf×Corr×DD×Streak fusion)")
    except Exception as e:
        log.warning("PositionSizer init failed: %s", e)

    get_bus().publish("system.startup", {"phase": "risk", "services": services}, source="runtime")
    if core_risk_failed:
        return PhaseResult(phase=Phase.RISK, ok=False, duration_sec=0.0,
                           services_registered=services,
                           error=f"Critical risk core failed: {core_risk_error}")
    return PhaseResult(phase=Phase.RISK, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_safety(registry: ServiceRegistry) -> PhaseResult:
    """Phase 13 — SafetyGuard + SpreadMonitor."""
    services = []
    safety_guard_failed = False
    safety_guard_error = None
    try:
        from broker.safety_guard import SafetyGuard
        registry.register("safety_guard", lambda r: SafetyGuard(paper_trader=r.try_resolve("paper_trader")))
        services.append("safety_guard")
    except Exception as e:
        # CRITICAL: SafetyGuard is the primary pre-trade safety gate. Report
        # ok=False rather than silently proceeding without it — same
        # reasoning as the risk-core failure in boot_risk above.
        log.critical("CRITICAL: SafetyGuard init failed — trading must not "
                     "proceed without this gate: %s", e, exc_info=True)
        safety_guard_failed = True
        safety_guard_error = str(e)

    try:
        from broker.spread_monitor import SpreadMonitor
        registry.register_instance("spread_monitor", SpreadMonitor())
        services.append("spread_monitor")
    except Exception as e:
        log.warning("SpreadMonitor not available (non-critical): %s", e)

    get_bus().publish("system.startup", {"phase": "safety", "services": services}, source="runtime")
    if safety_guard_failed:
        return PhaseResult(phase=Phase.SAFETY, ok=False, duration_sec=0.0,
                           services_registered=services,
                           error=f"SafetyGuard failed: {safety_guard_error}")
    return PhaseResult(phase=Phase.SAFETY, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_execution(registry: ServiceRegistry) -> PhaseResult:
    """Phase 14 — ExecutionRouter (MT5 demo only, paper trading removed).

    Day 90+ hotfix: এখন registry থেকে shared "mt5_connection" instance
    (Phase 4 / boot_market-এ register হওয়া) resolve করে ExecutionRouter-এ
    inject করা হয়। আগে ExecutionRouter নিজেই আলাদা MT5Connection()
    বানাতো — ফলে একই process-এ দুটো mt5.initialize() session চলতো,
    যেটা একে অপরের connection ভেঙে দিয়ে বারবার "MT5 connection lost"
    flapping তৈরি করছিল। এখন একটাই connection object পুরো system জুড়ে
    শেয়ার হয়।
    """
    services = []
    router_failed = False
    router_error = None

    try:
        from execution.execution_router import ExecutionRouter
        db = registry.try_resolve("db")
        shared_mt5_conn = registry.try_resolve("mt5_connection")
        router = ExecutionRouter(db=db, mt5_conn=shared_mt5_conn)
        registry.register_instance("execution_router", router)
        services.append("execution_router")
        if shared_mt5_conn is not None:
            log.info("ExecutionRouter wired in MT5_DEMO mode (using shared MT5 connection)")
        else:
            log.warning(
                "ExecutionRouter wired in MT5_DEMO mode — no shared mt5_connection "
                "found in registry, router created its own connection (check boot_market phase)"
            )
    except Exception as e:
        # CRITICAL: ExecutionRouter is the only path orders reach the broker.
        # If it fails to construct, nothing downstream can trade — report
        # ok=False instead of letting boot appear to succeed.
        log.critical("CRITICAL: ExecutionRouter init failed — no order path "
                     "exists: %s", e, exc_info=True)
        router_failed = True
        router_error = str(e)

    get_bus().publish("system.startup", {"phase": "execution", "services": services}, source="runtime")
    if router_failed:
        return PhaseResult(phase=Phase.EXECUTION, ok=False, duration_sec=0.0,
                           services_registered=services,
                           error=f"ExecutionRouter failed: {router_error}")
    return PhaseResult(phase=Phase.EXECUTION, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_broker(registry: ServiceRegistry) -> PhaseResult:
    """Phase 15 — broker subsystem (mt5_demo only).

    Day 90+ note: account_manager/order_manager এখানেও registry-র shared
    "mt5_connection" ব্যবহার করে — কোনো নতুন MT5Connection বানানো হয় না।
    এটা আগে থেকেই ঠিক ছিল (registry.try_resolve ব্যবহার করতো), তাই এই
    phase-এ কোনো পরিবর্তনের দরকার ছিল না।
    """
    services = []
    # Day 131 fix: same issue as boot_market() — "paper" is not a
    # supported EXECUTION_MODE (see config.py), so falling back to it here
    # would silently skip broker wiring if "execution_mode" were ever
    # missing from the registry. Fall back to config's own EXECUTION_MODE
    # instead, keeping this phase self-consistent with boot_market().
    try:
        from config import EXECUTION_MODE as _CONFIG_EXECUTION_MODE
    except Exception:
        _CONFIG_EXECUTION_MODE = "mt5_demo"
    mode = registry.get("execution_mode", _CONFIG_EXECUTION_MODE)
    if mode != "mt5_demo":
        get_bus().publish("system.startup", {"phase": "broker", "skipped": True}, source="runtime")
        return PhaseResult(phase=Phase.BROKER, ok=True, duration_sec=0.0,
                           services_registered=[], skipped=True)

    account_broker_failed = False
    account_broker_error = None
    om = None

    # ── CRITICAL: AccountManager + OrderManager — the actual order path.
    # Kept in its own try block, and each service is added to `services`
    # immediately on success, so a later failure in a non-critical service
    # (journal_bridge, economic_calendar, ...) can no longer erase these
    # from the reported PhaseResult the way a single batched extend() did.
    try:
        from broker.account_manager import AccountManager
        from broker.order_manager import OrderManager

        conn = registry.try_resolve("mt5_connection")
        account_mgr = AccountManager(conn)
        registry.register_instance("account_manager", account_mgr)
        services.append("account_manager")

        om = OrderManager(conn, account_mgr)
        registry.register_instance("order_manager", om)
        services.append("order_manager")
    except Exception as e:
        log.critical("CRITICAL: AccountManager/OrderManager init failed — no "
                     "order path exists: %s", e, exc_info=True)
        account_broker_failed = True
        account_broker_error = str(e)

    # ── NON-CRITICAL: journal, broker health, economic calendar — each
    # isolated so one failing import can't take out the others.
    db = registry.try_resolve("db")
    conn = registry.try_resolve("mt5_connection")
    try:
        from broker.journal_bridge import JournalBridge
        registry.register_instance("journal_bridge", JournalBridge(db=db))
        services.append("journal_bridge")
    except Exception as e:
        log.warning("JournalBridge init failed (non-critical): %s", e)

    try:
        from broker.health_monitor import HealthMonitor as BrokerHealthMonitor
        registry.register_instance("broker_health_monitor", BrokerHealthMonitor(conn))
        services.append("broker_health_monitor")
    except Exception as e:
        log.warning("BrokerHealthMonitor init failed (non-critical): %s", e)

    try:
        from broker.economic_calendar import EconomicCalendar
        registry.register_instance("economic_calendar", EconomicCalendar())
        services.append("economic_calendar")
    except Exception as e:
        log.warning("EconomicCalendar init failed (non-critical): %s", e)

    # ── Round-30: MT5 Trade State Recovery ────────────────────────────
    # Runs whenever OrderManager is available, independent of whether the
    # non-critical services above succeeded — recovery only needs `om`.
    if om is not None:
        try:
            from execution.trade_recovery import recover_mt5_state

            # Reuse the trade_memory instance registered by boot_persistence
            # instead of constructing a second, independent TradeMemory()
            # (which would open a second DB handle and redo seed_rules work
            # for no reason).
            learning_db = None
            try:
                tm = registry.try_resolve("trade_memory")
                if tm is None:
                    from memory.trade_memory import TradeMemory
                    tm = TradeMemory()
                learning_db = tm.db
            except Exception:
                pass

            # PositionManager may not exist yet (created per-AITrader later).
            # We can still run recovery for DB reconciliation — PositionManager
            # seeding will happen when each AITrader boots and calls
            # recover_from_mt5() on its own PositionManager instance.
            recovery_result = recover_mt5_state(
                order_manager=om,
                learning_db=learning_db,
                position_manager=None,  # Per-AITrader instances seed themselves
            )
            log.info(
                f"[Runtime] MT5 recovery: {recovery_result['open_positions']} open | "
                f"{recovery_result['closed_reconciled']} reconciled | "
                f"{recovery_result['orphan_mt5']} orphan"
            )
            services.append("trade_recovery")
        except Exception as _recovery_e:
            log.warning(f"[Runtime] MT5 trade recovery failed (non-fatal): {_recovery_e}")

    get_bus().publish("system.startup", {"phase": "broker", "services": services}, source="runtime")
    if account_broker_failed:
        return PhaseResult(phase=Phase.BROKER, ok=False, duration_sec=0.0,
                           services_registered=services,
                           error=f"AccountManager/OrderManager failed: {account_broker_error}")
    return PhaseResult(phase=Phase.BROKER, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_analytics(registry: ServiceRegistry) -> PhaseResult:
    """Phase 16 — PerformanceAnalyzer + StrategyTracker + RankingEngine."""
    services = []
    try:
        from analytics.analytics import PerformanceAnalyzer
        registry.register_instance("performance_analyzer", PerformanceAnalyzer())
        services.append("performance_analyzer")
    except Exception as e:
        log.warning("PerformanceAnalyzer init failed: %s", e)

    try:
        from analytics.strategy_tracker import StrategyTracker
        registry.register_instance("strategy_tracker", StrategyTracker())
        services.append("strategy_tracker")
    except Exception as e:
        log.warning("StrategyTracker init failed: %s", e)

    try:
        from analytics.ranking_engine import RankingEngine
        registry.register_instance("ranking_engine", RankingEngine())
        services.append("ranking_engine")
    except Exception as e:
        log.warning("RankingEngine init failed: %s", e)

    try:
        from analytics.performance_report import PerformanceReport
        registry.register(
            "performance_report",
            lambda r: PerformanceReport(tracker=r.try_resolve("strategy_tracker")),
        )
        services.append("performance_report")
    except Exception as e:
        log.warning("PerformanceReport not available: %s", e)

    get_bus().publish("system.startup", {"phase": "analytics", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.ANALYTICS, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_reports(registry: ServiceRegistry) -> PhaseResult:
    """Phase 17 — BacktestReport."""
    services = []
    try:
        from backtest.report import BacktestReport
        registry.register_instance("backtest_report", BacktestReport())
        services.append("backtest_report")
    except Exception as e:
        log.warning("BacktestReport init failed: %s", e)

    get_bus().publish("system.startup", {"phase": "reports", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.REPORTS, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_learning(registry: ServiceRegistry) -> PhaseResult:
    """Phase 18 — ConfidenceEngine + AutoOptimizer + LessonMemory + MistakeAnalyzer."""
    services = []
    try:
        from learning.confidence_engine import ConfidenceEngine
        registry.register_instance("confidence_engine", ConfidenceEngine())
        services.append("confidence_engine")
    except Exception as e:
        log.warning("ConfidenceEngine init failed: %s", e)

    try:
        from learning.rule_updater import RuleUpdater
        registry.register_instance("rule_updater", RuleUpdater())
        services.append("rule_updater")
    except Exception as e:
        log.warning("RuleUpdater init failed: %s", e)

    try:
        from learning.lesson_memory import LessonMemory
        registry.register_instance("lesson_memory", LessonMemory())
        services.append("lesson_memory")
    except Exception as e:
        log.warning("LessonMemory init failed: %s", e)

    try:
        from learning.performance_feedback import PerformanceFeedback
        registry.register_instance("performance_feedback", PerformanceFeedback())
        services.append("performance_feedback")
    except Exception as e:
        log.warning("PerformanceFeedback init failed: %s", e)

    try:
        from learning.auto_optimizer import AutoOptimizer
        registry.register("auto_optimizer", lambda r: AutoOptimizer())
        services.append("auto_optimizer")
    except Exception as e:
        log.warning("AutoOptimizer not available: %s", e)

    try:
        from learning.memory_integration import MemoryIntegration
        # NOTE: MemoryIntegration is registered but never actively called
        # in the live trading loop. It's a legacy module that was part of
        # the hybrid/ pipeline which is now dead. Keeping registration for
        # backward compat but it's effectively a no-op.
        registry.register("memory_integration", lambda r: MemoryIntegration())
        services.append("memory_integration")
    except Exception as e:
        log.warning("MemoryIntegration not available (non-critical, legacy): %s", e)

    try:
        from learning.mistake_analyzer import AdvancedMistakeAnalyzer
        registry.register_instance("mistake_analyzer", AdvancedMistakeAnalyzer())
        services.append("mistake_analyzer")
    except Exception as e:
        log.warning("MistakeAnalyzer init failed: %s", e)

    try:
        from learning.weekly_review import run_weekly_review
        registry.register_instance("weekly_review_fn", run_weekly_review)
        services.append("weekly_review_fn")
    except Exception as e:
        log.warning("weekly_review not available: %s", e)

    get_bus().publish("system.startup", {"phase": "learning", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.LEARNING, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_dashboard(registry: ServiceRegistry) -> PhaseResult:
    """Phase 19 — Dashboard path + bus subscriptions."""
    services = []
    try:
        dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
        registry.register_instance("dashboard_path", str(dashboard_path))
        services.append("dashboard_path")
    except Exception as e:
        log.warning("Dashboard path registration failed: %s", e)

    def _on_metric(evt):
        try:
            from dashboard.components import data_loader as dl
            if hasattr(dl, "_on_runtime_metric"):
                dl._on_runtime_metric(evt.payload)
        except Exception as e:
            log.warning("Suppressed exception in _on_metric dashboard callback: %s", e)
            pass

    try:
        get_bus().subscribe("analytics.metric", _on_metric)
        get_bus().subscribe("health.report", _on_metric)
    except Exception as e:
        log.warning("Suppressed exception while subscribing analytics/health events: %s", e)
        pass

    get_bus().publish("system.startup", {"phase": "dashboard", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.DASHBOARD, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_alerts(registry: ServiceRegistry) -> PhaseResult:
    """Phase 20 — TelegramNotifier + bot polling (FIXED: no duplicate) + bus subscribers."""
    services = []
    global _telegram_polling_started

    try:
        from alerts.telegram_bot import TelegramNotifier, start_telegram_bot_polling
        from config import ENABLE_TELEGRAM

        if ENABLE_TELEGRAM:
            notifier = TelegramNotifier()
            registry.register_instance("telegram_notifier", notifier)
            services.append("telegram_notifier")

            # ✅ FIX: শুধু একবারই polling start হবে — 409 Conflict আর হবে না
            with _telegram_polling_lock:
                if not _telegram_polling_started:
                    t = threading.Thread(
                        target=start_telegram_bot_polling,
                        name="telegram-polling",
                        daemon=True,
                    )
                    t.start()
                    _telegram_polling_started = True
                    services.append("telegram_polling_thread")
                    log.info("✅ Telegram polling started (daemon thread)")
                else:
                    log.warning("⚠️ Telegram polling already running — skipped duplicate")
        else:
            log.info("Telegram disabled (ENABLE_TELEGRAM=false)")

    except Exception as e:
        log.warning("Telegram init failed: %s", e)

    # ── Bus → Telegram event routing ──────────────────────────────
    def _send(msg: str):
        notifier = registry.try_resolve("telegram_notifier")
        if not notifier:
            return
        try:
            from utils.async_utils import run_coro_sync
            run_coro_sync(notifier.send_message(msg))
        except Exception as e:
            log.warning(f"[Runtime] Telegram bus notification failed: {e}")

    get_bus().subscribe("risk.event", lambda e: _send(f"⚠️ RISK: {e.payload}"))
    get_bus().subscribe("risk.circuit_breaker", lambda e: _send(f"⚠️ RISK: {e.payload}"))
    get_bus().subscribe("broker.failure", lambda e: _send(f"🔴 BROKER: {e.payload}"))
    get_bus().subscribe("system.error", lambda e: _send(f"❌ ERROR: {e.payload}"))

    get_bus().publish("system.startup", {"phase": "alerts", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.ALERTS, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_automation(registry: ServiceRegistry) -> PhaseResult:
    """Phase 21 — ErrorHandler + DailyReview + SystemHealth."""
    services = []
    try:
        from automation.error_handler import ErrorHandler
        registry.register_instance("error_handler", ErrorHandler())
        services.append("error_handler")
    except Exception as e:
        log.warning("ErrorHandler not available: %s", e)

    try:
        from automation.daily_review import DailyReview
        registry.register("daily_review", lambda r: DailyReview())
        services.append("daily_review")
    except Exception as e:
        log.warning("DailyReview not available: %s", e)

    try:
        from automation.system_health import SystemHealth
        registry.register("system_health_legacy", lambda r: SystemHealth())
        services.append("system_health_legacy")
    except Exception as e:
        log.warning("SystemHealth (legacy) not available: %s", e)

    def _on_sys_error(evt):
        try:
            h = registry.try_resolve("error_handler")
            if h and hasattr(h, "log_error"):
                h.log_error(
                    category=evt.payload.get("channel", "runtime") if isinstance(evt.payload, dict) else "runtime",
                    message=str(evt.payload),
                )
        except Exception as e:
            log.warning("Suppressed exception in _on_sys_error automation callback: %s", e)
            pass

    get_bus().subscribe("system.error", _on_sys_error)

    get_bus().publish("system.startup", {"phase": "automation", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.AUTOMATION, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_webhook(registry: ServiceRegistry) -> PhaseResult:
    """Phase 22 — Webhook server (DISABLED).

    FIX: The entire webhook server has been moved to legacy/server/.
    It was a security risk (auth bypass when WEBHOOK_SECRET unset) and
    ran a parallel, weaker trading pipeline that didn't share risk
    state with the main loop. Removed to eliminate both issues.

    This phase is now a no-op.
    """
    services = []
    log.info("boot_webhook: skipped (webhook moved to legacy/ — security risk)")
    get_bus().publish("system.startup", {"phase": "webhook", "services": services,
                                          "note": "skipped — moved to legacy/"}, source="runtime")
    return PhaseResult(phase=Phase.WEBHOOK, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_orchestrator(registry: ServiceRegistry) -> PhaseResult:
    """Phase 23 — TradingOrchestrator + DailyRoutineManager + TaskScheduler."""
    services = []
    try:
        from orchestrator.trading_orchestrator import TradingOrchestrator
        registry.register("trading_orchestrator", lambda r: TradingOrchestrator())
        services.append("trading_orchestrator")
    except Exception as e:
        log.warning("TradingOrchestrator not available: %s", e)

    try:
        from orchestrator.daily_routine import DailyRoutineManager
        registry.register(
            "daily_routine",
            lambda r: DailyRoutineManager(orchestrator=r.try_resolve("trading_orchestrator")),
        )
        services.append("daily_routine")
    except Exception as e:
        log.warning("DailyRoutineManager not available: %s", e)

    try:
        from orchestrator.scheduler import TaskScheduler
        registry.register_instance("task_scheduler", TaskScheduler())
        services.append("task_scheduler")
    except Exception as e:
        log.warning("TaskScheduler not available: %s", e)

    try:
        from orchestrator.audit_trail import AuditTrail
        registry.register_instance("audit_trail", AuditTrail())
        services.append("audit_trail")
    except Exception as e:
        log.warning("AuditTrail not available: %s", e)

    try:
        from orchestrator.human_override import HumanOverrideSystem

        def _make_human_override(r):
            hos = HumanOverrideSystem(
                state_manager=r.try_resolve("system_state_manager"),
                bus=r.try_resolve("message_bus"),
            )
            hos.start()  # FIX (bug 18): actually start the command-file poll thread
            return hos

        registry.register("human_override", _make_human_override)
        services.append("human_override")
    except Exception as e:
        log.warning("HumanOverrideSystem not available: %s", e)

    try:
        from orchestrator.communication_bus import AgentMessageBus
        registry.register_instance("message_bus", AgentMessageBus())
        services.append("message_bus")
    except Exception as e:
        log.warning("AgentMessageBus not available: %s", e)

    try:
        from orchestrator.system_state import SystemStateManager
        registry.register_instance("system_state_manager", SystemStateManager())
        services.append("system_state_manager")
    except Exception as e:
        log.warning("SystemStateManager not available: %s", e)

    get_bus().publish("system.startup", {"phase": "orchestrator", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.ORCHESTRATOR, ok=True, duration_sec=0.0,
                       services_registered=services)


def boot_runtime_phase(registry: ServiceRegistry) -> PhaseResult:
    """Phase 24 — TradingEngine (extends AutonomousTraderSystem, no double init)."""
    services = []
    try:
        from core.trading_engine import TradingEngine
        from config import (EXECUTION_MODE, USE_SCANNER, APPROVAL_MODE, SYMBOLS,
                            DEFAULT_TIMEFRAME, INITIAL_BALANCE, LOOP_INTERVAL_SEC,
                            BACKUP_INTERVAL_MIN, RECOVERY_COOLDOWN_MIN, ENABLE_TELEGRAM)

        symbols = registry.get("symbols", list(SYMBOLS))
        timeframe = registry.get("timeframe", DEFAULT_TIMEFRAME)
        execution_mode = registry.get("execution_mode", EXECUTION_MODE)

        # TradingEngine IS-A AutonomousTraderSystem — single instance only
        engine = TradingEngine(
            symbols=symbols, timeframe=timeframe, balance=INITIAL_BALANCE,
            poll_seconds=LOOP_INTERVAL_SEC,
            backup_interval_minutes=BACKUP_INTERVAL_MIN,
            cooldown_minutes=RECOVERY_COOLDOWN_MIN,
            enable_telegram=ENABLE_TELEGRAM, use_scanner=USE_SCANNER,
            execution_mode=execution_mode, approval_mode=APPROVAL_MODE,
            registry=registry,
        )
        registry.register_instance("trader", engine)          # backward compat
        services.append("trader")
        registry.register_instance("trading_engine", engine)  # also as engine
        services.append("trading_engine")

    except Exception as e:
        log.error("Trader init failed: %s", e, exc_info=True)
        registry.mark("trader", ServiceStatus.FAILED, str(e))

    get_bus().publish("system.startup", {"phase": "runtime", "services": services}, source="runtime")
    return PhaseResult(phase=Phase.RUNTIME, ok=True, duration_sec=0.0,
                       services_registered=services)


# ─────────────────────────────────────────────────────────────────────
# Phase registration
# ─────────────────────────────────────────────────────────────────────


def register_default_phases(lifecycle: LifecycleManager) -> None:
    """Wire every phase to its boot function."""
    lifecycle.register_phase(Phase.BOOTSTRAP,    boot_bootstrap)
    lifecycle.register_phase(Phase.PERSISTENCE,  boot_persistence)
    lifecycle.register_phase(Phase.DATA,         boot_data)
    lifecycle.register_phase(Phase.MARKET,       boot_market)
    lifecycle.register_phase(Phase.RESEARCH,     boot_research)
    lifecycle.register_phase(Phase.FUNDAMENTAL,  boot_fundamental)
    lifecycle.register_phase(Phase.ANALYSIS,     boot_analysis)
    lifecycle.register_phase(Phase.AI,           boot_ai)
    lifecycle.register_phase(Phase.AGENTS,       boot_agents)
    lifecycle.register_phase(Phase.STRATEGY,     boot_strategy)
    lifecycle.register_phase(Phase.HYBRID,       boot_hybrid)
    lifecycle.register_phase(Phase.RISK,         boot_risk)
    lifecycle.register_phase(Phase.SAFETY,       boot_safety)
    lifecycle.register_phase(Phase.EXECUTION,    boot_execution)
    lifecycle.register_phase(Phase.BROKER,       boot_broker)
    lifecycle.register_phase(Phase.ANALYTICS,    boot_analytics)
    lifecycle.register_phase(Phase.REPORTS,      boot_reports)
    lifecycle.register_phase(Phase.LEARNING,     boot_learning)
    lifecycle.register_phase(Phase.DASHBOARD,    boot_dashboard)
    lifecycle.register_phase(Phase.ALERTS,       boot_alerts)
    lifecycle.register_phase(Phase.AUTOMATION,   boot_automation)
    lifecycle.register_phase(Phase.WEBHOOK,      boot_webhook)
    lifecycle.register_phase(Phase.ORCHESTRATOR, boot_orchestrator)
    lifecycle.register_phase(Phase.RUNTIME,      boot_runtime_phase)


def boot_runtime(until: Optional[Phase] = None) -> Runtime:
    """Convenience: create Runtime, register phases, boot, return."""
    rt = get_runtime()
    rt.boot(until=until)
    return rt