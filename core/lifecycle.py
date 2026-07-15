"""
core/lifecycle.py — Lifecycle manager
======================================

Owns the orderly bring-up and tear-down of every runtime service. The
manager enforces a strict phase ordering so that no service starts before
its dependencies are healthy.

Phases (executed in this order on startup, reversed on shutdown):

  BOOTSTRAP   — logging, paths, config validation
  PERSISTENCE — database, memory stores, vector DB
  DATA        — fetchers, validators, indicators
  MARKET      — scanner, market data manager (MT5 if enabled)
  RESEARCH    — hypothesis engine, experiment runner, research agent
  FUNDAMENTAL — news filter, fundamental sentiment
  ANALYSIS    — every analysis/* engine
  AI          — AIAnalyst, MasterAnalyst, model versioning
  AGENTS      — Market / Analysis / Decision / Learning agents
  STRATEGY    — signal engine + strategies package
  HYBRID      — flow controller + decision validator
  RISK        — risk engine, circuit breaker, trade permission
  SAFETY      — safety guard, spread monitor, drawdown controller
  EXECUTION   — paper trader, execution router
  BROKER      — MT5 connection, order/position managers
  ANALYTICS   — performance analyzer, strategy tracker
  REPORTS     — report generators
  LEARNING    — auto-optimizer, lesson memory, memory integration
  DASHBOARD   — streamlit status (lazy)
  ALERTS      — telegram notifier + bot
  AUTOMATION  — daily review, error handler, runtime metrics
  WEBHOOK     — signal pipeline + flask server
  ORCHESTRATOR— trading orchestrator + daily routine
  RUNTIME     — the trader itself (AutonomousTraderSystem)

Each phase has a `boot_<phase>()` function that registers its services into
the ServiceRegistry. The LifecycleManager calls them in order and tracks
timing, failures, and skipped phases.

Public API:
  * `LifecycleManager` — the manager class.
  * `Phase` — enum of phases.
  * `get_lifecycle()` — singleton accessor.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from core.service_registry import ServiceRegistry, ServiceStatus, get_registry

log = logging.getLogger(__name__)


class Phase(str, Enum):
    BOOTSTRAP = "bootstrap"
    PERSISTENCE = "persistence"
    DATA = "data"
    MARKET = "market"
    RESEARCH = "research"
    FUNDAMENTAL = "fundamental"
    ANALYSIS = "analysis"
    AI = "ai"
    AGENTS = "agents"
    STRATEGY = "strategy"
    HYBRID = "hybrid"
    RISK = "risk"
    SAFETY = "safety"
    EXECUTION = "execution"
    BROKER = "broker"
    ANALYTICS = "analytics"
    REPORTS = "reports"
    LEARNING = "learning"
    DASHBOARD = "dashboard"
    ALERTS = "alerts"
    AUTOMATION = "automation"
    WEBHOOK = "webhook"
    ORCHESTRATOR = "orchestrator"
    RUNTIME = "runtime"


PHASE_ORDER: List[Phase] = [
    Phase.BOOTSTRAP,
    Phase.PERSISTENCE,
    Phase.DATA,
    Phase.MARKET,
    Phase.RESEARCH,
    Phase.FUNDAMENTAL,
    Phase.ANALYSIS,
    Phase.AI,
    Phase.AGENTS,
    Phase.STRATEGY,
    Phase.HYBRID,
    Phase.RISK,
    Phase.SAFETY,
    Phase.EXECUTION,
    Phase.BROKER,
    Phase.ANALYTICS,
    Phase.REPORTS,
    Phase.LEARNING,
    Phase.DASHBOARD,
    Phase.ALERTS,
    Phase.AUTOMATION,
    Phase.WEBHOOK,
    Phase.ORCHESTRATOR,
    Phase.RUNTIME,
]


@dataclass
class PhaseResult:
    phase: Phase
    ok: bool
    duration_sec: float
    services_registered: List[str] = field(default_factory=list)
    error: Optional[str] = None
    skipped: bool = False


PhaseBootFn = Callable[[ServiceRegistry], PhaseResult]


class LifecycleManager:
    """Drives the boot and shutdown of every runtime phase."""

    def __init__(self, registry: Optional[ServiceRegistry] = None):
        self.registry = registry or get_registry()
        self._boot_fns: Dict[Phase, PhaseBootFn] = {}
        self._results: List[PhaseResult] = []
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._on_phase_complete: List[Callable[[PhaseResult], None]] = []

    def register_phase(self, phase: Phase, boot_fn: PhaseBootFn) -> None:
        self._boot_fns[phase] = boot_fn
        log.debug("Phase '%s' boot fn registered", phase.value)

    def on_phase_complete(self, cb: Callable[[PhaseResult], None]) -> None:
        self._on_phase_complete.append(cb)

    def boot(self, until: Optional[Phase] = None) -> List[PhaseResult]:
        """Run every registered phase in PHASE_ORDER. If `until` is given,
        stop after that phase completes (inclusive)."""
        self._started_at = time.time()
        self._results.clear()
        cutoff = PHASE_ORDER.index(until) if until else len(PHASE_ORDER) - 1
        for phase in PHASE_ORDER[: cutoff + 1]:
            boot_fn = self._boot_fns.get(phase)
            if boot_fn is None:
                log.debug("Phase '%s' has no boot fn — skipping", phase.value)
                result = PhaseResult(phase=phase, ok=True, duration_sec=0.0, skipped=True)
            else:
                t0 = time.time()
                try:
                    result = boot_fn(self.registry)
                    if result is None:
                        result = PhaseResult(phase=phase, ok=True, duration_sec=time.time() - t0)
                    result.duration_sec = round(time.time() - t0, 3)
                except Exception as e:
                    log.error("Phase '%s' FAILED: %s", phase.value, e, exc_info=True)
                    result = PhaseResult(
                        phase=phase, ok=False, duration_sec=round(time.time() - t0, 3),
                        error=str(e),
                    )
            self._results.append(result)
            for cb in self._on_phase_complete:
                try:
                    cb(result)
                except Exception as e:
                    log.error("on_phase_complete callback failed: %s", e)
            if not result.ok and phase in (Phase.BOOTSTRAP, Phase.PERSISTENCE):
                log.critical("Phase '%s' is critical — aborting boot", phase.value)
                break
        self._stopped_at = time.time()
        return list(self._results)

    def shutdown(self) -> None:
        """Shutdown services in reverse order."""
        log.info("LifecycleManager.shutdown — calling registry.shutdown()")
        self.registry.shutdown()

    def report(self) -> Dict:
        return {
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "phases": [
                {
                    "phase": r.phase.value,
                    "ok": r.ok,
                    "skipped": r.skipped,
                    "duration_sec": r.duration_sec,
                    "services": r.services_registered,
                    "error": r.error,
                }
                for r in self._results
            ],
            "registry_health": self.registry.health(),
        }

    def is_phase_complete(self, phase: Phase) -> bool:
        return any(r.phase == phase for r in self._results)

    def last_result(self, phase: Phase) -> Optional[PhaseResult]:
        for r in reversed(self._results):
            if r.phase == phase:
                return r
        return None


# ── singleton ───────────────────────────────────────────────────────

_LIFECYCLE: Optional[LifecycleManager] = None


def get_lifecycle() -> LifecycleManager:
    global _LIFECYCLE
    if _LIFECYCLE is None:
        _LIFECYCLE = LifecycleManager()
    return _LIFECYCLE
