"""
core/service_registry.py — Central service registry & dependency injection
=========================================================================

A lightweight, type-safe service registry that the runtime uses for
dependency injection. Every long-lived service (database, memory, scanner,
risk engine, broker, etc.) is registered here exactly once and resolved by
name or by type. This replaces the ad-hoc `self._x = X()` wiring scattered
across `core/trader.py` and `main.py`.

Design goals:
  * Single source of truth for service instances.
  * Lazy registration: services are registered but instantiated on first
    resolve (or eagerly if `register_instance` is used).
  * Health-aware: the registry tracks which services are alive.
  * No external dependencies — uses only stdlib so it can boot before any
    third-party package is imported.

Public API:
  * `ServiceRegistry` — the registry class.
  * `get_registry()` — module-level singleton accessor.
  * `ServiceStatus` — enum of service states.
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceStatus(str, Enum):
    """Lifecycle states a registered service can be in."""
    REGISTERED = "registered"   # factory known, not yet built
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTDOWN = "shutdown"


@dataclass
class ServiceRecord:
    """Internal bookkeeping for a single registered service."""
    name: str
    factory: Optional[Callable[["ServiceRegistry"], Any]] = None
    instance: Any = None
    status: ServiceStatus = ServiceStatus.REGISTERED
    error: Optional[str] = None
    init_at: Optional[float] = None
    tags: set = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)


class ServiceNotFoundError(KeyError):
    """Raised when a service is requested that was never registered."""


class ServiceRegistrationError(RuntimeError):
    """Raised when a service is registered twice with different factories."""


class ServiceRegistry:
    """Thread-safe service registry & DI container.

    Usage:
        registry = get_registry()
        registry.register("db", lambda r: TraderDB())
        registry.register("risk_engine", lambda r: RiskEngine(...), dependencies=["db"])

        db = registry.resolve("db")              # builds & caches
        db2 = registry.resolve("db")             # returns cached
        registry.health()                        # -> {"db": "healthy", ...}
    """

    def __init__(self):
        self._services: Dict[str, ServiceRecord] = {}
        self._by_type: Dict[Type, str] = {}
        self._lock = threading.RLock()
        self._shutdown_hooks: List[Callable[["ServiceRegistry"], None]] = []

    # ── registration ────────────────────────────────────────────────

    def register(
        self,
        name: str,
        factory: Callable[["ServiceRegistry"], Any],
        *,
        tags: Optional[Iterable[str]] = None,
        dependencies: Optional[List[str]] = None,
        eager: bool = False,
    ) -> "ServiceRegistry":
        """Register a service by name with a factory.
        The factory receives the registry itself so it can resolve deps.
        If `eager=True`, the service is built immediately (use for critical
        infra like logging/db)."""
        with self._lock:
            if name in self._services:
                existing = self._services[name]
                if existing.factory is not factory and existing.instance is None:
                    raise ServiceRegistrationError(
                        f"Service '{name}' already registered with a different factory"
                    )
                log.debug("Service '%s' already registered — skipping", name)
                return self
            rec = ServiceRecord(
                name=name,
                factory=factory,
                tags=set(tags or []),
                dependencies=list(dependencies or []),
            )
            self._services[name] = rec
            log.debug("Registered service '%s' (eager=%s, deps=%s)", name, eager, rec.dependencies)
        if eager:
            self.resolve(name)
        return self

    def register_instance(self, name: str, instance: Any, *, tags: Optional[List[str]] = None) -> "ServiceRegistry":
        """Register an already-built instance (skips factory)."""
        with self._lock:
            if name in self._services and self._services[name].instance is not None:
                log.debug("Service '%s' already has an instance — replacing", name)
            self._services[name] = ServiceRecord(
                name=name,
                factory=None,
                instance=instance,
                status=ServiceStatus.HEALTHY,
                init_at=time.time(),
                tags=set(tags or []),
            )
            self._index_type(name, instance)
        return self

    def register_failed(self, name: str, error: str, *, tags: Optional[List[str]] = None) -> "ServiceRegistry":
        """Record a service that was attempted but failed to initialize
        (e.g. an optional external connection that couldn't be reached at
        boot). Unlike a normal registration, this has neither a factory
        nor an instance, so `resolve()` will raise (and `try_resolve()` /
        `get()` will cleanly return None/default) on every call — instead
        of a placeholder factory being invoked repeatedly and flapping the
        status between FAILED and HEALTHY. `health()` will correctly show
        this service as FAILED with its error message, so monitoring can
        alert on it.
        """
        with self._lock:
            self._services[name] = ServiceRecord(
                name=name,
                factory=None,
                instance=None,
                status=ServiceStatus.FAILED,
                error=error,
                tags=set(tags or []),
            )
        return self

    def register_type(self, name: str, cls: Type[T]) -> "ServiceRegistry":
        """Register a class — resolved by calling cls(registry) lazily."""
        return self.register(name, lambda r: cls(r))

    def _index_type(self, name: str, instance: Any) -> None:
        """Map instance's class (and bases) to the service name for resolve_type()."""
        for klass in type(instance).__mro__:
            if klass is object:
                continue
            self._by_type.setdefault(klass, name)

    # ── resolution ──────────────────────────────────────────────────

    def resolve(self, name: str) -> Any:
        """Resolve (build if needed) and return the service instance."""
        with self._lock:
            rec = self._services.get(name)
            if rec is None:
                raise ServiceNotFoundError(
                    f"Service '{name}' not registered. Available: {sorted(self._services)}"
                )
            if rec.instance is not None:
                return rec.instance
            if rec.factory is None:
                raise ServiceRegistrationError(f"Service '{name}' has no factory and no instance")
            if rec.status == ServiceStatus.INITIALIZING:
                raise RuntimeError(f"Circular dependency detected: service '{name}' is already being initialized")
            rec.status = ServiceStatus.INITIALIZING
        # Build outside the lock to allow deps to resolve back into us.
        try:
            instance = rec.factory(self)
        except Exception as e:
            with self._lock:
                rec.status = ServiceStatus.FAILED
                rec.error = str(e)
            log.error("Service '%s' initialization failed: %s", name, e, exc_info=True)
            raise
        with self._lock:
            rec.instance = instance
            rec.status = ServiceStatus.HEALTHY
            rec.init_at = time.time()
            self._index_type(name, instance)
        log.info("Service '%s' initialized", name)
        return instance

    def try_resolve(self, name: str) -> Optional[Any]:
        """Like resolve() but returns None on failure (never raises)."""
        try:
            return self.resolve(name)
        except Exception as e:
            log.debug("try_resolve('%s') failed: %s", name, e)
            return None

    def resolve_type(self, cls: Type[T]) -> Optional[T]:
        """Resolve a service by its class (or any base class)."""
        with self._lock:
            name = self._by_type.get(cls)
            if name is None:
                for klass, svc_name in self._by_type.items():
                    if issubclass(klass, cls):
                        name = svc_name
                        break
            if name is None:
                return None
        return self.resolve(name)

    def has(self, name: str) -> bool:
        return name in self._services

    def get(self, name: str, default: Any = None) -> Any:
        """Safe getter — returns default if missing or unresolvable."""
        try:
            return self.resolve(name)
        except Exception as e:
            return default

    # ── health & introspection ──────────────────────────────────────

    def mark(self, name: str, status: ServiceStatus, error: Optional[str] = None) -> None:
        with self._lock:
            rec = self._services.get(name)
            if rec is None:
                log.warning("mark('%s', %s) — service not registered", name, status)
                return
            rec.status = status
            if error:
                rec.error = error

    def health(self) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of every service's status."""
        with self._lock:
            return {
                name: {
                    "status": rec.status.value,
                    "error": rec.error,
                    "init_at": rec.init_at,
                    "tags": sorted(rec.tags),
                    "dependencies": rec.dependencies,
                }
                for name, rec in self._services.items()
            }

    def names(self, tag: Optional[str] = None) -> List[str]:
        with self._lock:
            if tag is None:
                return sorted(self._services.keys())
            return [n for n, r in self._services.items() if tag in r.tags]

    # ── lifecycle hooks ─────────────────────────────────────────────

    def on_shutdown(self, hook: Callable[["ServiceRegistry"], None]) -> None:
        self._shutdown_hooks.append(hook)

    def shutdown(self) -> None:
        """Run all registered shutdown hooks in reverse order."""
        log.info("ServiceRegistry shutdown — running %d hooks", len(self._shutdown_hooks))
        for hook in reversed(self._shutdown_hooks):
            try:
                hook(self)
            except Exception as e:
                log.error("Shutdown hook failed: %s", e, exc_info=True)
        # Then call .shutdown() / .close() on any service that exposes it.
        with self._lock:
            for name, rec in list(self._services.items()):
                if rec.instance is None:
                    continue
                closer = getattr(rec.instance, "shutdown", None) or getattr(rec.instance, "close", None)
                if closer is None:
                    continue
                try:
                    closer()
                    rec.status = ServiceStatus.SHUTDOWN
                    log.info("Service '%s' shut down", name)
                except Exception as e:
                    log.error("Service '%s' shutdown failed: %s", name, e, exc_info=True)
        log.info("ServiceRegistry shutdown complete")


# ── module-level singleton ──────────────────────────────────────────

_REGISTRY: Optional[ServiceRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> ServiceRegistry:
    """Get (or create) the global ServiceRegistry singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = ServiceRegistry()
    return _REGISTRY


def reset_registry() -> None:
    """Discard the global singleton — for tests only."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is not None:
            try:
                _REGISTRY.shutdown()
            except Exception as e:
                log.warning(f"Suppressed exception at line 299: {e}")
                pass
        _REGISTRY = None