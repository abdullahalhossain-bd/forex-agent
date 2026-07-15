# core/retry_with_failover.py — Retry utility + model failover (ported from K.I.T.)
# =============================================================================
# Ported from: https://github.com/Signal-Execution-Labs/forex-trading-ai-agent
# Original: src/core/retry.ts + src/core/model-failover.ts (TypeScript)
# Original author: Signal-Execution-Labs (K.I.T. Team) — MIT license
#
# Two utilities ported from the K.I.T. AI trading agent:
#
# 1. RETRY UTILITY — exponential backoff with jitter for reliable API calls.
#    Used for broker connections, Telegram API, exchange APIs, etc.
#    Preset policies: default, telegram, discord, exchange (conservative).
#
# 2. MODEL FAILOVER SERVICE — rotate between multiple API keys/profiles
#    when one fails (rate limit, timeout, etc.). Supports:
#    - Round-robin rotation among available profiles
#    - Cooldown period for failed profiles
#    - Session pinning (stick to one profile per session)
#    - Model fallback (primary → fallback1 → fallback2)
#
# ⚠️  STATUS (institutional review, see audit report):
#     retry_sync()/RetryPolicy (part 1) IS now used — see
#     core/production_hardening.py's _mt5_positions_get(), which
#     previously duplicated this logic with its own hand-rolled loop.
#     ModelFailoverService (part 2) is still NOT called from anywhere in
#     the uploaded codebase — left as-is since wiring it requires
#     knowing which LLM-calling module (not part of this review) should
#     own the profile rotation.
# =============================================================================

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

from utils.logger import get_logger

log = get_logger("retry_failover")

T = TypeVar("T")


# ── 1. Retry Utility ─────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    attempts: int = 3
    min_delay_ms: int = 1000
    max_delay_ms: int = 30000
    jitter: float = 0.2           # 0-1, randomize delays
    base: float = 2.0             # exponential base
    retry_status_codes: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    should_retry: Optional[Callable[[Exception, int], bool]] = None
    on_retry: Optional[Callable[[Exception, int, int], None]] = None


# Preset policies (matching the original TypeScript)
DEFAULT_RETRY = RetryPolicy()
TELEGRAM_RETRY = RetryPolicy(attempts=3, min_delay_ms=1000, max_delay_ms=60000, jitter=0.25)
DISCORD_RETRY = RetryPolicy(attempts=3, min_delay_ms=1000, max_delay_ms=30000, jitter=0.2)
EXCHANGE_RETRY = RetryPolicy(
    attempts=5, min_delay_ms=2000, max_delay_ms=60000, jitter=0.3,
    retry_status_codes=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524]
)
MT5_RETRY = RetryPolicy(attempts=3, min_delay_ms=500, max_delay_ms=10000, jitter=0.15)


def calculate_retry_delay(attempt: int, policy: RetryPolicy) -> float:
    """Calculate delay for a retry attempt with exponential backoff and jitter (ms)."""
    exp_delay = policy.min_delay_ms * (policy.base ** attempt)
    capped = min(exp_delay, policy.max_delay_ms)
    jitter_range = capped * policy.jitter
    jitter_offset = (random.random() - 0.5) * jitter_range
    return round(capped + jitter_offset)


def is_retryable_error(error: Exception, policy: RetryPolicy) -> bool:
    """Check if an error is retryable based on policy."""
    # Network errors
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True

    # HTTP status codes
    status = getattr(error, 'status', None) or getattr(error, 'code', None)
    if status and status in policy.retry_status_codes:
        return True

    # Rate limit
    msg = str(error).lower()
    if 'rate limit' in msg or 'too many requests' in msg:
        return True

    return False


async def retry_async(
    fn: Callable[[], Any],
    policy: RetryPolicy = DEFAULT_RETRY,
    label: str = "",
) -> Any:
    """
    Retry wrapper for async functions with exponential backoff.

    Example:
        result = await retry_async(
            lambda: broker.place_order(...),
            policy=EXCHANGE_RETRY,
            label="place_order",
        )
    """
    last_error = None
    for attempt in range(policy.attempts):
        try:
            return await fn()
        except Exception as e:
            last_error = e
            if attempt < policy.attempts - 1 and (
                is_retryable_error(e, policy) or
                (policy.should_retry and policy.should_retry(e, attempt))
            ):
                delay = calculate_retry_delay(attempt, policy)
                if policy.on_retry:
                    policy.on_retry(e, attempt, delay)
                log.warning(f"Retry {attempt+1}/{policy.attempts} for '{label}' "
                           f"after {delay}ms: {e}")
                await asyncio.sleep(delay / 1000.0)
            else:
                raise
    raise last_error  # type: ignore


def retry_sync(
    fn: Callable[[], Any],
    policy: RetryPolicy = DEFAULT_RETRY,
    label: str = "",
) -> Any:
    """Synchronous version of retry_async."""
    last_error = None
    for attempt in range(policy.attempts):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < policy.attempts - 1 and (
                is_retryable_error(e, policy) or
                (policy.should_retry and policy.should_retry(e, attempt))
            ):
                delay = calculate_retry_delay(attempt, policy)
                if policy.on_retry:
                    policy.on_retry(e, attempt, delay)
                log.warning(f"Retry {attempt+1}/{policy.attempts} for '{label}' "
                           f"after {delay}ms: {e}")
                time.sleep(delay / 1000.0)
            else:
                raise
    raise last_error  # type: ignore


# ── 2. Model Failover Service ────────────────────────────────────────────────

@dataclass
class AuthProfile:
    """An API key/profile for a provider (e.g., OpenAI, Groq)."""
    id: str
    provider: str
    key: str = ""
    profile_type: str = "api_key"  # "api_key" or "oauth"
    last_used: float = 0.0
    cooldown_until: float = 0.0
    disabled: bool = False
    requests: int = 0
    failures: int = 0


@dataclass
class FailoverConfig:
    """Configuration for model failover behavior."""
    enabled: bool = True
    max_retries: int = 3
    cooldown_ms: int = 60000  # 1 minute
    rotate_on_rate_limit: bool = True
    rotate_on_timeout: bool = True
    session_sticky: bool = True  # pin profile per session


class ModelFailoverService:
    """
    Rotate between multiple API keys/profiles when one fails.

    Usage:
        failover = ModelFailoverService()
        failover.register_profiles("openai", [
            AuthProfile(id="key1", provider="openai", key="sk-..."),
            AuthProfile(id="key2", provider="openai", key="sk-..."),
        ])

        # Get next available profile
        profile = failover.get_next_profile("openai")
        if profile:
            try:
                result = call_api(profile.key)
                failover.mark_used(profile.id)
            except RateLimitError:
                failover.cooldown(profile.id)
                # Next call will use a different profile
    """

    def __init__(self, config: Optional[FailoverConfig] = None):
        self.config = config or FailoverConfig()
        self._profiles: dict[str, list[AuthProfile]] = {}
        self._rotation_index: int = 0
        self._pinned: dict[str, str] = {}  # session_id → profile_id

    def register_profiles(self, provider: str, profiles: list[AuthProfile]) -> None:
        """Register auth profiles for a provider."""
        self._profiles[provider] = profiles
        log.info(f"Registered {len(profiles)} profiles for {provider}")

    def get_available_profiles(self, provider: str) -> list[AuthProfile]:
        """Get non-cooldown, non-disabled profiles sorted by last used."""
        profiles = self._profiles.get(provider, [])
        now = time.time()
        available = [
            p for p in profiles
            if not p.disabled and p.cooldown_until < now
        ]
        available.sort(key=lambda p: p.last_used)
        return available

    def get_next_profile(
        self, provider: str, session_id: Optional[str] = None
    ) -> Optional[AuthProfile]:
        """Get the next profile to try (round-robin within available)."""
        if not self.config.enabled:
            # Return first available without rotation
            available = self.get_available_profiles(provider)
            return available[0] if available else None

        # Session pinning
        if self.config.session_sticky and session_id:
            pinned_id = self._pinned.get(session_id)
            if pinned_id:
                for p in self._profiles.get(provider, []):
                    if p.id == pinned_id and p.cooldown_until < time.time():
                        return p

        available = self.get_available_profiles(provider)
        if not available:
            return None

        profile = available[self._rotation_index % len(available)]
        self._rotation_index += 1

        if self.config.session_sticky and session_id:
            self._pinned[session_id] = profile.id

        return profile

    def cooldown(self, profile_id: str) -> None:
        """Put a profile in cooldown after failure."""
        for profiles in self._profiles.values():
            for p in profiles:
                if p.id == profile_id:
                    p.cooldown_until = time.time() + (self.config.cooldown_ms / 1000)
                    p.failures += 1
                    log.warning(f"Profile {profile_id} in cooldown for "
                               f"{self.config.cooldown_ms}ms (failures: {p.failures})")
                    return

    def mark_used(self, profile_id: str) -> None:
        """Mark a profile as successfully used."""
        for profiles in self._profiles.values():
            for p in profiles:
                if p.id == profile_id:
                    p.last_used = time.time()
                    p.requests += 1
                    return

    def disable(self, profile_id: str) -> None:
        """Permanently disable a profile."""
        for profiles in self._profiles.values():
            for p in profiles:
                if p.id == profile_id:
                    p.disabled = True
                    log.warning(f"Profile {profile_id} disabled")
                    return

    def get_status(self) -> dict:
        """Return status of all profiles for debugging."""
        result = {}
        for provider, profiles in self._profiles.items():
            result[provider] = [
                {
                    "id": p.id,
                    "disabled": p.disabled,
                    "in_cooldown": p.cooldown_until > time.time(),
                    "requests": p.requests,
                    "failures": p.failures,
                }
                for p in profiles
            ]
        return result


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Test retry delay calculation
    for attempt in range(5):
        delay = calculate_retry_delay(attempt, EXCHANGE_RETRY)
        print(f"Attempt {attempt}: delay={delay}ms")

    # 2. Test retryable error detection
    assert is_retryable_error(ConnectionError("network"), DEFAULT_RETRY)
    assert is_retryable_error(TimeoutError("timeout"), DEFAULT_RETRY)

    class RateLimitError(Exception):
        status = 429
    assert is_retryable_error(RateLimitError("rate limit"), DEFAULT_RETRY)

    class ValueError2(Exception):
        pass
    assert not is_retryable_error(ValueError2("bad input"), DEFAULT_RETRY)

    # 3. Test retry_sync
    counter = {"n": 0}
    def flaky_function():
        counter["n"] += 1
        if counter["n"] < 3:
            raise ConnectionError("network error")
        return "success"

    result = retry_sync(flaky_function, RetryPolicy(attempts=5, min_delay_ms=10),
                        label="flaky")
    assert result == "success"
    assert counter["n"] == 3
    print(f"\nRetry succeeded after {counter['n']} attempts: {result}")

    # 4. Test ModelFailoverService
    failover = ModelFailoverService()
    failover.register_profiles("openai", [
        AuthProfile(id="key1", provider="openai", key="sk-key1"),
        AuthProfile(id="key2", provider="openai", key="sk-key2"),
        AuthProfile(id="key3", provider="openai", key="sk-key3"),
    ])

    # Get next profiles (round-robin)
    p1 = failover.get_next_profile("openai")
    p2 = failover.get_next_profile("openai")
    p3 = failover.get_next_profile("openai")
    print(f"\nRound-robin: {p1.id} → {p2.id} → {p3.id}")
    assert len({p1.id, p2.id, p3.id}) == 3  # all different

    # Cooldown one
    failover.cooldown("key2")
    available = failover.get_available_profiles("openai")
    ids = [p.id for p in available]
    print(f"After cooldown key2: available={ids}")
    assert "key2" not in ids

    # Status
    status = failover.get_status()
    print(f"Status: {status}")

    print("\nRetry + failover smoke test passed.")