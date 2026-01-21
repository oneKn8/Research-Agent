"""Circuit breaker pattern implementation for external service resilience.

Prevents cascading failures by tracking failure rates and temporarily
blocking requests to failing services.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service failing, requests are rejected immediately
- HALF_OPEN: Testing if service recovered, limited requests allowed
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from src.utils.logging import get_logger

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open to close
    timeout_seconds: float = 30.0  # Time before half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    current_state: CircuitState = CircuitState.CLOSED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/metrics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "current_state": self.current_state.value,
            "failure_rate": (
                self.failed_calls / self.total_calls if self.total_calls > 0 else 0.0
            ),
        }


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and rejecting requests."""

    def __init__(self, service_name: str, retry_after: float) -> None:
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker open for {service_name}. Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """Circuit breaker for protecting external service calls.

    Usage:
        breaker = CircuitBreaker("openai")

        async def call_openai():
            async with breaker:
                return await openai_client.chat(...)

    Or with decorator:
        @breaker.protect
        async def call_openai():
            return await openai_client.chat(...)
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Service name for logging/metrics
            config: Configuration options
        """
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._logger = get_logger(f"CircuitBreaker.{name}")
        self._stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        self._stats.current_state = self._state
        return self._stats

    async def _check_state(self) -> None:
        """Check and potentially transition state."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self._config.timeout_seconds:
                        self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._stats.state_changes += 1

            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._success_count = 0

            if new_state == CircuitState.CLOSED:
                self._failure_count = 0

            self._logger.info(
                "Circuit breaker state changed",
                service=self._name,
                old_state=old_state.value,
                new_state=new_state.value,
            )

    async def _can_execute(self) -> bool:
        """Check if a request can be executed."""
        await self._check_state()

        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                retry_after = 0.0
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    retry_after = max(0, self._config.timeout_seconds - elapsed)
                self._stats.rejected_calls += 1
                raise CircuitBreakerOpen(self._name, retry_after)

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._config.half_open_max_calls:
                    self._stats.rejected_calls += 1
                    raise CircuitBreakerOpen(self._name, 1.0)
                self._half_open_calls += 1
                return True

            return False

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.last_failure_time = time.time()
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._transition_to(CircuitState.OPEN)
                self._logger.warning(
                    "Circuit reopened after half-open failure",
                    service=self._name,
                    error=str(error) if error else None,
                )
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    self._logger.warning(
                        "Circuit opened due to failures",
                        service=self._name,
                        failure_count=self._failure_count,
                        threshold=self._config.failure_threshold,
                    )

    async def __aenter__(self) -> "CircuitBreaker":
        """Context manager entry - check if call is allowed."""
        await self._can_execute()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit - record result."""
        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure(exc_val if isinstance(exc_val, Exception) else None)

    def protect(self, func: Any) -> Any:
        """Decorator to protect an async function with this circuit breaker."""
        import functools

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                return await func(*args, **kwargs)

        return wrapper

    async def reset(self) -> None:
        """Reset circuit breaker to closed state (for testing)."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            self._stats = CircuitBreakerStats()
            self._logger.info("Circuit breaker reset", service=self._name)


# Global registry of circuit breakers for centralized management
_circuit_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = asyncio.Lock()
_MAX_CIRCUIT_BREAKERS = 100  # Prevent unbounded growth


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name.

    Note: For thread-safe creation in async context, use get_circuit_breaker_async.

    Args:
        name: Service name
        config: Configuration (only used on first creation)

    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        if len(_circuit_breakers) >= _MAX_CIRCUIT_BREAKERS:
            raise RuntimeError(
                f"Maximum circuit breakers ({_MAX_CIRCUIT_BREAKERS}) reached"
            )
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


async def get_circuit_breaker_async(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name (async/thread-safe).

    Args:
        name: Service name
        config: Configuration (only used on first creation)

    Returns:
        CircuitBreaker instance
    """
    async with _registry_lock:
        if name not in _circuit_breakers:
            if len(_circuit_breakers) >= _MAX_CIRCUIT_BREAKERS:
                raise RuntimeError(
                    f"Maximum circuit breakers ({_MAX_CIRCUIT_BREAKERS}) reached"
                )
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def get_all_circuit_breaker_stats() -> dict[str, dict[str, Any]]:
    """Get stats for all registered circuit breakers."""
    return {name: breaker.stats.to_dict() for name, breaker in _circuit_breakers.items()}
