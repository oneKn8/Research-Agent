"""Base worker class for all worker types.

Provides common functionality including:
- Retry logic with exponential backoff
- Circuit breaker for resilience
- Error handling
- Cost tracking
- Metrics collection
- Logging
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar

from src.config import get_settings
from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    get_circuit_breaker,
)
from src.utils.logging import get_logger
from src.utils.metrics import record_worker_execution

T = TypeVar("T")


@dataclass
class WorkerResult:
    """Result from a worker operation."""

    success: bool
    data: Any
    error: str | None = None
    cost_usd: float = 0.0
    tokens_used: int = 0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostTracker:
    """Tracks API costs for a worker."""

    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    def record_request(
        self,
        cost_usd: float = 0.0,
        tokens: int = 0,
        success: bool = True,
    ) -> None:
        """Record a request's cost and outcome."""
        self.total_cost_usd += cost_usd
        self.total_tokens += tokens
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
        }


class BaseWorker(ABC):
    """Abstract base class for all workers.

    Subclasses must implement:
    - _execute(): The actual work logic
    - worker_type: String identifying the worker type

    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker protection for external services
    - Cost tracking and metrics collection
    """

    # Circuit breaker configuration (can be overridden in subclasses)
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
    CIRCUIT_BREAKER_TIMEOUT_SECONDS = 30.0
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2

    def __init__(self, use_circuit_breaker: bool = True) -> None:
        """Initialize the worker.

        Args:
            use_circuit_breaker: Whether to enable circuit breaker protection
        """
        self._settings = get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._cost_tracker = CostTracker()
        self._use_circuit_breaker = use_circuit_breaker
        self._circuit_breaker: CircuitBreaker | None = None

    def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create the circuit breaker for this worker."""
        if self._circuit_breaker is None:
            config = CircuitBreakerConfig(
                failure_threshold=self.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                timeout_seconds=self.CIRCUIT_BREAKER_TIMEOUT_SECONDS,
                success_threshold=self.CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
            )
            self._circuit_breaker = get_circuit_breaker(
                f"worker.{self.worker_type}", config
            )
        return self._circuit_breaker

    @property
    @abstractmethod
    def worker_type(self) -> str:
        """Return the worker type identifier."""
        ...

    @abstractmethod
    async def _execute(self, *args: Any, **kwargs: Any) -> WorkerResult:
        """Execute the worker's main logic.

        Subclasses implement this method with their specific functionality.
        """
        ...

    async def execute(self, *args: Any, **kwargs: Any) -> WorkerResult:
        """Execute the worker with retry logic, circuit breaker, and metrics.

        This is the public interface for running workers.

        Raises:
            CircuitBreakerOpen: If circuit breaker is open and rejecting requests
        """
        start_time = datetime.now()
        max_retries = kwargs.pop("max_retries", self._settings.max_retries)
        last_error: Exception | None = None

        # Check circuit breaker first
        if self._use_circuit_breaker:
            breaker = self._get_circuit_breaker()
            try:
                await breaker._can_execute()
            except CircuitBreakerOpen:
                self._logger.warning(
                    "Circuit breaker open, rejecting request",
                    worker_type=self.worker_type,
                )
                raise

        for attempt in range(max_retries + 1):
            try:
                result = await self._execute(*args, **kwargs)

                # Track costs
                self._cost_tracker.record_request(
                    cost_usd=result.cost_usd,
                    tokens=result.tokens_used,
                    success=result.success,
                )

                # Add duration
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                result.duration_ms = duration_ms

                # Record metrics
                record_worker_execution(
                    worker_type=self.worker_type,
                    success=result.success,
                    duration_ms=duration_ms,
                    cost_usd=result.cost_usd,
                    tokens=result.tokens_used,
                )

                # Record circuit breaker success
                if self._use_circuit_breaker:
                    await self._get_circuit_breaker().record_success()

                self._logger.info(
                    "Worker execution completed",
                    worker_type=self.worker_type,
                    success=result.success,
                    duration_ms=round(duration_ms, 2),
                    cost_usd=result.cost_usd,
                )

                return result

            except Exception as e:
                last_error = e
                self._cost_tracker.record_request(success=False)

                if attempt < max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    self._logger.warning(
                        "Worker execution failed, retrying",
                        worker_type=self.worker_type,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        wait_seconds=wait_time,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self._logger.error(
                        "Worker execution failed after all retries",
                        worker_type=self.worker_type,
                        attempts=max_retries + 1,
                        error=str(e),
                    )

        # All retries exhausted - record circuit breaker failure
        if self._use_circuit_breaker:
            await self._get_circuit_breaker().record_failure(last_error)

        # Record metrics for failed execution
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        record_worker_execution(
            worker_type=self.worker_type,
            success=False,
            duration_ms=duration_ms,
        )

        return WorkerResult(
            success=False,
            data=None,
            error=str(last_error) if last_error else "Unknown error",
            duration_ms=duration_ms,
        )

    def get_cost_summary(self) -> dict[str, Any]:
        """Get the cost tracking summary."""
        return self._cost_tracker.to_dict()

    def reset_cost_tracker(self) -> None:
        """Reset the cost tracker."""
        self._cost_tracker = CostTracker()

    def get_circuit_breaker_stats(self) -> dict[str, Any] | None:
        """Get circuit breaker statistics if enabled."""
        if not self._use_circuit_breaker:
            return None
        return self._get_circuit_breaker().stats.to_dict()

    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is currently open."""
        if not self._use_circuit_breaker:
            return False
        from src.utils.circuit_breaker import CircuitState

        return self._get_circuit_breaker().state == CircuitState.OPEN


class LLMWorker(BaseWorker):
    """Base class for workers that use LLM APIs (OpenAI).

    Provides common LLM functionality including:
    - OpenAI client management
    - Token counting
    - Cost calculation for GPT-4o-mini
    """

    # GPT-4o-mini pricing (as of 2026)
    INPUT_COST_PER_1K = 0.00015  # $0.15 per 1M tokens = $0.00015 per 1K
    OUTPUT_COST_PER_1K = 0.0006  # $0.60 per 1M tokens = $0.0006 per 1K

    def __init__(self) -> None:
        """Initialize the LLM worker."""
        super().__init__()
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Get or create the OpenAI async client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._settings.openai_api_key.get_secret_value(),
            )
        return self._client

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for GPT-4o-mini usage."""
        input_cost = (input_tokens / 1000) * self.INPUT_COST_PER_1K
        output_cost = (output_tokens / 1000) * self.OUTPUT_COST_PER_1K
        return input_cost + output_cost

    async def close(self) -> None:
        """Close the OpenAI client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
