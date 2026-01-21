"""Utility modules for the research agent.

Components:
- logging: Structured logging with contextvars
- errors: Custom exception classes
- circuit_breaker: Circuit breaker pattern for resilience
- metrics: Prometheus-compatible metrics
"""

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerStats,
    CircuitState,
    get_all_circuit_breaker_stats,
    get_circuit_breaker,
)
from src.utils.errors import (
    BrainServiceError,
    ConfigurationError,
    RateLimitError,
    ResearchAgentError,
    ValidationError,
    WorkerError,
)
from src.utils.logging import bind_context, clear_context, get_logger, setup_logging
from src.utils.metrics import (
    MetricsMiddleware,
    MetricsRegistry,
    get_metrics,
    record_circuit_breaker_state,
    record_request,
    record_research_workflow,
    record_worker_execution,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "bind_context",
    "clear_context",
    # Errors
    "ResearchAgentError",
    "ValidationError",
    "ConfigurationError",
    "BrainServiceError",
    "WorkerError",
    "RateLimitError",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitBreakerStats",
    "CircuitState",
    "get_circuit_breaker",
    "get_all_circuit_breaker_stats",
    # Metrics
    "MetricsRegistry",
    "MetricsMiddleware",
    "get_metrics",
    "record_request",
    "record_worker_execution",
    "record_circuit_breaker_state",
    "record_research_workflow",
]
