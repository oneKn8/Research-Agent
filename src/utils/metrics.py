"""Prometheus metrics for observability.

Provides metrics for:
- API request counts and latencies
- Worker execution metrics
- Circuit breaker states
- Cost tracking
- Error rates
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MetricsRegistry:
    """Simple metrics registry without external dependencies.

    Can be exported to Prometheus format or used with OpenTelemetry.
    """

    def __init__(self) -> None:
        self._counters: dict[str, dict[str, Any]] = {}
        self._histograms: dict[str, dict[str, Any]] = {}
        self._gauges: dict[str, dict[str, Any]] = {}

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        if key not in self._counters:
            self._counters[key] = {"name": name, "labels": labels or {}, "value": 0.0}
        self._counters[key]["value"] += value

    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = {
                "name": name,
                "labels": labels or {},
                "values": [],
                "sum": 0.0,
                "count": 0,
            }
        self._histograms[key]["values"].append(value)
        self._histograms[key]["sum"] += value
        self._histograms[key]["count"] += 1

    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric value."""
        key = self._make_key(name, labels)
        self._gauges[key] = {"name": name, "labels": labels or {}, "value": value}

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines: list[str] = []

        # Export counters
        for data in self._counters.values():
            name = data["name"]
            labels = data["labels"]
            value = data["value"]
            label_str = self._format_labels(labels)
            lines.append(f"{name}{label_str} {value}")

        # Export histograms (simplified - just sum and count)
        for data in self._histograms.values():
            name = data["name"]
            labels = data["labels"]
            label_str = self._format_labels(labels)
            lines.append(f"{name}_sum{label_str} {data['sum']}")
            lines.append(f"{name}_count{label_str} {data['count']}")

        # Export gauges
        for data in self._gauges.values():
            name = data["name"]
            labels = data["labels"]
            value = data["value"]
            label_str = self._format_labels(labels)
            lines.append(f"{name}{label_str} {value}")

        return "\n".join(lines)

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{{{label_str}}}"

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "counters": list(self._counters.values()),
            "histograms": [
                {
                    "name": h["name"],
                    "labels": h["labels"],
                    "sum": h["sum"],
                    "count": h["count"],
                    "avg": h["sum"] / h["count"] if h["count"] > 0 else 0,
                }
                for h in self._histograms.values()
            ],
            "gauges": list(self._gauges.values()),
        }

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        self._counters.clear()
        self._histograms.clear()
        self._gauges.clear()


# Global metrics registry
_metrics = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _metrics


# Convenience functions for common metrics
def record_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Record HTTP request metrics."""
    labels = {"method": method, "path": _normalize_path(path), "status": str(status_code)}
    _metrics.counter("http_requests_total", labels=labels)
    _metrics.histogram("http_request_duration_ms", duration_ms, labels=labels)


def record_worker_execution(
    worker_type: str,
    success: bool,
    duration_ms: float,
    cost_usd: float = 0.0,
    tokens: int = 0,
) -> None:
    """Record worker execution metrics."""
    labels = {"worker_type": worker_type, "success": str(success).lower()}
    _metrics.counter("worker_executions_total", labels=labels)
    _metrics.histogram("worker_duration_ms", duration_ms, labels={"worker_type": worker_type})

    if cost_usd > 0:
        _metrics.counter(
            "worker_cost_usd_total", cost_usd, labels={"worker_type": worker_type}
        )

    if tokens > 0:
        _metrics.counter(
            "worker_tokens_total", float(tokens), labels={"worker_type": worker_type}
        )


def record_circuit_breaker_state(name: str, state: str) -> None:
    """Record circuit breaker state change."""
    # Use gauge for current state (1 = open, 0 = closed, 0.5 = half_open)
    state_value = {"closed": 0.0, "open": 1.0, "half_open": 0.5}.get(state, 0.0)
    _metrics.gauge("circuit_breaker_state", state_value, labels={"service": name})


def record_research_workflow(
    status: str,
    duration_ms: float,
    cost_usd: float,
    tokens: int,
) -> None:
    """Record research workflow completion metrics."""
    labels = {"status": status}
    _metrics.counter("research_workflows_total", labels=labels)
    _metrics.histogram("research_workflow_duration_ms", duration_ms, labels={})
    _metrics.counter("research_workflow_cost_usd_total", cost_usd, labels={})
    _metrics.counter("research_workflow_tokens_total", float(tokens), labels={})


def _normalize_path(path: str) -> str:
    """Normalize path for metrics (replace IDs with placeholders)."""
    # Replace UUIDs and numeric IDs with placeholders
    import re

    path = re.sub(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "/{id}", path)
    path = re.sub(r"/\d+", "/{id}", path)
    return path


class MetricsMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic request metrics."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process request and record metrics."""
        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        # Record metrics (skip health checks for cleaner data)
        if request.url.path not in ("/health", "/ready", "/metrics"):
            record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

        return response
