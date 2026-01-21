"""Research Agent API - Main entry point.

FastAPI application for the deep research AI system.
Handles research queries and generates publication-ready LaTeX papers.
"""

import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings
from src.pipeline import ResearchDomain, ResearchWorkflow, get_graph_ascii
from src.utils.circuit_breaker import get_all_circuit_breaker_stats
from src.utils.errors import RateLimitError, ResearchAgentError, ValidationError
from src.utils.logging import bind_context, clear_context, get_logger, setup_logging
from src.utils.metrics import MetricsMiddleware, get_metrics

# Version info
__version__ = "0.1.0"


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    setup_logging(
        level=settings.log_level,
        json_logs=settings.log_json,
        service_name="research-agent",
    )

    logger = get_logger()
    logger.info(
        "Starting Research Agent API",
        version=__version__,
        environment=settings.environment,
    )

    # Validate required services are configured
    logger.info("Configuration validated", brain_url=settings.brain_service_url)

    yield

    # Shutdown
    logger.info("Shutting down Research Agent API")


# =============================================================================
# Application Factory
# =============================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Research Agent API",
        description="Deep research AI system for generating publication-ready papers",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Add exception handlers
    app.add_exception_handler(ResearchAgentError, research_agent_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    # Add request middleware
    app.add_middleware(RequestMiddleware)

    # Add metrics middleware
    app.add_middleware(MetricsMiddleware)

    return app


# =============================================================================
# Middleware
# =============================================================================

# Simple in-memory rate limiter (replace with Redis in production)
_rate_limit_store: dict[str, list[float]] = {}


class RequestMiddleware(BaseHTTPMiddleware):
    """Request middleware for logging, rate limiting, and context."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process the request with logging, rate limiting, and context."""
        settings = get_settings()
        logger = get_logger()

        # Generate request ID
        request_id = str(uuid4())[:8]
        bind_context(request_id=request_id)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Rate limiting (skip for health checks)
        if request.url.path not in ("/health", "/ready"):
            current_time = time.time()
            window_start = current_time - settings.rate_limit_window_seconds

            # Clean old entries and check rate limit
            if client_ip not in _rate_limit_store:
                _rate_limit_store[client_ip] = []

            _rate_limit_store[client_ip] = [
                t for t in _rate_limit_store[client_ip] if t > window_start
            ]

            if len(_rate_limit_store[client_ip]) >= settings.rate_limit_requests:
                logger.warning("Rate limit exceeded", client_ip=client_ip)
                clear_context()
                raise RateLimitError(
                    "Rate limit exceeded. Please try again later.",
                    details={"retry_after_seconds": settings.rate_limit_window_seconds},
                )

            _rate_limit_store[client_ip].append(current_time)

        # Log request
        start_time = time.time()
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
        )

        # Process request
        response = await call_next(request)

        # Log response
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Clear context
        clear_context()

        return response


# =============================================================================
# Exception Handlers
# =============================================================================


async def research_agent_error_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle ResearchAgentError exceptions."""
    logger = get_logger()
    # Cast to ResearchAgentError since this handler is registered for that type
    err = exc if isinstance(exc, ResearchAgentError) else ResearchAgentError(str(exc))
    logger.error(
        "Request failed",
        error_code=err.error_code,
        message=err.message,
        details=err.details,
    )
    return JSONResponse(
        status_code=err.status_code,
        content=err.to_dict(),
    )


async def generic_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger = get_logger()
    logger.exception("Unexpected error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": {},
        },
    )


# =============================================================================
# Create Application
# =============================================================================

app = create_app()


# =============================================================================
# Health Check Endpoints
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    environment: str


class ReadyResponse(BaseModel):
    """Readiness check response model."""

    status: str
    checks: dict[str, Any]


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Basic health check endpoint.

    Returns 200 if the service is running.
    Used by container orchestration for liveness checks.
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=__version__,
        environment=settings.environment,
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def readiness_check() -> ReadyResponse:
    """Readiness check endpoint.

    Verifies that all required services are available.
    Used by container orchestration for readiness checks.
    """
    settings = get_settings()
    checks: dict[str, Any] = {
        "api": True,
        "brain_configured": bool(settings.brain_service_url),
        "openai_configured": bool(settings.openai_api_key.get_secret_value()),
        "tavily_configured": bool(settings.tavily_api_key.get_secret_value()),
    }

    # TODO: Add actual connectivity checks to brain service

    all_ready = all(checks.values())

    return ReadyResponse(
        status="ready" if all_ready else "not_ready",
        checks=checks,
    )


# =============================================================================
# Metrics Endpoints
# =============================================================================


class MetricsResponse(BaseModel):
    """Metrics response model."""

    counters: list[dict[str, Any]]
    histograms: list[dict[str, Any]]
    gauges: list[dict[str, Any]]
    circuit_breakers: dict[str, Any]


@app.get("/metrics", tags=["Monitoring"])
async def get_prometheus_metrics() -> Response:
    """Get metrics in Prometheus text format.

    Returns metrics suitable for Prometheus scraping.
    """
    metrics = get_metrics()
    return Response(
        content=metrics.to_prometheus_format(),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/metrics/json", response_model=MetricsResponse, tags=["Monitoring"])
async def get_json_metrics() -> MetricsResponse:
    """Get metrics in JSON format.

    Returns metrics for debugging and dashboards.
    """
    metrics = get_metrics()
    metrics_dict = metrics.to_dict()
    return MetricsResponse(
        counters=metrics_dict["counters"],
        histograms=metrics_dict["histograms"],
        gauges=metrics_dict["gauges"],
        circuit_breakers=get_all_circuit_breaker_stats(),
    )


# =============================================================================
# Main Entry Point
# =============================================================================


# =============================================================================
# Research Endpoints
# =============================================================================


class ResearchRequest(BaseModel):
    """Request model for research queries."""

    query: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The research question or topic",
    )
    domains: list[str] | None = Field(
        default=None,
        description="Research domains to focus on (ai_ml, quantum_physics, astrophysics, general)",
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum search iterations",
    )


class ResearchResponse(BaseModel):
    """Response model for research results."""

    thread_id: str
    status: str
    title: str | None = None
    abstract: str | None = None
    sections: dict[str, str] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    quality_score: float | None = None
    cost_usd: float = 0.0
    tokens_used: int = 0
    errors: list[str] = Field(default_factory=list)


class ResearchStatusResponse(BaseModel):
    """Response model for research status."""

    thread_id: str
    query: str
    status: str
    iteration_count: int
    sources_count: int
    sections_count: int
    has_synthesis: bool
    quality_score: float | None
    cost_usd: float
    tokens_used: int
    errors: list[str]


class WorkflowGraphResponse(BaseModel):
    """Response model for workflow graph visualization."""

    ascii_diagram: str
    node_descriptions: dict[str, str]


@app.post("/research", response_model=ResearchResponse, tags=["Research"])
async def start_research(request: ResearchRequest) -> ResearchResponse:
    """Start a new research workflow.

    Creates a research plan, searches relevant sources, analyzes findings,
    and generates a draft research paper.

    Note: This endpoint may take several minutes to complete.
    For real-time updates, use the /research/stream endpoint.
    """
    logger = get_logger()
    logger.info("Starting research request", query=request.query[:100])

    # Validate domains
    valid_domains = [d.value for d in ResearchDomain]
    if request.domains:
        invalid = [d for d in request.domains if d not in valid_domains]
        if invalid:
            raise ValidationError(
                f"Invalid domains: {invalid}. Valid options: {valid_domains}",
                details={"invalid_domains": invalid, "valid_domains": valid_domains},
            )

    workflow = ResearchWorkflow(use_checkpointing=False)

    result = await workflow.run(
        query=request.query,
        domains=request.domains,
        max_iterations=request.max_iterations,
        timeout_seconds=600,
    )

    return ResearchResponse(
        thread_id=result.thread_id,
        status=result.status,
        title=result.title,
        abstract=result.abstract,
        sections=result.sections,
        citations=result.citations,
        quality_score=result.quality_score,
        cost_usd=round(result.cost_usd, 6),
        tokens_used=result.tokens_used,
        errors=result.errors,
    )


@app.post("/research/stream", tags=["Research"])
async def stream_research(request: ResearchRequest) -> EventSourceResponse:
    """Start a research workflow with streaming updates.

    Returns Server-Sent Events (SSE) with real-time progress updates.
    Final result is sent as the last event.

    Events:
    - progress: Intermediate updates with current node and progress
    - complete: Final result with full research output
    - error: Error event if workflow fails
    """
    logger = get_logger()
    logger.info("Starting streaming research", query=request.query[:100])

    # Validate domains
    valid_domains = [d.value for d in ResearchDomain]
    if request.domains:
        invalid = [d for d in request.domains if d not in valid_domains]
        if invalid:
            raise ValidationError(
                f"Invalid domains: {invalid}",
                details={"invalid_domains": invalid},
            )

    async def event_generator() -> Any:
        """Generate SSE events from workflow stream."""
        import json

        workflow = ResearchWorkflow(use_checkpointing=False)

        try:
            async for update in workflow.stream(
                query=request.query,
                domains=request.domains,
                max_iterations=request.max_iterations,
            ):
                if update.node == "end":
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "status": update.status,
                            "message": update.message,
                            "data": update.data,
                        }),
                    }
                elif update.node == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "status": update.status,
                            "message": update.message,
                            "data": update.data,
                        }),
                    }
                else:
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "node": update.node,
                            "status": update.status,
                            "message": update.message,
                            "progress": update.progress,
                            "data": update.data,
                        }),
                    }
        except Exception as e:
            logger.error("Streaming error", error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())


@app.get("/research/graph", response_model=WorkflowGraphResponse, tags=["Research"])
async def get_workflow_graph() -> WorkflowGraphResponse:
    """Get the research workflow graph visualization.

    Returns an ASCII diagram of the workflow structure and
    descriptions of each node.
    """
    return WorkflowGraphResponse(
        ascii_diagram=get_graph_ascii(),
        node_descriptions={
            "plan": "Brain creates research plan from user query",
            "search": "Workers execute web and ArXiv searches in parallel",
            "analyze": "Brain analyzes gathered sources and identifies gaps",
            "generate_queries": "Brain generates additional search queries for gaps",
            "synthesize": "Brain synthesizes all findings into coherent narrative",
            "write": "Workers draft paper sections using GPT-4o-mini",
            "review": "Brain reviews paper for quality and accuracy",
        },
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the application using uvicorn."""
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
