# Research Agent

A deep research AI system that generates publication-ready LaTeX papers. Built with a self-hosted reasoning model (DeepSeek-R1-Distill-Qwen-14B) and GPT-4o-mini workers for search and extraction.

## Features

- **Deep Research**: Searches web and ArXiv for relevant papers and sources
- **Self-Hosted Brain**: 128K context reasoning model via vLLM
- **Publication-Ready Output**: LaTeX papers with proper BibTeX citations
- **Domain Focus**: AI/ML, Quantum Physics, Astrophysics
- **Resilient**: Circuit breaker protection for external services
- **Observable**: Prometheus-compatible metrics and structured logging

## Architecture

```
User Query
    |
    v
+------------------+
|   FastAPI        |  API Layer (rate limiting, validation)
+------------------+
    |
    v
+------------------+
|   LangGraph      |  Orchestration (state machine, checkpointing)
+------------------+
    |
    +---> Brain (DeepSeek-R1-Distill-14B, self-hosted vLLM)
    |       - Planning: Creates research strategy
    |       - Analysis: Evaluates gathered sources
    |       - Synthesis: Combines findings
    |       - Review: Quality checks output
    |
    +---> Workers (GPT-4o-mini API)
            - Web search (Tavily API)
            - ArXiv search and paper retrieval
            - LaTeX section drafting
    |
    v
+------------------+
|   LaTeX Output   |  Papers with BibTeX citations
+------------------+
```

## Requirements

- Python 3.12+
- Docker and Docker Compose
- NVIDIA GPU with 24GB+ VRAM (for local brain service)
- API keys: OpenAI, Tavily

## Quick Start

### 1. Clone and setup

```bash
git clone <repository-url>
cd research-agent
./scripts/setup.sh
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key
TAVILY_API_KEY=tvly-your-tavily-api-key

# Brain service (use mock for development)
BRAIN_SERVICE_URL=http://localhost:8001
BRAIN_API_KEY=your-internal-brain-api-key
```

### 3. Start services

```bash
# Development mode (external brain service or mock)
docker compose up -d redis postgres
./scripts/run_local.sh

# With local GPU brain service
docker compose up -d
```

### 4. Access the API

- API: http://localhost:8000
- Docs: http://localhost:8000/docs (development only)
- Metrics: http://localhost:8000/metrics

## API Reference

### Research Endpoints

#### POST /research

Start a synchronous research workflow.

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest advances in quantum error correction?",
    "domains": ["quantum_physics"],
    "max_iterations": 3
  }'
```

**Request Body:**
| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Research question (10-2000 chars) |
| `domains` | array | Optional: `["ai_ml", "quantum_physics", "astrophysics", "general"]` |
| `max_iterations` | int | Max search iterations (1-5, default: 3) |

**Response:**
```json
{
  "thread_id": "abc123",
  "status": "completed",
  "title": "Advances in Quantum Error Correction",
  "abstract": "...",
  "sections": {
    "introduction": "...",
    "methodology": "...",
    "results": "...",
    "conclusion": "..."
  },
  "citations": [...],
  "quality_score": 0.85,
  "cost_usd": 0.0234,
  "tokens_used": 15420
}
```

#### POST /research/stream

Start a research workflow with streaming updates (SSE).

```bash
curl -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "Explain transformer attention mechanisms"}'
```

#### GET /research/graph

Get the workflow graph visualization.

### Health and Monitoring

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic liveness check |
| `GET /ready` | Readiness check (all services configured) |
| `GET /metrics` | Prometheus metrics (text format) |
| `GET /metrics/json` | Metrics with circuit breaker stats (JSON) |

## Project Structure

```
research-agent/
├── src/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Pydantic settings
│   ├── brain/            # vLLM client and prompts
│   │   ├── client.py     # Async brain client
│   │   └── prompts.py    # 12 research prompt templates
│   ├── workers/          # GPT-4o-mini workers
│   │   ├── base.py       # BaseWorker with retry/circuit breaker
│   │   ├── search.py     # Tavily web search
│   │   ├── arxiv.py      # ArXiv paper retrieval
│   │   └── writer.py     # LaTeX section writer
│   ├── pipeline/         # LangGraph orchestration
│   │   ├── state.py      # ResearchState TypedDict
│   │   ├── nodes.py      # 7 workflow nodes
│   │   ├── graph.py      # StateGraph with routing
│   │   └── workflow.py   # Execution interface
│   ├── output/           # LaTeX generation
│   │   ├── latex.py      # Jinja2 template rendering
│   │   ├── bibtex.py     # Citation management
│   │   └── compiler.py   # PDF compilation
│   ├── security/         # Security utilities
│   │   └── latex.py      # LaTeX sanitization
│   └── utils/            # Shared utilities
│       ├── logging.py    # Structured logging
│       ├── errors.py     # Custom exceptions
│       ├── circuit_breaker.py  # Resilience pattern
│       └── metrics.py    # Prometheus metrics
├── services/brain/       # vLLM brain service Docker
├── infra/postgres/       # Database initialization
├── scripts/              # Setup and run scripts
├── tests/                # Test suite (190+ tests)
└── outputs/              # Generated papers
```

## Configuration

### Environment Variables

See `.env.example` for the complete list. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for workers | Required |
| `TAVILY_API_KEY` | Tavily API key for web search | Required |
| `BRAIN_SERVICE_URL` | vLLM brain service URL | `http://localhost:8001` |
| `BRAIN_API_KEY` | Internal auth for brain service | Required |
| `ENVIRONMENT` | `development`, `staging`, `production` | `development` |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `RATE_LIMIT_REQUESTS` | Requests per minute per IP | `60` |
| `MAX_SEARCH_ITERATIONS` | Max search loops | `3` |
| `API_TIMEOUT_SECONDS` | External API timeout | `30` |

### Brain Service (DeepSeek-R1)

The brain model requires specific settings:

- **Temperature**: 0.5-0.7 (recommended: 0.6)
- **Top-p**: 0.95
- **No system prompts**: R1-Distill models don't use system prompts
- **Uses `<think>` tags**: For chain-of-thought reasoning

## Development

### Running Tests

```bash
# Run all tests
./scripts/test.sh

# Run with coverage
./scripts/test.sh --coverage

# Run specific test file
pytest tests/test_workers.py -v

# Run specific test
pytest tests/test_workers.py::test_search_worker_success -v
```

### Code Quality

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/

# Security audit
pip-audit
```

### Local Development Without GPU

For development without a local GPU:

1. Use a cloud GPU service (Brev.dev, Vast.ai) for the brain service
2. Update `BRAIN_SERVICE_URL` in `.env` to point to the remote service
3. Run the API locally: `./scripts/run_local.sh`

## Troubleshooting

### Common Issues

**Circuit breaker open**
```
CircuitBreakerOpen: Circuit breaker open for worker.search
```
The external service has failed multiple times. Wait 30 seconds for recovery or check the service status.

**Rate limit exceeded**
```
RateLimitError: Rate limit exceeded. Please try again later.
```
The API has hit the rate limit (default: 60 requests/minute). Wait or increase `RATE_LIMIT_REQUESTS`.

**Brain service unreachable**
```
BrainServiceError: Failed to connect to brain service
```
Check that the vLLM service is running and `BRAIN_SERVICE_URL` is correct.

**LaTeX compilation failed**
```
LaTeX compilation failed: Missing package
```
Ensure `pdflatex` is installed with required packages. On Ubuntu:
```bash
sudo apt-get install texlive-latex-base texlive-fonts-recommended
```

### Checking Service Health

```bash
# API health
curl http://localhost:8000/health

# Service readiness
curl http://localhost:8000/ready

# Metrics and circuit breaker status
curl http://localhost:8000/metrics/json
```

### Logs

Logs are structured JSON in production mode. Enable debug logging:

```bash
LOG_LEVEL=DEBUG ./scripts/run_local.sh
```

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| API Framework | FastAPI | 0.115.6 |
| Python | Python | 3.12 |
| Orchestration | LangGraph | 1.0.6 |
| Brain Model | DeepSeek-R1-Distill-Qwen-14B | - |
| Inference Server | vLLM | 0.13.0 |
| Worker LLM | GPT-4o-mini | - |
| Web Search | Tavily API | 0.5.0 |
| Database | PostgreSQL | 16 |
| Cache | Redis | 7.4 |
| Output | LaTeX + BibTeX | - |
| Monitoring | OpenTelemetry | 1.29.0 |

## Cost Estimates

| Operation | Estimated Cost |
|-----------|----------------|
| Single research paper | $0.02-0.05 |
| Web search (Tavily) | Free tier: 1000/month |
| ArXiv search | Free |
| Brain inference | Self-hosted (GPU cost) |

## License

MIT
