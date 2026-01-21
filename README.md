# Research Agent

A deep research AI system that generates publication-ready LaTeX papers. Built with a self-hosted reasoning model (DeepSeek-R1-Distill-Qwen-14B) and GPT-4o-mini workers for search and extraction.

## Features

- **Deep Research**: Searches web and ArXiv for relevant papers and sources
- **Self-Hosted Brain**: 128K context reasoning model via vLLM
- **Publication-Ready Output**: LaTeX papers with proper BibTeX citations
- **Domain Focus**: AI/ML, Quantum Physics, Astrophysics

## Architecture

```
User Query
    |
    v
+------------------+
|   FastAPI        |
+------------------+
    |
    v
+------------------+
|   LangGraph      |  Orchestration
+------------------+
    |
    +---> Brain (DeepSeek-R1-Distill-14B, self-hosted)
    |       - Planning, Analysis, Synthesis
    |
    +---> Workers (GPT-4o-mini API)
            - Web search (Tavily)
            - ArXiv search
            - LaTeX drafting
    |
    v
+------------------+
|   LaTeX Output   |
+------------------+
```

## Requirements

- Python 3.12+
- Docker and Docker Compose
- NVIDIA GPU with 24GB+ VRAM (for local brain service)
- API keys: OpenAI, Tavily

## Quick Start

1. **Clone and setup**
   ```bash
   git clone git@github.com:Sant0-9/Research-Agent.git
   cd Research-Agent
   ./scripts/setup.sh
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start services**
   ```bash
   # Development mode (no local GPU)
   ./scripts/run_local.sh

   # With local GPU brain service
   ./scripts/run_local.sh --gpu
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs (development only)

## Project Structure

```
research-agent/
├── src/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration
│   ├── brain/            # Brain service client
│   ├── workers/          # Search and writing workers
│   ├── pipeline/         # LangGraph orchestration
│   ├── output/           # LaTeX generation
│   └── utils/            # Logging, errors
├── services/brain/       # vLLM brain service
├── infra/                # Infrastructure configs
├── scripts/              # Setup and run scripts
├── tests/                # Test suite
└── outputs/              # Generated papers
```

## Configuration

Key environment variables (see `.env.example` for full list):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o-mini workers |
| `TAVILY_API_KEY` | Tavily API key for web search |
| `BRAIN_SERVICE_URL` | URL of vLLM brain service |
| `BRAIN_API_KEY` | Internal auth key for brain service |

## Development

```bash
# Run tests
./scripts/test.sh

# Run tests with coverage
./scripts/test.sh --coverage

# Lint and type check
ruff check src/
mypy src/
```

## Tech Stack

- **Backend**: FastAPI 0.115.6, Python 3.12
- **Orchestration**: LangGraph 1.0.6
- **Brain Model**: DeepSeek-R1-Distill-Qwen-14B (128K context)
- **Inference**: vLLM 0.13.0
- **Workers**: GPT-4o-mini (OpenAI API)
- **Search**: Tavily API
- **Database**: PostgreSQL 16
- **Cache**: Redis 7.4
- **Output**: LaTeX with BibTeX

## License

MIT
