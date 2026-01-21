# Architecture Reference

## System Overview

```
USER INTERFACE
    |
    v
ORCHESTRATOR (LangGraph)
- State management, workflow control, error handling
    |
    +------------------+------------------+
    |                  |                  |
    v                  v                  v
RESEARCH BRAIN     SEARCH WORKERS    WRITING WORKERS
(DeepSeek-R1-14B)  (GPT-4o-mini)     (GPT-4o-mini)
- Planning         - Web search       - LaTeX drafting
- Analysis         - ArXiv API        - Citations
- Synthesis        - Paper retrieval  - BibTeX
SELF-HOSTED        API-BASED          API-BASED
    |                  |                  |
    +------------------+------------------+
                       |
                       v
               MEMORY SYSTEM
- Short-term: 128K context
- Long-term: Vector DB (deferred)
                       |
                       v
               OUTPUT LAYER
- LaTeX papers, JSON metadata, logs
```

## Directory Structure

```
research-agent/
├── CLAUDE.md              # Rules (lean)
├── BUILDPLAN.md           # Implementation phases
├── RESEARCH_REPORT.md     # Research findings
├── pyproject.toml         # Dependencies
├── docker-compose.yml     # Production
├── docker-compose.dev.yml # Development
├── .env.example           # Env template
├── docs/                  # Documentation
│   ├── PROGRESS.md        # Progress & notes
│   ├── ARCHITECTURE.md    # This file
│   └── PHASE_CHECKLISTS.md
├── src/
│   ├── main.py            # FastAPI entry
│   ├── config.py          # Configuration
│   ├── brain/             # vLLM client
│   ├── workers/           # API workers
│   ├── pipeline/          # LangGraph
│   ├── output/            # LaTeX gen
│   ├── security/          # Security utils
│   └── utils/             # Logging, errors
├── services/brain/        # vLLM Dockerfile
├── infra/postgres/        # DB init
├── scripts/               # Setup scripts
├── tests/                 # Test suite
├── outputs/               # Generated papers
└── storage/               # Local storage
```

## Key Files

| File | Purpose |
|------|---------|
| src/main.py | FastAPI app, research endpoints |
| src/config.py | Pydantic settings |
| src/brain/client.py | BrainClient for vLLM |
| src/brain/prompts.py | 12 research prompts |
| src/workers/search.py | Tavily search |
| src/workers/arxiv.py | ArXiv API |
| src/workers/writer.py | GPT-4o-mini writing |
| src/pipeline/state.py | ResearchState TypedDict |
| src/pipeline/nodes.py | 7 workflow nodes |
| src/pipeline/graph.py | StateGraph routing |
| src/pipeline/workflow.py | Execution interface |
| src/output/bibtex.py | Citation management |
| src/output/latex.py | LaTeX generator |
| src/output/compiler.py | PDF compilation |
| src/security/latex.py | LaTeX sanitization |

## Tech Stack

### Models
- **Brain**: DeepSeek-R1-Distill-Qwen-14B (128K, ~9GB Q4)
- **Workers**: GPT-4o-mini ($0.15/M in, $0.60/M out)

### Python Dependencies
- openai 1.59.9, tavily-python 0.5.0, arxiv 2.1.3
- fastapi 0.115.6, pydantic 2.10.4, structlog 24.4.0
- langgraph 1.0.6, langgraph-checkpoint-postgres 3.0.3
- bibtexparser 2.0.0b7, pylatexenc 2.10, jinja2 3.1.5

### Infrastructure
- Docker, Redis 7.4-alpine, PostgreSQL 16-alpine, vLLM 0.13.0

### Frontend (Planned)
- Next.js 15, TypeScript, Tailwind CSS, Aceternity UI, Motion

### Cloud
- Brev.dev: L40S dev ($1.03/hr), A100 training ($1.49/hr)
- Vast.ai: A100 backup ($0.82-1.27/hr)
