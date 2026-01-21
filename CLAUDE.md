# CLAUDE.md - Project Rules & Progress Tracker

## Project: Deep Research AI System

**Purpose**: Self-evolving AI research assistant for deep research across physics, quantum mechanics, astronomy, and ML/AI - outputs publication-ready LaTeX papers.

**Started**: 2026-01-20

**Run Mode**: COMPREHENSIVE - Research before implementing. Verify everything. Ask questions if uncertain.

---

## CURRENT BUILD SCOPE (IMPORTANT)

**DEFERRED FOR LATER:**
- Long-term memory (Vector DB, embeddings, persistent knowledge)
- Self-evolution system (trajectory logging, retraining loops)
- Fine-tuning pipeline (CPT, SFT) - will use base model first

**CURRENT FOCUS:**
- Core research agent with short-term memory only (128K context window)
- Brain + Workers architecture
- LaTeX output pipeline
- Security-hardened implementation

**WHY:** Get a working, secure, high-quality system first. Validate the core works before adding complexity. Long-term memory and self-evolution are enhancements, not MVP requirements.

---

## CRITICAL RULES (Check Before Every Major Action)

### 1. Search & Research Rules
- **ALWAYS RESEARCH BEFORE BUILDING** - Before starting any broad feature, service, or system, research current best practices (2026), available libraries and trade-offs, potential pitfalls, then create a plan before implementing
- **ALWAYS use 2026** as the current year in all web searches
- **NEVER guess** library versions, API syntax, model configurations, or hyperparameters
- **ALWAYS verify** with web search or official docs before using unfamiliar APIs
- **ALWAYS check HuggingFace** for model specifications (context window, VRAM, quantization)
- If search results are unclear, search again with different terms
- Prefer official documentation over blog posts
- For ML/AI libraries: check GitHub releases for latest stable versions

### 2. Anti-Hallucination Guidelines
- **DO NOT** invent package names, function signatures, or CLI flags
- **DO NOT** assume model capabilities - verify context window, VRAM requirements
- **DO NOT** make up training hyperparameters - reference working examples
- **DO NOT** guess benchmark scores - cite sources
- If uncertain about syntax: STOP and research
- If a tool/library seems unfamiliar: STOP and research
- When in doubt: ASK THE USER

### 3. Code Quality Rules
- **NO emojis** in code, commits, or documentation (per user preference)
- **NO placeholder code** - every file should be functional
- **Test incrementally** - verify each component works before moving on
- Follow existing patterns in the codebase
- Use type hints in Python, proper error handling everywhere
- All async code must handle timeouts and cancellation

### 4. Project-Specific Rules
- **Brain model**: DeepSeek-R1-Distill-Qwen-14B (128K context, best reasoning)
- **Worker model**: GPT-4o-mini API (web search, tool calling)
- **Self-hosted inference**: vLLM on RTX 4090
- **Training**: Unsloth + QLoRA on Brev.dev (A100)
- **Orchestration**: LangGraph for agent workflows
- **Output format**: LaTeX papers with BibTeX citations
- **Domains**: AI/ML, Quantum Physics, Astrophysics

### 5. Model & Training Rules
- **NEVER train on model outputs beyond generation depth 2**
- **ALWAYS maintain 60% human-curated data** in training corpus
- **MAX 40% synthetic data** in any training run
- **Validate diversity score** before adding new training data
- **Keep base model checkpoints** - never overwrite originals
- Follow anti-collapse guardrails from Section 7 of RESEARCH_REPORT.md

### 6. Infrastructure Rules
- All services must be containerized with Docker
- Use **multi-stage builds** for smaller images
- Include **health checks** in all containers
- Pin specific versions (no `latest` tags)
- GPU containers must specify CUDA version
- Use **non-root users** in containers

### 7. Git/Commit Rules
- **NEVER mention Claude, Anthropic, or AI** in commit messages
- **COMMIT AND PUSH after major development** - not tiny changes, but after completing a feature/service
- **Commit messages must be technical** - describe what was actually built
- Write commit messages as if a human developer wrote them
- Include technical details: what services, what features, what metrics

### 8. Security Rules (NON-NEGOTIABLE)
- **NO vulnerabilities** - Code must be secure, no shortcuts
- **Sanitize ALL inputs** - Never trust user input, API responses, or external data
- **No hardcoded secrets** - Use environment variables, never commit credentials
- **Validate and escape** - Prevent injection attacks (SQL, command, prompt)
- **Principle of least privilege** - Containers, services, and code get minimum permissions
- **Secure dependencies** - Pin versions, check for known vulnerabilities
- **No eval() or exec()** on untrusted data
- **Rate limiting** on all endpoints
- **HTTPS only** for external communications
- Someone WILL try to hack this - build like it's production from day one

### 9. Development Approach Rules
- **DO NOT SPEED RUN** - Take time to do it right
- **ASK what the user wants** - Don't guess or assume requirements
- **QUALITY OVER COST** - Don't cheap out. User can't afford to waste time/money on a bad project
- **Research before building** - Understand the problem space first
- **Validate assumptions** - If uncertain, ask or research
- **No throwaway code** - Every line should be production-worthy
- **NEVER suggest Streamlit** - User cares deeply about UI quality. Streamlit is banned.
- **UI must be polished** - No janky interfaces. If building UI, do it properly.
- **COMPREHENSIVE BRAINSTORMING** - When asked to brainstorm or research:
  - Give thorough, quality ideas - not minimal/cheap alternatives
  - Don't suggest cutting corners unless explicitly asked
  - Present the best options first, not the quickest
  - User decides what to scale back, not the assistant
  - "MVP" means focused scope, not low quality

### 10. UI/Frontend Rules (MANDATORY)

- **Framework**: Next.js 15 + TypeScript + Tailwind CSS
- **Components**: Aceternity UI (dark theme, premium animations, Tailwind + Motion based)
- **Animation**: Motion (Framer Motion) ONLY - no GSAP, no heavy libraries
- **Design Philosophy**: Intentionally minimal, premium feel
  - Clean whitespace, purposeful layouts
  - Premium typography (Inter, Geist, or similar system fonts)
  - Subtle, fast micro-interactions only
  - No flashy effects everywhere - animations must be purposeful
  - Smooth page transitions, not distracting
- **Performance**: Fast and responsive is non-negotiable
  - Animations must not block UI
  - No jank, no frame drops
  - Lazy load heavy components
- **Robustness**: UI must never break
  - Test all breakpoints (mobile, tablet, desktop)
  - Handle loading states gracefully
  - Error boundaries for component failures
  - Skeleton loaders, not spinners
- **Banned**: Streamlit, Gradio, shadcn/ui, heavy animation libraries

### 11. Progress Tracking Rules (MANDATORY)
**After completing ANY phase or significant task, you MUST update this file:**
1. **Update "Current Status"** line in Progress Tracker section
2. **Check off completed items** in the relevant Phase Checklist
3. **Add actual versions** used (not just "vLLM" but "vLLM 0.6.4")
4. **Add entry to NOTES & DECISIONS LOG** with date and details
5. **Update Tech Stack Reference** if new dependencies were added
6. **Update Context Recovery Instructions** if new services/directories were created

**This is NON-NEGOTIABLE** - the user relies on this file for context recovery.

---

## ARCHITECTURE OVERVIEW

```
+-------------------------------------------------------------------------+
|                           USER INTERFACE                                |
|  - Research query input                                                 |
|  - Progress monitoring                                                  |
|  - Paper review/editing                                                 |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                        ORCHESTRATOR (LangGraph)                         |
|  - State management                                                     |
|  - Workflow control                                                     |
|  - Error handling & retry                                               |
|  - Logging for self-evolution                                           |
+-------------------------------------------------------------------------+
                                    |
              +---------------------+---------------------+
              |                     |                     |
              v                     v                     v
+---------------------+ +---------------------+ +---------------------+
|   RESEARCH BRAIN    | |   SEARCH WORKERS    | |   WRITING WORKERS   |
| (DeepSeek-R1-14B)   | |   (GPT-4o-mini)     | |   (GPT-4o-mini)     |
|                     | |                     | |                     |
| - Planning          | | - Web search        | | - LaTeX drafting    |
| - Hypothesis gen    | | - ArXiv API         | | - Citation format   |
| - Deep analysis     | | - Paper retrieval   | | - Section writing   |
| - Synthesis         | | - Data extraction   | | - BibTeX generation |
| - Quality review    | |                     | |                     |
|                     | | (3-5 parallel)      | | (1-2 parallel)      |
| SELF-HOSTED (vLLM)  | | API-BASED           | | API-BASED           |
+---------------------+ +---------------------+ +---------------------+
         |                       |                       |
         +-----------------------+-----------------------+
                                 |
                                 v
+-------------------------------------------------------------------------+
|                          MEMORY SYSTEM                                  |
|  - Short-term: Current research context (128K tokens)                   |
|  - Long-term: Vector DB (papers read, facts learned)                    |
|  - Episodic: Research trajectories (for self-evolution)                 |
+-------------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------------+
|                          OUTPUT LAYER                                   |
|  - LaTeX paper (full format)                                            |
|  - JSON metadata (for structured storage)                               |
|  - Research logs (for training data)                                    |
+-------------------------------------------------------------------------+
```

---

## PROGRESS TRACKER

### Current Status: BUILDPLAN PHASE 4 READY - Orchestration Pipeline

### Completed Tasks
- [x] Research HPC/cloud model hosting options (Brev, Vast.ai)
- [x] Research model selection (DeepSeek, Qwen3, Mistral, Falcon)
- [x] Research fine-tuning strategies (QLoRA, Unsloth)
- [x] Research self-evolution and model collapse prevention
- [x] Research ArXiv data pipeline
- [x] Research multi-agent architectures
- [x] Create comprehensive research report (RESEARCH_REPORT.md)
- [x] Final model selection: DeepSeek-R1-Distill-Qwen-14B
- [x] Create project rules (this file)
- [x] **BUILDPLAN Phase 1: Foundation & Infrastructure** (2026-01-21)
- [x] **BUILDPLAN Phase 2: Brain Service** (2026-01-21)
- [x] **BUILDPLAN Phase 3: Worker Services** (2026-01-21)

### Pending Phases (per BUILDPLAN.md)

- [x] Phase 1: Foundation & Infrastructure
- [x] Phase 2: Brain Service
- [x] Phase 3: Worker Services
- [ ] Phase 4: Orchestration Pipeline
- [ ] Phase 5: LaTeX Output
- [ ] Phase 6: Security Hardening & Polish

---

## PHASE 0 DETAILED CHECKLIST

### 0.1 Project Scaffolding
- [ ] Create directory structure (brain/, hands/, pipeline/, training/, eval/, infra/, docs/)
- [ ] Initialize Python project with pyproject.toml
- [ ] Set up virtual environment
- [ ] Create root docker-compose.yml
- [ ] Set up .env.example for secrets

### 0.2 Development Environment
- [ ] Set up Brev.dev account and verify credits ($80)
- [ ] Test L40S instance connectivity
- [ ] Install vLLM and verify GPU access
- [ ] Download DeepSeek-R1-Distill-Qwen-14B
- [ ] Test inference speed and VRAM usage

### 0.3 Base Infrastructure
- [ ] Redis for task queue / pub-sub
- [ ] PostgreSQL for metadata and logs
- [ ] Vector DB setup (ChromaDB or Qdrant)
- [ ] MinIO or local storage for papers/embeddings

### 0.4 Verification
- [ ] Brain model runs and generates coherent output
- [ ] API keys configured (OpenAI for GPT-4o-mini)
- [ ] All containers start successfully
- [ ] Basic health checks pass

---

## PHASE 1 DETAILED CHECKLIST

### 1.1 LangGraph Orchestrator
- [ ] Define state schema for research workflow
- [ ] Implement node functions (plan, search, analyze, synthesize)
- [ ] Configure conditional edges
- [ ] Add checkpointing for long-running tasks
- [ ] Implement error handling and retry logic

### 1.2 Brain Service (Self-Hosted)
- [ ] vLLM server setup with DeepSeek-R1-Distill-Qwen-14B
- [ ] Prompt templates for research tasks
- [ ] Streaming response handling
- [ ] Context management (128K window)
- [ ] Dockerfile with CUDA support

### 1.3 Search Workers (API-Based)
- [ ] GPT-4o-mini client wrapper
- [ ] Web search tool (Tavily or SerpAPI)
- [ ] ArXiv API integration
- [ ] Parallel execution with asyncio
- [ ] Result aggregation and deduplication

### 1.4 Memory System (Basic)
- [ ] Short-term context buffer
- [ ] Paper storage and retrieval
- [ ] Basic embedding pipeline
- [ ] Vector similarity search

### 1.5 Verification
- [ ] End-to-end research query works
- [ ] Brain synthesizes worker outputs
- [ ] Logs capture full trajectory
- [ ] Cost tracking per query

---

## PHASE 2 DETAILED CHECKLIST

### 2.1 LaTeX Generation
- [ ] Paper template system (article, report formats)
- [ ] Section generation prompts
- [ ] Equation formatting (preserve LaTeX math)
- [ ] Figure/table placeholder system

### 2.2 Citation Management
- [ ] BibTeX entry generation from URLs
- [ ] Citation key management
- [ ] In-text citation formatting
- [ ] Bibliography compilation

### 2.3 Quality Assurance
- [ ] LaTeX compilation check (pdflatex)
- [ ] Citation validation
- [ ] Section coherence review
- [ ] Fact-check against sources

### 2.4 Verification
- [ ] Generated papers compile without errors
- [ ] Citations are valid and formatted correctly
- [ ] Output quality matches research paper standards

---

## PHASE 3 DETAILED CHECKLIST

### 3.1 Data Pipeline
- [ ] ArXiv S3 bulk download scripts
- [ ] LaTeX source extraction (TexSoup)
- [ ] Text cleaning pipeline
- [ ] Deduplication (MinHash LSH)
- [ ] JSONL formatting for training

### 3.2 Continual Pre-Training (CPT)
- [ ] Prepare physics/astro corpus (~250K papers)
- [ ] Configure QLoRA parameters (rank 64, alpha 128)
- [ ] Set up Unsloth for 2x speed
- [ ] Run CPT on Brev A100 (~12 hours)
- [ ] Validate no capability degradation

### 3.3 Supervised Fine-Tuning (SFT)
- [ ] Create research methodology examples (~10K)
- [ ] Create paper writing examples
- [ ] Create search query examples
- [ ] Run SFT on Brev A100 (~4 hours)
- [ ] Evaluate against baseline

### 3.4 Verification
- [ ] Fine-tuned model outperforms base on domain tasks
- [ ] Reasoning ability preserved (MATH-500 check)
- [ ] No hallucination increase

---

## PHASE 4 DETAILED CHECKLIST

### 4.1 Log Processing Pipeline
- [ ] Trajectory logging format
- [ ] Success/failure labeling
- [ ] Critical step identification (ATLAS method)
- [ ] Quality scoring system

### 4.2 Validation Gates
- [ ] Factual accuracy checker
- [ ] Reasoning coherence validator
- [ ] Diversity score calculator
- [ ] Generation depth tracker
- [ ] Synthetic ratio enforcer

### 4.3 Evolution Loop
- [ ] Batch processing (weekly)
- [ ] Training corpus composition (60/30/10 rule)
- [ ] Candidate training pipeline
- [ ] Evaluation arena
- [ ] Staged deployment with rollback

### 4.4 Verification
- [ ] Evolution cycle completes without errors
- [ ] Quality metrics improve or stay stable
- [ ] No model collapse indicators

---

## PHASE 5 DETAILED CHECKLIST

### 5.1 Error Handling
- [ ] Graceful degradation on API failures
- [ ] Automatic retry with backoff
- [ ] Circuit breaker for external services
- [ ] Dead letter queue for failed tasks

### 5.2 Monitoring
- [ ] Prometheus metrics exporter
- [ ] Grafana dashboards (inference, costs, quality)
- [ ] Alerting rules
- [ ] Cost tracking and budgets

### 5.3 Documentation
- [ ] README.md with setup instructions
- [ ] API documentation
- [ ] Architecture decision records
- [ ] Troubleshooting guide

### 5.4 Demo & Testing
- [ ] One-command setup script
- [ ] Example research queries
- [ ] Benchmark suite
- [ ] Load testing

---

## TECH STACK REFERENCE

### Models (Verified 2026-01-20)
- **Brain**: DeepSeek-R1-Distill-Qwen-14B
  - Context: 128K tokens
  - VRAM: ~9GB quantized (Q4_K_M), ~28GB FP16
  - License: MIT
  - Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

- **Workers**: GPT-4o-mini (API)
  - Cost: $0.15/M input, $0.60/M output
  - Best for: Tool calling, structured output

### Python Dependencies (Implemented)

- **openai**: 1.59.9 - AsyncOpenAI for GPT-4o-mini and vLLM
- **tavily-python**: 0.5.0 - Web search API client (async)
- **arxiv**: 2.1.3 - ArXiv API client with rate limiting
- **fastapi**: 0.115.6 - API server
- **httpx**: 0.28.1 - Async HTTP client
- **pydantic**: 2.10.4 - Data validation
- **structlog**: 24.4.0 - Structured logging

### Python Dependencies (Planned)

- **LangGraph**: Agent orchestration
- **vLLM**: Self-hosted inference
- **Unsloth**: 2x faster fine-tuning
- **transformers**: Model loading
- **datasets**: Data processing
- **TexSoup**: LaTeX parsing
- **pylatexenc**: LaTeX to text
- **datasketch**: MinHash deduplication
- **chromadb** or **qdrant-client**: Vector DB

### Frontend Stack (Decided 2026-01-21)

- **Next.js 15**: React framework with App Router
- **TypeScript**: Type safety
- **Tailwind CSS**: Utility-first styling
- **Aceternity UI**: Component library (premium animations, Tailwind + Motion based)
- **Motion (Framer Motion)**: Animations (subtle, purposeful only)
- **Geist or Inter**: Premium system fonts

### Infrastructure (Versions Confirmed 2026-01-21)

- **Docker**: Containerization (multi-stage builds)
- **Redis**: 7.4-alpine - Task queue, caching
- **PostgreSQL**: 16-alpine - Metadata storage
- **vLLM**: 0.13.0 - Brain inference server
- **Prometheus**: Metrics (planned)
- **Grafana**: Ops monitoring dashboards (planned)

### Cloud Resources
- **Brev.dev**: $80 credits
  - Development: L40S @ $1.03/hr
  - Training: A100 80GB @ $1.49/hr
- **Vast.ai**: Backup/long-term
  - A100 80GB @ $0.82-1.27/hr

### Hardware Target (Self-Hosted)
- **RTX 4090**: 24GB VRAM
  - Runs DeepSeek-R1-Distill-14B quantized easily
  - Target: 15-30 tok/s inference

---

## COST TRACKING

### Budget Allocation
| Category | Allocated | Spent | Remaining |
|----------|-----------|-------|-----------|
| Brev Credits | $80 | $0 | $80 |
| API Calls (GPT-4o-mini) | ~$20/month | $0 | - |
| Total Initial | $100 | $0 | $100 |

### Per-Task Cost Targets
- Research paper generation: < $1.00
- Training run (CPT): < $20
- Training run (SFT): < $10
- Monthly operation (self-hosted): < $25

---

## CONTEXT RECOVERY INSTRUCTIONS

If context is cleared or conversation is resumed:

1. **Read this file first** (CLAUDE.md)
2. **Check the Progress Tracker** section above
3. **Read RESEARCH_REPORT.md** for detailed research findings
4. **Check existing code** in the relevant directories
5. **Resume from the last incomplete task**

### Directory Structure

```
research-agent/
├── CLAUDE.md              # This file - rules and progress
├── BUILDPLAN.md           # Implementation plan (follow this)
├── RESEARCH_REPORT.md     # Comprehensive research findings
├── pyproject.toml         # Python dependencies (pinned versions)
├── docker-compose.yml     # Production docker setup
├── docker-compose.dev.yml # Development docker setup
├── .env.example           # Environment variables template
│
├── src/                   # Main application code
│   ├── main.py            # FastAPI entry point
│   ├── config.py          # Configuration management
│   ├── brain/             # Brain service client
│   ├── workers/           # Worker services
│   ├── pipeline/          # LangGraph orchestration
│   ├── output/            # LaTeX generation
│   ├── security/          # Security utilities
│   └── utils/             # Logging, errors
│
├── services/brain/        # vLLM brain service
├── infra/postgres/        # PostgreSQL init scripts
├── scripts/               # Setup and run scripts
├── tests/                 # Test suite
├── outputs/               # Generated papers
└── storage/               # Local file storage
```

### Key Files to Check

- `BUILDPLAN.md`: Implementation phases and deliverables
- `src/main.py`: FastAPI application (health endpoints working)
- `src/config.py`: Configuration with Pydantic settings
- `src/brain/client.py`: BrainClient for vLLM integration
- `src/brain/prompts.py`: Prompt templates for research tasks
- `src/workers/base.py`: BaseWorker and LLMWorker base classes
- `src/workers/search.py`: SearchWorker with Tavily API
- `src/workers/arxiv.py`: ArxivWorker with ArXiv API
- `src/workers/writer.py`: WriterWorker with GPT-4o-mini
- `src/utils/errors.py`: Custom exception hierarchy
- `src/utils/logging.py`: Structured logging with structlog
- `services/brain/`: vLLM Dockerfile and startup script
- `infra/postgres/init.sql`: Database schema
- `scripts/`: setup.sh, run_local.sh, test.sh

---

## NOTES & DECISIONS LOG

### 2026-01-21

- **BUILDPLAN Phase 3 Complete**: Worker Services
  - BaseWorker abstract class with retry logic (exponential backoff), error handling, cost tracking (src/workers/base.py - 216 lines)
  - LLMWorker base class for OpenAI-based workers with async client management
  - SearchWorker: Tavily API integration with score filtering, parallel search, context/QnA modes (src/workers/search.py - 291 lines)
  - ArxivWorker: ArXiv API with 3s rate limiting, category filtering, BibTeX generation, batch retrieval (src/workers/arxiv.py - 347 lines)
  - WriterWorker: GPT-4o-mini content generation for LaTeX sections, abstracts, citations (src/workers/writer.py - 361 lines)
  - CostTracker and WorkerResult for standardized cost/token tracking
  - GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output tokens
  - Tavily pricing: $0.01 per search
  - 27 comprehensive unit tests with mocked API clients (tests/test_workers.py - 490 lines)
  - Added mypy overrides for tavily and arxiv modules in pyproject.toml
  - All 91 tests passing (64 brain + 27 workers), linting and type checks clean

- **BUILDPLAN Phase 2 Complete**: Brain Service
  - BrainClient class with async vLLM integration (src/brain/client.py - 320 lines)
  - Uses OpenAI Python SDK with vLLM's OpenAI-compatible API
  - Streaming support, retry logic with exponential backoff
  - Extracts reasoning from `<think>` tags automatically
  - 12 prompt templates for research tasks (src/brain/prompts.py - 340 lines)
  - NO system prompts (R1-Distill requirement)
  - vLLM Dockerfile with DeepSeek-R1-Distill-14B (services/brain/)
  - Comprehensive test suite with 26 tests (tests/test_brain.py)
  - All 64 tests passing, linting and type checks clean

- **BUILDPLAN Phase 1 Complete**: Foundation & Infrastructure
  - FastAPI application with health endpoints (main.py - 297 lines)
  - Configuration management with Pydantic v2 (config.py - 98 lines)
  - Structured logging with structlog (utils/logging.py - 122 lines)
  - Custom exception hierarchy (utils/errors.py - 190 lines)
  - Docker Compose with Redis 7.4, PostgreSQL 16, vLLM 0.13.0
  - PostgreSQL schema for research queries, logs, papers, sources
  - Setup scripts: setup.sh, run_local.sh, test.sh
  - Test suite with pytest fixtures (conftest.py, test_*.py)
  - All dependencies pinned in pyproject.toml

- **Phase 2 Research Complete**: vLLM and DeepSeek-R1 Best Practices
  - vLLM 0.13.0 exposes OpenAI-compatible API at `/v1` endpoint
  - Use official OpenAI Python client with `base_url` set to vLLM server
  - vLLM-specific params via `extra_body` (e.g., `top_k`)
  - Streaming supported via standard OpenAI client pattern
  - AsyncOpenAI for async operations, aiohttp backend optional
  - DeepSeek-R1 settings: temperature 0.5-0.7 (0.6 optimal), top_p 0.95
  - NO system prompts for R1-Distill - put all instructions in user message
  - NO few-shot prompting - degrades performance, trust zero-shot
  - Math problems: add "Please reason step by step, put final answer in \\boxed{}"
  - Model uses `<think>` tags for reasoning output
  - Force thinking with `<think>\n` if model skips reasoning
  - Sources: vLLM docs, DeepSeek GitHub, Together.ai docs

- **Phase 1 Code Quality Review**
  - Fixed ruff linting issues (unused imports, argument naming)
  - Fixed mypy type annotations (lifespan, middleware, logger)
  - Added per-file-ignores for intentional patterns (S104 for Docker binding)
  - All linting and type checks now pass

- **UI Stack Decision**: Next.js 15 + Aceternity UI + Motion (Framer Motion)
  - Design philosophy: Intentionally minimal, premium typography, fast responsive
  - Animations: Subtle micro-interactions only, no flashy effects
  - Motion only - GSAP banned for simplicity
  - Aceternity UI chosen over shadcn/ui for premium look
  - Grafana for ops monitoring only, not main UI
  - Streamlit/Gradio/shadcn permanently banned

### 2026-01-20
- Project initiated with comprehensive research phase
- Created RESEARCH_REPORT.md with 2000+ lines of findings
- **Model Selection Decision**: DeepSeek-R1-Distill-Qwen-14B chosen over:
  - Mistral Small 3.2 (24B): Larger, requires quantization, weaker math
  - Qwen3-14B: Lower math scores (62% vs 93.9% MATH-500)
  - Falcon H1R-7B: Too new, less tested
- **Reasoning**: 93.9% MATH-500, 128K context, fits RTX 4090 easily (~9GB Q4)
- **Domain focus**: AI/ML, Quantum Physics, Astrophysics - all math-heavy
- **Architecture**: Brain (self-hosted) + Workers (API) pattern confirmed
- Created CLAUDE.md with project rules and progress tracking
- **Scope Decision**: MVP without long-term memory and self-evolution
  - Deferred: Vector DB, embeddings, trajectory logging, retraining
  - Focus: Core agent with 128K short-term memory, secure, production-quality
- Added security rules (non-negotiable) and development approach rules
- Created comprehensive BUILDPLAN.md with 6 phases

---

## QUESTIONS FOR USER (If Needed)

Before asking, check if the answer is in RESEARCH_REPORT.md or this file.

Current open questions:
- None

---

## REMINDERS

- **QUALITY OVER COST** - Don't cheap out. User can't afford wasted time/money on a bad project
- **ASK, DON'T GUESS** - When uncertain about requirements, ask the user
- **DON'T SPEED RUN** - Take time to do it right, research before building
- **SECURITY IS NON-NEGOTIABLE** - Build like someone will try to hack it
- This is a RESEARCH PROJECT - correctness over speed
- The goal is publication-ready papers, not quick summaries
- Every component should handle the 128K context properly
- MVP scope: Short-term memory only, no self-evolution, no fine-tuning yet
- Commit and push after major development milestones
- Never mention Claude/Anthropic in commits
- No emojis anywhere in the codebase
