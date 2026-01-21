"""Worker services for the research agent.

Workers handle specialized tasks using external APIs:
- SearchWorker: Web search via Tavily
- ArxivWorker: Academic paper search via ArXiv
- WriterWorker: Content generation via GPT-4o-mini
"""

from src.workers.arxiv import (
    ArxivPaper,
    ArxivSearchResponse,
    ArxivWorker,
    arxiv_search,
)
from src.workers.base import (
    BaseWorker,
    CostTracker,
    LLMWorker,
    WorkerResult,
)
from src.workers.search import (
    SearchResponse,
    SearchResult,
    SearchWorker,
    web_search,
)
from src.workers.writer import (
    WriterOutput,
    WriterWorker,
    write_content,
)

__all__ = [
    # ArXiv worker
    "ArxivPaper",
    "ArxivSearchResponse",
    "ArxivWorker",
    # Base classes
    "BaseWorker",
    "CostTracker",
    "LLMWorker",
    "SearchResponse",
    # Search worker
    "SearchResult",
    "SearchWorker",
    "WorkerResult",
    # Writer worker
    "WriterOutput",
    "WriterWorker",
    "arxiv_search",
    "web_search",
    "write_content",
]
