"""ArXiv paper search worker.

Provides ArXiv search and paper retrieval capabilities.
Uses the arxiv.py library with proper rate limiting.

Best practices from ArXiv API docs:
- No more than one request every 3 seconds
- Limit results per request (1000 max recommended)
- Use Client for connection pooling and rate limiting
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import arxiv

from src.utils.errors import ArxivError
from src.workers.base import BaseWorker, WorkerResult


@dataclass
class ArxivPaper:
    """Represents an ArXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime
    updated: datetime
    pdf_url: str
    entry_url: str
    comment: str | None = None
    journal_ref: str | None = None
    doi: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "categories": self.categories,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "pdf_url": self.pdf_url,
            "entry_url": self.entry_url,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
        }

    def to_bibtex(self) -> str:
        """Generate a BibTeX entry for this paper."""
        # Create citation key: FirstAuthorLastNameYear
        first_author = self.authors[0] if self.authors else "Unknown"
        last_name = first_author.split()[-1].lower()
        year = self.published.year
        key = f"{last_name}{year}"

        # Clean title for BibTeX
        title = self.title.replace("{", "").replace("}", "")

        # Format authors
        authors_str = " and ".join(self.authors)

        bibtex = f"""@article{{{key},
    title = {{{title}}},
    author = {{{authors_str}}},
    year = {{{year}}},
    eprint = {{{self.arxiv_id}}},
    archivePrefix = {{arXiv}},
    primaryClass = {{{self.categories[0] if self.categories else "cs.AI"}}},
    url = {{{self.entry_url}}}"""

        if self.doi:
            bibtex += f',\n    doi = {{{self.doi}}}'
        if self.journal_ref:
            bibtex += f',\n    journal = {{{self.journal_ref}}}'

        bibtex += "\n}"
        return bibtex


@dataclass
class ArxivSearchResponse:
    """Response from an ArXiv search."""

    query: str
    papers: list[ArxivPaper]
    total_results: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ArxivWorker(BaseWorker):
    """ArXiv paper search worker.

    Supports:
    - Keyword search
    - Author search
    - Category filtering
    - Date range filtering
    - Paper retrieval by ID
    """

    # ArXiv API is free, no cost
    COST_PER_SEARCH = 0.0

    # Rate limit: 3 seconds between requests (ArXiv Terms of Use)
    RATE_LIMIT_SECONDS = 3.0

    def __init__(self) -> None:
        """Initialize the ArXiv worker."""
        super().__init__()
        # Create a client with proper rate limiting
        self._client = arxiv.Client(
            page_size=100,
            delay_seconds=self.RATE_LIMIT_SECONDS,
            num_retries=3,
        )

    @property
    def worker_type(self) -> str:
        """Return the worker type identifier."""
        return "arxiv"

    def _result_to_paper(self, result: arxiv.Result) -> ArxivPaper:
        """Convert an arxiv.Result to ArxivPaper."""
        return ArxivPaper(
            arxiv_id=result.get_short_id(),
            title=result.title,
            authors=[str(author) for author in result.authors],
            abstract=result.summary,
            categories=result.categories,
            published=result.published,
            updated=result.updated,
            pdf_url=result.pdf_url or "",
            entry_url=result.entry_id,
            comment=result.comment,
            journal_ref=result.journal_ref,
            doi=result.doi,
        )

    async def _execute(
        self,
        query: str,
        *,
        max_results: int = 10,
        sort_by: Literal["relevance", "submitted", "updated"] = "relevance",
        sort_order: Literal["ascending", "descending"] = "descending",
        categories: list[str] | None = None,
    ) -> WorkerResult:
        """Execute an ArXiv search.

        Args:
            query: Search query. Supports ArXiv query syntax:
                   - ti:word (title contains word)
                   - au:name (author name)
                   - abs:word (abstract contains word)
                   - cat:cs.AI (category)
                   - all:word (all fields)
            max_results: Maximum papers to return (1-1000).
            sort_by: Sort criterion.
            sort_order: Sort direction.
            categories: Filter to specific ArXiv categories.

        Returns:
            WorkerResult with ArxivSearchResponse data.
        """
        try:
            # Build the full query with category filter
            full_query = query
            if categories:
                cat_filter = " OR ".join(f"cat:{cat}" for cat in categories)
                full_query = f"({query}) AND ({cat_filter})"

            # Map sort options
            sort_criterion_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "submitted": arxiv.SortCriterion.SubmittedDate,
                "updated": arxiv.SortCriterion.LastUpdatedDate,
            }
            sort_order_map = {
                "ascending": arxiv.SortOrder.Ascending,
                "descending": arxiv.SortOrder.Descending,
            }

            # Create search
            search = arxiv.Search(
                query=full_query,
                max_results=min(max_results, 1000),
                sort_by=sort_criterion_map[sort_by],
                sort_order=sort_order_map[sort_order],
            )

            # Execute search (blocking, but rate-limited)
            # Note: arxiv.py doesn't have async support, so we run in executor
            import asyncio

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(self._client.results(search)),
            )

            # Convert to ArxivPaper objects
            papers = [self._result_to_paper(r) for r in results]

            response = ArxivSearchResponse(
                query=query,
                papers=papers,
                total_results=len(papers),
                metadata={
                    "full_query": full_query,
                    "categories": categories,
                },
            )

            return WorkerResult(
                success=True,
                data=response,
                cost_usd=self.COST_PER_SEARCH,
                metadata={"result_count": len(papers)},
            )

        except Exception as e:
            self._logger.error("ArXiv search failed", query=query, error=str(e))
            raise ArxivError(f"ArXiv search failed: {e}") from e

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> ArxivSearchResponse:
        """Convenience method for searches.

        Returns ArxivSearchResponse directly instead of WorkerResult.
        """
        result = await self.execute(query, **kwargs)
        if not result.success:
            raise ArxivError(result.error or "ArXiv search failed")
        return result.data  # type: ignore[no-any-return]

    async def search_by_author(
        self,
        author: str,
        max_results: int = 10,
    ) -> ArxivSearchResponse:
        """Search papers by author name."""
        query = f'au:"{author}"'
        return await self.search(query, max_results=max_results)

    async def search_by_category(
        self,
        category: str,
        query: str = "",
        max_results: int = 10,
    ) -> ArxivSearchResponse:
        """Search papers in a specific category.

        Common categories:
        - cs.AI: Artificial Intelligence
        - cs.LG: Machine Learning
        - quant-ph: Quantum Physics
        - astro-ph: Astrophysics
        - physics: General Physics
        """
        full_query = f"({query}) AND cat:{category}" if query else f"cat:{category}"
        return await self.search(full_query, max_results=max_results)

    async def get_paper(self, arxiv_id: str) -> ArxivPaper | None:
        """Get a specific paper by ArXiv ID.

        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345" or "2301.12345v2")

        Returns:
            ArxivPaper if found, None otherwise.
        """
        try:
            # Clean the ID
            clean_id = arxiv_id.replace("arXiv:", "").strip()

            search = arxiv.Search(id_list=[clean_id])

            import asyncio

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(self._client.results(search)),
            )

            if results:
                return self._result_to_paper(results[0])
            return None

        except Exception as e:
            self._logger.error("Paper retrieval failed", arxiv_id=arxiv_id, error=str(e))
            raise ArxivError(f"Failed to retrieve paper {arxiv_id}: {e}") from e

    async def get_papers(self, arxiv_ids: list[str]) -> list[ArxivPaper]:
        """Get multiple papers by ArXiv IDs.

        Args:
            arxiv_ids: List of ArXiv paper IDs.

        Returns:
            List of ArxivPaper objects.
        """
        # Clean IDs
        clean_ids = [aid.replace("arXiv:", "").strip() for aid in arxiv_ids]

        try:
            search = arxiv.Search(id_list=clean_ids)

            import asyncio

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(self._client.results(search)),
            )

            return [self._result_to_paper(r) for r in results]

        except Exception as e:
            self._logger.error("Batch paper retrieval failed", error=str(e))
            raise ArxivError(f"Failed to retrieve papers: {e}") from e


# Convenience function for one-off searches
async def arxiv_search(
    query: str,
    max_results: int = 10,
    **kwargs: Any,
) -> ArxivSearchResponse:
    """Execute a one-off ArXiv search.

    Creates a temporary worker for simple use cases.
    For repeated searches, use ArxivWorker directly.
    """
    worker = ArxivWorker()
    return await worker.search(query, max_results=max_results, **kwargs)
