"""Web search worker using Tavily API.

Provides web search capabilities for the research agent.
Uses AsyncTavilyClient for concurrent searches.

Best practices from Tavily docs:
- Keep queries under 400 characters
- Use score filtering (>0.7 for high relevance)
- Break complex research into focused sub-queries
- Use include_domains for domain-specific searches
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from tavily import AsyncTavilyClient, TavilyClient

from src.utils.errors import SearchError
from src.workers.base import BaseWorker, WorkerResult


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    content: str
    score: float
    raw_content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "raw_content": self.raw_content,
        }


@dataclass
class SearchResponse:
    """Response from a search operation."""

    query: str
    results: list[SearchResult]
    answer: str | None = None  # From Tavily's QnA mode


class SearchWorker(BaseWorker):
    """Web search worker using Tavily API.

    Supports:
    - Basic search with relevance scoring
    - Context search for RAG applications
    - QnA search for direct answers
    - Domain filtering
    - Async parallel searches
    """

    # Tavily doesn't charge per token, but per search
    # Free tier: 1000 searches/month
    # Pro tier: $0.01 per search (approx)
    COST_PER_SEARCH = 0.01

    def __init__(self) -> None:
        """Initialize the search worker."""
        super().__init__()
        self._async_client: AsyncTavilyClient | None = None
        self._sync_client: TavilyClient | None = None

    @property
    def worker_type(self) -> str:
        """Return the worker type identifier."""
        return "search"

    def _get_async_client(self) -> AsyncTavilyClient:
        """Get or create the async Tavily client."""
        if self._async_client is None:
            api_key = self._settings.tavily_api_key.get_secret_value()
            self._async_client = AsyncTavilyClient(api_key=api_key)
        return self._async_client

    def _get_sync_client(self) -> TavilyClient:
        """Get or create the sync Tavily client."""
        if self._sync_client is None:
            api_key = self._settings.tavily_api_key.get_secret_value()
            self._sync_client = TavilyClient(api_key=api_key)
        return self._sync_client

    async def _execute(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: Literal["basic", "advanced"] = "basic",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_raw_content: bool = False,
        topic: Literal["general", "news"] = "general",
        days: int | None = None,
        min_score: float = 0.0,
    ) -> WorkerResult:
        """Execute a web search.

        Args:
            query: Search query (keep under 400 chars).
            max_results: Maximum results to return (1-20).
            search_depth: "basic" for speed, "advanced" for relevance.
            include_domains: Only search these domains.
            exclude_domains: Exclude these domains.
            include_raw_content: Include full page content.
            topic: "general" or "news".
            days: For news, limit to last N days.
            min_score: Filter results below this relevance score.

        Returns:
            WorkerResult with SearchResponse data.
        """
        # Validate query length
        if len(query) > 400:
            self._logger.warning(
                "Query exceeds 400 chars, may have reduced relevance",
                query_length=len(query),
            )

        try:
            client = self._get_async_client()

            # Build search parameters
            search_params: dict[str, Any] = {
                "query": query,
                "max_results": min(max_results, 20),
                "search_depth": search_depth,
                "include_raw_content": include_raw_content,
                "topic": topic,
            }

            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            if days and topic == "news":
                search_params["days"] = days

            # Execute search
            response = await client.search(**search_params)

            # Parse results
            results: list[SearchResult] = []
            for item in response.get("results", []):
                score = item.get("score", 0.0)

                # Apply score filtering
                if score < min_score:
                    continue

                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        score=score,
                        raw_content=item.get("raw_content"),
                    )
                )

            search_response = SearchResponse(
                query=query,
                results=results,
                answer=response.get("answer"),
            )

            return WorkerResult(
                success=True,
                data=search_response,
                cost_usd=self.COST_PER_SEARCH,
                metadata={"result_count": len(results)},
            )

        except Exception as e:
            self._logger.error("Search failed", query=query, error=str(e))
            raise SearchError(f"Web search failed: {e}") from e

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> SearchResponse:
        """Convenience method for simple searches.

        Returns SearchResponse directly instead of WorkerResult.
        """
        result = await self.execute(query, **kwargs)
        if not result.success:
            raise SearchError(result.error or "Search failed")
        return result.data  # type: ignore[no-any-return]

    async def search_multiple(
        self,
        queries: list[str],
        **kwargs: Any,
    ) -> list[SearchResponse]:
        """Execute multiple searches in parallel.

        Args:
            queries: List of search queries.
            **kwargs: Common search parameters for all queries.

        Returns:
            List of SearchResponse objects.
        """
        tasks = [self.search(q, **kwargs) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        responses: list[SearchResponse] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                self._logger.error(
                    "Parallel search failed",
                    query=queries[i],
                    error=str(result),
                )
            elif isinstance(result, SearchResponse):
                responses.append(result)

        return responses

    async def get_search_context(
        self,
        query: str,
        *,
        max_results: int = 5,
        max_tokens: int = 4000,
    ) -> str:
        """Get search context optimized for RAG applications.

        Returns a formatted string of search results suitable for
        injecting into an LLM context.
        """
        try:
            client = self._get_async_client()
            context: str = await client.get_search_context(
                query=query,
                max_results=max_results,
                max_tokens=max_tokens,
            )
            return context
        except Exception as e:
            self._logger.error("Context search failed", query=query, error=str(e))
            raise SearchError(f"Context search failed: {e}") from e

    async def qna_search(
        self,
        question: str,
        *,
        search_depth: Literal["basic", "advanced"] = "advanced",
    ) -> str:
        """Get a direct answer to a question.

        Uses Tavily's QnA mode for concise, accurate answers.
        """
        try:
            client = self._get_async_client()
            answer: str = await client.qna_search(
                query=question,
                search_depth=search_depth,
            )
            return answer
        except Exception as e:
            self._logger.error("QnA search failed", question=question, error=str(e))
            raise SearchError(f"QnA search failed: {e}") from e


# Convenience function for one-off searches
async def web_search(
    query: str,
    max_results: int = 5,
    **kwargs: Any,
) -> SearchResponse:
    """Execute a one-off web search.

    Creates a temporary worker for simple use cases.
    For repeated searches, use SearchWorker directly.
    """
    worker = SearchWorker()
    return await worker.search(query, max_results=max_results, **kwargs)
