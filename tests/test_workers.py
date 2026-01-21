"""Tests for worker services."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.workers import (
    ArxivPaper,
    ArxivSearchResponse,
    ArxivWorker,
    BaseWorker,
    CostTracker,
    LLMWorker,
    SearchResponse,
    SearchResult,
    SearchWorker,
    WorkerResult,
    WriterOutput,
    WriterWorker,
)


class TestCostTracker:
    """Test cases for CostTracker."""

    def test_initial_state(self):
        """Test initial tracker state."""
        tracker = CostTracker()

        assert tracker.total_cost_usd == 0.0
        assert tracker.total_tokens == 0
        assert tracker.total_requests == 0
        assert tracker.successful_requests == 0
        assert tracker.failed_requests == 0

    def test_record_successful_request(self):
        """Test recording a successful request."""
        tracker = CostTracker()
        tracker.record_request(cost_usd=0.01, tokens=100, success=True)

        assert tracker.total_cost_usd == 0.01
        assert tracker.total_tokens == 100
        assert tracker.total_requests == 1
        assert tracker.successful_requests == 1
        assert tracker.failed_requests == 0

    def test_record_failed_request(self):
        """Test recording a failed request."""
        tracker = CostTracker()
        tracker.record_request(success=False)

        assert tracker.total_requests == 1
        assert tracker.failed_requests == 1
        assert tracker.successful_requests == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        tracker = CostTracker()
        tracker.record_request(cost_usd=0.015, tokens=150, success=True)

        result = tracker.to_dict()

        assert result["total_cost_usd"] == 0.015
        assert result["total_tokens"] == 150
        assert result["total_requests"] == 1


class TestWorkerResult:
    """Test cases for WorkerResult."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = WorkerResult(
            success=True,
            data={"key": "value"},
            cost_usd=0.01,
            tokens_used=100,
        )

        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.cost_usd == 0.01

    def test_failed_result(self):
        """Test creating a failed result."""
        result = WorkerResult(
            success=False,
            data=None,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.data is None
        assert result.error == "Something went wrong"


class TestSearchResult:
    """Test cases for SearchResult."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            title="Test Article",
            url="https://example.com",
            content="Test content",
            score=0.95,
        )

        assert result.title == "Test Article"
        assert result.score == 0.95

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SearchResult(
            title="Test",
            url="https://test.com",
            content="Content",
            score=0.8,
            raw_content="Full content",
        )

        d = result.to_dict()
        assert d["title"] == "Test"
        assert d["raw_content"] == "Full content"


class TestSearchResponse:
    """Test cases for SearchResponse."""

    def test_search_response_creation(self):
        """Test creating a search response."""
        results = [
            SearchResult(title="A", url="http://a.com", content="a", score=0.9),
            SearchResult(title="B", url="http://b.com", content="b", score=0.8),
        ]

        response = SearchResponse(
            query="test query",
            results=results,
            answer="Direct answer",
        )

        assert response.query == "test query"
        assert len(response.results) == 2
        assert response.answer == "Direct answer"


class TestArxivPaper:
    """Test cases for ArxivPaper."""

    @pytest.fixture
    def sample_paper(self) -> ArxivPaper:
        """Create a sample paper."""
        return ArxivPaper(
            arxiv_id="2401.12345",
            title="Test Paper on Machine Learning",
            authors=["John Smith", "Jane Doe"],
            abstract="This is a test abstract.",
            categories=["cs.LG", "cs.AI"],
            published=datetime(2024, 1, 15),
            updated=datetime(2024, 1, 20),
            pdf_url="https://arxiv.org/pdf/2401.12345.pdf",
            entry_url="https://arxiv.org/abs/2401.12345",
            doi="10.1234/test",
        )

    def test_paper_creation(self, sample_paper: ArxivPaper):
        """Test creating a paper."""
        assert sample_paper.arxiv_id == "2401.12345"
        assert len(sample_paper.authors) == 2
        assert "cs.LG" in sample_paper.categories

    def test_to_dict(self, sample_paper: ArxivPaper):
        """Test conversion to dictionary."""
        d = sample_paper.to_dict()

        assert d["arxiv_id"] == "2401.12345"
        assert d["title"] == "Test Paper on Machine Learning"
        assert "2024-01-15" in d["published"]

    def test_to_bibtex(self, sample_paper: ArxivPaper):
        """Test BibTeX generation."""
        bibtex = sample_paper.to_bibtex()

        # First author is John Smith, so key should be smith2024
        assert "@article{smith2024" in bibtex
        assert "Test Paper on Machine Learning" in bibtex
        assert "eprint = {2401.12345}" in bibtex
        assert "doi = {10.1234/test}" in bibtex


class TestWriterOutput:
    """Test cases for WriterOutput."""

    def test_writer_output_creation(self):
        """Test creating writer output."""
        output = WriterOutput(
            content="\\section{Introduction}",
            format="latex",
            tokens_used=100,
            metadata={"model": "gpt-4o-mini"},
        )

        assert output.format == "latex"
        assert output.tokens_used == 100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = WriterOutput(
            content="Test",
            format="text",
            tokens_used=50,
            metadata={},
        )

        d = output.to_dict()
        assert d["format"] == "text"
        assert d["tokens_used"] == 50


class TestSearchWorker:
    """Test cases for SearchWorker."""

    @pytest.fixture
    def mock_tavily_response(self) -> dict:
        """Create a mock Tavily API response."""
        return {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com",
                    "content": "Test content",
                    "score": 0.95,
                },
            ],
            "answer": "Direct answer",
        }

    @pytest.mark.asyncio
    async def test_search_execution(self, mock_tavily_response: dict):
        """Test search execution."""
        with patch("src.workers.search.AsyncTavilyClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.search = AsyncMock(return_value=mock_tavily_response)
            mock_client_class.return_value = mock_client

            worker = SearchWorker()
            result = await worker.execute("test query")

            assert result.success is True
            assert isinstance(result.data, SearchResponse)
            assert len(result.data.results) == 1
            assert result.data.results[0].title == "Test Result"

    @pytest.mark.asyncio
    async def test_search_with_score_filtering(self, mock_tavily_response: dict):
        """Test search with score filtering."""
        # Add a low-score result
        mock_tavily_response["results"].append(
            {
                "title": "Low Score",
                "url": "https://low.com",
                "content": "Low",
                "score": 0.3,
            }
        )

        with patch("src.workers.search.AsyncTavilyClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.search = AsyncMock(return_value=mock_tavily_response)
            mock_client_class.return_value = mock_client

            worker = SearchWorker()
            result = await worker.execute("test", min_score=0.5)

            # Should filter out the low-score result
            assert len(result.data.results) == 1
            assert result.data.results[0].score >= 0.5

    @pytest.mark.asyncio
    async def test_search_cost_tracking(self, mock_tavily_response: dict):
        """Test that search tracks costs."""
        with patch("src.workers.search.AsyncTavilyClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.search = AsyncMock(return_value=mock_tavily_response)
            mock_client_class.return_value = mock_client

            worker = SearchWorker()
            await worker.execute("test")

            costs = worker.get_cost_summary()
            assert costs["total_cost_usd"] == SearchWorker.COST_PER_SEARCH
            assert costs["successful_requests"] == 1


class TestArxivWorker:
    """Test cases for ArxivWorker."""

    @pytest.fixture
    def mock_arxiv_result(self) -> MagicMock:
        """Create a mock arxiv.Result."""
        result = MagicMock()
        result.get_short_id.return_value = "2401.12345"
        result.title = "Test Paper"
        result.authors = [MagicMock(__str__=lambda s: "Author One")]
        result.summary = "Test abstract"
        result.categories = ["cs.AI"]
        result.published = datetime(2024, 1, 15)
        result.updated = datetime(2024, 1, 20)
        result.pdf_url = "https://arxiv.org/pdf/2401.12345.pdf"
        result.entry_id = "https://arxiv.org/abs/2401.12345"
        result.comment = None
        result.journal_ref = None
        result.doi = None
        return result

    @pytest.mark.asyncio
    async def test_arxiv_search(self, mock_arxiv_result: MagicMock):
        """Test ArXiv search execution."""
        with patch("src.workers.arxiv.arxiv.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.results = MagicMock(return_value=iter([mock_arxiv_result]))
            mock_client_class.return_value = mock_client

            worker = ArxivWorker()
            worker._client = mock_client

            # Patch the executor to run synchronously
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    return_value=[mock_arxiv_result]
                )

                result = await worker.execute("machine learning")

                assert result.success is True
                assert isinstance(result.data, ArxivSearchResponse)
                assert len(result.data.papers) == 1
                assert result.data.papers[0].arxiv_id == "2401.12345"

    def test_worker_type(self):
        """Test worker type identifier."""
        worker = ArxivWorker()
        assert worker.worker_type == "arxiv"


class TestWriterWorker:
    """Test cases for WriterWorker."""

    @pytest.fixture
    def mock_openai_response(self) -> MagicMock:
        """Create a mock OpenAI response."""
        mock_choice = MagicMock()
        mock_choice.message.content = "\\section{Introduction}\nTest content"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 100

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        return mock_response

    @pytest.mark.asyncio
    async def test_writer_execution(self, mock_openai_response: MagicMock):
        """Test writer execution."""
        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            worker = WriterWorker()
            result = await worker.execute("Write an introduction")

            assert result.success is True
            assert isinstance(result.data, WriterOutput)
            assert result.data.format == "latex"
            assert "Introduction" in result.data.content

            await worker.close()

    @pytest.mark.asyncio
    async def test_writer_cost_calculation(self, mock_openai_response: MagicMock):
        """Test writer cost calculation."""
        with patch("openai.AsyncOpenAI") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            mock_client.close = AsyncMock()
            mock_client_class.return_value = mock_client

            worker = WriterWorker()
            result = await worker.execute("Test")

            # Cost should be calculated
            assert result.cost_usd > 0
            assert result.tokens_used == 150  # 50 + 100

            await worker.close()

    def test_worker_type(self):
        """Test worker type identifier."""
        worker = WriterWorker()
        assert worker.worker_type == "writer"

    def test_llm_cost_calculation(self):
        """Test LLM cost calculation."""
        worker = WriterWorker()

        # 1000 input, 1000 output
        cost = worker.calculate_cost(1000, 1000)

        # $0.15/1M input + $0.60/1M output = $0.00015 + $0.0006 per 1K
        expected = 0.00015 + 0.0006
        assert abs(cost - expected) < 0.0001


class TestBaseWorkerRetry:
    """Test retry logic in BaseWorker."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that worker retries on failure."""

        class TestWorker(BaseWorker):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            @property
            def worker_type(self) -> str:
                return "test"

            async def _execute(self) -> WorkerResult:
                self.call_count += 1
                if self.call_count < 3:
                    raise ValueError("Temporary failure")
                return WorkerResult(success=True, data="success")

        worker = TestWorker()
        result = await worker.execute(max_retries=3)

        assert result.success is True
        assert worker.call_count == 3  # Failed 2x, succeeded on 3rd

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test that worker gives up after max retries."""

        class FailingWorker(BaseWorker):
            @property
            def worker_type(self) -> str:
                return "failing"

            async def _execute(self) -> WorkerResult:
                raise ValueError("Always fails")

        worker = FailingWorker()
        result = await worker.execute(max_retries=2)

        assert result.success is False
        assert "Always fails" in result.error


class TestLLMWorkerCostCalculation:
    """Test LLM cost calculations."""

    def test_gpt4o_mini_pricing(self):
        """Test GPT-4o-mini pricing constants."""
        assert LLMWorker.INPUT_COST_PER_1K == 0.00015
        assert LLMWorker.OUTPUT_COST_PER_1K == 0.0006

    def test_cost_calculation_large_tokens(self):
        """Test cost calculation for large token counts."""
        worker = WriterWorker()

        # 100K input, 10K output
        cost = worker.calculate_cost(100_000, 10_000)

        # (100000/1000)*0.00015 + (10000/1000)*0.0006 = 0.015 + 0.006 = 0.021
        expected = 0.021
        assert abs(cost - expected) < 0.0001
