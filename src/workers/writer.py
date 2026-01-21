"""Writer worker using GPT-4o-mini for LaTeX content generation.

Provides writing assistance for research papers including:
- Section drafting
- LaTeX formatting
- Citation formatting
- BibTeX generation

Uses GPT-4o-mini for cost-effective content generation.
"""

from dataclasses import dataclass
from typing import Any, Literal

from src.utils.errors import WorkerError
from src.workers.base import LLMWorker, WorkerResult


@dataclass
class WriterOutput:
    """Output from the writer worker."""

    content: str
    format: Literal["latex", "markdown", "text", "bibtex"]
    tokens_used: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "format": self.format,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
        }


class WriterWorker(LLMWorker):
    """Writer worker using GPT-4o-mini.

    Handles various writing tasks for research papers:
    - Section content generation
    - Abstract writing
    - Conclusion writing
    - Citation formatting
    - BibTeX entry generation
    """

    MODEL = "gpt-4o-mini"

    @property
    def worker_type(self) -> str:
        """Return the worker type identifier."""
        return "writer"

    async def _execute(
        self,
        prompt: str,
        *,
        output_format: Literal["latex", "markdown", "text", "bibtex"] = "latex",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> WorkerResult:
        """Execute a writing task.

        Args:
            prompt: The writing instruction/prompt.
            output_format: Expected output format.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            system_prompt: Optional system prompt override.

        Returns:
            WorkerResult with WriterOutput data.
        """
        try:
            client = await self._get_client()

            # Default system prompt for academic writing
            if system_prompt is None:
                system_prompt = self._get_default_system_prompt(output_format)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            response = await client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens

            # Calculate cost
            cost = self.calculate_cost(input_tokens, output_tokens)

            output = WriterOutput(
                content=content,
                format=output_format,
                tokens_used=total_tokens,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model": self.MODEL,
                },
            )

            return WorkerResult(
                success=True,
                data=output,
                cost_usd=cost,
                tokens_used=total_tokens,
            )

        except Exception as e:
            self._logger.error("Writing task failed", error=str(e))
            raise WorkerError(f"Writing task failed: {e}") from e

    def _get_default_system_prompt(
        self,
        output_format: Literal["latex", "markdown", "text", "bibtex"],
    ) -> str:
        """Get the default system prompt for the output format."""
        base = (
            "You are an expert academic writer specializing in research papers. "
            "Your writing is clear, precise, and follows academic conventions. "
            "You cite sources properly and maintain technical accuracy."
        )

        format_instructions = {
            "latex": (
                "\n\nOutput format: LaTeX\n"
                "- Use proper LaTeX formatting (sections, equations, etc.)\n"
                "- Use \\cite{key} for citations\n"
                "- Format equations using $...$ or \\[...\\]\n"
                "- Do not include document class or preamble"
            ),
            "markdown": (
                "\n\nOutput format: Markdown\n"
                "- Use proper Markdown formatting\n"
                "- Use [Author, Year] for citations\n"
                "- Use LaTeX syntax for equations: $...$"
            ),
            "text": (
                "\n\nOutput format: Plain text\n"
                "- Write in clear, readable prose\n"
                "- Use (Author, Year) for citations"
            ),
            "bibtex": (
                "\n\nOutput format: BibTeX\n"
                "- Generate valid BibTeX entries\n"
                "- Use appropriate entry types (@article, @book, etc.)\n"
                "- Create sensible citation keys (authorYear format)"
            ),
        }

        return base + format_instructions[output_format]

    async def write_section(
        self,
        topic: str,
        section_name: str,
        outline: str,
        sources: str,
        **kwargs: Any,
    ) -> WriterOutput:
        """Write a paper section.

        Args:
            topic: The paper topic.
            section_name: Name of the section (e.g., "Introduction").
            outline: Outline or key points to cover.
            sources: Relevant source information to cite.
            **kwargs: Additional arguments passed to execute.

        Returns:
            WriterOutput with LaTeX content.
        """
        prompt = f"""Write the "{section_name}" section for a research paper on the following topic:

TOPIC: {topic}

OUTLINE:
{outline}

RELEVANT SOURCES TO CITE:
{sources}

Write a well-structured section with proper citations. Be thorough but concise."""

        result = await self.execute(prompt, output_format="latex", **kwargs)
        if not result.success:
            raise WorkerError(result.error or "Section writing failed")
        return result.data  # type: ignore[no-any-return]

    async def write_abstract(
        self,
        title: str,
        content_summary: str,
        **kwargs: Any,
    ) -> WriterOutput:
        """Write a paper abstract.

        Args:
            title: Paper title.
            content_summary: Summary of paper content/findings.
            **kwargs: Additional arguments.

        Returns:
            WriterOutput with abstract text.
        """
        prompt = f"""Write an abstract for the following research paper:

TITLE: {title}

CONTENT SUMMARY:
{content_summary}

Write a concise abstract (150-250 words) that:
1. States the research question/problem
2. Briefly describes the methodology
3. Summarizes key findings
4. States main conclusions and implications

The abstract should stand alone and be understandable without the full paper."""

        kwargs.setdefault("max_tokens", 500)
        result = await self.execute(prompt, output_format="text", **kwargs)
        if not result.success:
            raise WorkerError(result.error or "Abstract writing failed")
        return result.data  # type: ignore[no-any-return]

    async def format_citation(
        self,
        source_info: dict[str, Any],
        **kwargs: Any,
    ) -> WriterOutput:
        """Format a source as a BibTeX citation.

        Args:
            source_info: Dictionary with source information:
                - title: Source title
                - authors: Author names
                - url: URL
                - date: Publication date
                - source_type: Type of source
            **kwargs: Additional arguments.

        Returns:
            WriterOutput with BibTeX entry.
        """
        prompt = f"""Generate a BibTeX entry for the following source:

Title: {source_info.get('title', 'Unknown')}
Authors: {source_info.get('authors', 'Unknown')}
URL: {source_info.get('url', '')}
Publication Date: {source_info.get('date', 'Unknown')}
Source Type: {source_info.get('source_type', 'misc')}

Additional metadata:
{source_info.get('metadata', {})}

Generate a properly formatted BibTeX entry with an appropriate entry type.
Create a sensible citation key in AuthorYear format.
Output only the BibTeX entry, nothing else."""

        kwargs.setdefault("max_tokens", 300)
        kwargs.setdefault("temperature", 0.3)
        result = await self.execute(prompt, output_format="bibtex", **kwargs)
        if not result.success:
            raise WorkerError(result.error or "Citation formatting failed")
        return result.data  # type: ignore[no-any-return]

    async def improve_latex(
        self,
        content: str,
        instructions: str = "Improve clarity and formatting",
        **kwargs: Any,
    ) -> WriterOutput:
        """Improve existing LaTeX content.

        Args:
            content: Existing LaTeX content to improve.
            instructions: Specific improvement instructions.
            **kwargs: Additional arguments.

        Returns:
            WriterOutput with improved LaTeX.
        """
        prompt = f"""Improve the following LaTeX content:

CONTENT:
{content}

INSTRUCTIONS:
{instructions}

Maintain the overall structure but improve clarity, formatting, and academic tone.
Ensure all LaTeX syntax is correct."""

        result = await self.execute(prompt, output_format="latex", **kwargs)
        if not result.success:
            raise WorkerError(result.error or "LaTeX improvement failed")
        return result.data  # type: ignore[no-any-return]

    async def extract_key_points(
        self,
        text: str,
        max_points: int = 5,
        **kwargs: Any,
    ) -> WriterOutput:
        """Extract key points from text.

        Args:
            text: Source text to analyze.
            max_points: Maximum number of key points.
            **kwargs: Additional arguments.

        Returns:
            WriterOutput with bullet-pointed key points.
        """
        prompt = f"""Extract the {max_points} most important key points from the following text:

TEXT:
{text}

Output as a numbered list of concise key points."""

        kwargs.setdefault("max_tokens", 500)
        result = await self.execute(prompt, output_format="text", **kwargs)
        if not result.success:
            raise WorkerError(result.error or "Key point extraction failed")
        return result.data  # type: ignore[no-any-return]


# Convenience function for one-off writing tasks
async def write_content(
    prompt: str,
    output_format: Literal["latex", "markdown", "text", "bibtex"] = "latex",
    **kwargs: Any,
) -> WriterOutput:
    """Execute a one-off writing task.

    Creates a temporary worker for simple use cases.
    For repeated tasks, use WriterWorker directly.
    """
    worker = WriterWorker()
    try:
        result = await worker.execute(prompt, output_format=output_format, **kwargs)
        if not result.success:
            raise WorkerError(result.error or "Writing task failed")
        return result.data  # type: ignore[no-any-return]
    finally:
        await worker.close()
