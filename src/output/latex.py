r"""LaTeX document generation.

Generates publication-ready LaTeX papers using Jinja2 templates.
Supports multiple formats (article, report) and handles proper
sanitization of content to prevent LaTeX injection attacks.

Security notes:
- All user content is sanitized before insertion
- No \write18 or \input from untrusted sources
- Template delimiters use LaTeX-style syntax to avoid conflicts
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader, Undefined, select_autoescape

from src.output.bibtex import BibTeXManager
from src.utils.errors import LaTeXError

logger = structlog.get_logger(__name__)

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


class PaperFormat(str, Enum):
    """Supported paper formats."""

    ARTICLE = "article"
    REPORT = "report"
    MINIMAL = "minimal"


@dataclass
class Section:
    """Represents a paper section."""

    name: str
    content: str
    level: int = 1  # 0=chapter, 1=section, 2=subsection, 3=subsubsection
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "content": self.content,
            "level": self.level,
            "order": self.order,
        }


@dataclass
class PaperMetadata:
    """Metadata for a research paper."""

    title: str
    author: str
    date: str | None = None
    affiliation: str | None = None
    keywords: list[str] = field(default_factory=list)
    subtitle: str | None = None
    report_number: str | None = None

    def __post_init__(self) -> None:
        if self.date is None:
            self.date = datetime.now().strftime("%B %d, %Y")


@dataclass
class Paper:
    """Represents a complete research paper."""

    metadata: PaperMetadata
    abstract: str | None = None
    sections: list[Section] = field(default_factory=list)
    acknowledgments: str | None = None
    appendices: list[Section] = field(default_factory=list)
    bibliography_file: str | None = None
    bibliography_style: str = "plainnat"
    references: list[str] = field(default_factory=list)  # For inline references

    def add_section(
        self,
        name: str,
        content: str,
        level: int = 1,
    ) -> None:
        """Add a section to the paper."""
        order = len(self.sections)
        section = Section(name=name, content=content, level=level, order=order)
        self.sections.append(section)

    def add_appendix(self, name: str, content: str) -> None:
        """Add an appendix to the paper."""
        appendix = Section(name=name, content=content, level=0, order=len(self.appendices))
        self.appendices.append(appendix)


class LaTeXGenerator:
    """Generates LaTeX documents from templates.

    Uses Jinja2 with LaTeX-compatible delimiters:
    - Variables: \\VAR{variable}
    - Blocks: \\BLOCK{...}
    - Comments: \\#{...}
    """

    def __init__(self, templates_dir: Path | None = None) -> None:
        """Initialize the generator.

        Args:
            templates_dir: Custom templates directory (default: built-in templates)
        """
        self._templates_dir = templates_dir or TEMPLATES_DIR
        self._env = self._create_environment()
        self._logger = structlog.get_logger(__name__)

    def _create_environment(self) -> Environment:
        """Create Jinja2 environment with LaTeX-safe delimiters."""
        env = Environment(
            loader=FileSystemLoader(str(self._templates_dir)),
            # LaTeX-compatible delimiters
            block_start_string=r"\BLOCK{",
            block_end_string="}",
            variable_start_string=r"\VAR{",
            variable_end_string="}",
            comment_start_string=r"\#{",
            comment_end_string="}",
            # Whitespace handling
            trim_blocks=True,
            lstrip_blocks=True,
            # No auto-escaping (we do manual LaTeX sanitization)
            autoescape=select_autoescape([]),
            # Keep undefined variables as empty
            undefined=SilentUndefined,
        )

        # Add custom filters
        env.filters["latex_escape"] = latex_escape_text
        env.filters["truncate"] = _jinja_truncate

        return env

    def render(
        self,
        paper: Paper,
        format: PaperFormat = PaperFormat.ARTICLE,
        sanitize: bool = True,
    ) -> str:
        """Render a paper to LaTeX.

        Args:
            paper: Paper object with content
            format: Output format (article, report, minimal)
            sanitize: Whether to sanitize content (default: True)

        Returns:
            Complete LaTeX document string

        Raises:
            LaTeXError: If rendering fails
        """
        try:
            # Sanitize content if requested
            if sanitize:
                paper = self._sanitize_paper(paper)

            # Load template
            template_name = f"{format.value}.tex"
            template = self._env.get_template(template_name)

            # Prepare context
            context = self._build_context(paper)

            # Render
            latex_content = template.render(**context)

            self._logger.info(
                "Generated LaTeX document",
                format=format.value,
                sections=len(paper.sections),
                title=paper.metadata.title[:50],
            )

            return latex_content

        except Exception as e:
            self._logger.error("LaTeX generation failed", error=str(e))
            raise LaTeXError(f"Failed to generate LaTeX: {e}") from e

    def _build_context(self, paper: Paper) -> dict[str, Any]:
        """Build template context from paper."""
        return {
            # Metadata
            "title": paper.metadata.title,
            "author": paper.metadata.author,
            "date": paper.metadata.date,
            "affiliation": paper.metadata.affiliation,
            "keywords": paper.metadata.keywords,
            "subtitle": paper.metadata.subtitle,
            "report_number": paper.metadata.report_number,
            # Content
            "abstract": paper.abstract,
            "sections": [s.to_dict() for s in paper.sections],
            "acknowledgments": paper.acknowledgments,
            "appendices": [a.to_dict() for a in paper.appendices],
            # Bibliography
            "bibliography_file": paper.bibliography_file,
            "bibliography_style": paper.bibliography_style,
            "references": paper.references,
        }

    def _sanitize_paper(self, paper: Paper) -> Paper:
        """Sanitize all paper content for LaTeX safety.

        Creates a new Paper object with sanitized content.
        """
        from copy import deepcopy

        sanitized = deepcopy(paper)

        # Sanitize metadata
        sanitized.metadata.title = sanitize_latex_text(paper.metadata.title)
        sanitized.metadata.author = sanitize_latex_text(paper.metadata.author)
        if paper.metadata.affiliation:
            sanitized.metadata.affiliation = sanitize_latex_text(paper.metadata.affiliation)
        if paper.metadata.subtitle:
            sanitized.metadata.subtitle = sanitize_latex_text(paper.metadata.subtitle)
        sanitized.metadata.keywords = [sanitize_latex_text(k) for k in paper.metadata.keywords]

        # Sanitize content
        if paper.abstract:
            sanitized.abstract = sanitize_latex_content(paper.abstract)

        sanitized.sections = [
            Section(
                name=sanitize_latex_text(s.name),
                content=sanitize_latex_content(s.content),
                level=s.level,
                order=s.order,
            )
            for s in paper.sections
        ]

        if paper.acknowledgments:
            sanitized.acknowledgments = sanitize_latex_content(paper.acknowledgments)

        sanitized.appendices = [
            Section(
                name=sanitize_latex_text(a.name),
                content=sanitize_latex_content(a.content),
                level=a.level,
                order=a.order,
            )
            for a in paper.appendices
        ]

        sanitized.references = [sanitize_latex_text(r) for r in paper.references]

        return sanitized

    def render_from_state(
        self,
        state: dict[str, Any],
        format: PaperFormat = PaperFormat.ARTICLE,
        bib_manager: BibTeXManager | None = None,
    ) -> tuple[str, str | None]:
        """Render paper from workflow state.

        Args:
            state: ResearchState dictionary
            format: Output format
            bib_manager: Optional BibTeX manager with citations

        Returns:
            Tuple of (latex_content, bibtex_content or None)
        """
        # Extract metadata
        title = state.get("title") or "Untitled Research Paper"
        author = "Research Agent"  # Could be configurable

        metadata = PaperMetadata(
            title=title,
            author=author,
            keywords=state.get("domains", []),
        )

        # Create paper
        paper = Paper(
            metadata=metadata,
            abstract=state.get("abstract"),
            bibliography_file="references" if bib_manager else None,
        )

        # Add sections from state
        sections = state.get("sections", {})
        section_order = [
            "Introduction",
            "Background",
            "Literature Review",
            "Methodology",
            "Methods",
            "Analysis",
            "Results",
            "Discussion",
            "Conclusion",
            "Future Work",
        ]

        added_sections = set()
        for section_name in section_order:
            if section_name in sections:
                paper.add_section(section_name, sections[section_name])
                added_sections.add(section_name)

        # Add any remaining sections not in the order list
        for section_name, content in sections.items():
            if section_name not in added_sections:
                paper.add_section(section_name, content)

        # Render LaTeX
        latex_content = self.render(paper, format=format)

        # Generate BibTeX if manager provided
        bibtex_content = None
        if bib_manager:
            bibtex_content = bib_manager.to_bibtex()

        return latex_content, bibtex_content


# =============================================================================
# Sanitization Functions
# =============================================================================


def sanitize_latex_text(text: str) -> str:
    """Sanitize text for safe use in LaTeX.

    Escapes special characters that have meaning in LaTeX.
    Does NOT preserve LaTeX commands.

    Args:
        text: Input text

    Returns:
        Sanitized text safe for LaTeX
    """
    if not text:
        return ""

    # Characters to escape
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]

    for char, replacement in replacements:
        text = text.replace(char, replacement)

    return text


def sanitize_latex_content(content: str) -> str:
    """Sanitize content that may contain intended LaTeX commands.

    Allows safe LaTeX commands (math, formatting) but blocks
    dangerous commands that could execute code or access files.

    Args:
        content: Input content (may contain LaTeX)

    Returns:
        Sanitized content with dangerous commands removed
    """
    if not content:
        return ""

    # Remove dangerous commands
    content = remove_dangerous_commands(content)

    # Remove potential shell escapes
    content = remove_shell_escapes(content)

    return content


def remove_dangerous_commands(content: str) -> str:
    """Remove dangerous LaTeX commands from content.

    Blocks commands that can:
    - Execute shell commands (\\write18)
    - Read/write files (\\input, \\include, \\openin, \\openout)
    - Define new commands unsafely
    """
    # Dangerous command patterns
    dangerous_patterns = [
        # Shell escape
        r"\\write18\s*\{[^}]*\}",
        r"\\immediate\\write18\s*\{[^}]*\}",
        r"\\ShellEscape\s*\{[^}]*\}",
        # File operations
        r"\\input\s*\{[^}]*\}",
        r"\\include\s*\{[^}]*\}",
        r"\\openin\s*[^\s\\]+",
        r"\\openout\s*[^\s\\]+",
        r"\\read\s*[^\s\\]+",
        r"\\write\s*[^\s\\]+",
        r"\\closein\s*[^\s\\]+",
        r"\\closeout\s*[^\s\\]+",
        # Catcode manipulation (can be used to bypass restrictions)
        r"\\catcode\s*[`'].[=\s]*\d+",
        # Direct Lua code (LuaTeX)
        r"\\directlua\s*\{[^}]*\}",
        r"\\luacode\s*\{[^}]*\}",
        r"\\luadirect\s*\{[^}]*\}",
        # Python (PythonTeX)
        r"\\py\s*\{[^}]*\}",
        r"\\pyc\s*\{[^}]*\}",
        # Special expansion
        r"\\special\s*\{[^}]*\}",
    ]

    for pattern in dangerous_patterns:
        content = re.sub(pattern, "[REMOVED]", content, flags=re.IGNORECASE)

    return content


def remove_shell_escapes(content: str) -> str:
    """Remove potential shell escape sequences."""
    # Backticks (potential shell execution in some systems)
    content = content.replace("`", "'")

    # Remove any remaining write18 variants
    content = re.sub(r"\\write\d+", "[REMOVED]", content)

    return content


def latex_escape_text(text: str) -> str:
    """Jinja2 filter for escaping text.

    Alias for sanitize_latex_text for use in templates.
    """
    return sanitize_latex_text(text)


def _jinja_truncate(text: str, length: int = 50, end: bool = False) -> str:
    """Truncate text for Jinja2 template."""
    if not text:
        return ""
    if len(text) <= length:
        return text
    if end:
        return text[:length] + "..."
    return text[:length]


# =============================================================================
# Jinja2 Undefined Handler
# =============================================================================


class SilentUndefined(Undefined):
    """Jinja2 undefined that returns empty string instead of raising errors."""

    def _fail_with_undefined_error(  # type: ignore[override]
        self, *_args: Any, **_kwargs: Any
    ) -> None:
        """Override to silently return empty instead of raising."""
        pass

    def __str__(self) -> str:
        return ""

    def __bool__(self) -> bool:
        return False

    def __iter__(self) -> Any:
        return iter([])

    def __len__(self) -> int:
        return 0

    def __getattr__(self, name: str) -> "SilentUndefined":
        return SilentUndefined()

    def __call__(  # type: ignore[override]
        self, *_args: Any, **_kwargs: Any
    ) -> "SilentUndefined":
        return SilentUndefined()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_paper(
    title: str,
    author: str,
    abstract: str | None = None,
    sections: list[tuple[str, str]] | None = None,
    keywords: list[str] | None = None,
    format: PaperFormat = PaperFormat.ARTICLE,
) -> str:
    """Quick paper creation helper.

    Args:
        title: Paper title
        author: Author name
        abstract: Paper abstract
        sections: List of (name, content) tuples
        keywords: List of keywords
        format: Output format

    Returns:
        Complete LaTeX document string
    """
    metadata = PaperMetadata(
        title=title,
        author=author,
        keywords=keywords or [],
    )

    paper = Paper(metadata=metadata, abstract=abstract)

    if sections:
        for name, content in sections:
            paper.add_section(name, content)

    generator = LaTeXGenerator()
    return generator.render(paper, format=format)


def validate_latex_syntax(content: str) -> list[str]:
    """Basic validation of LaTeX syntax.

    Checks for common issues like unbalanced braces and
    undefined environments. Not a full LaTeX compiler check.

    Args:
        content: LaTeX content to validate

    Returns:
        List of warning/error messages (empty if valid)
    """
    issues: list[str] = []

    # Check brace balance
    open_braces = content.count("{")
    close_braces = content.count("}")
    if open_braces != close_braces:
        issues.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")

    # Check for common environment issues
    begin_count = len(re.findall(r"\\begin\{(\w+)\}", content))
    end_count = len(re.findall(r"\\end\{(\w+)\}", content))
    if begin_count != end_count:
        issues.append(f"Unbalanced environments: {begin_count} begin, {end_count} end")

    # Check for document structure
    if "\\documentclass" not in content:
        issues.append("Missing \\documentclass")

    if "\\begin{document}" not in content:
        issues.append("Missing \\begin{document}")

    if "\\end{document}" not in content:
        issues.append("Missing \\end{document}")

    return issues
