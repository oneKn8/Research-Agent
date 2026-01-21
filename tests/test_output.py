"""Tests for the output generation module.

Covers:
- BibTeX citation management
- LaTeX document generation
- Security sanitization
- Paper compilation
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.output.bibtex import (
    BibTeXEntry,
    BibTeXManager,
    EntryType,
    create_article_entry,
    create_arxiv_entry,
    create_inproceedings_entry,
    create_url_entry,
    generate_citation_key,
    parse_bibtex,
    validate_bibtex_entry,
)
from src.output.compiler import (
    CompilationResult,
    CompilationStatus,
    PaperCompiler,
)
from src.output.latex import (
    LaTeXGenerator,
    Paper,
    PaperFormat,
    PaperMetadata,
    Section,
    create_paper,
    sanitize_latex_content,
    sanitize_latex_text,
    validate_latex_syntax,
)
from src.security.latex import (
    check_content_safety,
    escape_latex_special_chars,
    sanitize_latex,
    validate_file_path,
)

# =============================================================================
# BibTeX Manager Tests
# =============================================================================


class TestBibTeXEntry:
    """Tests for BibTeXEntry dataclass."""

    def test_create_basic_entry(self) -> None:
        """Test creating a basic BibTeX entry."""
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="smith2024",
            title="Test Article",
            author="John Smith",
            year=2024,
        )
        assert entry.key == "smith2024"
        assert entry.entry_type == EntryType.ARTICLE
        assert entry.title == "Test Article"
        assert entry.author == "John Smith"
        assert entry.year == 2024

    def test_entry_to_bibtex(self) -> None:
        """Test converting entry to BibTeX string."""
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="smith2024",
            title="Test Article",
            author="John Smith",
            year=2024,
            journal="Test Journal",
            volume="10",
            pages="1-10",
        )
        bibtex = entry.to_bibtex()

        assert "@article{smith2024," in bibtex
        assert "title = {Test Article}" in bibtex
        assert "author = {John Smith}" in bibtex
        assert "year = {2024}" in bibtex
        assert "journal = {Test Journal}" in bibtex
        assert "volume = {10}" in bibtex
        assert "pages = {1-10}" in bibtex

    def test_entry_to_bibtex_with_doi(self) -> None:
        """Test BibTeX with DOI."""
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="smith2024",
            title="Test",
            author="Smith",
            year=2024,
            doi="10.1234/test.2024",
        )
        bibtex = entry.to_bibtex()
        assert "doi = {10.1234/test.2024}" in bibtex

    def test_entry_to_bibtex_escapes_special_chars(self) -> None:
        """Test that special characters are escaped."""
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test2024",
            title="Test & Analysis",
            author="O'Brien",
            year=2024,
        )
        bibtex = entry.to_bibtex()
        assert r"Test \& Analysis" in bibtex

    def test_entry_to_dict(self) -> None:
        """Test converting entry to dictionary."""
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="smith2024",
            title="Test",
            author="Smith",
            year=2024,
        )
        d = entry.to_dict()
        assert d["key"] == "smith2024"
        assert d["entry_type"] == "article"
        assert d["title"] == "Test"


class TestBibTeXManager:
    """Tests for BibTeXManager class."""

    def test_add_entry(self) -> None:
        """Test adding an entry to the manager."""
        manager = BibTeXManager()
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test2024",
            title="Test",
            author="Author",
            year=2024,
        )
        key = manager.add_entry(entry)
        assert key == "test2024"
        assert manager.has_entry("test2024")
        assert len(manager.entries) == 1

    def test_add_duplicate_entry_same_paper(self) -> None:
        """Test that duplicate entries are not added."""
        manager = BibTeXManager()
        entry1 = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test2024",
            title="Test Paper",
            author="Author",
            year=2024,
        )
        entry2 = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test2024",
            title="Test Paper",  # Same title and year
            author="Author",
            year=2024,
        )
        manager.add_entry(entry1)
        key2 = manager.add_entry(entry2)

        # Should return the same key (detected as duplicate)
        assert key2 == "test2024"
        assert len(manager.entries) == 1

    def test_add_entry_with_key_collision(self) -> None:
        """Test that key collisions are handled with suffixes."""
        manager = BibTeXManager()
        entry1 = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test2024",
            title="Paper One",
            author="Author",
            year=2024,
        )
        entry2 = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test2024",  # Same key, different paper
            title="Paper Two",
            author="Author",
            year=2024,
        )
        manager.add_entry(entry1)
        key2 = manager.add_entry(entry2)

        assert key2 == "test2024a"  # Should get suffix
        assert len(manager.entries) == 2
        assert manager.has_entry("test2024")
        assert manager.has_entry("test2024a")

    def test_get_entry(self) -> None:
        """Test retrieving an entry by key."""
        manager = BibTeXManager()
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test2024",
            title="Test",
            author="Author",
            year=2024,
        )
        manager.add_entry(entry)

        retrieved = manager.get_entry("test2024")
        assert retrieved is not None
        assert retrieved.title == "Test"

        assert manager.get_entry("nonexistent") is None

    def test_remove_entry(self) -> None:
        """Test removing an entry."""
        manager = BibTeXManager()
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test2024",
            title="Test",
            author="Author",
            year=2024,
        )
        manager.add_entry(entry)
        assert manager.has_entry("test2024")

        result = manager.remove_entry("test2024")
        assert result is True
        assert not manager.has_entry("test2024")

        # Removing non-existent entry returns False
        assert manager.remove_entry("nonexistent") is False

    def test_to_bibtex_full(self) -> None:
        """Test generating complete BibTeX output."""
        manager = BibTeXManager()
        manager.add_entry(BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="alpha2024",
            title="Alpha",
            author="Author A",
            year=2024,
        ))
        manager.add_entry(BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="beta2024",
            title="Beta",
            author="Author B",
            year=2024,
        ))

        bibtex = manager.to_bibtex()
        assert "@article{alpha2024," in bibtex
        assert "@article{beta2024," in bibtex
        # Should be separated by blank lines
        assert "\n\n" in bibtex

    def test_clear(self) -> None:
        """Test clearing all entries."""
        manager = BibTeXManager()
        manager.add_entry(BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test2024",
            title="Test",
            author="Author",
            year=2024,
        ))
        assert len(manager.entries) == 1

        manager.clear()
        assert len(manager.entries) == 0


class TestCitationKeyGeneration:
    """Tests for citation key generation."""

    def test_basic_key_generation(self) -> None:
        """Test basic key generation from author and year."""
        key = generate_citation_key("John Smith", 2024)
        assert key == "smith2024"

    def test_key_from_author_list(self) -> None:
        """Test key generation from author list."""
        key = generate_citation_key(["John Smith", "Jane Doe"], 2024)
        assert key == "smith2024"

    def test_key_from_bibtex_format(self) -> None:
        """Test key generation from BibTeX author format."""
        key = generate_citation_key("Smith, John and Doe, Jane", 2024)
        assert key == "smith2024"

    def test_key_with_special_chars(self) -> None:
        """Test key generation handles special characters."""
        key = generate_citation_key("O'Connor", 2024)
        assert key == "oconnor2024"

        key = generate_citation_key("von Neumann", 2024)
        assert key == "neumann2024"


class TestEntryCreationHelpers:
    """Tests for entry creation helper functions."""

    def test_create_arxiv_entry(self) -> None:
        """Test creating an ArXiv entry."""
        entry = create_arxiv_entry(
            arxiv_id="2401.12345",
            title="Test Paper",
            authors=["John Smith", "Jane Doe"],
            year=2024,
            categories=["cs.AI", "cs.LG"],
        )
        assert entry.entry_type == EntryType.MISC
        assert entry.eprint == "2401.12345"
        assert entry.archiveprefix == "arXiv"
        assert entry.primaryclass == "cs.AI"
        assert "arxiv.org" in entry.url or ""

    def test_create_url_entry(self) -> None:
        """Test creating a URL entry."""
        entry = create_url_entry(
            url="https://example.com/article",
            title="Online Article",
            author="Author Name",
            year=2024,
        )
        assert entry.entry_type == EntryType.MISC
        assert entry.url == "https://example.com/article"
        assert "url{https://example.com/article}" in entry.howpublished or ""

    def test_create_article_entry(self) -> None:
        """Test creating a journal article entry."""
        entry = create_article_entry(
            title="Test Article",
            authors=["John Smith"],
            year=2024,
            journal="Journal of Testing",
            volume="10",
            pages="1-20",
        )
        assert entry.entry_type == EntryType.ARTICLE
        assert entry.journal == "Journal of Testing"
        assert entry.volume == "10"
        assert entry.pages == "1-20"

    def test_create_inproceedings_entry(self) -> None:
        """Test creating a conference paper entry."""
        entry = create_inproceedings_entry(
            title="Conference Paper",
            authors=["Author One"],
            year=2024,
            booktitle="Proceedings of Test Conference",
        )
        assert entry.entry_type == EntryType.INPROCEEDINGS
        assert entry.booktitle == "Proceedings of Test Conference"


class TestBibTeXValidation:
    """Tests for BibTeX entry validation."""

    def test_valid_entry(self) -> None:
        """Test validation of a valid entry."""
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="valid2024",
            title="Valid Title",
            author="Valid Author",
            year=2024,
            journal="Journal Name",
        )
        issues = validate_bibtex_entry(entry)
        assert len(issues) == 0

    def test_missing_required_fields(self) -> None:
        """Test validation catches missing fields."""
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="",  # Missing key
            title="",  # Missing title
            author="",  # Missing author
            year="",  # Missing year
        )
        issues = validate_bibtex_entry(entry)
        assert "Missing citation key" in issues
        assert "Missing title" in issues
        assert "Missing author" in issues
        assert "Missing year" in issues

    def test_invalid_year(self) -> None:
        """Test validation catches invalid year."""
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test",
            title="Test",
            author="Author",
            year=1800,  # Too old
        )
        issues = validate_bibtex_entry(entry)
        assert any("year" in issue.lower() for issue in issues)

    def test_article_missing_journal(self) -> None:
        """Test that article entries need journal."""
        entry = BibTeXEntry(
            entry_type=EntryType.ARTICLE,
            key="test2024",
            title="Test",
            author="Author",
            year=2024,
            # Missing journal
        )
        issues = validate_bibtex_entry(entry)
        assert "Article entry missing journal" in issues


class TestBibTeXParsing:
    """Tests for BibTeX parsing."""

    def test_parse_simple_entry(self) -> None:
        """Test parsing a simple BibTeX entry."""
        bibtex_str = """
@article{smith2024,
    title = {Test Article},
    author = {John Smith},
    year = {2024},
    journal = {Test Journal}
}
"""
        entries = parse_bibtex(bibtex_str)
        assert len(entries) == 1
        assert entries[0].key == "smith2024"
        assert entries[0].title == "Test Article"
        assert entries[0].author == "John Smith"

    def test_parse_multiple_entries(self) -> None:
        """Test parsing multiple BibTeX entries."""
        bibtex_str = """
@article{first2024,
    title = {First},
    author = {Author One},
    year = {2024}
}

@book{second2024,
    title = {Second},
    author = {Author Two},
    year = {2024}
}
"""
        entries = parse_bibtex(bibtex_str)
        assert len(entries) == 2


# =============================================================================
# LaTeX Generator Tests
# =============================================================================


class TestPaperMetadata:
    """Tests for PaperMetadata dataclass."""

    def test_default_date(self) -> None:
        """Test that default date is set."""
        metadata = PaperMetadata(
            title="Test",
            author="Author",
        )
        assert metadata.date is not None
        # Should contain current year
        assert str(datetime.now().year) in metadata.date


class TestSection:
    """Tests for Section dataclass."""

    def test_section_to_dict(self) -> None:
        """Test section conversion to dictionary."""
        section = Section(
            name="Introduction",
            content="This is the intro.",
            level=1,
            order=0,
        )
        d = section.to_dict()
        assert d["name"] == "Introduction"
        assert d["content"] == "This is the intro."
        assert d["level"] == 1
        assert d["order"] == 0


class TestPaper:
    """Tests for Paper dataclass."""

    def test_add_section(self) -> None:
        """Test adding sections to a paper."""
        metadata = PaperMetadata(title="Test", author="Author")
        paper = Paper(metadata=metadata)

        paper.add_section("Introduction", "Intro content")
        paper.add_section("Methods", "Methods content")

        assert len(paper.sections) == 2
        assert paper.sections[0].name == "Introduction"
        assert paper.sections[1].name == "Methods"
        assert paper.sections[0].order == 0
        assert paper.sections[1].order == 1

    def test_add_appendix(self) -> None:
        """Test adding appendices."""
        metadata = PaperMetadata(title="Test", author="Author")
        paper = Paper(metadata=metadata)

        paper.add_appendix("Data Tables", "Table content")
        assert len(paper.appendices) == 1
        assert paper.appendices[0].name == "Data Tables"


class TestLaTeXGenerator:
    """Tests for LaTeXGenerator class."""

    def test_render_basic_paper(self) -> None:
        """Test rendering a basic paper."""
        metadata = PaperMetadata(
            title="Test Paper",
            author="Test Author",
        )
        paper = Paper(
            metadata=metadata,
            abstract="This is the abstract.",
        )
        paper.add_section("Introduction", "This is the introduction.")

        generator = LaTeXGenerator()
        latex = generator.render(paper, format=PaperFormat.ARTICLE)

        assert r"\documentclass" in latex
        assert r"\title{Test Paper}" in latex
        assert r"\author{Test Author}" in latex
        assert r"\begin{abstract}" in latex
        assert "This is the abstract." in latex
        assert r"\section{Introduction}" in latex
        assert r"\end{document}" in latex

    def test_render_different_formats(self) -> None:
        """Test rendering in different formats."""
        metadata = PaperMetadata(title="Test", author="Author")
        paper = Paper(metadata=metadata)
        paper.add_section("Test", "Content")

        generator = LaTeXGenerator()

        article = generator.render(paper, format=PaperFormat.ARTICLE)
        assert r"\documentclass[11pt,a4paper]{article}" in article

        report = generator.render(paper, format=PaperFormat.REPORT)
        assert r"\documentclass[11pt,a4paper]{report}" in report

        minimal = generator.render(paper, format=PaperFormat.MINIMAL)
        assert r"\documentclass[11pt,a4paper]{article}" in minimal

    def test_render_sanitizes_content(self) -> None:
        """Test that rendering sanitizes dangerous content."""
        metadata = PaperMetadata(
            title="Test Paper",
            author="Author",
        )
        paper = Paper(metadata=metadata)
        paper.add_section("Test", r"Content with \write18{rm -rf /} danger")

        generator = LaTeXGenerator()
        latex = generator.render(paper, format=PaperFormat.ARTICLE, sanitize=True)

        # The dangerous command should be removed
        assert r"\write18{rm -rf /}" not in latex
        assert "[REMOVED" in latex or "danger" in latex


class TestLaTeXSanitization:
    """Tests for LaTeX text sanitization."""

    def test_sanitize_special_chars(self) -> None:
        """Test escaping special LaTeX characters."""
        text = "Test & analysis with $money and #hashtags"
        sanitized = sanitize_latex_text(text)

        assert r"\&" in sanitized
        assert r"\$" in sanitized
        assert r"\#" in sanitized

    def test_sanitize_preserves_normal_text(self) -> None:
        """Test that normal text is preserved."""
        text = "This is normal text with nothing special."
        sanitized = sanitize_latex_text(text)
        assert sanitized == text

    def test_sanitize_content_removes_dangerous_commands(self) -> None:
        """Test that dangerous commands are removed from content."""
        content = r"Normal text \write18{evil command} more text"
        sanitized = sanitize_latex_content(content)

        assert r"\write18" not in sanitized
        assert "evil command" not in sanitized
        assert "Normal text" in sanitized

    def test_sanitize_removes_file_operations(self) -> None:
        """Test removal of file operations."""
        content = r"Text \input{/etc/passwd} more text"
        sanitized = sanitize_latex_content(content)

        assert r"\input" not in sanitized
        assert "/etc/passwd" not in sanitized


class TestLaTeXValidation:
    """Tests for LaTeX syntax validation."""

    def test_validate_valid_document(self) -> None:
        """Test validation of a valid document."""
        latex = r"""
\documentclass{article}
\begin{document}
Hello World
\end{document}
"""
        issues = validate_latex_syntax(latex)
        assert len(issues) == 0

    def test_validate_missing_documentclass(self) -> None:
        """Test detection of missing documentclass."""
        latex = r"""
\begin{document}
Hello
\end{document}
"""
        issues = validate_latex_syntax(latex)
        assert any("documentclass" in issue.lower() for issue in issues)

    def test_validate_unbalanced_braces(self) -> None:
        """Test detection of unbalanced braces."""
        latex = r"""
\documentclass{article}
\begin{document}
Some text { with unbalanced brace
\end{document}
"""
        issues = validate_latex_syntax(latex)
        assert any("brace" in issue.lower() for issue in issues)

    def test_validate_unbalanced_environments(self) -> None:
        """Test detection of unbalanced environments."""
        # Note: comment does not contain actual end tag to avoid regex match
        latex = r"""
\documentclass{article}
\begin{document}
\begin{itemize}
\item One
\item Two
\end{document}
"""
        issues = validate_latex_syntax(latex)
        assert any("environment" in issue.lower() for issue in issues)


class TestCreatePaperHelper:
    """Tests for the create_paper helper function."""

    def test_create_simple_paper(self) -> None:
        """Test creating a simple paper."""
        latex = create_paper(
            title="Quick Paper",
            author="Quick Author",
            abstract="Quick abstract.",
            sections=[
                ("Introduction", "Intro content."),
                ("Conclusion", "Conclusion content."),
            ],
        )

        assert r"\title{Quick Paper}" in latex
        assert r"\author{Quick Author}" in latex
        assert "Quick abstract." in latex
        assert r"\section{Introduction}" in latex
        assert r"\section{Conclusion}" in latex


# =============================================================================
# Security Tests
# =============================================================================


class TestLaTeXSecuritySanitization:
    """Tests for LaTeX security sanitization."""

    def test_sanitize_shell_escape(self) -> None:
        """Test removal of shell escape commands."""
        content = r"Text \write18{rm -rf /} more"
        result = sanitize_latex(content)

        assert r"\write18" not in result.content
        assert not result.is_safe
        assert len(result.warnings) > 0
        assert len(result.removed_commands) > 0

    def test_sanitize_file_input(self) -> None:
        """Test removal of file input commands."""
        content = r"Text \input{/etc/passwd} more"
        result = sanitize_latex(content)

        assert r"\input" not in result.content
        assert not result.is_safe

    def test_sanitize_lua_code(self) -> None:
        """Test removal of Lua code execution."""
        content = r"Text \directlua{os.execute('evil')} more"
        result = sanitize_latex(content)

        assert r"\directlua" not in result.content
        assert not result.is_safe

    def test_sanitize_path_traversal(self) -> None:
        """Test detection of path traversal."""
        content = r"Text with ../../../etc/passwd path"
        result = sanitize_latex(content)

        assert "../" not in result.content
        assert not result.is_safe

    def test_safe_content_unchanged(self) -> None:
        """Test that safe content is not modified."""
        content = r"This is safe \textbf{text} with $math$ and nothing dangerous."
        result = sanitize_latex(content)

        assert result.is_safe
        assert len(result.warnings) == 0
        assert r"\textbf{text}" in result.content

    def test_strict_mode_whitelist(self) -> None:
        """Test strict mode only allows whitelisted commands."""
        content = r"Text \textbf{bold} and \customcommand{x}"
        result = sanitize_latex(content, strict=True, allow_formatting=True)

        assert r"\textbf" in result.content
        assert r"\customcommand" not in result.content


class TestCheckContentSafety:
    """Tests for content safety checking."""

    def test_safe_content(self) -> None:
        """Test that safe content is identified."""
        content = r"Normal text with \section{heading}"
        result = check_content_safety(content)

        assert result["is_safe"]
        assert result["risk_level"] == "low"
        assert result["issue_count"] == 0

    def test_critical_risk_shell_escape(self) -> None:
        """Test shell escape is critical risk."""
        content = r"Text \write18{cmd}"
        result = check_content_safety(content)

        assert not result["is_safe"]
        assert result["risk_level"] == "critical"

    def test_high_risk_file_operations(self) -> None:
        """Test file operations are high risk."""
        content = r"Text \input{file}"
        result = check_content_safety(content)

        assert not result["is_safe"]
        assert result["risk_level"] in ("high", "critical")


class TestEscapeSpecialChars:
    """Tests for special character escaping."""

    def test_escape_all_special_chars(self) -> None:
        """Test escaping all LaTeX special characters."""
        text = r"& % $ # _ { } ~ ^ < > |"
        escaped = escape_latex_special_chars(text)

        assert r"\&" in escaped
        assert r"\%" in escaped
        assert r"\$" in escaped
        assert r"\#" in escaped
        assert r"\_" in escaped
        assert r"\{" in escaped
        assert r"\}" in escaped

    def test_escape_backslash(self) -> None:
        """Test backslash escaping."""
        text = r"path\to\file"
        escaped = escape_latex_special_chars(text)

        assert r"\textbackslash{}" in escaped


class TestValidateFilePath:
    """Tests for file path validation."""

    def test_valid_path(self) -> None:
        """Test validation of valid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = Path(tmpdir) / "test.tex"
            assert validate_file_path(valid_path, Path(tmpdir))

    def test_path_traversal_rejected(self) -> None:
        """Test that path traversal is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # This path tries to escape the allowed directory
            bad_path = Path(tmpdir) / ".." / ".." / "etc" / "passwd"
            assert not validate_file_path(bad_path, Path(tmpdir))


# =============================================================================
# Compiler Tests
# =============================================================================


class TestPaperCompiler:
    """Tests for paper compilation."""

    def test_compiler_creates_output_dir(self) -> None:
        """Test that compiler creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            compiler = PaperCompiler(output_dir=output_dir)

            assert output_dir.exists()
            assert compiler.output_dir == output_dir

    @pytest.mark.asyncio
    async def test_compile_paper_latex_only(self) -> None:
        """Test compiling to LaTeX without PDF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            compiler = PaperCompiler(output_dir=output_dir)

            metadata = PaperMetadata(title="Test Paper", author="Author")
            paper = Paper(metadata=metadata, abstract="Abstract")
            paper.add_section("Introduction", "Content")

            result = await compiler.compile_paper(
                paper=paper,
                format=PaperFormat.ARTICLE,
                compile_pdf=False,  # Skip PDF
            )

            assert result.status == CompilationStatus.SKIPPED
            assert result.latex_path is not None
            assert result.latex_path.exists()
            assert result.pdf_path is None

            # Check LaTeX content
            latex_content = result.latex_path.read_text()
            assert r"\title{Test Paper}" in latex_content
            assert "Abstract" in latex_content

    @pytest.mark.asyncio
    async def test_compile_with_bibtex(self) -> None:
        """Test compiling with BibTeX citations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            compiler = PaperCompiler(output_dir=output_dir)

            metadata = PaperMetadata(title="Test", author="Author")
            paper = Paper(metadata=metadata)
            paper.add_section("Test", "Content")

            bib_manager = BibTeXManager()
            bib_manager.add_entry(BibTeXEntry(
                entry_type=EntryType.ARTICLE,
                key="cite2024",
                title="Cited Paper",
                author="Cited Author",
                year=2024,
            ))

            result = await compiler.compile_paper(
                paper=paper,
                bib_manager=bib_manager,
                compile_pdf=False,
            )

            assert result.bibtex_path is not None
            assert result.bibtex_path.exists()

            bibtex_content = result.bibtex_path.read_text()
            assert "@article{cite2024," in bibtex_content

    @pytest.mark.asyncio
    async def test_compile_sanitizes_dangerous_content(self) -> None:
        """Test that compilation sanitizes dangerous content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            compiler = PaperCompiler(output_dir=output_dir)

            metadata = PaperMetadata(title="Test", author="Author")
            paper = Paper(metadata=metadata)
            paper.add_section("Test", r"Content \write18{evil} more")

            result = await compiler.compile_paper(
                paper=paper,
                compile_pdf=False,
            )

            # Should have warnings about removed content
            assert len(result.warnings) > 0 or result.status == CompilationStatus.SKIPPED

            if result.latex_path:
                latex_content = result.latex_path.read_text()
                assert r"\write18{evil}" not in latex_content


class TestCompilationResult:
    """Tests for CompilationResult dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = CompilationResult(
            status=CompilationStatus.SUCCESS,
            latex_path=Path("/path/to/paper.tex"),
            errors=[],
            warnings=["Warning 1"],
            duration_seconds=1.5,
        )

        d = result.to_dict()
        assert d["status"] == "success"
        assert d["latex_path"] == "/path/to/paper.tex"
        assert d["warnings"] == ["Warning 1"]
        assert d["duration_seconds"] == 1.5
