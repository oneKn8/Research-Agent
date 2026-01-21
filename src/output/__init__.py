"""Output generation module for research papers.

This module provides tools for generating publication-ready
LaTeX papers with proper citation management.

Components:
- bibtex: BibTeX citation management
- latex: LaTeX document generation with Jinja2 templates
- compiler: LaTeX compilation to PDF
"""

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
    compile_latex_string,
    compile_paper,
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

__all__ = [
    "BibTeXEntry",
    "BibTeXManager",
    "CompilationResult",
    "CompilationStatus",
    "EntryType",
    "LaTeXGenerator",
    "Paper",
    "PaperCompiler",
    "PaperFormat",
    "PaperMetadata",
    "Section",
    "compile_latex_string",
    "compile_paper",
    "create_article_entry",
    "create_arxiv_entry",
    "create_inproceedings_entry",
    "create_paper",
    "create_url_entry",
    "generate_citation_key",
    "parse_bibtex",
    "sanitize_latex_content",
    "sanitize_latex_text",
    "validate_bibtex_entry",
    "validate_latex_syntax",
]
