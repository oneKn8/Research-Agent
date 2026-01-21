"""BibTeX citation management.

Handles generation, parsing, and management of BibTeX citations.
Supports entries from ArXiv, URLs, DOIs, and manual creation.

Key features:
- Generate unique citation keys
- Deduplicate citations
- Validate BibTeX entries
- Format for LaTeX bibliography
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from src.utils.errors import CitationError

logger = structlog.get_logger(__name__)


class EntryType(str, Enum):
    """BibTeX entry types."""

    ARTICLE = "article"
    BOOK = "book"
    INPROCEEDINGS = "inproceedings"
    MISC = "misc"
    TECHREPORT = "techreport"
    PHDTHESIS = "phdthesis"
    MASTERSTHESIS = "mastersthesis"
    UNPUBLISHED = "unpublished"
    ONLINE = "online"


@dataclass
class BibTeXEntry:
    """Represents a BibTeX citation entry.

    Follows the standard BibTeX format with common fields.
    """

    entry_type: EntryType
    key: str
    title: str
    author: str
    year: int | str
    # Optional fields
    journal: str | None = None
    booktitle: str | None = None
    volume: str | None = None
    number: str | None = None
    pages: str | None = None
    month: str | None = None
    publisher: str | None = None
    address: str | None = None
    doi: str | None = None
    url: str | None = None
    eprint: str | None = None
    archiveprefix: str | None = None
    primaryclass: str | None = None
    abstract: str | None = None
    note: str | None = None
    howpublished: str | None = None
    # Metadata (not included in BibTeX output)
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_bibtex(self, include_abstract: bool = False) -> str:
        """Convert to BibTeX string format.

        Args:
            include_abstract: Whether to include abstract field.

        Returns:
            Formatted BibTeX entry string.
        """
        lines = [f"@{self.entry_type.value}{{{self.key},"]

        # Required fields first
        lines.append(f"    title = {{{self._escape_latex(self.title)}}},")
        lines.append(f"    author = {{{self._escape_latex(self.author)}}},")
        lines.append(f"    year = {{{self.year}}},")

        # Optional fields in standard order
        optional_fields = [
            ("journal", self.journal),
            ("booktitle", self.booktitle),
            ("volume", self.volume),
            ("number", self.number),
            ("pages", self.pages),
            ("month", self.month),
            ("publisher", self.publisher),
            ("address", self.address),
            ("doi", self.doi),
            ("url", self.url),
            ("eprint", self.eprint),
            ("archiveprefix", self.archiveprefix),
            ("primaryclass", self.primaryclass),
            ("howpublished", self.howpublished),
            ("note", self.note),
        ]

        if include_abstract:
            optional_fields.append(("abstract", self.abstract))

        for field_name, value in optional_fields:
            if value:
                escaped = self._escape_latex(str(value))
                lines.append(f"    {field_name} = {{{escaped}}},")

        # Remove trailing comma from last field
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]

        lines.append("}")
        return "\n".join(lines)

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters in text.

        Note: Braces are preserved as they may be intentional for LaTeX commands.
        """
        # Only escape truly problematic characters
        # Don't escape braces as they're used in LaTeX
        replacements = [
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
        ]
        for char, replacement in replacements:
            text = text.replace(char, replacement)
        return text

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entry_type": self.entry_type.value,
            "key": self.key,
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "journal": self.journal,
            "booktitle": self.booktitle,
            "volume": self.volume,
            "number": self.number,
            "pages": self.pages,
            "month": self.month,
            "publisher": self.publisher,
            "address": self.address,
            "doi": self.doi,
            "url": self.url,
            "eprint": self.eprint,
            "archiveprefix": self.archiveprefix,
            "primaryclass": self.primaryclass,
            "note": self.note,
            "source_id": self.source_id,
        }


class BibTeXManager:
    """Manages BibTeX citations for a research paper.

    Handles:
    - Citation key generation
    - Entry deduplication
    - Bibliography formatting
    - Validation
    """

    def __init__(self) -> None:
        """Initialize the BibTeX manager."""
        self._entries: dict[str, BibTeXEntry] = {}
        self._key_counter: dict[str, int] = {}
        self._logger = structlog.get_logger(__name__)

    @property
    def entries(self) -> list[BibTeXEntry]:
        """Get all entries."""
        return list(self._entries.values())

    @property
    def keys(self) -> list[str]:
        """Get all citation keys."""
        return list(self._entries.keys())

    def add_entry(self, entry: BibTeXEntry) -> str:
        """Add a citation entry.

        If entry with same key exists, generates a unique key.

        Args:
            entry: BibTeX entry to add.

        Returns:
            The key used for the entry (may be modified for uniqueness).
        """
        key = entry.key

        # Ensure key uniqueness
        if key in self._entries:
            # Check if it's actually a duplicate entry
            if self._is_duplicate(entry, self._entries[key]):
                self._logger.debug("Duplicate citation detected", key=key)
                return key

            # Generate unique key
            key = self._make_unique_key(key)
            entry.key = key

        self._entries[key] = entry
        self._logger.debug("Added citation", key=key, title=entry.title[:50])
        return key

    def get_entry(self, key: str) -> BibTeXEntry | None:
        """Get entry by key."""
        return self._entries.get(key)

    def remove_entry(self, key: str) -> bool:
        """Remove entry by key."""
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def has_entry(self, key: str) -> bool:
        """Check if entry exists."""
        return key in self._entries

    def _is_duplicate(self, entry1: BibTeXEntry, entry2: BibTeXEntry) -> bool:
        """Check if two entries are duplicates.

        Compares title and year, allowing for minor differences.
        """
        # Normalize titles for comparison
        title1 = self._normalize_text(entry1.title)
        title2 = self._normalize_text(entry2.title)

        # Same title and year = duplicate
        if title1 == title2 and entry1.year == entry2.year:
            return True

        # Same DOI = duplicate
        if entry1.doi and entry2.doi and entry1.doi.lower() == entry2.doi.lower():
            return True

        # Same ArXiv ID = duplicate
        return bool(entry1.eprint and entry2.eprint and entry1.eprint == entry2.eprint)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove accents
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        # Lowercase and remove extra whitespace
        text = " ".join(text.lower().split())
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def _make_unique_key(self, base_key: str) -> str:
        """Generate a unique key by appending a letter suffix."""
        if base_key not in self._key_counter:
            self._key_counter[base_key] = 0

        self._key_counter[base_key] += 1
        suffix = chr(ord("a") + self._key_counter[base_key] - 1)
        return f"{base_key}{suffix}"

    def to_bibtex(self, include_abstracts: bool = False) -> str:
        """Generate complete BibTeX file content.

        Args:
            include_abstracts: Whether to include abstract fields.

        Returns:
            Complete BibTeX file content.
        """
        if not self._entries:
            return ""

        entries_str = []
        # Sort by key for consistent output
        for key in sorted(self._entries.keys()):
            entry = self._entries[key]
            entries_str.append(entry.to_bibtex(include_abstract=include_abstracts))

        return "\n\n".join(entries_str) + "\n"

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        self._key_counter.clear()


# =============================================================================
# Citation Key Generation
# =============================================================================


def generate_citation_key(
    authors: list[str] | str,
    year: int | str,
    _title: str | None = None,
) -> str:
    """Generate a citation key from author and year.

    Format: FirstAuthorLastNameYear (e.g., smith2024)

    Args:
        authors: Author name(s). Can be string or list.
        year: Publication year.
        _title: Reserved for future disambiguation (currently unused).

    Returns:
        Citation key string.
    """
    # Handle author list
    if isinstance(authors, list):
        first_author = authors[0] if authors else "unknown"
    else:
        # Split on "and" for BibTeX format
        parts = authors.split(" and ")
        first_author = parts[0].strip() if parts else "unknown"

    # Extract last name
    # Handle "Last, First" and "First Last" formats
    if "," in first_author:
        last_name = first_author.split(",")[0].strip()
    else:
        parts = first_author.split()
        last_name = parts[-1] if parts else "unknown"

    # Clean the last name
    last_name = _clean_for_key(last_name)

    # Handle year
    year_str = str(year)

    key = f"{last_name}{year_str}"

    return key.lower()


def _clean_for_key(text: str) -> str:
    """Clean text for use in citation key."""
    # Remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Keep only alphanumeric
    text = re.sub(r"[^a-zA-Z0-9]", "", text)
    return text


# =============================================================================
# Entry Creation Helpers
# =============================================================================


def create_arxiv_entry(
    arxiv_id: str,
    title: str,
    authors: list[str],
    year: int,
    abstract: str | None = None,
    categories: list[str] | None = None,
    url: str | None = None,
    doi: str | None = None,
    journal_ref: str | None = None,
) -> BibTeXEntry:
    """Create BibTeX entry from ArXiv paper data.

    Args:
        arxiv_id: ArXiv paper ID (e.g., "2301.12345")
        title: Paper title
        authors: List of author names
        year: Publication year
        abstract: Paper abstract
        categories: ArXiv categories
        url: Entry URL
        doi: DOI if published
        journal_ref: Journal reference if published

    Returns:
        BibTeXEntry for the ArXiv paper
    """
    # Generate citation key
    key = generate_citation_key(authors, year, title)

    # Format authors for BibTeX
    authors_str = " and ".join(authors)

    # Determine entry type
    entry_type = EntryType.ARTICLE if journal_ref else EntryType.MISC

    return BibTeXEntry(
        entry_type=entry_type,
        key=key,
        title=title,
        author=authors_str,
        year=year,
        journal=journal_ref,
        doi=doi,
        url=url or f"https://arxiv.org/abs/{arxiv_id}",
        eprint=arxiv_id,
        archiveprefix="arXiv",
        primaryclass=categories[0] if categories else None,
        abstract=abstract,
        source_id=f"arxiv:{arxiv_id}",
    )


def create_url_entry(
    url: str,
    title: str,
    author: str | None = None,
    year: int | str | None = None,
    accessed_date: str | None = None,
    note: str | None = None,
) -> BibTeXEntry:
    """Create BibTeX entry for a web resource.

    Args:
        url: Resource URL
        title: Page/resource title
        author: Author if known
        year: Year if known
        accessed_date: Date the URL was accessed
        note: Additional notes

    Returns:
        BibTeXEntry for the web resource
    """
    # Use current year if not provided
    if year is None:
        year = datetime.now().year

    # Generate key from URL hash if no author
    if author:
        key = generate_citation_key(author, year)
    else:
        # Create key from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]  # noqa: S324
        key = f"web{year}{url_hash}"
        author = "Unknown"

    # Add access date to note
    full_note = note or ""
    if accessed_date:
        access_note = f"Accessed: {accessed_date}"
        full_note = f"{full_note}. {access_note}" if full_note else access_note

    return BibTeXEntry(
        entry_type=EntryType.MISC,
        key=key,
        title=title,
        author=author,
        year=year,
        url=url,
        howpublished=f"\\url{{{url}}}",
        note=full_note if full_note else None,
        source_id=f"url:{url}",
    )


def create_article_entry(
    title: str,
    authors: list[str],
    year: int,
    journal: str,
    volume: str | None = None,
    number: str | None = None,
    pages: str | None = None,
    doi: str | None = None,
    url: str | None = None,
) -> BibTeXEntry:
    """Create BibTeX entry for a journal article.

    Args:
        title: Article title
        authors: List of author names
        year: Publication year
        journal: Journal name
        volume: Volume number
        number: Issue number
        pages: Page range
        doi: Digital Object Identifier
        url: Article URL

    Returns:
        BibTeXEntry for the journal article
    """
    key = generate_citation_key(authors, year, title)
    authors_str = " and ".join(authors)

    return BibTeXEntry(
        entry_type=EntryType.ARTICLE,
        key=key,
        title=title,
        author=authors_str,
        year=year,
        journal=journal,
        volume=volume,
        number=number,
        pages=pages,
        doi=doi,
        url=url,
        source_id=f"doi:{doi}" if doi else None,
    )


def create_inproceedings_entry(
    title: str,
    authors: list[str],
    year: int,
    booktitle: str,
    pages: str | None = None,
    publisher: str | None = None,
    address: str | None = None,
    doi: str | None = None,
    url: str | None = None,
) -> BibTeXEntry:
    """Create BibTeX entry for a conference paper.

    Args:
        title: Paper title
        authors: List of author names
        year: Publication year
        booktitle: Conference proceedings name
        pages: Page range
        publisher: Publisher name
        address: Conference location
        doi: Digital Object Identifier
        url: Paper URL

    Returns:
        BibTeXEntry for the conference paper
    """
    key = generate_citation_key(authors, year, title)
    authors_str = " and ".join(authors)

    return BibTeXEntry(
        entry_type=EntryType.INPROCEEDINGS,
        key=key,
        title=title,
        author=authors_str,
        year=year,
        booktitle=booktitle,
        pages=pages,
        publisher=publisher,
        address=address,
        doi=doi,
        url=url,
        source_id=f"doi:{doi}" if doi else None,
    )


# =============================================================================
# Validation
# =============================================================================


def validate_bibtex_entry(entry: BibTeXEntry) -> list[str]:
    """Validate a BibTeX entry for common issues.

    Args:
        entry: Entry to validate

    Returns:
        List of validation warnings/errors (empty if valid)
    """
    issues: list[str] = []

    # Check required fields
    if not entry.key:
        issues.append("Missing citation key")
    if not entry.title:
        issues.append("Missing title")
    if not entry.author:
        issues.append("Missing author")
    if not entry.year:
        issues.append("Missing year")

    # Check key format
    if entry.key and not re.match(r"^[a-zA-Z][a-zA-Z0-9_:-]*$", entry.key):
        issues.append(f"Invalid key format: {entry.key}")

    # Check year is reasonable
    if entry.year:
        try:
            year_int = int(entry.year)
            current_year = datetime.now().year
            if year_int < 1900 or year_int > current_year + 1:
                issues.append(f"Suspicious year: {entry.year}")
        except ValueError:
            issues.append(f"Invalid year format: {entry.year}")

    # Type-specific checks
    if entry.entry_type == EntryType.ARTICLE and not entry.journal:
        issues.append("Article entry missing journal")

    if entry.entry_type == EntryType.INPROCEEDINGS and not entry.booktitle:
        issues.append("Inproceedings entry missing booktitle")

    return issues


# =============================================================================
# Parsing (using bibtexparser if available)
# =============================================================================


def parse_bibtex(bibtex_str: str) -> list[BibTeXEntry]:
    """Parse BibTeX string into entries.

    Uses bibtexparser library for robust parsing.

    Args:
        bibtex_str: BibTeX format string

    Returns:
        List of parsed BibTeXEntry objects

    Raises:
        CitationError: If parsing fails
    """
    try:
        import bibtexparser
        from bibtexparser import Library
        from bibtexparser.middlewares.names import SeparateCoAuthors

        # Parse with bibtexparser v2
        library: Library = bibtexparser.parse_string(
            bibtex_str,
            append_middleware=[SeparateCoAuthors()],
        )

        entries: list[BibTeXEntry] = []
        for bib_entry in library.entries:
            # Map entry type
            entry_type_str = bib_entry.entry_type.lower()
            try:
                entry_type = EntryType(entry_type_str)
            except ValueError:
                entry_type = EntryType.MISC

            # Extract authors
            authors_field = bib_entry.fields_dict.get("author")
            if authors_field:
                # Handle list of authors from SeparateCoAuthors middleware
                if isinstance(authors_field.value, list):
                    authors_str = " and ".join(str(a) for a in authors_field.value)
                else:
                    authors_str = str(authors_field.value)
            else:
                authors_str = "Unknown"

            # Build entry
            entry = BibTeXEntry(
                entry_type=entry_type,
                key=bib_entry.key,
                title=_get_field(bib_entry, "title") or "Untitled",
                author=authors_str,
                year=_get_field(bib_entry, "year") or "",
                journal=_get_field(bib_entry, "journal"),
                booktitle=_get_field(bib_entry, "booktitle"),
                volume=_get_field(bib_entry, "volume"),
                number=_get_field(bib_entry, "number"),
                pages=_get_field(bib_entry, "pages"),
                month=_get_field(bib_entry, "month"),
                publisher=_get_field(bib_entry, "publisher"),
                address=_get_field(bib_entry, "address"),
                doi=_get_field(bib_entry, "doi"),
                url=_get_field(bib_entry, "url"),
                eprint=_get_field(bib_entry, "eprint"),
                archiveprefix=_get_field(bib_entry, "archiveprefix"),
                primaryclass=_get_field(bib_entry, "primaryclass"),
                abstract=_get_field(bib_entry, "abstract"),
                note=_get_field(bib_entry, "note"),
                howpublished=_get_field(bib_entry, "howpublished"),
            )
            entries.append(entry)

        return entries

    except ImportError:
        logger.warning("bibtexparser not available, using fallback parser")
        return _parse_bibtex_fallback(bibtex_str)
    except Exception as e:
        raise CitationError(f"Failed to parse BibTeX: {e}") from e


def _get_field(entry: Any, field_name: str, default: str | None = None) -> str | None:
    """Get field value from bibtexparser entry."""
    field = entry.fields_dict.get(field_name)
    if field is not None:
        return str(field.value)
    return default


def _parse_bibtex_fallback(bibtex_str: str) -> list[BibTeXEntry]:
    """Simple fallback BibTeX parser.

    Basic regex-based parser for when bibtexparser is not available.
    Handles common cases but may miss edge cases.
    """
    entries: list[BibTeXEntry] = []

    # Match entry blocks
    entry_pattern = r"@(\w+)\s*\{\s*([^,]+)\s*,(.*?)\n\}"
    matches = re.findall(entry_pattern, bibtex_str, re.DOTALL)

    for entry_type_str, key, fields_str in matches:
        try:
            entry_type = EntryType(entry_type_str.lower())
        except ValueError:
            entry_type = EntryType.MISC

        # Parse fields
        fields: dict[str, str] = {}
        field_pattern = r"(\w+)\s*=\s*\{([^}]*)\}"
        for field_name, field_value in re.findall(field_pattern, fields_str):
            fields[field_name.lower()] = field_value.strip()

        entry = BibTeXEntry(
            entry_type=entry_type,
            key=key.strip(),
            title=fields.get("title", "Untitled"),
            author=fields.get("author", "Unknown"),
            year=fields.get("year", ""),
            journal=fields.get("journal"),
            booktitle=fields.get("booktitle"),
            volume=fields.get("volume"),
            number=fields.get("number"),
            pages=fields.get("pages"),
            doi=fields.get("doi"),
            url=fields.get("url"),
            eprint=fields.get("eprint"),
            archiveprefix=fields.get("archiveprefix"),
            primaryclass=fields.get("primaryclass"),
            abstract=fields.get("abstract"),
            note=fields.get("note"),
        )
        entries.append(entry)

    return entries
