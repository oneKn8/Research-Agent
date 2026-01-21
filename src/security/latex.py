"""LaTeX security utilities.

Provides comprehensive sanitization and validation for LaTeX content
to prevent injection attacks and ensure safe compilation.

Security considerations based on research:
- LaTeX can execute shell commands via \\write18 (shell-escape)
- File operations (\\input, \\include) can read sensitive files
- Catcode manipulation can bypass restrictions
- LuaTeX has direct Lua execution capabilities
- Infinite loops can cause DoS (must use compilation timeout)

References:
- PayloadsAllTheThings LaTeX Injection: https://github.com/swisskyrepo/PayloadsAllTheThings
- "Don't take LaTeX files from strangers": https://hovav.net/ucsd/dist/tex-login.pdf
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SanitizationResult:
    """Result of content sanitization."""

    content: str
    is_safe: bool
    warnings: list[str]
    removed_commands: list[str]


# =============================================================================
# Dangerous Command Patterns
# =============================================================================

# Commands that can execute shell code
SHELL_ESCAPE_PATTERNS = [
    r"\\write18\s*\{",
    r"\\immediate\s*\\write18",
    r"\\ShellEscape\s*\{",
    r"\\usepackage\[shell\-escape\]",
    r"\\usepackage\{\s*shellesc\s*\}",
    r"\\usepackage\{\s*epstopdf\s*\}.*\\epstopdfsetup\{.*shell\-escape",
]

# Commands that can read/write files
FILE_OPERATION_PATTERNS = [
    r"\\input\s*\{",
    r"\\include\s*\{",
    r"\\openin\s*\d*\s*=?",
    r"\\openout\s*\d*\s*=?",
    r"\\read\s*\d+",
    r"\\write\s*\d+",
    r"\\closein\s*\d*",
    r"\\closeout\s*\d*",
    r"\\InputIfFileExists\s*\{",
    r"\\IfFileExists\s*\{",
    r"\\@@input\s*\{",
    r"\\@input\s*\{",
    r"\\verbatiminput\s*\{",
    r"\\lstinputlisting\s*\{",
    r"\\lstinputlisting\s*\[",
]

# Commands that can manipulate categories (bypass restrictions)
CATCODE_PATTERNS = [
    r"\\catcode\s*[`']",
    r"\\catcode\s*\d+\s*=",
    r"\\@makeother",
]

# Commands for direct code execution (LuaTeX, PythonTeX, etc.)
CODE_EXECUTION_PATTERNS = [
    r"\\directlua\s*\{",
    r"\\luacode",
    r"\\luadirect\s*\{",
    r"\\luatexbase",
    r"\\py\s*\{",
    r"\\pyc\s*\{",
    r"\\pyv\s*\{",
    r"\\pyb\s*\{",
    r"\\pys\s*\{",
    r"\\pyblock",
    r"\\begin\{pycode\}",
    r"\\begin\{pyverbatim\}",
]

# Special commands that can have security implications
SPECIAL_PATTERNS = [
    r"\\special\s*\{",
    r"\\pdfprimitive",
    r"\\primitive",
]

# Patterns that could indicate path traversal
PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",  # Parent directory
    r"\.\.\\",  # Windows parent
    r"/etc/",  # Unix system files
    r"/proc/",  # Linux proc
    r"/dev/",  # Device files
    r"C:\\",  # Windows root
    r"~",  # Home directory expansion
]


# =============================================================================
# Sanitization Functions
# =============================================================================


def sanitize_latex(
    content: str,
    allow_math: bool = True,
    allow_formatting: bool = True,
    strict: bool = False,
) -> SanitizationResult:
    """Sanitize LaTeX content for safe compilation.

    Args:
        content: Raw LaTeX content
        allow_math: Allow math environments and commands
        allow_formatting: Allow formatting commands (bold, italic, etc.)
        strict: If True, remove ALL commands not in whitelist

    Returns:
        SanitizationResult with sanitized content and metadata
    """
    warnings: list[str] = []
    removed_commands: list[str] = []
    is_safe = True

    if not content:
        return SanitizationResult(
            content="",
            is_safe=True,
            warnings=[],
            removed_commands=[],
        )

    # Check and remove shell escape commands
    for pattern in SHELL_ESCAPE_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            is_safe = False
            removed_commands.extend(matches)
            warnings.append(f"Removed shell escape command: {pattern}")
            content = re.sub(pattern + r"[^}]*\}", "[REMOVED-SHELL-ESCAPE]", content, flags=re.IGNORECASE)

    # Check and remove file operations
    for pattern in FILE_OPERATION_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            is_safe = False
            removed_commands.extend(matches)
            warnings.append(f"Removed file operation: {pattern}")
            # Remove the full command
            content = re.sub(pattern + r"[^}]*\}", "[REMOVED-FILE-OP]", content, flags=re.IGNORECASE)
            # Also remove commands without braces (like \read12)
            content = re.sub(pattern + r"[^\s\\]*", "[REMOVED-FILE-OP]", content, flags=re.IGNORECASE)

    # Check and remove catcode manipulation
    for pattern in CATCODE_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            is_safe = False
            removed_commands.extend(matches)
            warnings.append(f"Removed catcode manipulation: {pattern}")
            content = re.sub(pattern + r"[^\n]*", "[REMOVED-CATCODE]", content)

    # Check and remove code execution
    for pattern in CODE_EXECUTION_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            is_safe = False
            removed_commands.extend(matches)
            warnings.append(f"Removed code execution: {pattern}")
            content = re.sub(pattern + r"[^}]*\}", "[REMOVED-CODE-EXEC]", content, flags=re.IGNORECASE)

    # Check and remove special commands
    for pattern in SPECIAL_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            warnings.append(f"Removed special command: {pattern}")
            removed_commands.extend(matches)
            content = re.sub(pattern + r"[^}]*\}", "[REMOVED-SPECIAL]", content, flags=re.IGNORECASE)

    # Check for path traversal
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if re.search(pattern, content):
            warnings.append(f"Potential path traversal detected: {pattern}")
            is_safe = False
            content = re.sub(pattern, "[REMOVED-PATH]", content)

    # Remove backticks (potential shell execution in some contexts)
    if "`" in content:
        warnings.append("Removed backticks (potential shell execution)")
        content = content.replace("`", "'")

    # Strict mode: only allow whitelisted commands
    if strict:
        content, strict_removed = _apply_strict_whitelist(content, allow_math, allow_formatting)
        removed_commands.extend(strict_removed)
        if strict_removed:
            warnings.append(f"Strict mode removed {len(strict_removed)} non-whitelisted commands")

    if removed_commands:
        logger.warning(
            "Removed dangerous LaTeX commands",
            count=len(removed_commands),
            commands=removed_commands[:10],  # Log first 10
        )

    return SanitizationResult(
        content=content,
        is_safe=is_safe,
        warnings=warnings,
        removed_commands=removed_commands,
    )


def _apply_strict_whitelist(
    content: str,
    allow_math: bool,
    allow_formatting: bool,
) -> tuple[str, list[str]]:
    """Apply strict whitelist filtering.

    Returns:
        Tuple of (filtered content, removed commands)
    """
    # Whitelist of safe commands
    safe_commands = {
        # Document structure
        "documentclass", "usepackage", "begin", "end",
        "section", "subsection", "subsubsection", "paragraph",
        "chapter", "part", "maketitle", "title", "author", "date",
        "tableofcontents", "listoffigures", "listoftables",
        "appendix", "abstract", "keywords",
        # References
        "label", "ref", "pageref", "cite", "citep", "citet",
        "bibliography", "bibliographystyle",
        # Figures and tables
        "includegraphics", "caption", "table", "figure",
        "tabular", "hline", "toprule", "midrule", "bottomrule",
        # Lists
        "item", "enumerate", "itemize", "description",
        # Misc
        "newpage", "clearpage", "vspace", "hspace",
        "footnote", "thanks", "url", "href",
    }

    if allow_math:
        safe_commands.update({
            "frac", "sqrt", "sum", "prod", "int", "oint",
            "lim", "infty", "partial", "nabla",
            "sin", "cos", "tan", "log", "ln", "exp",
            "alpha", "beta", "gamma", "delta", "epsilon",
            "theta", "lambda", "mu", "nu", "pi", "sigma",
            "omega", "phi", "psi", "xi", "rho", "tau",
            "vec", "hat", "bar", "dot", "ddot",
            "left", "right", "cdot", "times", "div",
            "leq", "geq", "neq", "approx", "equiv",
            "subset", "supset", "in", "notin",
            "rightarrow", "leftarrow", "Rightarrow", "Leftarrow",
            "text", "mathrm", "mathbf", "mathcal", "mathbb",
            "binom", "matrix", "pmatrix", "bmatrix",
            "equation", "align", "gather", "multline",
            "boxed", "cases",
        })

    if allow_formatting:
        safe_commands.update({
            "textbf", "textit", "texttt", "textsc", "textrm",
            "emph", "underline",
            "centering", "raggedright", "raggedleft",
            "small", "large", "Large", "LARGE", "huge", "Huge",
            "normalsize", "footnotesize", "scriptsize", "tiny",
        })

    removed: list[str] = []

    # Find all commands
    command_pattern = r"\\([a-zA-Z]+)"

    def filter_command(match: re.Match[str]) -> str:
        cmd = match.group(1)
        if cmd.lower() in {c.lower() for c in safe_commands}:
            return match.group(0)
        removed.append(cmd)
        return ""

    content = re.sub(command_pattern, filter_command, content)

    return content, removed


def validate_file_path(path: str | Path, allowed_dir: Path | None = None) -> bool:
    """Validate a file path for safety.

    Checks for path traversal attempts and ensures path
    is within allowed directory.

    Args:
        path: Path to validate
        allowed_dir: Directory paths must be within (if specified)

    Returns:
        True if path is safe, False otherwise
    """
    try:
        path = Path(path).resolve()

        # Check for path traversal patterns in string representation
        path_str = str(path)
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if pattern in path_str:
                logger.warning("Path traversal detected", path=path_str, pattern=pattern)
                return False

        # Check if within allowed directory
        if allowed_dir:
            allowed_dir = allowed_dir.resolve()
            try:
                path.relative_to(allowed_dir)
            except ValueError:
                logger.warning(
                    "Path outside allowed directory",
                    path=path_str,
                    allowed=str(allowed_dir),
                )
                return False

        return True

    except Exception as e:
        logger.error("Path validation error", path=str(path), error=str(e))
        return False


def escape_latex_special_chars(text: str) -> str:
    """Escape LaTeX special characters in plain text.

    Use this for user-provided text that should NOT contain
    LaTeX commands (like names, titles).

    Args:
        text: Plain text to escape

    Returns:
        Text safe for use in LaTeX
    """
    if not text:
        return ""

    # Use placeholder for backslash to avoid double-escaping
    placeholder = "\x00BACKSLASH\x00"
    text = text.replace("\\", placeholder)

    # Escape special characters
    special_chars = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("<", r"\textless{}"),
        (">", r"\textgreater{}"),
        ("|", r"\textbar{}"),
    ]

    for char, replacement in special_chars:
        text = text.replace(char, replacement)

    # Replace placeholder with proper LaTeX backslash
    text = text.replace(placeholder, r"\textbackslash{}")

    return text


def check_content_safety(content: str) -> dict[str, Any]:
    """Check content for potential security issues without modifying it.

    Args:
        content: LaTeX content to check

    Returns:
        Dictionary with safety analysis
    """
    issues: list[dict[str, Any]] = []
    risk_level = "low"

    # Check for shell escapes
    for pattern in SHELL_ESCAPE_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append({
                "type": "shell_escape",
                "severity": "critical",
                "matches": matches,
            })
            risk_level = "critical"

    # Check for file operations
    for pattern in FILE_OPERATION_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append({
                "type": "file_operation",
                "severity": "high",
                "matches": matches,
            })
            if risk_level != "critical":
                risk_level = "high"

    # Check for code execution
    for pattern in CODE_EXECUTION_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append({
                "type": "code_execution",
                "severity": "critical",
                "matches": matches,
            })
            risk_level = "critical"

    # Check for catcode manipulation
    for pattern in CATCODE_PATTERNS:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append({
                "type": "catcode_manipulation",
                "severity": "high",
                "matches": matches,
            })
            if risk_level not in ("critical", "high"):
                risk_level = "medium"

    # Check for path traversal
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if re.search(pattern, content):
            issues.append({
                "type": "path_traversal",
                "severity": "high",
                "pattern": pattern,
            })
            if risk_level != "critical":
                risk_level = "high"

    return {
        "is_safe": len(issues) == 0,
        "risk_level": risk_level,
        "issues": issues,
        "issue_count": len(issues),
    }
