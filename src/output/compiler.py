"""LaTeX paper compilation.

Handles compilation of LaTeX documents to PDF with proper
security measures and timeout handling.

Security measures:
- Shell-escape disabled by default
- No write18 execution
- Timeout on compilation to prevent DoS
- Output directory restrictions
- Non-root execution
"""

import asyncio
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from src.output.bibtex import BibTeXManager
from src.output.latex import LaTeXGenerator, Paper, PaperFormat, validate_latex_syntax
from src.security.latex import check_content_safety, sanitize_latex

logger = structlog.get_logger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = Path("outputs")

# Compilation settings
DEFAULT_TIMEOUT_SECONDS = 120  # 2 minutes
MAX_COMPILATION_PASSES = 3  # pdflatex runs for resolving references


class CompilationStatus(str, Enum):
    """Status of LaTeX compilation."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"  # When PDF compilation not requested


@dataclass
class CompilationResult:
    """Result of paper compilation."""

    status: CompilationStatus
    latex_path: Path | None = None
    bibtex_path: Path | None = None
    pdf_path: Path | None = None
    log_path: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "latex_path": str(self.latex_path) if self.latex_path else None,
            "bibtex_path": str(self.bibtex_path) if self.bibtex_path else None,
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "log_path": str(self.log_path) if self.log_path else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
        }


class PaperCompiler:
    """Compiles LaTeX papers to PDF.

    Handles the full compilation pipeline:
    1. Write .tex and .bib files
    2. Run pdflatex (multiple passes for references)
    3. Run bibtex if citations present
    4. Collect and report errors
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the compiler.

        Args:
            output_dir: Directory for output files (created if not exists)
            timeout_seconds: Timeout for compilation
        """
        self._output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout_seconds
        self._logger = structlog.get_logger(__name__)
        self._generator = LaTeXGenerator()

    @property
    def output_dir(self) -> Path:
        """Get the output directory."""
        return self._output_dir

    async def compile_paper(
        self,
        paper: Paper,
        bib_manager: BibTeXManager | None = None,
        format: PaperFormat = PaperFormat.ARTICLE,
        base_name: str = "paper",
        compile_pdf: bool = True,
    ) -> CompilationResult:
        """Compile a paper to LaTeX and optionally PDF.

        Args:
            paper: Paper object to compile
            bib_manager: Optional BibTeX manager with citations
            format: Paper format (article, report, minimal)
            base_name: Base name for output files (e.g., "paper" -> "paper.tex")
            compile_pdf: Whether to compile to PDF (requires pdflatex)

        Returns:
            CompilationResult with paths and status
        """
        import time

        start_time = time.time()
        errors: list[str] = []
        warnings: list[str] = []

        # Sanitize base name
        safe_base_name = self._sanitize_filename(base_name)

        # Create unique directory for this compilation
        compile_dir = self._output_dir / safe_base_name
        compile_dir.mkdir(parents=True, exist_ok=True)

        tex_path = compile_dir / f"{safe_base_name}.tex"
        bib_path = compile_dir / f"{safe_base_name}.bib" if bib_manager else None
        pdf_path = compile_dir / f"{safe_base_name}.pdf"
        log_path = compile_dir / f"{safe_base_name}.log"

        try:
            # Generate LaTeX content
            if bib_manager:
                paper.bibliography_file = safe_base_name

            latex_content = self._generator.render(paper, format=format, sanitize=True)

            # Validate syntax
            syntax_issues = validate_latex_syntax(latex_content)
            if syntax_issues:
                warnings.extend([f"Syntax warning: {issue}" for issue in syntax_issues])

            # Check content safety
            safety_check = check_content_safety(latex_content)
            if not safety_check["is_safe"]:
                warnings.append(f"Content safety issues detected: {safety_check['issues']}")
                # Re-sanitize if needed
                sanitize_result = sanitize_latex(latex_content)
                latex_content = sanitize_result.content
                if sanitize_result.removed_commands:
                    warnings.extend([
                        f"Removed unsafe command: {cmd}"
                        for cmd in sanitize_result.removed_commands[:5]
                    ])

            # Write LaTeX file
            tex_path.write_text(latex_content, encoding="utf-8")
            self._logger.info("Wrote LaTeX file", path=str(tex_path))

            # Write BibTeX file if citations provided
            if bib_manager and bib_manager.entries:
                bibtex_content = bib_manager.to_bibtex()
                if bib_path:
                    bib_path.write_text(bibtex_content, encoding="utf-8")
                    self._logger.info("Wrote BibTeX file", path=str(bib_path))

            # Compile to PDF if requested
            if compile_pdf:
                compile_result = await self._compile_latex(
                    tex_path=tex_path,
                    bib_path=bib_path,
                    compile_dir=compile_dir,
                )
                errors.extend(compile_result.get("errors", []))
                warnings.extend(compile_result.get("warnings", []))

                if compile_result["success"]:
                    if not pdf_path.exists():
                        errors.append("PDF file not created")
                else:
                    errors.extend(compile_result.get("errors", ["Compilation failed"]))

            duration = time.time() - start_time

            # Determine status
            if compile_pdf:
                if errors:
                    status = CompilationStatus.FAILED
                elif pdf_path.exists():
                    status = CompilationStatus.SUCCESS
                else:
                    status = CompilationStatus.FAILED
                    errors.append("PDF file was not created")
            else:
                status = CompilationStatus.SKIPPED

            return CompilationResult(
                status=status,
                latex_path=tex_path,
                bibtex_path=bib_path,
                pdf_path=pdf_path if compile_pdf and pdf_path.exists() else None,
                log_path=log_path if log_path.exists() else None,
                errors=errors,
                warnings=warnings,
                duration_seconds=duration,
            )

        except Exception as e:
            self._logger.error("Compilation error", error=str(e))
            return CompilationResult(
                status=CompilationStatus.FAILED,
                latex_path=tex_path if tex_path.exists() else None,
                errors=[str(e)],
                duration_seconds=time.time() - start_time,
            )

    async def compile_from_state(
        self,
        state: dict[str, Any],
        bib_manager: BibTeXManager | None = None,
        format: PaperFormat = PaperFormat.ARTICLE,
        compile_pdf: bool = True,
    ) -> CompilationResult:
        """Compile paper directly from workflow state.

        Args:
            state: ResearchState dictionary
            bib_manager: Optional BibTeX manager
            format: Paper format
            compile_pdf: Whether to compile to PDF

        Returns:
            CompilationResult
        """
        # Generate latex and bibtex from state (validation/logging only)
        _latex_content, _bibtex_content = self._generator.render_from_state(
            state, format=format, bib_manager=bib_manager
        )

        # Create paper name from title or thread_id
        title = state.get("title", "")
        thread_id = state.get("thread_id", "paper")
        base_name = self._sanitize_filename(title[:50]) if title else thread_id

        # Create paper object for compile_paper
        from src.output.latex import Paper, PaperMetadata

        metadata = PaperMetadata(
            title=state.get("title", "Untitled"),
            author="Research Agent",
        )
        paper = Paper(metadata=metadata, abstract=state.get("abstract"))

        # Add sections
        for name, content in state.get("sections", {}).items():
            paper.add_section(name, content)

        if bib_manager:
            paper.bibliography_file = base_name

        return await self.compile_paper(
            paper=paper,
            bib_manager=bib_manager,
            format=format,
            base_name=base_name,
            compile_pdf=compile_pdf,
        )

    async def _compile_latex(
        self,
        tex_path: Path,
        bib_path: Path | None,
        compile_dir: Path,
    ) -> dict[str, Any]:
        """Run pdflatex and bibtex compilation.

        Args:
            tex_path: Path to .tex file
            bib_path: Path to .bib file (optional)
            compile_dir: Working directory

        Returns:
            Dictionary with success, errors, warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check if pdflatex is available
        if not self._is_latex_available():
            return {
                "success": False,
                "errors": ["pdflatex not found. Install TeX Live or MiKTeX to compile PDFs."],
                "warnings": [],
            }

        try:
            # First pdflatex pass
            result = await self._run_pdflatex(tex_path, compile_dir)
            if not result["success"]:
                return result

            # Run bibtex if bibliography present
            if bib_path and bib_path.exists():
                bib_result = await self._run_bibtex(tex_path, compile_dir)
                if bib_result.get("warnings"):
                    warnings.extend(bib_result["warnings"])

            # Second and third pdflatex passes (resolve references)
            for _ in range(2):
                result = await self._run_pdflatex(tex_path, compile_dir)
                if not result["success"]:
                    errors.extend(result.get("errors", []))
                    # Continue anyway, PDF might still be generated

            # Check for common warnings in log
            log_path = tex_path.with_suffix(".log")
            if log_path.exists():
                log_warnings = self._parse_log_warnings(log_path)
                warnings.extend(log_warnings)

            return {
                "success": True,
                "errors": errors,
                "warnings": warnings,
            }

        except TimeoutError:
            return {
                "success": False,
                "errors": [f"Compilation timed out after {self._timeout} seconds"],
                "warnings": [],
            }
        except Exception as e:
            return {
                "success": False,
                "errors": [f"Compilation error: {e}"],
                "warnings": [],
            }

    async def _run_pdflatex(self, tex_path: Path, work_dir: Path) -> dict[str, Any]:
        r"""Run pdflatex with security restrictions.

        Security: --no-shell-escape prevents \write18 execution
        """
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",  # Don't stop on errors
            "-halt-on-error",  # Stop on first error
            "-no-shell-escape",  # SECURITY: Disable shell commands
            "-output-directory", str(work_dir),
            str(tex_path),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout,
            )

            if process.returncode != 0:
                # Parse error from output
                error_msg = self._parse_latex_errors(stdout.decode("utf-8", errors="replace"))
                return {
                    "success": False,
                    "errors": [error_msg] if error_msg else ["pdflatex failed"],
                    "warnings": [],
                }

            return {"success": True, "errors": [], "warnings": []}

        except TimeoutError:
            self._logger.error("pdflatex timeout")
            raise
        except FileNotFoundError:
            return {
                "success": False,
                "errors": ["pdflatex not found"],
                "warnings": [],
            }

    async def _run_bibtex(self, tex_path: Path, work_dir: Path) -> dict[str, Any]:
        """Run bibtex to process citations."""
        aux_path = tex_path.with_suffix(".aux")

        cmd = ["bibtex", str(aux_path.stem)]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60,  # BibTeX should be quick
            )

            warnings: list[str] = []
            if process.returncode != 0:
                output = stdout.decode("utf-8", errors="replace")
                if "Warning" in output:
                    warnings.append("BibTeX warnings detected")
                # BibTeX errors are often non-fatal

            return {"success": True, "errors": [], "warnings": warnings}

        except FileNotFoundError:
            return {
                "success": True,  # Continue without bibtex
                "errors": [],
                "warnings": ["bibtex not found, citations may not resolve"],
            }
        except TimeoutError:
            return {
                "success": True,  # Continue anyway
                "errors": [],
                "warnings": ["bibtex timed out"],
            }

    def _parse_latex_errors(self, output: str) -> str:
        """Extract meaningful error from pdflatex output."""
        lines = output.split("\n")

        # Look for error lines
        for i, line in enumerate(lines):
            if line.startswith("!"):
                # Found an error, get context
                error_lines = [line]
                for j in range(i + 1, min(i + 3, len(lines))):
                    error_lines.append(lines[j])
                return " ".join(error_lines)

        # Look for "LaTeX Error"
        for line in lines:
            if "LaTeX Error" in line:
                return line

        return "Unknown LaTeX error"

    def _parse_log_warnings(self, log_path: Path) -> list[str]:
        """Extract warnings from log file."""
        warnings: list[str] = []

        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")

            # Common warning patterns
            warning_patterns = [
                r"LaTeX Warning: Reference .* undefined",
                r"LaTeX Warning: Citation .* undefined",
                r"Overfull \\hbox",
                r"Underfull \\hbox",
            ]

            for pattern in warning_patterns:
                matches = re.findall(pattern, content)
                for match in matches[:3]:  # Limit to first 3 of each type
                    warnings.append(match)

        except Exception as e:
            self._logger.debug("Failed to parse log file", error=str(e))

        return warnings

    def _is_latex_available(self) -> bool:
        """Check if pdflatex is installed."""
        return shutil.which("pdflatex") is not None

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as filename."""
        # Remove/replace problematic characters
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"[\s]+", "_", name)
        name = name.strip("_")

        # Ensure not empty
        if not name:
            name = "paper"

        # Limit length
        if len(name) > 100:
            name = name[:100]

        return name.lower()

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up old compilation outputs.

        Args:
            max_age_hours: Maximum age of files to keep

        Returns:
            Number of directories removed
        """
        import time

        removed = 0
        cutoff_time = time.time() - (max_age_hours * 3600)

        for item in self._output_dir.iterdir():
            if item.is_dir():
                try:
                    # Check modification time
                    mtime = item.stat().st_mtime
                    if mtime < cutoff_time:
                        shutil.rmtree(item)
                        removed += 1
                        self._logger.info("Removed old output", path=str(item))
                except Exception as e:
                    self._logger.warning("Failed to remove", path=str(item), error=str(e))

        return removed


# =============================================================================
# Convenience Functions
# =============================================================================


async def compile_paper(
    paper: Paper,
    bib_manager: BibTeXManager | None = None,
    format: PaperFormat = PaperFormat.ARTICLE,
    output_dir: Path | None = None,
    compile_pdf: bool = True,
) -> CompilationResult:
    """Quick paper compilation function.

    Args:
        paper: Paper to compile
        bib_manager: Optional BibTeX manager
        format: Output format
        output_dir: Output directory
        compile_pdf: Whether to compile PDF

    Returns:
        CompilationResult
    """
    compiler = PaperCompiler(output_dir=output_dir)
    return await compiler.compile_paper(
        paper=paper,
        bib_manager=bib_manager,
        format=format,
        compile_pdf=compile_pdf,
    )


async def compile_latex_string(
    latex_content: str,
    bibtex_content: str | None = None,
    base_name: str = "paper",
    output_dir: Path | None = None,
) -> CompilationResult:
    """Compile raw LaTeX string to PDF.

    Args:
        latex_content: LaTeX document content
        bibtex_content: Optional BibTeX content
        base_name: Base filename
        output_dir: Output directory

    Returns:
        CompilationResult
    """
    import time

    start_time = time.time()
    errors: list[str] = []
    warnings: list[str] = []

    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize the content
    safety_check = check_content_safety(latex_content)
    if not safety_check["is_safe"]:
        sanitize_result = sanitize_latex(latex_content)
        latex_content = sanitize_result.content
        warnings.extend(sanitize_result.warnings)

    # Validate syntax
    syntax_issues = validate_latex_syntax(latex_content)
    warnings.extend(syntax_issues)

    # Create compiler and use internal methods
    compiler = PaperCompiler(output_dir=output_dir)

    compile_dir = output_dir / compiler._sanitize_filename(base_name)
    compile_dir.mkdir(parents=True, exist_ok=True)

    tex_path = compile_dir / f"{base_name}.tex"
    bib_path = compile_dir / f"{base_name}.bib" if bibtex_content else None
    pdf_path = compile_dir / f"{base_name}.pdf"

    try:
        # Write files
        tex_path.write_text(latex_content, encoding="utf-8")
        if bibtex_content and bib_path:
            bib_path.write_text(bibtex_content, encoding="utf-8")

        # Compile
        result = await compiler._compile_latex(tex_path, bib_path, compile_dir)
        errors.extend(result.get("errors", []))
        warnings.extend(result.get("warnings", []))

        status = CompilationStatus.SUCCESS if result["success"] else CompilationStatus.FAILED

        return CompilationResult(
            status=status,
            latex_path=tex_path,
            bibtex_path=bib_path,
            pdf_path=pdf_path if pdf_path.exists() else None,
            errors=errors,
            warnings=warnings,
            duration_seconds=time.time() - start_time,
        )

    except Exception as e:
        return CompilationResult(
            status=CompilationStatus.FAILED,
            latex_path=tex_path if tex_path.exists() else None,
            errors=[str(e)],
            duration_seconds=time.time() - start_time,
        )
