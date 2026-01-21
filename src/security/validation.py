"""Input validation utilities for API security.

Provides comprehensive validation for user inputs including:
- Query sanitization
- Character whitelisting
- Length limits
- Pattern detection for malicious inputs
- Domain validation
"""

import re
from dataclasses import dataclass
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    sanitized_value: str | None
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "sanitized_value": self.sanitized_value,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class InputValidator:
    """Validates and sanitizes user inputs."""

    # Allowed characters for research queries (Unicode-aware)
    ALLOWED_QUERY_PATTERN = re.compile(
        r"^[\w\s\-.,;:?!()'\"\[\]{}@#$%&*+=/<>~`^|\\]+$",
        re.UNICODE,
    )

    # Patterns that might indicate injection attempts
    # Note: Designed to minimize false positives on legitimate research queries
    SUSPICIOUS_PATTERNS = [
        # SQL injection patterns - require context (not just keywords alone)
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\s+(FROM|INTO|TABLE|ALL)\b)",
        r"(--\s*$|;\s*--)",  # SQL comment at end
        r"('\s*OR\s+'|'\s*AND\s+')",  # SQL injection via quotes
        # Command injection patterns
        r"(\$\([^)]+\)|`[^`]+`)",  # Command substitution with content
        r"(\|\s*\w+|\w+\s*\|)",  # Pipe with commands
        r"(&&\s*\w+|\w+\s*&&)",  # AND with commands
        r"(\\x[0-9a-fA-F]{2}){3,}",  # Multiple hex escape sequences
        # Path traversal
        r"(\.\./|\.\.\\){2,}",  # Multiple levels of traversal
        # Script injection
        r"(<script[\s>]|javascript:\s*\w|on\w+\s*=\s*[\"'])",
    ]

    # Compiled suspicious patterns
    _suspicious_compiled: list[re.Pattern[str]] | None = None

    # Length limits
    MIN_QUERY_LENGTH = 10
    MAX_QUERY_LENGTH = 2000
    MAX_DOMAIN_COUNT = 4

    # Valid domains
    VALID_DOMAINS = {"ai_ml", "quantum_physics", "astrophysics", "general"}

    @classmethod
    def _get_suspicious_patterns(cls) -> list[re.Pattern[str]]:
        """Get compiled suspicious patterns (lazy initialization)."""
        if cls._suspicious_compiled is None:
            cls._suspicious_compiled = [
                re.compile(pattern, re.IGNORECASE) for pattern in cls.SUSPICIOUS_PATTERNS
            ]
        return cls._suspicious_compiled

    @classmethod
    def validate_query(cls, query: str | None) -> ValidationResult:
        """Validate and sanitize a research query.

        Args:
            query: The research query to validate

        Returns:
            ValidationResult with validation status and sanitized value
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check for None or empty
        if query is None:
            return ValidationResult(
                is_valid=False,
                sanitized_value=None,
                errors=["Query is required"],
                warnings=[],
            )

        # Strip whitespace
        sanitized = query.strip()

        # Check length
        if len(sanitized) < cls.MIN_QUERY_LENGTH:
            errors.append(f"Query must be at least {cls.MIN_QUERY_LENGTH} characters")

        if len(sanitized) > cls.MAX_QUERY_LENGTH:
            errors.append(f"Query must be at most {cls.MAX_QUERY_LENGTH} characters")
            sanitized = sanitized[: cls.MAX_QUERY_LENGTH]

        # Check for suspicious patterns - detect but don't remove (prevents bypass)
        suspicious_found = False
        for pattern in cls._get_suspicious_patterns():
            if pattern.search(sanitized):
                logger.warning(
                    "Suspicious pattern detected in query",
                    pattern=pattern.pattern,
                    query_preview=sanitized[:50],
                )
                suspicious_found = True

        if suspicious_found:
            warnings.append("Query contains potentially suspicious characters")

        # Normalize whitespace
        sanitized = " ".join(sanitized.split())

        # Check if query is mostly valid characters
        if sanitized and not cls.ALLOWED_QUERY_PATTERN.match(sanitized):
            # Allow the query but warn
            warnings.append("Query contains unusual characters")

        # Check for very short sanitized result
        if len(sanitized) < cls.MIN_QUERY_LENGTH:
            errors.append("Query is too short after sanitization")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized if sanitized else None,
            errors=errors,
            warnings=warnings,
        )

    @classmethod
    def validate_domains(cls, domains: list[str] | None) -> ValidationResult:
        """Validate research domains.

        Args:
            domains: List of domain strings

        Returns:
            ValidationResult with validation status
        """
        errors: list[str] = []
        warnings: list[str] = []

        if domains is None:
            return ValidationResult(
                is_valid=True,
                sanitized_value=None,
                errors=[],
                warnings=[],
            )

        if not isinstance(domains, list):
            return ValidationResult(
                is_valid=False,
                sanitized_value=None,
                errors=["Domains must be a list"],
                warnings=[],
            )

        if len(domains) > cls.MAX_DOMAIN_COUNT:
            errors.append(f"Maximum {cls.MAX_DOMAIN_COUNT} domains allowed")

        # Validate each domain
        valid_domains: list[str] = []
        for domain in domains:
            if not isinstance(domain, str):
                errors.append(f"Invalid domain type: {type(domain).__name__}")
                continue

            domain_lower = domain.lower().strip()
            if domain_lower in cls.VALID_DOMAINS:
                valid_domains.append(domain_lower)
            else:
                errors.append(
                    f"Invalid domain: {domain}. Valid options: {cls.VALID_DOMAINS}"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=",".join(valid_domains) if valid_domains else None,
            errors=errors,
            warnings=warnings,
        )

    @classmethod
    def validate_max_iterations(cls, value: int | None) -> ValidationResult:
        """Validate max_iterations parameter.

        Args:
            value: The max_iterations value

        Returns:
            ValidationResult with validation status
        """
        errors: list[str] = []

        if value is None:
            return ValidationResult(
                is_valid=True,
                sanitized_value="3",  # Default
                errors=[],
                warnings=[],
            )

        if not isinstance(value, int):
            return ValidationResult(
                is_valid=False,
                sanitized_value=None,
                errors=["max_iterations must be an integer"],
                warnings=[],
            )

        if value < 1:
            errors.append("max_iterations must be at least 1")
        elif value > 5:
            errors.append("max_iterations must be at most 5")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=str(value) if errors == [] else None,
            errors=errors,
            warnings=[],
        )


def sanitize_for_logging(value: str, max_length: int = 100) -> str:
    """Sanitize a value for safe logging.

    Removes or escapes characters that could cause log injection.

    Args:
        value: The value to sanitize
        max_length: Maximum length of output

    Returns:
        Sanitized string safe for logging
    """
    if not value:
        return ""

    # Remove control characters
    sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)

    # Truncate if needed
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized


def is_safe_url(url: str) -> bool:
    """Check if a URL is safe to fetch.

    Defends against SSRF by blocking internal/private IPs.

    Args:
        url: The URL to check

    Returns:
        True if the URL appears safe
    """
    if not url:
        return False

    # Must start with http:// or https://
    if not url.startswith(("http://", "https://")):
        return False

    # Decode URL encoding to prevent bypass
    from urllib.parse import unquote, urlparse

    try:
        decoded_url = unquote(unquote(url))  # Double decode for nested encoding
        parsed = urlparse(decoded_url)
        host = parsed.hostname or ""
        host_lower = host.lower()
    except (ValueError, AttributeError):
        return False

    # Block localhost and internal IPs
    blocked_patterns = [
        r"^localhost$",
        r"^127\.",
        r"^0\.0\.0\.0$",
        r"^10\.",
        r"^172\.(1[6-9]|2\d|3[01])\.",
        r"^192\.168\.",
        r"^169\.254\.",  # Link-local
        r"^\[?::1\]?$",  # IPv6 localhost
        r"^\[?0:0:0:0:0:0:0:1\]?$",  # IPv6 localhost expanded
        r"^\[?fe80:",  # IPv6 link-local
        r"^\[?fc00:",  # IPv6 unique local
        r"^\[?fd00:",  # IPv6 unique local
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, host_lower, re.IGNORECASE):
            return False

    # Block numeric IP that could be octal/hex encoded
    # e.g., 0x7f.0.0.1, 017700000001
    if re.match(r"^[\d.x]+$", host_lower, re.IGNORECASE):
        # Suspicious numeric-only hostname, validate it's a real public IP
        parts = host.split(".")
        if len(parts) == 4:
            try:
                octets = [int(p, 0) for p in parts]  # int(x, 0) handles 0x and 0o
                if octets[0] in (0, 10, 127) or (octets[0] == 172 and 16 <= octets[1] <= 31):
                    return False
                if octets[0] == 192 and octets[1] == 168:
                    return False
                if octets[0] == 169 and octets[1] == 254:
                    return False
            except (ValueError, IndexError):
                return False

    return True


def validate_file_path_safe(path: str, allowed_base: str) -> bool:
    """Validate that a file path is within an allowed directory.

    Args:
        path: The path to validate
        allowed_base: The base directory that paths must be within

    Returns:
        True if the path is safe
    """
    import os

    try:
        # Resolve to absolute path
        resolved = os.path.realpath(path)
        base_resolved = os.path.realpath(allowed_base)

        # Check if resolved path starts with the allowed base
        return resolved.startswith(base_resolved + os.sep) or resolved == base_resolved
    except (OSError, ValueError):
        return False
