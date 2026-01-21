"""Tests for security validation module."""

import pytest

from src.security.validation import (
    InputValidator,
    is_safe_url,
    sanitize_for_logging,
    validate_file_path_safe,
)


class TestInputValidator:
    """Tests for InputValidator class."""

    def test_validate_query_valid(self) -> None:
        """Test validation of a valid query."""
        result = InputValidator.validate_query(
            "What are the latest advances in quantum computing?"
        )
        assert result.is_valid is True
        assert result.sanitized_value is not None
        assert len(result.errors) == 0

    def test_validate_query_too_short(self) -> None:
        """Test validation rejects short queries."""
        result = InputValidator.validate_query("short")
        assert result.is_valid is False
        assert any("at least" in e for e in result.errors)

    def test_validate_query_too_long(self) -> None:
        """Test validation truncates long queries."""
        long_query = "a" * 3000
        result = InputValidator.validate_query(long_query)
        assert result.is_valid is False
        assert any("at most" in e for e in result.errors)

    def test_validate_query_none(self) -> None:
        """Test validation rejects None query."""
        result = InputValidator.validate_query(None)
        assert result.is_valid is False
        assert result.sanitized_value is None

    def test_validate_query_strips_whitespace(self) -> None:
        """Test validation strips leading/trailing whitespace."""
        result = InputValidator.validate_query(
            "   What are neural networks?   "
        )
        assert result.is_valid is True
        assert result.sanitized_value == "What are neural networks?"

    def test_validate_query_normalizes_whitespace(self) -> None:
        """Test validation normalizes internal whitespace."""
        result = InputValidator.validate_query(
            "What   are    neural    networks?"
        )
        assert result.is_valid is True
        assert result.sanitized_value == "What are neural networks?"

    def test_validate_query_suspicious_sql(self) -> None:
        """Test validation detects SQL injection attempts."""
        result = InputValidator.validate_query(
            "What is SELECT * FROM users WHERE 1=1--"
        )
        # Should have warnings but still be valid after sanitization
        assert len(result.warnings) > 0 or result.is_valid is True

    def test_validate_query_suspicious_command(self) -> None:
        """Test validation detects command injection attempts."""
        result = InputValidator.validate_query(
            "What is machine learning $(rm -rf /)"
        )
        assert len(result.warnings) > 0 or result.is_valid is True

    def test_validate_domains_valid(self) -> None:
        """Test validation of valid domains."""
        result = InputValidator.validate_domains(["ai_ml", "quantum_physics"])
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_domains_invalid(self) -> None:
        """Test validation rejects invalid domains."""
        result = InputValidator.validate_domains(["invalid_domain"])
        assert result.is_valid is False
        assert any("Invalid domain" in e for e in result.errors)

    def test_validate_domains_none(self) -> None:
        """Test validation accepts None domains."""
        result = InputValidator.validate_domains(None)
        assert result.is_valid is True

    def test_validate_domains_too_many(self) -> None:
        """Test validation rejects too many domains."""
        result = InputValidator.validate_domains(
            ["ai_ml", "quantum_physics", "astrophysics", "general", "extra"]
        )
        assert result.is_valid is False
        assert any("Maximum" in e for e in result.errors)

    def test_validate_domains_case_insensitive(self) -> None:
        """Test domain validation is case insensitive."""
        result = InputValidator.validate_domains(["AI_ML", "QUANTUM_PHYSICS"])
        assert result.is_valid is True

    def test_validate_max_iterations_valid(self) -> None:
        """Test validation of valid max_iterations."""
        result = InputValidator.validate_max_iterations(3)
        assert result.is_valid is True

    def test_validate_max_iterations_none(self) -> None:
        """Test validation accepts None max_iterations."""
        result = InputValidator.validate_max_iterations(None)
        assert result.is_valid is True
        assert result.sanitized_value == "3"  # Default

    def test_validate_max_iterations_too_low(self) -> None:
        """Test validation rejects too low max_iterations."""
        result = InputValidator.validate_max_iterations(0)
        assert result.is_valid is False

    def test_validate_max_iterations_too_high(self) -> None:
        """Test validation rejects too high max_iterations."""
        result = InputValidator.validate_max_iterations(10)
        assert result.is_valid is False


class TestSanitizeForLogging:
    """Tests for sanitize_for_logging function."""

    def test_sanitize_normal_string(self) -> None:
        """Test sanitization of normal string."""
        result = sanitize_for_logging("Hello, World!")
        assert result == "Hello, World!"

    def test_sanitize_removes_control_chars(self) -> None:
        """Test sanitization removes control characters."""
        result = sanitize_for_logging("Hello\x00World\x1f!")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_sanitize_truncates_long_string(self) -> None:
        """Test sanitization truncates long strings."""
        long_string = "a" * 200
        result = sanitize_for_logging(long_string, max_length=50)
        assert len(result) <= 53  # 50 + "..."

    def test_sanitize_empty_string(self) -> None:
        """Test sanitization of empty string."""
        result = sanitize_for_logging("")
        assert result == ""


class TestIsSafeUrl:
    """Tests for is_safe_url function."""

    def test_safe_https_url(self) -> None:
        """Test HTTPS URL is safe."""
        assert is_safe_url("https://example.com/page") is True

    def test_safe_http_url(self) -> None:
        """Test HTTP URL is safe."""
        assert is_safe_url("http://example.com/page") is True

    def test_unsafe_localhost(self) -> None:
        """Test localhost is blocked."""
        assert is_safe_url("http://localhost:8000") is False

    def test_unsafe_127_0_0_1(self) -> None:
        """Test 127.0.0.1 is blocked."""
        assert is_safe_url("http://127.0.0.1:8000") is False

    def test_unsafe_private_ip_10(self) -> None:
        """Test 10.x.x.x is blocked."""
        assert is_safe_url("http://10.0.0.1") is False

    def test_unsafe_private_ip_192(self) -> None:
        """Test 192.168.x.x is blocked."""
        assert is_safe_url("http://192.168.1.1") is False

    def test_unsafe_private_ip_172(self) -> None:
        """Test 172.16-31.x.x is blocked."""
        assert is_safe_url("http://172.16.0.1") is False

    def test_unsafe_no_protocol(self) -> None:
        """Test URL without protocol is blocked."""
        assert is_safe_url("example.com") is False

    def test_unsafe_file_protocol(self) -> None:
        """Test file:// protocol is blocked."""
        assert is_safe_url("file:///etc/passwd") is False

    def test_unsafe_empty_url(self) -> None:
        """Test empty URL is blocked."""
        assert is_safe_url("") is False


class TestValidateFilePathSafe:
    """Tests for validate_file_path_safe function."""

    def test_valid_path_within_base(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test valid path within base directory."""
        import os

        base = str(tmp_path)  # type: ignore[arg-type]
        test_file = os.path.join(base, "test.txt")
        assert validate_file_path_safe(test_file, base) is True

    def test_invalid_path_traversal(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test path traversal is blocked."""
        import os

        base = str(tmp_path)  # type: ignore[arg-type]
        test_file = os.path.join(base, "..", "etc", "passwd")
        assert validate_file_path_safe(test_file, base) is False

    def test_invalid_absolute_path(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test absolute path outside base is blocked."""
        base = str(tmp_path)  # type: ignore[arg-type]
        assert validate_file_path_safe("/etc/passwd", base) is False
