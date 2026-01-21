"""Tests for configuration management."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError


class TestSettings:
    """Test cases for Settings configuration."""

    def test_settings_loads_from_environment(self, test_settings):
        """Test that settings load correctly from environment."""
        assert test_settings.environment == "development"
        assert test_settings.brain_service_url == "http://localhost:8001"
        assert test_settings.log_level == "DEBUG"

    def test_settings_is_development(self, test_settings):
        """Test is_development property."""
        assert test_settings.is_development is True
        assert test_settings.is_production is False

    def test_settings_default_values(self, test_settings):
        """Test default configuration values."""
        assert test_settings.api_host == "0.0.0.0"
        assert test_settings.api_port == 8000
        assert test_settings.rate_limit_requests == 60
        assert test_settings.max_search_workers == 5

    def test_settings_brain_config(self, test_settings):
        """Test brain service configuration."""
        assert test_settings.brain_temperature == 0.6
        assert test_settings.brain_top_p == 0.95
        assert test_settings.max_context_tokens == 128000

    def test_settings_missing_required_raises_error(self):
        """Test that missing required fields raise validation error."""
        with patch.dict(os.environ, {}, clear=True):
            from src.config import Settings

            with pytest.raises(ValidationError):
                # Use nonexistent .env file to prevent reading actual .env
                Settings(_env_file="/nonexistent/.env")

    def test_settings_output_dir_created(self, test_settings, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        from src.config import Settings

        test_output_dir = tmp_path / "test_outputs"

        with patch.dict(
            os.environ,
            {
                "OUTPUT_DIR": str(test_output_dir),
                "OPENAI_API_KEY": "sk-test",
                "TAVILY_API_KEY": "tvly-test",
                "BRAIN_API_KEY": "test",
            },
        ):
            settings = Settings()
            assert settings.output_dir.exists()


class TestGetSettings:
    """Test cases for get_settings function."""

    def test_get_settings_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        from src.config import get_settings

        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, "environment")
        assert hasattr(settings, "brain_service_url")

    def test_get_settings_is_cached(self):
        """Test that get_settings returns cached instance."""
        from src.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
