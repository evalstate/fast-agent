"""
Tests for model selection source detection logic.
"""

import os

from fast_agent.core.direct_factory import get_default_model_source


class TestGetDefaultModelSource:
    """Tests for get_default_model_source function."""

    def test_cli_model_returns_none(self):
        """When CLI model is specified, returns None (no message needed)."""
        result = get_default_model_source(
            config_default_model="sonnet",
            cli_model="haiku",
        )
        assert result is None

    def test_config_model_returns_config_file(self):
        """When config model is set and no CLI, returns 'config file'."""
        result = get_default_model_source(
            config_default_model="sonnet",
            cli_model=None,
        )
        assert result == "config file"

    def test_config_model_alias_returns_config_file(self):
        """Alias defaults still report config as the source."""
        result = get_default_model_source(
            config_default_model="$system.default",
            cli_model=None,
            model_aliases={"system": {"default": "passthrough"}},
        )
        assert result == "config file"

    def test_env_var_returns_environment_variable(self):
        """When env var is set and no config/CLI, returns 'environment variable'."""
        # Store original value if any
        original = os.environ.get("FAST_AGENT_MODEL")

        try:
            os.environ["FAST_AGENT_MODEL"] = "gpt-4o"
            result = get_default_model_source(
                config_default_model=None,
                cli_model=None,
            )
            assert result == "environment variable"
        finally:
            # Restore original state
            if original is not None:
                os.environ["FAST_AGENT_MODEL"] = original
            elif "FAST_AGENT_MODEL" in os.environ:
                del os.environ["FAST_AGENT_MODEL"]

    def test_no_source_returns_none(self):
        """When nothing is set, returns None (hardcoded default used)."""
        # Store original value if any
        original = os.environ.get("FAST_AGENT_MODEL")

        try:
            # Ensure env var is not set
            if "FAST_AGENT_MODEL" in os.environ:
                del os.environ["FAST_AGENT_MODEL"]

            result = get_default_model_source(
                config_default_model=None,
                cli_model=None,
            )
            assert result is None
        finally:
            # Restore original state
            if original is not None:
                os.environ["FAST_AGENT_MODEL"] = original

    def test_config_takes_precedence_over_env_var(self):
        """Config file setting takes precedence over environment variable."""
        original = os.environ.get("FAST_AGENT_MODEL")

        try:
            os.environ["FAST_AGENT_MODEL"] = "gpt-4o"
            result = get_default_model_source(
                config_default_model="sonnet",
                cli_model=None,
            )
            # Config is checked first, so should return "config file"
            assert result == "config file"
        finally:
            if original is not None:
                os.environ["FAST_AGENT_MODEL"] = original
            elif "FAST_AGENT_MODEL" in os.environ:
                del os.environ["FAST_AGENT_MODEL"]

    def test_cli_takes_precedence_over_all(self):
        """CLI model takes precedence over config and env var."""
        original = os.environ.get("FAST_AGENT_MODEL")

        try:
            os.environ["FAST_AGENT_MODEL"] = "gpt-4o"
            result = get_default_model_source(
                config_default_model="sonnet",
                cli_model="haiku",
            )
            # CLI is explicit, so should return None
            assert result is None
        finally:
            if original is not None:
                os.environ["FAST_AGENT_MODEL"] = original
            elif "FAST_AGENT_MODEL" in os.environ:
                del os.environ["FAST_AGENT_MODEL"]
