"""Unit tests for HuggingFace token display in check config command.

WARNING: This test suite modifies environment variables directly during testing.
Environment variables are volatile and may be temporarily modified during test execution.
"""

import os

from fast_agent.cli.commands.check_config import check_api_keys
from fast_agent.llm.provider_key_manager import API_KEY_HINT_TEXT


def _set_hf_token(value: str | None) -> str | None:
    """Set HF_TOKEN environment variable and return the original value."""
    original = os.getenv("HF_TOKEN")
    if value is None:
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
    else:
        os.environ["HF_TOKEN"] = value
    return original


def _restore_hf_token(original_value: str | None) -> None:
    """Restore HF_TOKEN environment variable to its original value."""
    if original_value is None:
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
    else:
        os.environ["HF_TOKEN"] = original_value


def test_check_api_keys_includes_huggingface():
    """Test that HuggingFace provider is included in API key check results."""
    # Empty config/secrets
    config_summary = {}
    secrets_summary = {"status": "not_found", "secrets": {}}

    original = _set_hf_token(None)
    try:
        results = check_api_keys(secrets_summary, config_summary)

        # HuggingFace should be in the results
        assert "huggingface" in results
        assert results["huggingface"]["env"] == ""
        assert results["huggingface"]["config"] == ""
    finally:
        _restore_hf_token(original)


def test_check_api_keys_detects_hf_token_in_env():
    """Test that HF_TOKEN is detected from environment variables."""
    config_summary = {}
    secrets_summary = {"status": "not_found", "secrets": {}}

    original = _set_hf_token("hf_1234567890abcdef")
    try:
        results = check_api_keys(secrets_summary, config_summary)

        assert "huggingface" in results
        assert results["huggingface"]["env"] == "...bcdef"  # Shows last 5 chars
        assert results["huggingface"]["config"] == ""
    finally:
        _restore_hf_token(original)


def test_check_api_keys_detects_hf_token_in_config():
    """Test that HuggingFace token is detected from config/secrets file."""
    config_summary = {}
    secrets_summary = {
        "status": "parsed",
        "secrets": {"huggingface": {"api_key": "hf_config_token_12345"}},
    }

    original = _set_hf_token(None)
    try:
        results = check_api_keys(secrets_summary, config_summary)

        assert "huggingface" in results
        assert results["huggingface"]["env"] == ""
        assert results["huggingface"]["config"] == "...12345"  # Shows last 5 chars
    finally:
        _restore_hf_token(original)


def test_check_api_keys_config_takes_precedence_over_env():
    """Test that config file token takes precedence over environment variable."""
    config_summary = {}
    secrets_summary = {
        "status": "parsed",
        "secrets": {"huggingface": {"api_key": "hf_config_priority"}},
    }

    original = _set_hf_token("hf_env_token")
    try:
        results = check_api_keys(secrets_summary, config_summary)

        assert "huggingface" in results
        assert results["huggingface"]["env"] == "...token"  # Env token detected
        assert results["huggingface"]["config"] == "...ority"  # Config token takes precedence
    finally:
        _restore_hf_token(original)


def test_check_api_keys_ignores_hint_text():
    """Test that API_KEY_HINT_TEXT placeholder is ignored."""
    config_summary = {}
    secrets_summary = {
        "status": "parsed",
        "secrets": {"huggingface": {"api_key": API_KEY_HINT_TEXT}},
    }

    original = _set_hf_token(None)
    try:
        results = check_api_keys(secrets_summary, config_summary)

        assert "huggingface" in results
        assert results["huggingface"]["env"] == ""
        assert results["huggingface"]["config"] == ""  # Hint text is ignored
    finally:
        _restore_hf_token(original)


def test_check_api_keys_short_token():
    """Test handling of short tokens (less than 5 characters)."""
    config_summary = {}
    secrets_summary = {"status": "parsed", "secrets": {"huggingface": {"api_key": "hf"}}}

    original = _set_hf_token("env")
    try:
        results = check_api_keys(secrets_summary, config_summary)

        assert "huggingface" in results
        assert results["huggingface"]["env"] == "...***"  # Short token masked
        assert results["huggingface"]["config"] == "...***"  # Short token masked
    finally:
        _restore_hf_token(original)
