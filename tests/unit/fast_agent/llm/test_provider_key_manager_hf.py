"""Unit tests for HuggingFace provider key management.

WARNING: This test suite modifies environment variables directly during testing.
Environment variables are volatile and may be temporarily modified during test execution.
"""

import os

from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.llm.provider_key_manager import ProviderKeyManager


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


def test_huggingface_env_var_name():
    """Test that HuggingFace uses HF_TOKEN as the environment variable name."""
    assert ProviderKeyManager.get_env_key_name("huggingface") == "HF_TOKEN"


def test_get_api_key_from_env():
    """Test getting HuggingFace API key from environment variable."""
    original = _set_hf_token("hf_env_token_12345")
    try:
        config = Settings()
        api_key = ProviderKeyManager.get_api_key("huggingface", config)
        assert api_key == "hf_env_token_12345"
    finally:
        _restore_hf_token(original)


def test_get_api_key_from_config():
    """Test getting HuggingFace API key from config."""
    original = _set_hf_token(None)
    try:
        config = Settings(huggingface=HuggingFaceSettings(api_key="hf_config_token"))
        api_key = ProviderKeyManager.get_api_key("huggingface", config)
        assert api_key == "hf_config_token"
    finally:
        _restore_hf_token(original)


def test_config_takes_precedence_over_env():
    """Test that config API key takes precedence over environment variable."""
    original = _set_hf_token("hf_env_token")
    try:
        config = Settings(huggingface=HuggingFaceSettings(api_key="hf_config_priority"))
        api_key = ProviderKeyManager.get_api_key("huggingface", config)
        assert api_key == "hf_config_priority"
    finally:
        _restore_hf_token(original)


def test_get_config_file_key():
    """Test extracting HuggingFace API key from config object."""
    config = {"huggingface": {"api_key": "hf_test_key"}}
    key = ProviderKeyManager.get_config_file_key("huggingface", config)
    assert key == "hf_test_key"


def test_get_config_file_key_no_provider():
    """Test extracting API key when provider is not in config."""
    config = {"other_provider": {"api_key": "other_key"}}
    key = ProviderKeyManager.get_config_file_key("huggingface", config)
    assert key is None


def test_get_config_file_key_hint_text():
    """Test that hint text is treated as no key."""
    config = {"huggingface": {"api_key": "<your-api-key-here>"}}
    key = ProviderKeyManager.get_config_file_key("huggingface", config)
    assert key is None
