"""Unit tests for HuggingFace provider key management.

WARNING: This test suite modifies environment variables directly during testing.
Environment variables are volatile and may be temporarily modified during test execution.
"""

import os
from types import SimpleNamespace

import pytest
from mcp.types import Tool

from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.auth.context import request_bearer_token
from fast_agent.types import PromptMessageExtended, RequestParams


class _HuggingFaceStubLLM(
    FastAgentLLM[PromptMessageExtended, PromptMessageExtended]
):
    def __init__(self, *, api_key: str) -> None:
        super().__init__(provider=Provider.HUGGINGFACE, api_key=api_key)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        del request_params, tools, is_template
        return multipart_messages[-1]

    def _convert_extended_messages_to_provider(
        self,
        messages: list[PromptMessageExtended],
    ) -> list[PromptMessageExtended]:
        return messages


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
    assert ProviderKeyManager.get_env_key_name("hf") == "HF_TOKEN"


def test_provider_env_var_name_normalizes_provider_name():
    assert ProviderKeyManager.get_env_key_name(" HF ") == "HF_TOKEN"
    assert ProviderKeyManager.get_env_key_name(" ANTHROPIC-VERTEX ") is None


def test_get_api_key_from_env():
    """Test getting HuggingFace API key from environment variable."""
    original = _set_hf_token("hf_env_token_12345")
    try:
        config = Settings()
        api_key = ProviderKeyManager.get_api_key("hf", config)
        assert api_key == "hf_env_token_12345"
    finally:
        _restore_hf_token(original)


def test_get_api_key_from_config():
    """Test getting HuggingFace API key from config."""
    original = _set_hf_token(None)
    try:
        config = Settings(hf=HuggingFaceSettings(api_key="hf_config_token"))
        api_key = ProviderKeyManager.get_api_key("hf", config)
        assert api_key == "hf_config_token"
    finally:
        _restore_hf_token(original)


def test_get_api_key_normalizes_provider_name():
    original = _set_hf_token(None)
    try:
        config = Settings(hf=HuggingFaceSettings(api_key="hf_config_token"))
        api_key = ProviderKeyManager.get_api_key(" HF ", config)
        assert api_key == "hf_config_token"
    finally:
        _restore_hf_token(original)


def test_config_takes_precedence_over_env():
    """Test that config API key takes precedence over environment variable."""
    original = _set_hf_token("hf_env_token")
    try:
        config = Settings(hf=HuggingFaceSettings(api_key="hf_config_priority"))
        api_key = ProviderKeyManager.get_api_key("hf", config)
        assert api_key == "hf_config_priority"
    finally:
        _restore_hf_token(original)


def test_request_context_token_takes_precedence_over_config_and_env():
    """Test that request-scoped auth tokens override config and environment values."""
    original = _set_hf_token("hf_env_token")
    saved_request_token = request_bearer_token.set("hf_request_token")
    try:
        config = Settings(hf=HuggingFaceSettings(api_key="hf_config_priority"))
        api_key = ProviderKeyManager.get_api_key("hf", config)
        assert api_key == "hf_request_token"
    finally:
        request_bearer_token.reset(saved_request_token)
        _restore_hf_token(original)


def _set_serve_oauth(value: str | None) -> str | None:
    original = os.getenv("FAST_AGENT_SERVE_OAUTH")
    if value is None:
        os.environ.pop("FAST_AGENT_SERVE_OAUTH", None)
    else:
        os.environ["FAST_AGENT_SERVE_OAUTH"] = value
    return original


def test_serve_oauth_fails_closed_without_request_token():
    """Under FAST_AGENT_SERVE_OAUTH=huggingface, a missing caller token must not
    silently fall back to the server's HF_TOKEN."""
    original_token = _set_hf_token("hf_server_secret")
    original_oauth = _set_serve_oauth("huggingface")
    try:
        config = Settings(hf=HuggingFaceSettings(api_key="hf_config_key"))
        try:
            ProviderKeyManager.get_optional_api_key("hf", config)
        except ProviderKeyError:
            pass
        else:  # pragma: no cover - failure path
            raise AssertionError("expected fail-closed ProviderKeyError")
    finally:
        _set_serve_oauth(original_oauth)
        _restore_hf_token(original_token)


def test_serve_oauth_uses_request_token_when_present():
    original_token = _set_hf_token("hf_server_secret")
    original_oauth = _set_serve_oauth("huggingface")
    saved_request_token = request_bearer_token.set("hf_caller_token")
    try:
        config = Settings(hf=HuggingFaceSettings(api_key="hf_config_key"))
        assert ProviderKeyManager.get_api_key("hf", config) == "hf_caller_token"
    finally:
        request_bearer_token.reset(saved_request_token)
        _set_serve_oauth(original_oauth)
        _restore_hf_token(original_token)


@pytest.mark.asyncio
async def test_serve_oauth_rejects_explicit_agent_key_without_request_token():
    original_oauth = _set_serve_oauth("huggingface")
    try:
        llm = _HuggingFaceStubLLM(api_key="hf_explicit_server_key")

        # Startup validation must remain request-neutral, but resolving the key for
        # an actual provider call must fail closed without a caller token.
        llm.validate_provider_credentials()
        with pytest.raises(ProviderKeyError, match="caller token missing"):
            llm._api_key()
    finally:
        _set_serve_oauth(original_oauth)


@pytest.mark.asyncio
async def test_serve_oauth_caller_token_overrides_explicit_agent_key():
    original_oauth = _set_serve_oauth("huggingface")
    saved_request_token = request_bearer_token.set("hf_caller_token")
    try:
        llm = _HuggingFaceStubLLM(api_key="hf_explicit_server_key")

        assert llm._api_key() == "hf_caller_token"
    finally:
        request_bearer_token.reset(saved_request_token)
        _set_serve_oauth(original_oauth)


def test_serve_oauth_request_requirement_uses_provider_aliases():
    original_oauth = _set_serve_oauth("hf")
    try:
        assert ProviderKeyManager.serve_oauth_requires_request_token("hf")
        assert ProviderKeyManager.serve_oauth_requires_request_token("HuggingFace")
        assert not ProviderKeyManager.serve_oauth_requires_request_token("anthropic")
    finally:
        _set_serve_oauth(original_oauth)


def test_serve_oauth_off_allows_server_token_fallback():
    original_token = _set_hf_token("hf_server_secret")
    original_oauth = _set_serve_oauth(None)
    try:
        assert ProviderKeyManager.get_api_key("hf", Settings()) == "hf_server_secret"
    finally:
        _set_serve_oauth(original_oauth)
        _restore_hf_token(original_token)


def test_serve_oauth_guard_does_not_affect_other_providers():
    original_oauth = _set_serve_oauth("huggingface")
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
    try:
        assert (
            ProviderKeyManager.get_api_key("anthropic", Settings()) == "sk-ant-test"
        )
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        _set_serve_oauth(original_oauth)


def test_get_config_file_key():
    """Test extracting HuggingFace API key from config object."""
    config = {"hf": {"api_key": "hf_test_key"}}
    key = ProviderKeyManager.get_config_file_key("hf", config)
    assert key == "hf_test_key"


def test_get_config_file_key_no_provider():
    """Test extracting API key when provider is not in config."""
    config = {"other_provider": {"api_key": "other_key"}}
    key = ProviderKeyManager.get_config_file_key("hf", config)
    assert key is None


def test_get_config_file_key_ignores_missing_or_non_mapping_config():
    assert ProviderKeyManager.get_config_file_key("hf", None) is None
    assert ProviderKeyManager.get_config_file_key("hf", SimpleNamespace()) is None


def test_get_config_file_key_hint_text():
    """Test that hint text is treated as no key."""
    config = {"hf": {"api_key": "<your-api-key-here>"}}
    key = ProviderKeyManager.get_config_file_key("hf", config)
    assert key is None
