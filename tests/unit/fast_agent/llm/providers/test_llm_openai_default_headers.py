"""Tests for custom default_headers configuration in OpenAI LLM provider."""

import pytest

from fast_agent.config import OpenAISettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM


class TestOpenAIDefaultHeaders:
    """Test suite for default_headers configuration feature."""

    def test_default_headers_returns_none_when_not_configured(self):
        """When no headers are configured, _default_headers() returns None."""
        context = Context()
        llm = OpenAILLM(context=context)

        assert llm._default_headers() is None

    def test_default_headers_returns_none_when_openai_settings_missing(self):
        """When openai settings is None, _default_headers() returns None."""
        settings = Settings()
        settings.openai = None
        context = Context(config=settings)
        llm = OpenAILLM(context=context)

        assert llm._default_headers() is None

    def test_default_headers_returns_configured_headers(self):
        """When headers are configured, _default_headers() returns them."""
        custom_headers = {
            "x-portkey-api-key": "pk-test-123",
            "x-portkey-provider": "openai",
        }
        openai_settings = OpenAISettings(
            api_key="test-key",
            default_headers=custom_headers,
        )
        settings = Settings(openai=openai_settings)
        context = Context(config=settings)
        llm = OpenAILLM(context=context)

        result = llm._default_headers()

        assert result == custom_headers
        assert result["x-portkey-api-key"] == "pk-test-123"
        assert result["x-portkey-provider"] == "openai"

    def test_default_headers_empty_dict_is_preserved(self):
        """When headers are set to empty dict, it is preserved (not None)."""
        openai_settings = OpenAISettings(
            api_key="test-key",
            default_headers={},
        )
        settings = Settings(openai=openai_settings)
        context = Context(config=settings)
        llm = OpenAILLM(context=context)

        result = llm._default_headers()

        assert result == {}

    def test_openai_client_receives_default_headers(self):
        """Verify that AsyncOpenAI client is created with custom headers."""
        custom_headers = {
            "x-custom-header": "custom-value",
            "x-another-header": "another-value",
        }
        openai_settings = OpenAISettings(
            api_key="test-key-for-headers-test",
            default_headers=custom_headers,
        )
        settings = Settings(openai=openai_settings)
        context = Context(config=settings)
        llm = OpenAILLM(context=context)

        # Create the client and verify headers are passed
        client = llm._openai_client()

        # The OpenAI client stores custom headers in _custom_headers
        # This is a non-invasive way to verify headers were passed
        assert client._custom_headers is not None
        assert "x-custom-header" in client._custom_headers
        assert client._custom_headers["x-custom-header"] == "custom-value"
        assert "x-another-header" in client._custom_headers
        assert client._custom_headers["x-another-header"] == "another-value"

    def test_openai_client_works_without_custom_headers(self):
        """Verify client creation works when no custom headers are configured."""
        openai_settings = OpenAISettings(api_key="test-key-no-headers")
        settings = Settings(openai=openai_settings)
        context = Context(config=settings)
        llm = OpenAILLM(context=context)

        # Should not raise
        client = llm._openai_client()

        # Client should be created successfully
        assert client is not None
