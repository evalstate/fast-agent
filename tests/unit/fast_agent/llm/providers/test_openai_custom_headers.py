"""Tests for custom headers configuration in OpenAI-compatible providers.

These tests verify that custom headers can be configured via settings
and are correctly passed to the OpenAI client.
"""


from fast_agent.config import (
    AzureSettings,
    DeepSeekSettings,
    GenericSettings,
    GoogleSettings,
    GroqSettings,
    HuggingFaceSettings,
    OpenAISettings,
    OpenRouterSettings,
    Settings,
    TensorZeroSettings,
    XAISettings,
)
from fast_agent.context import Context


class TestSettingsDefaultHeaders:
    """Test that settings classes correctly handle default_headers field."""

    def test_openai_settings_default_headers_none_by_default(self):
        """OpenAI settings should have None default_headers by default."""
        settings = OpenAISettings()
        assert settings.default_headers is None

    def test_openai_settings_default_headers_can_be_set(self):
        """OpenAI settings should accept default_headers dictionary."""
        headers = {"X-Custom-Header": "value", "X-Another": "test"}
        settings = OpenAISettings(default_headers=headers)
        assert settings.default_headers == headers

    def test_generic_settings_default_headers(self):
        """Generic settings should support default_headers."""
        headers = {"X-Portkey-Config": "abc123"}
        settings = GenericSettings(default_headers=headers)
        assert settings.default_headers == headers

    def test_openrouter_settings_default_headers(self):
        """OpenRouter settings should support default_headers."""
        headers = {"HTTP-Referer": "https://myapp.com"}
        settings = OpenRouterSettings(default_headers=headers)
        assert settings.default_headers == headers

    def test_deepseek_settings_default_headers(self):
        """DeepSeek settings should support default_headers."""
        headers = {"X-Custom": "value"}
        settings = DeepSeekSettings(default_headers=headers)
        assert settings.default_headers == headers

    def test_xai_settings_default_headers(self):
        """xAI settings should support default_headers."""
        headers = {"Authorization-Extra": "token"}
        settings = XAISettings(default_headers=headers)
        assert settings.default_headers == headers

    def test_groq_settings_default_headers(self):
        """Groq settings should support default_headers."""
        headers = {"X-Groq-Custom": "value"}
        settings = GroqSettings(default_headers=headers)
        assert settings.default_headers == headers

    def test_google_settings_default_headers(self):
        """Google settings should support default_headers."""
        headers = {"X-Google-Custom": "value"}
        settings = GoogleSettings(default_headers=headers)
        assert settings.default_headers == headers

    def test_tensorzero_settings_default_headers(self):
        """TensorZero settings should support default_headers."""
        headers = {"X-TensorZero": "value"}
        settings = TensorZeroSettings(default_headers=headers)
        assert settings.default_headers == headers

    def test_huggingface_settings_default_headers(self):
        """HuggingFace settings should support default_headers."""
        headers = {"X-HF-Custom": "value"}
        settings = HuggingFaceSettings(default_headers=headers)
        assert settings.default_headers == headers

    def test_azure_settings_default_headers(self):
        """Azure settings should support default_headers."""
        headers = {"Ocp-Apim-Subscription-Key": "value"}
        settings = AzureSettings(default_headers=headers)
        assert settings.default_headers == headers


class TestMainSettingsIntegration:
    """Test that the main Settings class correctly handles provider headers."""

    def test_settings_openai_headers_from_dict(self):
        """Settings should correctly parse OpenAI headers from dict."""
        settings = Settings.model_validate(
            {"openai": {"api_key": "test-key", "default_headers": {"X-Custom": "value"}}}
        )
        assert settings.openai is not None
        assert settings.openai.default_headers == {"X-Custom": "value"}

    def test_settings_generic_headers_from_dict(self):
        """Settings should correctly parse generic provider headers from dict."""
        settings = Settings.model_validate(
            {
                "generic": {
                    "base_url": "http://localhost:11434/v1",
                    "default_headers": {"X-Test": "123"},
                }
            }
        )
        assert settings.generic is not None
        assert settings.generic.default_headers == {"X-Test": "123"}

    def test_settings_multiple_providers_headers(self):
        """Settings should correctly handle headers for multiple providers."""
        settings = Settings.model_validate(
            {
                "openai": {"default_headers": {"X-OpenAI": "openai-value"}},
                "openrouter": {"default_headers": {"X-OpenRouter": "openrouter-value"}},
                "generic": {"default_headers": {"X-Generic": "generic-value"}},
            }
        )
        assert settings.openai is not None
        assert settings.openrouter is not None
        assert settings.generic is not None
        assert settings.openai.default_headers == {"X-OpenAI": "openai-value"}
        assert settings.openrouter.default_headers == {"X-OpenRouter": "openrouter-value"}
        assert settings.generic.default_headers == {"X-Generic": "generic-value"}

    def test_settings_azure_headers_from_dict(self):
        """Settings should correctly parse Azure headers from dict."""
        settings = Settings.model_validate(
            {
                "azure": {
                    "api_key": "test-key",
                    "base_url": "https://example.openai.azure.com/",
                    "azure_deployment": "gpt-4o",
                    "default_headers": {"Ocp-Apim-Subscription-Key": "value"},
                }
            }
        )
        assert settings.azure is not None
        assert settings.azure.default_headers == {"Ocp-Apim-Subscription-Key": "value"}


class TestLLMDefaultHeadersMethod:
    """Test that LLM classes correctly return headers from config."""

    def test_openai_llm_default_headers_returns_none_without_config(self):
        """OpenAI LLM should return None when no headers configured."""
        from fast_agent.llm.provider.openai.llm_openai import OpenAILLM

        context = Context()
        llm = OpenAILLM(context=context)
        assert llm._default_headers() is None

    def test_openai_llm_default_headers_returns_configured_headers(self):
        """OpenAI LLM should return headers from config."""
        from fast_agent.llm.provider.openai.llm_openai import OpenAILLM

        headers = {"X-Portkey-Trace-Id": "trace-123"}
        settings = Settings.model_validate({"openai": {"default_headers": headers}})
        context = Context(config=settings)
        llm = OpenAILLM(context=context)

        assert llm._default_headers() == headers

    def test_generic_llm_default_headers_returns_configured_headers(self):
        """Generic LLM should return headers from generic config."""
        from fast_agent.llm.provider.openai.llm_generic import GenericLLM

        headers = {"X-Custom-Gateway": "gateway-value"}
        settings = Settings.model_validate({"generic": {"default_headers": headers}})
        context = Context(config=settings)
        llm = GenericLLM(context=context)

        assert llm._default_headers() == headers

    def test_openrouter_llm_default_headers_returns_configured_headers(self):
        """OpenRouter LLM should return headers from openrouter config."""
        from fast_agent.llm.provider.openai.llm_openrouter import OpenRouterLLM

        headers = {"HTTP-Referer": "https://myapp.com", "X-Title": "My App"}
        settings = Settings.model_validate({"openrouter": {"default_headers": headers}})
        context = Context(config=settings)
        llm = OpenRouterLLM(context=context)

        assert llm._default_headers() == headers

    def test_deepseek_llm_default_headers_returns_configured_headers(self):
        """DeepSeek LLM should return headers from deepseek config."""
        from fast_agent.llm.provider.openai.llm_deepseek import DeepSeekLLM

        headers = {"X-DeepSeek-Custom": "value"}
        settings = Settings.model_validate({"deepseek": {"default_headers": headers}})
        context = Context(config=settings)
        llm = DeepSeekLLM(context=context)

        assert llm._default_headers() == headers

    def test_xai_llm_default_headers_returns_configured_headers(self):
        """xAI LLM should return headers from xai config."""
        from fast_agent.llm.provider.openai.llm_xai import XAILLM

        headers = {"X-XAI-Custom": "value"}
        settings = Settings.model_validate({"xai": {"default_headers": headers}})
        context = Context(config=settings)
        llm = XAILLM(context=context)

        assert llm._default_headers() == headers

    def test_groq_llm_default_headers_returns_configured_headers(self):
        """Groq LLM should return headers from groq config."""
        from fast_agent.llm.provider.openai.llm_groq import GroqLLM

        headers = {"X-Groq-Custom": "value"}
        settings = Settings.model_validate({"groq": {"default_headers": headers}})
        context = Context(config=settings)
        llm = GroqLLM(context=context)

        assert llm._default_headers() == headers

    def test_azure_llm_default_headers_returns_configured_headers(self):
        """Azure LLM should return headers from azure config."""
        from fast_agent.llm.provider.openai.llm_azure import AzureOpenAILLM

        headers = {"Ocp-Apim-Subscription-Key": "value"}
        settings = Settings.model_validate(
            {
                "azure": {
                    "api_key": "test-key",
                    "base_url": "https://example.openai.azure.com/",
                    "azure_deployment": "gpt-4o",
                    "default_headers": headers,
                }
            }
        )
        context = Context(config=settings)
        llm = AzureOpenAILLM(context=context)

        assert llm._default_headers() == headers


class TestOpenAIClientCreation:
    """Test that the OpenAI client is created with custom headers."""

    def test_openai_client_includes_custom_headers(self):
        """OpenAI client should include custom headers when configured."""
        from fast_agent.llm.provider.openai.llm_openai import OpenAILLM

        headers = {"X-Portkey-Config": "config-id", "X-Custom-Header": "custom-value"}
        settings = Settings.model_validate(
            {"openai": {"api_key": "test-key", "default_headers": headers}}
        )
        context = Context(config=settings)
        llm = OpenAILLM(context=context)

        # Create the client
        client = llm._openai_client()

        # Verify the client has the custom headers set
        # The OpenAI SDK stores default headers in _custom_headers
        assert client._custom_headers is not None
        assert client._custom_headers.get("X-Portkey-Config") == "config-id"
        assert client._custom_headers.get("X-Custom-Header") == "custom-value"

    def test_openai_client_without_headers_has_no_custom_headers(self):
        """OpenAI client should not have custom headers when none configured."""
        from fast_agent.llm.provider.openai.llm_openai import OpenAILLM

        settings = Settings.model_validate({"openai": {"api_key": "test-key"}})
        context = Context(config=settings)
        llm = OpenAILLM(context=context)

        # Create the client
        client = llm._openai_client()

        # Verify no custom headers (or empty dict)
        assert client._custom_headers is None or client._custom_headers == {}

    def test_generic_client_includes_custom_headers(self):
        """Generic LLM client should include custom headers when configured."""
        from fast_agent.llm.provider.openai.llm_generic import GenericLLM

        headers = {"X-Gateway-Auth": "token123"}
        settings = Settings.model_validate(
            {"generic": {"api_key": "test-key", "default_headers": headers}}
        )
        context = Context(config=settings)
        llm = GenericLLM(context=context)

        # Create the client
        client = llm._openai_client()

        # Verify the client has the custom headers set
        assert client._custom_headers is not None
        assert client._custom_headers.get("X-Gateway-Auth") == "token123"

    def test_azure_client_includes_custom_headers(self):
        """Azure LLM client should include custom headers when configured."""
        from fast_agent.llm.provider.openai.llm_azure import AzureOpenAILLM

        headers = {"Ocp-Apim-Subscription-Key": "value"}
        settings = Settings.model_validate(
            {
                "azure": {
                    "api_key": "test-key",
                    "base_url": "https://example.openai.azure.com/",
                    "azure_deployment": "gpt-4o",
                    "default_headers": headers,
                }
            }
        )
        context = Context(config=settings)
        llm = AzureOpenAILLM(context=context)

        client = llm._openai_client()

        assert client._custom_headers is not None
        assert client._custom_headers.get("Ocp-Apim-Subscription-Key") == "value"
