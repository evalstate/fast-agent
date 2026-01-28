"""Tests for custom default headers support in Anthropic provider."""

import yaml

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM


class TestDefaultHeaders:
    """Test custom headers support for Anthropic provider."""

    def test_default_headers_passed_to_config(self):
        """Test that default_headers are properly configured."""
        custom_headers = {
            "Ocp-Apim-Subscription-Key": "test-key-123",
            "X-Custom-Header": "custom-value",
        }
        
        settings = AnthropicSettings(
            api_key="test-api-key",
            default_headers=custom_headers,
        )
        
        assert settings.default_headers is not None
        assert settings.default_headers == custom_headers
        assert settings.default_headers["Ocp-Apim-Subscription-Key"] == "test-key-123"
        assert settings.default_headers["X-Custom-Header"] == "custom-value"

    def test_default_headers_retrieved_by_llm(self):
        """Test that LLM can retrieve default headers from config."""
        custom_headers = {
            "Ocp-Apim-Subscription-Key": "test-key-456",
        }
        
        settings = Settings()
        settings.anthropic = AnthropicSettings(
            api_key="test-api-key",
            default_headers=custom_headers,
        )
        
        ctx = Context()
        ctx.config = settings
        
        llm = AnthropicLLM(context=ctx)
        retrieved_headers = llm._default_headers()
        
        assert retrieved_headers is not None
        assert retrieved_headers == custom_headers
        assert retrieved_headers["Ocp-Apim-Subscription-Key"] == "test-key-456"

    def test_no_default_headers_returns_none(self):
        """Test that when no headers are configured, None is returned."""
        settings = Settings()
        settings.anthropic = AnthropicSettings(api_key="test-api-key")
        
        ctx = Context()
        ctx.config = settings
        
        llm = AnthropicLLM(context=ctx)
        retrieved_headers = llm._default_headers()
        
        assert retrieved_headers is None

    def test_empty_default_headers(self):
        """Test that empty headers dict can be configured."""
        settings = AnthropicSettings(
            api_key="test-api-key",
            default_headers={},
        )
        
        assert settings.default_headers == {}

    def test_default_headers_from_yaml(self):
        """Test that default_headers can be loaded from YAML configuration."""
        yaml_config = """
anthropic:
  api_key: "dummy"
  base_url: "https://llm-api.amd.com/Anthropic"
  default_headers:
    "Ocp-Apim-Subscription-Key": "test-subscription-key"
    "X-Custom-Header": "custom-value"
"""
        config_dict = yaml.safe_load(yaml_config)
        
        settings = AnthropicSettings(**config_dict["anthropic"])
        
        assert settings.api_key == "dummy"
        assert settings.base_url == "https://llm-api.amd.com/Anthropic"
        assert settings.default_headers is not None
        assert settings.default_headers["Ocp-Apim-Subscription-Key"] == "test-subscription-key"
        assert settings.default_headers["X-Custom-Header"] == "custom-value"

    def test_yaml_config_with_multiple_headers(self):
        """Test YAML configuration with multiple custom headers."""
        yaml_config = """
anthropic:
  api_key: "test-key"
  base_url: "https://custom-api.example.com"
  default_headers:
    "Authorization": "Bearer custom-token"
    "X-API-Key": "api-key-value"
    "X-Request-ID": "req-12345"
    "X-Tenant-ID": "tenant-abc"
"""
        config_dict = yaml.safe_load(yaml_config)
        
        settings = AnthropicSettings(**config_dict["anthropic"])
        
        assert settings.default_headers is not None
        assert len(settings.default_headers) == 4
        assert settings.default_headers["Authorization"] == "Bearer custom-token"
        assert settings.default_headers["X-API-Key"] == "api-key-value"
        assert settings.default_headers["X-Request-ID"] == "req-12345"
        assert settings.default_headers["X-Tenant-ID"] == "tenant-abc"
