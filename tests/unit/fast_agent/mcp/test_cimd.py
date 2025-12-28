"""Tests for Client ID Metadata Document (CIMD) support."""

import pytest
from pydantic import ValidationError

from fast_agent.config import MCPServerAuthSettings, MCPServerSettings
from fast_agent.mcp.oauth_client import build_oauth_provider


class TestCIMDConfigValidation:
    """Test CIMD URL validation in MCPServerAuthSettings."""

    def test_valid_cimd_url(self):
        """A valid HTTPS URL with non-root path should be accepted."""
        auth = MCPServerAuthSettings(
            client_metadata_url="https://example.com/client.json"
        )
        assert auth.client_metadata_url == "https://example.com/client.json"

    def test_valid_cimd_url_with_path(self):
        """A valid HTTPS URL with a deep path should be accepted."""
        auth = MCPServerAuthSettings(
            client_metadata_url="https://example.com/oauth/client-metadata.json"
        )
        assert auth.client_metadata_url == "https://example.com/oauth/client-metadata.json"

    def test_cimd_url_rejects_http(self):
        """HTTP URLs should be rejected (must be HTTPS)."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerAuthSettings(
                client_metadata_url="http://example.com/client.json"
            )
        assert "client_metadata_url must use HTTPS scheme" in str(exc_info.value)

    def test_cimd_url_rejects_root_path(self):
        """URLs with root path (/) should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerAuthSettings(
                client_metadata_url="https://example.com/"
            )
        assert "client_metadata_url must have a non-root pathname" in str(exc_info.value)

    def test_cimd_url_rejects_no_path(self):
        """URLs with no path should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerAuthSettings(
                client_metadata_url="https://example.com"
            )
        assert "client_metadata_url must have a non-root pathname" in str(exc_info.value)

    def test_cimd_url_none_by_default(self):
        """client_metadata_url should be None by default."""
        auth = MCPServerAuthSettings()
        assert auth.client_metadata_url is None


class TestCIMDOAuthProvider:
    """Test that CIMD URL is passed to OAuthClientProvider."""

    def test_build_oauth_provider_with_cimd_url(self, monkeypatch):
        """build_oauth_provider should pass client_metadata_url to OAuthClientProvider."""
        captured_kwargs = {}

        class MockOAuthClientProvider:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(
            "fast_agent.mcp.oauth_client.OAuthClientProvider",
            MockOAuthClientProvider,
        )

        auth = MCPServerAuthSettings(
            client_metadata_url="https://example.com/client.json"
        )
        config = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
            auth=auth,
        )

        build_oauth_provider(config)

        assert captured_kwargs.get("client_metadata_url") == "https://example.com/client.json"

    def test_build_oauth_provider_without_cimd_url(self, monkeypatch):
        """build_oauth_provider should pass None for client_metadata_url when not configured."""
        captured_kwargs = {}

        class MockOAuthClientProvider:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(
            "fast_agent.mcp.oauth_client.OAuthClientProvider",
            MockOAuthClientProvider,
        )

        config = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
        )

        build_oauth_provider(config)

        assert captured_kwargs.get("client_metadata_url") is None

    def test_build_oauth_provider_cimd_with_sse_transport(self, monkeypatch):
        """build_oauth_provider should work with SSE transport and CIMD."""
        captured_kwargs = {}

        class MockOAuthClientProvider:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(
            "fast_agent.mcp.oauth_client.OAuthClientProvider",
            MockOAuthClientProvider,
        )

        auth = MCPServerAuthSettings(
            client_metadata_url="https://example.com/client.json"
        )
        config = MCPServerSettings(
            name="test",
            transport="sse",
            url="https://example.com/sse",
            auth=auth,
        )

        build_oauth_provider(config)

        assert captured_kwargs.get("client_metadata_url") == "https://example.com/client.json"

    def test_build_oauth_provider_stdio_ignores_cimd(self):
        """build_oauth_provider should return None for stdio transport (no OAuth)."""
        auth = MCPServerAuthSettings(
            client_metadata_url="https://example.com/client.json"
        )
        config = MCPServerSettings(
            name="test",
            transport="stdio",
            command="echo",
            auth=auth,
        )

        result = build_oauth_provider(config)

        assert result is None

    def test_build_oauth_provider_oauth_disabled_ignores_cimd(self):
        """build_oauth_provider should return None when OAuth is disabled."""
        auth = MCPServerAuthSettings(
            oauth=False,
            client_metadata_url="https://example.com/client.json"
        )
        config = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
            auth=auth,
        )

        result = build_oauth_provider(config)

        assert result is None
