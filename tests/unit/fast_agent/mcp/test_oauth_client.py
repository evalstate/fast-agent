"""Tests for OAuth client functionality, including CIMD support."""

from unittest.mock import MagicMock, patch

import pytest

from fast_agent.config import MCPServerAuthSettings, MCPServerSettings
from fast_agent.mcp.oauth_client import build_oauth_provider


class TestBuildOAuthProvider:
    """Tests for the build_oauth_provider function."""

    def test_returns_none_for_stdio_transport(self):
        """OAuth should not be used for stdio transport."""
        config = MCPServerSettings(
            name="test",
            transport="stdio",
            command="echo",
        )
        result = build_oauth_provider(config)
        assert result is None

    def test_returns_none_when_oauth_disabled(self):
        """OAuth should not be used when explicitly disabled."""
        config = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
            auth=MCPServerAuthSettings(oauth=False),
        )
        result = build_oauth_provider(config)
        assert result is None

    def test_returns_none_when_no_url(self):
        """OAuth should not be used when there's no URL to derive base from."""
        config = MCPServerSettings(
            name="test",
            transport="http",
            url=None,
        )
        result = build_oauth_provider(config)
        assert result is None

    @patch("fast_agent.mcp.oauth_client.OAuthClientProvider")
    def test_basic_provider_creation(self, mock_provider_class):
        """Test basic OAuth provider creation without CIMD."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
        )

        result = build_oauth_provider(config)

        assert result is mock_provider
        mock_provider_class.assert_called_once()
        call_kwargs = mock_provider_class.call_args.kwargs

        # Should not have client_metadata_url when not using CIMD
        assert "client_metadata_url" not in call_kwargs
        assert call_kwargs["server_url"] == "https://example.com"

    @patch("fast_agent.mcp.oauth_client.OAuthClientProvider")
    def test_cimd_client_id_passed_to_provider(self, mock_provider_class):
        """Test that client_id is passed as client_metadata_url when provided."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        cimd_url = "https://app.example.com/oauth/client-metadata.json"
        config = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
            auth=MCPServerAuthSettings(
                oauth=True,
                client_id=cimd_url,
            ),
        )

        result = build_oauth_provider(config)

        assert result is mock_provider
        mock_provider_class.assert_called_once()
        call_kwargs = mock_provider_class.call_args.kwargs

        # Should have client_metadata_url set to the CIMD URL
        assert call_kwargs["client_metadata_url"] == cimd_url
        assert call_kwargs["server_url"] == "https://example.com"

    @patch("fast_agent.mcp.oauth_client.OAuthClientProvider")
    def test_cimd_with_sse_transport(self, mock_provider_class):
        """Test CIMD works with SSE transport as well."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        cimd_url = "https://myapp.example.com/client.json"
        config = MCPServerSettings(
            name="test",
            transport="sse",
            url="https://server.example.com/sse",
            auth=MCPServerAuthSettings(
                client_id=cimd_url,
            ),
        )

        result = build_oauth_provider(config)

        assert result is mock_provider
        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["client_metadata_url"] == cimd_url

    @patch("fast_agent.mcp.oauth_client.OAuthClientProvider")
    def test_scope_is_passed_to_metadata(self, mock_provider_class):
        """Test that scope configuration is passed through."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
            auth=MCPServerAuthSettings(
                scope=["read", "write"],
            ),
        )

        build_oauth_provider(config)

        call_kwargs = mock_provider_class.call_args.kwargs
        client_metadata = call_kwargs["client_metadata"]
        assert client_metadata.scope == "read write"

    @patch("fast_agent.mcp.oauth_client.OAuthClientProvider")
    def test_custom_redirect_port_and_path(self, mock_provider_class):
        """Test that custom redirect configuration is used."""
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        config = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
            auth=MCPServerAuthSettings(
                redirect_port=8080,
                redirect_path="/oauth/callback",
            ),
        )

        build_oauth_provider(config)

        call_kwargs = mock_provider_class.call_args.kwargs
        client_metadata = call_kwargs["client_metadata"]
        redirect_uris = [str(uri) for uri in client_metadata.redirect_uris]
        assert "http://localhost:8080/oauth/callback" in redirect_uris


class TestMCPServerAuthSettingsClientId:
    """Tests for the client_id field in MCPServerAuthSettings."""

    def test_client_id_defaults_to_none(self):
        """client_id should default to None."""
        auth = MCPServerAuthSettings()
        assert auth.client_id is None

    def test_client_id_can_be_set(self):
        """client_id can be set to a URL string."""
        cimd_url = "https://example.com/oauth/client.json"
        auth = MCPServerAuthSettings(client_id=cimd_url)
        assert auth.client_id == cimd_url

    def test_client_id_with_other_settings(self):
        """client_id can be used alongside other auth settings."""
        auth = MCPServerAuthSettings(
            oauth=True,
            client_id="https://example.com/client.json",
            redirect_port=9000,
            scope="openid profile",
            persist="memory",
        )
        assert auth.client_id == "https://example.com/client.json"
        assert auth.redirect_port == 9000
        assert auth.scope == "openid profile"
        assert auth.persist == "memory"
