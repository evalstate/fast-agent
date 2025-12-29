"""Tests for Client ID Metadata Document (CIMD) support."""

import socket

import pytest
from pydantic import ValidationError

from fast_agent.config import MCPServerAuthSettings, MCPServerSettings
from fast_agent.mcp.oauth_client import _CallbackServer, build_oauth_provider


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

    def test_build_oauth_provider_uses_loopback_ip(self, monkeypatch):
        """build_oauth_provider should use 127.0.0.1 (loopback IP) for RFC 8252 compliance."""
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

        # Check that redirect_uris use 127.0.0.1 instead of localhost
        client_metadata = captured_kwargs.get("client_metadata")
        assert client_metadata is not None
        redirect_uris = [str(uri) for uri in client_metadata.redirect_uris]
        assert all("127.0.0.1" in uri for uri in redirect_uris)
        assert not any("localhost" in uri for uri in redirect_uris)

    def test_build_oauth_provider_registers_fallback_ports(self, monkeypatch):
        """build_oauth_provider should register multiple ports for fallback support."""
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

        client_metadata = captured_kwargs.get("client_metadata")
        assert client_metadata is not None
        redirect_uris = [str(uri) for uri in client_metadata.redirect_uris]
        # Should have multiple redirect URIs for port fallback
        assert len(redirect_uris) >= 3
        # Should include default port 3030
        assert any(":3030/" in uri for uri in redirect_uris)


class TestCallbackServerPortFallback:
    """Test RFC 8252 compliant port fallback in _CallbackServer."""

    def test_callback_server_uses_loopback_ip(self):
        """_CallbackServer should bind to 127.0.0.1 for RFC 8252 compliance."""
        server = _CallbackServer(port=0, path="/callback")  # Use ephemeral port
        try:
            server.start()
            assert server.actual_port is not None
            assert server.actual_port > 0
            # The server is bound to 127.0.0.1
            assert server._server is not None
            assert server._server.server_address[0] == "127.0.0.1"
        finally:
            server.stop()

    def test_callback_server_ephemeral_port(self):
        """_CallbackServer should work with ephemeral port (0)."""
        server = _CallbackServer(port=0, path="/callback")
        try:
            server.start()
            # Should get a real port assigned
            assert server.actual_port is not None
            assert server.actual_port > 0
        finally:
            server.stop()

    def test_callback_server_get_redirect_uri(self):
        """get_redirect_uri should return the actual bound port."""
        server = _CallbackServer(port=0, path="/callback")
        try:
            server.start()
            redirect_uri = server.get_redirect_uri()
            assert redirect_uri.startswith("http://127.0.0.1:")
            assert redirect_uri.endswith("/callback")
            assert f":{server.actual_port}/" in redirect_uri
        finally:
            server.stop()

    def test_callback_server_port_fallback(self):
        """_CallbackServer should fall back to next port if preferred is in use."""
        # Occupy the preferred port
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            blocker.bind(("127.0.0.1", 13030))  # Use unusual port to avoid conflicts
            blocker.listen(1)

            # Try to start server on blocked port
            server = _CallbackServer(port=13030, path="/callback")
            try:
                server.start()
                # Should have fallen back to a different port
                assert server.actual_port != 13030
                assert server.actual_port is not None
            finally:
                server.stop()
        finally:
            blocker.close()

    def test_callback_server_get_redirect_uri_before_start_raises(self):
        """get_redirect_uri should raise if called before start()."""
        server = _CallbackServer(port=3030, path="/callback")
        with pytest.raises(RuntimeError, match="Server not started"):
            server.get_redirect_uri()
