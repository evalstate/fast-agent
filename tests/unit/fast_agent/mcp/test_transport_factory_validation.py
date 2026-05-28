"""Tests for transport factory validation with inferred transport types."""

import pytest

from fast_agent.config import MCPServerAuthSettings, MCPServerSettings
from fast_agent.mcp.mcp_connection_manager import create_transport_context


def test_transport_factory_validation_stdio_without_command():
    """Test that stdio transport without command raises appropriate error."""
    server_config = MCPServerSettings(transport="stdio")

    with pytest.raises(ValueError, match="uses stdio transport but no command is specified"):
        create_transport_context(server_name="test_server", config=server_config)


def test_transport_factory_validation_http_without_url():
    """Test that http transport without URL raises appropriate error."""
    server_config = MCPServerSettings(transport="http")

    with pytest.raises(ValueError, match="uses http transport but no url is specified"):
        create_transport_context(server_name="test_server", config=server_config)


def test_transport_factory_validation_sse_without_url():
    """Test that sse transport without URL raises appropriate error."""
    server_config = MCPServerSettings(transport="sse")

    with pytest.raises(ValueError, match="uses sse transport but no url is specified"):
        create_transport_context(server_name="test_server", config=server_config)


def test_inferred_http_transport_has_url():
    """Test that inferred HTTP transport always has a URL (from our inference logic)."""
    server_config = MCPServerSettings(url="http://example.com/mcp")

    assert server_config.transport == "http"
    assert server_config.url == "http://example.com/mcp"

    # create_transport_context should not raise for valid config
    ctx = create_transport_context(server_name="test_server", config=server_config)
    assert ctx is not None


def test_inferred_stdio_transport_has_command():
    """Test that inferred stdio transport always has a command (when provided)."""
    server_config = MCPServerSettings(command="npx server")

    assert server_config.transport == "stdio"
    assert server_config.command == "npx server"

    # create_transport_context should not raise for valid config
    ctx = create_transport_context(server_name="test_server", config=server_config)
    assert ctx is not None


def test_explicit_transport_validation_still_works():
    """Test that explicit transport settings still get validated properly."""
    # Explicit http transport without URL should fail validation
    server_config = MCPServerSettings(transport="http", command="some_command")
    with pytest.raises(ValueError, match="uses http transport but no url is specified"):
        create_transport_context(server_name="test_server", config=server_config)

    # Explicit stdio transport without command should fail validation
    server_config = MCPServerSettings(transport="stdio", url="http://example.com")
    with pytest.raises(ValueError, match="uses stdio transport but no command is specified"):
        create_transport_context(server_name="test_server", config=server_config)


def test_nonpersistent_transport_avoids_speculative_oauth(monkeypatch: pytest.MonkeyPatch):
    captured: list[bool | None] = []

    def _fake_prepare_headers_and_auth(server_config, **kwargs):
        del server_config
        captured.append(kwargs.get("trigger_oauth"))
        return {}, None, set()

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager._prepare_headers_and_auth",
        _fake_prepare_headers_and_auth,
    )
    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.tracking_sse_client",
        lambda *args, **kwargs: object(),
    )

    server_config = MCPServerSettings(transport="sse", url="http://example.com/sse")
    ctx = create_transport_context(server_name="test_server", config=server_config)

    assert ctx is not None
    assert captured == [False]


def test_nonpersistent_transport_honors_explicit_oauth_config(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[bool | None] = []

    def _fake_prepare_headers_and_auth(server_config, **kwargs):
        del server_config
        captured.append(kwargs.get("trigger_oauth"))
        return {}, None, set()

    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager._prepare_headers_and_auth",
        _fake_prepare_headers_and_auth,
    )
    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.tracking_sse_client",
        lambda *args, **kwargs: object(),
    )

    server_config = MCPServerSettings(
        transport="sse",
        url="http://example.com/sse",
        auth=MCPServerAuthSettings(oauth=True),
    )
    ctx = create_transport_context(server_name="test_server", config=server_config)

    assert ctx is not None
    assert captured == [True]
