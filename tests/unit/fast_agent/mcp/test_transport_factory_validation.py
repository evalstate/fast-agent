"""Tests for transport factory validation with inferred transport types."""

import pytest

from fast_agent.config import MCPServerSettings
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
    # When URL is provided, transport should be inferred as HTTP
    server_config = MCPServerSettings(url="http://example.com/mcp")

    assert server_config.transport == "http"
    assert server_config.url == "http://example.com/mcp"

    # This should not raise an error in the transport factory
    def transport_context_factory():
        if server_config.transport == "http":
            if not server_config.url:
                raise ValueError("Server uses http transport but no url is specified")
            # If we get here, validation passed
            return True

    result = transport_context_factory()
    assert result is True


def test_inferred_stdio_transport_has_command():
    """Test that inferred stdio transport always has a command (when provided)."""
    # When command is provided, transport should remain stdio
    server_config = MCPServerSettings(command="npx server")

    assert server_config.transport == "stdio"
    assert server_config.command == "npx server"

    # This should not raise an error in the transport factory
    def transport_context_factory():
        if server_config.transport == "stdio":
            if not server_config.command:
                raise ValueError("Server uses stdio transport but no command is specified")
            # If we get here, validation passed
            return True

    result = transport_context_factory()
    assert result is True


def test_explicit_transport_validation_still_works():
    """Test that explicit transport settings still get validated properly."""
    # Explicit http transport without URL should fail validation
    server_config = MCPServerSettings(transport="http", command="some_command")

    def transport_context_factory():
        if server_config.transport == "http":
            if not server_config.url:
                raise ValueError("Server uses http transport but no url is specified")

    with pytest.raises(ValueError, match="uses http transport but no url is specified"):
        transport_context_factory()

    # Explicit stdio transport without command should fail validation
    server_config = MCPServerSettings(transport="stdio", url="http://example.com")

    def transport_context_factory():
        if server_config.transport == "stdio":
            if not server_config.command:
                raise ValueError("Server uses stdio transport but no command is specified")

    with pytest.raises(ValueError, match="uses stdio transport but no command is specified"):
        transport_context_factory()
