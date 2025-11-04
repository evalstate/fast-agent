"""Tests for automatic transport inference in MCPServerSettings."""

import warnings

from fast_agent.config import MCPServerSettings


def test_transport_inference_url_only():
    """Test that providing only a URL infers HTTP transport."""
    config = MCPServerSettings(url="http://example.com/mcp")
    assert config.transport == "http"
    assert config.url == "http://example.com/mcp"
    assert config.command is None


def test_transport_inference_command_only():
    """Test that providing only a command keeps stdio transport."""
    config = MCPServerSettings(command="npx server")
    assert config.transport == "stdio"
    assert config.command == "npx server"
    assert config.url is None


def test_transport_inference_both_url_and_command():
    """Test that providing both URL and command prefers HTTP and warns."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = MCPServerSettings(url="http://example.com/mcp", command="npx server")

        # Check that warning was issued
        assert len(w) == 1
        assert "both 'url'" in str(w[0].message)
        assert "Preferring HTTP transport" in str(w[0].message)

        # Check that HTTP transport is selected and command is cleared
        assert config.transport == "http"
        assert config.url == "http://example.com/mcp"
        assert config.command is None  # Should be cleared


def test_transport_inference_explicit_transport_overrides():
    """Test that explicitly setting transport overrides inference."""
    # URL with explicit SSE transport should keep SSE
    config = MCPServerSettings(url="http://example.com/sse", transport="sse")
    assert config.transport == "sse"
    assert config.url == "http://example.com/sse"

    # Command with explicit HTTP transport should keep HTTP
    config = MCPServerSettings(command="npx server", transport="http")
    assert config.transport == "http"
    assert config.command == "npx server"


def test_transport_inference_empty_url():
    """Test that empty URL doesn't trigger HTTP inference."""
    config = MCPServerSettings(url="")
    assert config.transport == "stdio"
    assert config.url == ""

    config = MCPServerSettings(url="   ")
    assert config.transport == "stdio"
    assert config.url == "   "


def test_transport_inference_empty_command():
    """Test that empty command doesn't affect inference."""
    config = MCPServerSettings(command="")
    assert config.transport == "stdio"
    assert config.command == ""

    config = MCPServerSettings(command="   ")
    assert config.transport == "stdio"
    assert config.command == "   "


def test_transport_inference_neither_url_nor_command():
    """Test that providing neither URL nor command keeps default stdio."""
    config = MCPServerSettings()
    assert config.transport == "stdio"
    assert config.url is None
    assert config.command is None


def test_transport_inference_both_empty():
    """Test behavior when both URL and command are empty."""
    config = MCPServerSettings(url="", command="")
    assert config.transport == "stdio"
    assert config.url == ""
    assert config.command == ""


def test_transport_inference_url_with_empty_command():
    """Test URL with empty command should infer HTTP."""
    config = MCPServerSettings(url="http://example.com/mcp", command="")
    assert config.transport == "http"
    assert config.url == "http://example.com/mcp"
    assert config.command == ""


def test_transport_inference_command_with_empty_url():
    """Test command with empty URL should keep stdio."""
    config = MCPServerSettings(command="npx server", url="")
    assert config.transport == "stdio"
    assert config.command == "npx server"
    assert config.url == ""


def test_transport_inference_both_with_whitespace():
    """Test that whitespace-only values don't trigger inference conflicts."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = MCPServerSettings(
            url="http://example.com/mcp",
            command="   ",  # Only whitespace
        )

        # Should not warn because whitespace-only command is considered empty
        assert len(w) == 0
        assert config.transport == "http"
        assert config.url == "http://example.com/mcp"
        assert config.command == "   "


def test_transport_inference_preserves_other_fields():
    """Test that inference doesn't affect other configuration fields."""
    config = MCPServerSettings(
        url="http://example.com/mcp",
        name="test_server",
        description="A test server",
        args=["--verbose"],
        read_timeout_seconds=30,
        env={"KEY": "value"},
    )

    assert config.transport == "http"
    assert config.name == "test_server"
    assert config.description == "A test server"
    assert config.args == ["--verbose"]
    assert config.read_timeout_seconds == 30
    assert config.env == {"KEY": "value"}
