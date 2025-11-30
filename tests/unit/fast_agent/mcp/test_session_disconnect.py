"""Tests for session disconnect detection and reconnection handling."""

import asyncio
from types import SimpleNamespace

from fast_agent.config import MCPServerSettings
from fast_agent.core.exceptions import SessionDisconnectError


def _is_session_disconnect_error(error: Exception) -> bool:
    """Standalone implementation of session disconnect detection for testing.

    This mirrors the logic in MCPAgentClientSession._is_session_disconnect_error
    to allow unit testing without instantiating the full session.

    The MCP SDK converts HTTP 404 responses to JSONRPCError with code 32600
    and message 'Session terminated', which becomes an McpError.
    """
    try:
        from mcp.shared.exceptions import McpError

        if isinstance(error, McpError):
            error_data = getattr(error, "error", None)
            if error_data:
                code = getattr(error_data, "code", None)
                message = getattr(error_data, "message", "").lower()
                if code == 32600 or "session terminated" in message:
                    return True
    except ImportError:
        pass
    return False


class TestSessionDisconnectConfig:
    """Test cases for the reconnect_on_disconnect config option."""

    def test_reconnect_on_disconnect_defaults_to_false(self):
        """By default, reconnect_on_disconnect should be disabled."""
        config = MCPServerSettings()
        assert config.reconnect_on_disconnect is False

    def test_reconnect_on_disconnect_can_be_enabled(self):
        """reconnect_on_disconnect can be explicitly enabled."""
        config = MCPServerSettings(reconnect_on_disconnect=True)
        assert config.reconnect_on_disconnect is True

    def test_reconnect_on_disconnect_with_http_transport(self):
        """reconnect_on_disconnect works with HTTP transport."""
        config = MCPServerSettings(
            url="http://example.com/mcp",
            reconnect_on_disconnect=True,
        )
        assert config.transport == "http"
        assert config.reconnect_on_disconnect is True

    def test_reconnect_on_disconnect_preserves_other_settings(self):
        """reconnect_on_disconnect doesn't affect other settings."""
        config = MCPServerSettings(
            name="test-server",
            url="http://example.com/mcp",
            reconnect_on_disconnect=True,
            include_instructions=False,
            read_timeout_seconds=60,
        )
        assert config.name == "test-server"
        assert config.reconnect_on_disconnect is True
        assert config.include_instructions is False
        assert config.read_timeout_seconds == 60


class TestSessionDisconnectError:
    """Test cases for the SessionDisconnectError exception."""

    def test_exception_message_format(self):
        """SessionDisconnectError has proper message format."""
        error = SessionDisconnectError("my-server")
        assert "my-server" in str(error)
        assert "session disconnected" in str(error).lower()

    def test_exception_with_details(self):
        """SessionDisconnectError includes details when provided."""
        error = SessionDisconnectError(
            "my-server",
            details="Server returned HTTP 404",
        )
        assert "my-server" in str(error)
        assert "HTTP 404" in str(error)

    def test_exception_inherits_from_fastagent_error(self):
        """SessionDisconnectError is a FastAgentError subclass."""
        from fast_agent.core.exceptions import FastAgentError

        error = SessionDisconnectError("server")
        assert isinstance(error, FastAgentError)


class TestSessionDisconnectDetection:
    """Test cases for detecting session termination errors."""

    def test_is_session_disconnect_error_with_mcp_error_code_32600(self):
        """Detect MCP error with code 32600 (session terminated)."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        # Create a real McpError with session terminated error data
        error_data = ErrorData(code=32600, message="Session terminated")
        error = McpError(error_data)

        result = _is_session_disconnect_error(error)
        assert result is True

    def test_is_session_disconnect_error_with_different_code(self):
        """Don't detect non-session-terminated errors."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        # Different error code should not be detected
        error_data = ErrorData(code=32601, message="Method not found")
        error = McpError(error_data)

        result = _is_session_disconnect_error(error)
        assert result is False

    def test_is_session_disconnect_error_with_regular_exception(self):
        """Regular exceptions are not session disconnect errors."""
        error = ValueError("Something went wrong")
        result = _is_session_disconnect_error(error)
        assert result is False

    def test_is_session_disconnect_error_with_connection_error(self):
        """ConnectionError is not treated as session disconnect."""
        error = ConnectionError("Connection refused")
        result = _is_session_disconnect_error(error)
        assert result is False

    def test_is_session_disconnect_by_message_content(self):
        """Detect session termination by message content even with different code."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        # Message containing "session terminated" should be detected
        error_data = ErrorData(code=12345, message="Session terminated unexpectedly")
        error = McpError(error_data)

        result = _is_session_disconnect_error(error)
        assert result is True

    def test_is_session_disconnect_case_insensitive(self):
        """Session termination detection is case insensitive."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        # Upper case should also work
        error_data = ErrorData(code=12345, message="SESSION TERMINATED")
        error = McpError(error_data)

        result = _is_session_disconnect_error(error)
        assert result is True


class TestGetServerConfig:
    """Test cases for _get_server_config helper method."""

    def test_get_server_config_without_context(self):
        """Return None when no context is available."""
        from fast_agent.mcp.mcp_aggregator import MCPAggregator

        aggregator = MCPAggregator(
            server_names=["test"],
            connection_persistence=False,
            context=None,
        )
        aggregator.display = None  # Skip display setup

        result = asyncio.run(aggregator._get_server_config("test"))
        assert result is None

    def test_get_server_config_with_registry(self):
        """Return config from registry when available."""
        from fast_agent.mcp.mcp_aggregator import MCPAggregator

        # Create a stub registry
        expected_config = MCPServerSettings(name="test-server")

        class StubRegistry:
            def get_server_config(self, name):
                return expected_config if name == "test" else None

        # Create stub context
        context = SimpleNamespace(server_registry=StubRegistry())

        aggregator = MCPAggregator(
            server_names=["test"],
            connection_persistence=False,
            context=context,
        )
        aggregator.display = None

        result = asyncio.run(aggregator._get_server_config("test"))
        assert result is expected_config
