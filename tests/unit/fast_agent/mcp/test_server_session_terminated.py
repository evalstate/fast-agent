"""
Tests for server session termination handling and reconnection functionality.
"""

from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from fast_agent.config import MCPServerSettings
from fast_agent.core.exceptions import FastAgentError, ServerSessionTerminatedError
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession


class TestServerSessionTerminatedError:
    """Tests for the ServerSessionTerminatedError exception class."""

    def test_error_code_constant(self):
        """MCP SDK uses positive 32600 for session terminated."""
        assert ServerSessionTerminatedError.SESSION_TERMINATED_CODE == 32600

    def test_error_creation(self):
        """Exception captures server name and details."""
        error = ServerSessionTerminatedError(server_name="test-server", details="404")
        assert error.server_name == "test-server"
        assert "test-server" in str(error)

    def test_inherits_from_fast_agent_error(self):
        """Exception inherits from FastAgentError."""
        assert isinstance(ServerSessionTerminatedError(server_name="x"), FastAgentError)


class TestSessionTerminationDetection:
    """Tests for detecting session terminated errors."""

    def _make_session(self):
        session = object.__new__(MCPAgentClientSession)
        session.session_server_name = "test"
        return session

    def test_detects_mcp_error_code_32600(self):
        """Detects McpError with code 32600."""
        error = McpError(ErrorData(code=32600, message="Session terminated"))
        assert self._make_session()._is_session_terminated_error(error) is True

    def test_ignores_different_error_codes(self):
        """Ignores McpError with different codes."""
        error = McpError(ErrorData(code=-32601, message="Method not found"))
        assert self._make_session()._is_session_terminated_error(error) is False

    def test_ignores_non_mcp_errors(self):
        """Ignores non-McpError exceptions."""
        session = self._make_session()
        assert session._is_session_terminated_error(ValueError("test")) is False
        assert session._is_session_terminated_error(ConnectionError("test")) is False


class TestReconnectConfig:
    """Tests for reconnect_on_disconnect config option."""

    def test_defaults_to_false(self):
        """reconnect_on_disconnect defaults to False."""
        settings = MCPServerSettings(name="test", url="https://example.com/mcp")
        assert settings.reconnect_on_disconnect is False

    def test_can_be_enabled(self):
        """reconnect_on_disconnect can be set to True."""
        settings = MCPServerSettings(
            name="test", url="https://example.com/mcp", reconnect_on_disconnect=True
        )
        assert settings.reconnect_on_disconnect is True
