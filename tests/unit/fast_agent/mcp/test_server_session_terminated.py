"""
Tests for server session termination handling and reconnection functionality.

These tests verify the ServerSessionTerminatedError exception,
the session termination detection logic, and the config options.
"""

from fast_agent.config import MCPServerSettings
from fast_agent.core.exceptions import FastAgentError, ServerSessionTerminatedError


class TestServerSessionTerminatedError:
    """Tests for the ServerSessionTerminatedError exception class."""

    def test_error_creation_with_server_name(self):
        """Test that the exception captures the server name correctly."""
        error = ServerSessionTerminatedError(server_name="test-server")

        assert error.server_name == "test-server"
        assert "test-server" in str(error)
        assert "session terminated" in str(error).lower()

    def test_error_creation_with_details(self):
        """Test that the exception captures details correctly."""
        error = ServerSessionTerminatedError(
            server_name="my-server", details="Server returned 404"
        )

        assert error.server_name == "my-server"
        assert error.details == "Server returned 404"
        assert "Server returned 404" in str(error)

    def test_error_code_constant(self):
        """Test that the SESSION_TERMINATED_CODE constant is defined correctly."""
        # The MCP SDK uses -32600 for session terminated errors
        assert ServerSessionTerminatedError.SESSION_TERMINATED_CODE == -32600

    def test_error_inherits_from_fast_agent_error(self):
        """Test that the exception inherits from FastAgentError."""
        error = ServerSessionTerminatedError(server_name="test")
        assert isinstance(error, FastAgentError)


class TestMCPServerSettingsReconnect:
    """Tests for the reconnect_on_disconnect config option."""

    def test_reconnect_option_defaults_to_false(self):
        """Test that reconnect_on_disconnect defaults to False."""
        settings = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
        )

        assert settings.reconnect_on_disconnect is False

    def test_reconnect_option_can_be_enabled(self):
        """Test that reconnect_on_disconnect can be set to True."""
        settings = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
            reconnect_on_disconnect=True,
        )

        assert settings.reconnect_on_disconnect is True

    def test_reconnect_option_can_be_explicitly_disabled(self):
        """Test that reconnect_on_disconnect can be explicitly set to False."""
        settings = MCPServerSettings(
            name="test",
            transport="http",
            url="https://example.com/mcp",
            reconnect_on_disconnect=False,
        )

        assert settings.reconnect_on_disconnect is False

    def test_reconnect_option_with_stdio_transport(self):
        """Test that reconnect option works with stdio transport (though not useful)."""
        settings = MCPServerSettings(
            name="test",
            command="python",
            args=["server.py"],
            reconnect_on_disconnect=True,
        )

        # Setting is accepted even for stdio, but won't have effect
        assert settings.reconnect_on_disconnect is True
        assert settings.transport == "stdio"

    def test_reconnect_preserves_other_settings(self):
        """Test that reconnect option doesn't affect other settings."""
        settings = MCPServerSettings(
            name="my-server",
            url="https://example.com/mcp",
            reconnect_on_disconnect=True,
            include_instructions=False,
            read_timeout_seconds=60,
        )

        assert settings.name == "my-server"
        assert settings.reconnect_on_disconnect is True
        assert settings.include_instructions is False
        assert settings.read_timeout_seconds == 60


class TestSessionTerminationDetection:
    """Tests for detecting session terminated errors."""

    def test_detection_with_mcp_error_code_negative_32600(self):
        """Test that errors with code -32600 are detected as session terminated."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

        # Create an actual McpError with code -32600
        error_data = ErrorData(code=-32600, message="Session terminated")
        mcp_error = McpError(error_data)

        # Create a minimal session instance to test the method
        # We need to bypass the constructor since it requires streams
        session = object.__new__(MCPAgentClientSession)
        session.session_server_name = "test"

        # Test the detection method
        assert session._is_session_terminated_error(mcp_error) is True

    def test_detection_with_different_error_code(self):
        """Test that errors with different codes are not detected."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

        # Create an actual McpError with a different code
        error_data = ErrorData(code=-32601, message="Method not found")
        mcp_error = McpError(error_data)

        session = object.__new__(MCPAgentClientSession)
        session.session_server_name = "test"

        assert session._is_session_terminated_error(mcp_error) is False

    def test_detection_with_regular_exception(self):
        """Test that regular exceptions are not detected as session terminated."""
        from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

        session = object.__new__(MCPAgentClientSession)
        session.session_server_name = "test"

        assert session._is_session_terminated_error(ValueError("test")) is False
        assert session._is_session_terminated_error(ConnectionError("test")) is False
        assert session._is_session_terminated_error(TimeoutError("test")) is False

    def test_detection_with_error_without_matching_code(self):
        """Test that errors without a matching code are not detected."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

        # Create a McpError with a different code (not -32600)
        error_data = ErrorData(code=0, message="No specific code")
        mcp_error = McpError(error_data)

        session = object.__new__(MCPAgentClientSession)
        session.session_server_name = "test"

        assert session._is_session_terminated_error(mcp_error) is False

    def test_detection_by_message_fallback(self):
        """Test detection by 'session terminated' in message as fallback."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

        # Create an error with a different code but "session terminated" in message
        error_data = ErrorData(code=12345, message="Session terminated unexpectedly")
        mcp_error = McpError(error_data)

        session = object.__new__(MCPAgentClientSession)
        session.session_server_name = "test"

        assert session._is_session_terminated_error(mcp_error) is True

    def test_detection_case_insensitive(self):
        """Test that session termination detection is case insensitive."""
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

        # Upper case should also work
        error_data = ErrorData(code=12345, message="SESSION TERMINATED")
        mcp_error = McpError(error_data)

        session = object.__new__(MCPAgentClientSession)
        session.session_server_name = "test"

        assert session._is_session_terminated_error(mcp_error) is True


class TestConfigParsing:
    """Tests for parsing reconnect config from YAML-like dicts."""

    def test_config_from_dict_with_reconnect_enabled(self):
        """Test parsing config with reconnect_on_disconnect enabled."""
        config_dict = {
            "name": "remote-server",
            "url": "https://api.example.com/mcp",
            "reconnect_on_disconnect": True,
        }

        settings = MCPServerSettings(**config_dict)

        assert settings.name == "remote-server"
        assert settings.reconnect_on_disconnect is True
        assert settings.transport == "http"  # inferred from url

    def test_config_from_dict_without_reconnect(self):
        """Test parsing config without reconnect option (should default to False)."""
        config_dict = {
            "name": "remote-server",
            "url": "https://api.example.com/mcp",
        }

        settings = MCPServerSettings(**config_dict)

        assert settings.reconnect_on_disconnect is False
