"""
Custom exceptions for the FastAgent framework.
Enables user-friendly error handling for common issues.
"""


class FastAgentError(Exception):
    """Base exception class for FastAgent errors"""

    def __init__(self, message: str, details: str = "") -> None:
        self.message = message
        self.details = details
        super().__init__(f"{message}\n\n{details}" if details else message)


class ServerConfigError(FastAgentError):
    """Raised when there are issues with MCP server configuration
    Example: Server name referenced in agent.servers[] but not defined in config
    """

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message, details)


class AgentConfigError(FastAgentError):
    """Raised when there are issues with Agent or Workflow configuration
    Example: Parallel fan-in references unknown agent
    """

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message, details)


class ProviderKeyError(FastAgentError):
    """Raised when there are issues with LLM provider API keys
    Example: OpenAI/Anthropic key not configured but model requires it
    """

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message, details)


class ServerInitializationError(FastAgentError):
    """Raised when a server fails to initialize properly."""

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message, details)


class ModelConfigError(FastAgentError):
    """Raised when there are issues with LLM model configuration
    Example: Unknown model name in model specification string
    """

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message, details)


class CircularDependencyError(FastAgentError):
    """Raised when we detect a Circular Dependency in the workflow"""

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message, details)


class PromptExitError(FastAgentError):
    """Raised from enhanced_prompt when the user requests hard exits"""

    # TODO an exception for flow control :(
    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message, details)


class ServerSessionTerminatedError(FastAgentError):
    """Raised when a server session has been terminated (e.g., 404 from server).

    This typically occurs when a remote StreamableHTTP server restarts and the
    session is no longer valid. When reconnect_on_disconnect is enabled, this
    error triggers automatic reconnection.
    """

    # Error code for session terminated from MCP SDK streamable_http.py
    # Note: The SDK uses positive 32600 (not the standard JSON-RPC -32600)
    # See: https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/streamable_http.py
    SESSION_TERMINATED_CODE = 32600

    def __init__(self, server_name: str, details: str = "") -> None:
        self.server_name = server_name
        message = f"MCP server '{server_name}' session terminated"
        super().__init__(message, details)
