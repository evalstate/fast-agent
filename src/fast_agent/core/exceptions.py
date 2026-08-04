"""
Custom exceptions for the FastAgent framework.
Enables user-friendly error handling for common issues.
"""

from collections.abc import Iterator


def walk_exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception, nested groups, and chained causes once."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)


class FastAgentError(Exception):
    """Base exception class for FastAgent errors"""

    def __init__(self, message: str, details: str = "") -> None:
        self.message = message
        self.details = details
        super().__init__(f"{message}\n\n{details}" if details else message)


def format_fast_agent_error(error: "FastAgentError", *, single_line: bool = True) -> str:
    message = f"{error.message}: {error.details}" if error.details else error.message
    if not single_line:
        return message
    return " ".join(message.splitlines())


class ServerConfigError(FastAgentError):
    """Raised when there are issues with MCP server configuration
    Example: Server name referenced in agent.servers[] but not defined in config
    """

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message, details)


class ConfigFileError(FastAgentError):
    """Raised when a YAML-backed config file cannot be loaded or parsed."""

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


class ProviderSafetyBufferingError(FastAgentError):
    """Raised when a provider withholds a response for additional safety checks."""

    def __init__(
        self,
        model: str,
        retry_model: str | None = None,
        *,
        reasons: list[str] | None = None,
        use_cases: list[str] | None = None,
    ) -> None:
        self.model = model
        self.retry_model = retry_model
        self.reasons = reasons
        self.use_cases = use_cases
        retry_hint = (
            f"Retry with the suggested faster model '{retry_model}' or choose another model."
            if retry_model
            else "Retry with another model."
        )
        super().__init__(f"Codex safety-buffered the request for model '{model}'. {retry_hint}")


class ServerInitializationError(FastAgentError):
    """Raised when a server fails to initialize properly."""

    def __init__(
        self,
        message: str,
        details: str = "",
        *,
        server_name: str | None = None,
    ) -> None:
        self.server_name = server_name
        super().__init__(message, details)


class EnvironmentStartupError(FastAgentError):
    """Raised when an execution environment fails to start or attach."""

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

    def __init__(self, server_name: str, details: str = "") -> None:
        self.server_name = server_name
        message = f"MCP server '{server_name}' session terminated"
        super().__init__(message, details)
