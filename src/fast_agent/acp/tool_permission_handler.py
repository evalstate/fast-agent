"""
Tool Permission Handler Protocol and Types.

Provides a clean interface for requesting tool execution permission
before tools run. Works with MCP tools, terminal, and filesystem.

This follows the same pattern as elicitation handlers - a protocol that
can be implemented by different backends (ACP, local UI, policy-based, etc.)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class PermissionDecision(Enum):
    """Result of a permission check."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass
class PermissionResult:
    """
    Result from a permission check.

    Attributes:
        decision: Whether to allow or deny the tool execution
        remember: Whether this decision should be persisted for future calls
        message: Optional message explaining the decision (for logging/display)
    """

    decision: PermissionDecision
    remember: bool = False
    message: str | None = None

    @property
    def allowed(self) -> bool:
        """Convenience property to check if execution is allowed."""
        return self.decision == PermissionDecision.ALLOW

    @classmethod
    def allow(cls, remember: bool = False) -> "PermissionResult":
        """Create an allow result."""
        return cls(decision=PermissionDecision.ALLOW, remember=remember)

    @classmethod
    def deny(cls, remember: bool = False, message: str | None = None) -> "PermissionResult":
        """Create a deny result."""
        return cls(decision=PermissionDecision.DENY, remember=remember, message=message)


# Standard denial message for rejected tool calls
PERMISSION_DENIED_MESSAGE = "The User declined this operation."


@runtime_checkable
class ToolPermissionHandler(Protocol):
    """
    Protocol for checking tool execution permission.

    Implementations can:
    - Request permission from users (e.g., via ACP client UI)
    - Check cached/persisted permissions
    - Apply policy rules
    - Auto-approve in non-interactive contexts

    The handler is called before tool execution. If it returns DENY,
    the tool is not executed and an error result is returned to the LLM.
    """

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None,
    ) -> PermissionResult:
        """
        Check if tool execution is permitted.

        This method is called before each tool execution. Implementations
        should be efficient (use caching) since this is on the hot path.

        Args:
            tool_name: Name of the tool (e.g., "execute", "write_text_file")
            server_name: Name of the server/runtime providing the tool
                        (e.g., "acp_terminal", "acp_filesystem", "mcp_server_name")
            arguments: Tool arguments (may be None)

        Returns:
            PermissionResult indicating whether to proceed and whether to
            remember the decision for future calls.
        """
        ...
