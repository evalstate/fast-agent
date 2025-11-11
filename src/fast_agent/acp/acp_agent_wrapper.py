"""
ACP Agent Wrapper for intercepting tool calls.

This module provides a wrapper that intercepts tool calls from agents
and sends ACP notifications.
"""

from typing import Any, Dict

from mcp.types import CallToolResult

from fast_agent.acp.tool_call_integration import (
    ACPToolCallMiddleware,
    create_tool_title,
    infer_tool_kind,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import AgentProtocol

logger = get_logger(__name__)


class ACPAgentWrapper:
    """
    Wraps an agent to intercept tool calls and send ACP notifications.

    This wrapper delegates all method calls to the underlying agent,
    but intercepts call_tool to send ACP tool call notifications.
    """

    def __init__(self, agent: AgentProtocol, middleware: ACPToolCallMiddleware):
        """
        Initialize the wrapper.

        Args:
            agent: The underlying agent to wrap
            middleware: Middleware for handling tool call notifications
        """
        self._wrapped_agent = agent
        self._middleware = middleware

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped agent."""
        return getattr(self._wrapped_agent, name)

    async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> CallToolResult:
        """
        Call a tool with ACP notifications.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool call result
        """
        # Infer tool kind and create title
        tool_kind = infer_tool_kind(name)
        tool_title = create_tool_title(name, arguments)

        # Wrap the tool call
        async def execute_fn():
            return await self._wrapped_agent.call_tool(name, arguments)

        return await self._middleware.wrap_tool_call(
            tool_name=name,
            arguments=arguments,
            execute_fn=execute_fn,
            tool_kind=tool_kind,
            tool_title=tool_title,
        )
