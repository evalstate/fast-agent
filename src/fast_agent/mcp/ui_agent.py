"""
MCP Agent with UI support using mixin pattern.

This module provides a concrete agent class that combines McpAgent
with UI functionality through the mixin pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fast_agent.agents import McpAgent
from fast_agent.mcp.ui_mixin import McpUIMixin

if TYPE_CHECKING:
    from fast_agent.context import Context


class McpAgentWithUI(McpUIMixin, McpAgent):
    """
    MCP Agent with UI resource handling capabilities.

    This class combines the base McpAgent functionality with UI resource
    processing using the mixin pattern. It's a clean, type-safe way to add
    UI functionality without the complexity of wrapper classes.

    Usage:
        agent = McpAgentWithUI(config, context=context, ui_mode="auto")
    """

    def __init__(
        self,
        config,
        context: "Context | None" = None,
        ui_mode: str = "auto",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the agent with UI capabilities.

        Args:
            config: Agent configuration
            context: Application context
            ui_mode: UI mode - "disabled", "enabled", or "auto"
            **kwargs: Additional arguments passed to parent classes
        """
        # Initialize both parent classes with the ui_mode parameter
        super().__init__(config=config, context=context, ui_mode=ui_mode, **kwargs)
