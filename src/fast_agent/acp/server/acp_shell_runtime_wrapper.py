"""
ACPShellRuntimeWrapper - Wraps ACPTerminalExecutor to provide ShellRuntime-compatible interface.

This wrapper allows ACP terminal execution to be injected into agents that expect
a ShellRuntime interface.
"""

from typing import Any

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


class ACPShellRuntimeWrapper:
    """
    Provides a ShellRuntime-compatible interface that delegates to ACPTerminalExecutor.

    This allows seamless integration with existing agent code that expects a ShellRuntime,
    but actually executes commands via ACP terminal protocol on the client side.
    """

    def __init__(self, terminal_executor):
        """
        Initialize the wrapper.

        Args:
            terminal_executor: An ACPTerminalExecutor instance
        """
        self.terminal_executor = terminal_executor
        self.tool = terminal_executor.tool  # Expose the tool for agent routing

        logger.info(
            "ACPShellRuntimeWrapper initialized",
            name="acp_shell_runtime_wrapper_init",
            session_id=terminal_executor.session_id,
        )

    async def execute(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Execute a command via ACP terminal protocol.

        This method matches the ShellRuntime.execute() signature and delegates
        to the ACPTerminalExecutor.

        Args:
            arguments: Tool arguments (command, cwd, etc.)

        Returns:
            MCP tool result
        """
        logger.debug(
            "Delegating execute call to ACP terminal executor",
            name="acp_shell_runtime_wrapper_execute",
            session_id=self.terminal_executor.session_id,
        )

        return await self.terminal_executor.execute(arguments)
