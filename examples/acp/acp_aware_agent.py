"""
Example: ACP-Aware Agent

This example demonstrates how to create a custom agent that is aware of
the Agent Client Protocol (ACP) context. When running via ACP (e.g., through
Claude Code or another ACP client), the agent can:

- Detect it's running in ACP mode
- Access client capabilities (terminal, filesystem, etc.)
- Declare slash commands via the acp_commands property
- Switch modes to other agents
- Access terminal and filesystem runtimes provided by the client

When NOT running via ACP (e.g., directly via CLI), the agent gracefully
falls back to standard behavior.
"""

import asyncio
from typing import TYPE_CHECKING

from fast_agent import FastAgent
from fast_agent.acp import ACPAwareMixin, ACPCommand
from fast_agent.acp.acp_aware_mixin import ACPModeInfo
from fast_agent.agents import McpAgent

if TYPE_CHECKING:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.context import Context

# Create the FastAgent application
fast = FastAgent("ACP Aware Example")


class ACPAwareAgent(ACPAwareMixin, McpAgent):
    """
    A custom agent that is aware of the ACP context.

    This agent demonstrates how to use the ACPAwareMixin to access
    ACP features when available, while maintaining compatibility
    with non-ACP execution modes.
    """

    def __init__(
        self,
        config: "AgentConfig",
        context: "Context | None" = None,
        **kwargs,
    ) -> None:
        """Initialize the agent with proper MRO handling."""
        # Call McpAgent's __init__ directly to ensure proper initialization
        McpAgent.__init__(self, config=config, context=context, **kwargs)
        # Store context for ACPAwareMixin (inherited from ContextDependent)
        self._context = context

    @property
    def acp_commands(self) -> dict[str, ACPCommand]:
        """
        Declare slash commands for this agent.

        These commands are automatically available when this agent is the
        active mode in an ACP session. Commands are queried dynamically,
        so they update when the mode changes.
        """
        return {
            "agent-status": ACPCommand(
                description="Show ACP connection status for this agent",
                handler=self._handle_status_command,
            ),
            "capabilities": ACPCommand(
                description="List client capabilities",
                handler=self._handle_capabilities_command,
            ),
        }

    async def _handle_status_command(self, arguments: str) -> str:
        """Handler for the /agent-status slash command."""
        return (
            f"ACP Status:\n"
            f"  Session: {self.acp_session_id}\n"
            f"  Mode: {self.acp_current_mode}\n"
            f"  Available modes: {', '.join(self.acp_available_modes())}"
        )

    async def _handle_capabilities_command(self, arguments: str) -> str:
        """Handler for the /capabilities slash command."""
        caps = []
        if self.acp_supports_terminal:
            caps.append("terminal")
        if self.acp_supports_fs_read:
            caps.append("filesystem-read")
        if self.acp_supports_fs_write:
            caps.append("filesystem-write")

        if caps:
            return f"Client capabilities: {', '.join(caps)}"
        return "No special capabilities detected"


class ACPAwareAgent2(ACPAwareMixin, McpAgent):
    """
    A custom agent that is aware of the ACP context.

    This agent demonstrates how to use the ACPAwareMixin to access
    ACP features when available, while maintaining compatibility
    with non-ACP execution modes.
    """

    def __init__(
        self,
        config: "AgentConfig",
        context: "Context | None" = None,
        **kwargs,
    ) -> None:
        """Initialize the agent with proper MRO handling."""
        # Call McpAgent's __init__ directly to ensure proper initialization
        McpAgent.__init__(self, config=config, context=context, **kwargs)
        # Store context for ACPAwareMixin (inherited from ContextDependent)
        self._context = context

    @property
    def acp_commands(self) -> dict[str, ACPCommand]:
        """
        Declare slash commands for this agent.

        These commands are automatically available when this agent is the
        active mode in an ACP session. Commands are queried dynamically,
        so they update when the mode changes.
        """
        return {
            "foo": ACPCommand(
                input_hint="input hint",
                description="Show ACP connection status for this agent",
                handler=self._handle_foo_command,
            ),
            "bar": ACPCommand(
                description="List client capabilities",
                handler=self._handle_bar_command,
            ),
        }

    async def _handle_foo_command(self, arguments: str) -> str:
        """Handler for the /foo slash command."""
        return "FOO"

    async def _handle_bar_command(self, arguments: str) -> str:
        """Handler for the /bar slash command."""
        return "BAR"

    def acp_mode_info(self) -> ACPModeInfo | None:
        return ACPModeInfo(name="FooBar Agent", description="A custom agent with custom commands")


# Use the @fast.custom decorator to register our custom agent class
@fast.custom(
    ACPAwareAgent,
    name="acp_agent",
    instruction="""You are an ACP-aware assistant. When running via ACP (Agent Client Protocol),
you have access to additional capabilities provided by the client such as terminal
and filesystem access.

You can help users with tasks that leverage these capabilities when available.
When not running via ACP, you function as a standard helpful assistant.""",
    default=True,
)
@fast.custom(ACPAwareAgent2, name="another_agent", instruction="do it!")
async def main() -> None:
    """Run the ACP-aware agent."""
    async with fast.run() as agent:
        # In interactive mode, the agent will:
        # - Detect if running via ACP and configure itself accordingly
        # - Provide slash commands if running via ACP client
        # - Fall back to standard CLI behavior otherwise
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
