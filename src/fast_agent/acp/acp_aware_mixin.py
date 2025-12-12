"""
ACPAwareMixin - Mixin providing convenient ACP access for agents.

This mixin extends ContextDependent to provide easy access to ACP features
when running in ACP mode. Agents inheriting this mixin can:

- Check if they're running in ACP mode
- Access ACP capabilities (terminal, filesystem, etc.)
- Switch modes programmatically
- Define slash commands declaratively via acp_commands property
- Query client capabilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

from fast_agent.context_dependent import ContextDependent

if TYPE_CHECKING:
    from fast_agent.acp.acp_context import ACPContext


# Type alias for slash command handlers
SlashCommandHandlerFunc = Callable[[str], Awaitable[str]]


@dataclass
class ACPCommand:
    """
    Defines a slash command that an agent can expose via ACP.

    Agents declare their commands by overriding the `acp_commands` property
    to return a dict mapping command names to ACPCommand instances.

    Example:
        @property
        def acp_commands(self) -> dict[str, ACPCommand]:
            return {
                "analyze": ACPCommand(
                    description="Analyze the current context",
                    handler=self._handle_analyze,
                ),
            }
    """

    description: str
    handler: SlashCommandHandlerFunc
    input_hint: str | None = None


@dataclass
class ACPModeInfo:
    """
    Optional display metadata for ACP modes.

    Agents can override `acp_mode_info()` to provide a custom name and/or
    description that will be surfaced to ACP clients.
    """

    name: str | None = None
    description: str | None = None


class ACPAwareMixin(ContextDependent):
    """
    Mixin providing convenient ACP access for agents.

    This mixin builds on ContextDependent to provide easy access to ACP features.
    It checks for the presence of an ACPContext in the application Context and
    provides convenient properties and methods for ACP operations.

    Usage:
        class MyAgent(ACPAwareMixin, McpAgent):
            async def initialize(self):
                await super().initialize()

                # Check if running in ACP mode
                if self.is_acp_mode:
                    # Register a dynamic slash command
                    await self.acp_add_command(
                        name="analyze",
                        description="Run analysis",
                        handler=self._handle_analyze
                    )

            async def generate_impl(self, request):
                if self.is_acp_mode:
                    # Check client capabilities
                    if self.acp.supports_terminal:
                        ...

                    # Get current mode
                    print(f"Current mode: {self.acp_current_mode}")

                    # Maybe switch modes
                    if should_delegate:
                        await self.acp_switch_mode("specialist_agent")

                return await super().generate_impl(request)

            async def _handle_analyze(self, arguments: str) -> str:
                return f"Analysis result for: {arguments}"
    """

    @property
    def acp(self) -> "ACPContext | None":
        """
        Get the ACP context if available.

        Returns:
            ACPContext if running in ACP mode, None otherwise
        """
        try:
            ctx = self.context
            return getattr(ctx, "acp", None)
        except RuntimeError:
            # No context available
            return None

    @property
    def is_acp_mode(self) -> bool:
        """
        Check if the agent is running in ACP mode.

        Returns:
            True if ACP context is available, False otherwise
        """
        return self.acp is not None

    # =========================================================================
    # Mode Management Shortcuts
    # =========================================================================

    @property
    def acp_current_mode(self) -> str | None:
        """
        Get the current ACP mode (agent ID).

        Returns:
            Current mode ID, or None if not in ACP mode
        """
        acp = self.acp
        return acp.current_mode if acp else None

    @property
    def acp_session_id(self) -> str | None:
        """
        Get the ACP session ID.

        Returns:
            Session ID, or None if not in ACP mode
        """
        acp = self.acp
        return acp.session_id if acp else None

    async def acp_switch_mode(self, mode_id: str) -> bool:
        """
        Switch to a different mode (agent).

        This sends a CurrentModeUpdate notification to the ACP client,
        telling it that the agent has autonomously switched modes.

        Args:
            mode_id: The mode ID to switch to

        Returns:
            True if switch was successful, False if not in ACP mode

        Raises:
            ValueError: If the mode_id is not in available modes
        """
        acp = self.acp
        if not acp:
            return False

        await acp.switch_mode(mode_id)
        return True

    def acp_available_modes(self) -> list[str]:
        """
        Get list of available mode IDs.

        Returns:
            List of mode IDs, or empty list if not in ACP mode
        """
        acp = self.acp
        return list(acp.available_modes.keys()) if acp else []

    # =========================================================================
    # Client Capability Shortcuts
    # =========================================================================

    @property
    def acp_supports_terminal(self) -> bool:
        """Check if ACP client supports terminal operations."""
        acp = self.acp
        return acp.supports_terminal if acp else False

    @property
    def acp_supports_fs_read(self) -> bool:
        """Check if ACP client supports file reading."""
        acp = self.acp
        return acp.supports_fs_read if acp else False

    @property
    def acp_supports_fs_write(self) -> bool:
        """Check if ACP client supports file writing."""
        acp = self.acp
        return acp.supports_fs_write if acp else False

    @property
    def acp_supports_filesystem(self) -> bool:
        """Check if ACP client supports any filesystem operations."""
        acp = self.acp
        return acp.supports_filesystem if acp else False

    # =========================================================================
    # Slash Commands (Declarative)
    # =========================================================================

    @property
    def acp_commands(self) -> dict[str, "ACPCommand"]:
        """
        Declare slash commands this agent exposes via ACP.

        Override this property to provide agent-specific commands.
        Commands are queried dynamically when the agent is the active mode.

        Returns:
            Dict mapping command names (without slash) to ACPCommand instances.

        Example:
            @property
            def acp_commands(self) -> dict[str, ACPCommand]:
                return {
                    "analyze": ACPCommand(
                        description="Analyze the current context",
                        handler=self._handle_analyze,
                    ),
                    "report": ACPCommand(
                        description="Generate a report",
                        handler=self._handle_report,
                        input_hint="<format>",
                    ),
                }
        """
        return {}

    def acp_mode_info(self) -> ACPModeInfo | None:
        """
        Optional ACP mode metadata (name/description) for display in clients.
        Override to customize the label and description for this agent's mode.
        """
        return None

    # =========================================================================
    # Runtime Access Shortcuts
    # =========================================================================

    @property
    def acp_terminal_runtime(self):
        """
        Get the ACP terminal runtime (if available).

        Returns:
            ACPTerminalRuntime or None
        """
        acp = self.acp
        return acp.terminal_runtime if acp else None

    @property
    def acp_filesystem_runtime(self):
        """
        Get the ACP filesystem runtime (if available).

        Returns:
            ACPFilesystemRuntime or None
        """
        acp = self.acp
        return acp.filesystem_runtime if acp else None

    @property
    def acp_permission_handler(self):
        """
        Get the ACP permission handler (if available).

        Returns:
            ACPToolPermissionAdapter or None
        """
        acp = self.acp
        return acp.permission_handler if acp else None

    @property
    def acp_progress_manager(self):
        """
        Get the ACP progress manager (if available).

        Returns:
            ACPToolProgressManager or None
        """
        acp = self.acp
        return acp.progress_manager if acp else None
