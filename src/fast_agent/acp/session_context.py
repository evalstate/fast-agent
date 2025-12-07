"""
ACP Session Context - provides agents with access to ACP infrastructure.

This module defines the ACPSessionContext class which gives agents a clean
interface to interact with ACP capabilities like:
- Registering temporary slash commands
- Sending session updates to the client
- Listening for and triggering mode changes
- Querying client capabilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from acp import Connection
    from acp.schema import SessionUpdate

    from fast_agent.acp.slash_commands import SlashCommandHandler
    from fast_agent.acp.tool_permission_adapter import ACPToolPermissionAdapter
    from fast_agent.acp.tool_progress import ACPToolProgressManager


@dataclass
class ACPSessionContext:
    """
    Context for agent interaction with ACP infrastructure.

    Provides agents with a clean interface to:
    - Register temporary slash commands
    - Send session updates to the client
    - Listen for and trigger mode changes
    - Access ACP-specific capabilities

    This context is injected into agents when running under ACP transport.
    When not running under ACP, agents will have acp_context = None.

    Example usage in an agent:

        if self.acp_context:
            # Register a custom command
            self.acp_context.register_command(
                "mystatus",
                self._handle_mystatus,
                "Show custom status"
            )

            # Listen for mode changes
            self.acp_context.on_mode_change(self._handle_mode_change)

            # Send a message to the client
            await self.acp_context.send_text("Processing complete!")
    """

    session_id: str
    _connection: Connection
    _server: Any  # AgentACPServer - use Any to avoid circular import

    # Components (may be None if not enabled)
    tool_handler: ACPToolProgressManager | None = None
    permission_handler: ACPToolPermissionAdapter | None = None
    slash_handler: SlashCommandHandler | None = None

    # Mode change callbacks: (session_id, old_mode, new_mode) -> None
    _mode_change_listeners: list[Callable[[str, str, str], Awaitable[None]]] = field(
        default_factory=list
    )

    # --- Slash Commands ---

    def register_command(
        self,
        name: str,
        handler: Callable[[str], Awaitable[str]],
        description: str,
        hint: str | None = None,
    ) -> Callable[[], None]:
        """
        Register a temporary slash command for this session.

        Args:
            name: Command name (without the leading /)
            handler: Async function that takes arguments string, returns response
            description: Short description shown in command list
            hint: Optional hint for command arguments (e.g., "<filename>")

        Returns:
            A function that unregisters the command when called.

        Example:
            async def handle_ping(args: str) -> str:
                return f"Pong! Args: {args}"

            unregister = ctx.register_command("ping", handle_ping, "Ping test")
            # Later:
            unregister()  # Remove the command
        """
        if self.slash_handler:
            return self.slash_handler.register_dynamic_command(
                name, handler, description, hint
            )
        return lambda: None

    def unregister_command(self, name: str) -> bool:
        """
        Unregister a previously registered dynamic command.

        Args:
            name: Command name to unregister

        Returns:
            True if command was found and removed, False otherwise.
        """
        if self.slash_handler:
            return self.slash_handler.unregister_dynamic_command(name)
        return False

    # --- Session Updates ---

    async def send_update(self, update: SessionUpdate) -> None:
        """
        Send an arbitrary session update to the client.

        Args:
            update: The SessionUpdate to send (from acp.schema)
        """
        await self._connection.session_update(
            session_id=self.session_id,
            update=update,
        )

    async def send_text(self, text: str) -> None:
        """
        Convenience method to send a text message update.

        Args:
            text: The text message to send to the client
        """
        from fast_agent.acp.server.agent_acp_server import update_agent_message_text

        await self.send_update(update_agent_message_text(text))

    # --- Mode Management ---

    @property
    def current_mode(self) -> str:
        """
        Get the current mode (agent name) for this session.

        Returns:
            The ID of the currently active mode/agent.
        """
        return self._server._session_current_agent.get(
            self.session_id,
            self._server.primary_agent_name,
        )

    @property
    def available_modes(self) -> list[str]:
        """
        Get list of available mode IDs for this session.

        Returns:
            List of agent names that can be switched to.
        """
        instance = self._server.sessions.get(self.session_id)
        return list(instance.agents.keys()) if instance else []

    async def set_mode(self, mode_id: str) -> None:
        """
        Programmatically change the current mode.

        This updates internal state and notifies registered listeners.
        Note: This does not send a notification to the client - the client
        initiates mode changes via setSessionMode.

        Args:
            mode_id: The ID of the mode/agent to switch to

        Raises:
            ValueError: If mode_id is not in available_modes
        """
        if mode_id not in self.available_modes:
            raise ValueError(
                f"Invalid mode ID: {mode_id}. "
                f"Available modes: {self.available_modes}"
            )

        old_mode = self.current_mode
        self._server._session_current_agent[self.session_id] = mode_id

        # Notify listeners
        for listener in self._mode_change_listeners:
            await listener(self.session_id, old_mode, mode_id)

    def on_mode_change(
        self,
        callback: Callable[[str, str, str], Awaitable[None]],
    ) -> Callable[[], None]:
        """
        Register a callback for mode changes.

        The callback receives (session_id, old_mode_id, new_mode_id).

        Args:
            callback: Async function to call when mode changes

        Returns:
            A function that unregisters the callback when called.

        Example:
            async def on_mode_changed(session_id, old_mode, new_mode):
                print(f"Mode changed from {old_mode} to {new_mode}")

            unregister = ctx.on_mode_change(on_mode_changed)
        """
        self._mode_change_listeners.append(callback)
        return lambda: self._mode_change_listeners.remove(callback)

    async def _notify_mode_change(self, old_mode: str, new_mode: str) -> None:
        """
        Internal method called by ACP server when mode changes.

        This notifies all registered listeners.
        """
        for listener in self._mode_change_listeners:
            await listener(self.session_id, old_mode, new_mode)

    # --- Client Capabilities ---

    @property
    def supports_terminal(self) -> bool:
        """Whether the ACP client supports terminal/command execution."""
        return self._server._client_supports_terminal

    @property
    def supports_filesystem_read(self) -> bool:
        """Whether the ACP client supports reading files."""
        return self._server._client_supports_fs_read

    @property
    def supports_filesystem_write(self) -> bool:
        """Whether the ACP client supports writing files."""
        return self._server._client_supports_fs_write

    @property
    def client_info(self) -> dict | None:
        """
        Get client information from ACP initialize.

        Returns dict with keys like 'name', 'version', etc.
        """
        return self._server._client_info

    @property
    def client_capabilities(self) -> dict | None:
        """Get raw client capabilities from ACP initialize."""
        return self._server._client_capabilities

    # --- Utility ---

    def __repr__(self) -> str:
        return (
            f"ACPSessionContext(session_id={self.session_id!r}, "
            f"current_mode={self.current_mode!r}, "
            f"modes={self.available_modes})"
        )
