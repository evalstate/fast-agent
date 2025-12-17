"""
ACPContext - Centralized ACP runtime state for context-aware agents.

Provides a unified interface for agents to access ACP capabilities including:
- Session information and mode management
- Terminal and filesystem runtimes
- Tool permissions and progress tracking
- Slash command management
- Client capability queries
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from acp.schema import (
    AvailableCommandsUpdate,
    CurrentModeUpdate,
    SessionMode,
)

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from acp import AgentSideConnection

    from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
    from fast_agent.acp.slash_commands import SlashCommandHandler
    from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
    from fast_agent.acp.tool_permission_adapter import ACPToolPermissionAdapter
    from fast_agent.acp.tool_progress import ACPToolProgressManager

logger = get_logger(__name__)


@dataclass
class ClientCapabilities:
    """Client capabilities from ACP initialization."""

    terminal: bool = False
    fs_read: bool = False
    fs_write: bool = False
    _meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_acp_capabilities(cls, caps: Any) -> "ClientCapabilities":
        """Create from ACP ClientCapabilities object."""
        result = cls()
        if caps is None:
            return result

        result.terminal = bool(getattr(caps, "terminal", False))

        if hasattr(caps, "fs") and caps.fs:
            fs_caps = caps.fs
            result.fs_read = bool(getattr(fs_caps, "readTextFile", False))
            result.fs_write = bool(getattr(fs_caps, "writeTextFile", False))

        if hasattr(caps, "_meta") and caps._meta:
            result._meta = dict(caps._meta) if isinstance(caps._meta, dict) else {}

        return result


@dataclass
class ClientInfo:
    """Client information from ACP initialization."""

    name: str = "unknown"
    version: str = "unknown"
    title: str | None = None

    @classmethod
    def from_acp_info(cls, info: Any) -> "ClientInfo":
        """Create from ACP Implementation object."""
        if info is None:
            return cls()
        return cls(
            name=getattr(info, "name", "unknown"),
            version=getattr(info, "version", "unknown"),
            title=getattr(info, "title", None),
        )


class ACPContext:
    """
    Centralized ACP runtime context.

    This class provides agents with access to all ACP-related capabilities
    when running in ACP mode. It centralizes:
    - Session and connection state
    - Mode management (current mode, switching)
    - Runtimes (terminal, filesystem)
    - Handlers (permissions, progress, slash commands)
    - Client capabilities

    Usage:
        if agent.is_acp_mode:
            # Check capabilities
            if agent.acp.supports_terminal:
                ...

            # Access current mode
            current = agent.acp.current_mode

            # Switch modes
            await agent.acp.switch_mode("specialist_agent")
    """

    def __init__(
        self,
        connection: "AgentSideConnection",
        session_id: str,
        *,
        client_capabilities: ClientCapabilities | None = None,
        client_info: ClientInfo | None = None,
        protocol_version: int | None = None,
    ) -> None:
        """
        Initialize the ACP context.

        Args:
            connection: The ACP connection for sending requests/notifications
            session_id: The ACP session ID
            client_capabilities: Client capabilities from initialization
            client_info: Client information from initialization
            protocol_version: ACP protocol version
        """
        self._connection = connection
        self._session_id = session_id
        self._client_capabilities = client_capabilities or ClientCapabilities()
        self._client_info = client_info or ClientInfo()
        self._protocol_version = protocol_version

        # Mode management
        self._current_mode: str = "default"
        self._available_modes: dict[str, SessionMode] = {}

        # Runtimes (set by AgentACPServer during session setup)
        self._terminal_runtime: "ACPTerminalRuntime | None" = None
        self._filesystem_runtime: "ACPFilesystemRuntime | None" = None

        # Handlers (set by AgentACPServer during session setup)
        self._permission_handler: "ACPToolPermissionAdapter | None" = None
        self._progress_manager: "ACPToolProgressManager | None" = None
        self._slash_handler: "SlashCommandHandler | None" = None

        # Reference to session-level resolved instructions cache
        # (shared with ACPSessionState.resolved_instructions)
        self._resolved_instructions: dict[str, str] | None = None

        # Lock for async operations
        self._lock = asyncio.Lock()

        logger.debug(
            "ACPContext initialized",
            name="acp_context_init",
            session_id=session_id,
            supports_terminal=self._client_capabilities.terminal,
            supports_fs_read=self._client_capabilities.fs_read,
            supports_fs_write=self._client_capabilities.fs_write,
        )

    # =========================================================================
    # Properties - Session Info
    # =========================================================================

    @property
    def session_id(self) -> str:
        """Get the ACP session ID."""
        return self._session_id

    @property
    def connection(self) -> "AgentSideConnection":
        """Get the ACP connection (for advanced use cases)."""
        return self._connection

    @property
    def protocol_version(self) -> int | None:
        """Get the ACP protocol version."""
        return self._protocol_version

    # =========================================================================
    # Properties - Client Info
    # =========================================================================

    @property
    def client_info(self) -> ClientInfo:
        """Get client information."""
        return self._client_info

    @property
    def client_capabilities(self) -> ClientCapabilities:
        """Get client capabilities."""
        return self._client_capabilities

    @property
    def supports_terminal(self) -> bool:
        """Check if the client supports terminal operations."""
        return self._client_capabilities.terminal

    @property
    def supports_fs_read(self) -> bool:
        """Check if the client supports file reading."""
        return self._client_capabilities.fs_read

    @property
    def supports_fs_write(self) -> bool:
        """Check if the client supports file writing."""
        return self._client_capabilities.fs_write

    @property
    def supports_filesystem(self) -> bool:
        """Check if the client supports any filesystem operations."""
        return self._client_capabilities.fs_read or self._client_capabilities.fs_write

    # =========================================================================
    # Properties - Mode Management
    # =========================================================================

    @property
    def current_mode(self) -> str:
        """Get the current mode (agent) ID."""
        return self._current_mode

    @property
    def available_modes(self) -> dict[str, SessionMode]:
        """Get available modes (agents) for this session."""
        return self._available_modes.copy()

    def set_current_mode(self, mode_id: str) -> None:
        """
        Set the current mode (called by server when mode changes).

        Args:
            mode_id: The mode ID to set as current
        """
        self._current_mode = mode_id

    def set_available_modes(self, modes: list[SessionMode]) -> None:
        """
        Set available modes (called by server during session setup).

        Args:
            modes: List of available session modes
        """
        self._available_modes = {mode.id: mode for mode in modes}

    async def switch_mode(self, mode_id: str) -> None:
        """
        Force-switch to a different mode/agent.

        This sends a CurrentModeUpdate notification to the client,
        telling it that the agent has autonomously switched modes.

        Args:
            mode_id: The mode ID to switch to

        Raises:
            ValueError: If the mode_id is not in available modes
        """
        if mode_id not in self._available_modes:
            raise ValueError(
                f"Invalid mode ID '{mode_id}'. Available modes: {list(self._available_modes.keys())}"
            )

        async with self._lock:
            old_mode = self._current_mode
            self._current_mode = mode_id

            # Send CurrentModeUpdate notification to client
            mode_update = CurrentModeUpdate(
                session_update="current_mode_update",
                current_mode_id=mode_id,
            )

            try:
                await self._connection.session_update(
                    session_id=self._session_id,
                    update=mode_update,
                )

                # Keep server-side slash command routing and command lists consistent with
                # agent-initiated mode switches.
                if self._slash_handler:
                    try:
                        self._slash_handler.set_current_agent(mode_id)
                    except Exception:
                        pass
                await self.send_available_commands_update()

                logger.info(
                    "Mode switched via agent request",
                    name="acp_mode_switch",
                    session_id=self._session_id,
                    old_mode=old_mode,
                    new_mode=mode_id,
                )
            except Exception as e:
                # Revert on failure
                self._current_mode = old_mode
                logger.error(
                    f"Failed to switch mode: {e}",
                    name="acp_mode_switch_error",
                    exc_info=True,
                )
                raise

    # =========================================================================
    # Properties - Runtimes
    # =========================================================================

    @property
    def terminal_runtime(self) -> "ACPTerminalRuntime | None":
        """Get the terminal runtime (if available)."""
        return self._terminal_runtime

    @property
    def filesystem_runtime(self) -> "ACPFilesystemRuntime | None":
        """Get the filesystem runtime (if available)."""
        return self._filesystem_runtime

    def set_terminal_runtime(self, runtime: "ACPTerminalRuntime") -> None:
        """Set the terminal runtime (called by server)."""
        self._terminal_runtime = runtime

    def set_filesystem_runtime(self, runtime: "ACPFilesystemRuntime") -> None:
        """Set the filesystem runtime (called by server)."""
        self._filesystem_runtime = runtime

    # =========================================================================
    # Properties - Handlers
    # =========================================================================

    @property
    def permission_handler(self) -> "ACPToolPermissionAdapter | None":
        """Get the permission handler (if available)."""
        return self._permission_handler

    @property
    def progress_manager(self) -> "ACPToolProgressManager | None":
        """Get the progress manager (if available)."""
        return self._progress_manager

    @property
    def slash_handler(self) -> "SlashCommandHandler | None":
        """Get the slash command handler."""
        return self._slash_handler

    def set_permission_handler(self, handler: "ACPToolPermissionAdapter") -> None:
        """Set the permission handler (called by server)."""
        self._permission_handler = handler

    def set_progress_manager(self, manager: "ACPToolProgressManager") -> None:
        """Set the progress manager (called by server)."""
        self._progress_manager = manager

    def set_slash_handler(self, handler: "SlashCommandHandler") -> None:
        """Set the slash command handler (called by server)."""
        self._slash_handler = handler

    def set_resolved_instructions(self, resolved_instructions: dict[str, str]) -> None:
        """
        Set the reference to resolved instructions cache (called by server).

        This should point to the same dict as ACPSessionState.resolved_instructions
        so that updates are reflected in both places.
        """
        self._resolved_instructions = resolved_instructions

    # =========================================================================
    # Slash Command Updates
    # =========================================================================

    async def send_available_commands_update(self) -> None:
        """
        Send AvailableCommandsUpdate notification to client.

        Call this when the available commands may have changed (e.g., after mode switch).
        Commands are queried from the SlashCommandHandler which combines session
        commands with agent-specific commands.
        """
        if not self._slash_handler:
            return

        all_commands = self._slash_handler.get_available_commands()

        commands_update = AvailableCommandsUpdate(
            session_update="available_commands_update",
            available_commands=all_commands,
        )

        try:
            await self._connection.session_update(
                session_id=self._session_id,
                update=commands_update,
            )
            logger.debug(
                "Sent available_commands_update",
                name="acp_commands_update_sent",
                session_id=self._session_id,
                command_count=len(all_commands),
            )
        except Exception as e:
            logger.error(
                f"Error sending available_commands_update: {e}",
                name="acp_commands_update_error",
                exc_info=True,
            )

    # =========================================================================
    # Session Updates
    # =========================================================================

    async def send_session_update(self, update: Any) -> None:
        """
        Send a session update notification to the client.

        This is a low-level method for sending arbitrary session updates.
        Prefer using higher-level methods like switch_mode() when available.

        Args:
            update: The session update payload (must be a valid ACP session update type)
        """
        await self._connection.session_update(
            session_id=self._session_id,
            update=update,
        )

    async def invalidate_instruction_cache(
        self, agent_name: str, new_instruction: str | None
    ) -> None:
        """
        Invalidate the session instruction cache for an agent.

        Call this when an agent's system prompt has been rebuilt (e.g., after
        connecting new MCP servers) to ensure the ACP session uses the fresh
        instruction on subsequent prompts.

        Args:
            agent_name: Name of the agent whose instruction was updated
            new_instruction: The new resolved instruction (or None to remove)
        """
        async with self._lock:
            # Update the session-level resolved instructions cache
            # (used by _build_session_request_params in AgentACPServer)
            if self._resolved_instructions is not None:
                if new_instruction:
                    self._resolved_instructions[agent_name] = new_instruction
                elif agent_name in self._resolved_instructions:
                    del self._resolved_instructions[agent_name]

            # Update the SlashCommandHandler's session instructions cache
            # (used by /system command)
            if self._slash_handler:
                self._slash_handler.update_session_instruction(agent_name, new_instruction)

            logger.info(
                "Invalidated instruction cache for agent",
                name="acp_instruction_cache_invalidated",
                session_id=self._session_id,
                agent_name=agent_name,
            )

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup(self) -> None:
        """Clean up ACP context resources."""
        async with self._lock:
            # Clear permission cache if handler exists
            if self._permission_handler:
                try:
                    await self._permission_handler.clear_session_cache()
                except Exception as e:
                    logger.error(f"Error clearing permission cache: {e}", exc_info=True)

            logger.debug(
                "ACPContext cleaned up",
                name="acp_context_cleanup",
                session_id=self._session_id,
            )
