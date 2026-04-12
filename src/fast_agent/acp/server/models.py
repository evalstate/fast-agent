from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fast_agent.acp.acp_context import ACPContext
    from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
    from fast_agent.acp.slash_commands import SlashCommandHandler
    from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
    from fast_agent.acp.tool_permission_adapter import ACPToolPermissionAdapter
    from fast_agent.acp.tool_progress import ACPToolProgressManager
    from fast_agent.config import MCPServerSettings
    from fast_agent.core.fastagent import AgentInstance


@dataclass
class SessionMCPServerState:
    server_name: str
    server_config: MCPServerSettings | None = None
    attached: bool = True


@dataclass
class ACPSessionState:
    """Aggregated per-session ACP state for easier lifecycle management."""

    session_id: str
    instance: AgentInstance
    session_cwd: str | None = None
    session_store_scope: Literal["workspace", "app"] = "workspace"
    session_store_cwd: str | None = None
    current_agent_name: str | None = None
    progress_manager: ACPToolProgressManager | None = None
    permission_handler: ACPToolPermissionAdapter | None = None
    terminal_runtime: ACPTerminalRuntime | None = None
    filesystem_runtime: ACPFilesystemRuntime | None = None
    slash_handler: SlashCommandHandler | None = None
    acp_context: ACPContext | None = None
    prompt_context: dict[str, str] = field(default_factory=dict)
    resolved_instructions: dict[str, str] = field(default_factory=dict)
    session_mcp_servers: dict[str, SessionMCPServerState] = field(default_factory=dict)
    agent_mcp_servers: dict[str, dict[str, SessionMCPServerState]] = field(default_factory=dict)
