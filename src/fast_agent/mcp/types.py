from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from fast_agent.interfaces import AgentProtocol

if TYPE_CHECKING:
    from rich.text import Text

    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context
    from fast_agent.mcp.experimental_session_client import ExperimentalSessionClient
    from fast_agent.mcp.mcp_aggregator import (
        MCPAggregator,
        MCPAttachOptions,
        MCPAttachResult,
        MCPDetachResult,
    )
    from fast_agent.skills import SkillManifest
    from fast_agent.skills.registry import SkillRegistry
    from fast_agent.tools.shell_runtime import ShellRuntime
    from fast_agent.ui.console_display import ConsoleDisplay


@runtime_checkable
class McpAgentProtocol(AgentProtocol, Protocol):
    """Agent protocol with MCP-specific surface area."""

    @property
    def aggregator(self) -> MCPAggregator: ...

    @property
    def experimental_sessions(self) -> "ExperimentalSessionClient": ...

    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        server_config: "MCPServerSettings | None" = None,
        options: "MCPAttachOptions | None" = None,
    ) -> "MCPAttachResult": ...

    async def detach_mcp_server(self, server_name: str) -> "MCPDetachResult": ...

    def list_attached_mcp_servers(self) -> list[str]: ...

    @property
    def display(self) -> "ConsoleDisplay": ...

    @property
    def context(self) -> "Context | None": ...

    @property
    def instruction_template(self) -> str: ...

    @property
    def instruction_context(self) -> dict[str, str]: ...

    @property
    def skill_manifests(self) -> Sequence["SkillManifest"]: ...

    @property
    def has_filesystem_runtime(self) -> bool: ...

    def set_skill_manifests(self, manifests: Sequence["SkillManifest"]) -> None: ...

    def set_instruction_context(self, context: dict[str, str]) -> None: ...

    @property
    def skill_registry(self) -> "SkillRegistry | None": ...

    @skill_registry.setter
    def skill_registry(self, value: "SkillRegistry | None") -> None: ...

    @property
    def shell_runtime_enabled(self) -> bool: ...

    @property
    def shell_access_modes(self) -> tuple[str, ...]: ...

    @property
    def shell_runtime(self) -> "ShellRuntime | None": ...

    def shell_notice_line(self) -> "Text | None": ...
