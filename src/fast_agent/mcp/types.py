from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from fast_agent.interfaces import AgentProtocol

if TYPE_CHECKING:
    from fast_agent.context import Context
    from fast_agent.mcp.mcp_aggregator import MCPAggregator
    from fast_agent.skills import SkillManifest
    from fast_agent.skills.registry import SkillRegistry
    from fast_agent.ui.console_display import ConsoleDisplay


@runtime_checkable
class McpAgentProtocol(AgentProtocol, Protocol):
    """Agent protocol with MCP-specific surface area."""

    @property
    def aggregator(self) -> MCPAggregator: ...

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
