"""Canonical source metadata for fast-agent runtime tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from mcp.types import Tool

FAST_AGENT_TOOL_SOURCE_META: Final = "fast-agent/toolSource"

ToolSource = Literal[
    "shell",
    "acp_terminal",
    "acp_filesystem",
    "skill",
    "mcp",
    "provider_managed",
]

SHELL_TOOL_SOURCE: Final[ToolSource] = "shell"
ACP_TERMINAL_TOOL_SOURCE: Final[ToolSource] = "acp_terminal"
ACP_FILESYSTEM_TOOL_SOURCE: Final[ToolSource] = "acp_filesystem"
SKILL_TOOL_SOURCE: Final[ToolSource] = "skill"
MCP_TOOL_SOURCE: Final[ToolSource] = "mcp"
PROVIDER_MANAGED_TOOL_SOURCE: Final[ToolSource] = "provider_managed"

TOOL_SOURCE_LABELS: Final[dict[ToolSource, str]] = {
    SHELL_TOOL_SOURCE: "Shell",
    ACP_TERMINAL_TOOL_SOURCE: "Shell",
    ACP_FILESYSTEM_TOOL_SOURCE: "ACP Filesystem",
    SKILL_TOOL_SOURCE: "Skill",
    MCP_TOOL_SOURCE: "MCP",
    PROVIDER_MANAGED_TOOL_SOURCE: "Provider Managed",
}


def set_tool_source(tool: "Tool", source: ToolSource) -> "Tool":
    """Return a copy of ``tool`` stamped with fast-agent source metadata."""
    meta = dict(tool.meta or {})
    meta[FAST_AGENT_TOOL_SOURCE_META] = source
    return tool.model_copy(update={"meta": meta})


def tool_source(tool: "Tool") -> ToolSource | None:
    """Return the fast-agent source metadata for ``tool``, when present and valid."""
    meta = tool.meta or tool.model_dump().get("meta")
    if not isinstance(meta, dict):
        return None

    value = meta.get(FAST_AGENT_TOOL_SOURCE_META)
    return value if value in TOOL_SOURCE_LABELS else None
