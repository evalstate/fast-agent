"""Shared helpers to summarize tool metadata for rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fast_agent.interfaces import (
    AgentBackedToolProvider,
    CardToolProvider,
    SmartToolingCapable,
)
from fast_agent.mcp.common import is_namespaced_name

if TYPE_CHECKING:
    from mcp.types import Tool


@dataclass(slots=True)
class ToolSummary:
    name: str
    title: str | None
    description: str | None
    args: list[str] | None
    suffix: str | None
    template: str | None


def _format_tool_args(schema: dict[str, Any] | None) -> list[str] | None:
    if not schema:
        return None

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None

    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []

    arg_list: list[str] = []
    for prop_name in properties:
        arg_list.append(f"{prop_name}*" if prop_name in required else prop_name)

    return arg_list or None


def _tool_meta(tool: "Tool") -> dict[str, Any]:
    """Return MCP tool metadata, working around upstream model access quirks."""
    if tool.meta:
        return tool.meta
    dumped = tool.model_dump().get("meta")
    return dumped if isinstance(dumped, dict) else {}


def _collect_tool_name_sets(agent: object) -> tuple[set[str], set[str], set[str]]:
    card_tool_names = set(agent.card_tool_names) if isinstance(agent, CardToolProvider) else set()
    smart_tool_names = set(agent.smart_tool_names) if isinstance(agent, SmartToolingCapable) else set()
    agent_tool_names = (
        set(agent.agent_backed_tools.keys()) if isinstance(agent, AgentBackedToolProvider) else set()
    )
    return card_tool_names, smart_tool_names, agent_tool_names


def build_tool_summaries(agent: object, tools: list[Tool]) -> list[ToolSummary]:
    card_tool_names, smart_tool_names, agent_tool_names = _collect_tool_name_sets(agent)
    child_agent_tool_names = agent_tool_names
    internal_tool_names = {"execute", "read_skill"}

    summaries: list[ToolSummary] = []

    for tool in tools:
        name = tool.name
        title = tool.title
        description = (tool.description or "").strip() or None
        meta = _tool_meta(tool)

        suffix = None
        if name in internal_tool_names:
            suffix = "(Internal)"
        elif name in smart_tool_names:
            suffix = "(Smart)"
        elif name in card_tool_names:
            suffix = "(Card Function)"
        elif name in child_agent_tool_names:
            suffix = "(Subagent)"
        elif name not in agent_tool_names and is_namespaced_name(name):
            suffix = "(MCP)"

        if meta.get("openai/skybridgeEnabled"):
            suffix = f"{suffix} (skybridge)" if suffix else "(skybridge)"

        args = _format_tool_args(tool.inputSchema)
        template = meta.get("openai/skybridgeTemplate")

        summaries.append(
            ToolSummary(
                name=name,
                title=title,
                description=description,
                args=args,
                suffix=suffix,
                template=template,
            )
        )

    return summaries
