"""Shared helpers to summarize tool metadata for rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fast_agent.mcp.common import is_namespaced_name


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


def build_tool_summaries(agent: object, tools: list[object]) -> list[ToolSummary]:
    card_tool_names = set(getattr(agent, "_card_tool_names", []) or [])
    smart_tool_names = set(getattr(agent, "_smart_tool_names", []) or [])
    agent_tool_names = set(getattr(agent, "_agent_tools", {}).keys())
    child_agent_tool_names = set(getattr(agent, "_child_agents", {}).keys())
    agent_tool_names |= child_agent_tool_names
    internal_tool_names = {"execute", "read_skill"}

    summaries: list[ToolSummary] = []

    for tool in tools:
        name = getattr(tool, "name", None) or "unnamed"
        title = getattr(tool, "title", None)
        description = (getattr(tool, "description", None) or "").strip() or None
        meta = getattr(tool, "meta", {}) or {}

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

        schema = getattr(tool, "inputSchema", None)
        args = _format_tool_args(schema) if isinstance(schema, dict) else None
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
