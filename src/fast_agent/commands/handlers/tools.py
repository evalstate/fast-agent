"""Shared tools command handlers."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

from rich.text import Text

from fast_agent.commands.results import CommandOutcome
from fast_agent.mcp.common import is_namespaced_name

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


def _truncate_tool_description(description: str, char_limit: int = 240) -> str:
    description = description.strip()
    if len(description) <= char_limit:
        return description

    truncate_pos = char_limit
    sentence_break = description.rfind(". ", 0, char_limit + 20)
    if sentence_break > char_limit - 50:
        truncate_pos = sentence_break + 1
    else:
        word_break = description.rfind(" ", 0, char_limit + 10)
        if word_break > char_limit - 30:
            truncate_pos = word_break

    return description[:truncate_pos].rstrip() + "..."


def _format_tool_line(tool_name: str, title: str | None, suffix: str | None) -> Text:
    line = Text()
    line.append(tool_name, style="bright_blue bold")
    if suffix:
        line.append(f" {suffix}", style="dim cyan")
    if title and title.strip():
        line.append(f" {title}", style="default")
    return line


def _format_tool_description(description: str) -> list[Text]:
    truncated = _truncate_tool_description(description)
    wrapped_lines = textwrap.wrap(truncated, width=72)
    return [Text(line, style="white") for line in wrapped_lines]


def _format_tool_args(schema: dict[str, Any]) -> str | None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None

    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []

    arg_list: list[str] = []
    for prop_name in properties:
        arg_list.append(f"{prop_name}*" if prop_name in required else prop_name)

    if not arg_list:
        return None

    args_text = ", ".join(arg_list)
    if len(args_text) > 80:
        args_text = args_text[:77] + "..."
    return args_text


async def handle_list_tools(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()

    agent = ctx.agent_provider._agent(agent_name)
    tools_result = await agent.list_tools()

    if not tools_result or not hasattr(tools_result, "tools") or not tools_result.tools:
        outcome.add_message(
            "No tools available for this agent.",
            channel="warning",
            right_info="tools",
            agent_name=agent_name,
        )
        return outcome

    card_tool_names = set(getattr(agent, "_card_tool_names", []) or [])
    agent_tool_names = set(getattr(agent, "_agent_tools", {}).keys())
    child_agent_tool_names = set(getattr(agent, "_child_agents", {}).keys())
    agent_tool_names |= child_agent_tool_names
    internal_tool_names = {"execute", "read_skill"}

    content = Text()
    header = Text(f"Tools for agent {agent_name}:", style="bold")
    content.append_text(header)
    content.append("\n\n")

    for index, tool in enumerate(tools_result.tools, 1):
        meta = getattr(tool, "meta", {}) or {}
        suffix = None
        if tool.name in internal_tool_names:
            suffix = "(Internal)"
        elif tool.name in card_tool_names:
            suffix = "(Card Function)"
        elif tool.name in child_agent_tool_names:
            suffix = "(Subagent)"
        elif tool.name not in agent_tool_names and is_namespaced_name(tool.name):
            suffix = "(MCP)"

        if meta.get("openai/skybridgeEnabled"):
            suffix = f"{suffix} (skybridge)" if suffix else "(skybridge)"

        line = Text()
        line.append(f"[{index:2}] ", style="dim cyan")
        line.append_text(_format_tool_line(tool.name, tool.title, suffix))
        content.append_text(line)
        content.append("\n")

        if tool.description and tool.description.strip():
            for wrapped_line in _format_tool_description(tool.description.strip()):
                content.append("     ", style="dim")
                content.append_text(wrapped_line)
                content.append("\n")

        schema = getattr(tool, "inputSchema", None)
        if isinstance(schema, dict):
            args_text = _format_tool_args(schema)
            if args_text:
                content.append("     ", style="dim")
                content.append(f"args: {args_text}", style="dim magenta")
                content.append("\n")

        if meta.get("openai/skybridgeEnabled"):
            template = meta.get("openai/skybridgeTemplate")
            if template:
                content.append("     ", style="dim")
                content.append("template: ", style="dim magenta")
                content.append(str(template))
                content.append("\n")

        content.append("\n")

    outcome.add_message(
        content,
        right_info="tools",
        agent_name=agent_name,
    )
    return outcome
