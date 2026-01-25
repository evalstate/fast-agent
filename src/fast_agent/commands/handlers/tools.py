"""Shared tools command handlers."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.commands.handlers._text_utils import truncate_description
from fast_agent.commands.results import CommandOutcome
from fast_agent.commands.tool_summaries import ToolSummary, build_tool_summaries

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


def _format_tool_line(tool_name: str, title: str | None, suffix: str | None) -> Text:
    line = Text()
    line.append(tool_name, style="bright_blue bold")
    if suffix:
        line.append(f" {suffix}", style="dim cyan")
    if title and title.strip():
        line.append(f" {title}", style="default")
    return line


def _format_tool_description(description: str) -> list[Text]:
    truncated = truncate_description(description)
    wrapped_lines = textwrap.wrap(truncated, width=72)
    return [Text(line, style="white") for line in wrapped_lines]


def _summaries_from_tools(agent: object, tools: list[object]) -> list[ToolSummary]:
    return build_tool_summaries(agent, tools)


def _format_args_text(args: list[str]) -> str:
    args_text = ", ".join(args)
    if len(args_text) > 80:
        return args_text[:77] + "..."
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

    summaries = _summaries_from_tools(agent, list(tools_result.tools))

    content = Text()
    header = Text(f"Tools for agent {agent_name}:", style="bold")
    content.append_text(header)
    content.append("\n\n")

    for index, summary in enumerate(summaries, 1):
        line = Text()
        line.append(f"[{index:2}] ", style="dim cyan")
        line.append_text(_format_tool_line(summary.name, summary.title, summary.suffix))
        content.append_text(line)
        content.append("\n")

        description = summary.description
        if description:
            for wrapped_line in _format_tool_description(description):
                content.append("     ", style="dim")
                content.append_text(wrapped_line)
                content.append("\n")

        if summary.args:
            args_text = _format_args_text(summary.args)
            if args_text:
                content.append("     ", style="dim")
                content.append(f"args: {args_text}", style="dim magenta")
                content.append("\n")

        if summary.template:
            content.append("     ", style="dim")
            content.append("template: ", style="dim magenta")
            content.append(str(summary.template))
            content.append("\n")

        content.append("\n")

    outcome.add_message(
        content,
        right_info="tools",
        agent_name=agent_name,
    )
    return outcome
