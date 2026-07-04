"""Shared tools command handlers."""

from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.commands.handlers._text_formatting import indexed_row
from fast_agent.commands.handlers._text_utils import truncate_description
from fast_agent.commands.results import CommandOutcome
from fast_agent.commands.tool_summaries import (
    ProviderToolSummary,
    build_provider_tool_summaries,
    build_tool_summaries,
    provider_tool_status_label,
)
from fast_agent.interfaces import AgentProtocol
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


APP_SUFFIX_BADGES = ("(Apps SDK)", "(MCP App)")
_APP_SUFFIX_BADGE_PATTERN = re.compile(
    f"({'|'.join(re.escape(badge) for badge in APP_SUFFIX_BADGES)})"
)


def _append_suffix(line: Text, suffix: str) -> None:
    line.append(" ", style="dim cyan")
    for segment in _APP_SUFFIX_BADGE_PATTERN.split(suffix):
        if not segment:
            continue
        style = "bright_yellow" if segment in APP_SUFFIX_BADGES else "dim cyan"
        line.append(segment, style=style)


def _format_tool_line(tool_name: str, title: str | None, suffix: str | None) -> Text:
    line = Text()
    line.append(tool_name, style="bright_blue bold")
    if suffix:
        _append_suffix(line, suffix)
    title_label = strip_to_none(title)
    if title_label is not None:
        line.append(f" {title_label}", style="default")
    return line


def _format_tool_description(description: str) -> list[Text]:
    truncated = truncate_description(description)
    wrapped_lines = textwrap.wrap(truncated, width=72)
    return [Text(line, style="white") for line in wrapped_lines]


def _append_provider_tool_section(
    content: Text,
    provider_summaries: list[ProviderToolSummary],
) -> None:
    if not provider_summaries:
        return

    content.append("Provider-managed / hosted tools", style="bold")
    content.append("\n\n")

    for summary in provider_summaries:
        line = Text()
        line.append("  • ", style="dim cyan")
        line.append(summary.name, style="bright_blue bold")
        line.append(f" ({provider_tool_status_label(summary)})", style="dim cyan")
        line.append(f" {summary.description}", style="white")
        content.append_text(line)
        content.append("\n")


def _format_args_text(args: list[str]) -> str:
    args_text = ", ".join(args)
    if len(args_text) > 80:
        return args_text[:77] + "..."
    return args_text


def _append_tool_detail(
    content: Text,
    label: str,
    value: str,
    *,
    label_style: str = "dim magenta",
    value_style: str | None = None,
) -> None:
    content.append("     ", style="dim")
    content.append(f"{label}: ", style=label_style)
    content.append(value, style=value_style)
    content.append("\n")


async def handle_list_tools(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()

    agent = ctx.agent_provider._agent(agent_name)
    if not isinstance(agent, AgentProtocol):
        outcome.add_message(
            "This agent does not support tool listing.",
            channel="warning",
            right_info="tools",
            agent_name=agent_name,
        )
        return outcome

    tools_result = await agent.list_tools()
    provider_summaries = build_provider_tool_summaries(agent)

    if not tools_result.tools and not provider_summaries:
        outcome.add_message(
            "No tools available for this agent.",
            channel="warning",
            right_info="tools",
            agent_name=agent_name,
        )
        return outcome

    summaries = build_tool_summaries(agent, list(tools_result.tools))

    content = Text()
    header = Text(f"Tools for agent {agent_name}:", style="bold")
    content.append_text(header)
    content.append("\n\n")

    if tools_result.tools:
        content.append("fast-agent managed tools", style="bold")
        content.append("\n\n")

        for index, summary in enumerate(summaries, 1):
            line = indexed_row(index)
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
                    _append_tool_detail(
                        content,
                        "args",
                        args_text,
                        value_style="dim magenta",
                    )

            if summary.template:
                _append_tool_detail(content, "template", str(summary.template))

            content.append("\n")

    _append_provider_tool_section(content, provider_summaries)

    outcome.add_message(
        content,
        right_info="tools",
        agent_name=agent_name,
    )
    return outcome
