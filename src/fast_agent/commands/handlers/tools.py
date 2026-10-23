"""Shared tools command handlers."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.commands.handlers._text_formatting import indexed_row
from fast_agent.commands.handlers._text_utils import truncate_description
from fast_agent.commands.renderers.tools_markdown import render_tool_query_markdown
from fast_agent.commands.results import CommandOutcome
from fast_agent.commands.tool_summaries import (
    ProviderToolSummary,
    ToolsListing,
    ToolSummary,
    build_tools_listing,
    code_tool_description,
    format_token_estimate,
    provider_tool_status_label,
    ptc_settings_label,
)
from fast_agent.interfaces import AgentProtocol
from fast_agent.tools.tool_surface import LARGE_STUB_TOKENS, resolve_tool_surface
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


APP_SUFFIX_BADGES = ("(Apps SDK)", "(MCP App)")
_APP_SUFFIX_BADGE_PATTERN = re.compile(
    f"({'|'.join(re.escape(badge) for badge in APP_SUFFIX_BADGES)})"
)
SHOW_ALL_FLAG = "--all"
TOOLS_USAGE = f"/tools [{SHOW_ALL_FLAG}|ptc|<tool-name>]"
_GROUP_NAMES_WIDTH = 60
_SIGNATURE_WIDTH = 96


@dataclass(frozen=True, slots=True)
class ToolsQuery:
    tool_name: str | None = None
    show_all: bool = False


def parse_tools_query(argument: str | None) -> ToolsQuery:
    """Parse ``/tools`` arguments; raises ValueError with usage text when invalid."""
    tokens = (argument or "").split()
    names = [token for token in tokens if token != SHOW_ALL_FLAG and token.casefold() != "summary"]
    if len(names) > 1 or any(name.startswith("-") for name in names):
        raise ValueError(f"Usage: {TOOLS_USAGE}")
    return ToolsQuery(tool_name=names[0] if names else None, show_all=SHOW_ALL_FLAG in tokens)


def _append_suffix(line: Text, suffix: str) -> None:
    line.append(" ", style="dim cyan")
    for segment in _APP_SUFFIX_BADGE_PATTERN.split(suffix):
        if not segment:
            continue
        style = "bright_yellow" if segment in APP_SUFFIX_BADGES else "dim cyan"
        line.append(segment, style=style)


def _append_context(line: Text, summary: ToolSummary) -> None:
    if summary.context_tokens is not None:
        line.append(f" {format_token_estimate(summary.context_tokens)}", style="dim")
    if summary.large:
        line.append(" large", style="bright_yellow")


def _format_tool_line(summary: ToolSummary) -> Text:
    line = Text()
    line.append(summary.name, style="bright_blue bold")
    if summary.suffix:
        _append_suffix(line, summary.suffix)
    title_label = strip_to_none(summary.title)
    if title_label is not None:
        line.append(f" {title_label}", style="default")
    _append_context(line, summary)
    return line


def _format_function_line(summary: ToolSummary) -> Text:
    line = Text()
    line.append(summary.python_name or summary.name, style="bright_blue bold")
    if summary.suffix:
        _append_suffix(line, summary.suffix)
    _append_context(line, summary)
    return line


def _append_signature(content: Text, signature: str) -> None:
    for wrapped_line in textwrap.wrap(
        f"await {signature}",
        width=_SIGNATURE_WIDTH,
        initial_indent="     ",
        subsequent_indent="         ",
        break_on_hyphens=False,
    ):
        content.append(wrapped_line, style="magenta")
        content.append("\n")


def _format_tool_description(description: str) -> list[Text]:
    truncated = truncate_description(description)
    wrapped_lines = textwrap.wrap(truncated, width=72)
    return [Text(line, style="white") for line in wrapped_lines]


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


def _append_description(content: Text, description: str | None) -> None:
    if not description:
        return
    for wrapped_line in _format_tool_description(description):
        content.append("     ", style="dim")
        content.append_text(wrapped_line)
        content.append("\n")


def _append_section_heading(content: Text, title: str, detail: str) -> None:
    content.append(title, style="bold")
    content.append(f" ({detail})", style="dim")
    content.append("\n\n")


def _append_model_tools(content: Text, listing: ToolsListing) -> int:
    count = len(listing.model)
    _append_section_heading(
        content,
        "Sent to the model",
        f"{count} · {format_token_estimate(listing.model_tokens)}",
    )
    code_tool_name = listing.ptc.tool_name if listing.ptc else None
    for index, summary in enumerate(listing.model, 1):
        line = indexed_row(index)
        line.append_text(_format_tool_line(summary))
        content.append_text(line)
        content.append("\n")
        if summary.name == code_tool_name:
            _append_description(content, code_tool_description(listing))
        else:
            _append_description(content, summary.description)
        if summary.args:
            _append_tool_detail(
                content, "args", _format_args_text(summary.args), value_style="dim magenta"
            )
        if summary.template:
            _append_tool_detail(content, "template", str(summary.template))
        if summary.direct_reason:
            _append_tool_detail(content, "direct", summary.direct_reason)
        content.append("\n")
    return count


def _group_names(summaries: list[ToolSummary]) -> str:
    names: list[str] = []
    used = 0
    for summary in summaries:
        name = summary.python_name or summary.name
        if names and used + len(name) > _GROUP_NAMES_WIDTH:
            return ", ".join(names) + f", +{len(summaries) - len(names)} more"
        names.append(name)
        used += len(name) + 2
    return ", ".join(names)


def _append_programmatic_groups(content: Text, listing: ToolsListing) -> None:
    groups = listing.programmatic_groups()
    width = max(len(group) for group, _ in groups)
    for group, summaries in groups:
        line = Text("     ")
        line.append(group.ljust(width), style="cyan")
        line.append(f"  {len(summaries):>3}  ", style="bright_blue bold")
        line.append(_group_names(summaries), style="dim")
        content.append_text(line)
        content.append("\n")


def _append_programmatic_functions(content: Text, listing: ToolsListing, start: int) -> None:
    for index, summary in enumerate(listing.programmatic, start):
        line = indexed_row(index)
        line.append_text(_format_function_line(summary))
        content.append_text(line)
        content.append("\n")
        if summary.signature:
            _append_signature(content, summary.signature)
        if summary.python_name and summary.python_name != summary.name:
            _append_tool_detail(content, "tool", summary.name)
        _append_description(content, summary.description)
        content.append("\n")


def _append_large_stub_warning(content: Text, listing: ToolsListing) -> None:
    large = listing.large_programmatic
    if not large:
        return
    content.append("     ⚠ ", style="bright_yellow")
    content.append(
        f"Large stubs (≥ ~{LARGE_STUB_TOKENS} tokens each) inflate the code tool: ",
        style="yellow",
    )
    content.append(
        ", ".join(
            f"{summary.python_name or summary.name} "
            f"{format_token_estimate(summary.context_tokens or 0)}"
            for summary in large
        ),
        style="default",
    )
    content.append("\n")


def _append_programmatic_tools(
    content: Text, listing: ToolsListing, *, show_all: bool, start: int
) -> None:
    if listing.ptc is None or not listing.programmatic:
        return
    _append_section_heading(
        content,
        f"Callable from {listing.ptc.tool_name}",
        f"{len(listing.programmatic)} · "
        f"{format_token_estimate(listing.programmatic_tokens)} "
        f"inside {listing.ptc.tool_name}",
    )
    if show_all:
        _append_programmatic_functions(content, listing, start)
    else:
        _append_programmatic_groups(content, listing)
    _append_large_stub_warning(content, listing)
    if not show_all:
        content.append("\n")
        content.append(
            f"     /tools {SHOW_ALL_FLAG} lists signatures · /tools <name> shows one function · "
            "/tools ptc shows the full code tool description",
            style="dim",
        )
        content.append("\n\n")


def _append_provider_tool_section(
    content: Text,
    provider_summaries: list[ProviderToolSummary],
    *,
    programmatic: bool,
) -> None:
    if not provider_summaries:
        return

    content.append("Provider-managed / hosted tools", style="bold")
    if programmatic:
        content.append(" (not callable from code)", style="dim")
    content.append("\n\n")

    for summary in provider_summaries:
        line = Text()
        line.append("  • ", style="dim cyan")
        line.append(summary.name, style="bright_blue bold")
        line.append(f" ({provider_tool_status_label(summary)})", style="dim cyan")
        line.append(f" {summary.description}", style="white")
        content.append_text(line)
        content.append("\n")


def render_tools_listing(agent_name: str, listing: ToolsListing, *, show_all: bool) -> Text:
    content = Text()
    content.append(f"Tools for agent {agent_name}:", style="bold")
    content.append("\n")
    if listing.ptc is not None:
        content.append(ptc_settings_label(listing.ptc), style="dim")
        content.append("\n")
    content.append("\n")

    shown = _append_model_tools(content, listing) if listing.model else 0
    _append_programmatic_tools(content, listing, show_all=show_all, start=shown + 1)
    _append_provider_tool_section(content, listing.provider, programmatic=listing.ptc is not None)
    return content


async def handle_list_tools(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None = None,
) -> CommandOutcome:
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

    try:
        query = parse_tools_query(argument)
    except ValueError as exc:
        outcome.add_message(str(exc), channel="warning", right_info="tools", agent_name=agent_name)
        return outcome

    surface = await resolve_tool_surface(agent)

    if query.tool_name is not None:
        detail = render_tool_query_markdown(query.tool_name, surface)
        if detail is None:
            outcome.add_message(
                f"Tool not found: {query.tool_name}",
                channel="warning",
                right_info="tools",
                agent_name=agent_name,
            )
            return outcome
        outcome.add_message(
            detail,
            right_info="tools",
            agent_name=agent_name,
            render_markdown=True,
        )
        return outcome

    listing = build_tools_listing(agent, surface)
    if listing.is_empty:
        outcome.add_message(
            "No tools available for this agent.",
            channel="warning",
            right_info="tools",
            agent_name=agent_name,
        )
        return outcome

    outcome.add_message(
        render_tools_listing(agent_name, listing, show_all=query.show_all),
        right_info="tools",
        agent_name=agent_name,
    )
    return outcome
