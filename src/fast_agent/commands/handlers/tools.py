"""Shared tools command handlers."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
from fast_agent.ptc.mentions import (
    ToolMentionError,
    find_programmatic_function,
    resolve_tool_definitions,
    tool_mention_token,
)
from fast_agent.tools.tool_surface import LARGE_STUB_TOKENS, resolve_tool_surface
from fast_agent.utils.count_display import format_count
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import CommandContext
    from fast_agent.tools.tool_surface import SurfaceTool, ToolSurface


APP_SUFFIX_BADGES = ("(Apps SDK)", "(MCP App)")
_APP_SUFFIX_BADGE_PATTERN = re.compile(
    f"({'|'.join(re.escape(badge) for badge in APP_SUFFIX_BADGES)})"
)
SHOW_ALL_FLAG = "--all"
ADD_ACTION = "add"
SELECT_ACTIONS = ("select", "browse", "explore")
TOOLS_ADD_USAGE = f"/tools {ADD_ACTION} <function> [<function>...]"
TOOLS_USAGE = f"/tools [{SHOW_ALL_FLAG}|ptc|<tool-name>|{ADD_ACTION} <function>...|select]"
TOOL_USAGE = "/tool <tool-name>"
END_OF_ACTIONS = "--"
_GROUP_NAMES_WIDTH = 60
_SIGNATURE_WIDTH = 96


@dataclass(frozen=True, slots=True)
class ToolsQuery:
    """List tools, or show one (``tool_name``)."""

    tool_name: str | None = None
    show_all: bool = False


@dataclass(frozen=True, slots=True)
class ToolsAddQuery:
    """Add ``^tool:`` mentions for PTC functions to the input draft."""

    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolsSelectQuery:
    """Pick PTC functions interactively, then add their ``^tool:`` mentions to the draft."""


type ToolsCommand = ToolsQuery | ToolsAddQuery | ToolsSelectQuery


@runtime_checkable
class ToolPickerIO(Protocol):
    """Command IO that can run the interactive PTC function picker."""

    async def pick_tools(self, surface: ToolSurface) -> Sequence[str] | None:
        """Return the chosen functions' Python names, or None when cancelled."""
        ...


def tool_detail_argument(tool_name: str) -> str:
    """``/tools`` argument that looks up ``tool_name`` even when it is named like an action."""
    return f"{END_OF_ACTIONS} {tool_name}"


def parse_tools_query(argument: str | None) -> ToolsCommand:
    """Parse ``/tools`` arguments; raises ValueError with usage text when invalid.

    A leading ``--`` ends action parsing: ``/tools -- add`` (and singular ``/tool add``)
    inspects a tool named ``add``.
    """
    tokens = (argument or "").split()
    action = tokens[0].casefold() if tokens else ""
    if action == END_OF_ACTIONS:
        if len(tokens) != 2:
            raise ValueError(f"Usage: {TOOL_USAGE}")
        return ToolsQuery(tool_name=tokens[1])
    if action == ADD_ACTION:
        mention_prefix = tool_mention_token("")
        names = tuple(token.removeprefix(mention_prefix) for token in tokens[1:])
        if not names or any(not name or name.startswith("-") for name in names):
            raise ValueError(f"Usage: {TOOLS_ADD_USAGE}")
        return ToolsAddQuery(names=names)
    if action in SELECT_ACTIONS:
        if len(tokens) > 1:
            raise ValueError(f"Usage: /tools {tokens[0]}")
        return ToolsSelectQuery()

    names = [token for token in tokens if token != SHOW_ALL_FLAG and token.casefold() != "summary"]
    if len(names) > 1 or any(name.startswith("-") for name in names):
        raise ValueError(f"Usage: {TOOLS_USAGE}")
    return ToolsQuery(tool_name=names[0] if names else None, show_all=SHOW_ALL_FLAG in tokens)


def resolve_tool_mentions(surface: ToolSurface, names: Sequence[str]) -> list[SurfaceTool]:
    """Resolve names for ``/tools add`` to PTC functions, deduplicated in order.

    Raises ``ToolMentionError`` when PTC is off or a name is not a PTC function.
    """
    ptc = surface.programmatic
    if ptc is None:
        raise ToolMentionError(
            "Programmatic tool calling is not enabled for this agent, so there are no "
            "function definitions to add."
        )
    for name in names:
        if find_programmatic_function(surface, name) is None and surface.find(name) is not None:
            raise ToolMentionError(
                f"{name} is sent to the model as a tool, not called from {ptc.tool_name}; "
                f"only {ptc.tool_name} functions can be added."
            )
    return resolve_tool_definitions(surface, names)


def tool_mentions_text(functions: Sequence[SurfaceTool]) -> str:
    """Space-joined ``^tool:<python_name>`` mentions for ``functions``."""
    return " ".join(
        tool_mention_token(function.python_name or function.tool.name) for function in functions
    )


def _append_suffix(line: Text, suffix: str) -> None:
    line.append(" ", style="dim cyan")
    for segment in _APP_SUFFIX_BADGE_PATTERN.split(suffix):
        if not segment:
            continue
        style = "bright_yellow" if segment in APP_SUFFIX_BADGES else "dim cyan"
        line.append(segment, style=style)


def _append_context(line: Text, summary: ToolSummary) -> None:
    if summary.deferred:
        line.append(f" {format_token_estimate(summary.stub_tokens)} on demand", style="dim")
        return
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


def _append_function_groups(content: Text, groups: list[tuple[str, list[ToolSummary]]]) -> None:
    if not groups:
        return
    width = max(len(group) for group, _ in groups)
    for group, summaries in groups:
        line = Text("     ")
        line.append(group.ljust(width), style="cyan")
        line.append(f"  {len(summaries):>3}  ", style="bright_blue bold")
        line.append(_group_names(summaries), style="dim")
        content.append_text(line)
        content.append("\n")


def _append_programmatic_functions(content: Text, summaries: list[ToolSummary], start: int) -> None:
    for index, summary in enumerate(summaries, start):
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


def _append_note(content: Text, note: str) -> None:
    for wrapped_line in textwrap.wrap(
        note,
        width=_SIGNATURE_WIDTH,
        initial_indent="     ",
        subsequent_indent="     ",
        break_on_hyphens=False,
    ):
        content.append(wrapped_line, style="dim")
        content.append("\n")


def _append_deferred_tools(
    content: Text, listing: ToolsListing, *, show_all: bool, start: int
) -> None:
    deferred = listing.deferred
    if listing.ptc is None or not deferred:
        return
    tool_name = listing.ptc.tool_name
    if not show_all:
        content.append("\n")
    _append_section_heading(
        content,
        f"Searchable from {tool_name}",
        f"{len(deferred)} · {format_token_estimate(listing.deferred_tokens)} on demand",
    )
    if show_all:
        _append_programmatic_functions(content, deferred, start)
    else:
        _append_function_groups(content, listing.deferred_groups())
        content.append("\n")
    _append_note(
        content,
        f"Not in the {tool_name} description; still callable. "
        "search_tools(query=..., server=...) finds them; "
        "^tool:<name> or /tools select adds definitions to your message.",
    )
    if show_all:
        content.append("\n")


def _append_programmatic_tools(
    content: Text, listing: ToolsListing, *, show_all: bool, start: int
) -> None:
    if listing.ptc is None or not listing.programmatic:
        return
    upfront = listing.upfront
    count = f"{len(upfront)} upfront" if listing.deferred else str(len(upfront))
    _append_section_heading(
        content,
        f"Callable from {listing.ptc.tool_name}",
        f"{count} · {format_token_estimate(listing.programmatic_tokens)} "
        f"inside {listing.ptc.tool_name}",
    )
    if show_all:
        _append_programmatic_functions(content, upfront, start)
    else:
        _append_function_groups(content, listing.programmatic_groups())
    _append_large_stub_warning(content, listing)
    _append_deferred_tools(content, listing, show_all=show_all, start=start + len(upfront))
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


def _add_tool_mentions(
    outcome: CommandOutcome,
    surface: ToolSurface,
    names: Sequence[str],
    *,
    agent_name: str,
) -> None:
    try:
        functions = resolve_tool_mentions(surface, names)
    except ToolMentionError as exc:
        outcome.add_message(str(exc), channel="error", right_info="tools", agent_name=agent_name)
        return
    mentions = tool_mentions_text(functions)
    tokens = sum(function.stub_tokens for function in functions)
    outcome.buffer_prefill = mentions
    outcome.add_message(
        f"{mentions} attaches {format_count(len(functions), 'function definition')} "
        f"({format_token_estimate(tokens)}) to your next message.",
        channel="info",
        right_info="tools",
        agent_name=agent_name,
    )


async def _select_tool_mentions(
    ctx: CommandContext,
    outcome: CommandOutcome,
    surface: ToolSurface,
    *,
    agent_name: str,
) -> None:
    def warn(message: str) -> None:
        outcome.add_message(message, channel="warning", right_info="tools", agent_name=agent_name)

    if surface.programmatic is None:
        warn("Programmatic tool calling is not enabled for this agent; there are no functions.")
    elif not surface.programmatic_functions():
        warn(f"No functions are callable from {surface.programmatic.tool_name}.")
    elif not isinstance(ctx.io, ToolPickerIO):
        warn(
            "Interactive tool selection needs the terminal UI. "
            f"Use {TOOLS_ADD_USAGE} to add function definitions to your message."
        )
    elif names := await ctx.io.pick_tools(surface):
        _add_tool_mentions(outcome, surface, names, agent_name=agent_name)


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

    try:
        surface = await resolve_tool_surface(agent)
    except Exception as exc:
        outcome.add_message(
            f"Failed to fetch tools from the agent.\nDetails: {exc}",
            channel="error",
            right_info="tools",
            agent_name=agent_name,
        )
        return outcome

    if isinstance(query, ToolsAddQuery):
        _add_tool_mentions(outcome, surface, query.names, agent_name=agent_name)
        return outcome
    if isinstance(query, ToolsSelectQuery):
        await _select_tool_mentions(ctx, outcome, surface, agent_name=agent_name)
        return outcome

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
