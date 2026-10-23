"""Markdown renderers for tool summaries."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from fast_agent.commands.renderers.markdown_blocks import (
    markdown_heading,
    wrapped_quote_lines,
)
from fast_agent.commands.summary_utils import optional_string
from fast_agent.commands.tool_summaries import (
    code_tool_description,
    format_token_estimate,
    provider_tool_status_label,
    ptc_settings_label,
)
from fast_agent.tools.tool_surface import LARGE_STUB_TOKENS, ToolExposure
from fast_agent.utils.markdown import escape_markdown_text, markdown_code_span

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp_types import Tool

    from fast_agent.commands.tool_summaries import ProviderToolSummary, ToolsListing, ToolSummary
    from fast_agent.tools.tool_surface import ProgrammaticSurface, SurfaceTool, ToolSurface

TOOLS_LIST_HINT = (
    "`/tools --all` lists signatures · `/tools <name>` shows one function · "
    "`/tools ptc` shows the full code tool description"
)
PTC_QUERY = "ptc"
"""``/tools ptc`` shows the code tool description exactly as the model receives it."""


def render_tool_schema_markdown(tool: "Tool", *, preamble: "Sequence[str]" = ()) -> str:
    name = escape_markdown_text(tool.name)
    input_schema = json.dumps(tool.input_schema, ensure_ascii=False, indent=2)
    sections = [
        f"# Tool schema: {name}",
        *preamble,
        f"## Input schema\n\n```json\n{input_schema}\n```",
    ]
    if tool.output_schema is not None:
        output_schema = json.dumps(tool.output_schema, ensure_ascii=False, indent=2)
        sections.append(
            "## Structured output schema\n\n"
            "Supplied by the MCP tool declaration.\n\n"
            f"```json\n{output_schema}\n```"
        )
    return "\n\n".join(sections)


def _large_stub_note() -> str:
    return f"⚠ Large stub (≥ ~{LARGE_STUB_TOKENS} tokens)."


def _render_code_tool_markdown(
    tool: "Tool", surface: "ToolSurface", ptc: "ProgrammaticSurface"
) -> str:
    functions = surface.exposed(ToolExposure.PROGRAMMATIC)
    tokens = sum(function.context_tokens for function in functions)
    lines = [
        f"## Callable functions ({len(functions)} · {format_token_estimate(tokens)})",
        "",
    ]
    for function in functions:
        line = f"- {markdown_code_span(function.signature or function.tool.name)}"
        if function.large:
            line += f" ⚠ {format_token_estimate(function.context_tokens)}"
        lines.append(line)
    return render_tool_schema_markdown(
        tool,
        preamble=(
            f"_{escape_markdown_text(ptc_settings_label(ptc))}_",
            f"## Instructions\n\n{ptc.instructions.strip()}",
            "\n".join(lines),
        ),
    )


def render_ptc_description_markdown(surface: "ToolSurface") -> str | None:
    """Render the code tool description verbatim, with its stubs; None when PTC is off."""
    ptc = surface.programmatic
    code_tool = surface.find(ptc.tool_name) if ptc is not None else None
    if ptc is None or code_tool is None:
        return None
    description = code_tool.tool.description or ""
    longest_run = max((len(run) for run in re.findall(r"`+", description)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return "\n\n".join(
        [
            f"# PTC: {escape_markdown_text(ptc.tool_name)} description",
            f"_{escape_markdown_text(ptc_settings_label(ptc))}_",
            (
                f"Sent to the model verbatim as the {markdown_code_span(ptc.tool_name)} tool "
                f"description ({format_token_estimate(code_tool.context_tokens)} with its schema)."
            ),
            f"{fence}text\n{description.rstrip()}\n{fence}",
        ]
    )


def render_tool_query_markdown(name: str, surface: "ToolSurface") -> str | None:
    """Render ``/tools <name>``: one tool, or the full PTC description for ``ptc``."""
    surface_tool = surface.find(name)
    if surface_tool is not None:
        return render_tool_detail_markdown(surface_tool, surface)
    if name.casefold() == PTC_QUERY:
        return render_ptc_description_markdown(surface)
    return None


def render_tool_detail_markdown(surface_tool: "SurfaceTool", surface: "ToolSurface") -> str:
    """Render one tool, explaining how the model reaches it."""
    tool = surface_tool.tool
    ptc = surface.programmatic
    if ptc is not None and tool.name == ptc.tool_name:
        return _render_code_tool_markdown(tool, surface, ptc)

    preamble: list[str] = []
    if ptc is not None and surface_tool.exposure is ToolExposure.PROGRAMMATIC:
        python_name = surface_tool.python_name or tool.name
        note = (
            f"Callable from {markdown_code_span(ptc.tool_name)} as "
            f"{markdown_code_span(f'await {python_name}(...)')}; its definition is not sent "
            f"to the model as a tool. The stub adds "
            f"{format_token_estimate(surface_tool.context_tokens)} to the "
            f"{markdown_code_span(ptc.tool_name)} description."
        )
        if surface_tool.large:
            note += f" {_large_stub_note()}"
        preamble.append(note)
        preamble.append(f"## Python stub\n\n```python\n{surface_tool.stub or ''}\n```")
    elif ptc is not None and surface_tool.direct_reason is not None:
        preamble.append(
            f"Sent to the model directly alongside {markdown_code_span(ptc.tool_name)} "
            f"({escape_markdown_text(surface_tool.direct_reason)})."
        )
    return render_tool_schema_markdown(tool, preamble=preamble)


def _format_args(args: list[str] | None) -> str | None:
    if not args:
        return None
    normalized_args = [arg for value in args if (arg := optional_string(value)) is not None]
    if not normalized_args:
        return None
    return ", ".join(markdown_code_span(arg) for arg in normalized_args)


def _format_context(summary: "ToolSummary") -> str:
    if summary.context_tokens is None:
        return ""
    context = f" · {format_token_estimate(summary.context_tokens)}"
    return f"{context} ⚠ large" if summary.large else context


def _format_header(*, index: int, summary: "ToolSummary") -> str:
    header = f"{index}. **{escape_markdown_text(summary.name)}**"
    suffix = optional_string(summary.suffix)
    title = optional_string(summary.title)

    if suffix is not None:
        header = f"{header} _{escape_markdown_text(suffix)}_"
    if title is not None:
        header = f"{header} — {escape_markdown_text(title)}"

    return header + _format_context(summary)


def _format_function_header(*, index: int, summary: "ToolSummary") -> str:
    header = f"{index}. {markdown_code_span(summary.signature or summary.name)}"
    suffix = optional_string(summary.suffix)
    if suffix is not None:
        header = f"{header} _{escape_markdown_text(suffix)}_"
    return header + _format_context(summary)


def _format_provider_tool(summary: "ProviderToolSummary") -> str:
    status = provider_tool_status_label(summary)
    return (
        f"- **{escape_markdown_text(summary.name)}** "
        f"_({escape_markdown_text(status)})_ — "
        f"{escape_markdown_text(summary.description)}"
    )


def _format_tool_detail(label: str, value: str) -> str:
    return f"    > **{label}:** {value}"


def _format_wrapped_description(description: str) -> list[str]:
    return wrapped_quote_lines(escape_markdown_text(description), prefix="    > ")


def _model_tool_lines(listing: "ToolsListing") -> list[str]:
    lines = [
        (
            f"## Sent to the model ({len(listing.model)} · "
            f"{format_token_estimate(listing.model_tokens)})"
        ),
        "",
    ]
    code_tool_name = listing.ptc.tool_name if listing.ptc else None
    for index, summary in enumerate(listing.model, start=1):
        lines.append(_format_header(index=index, summary=summary))

        description = (
            code_tool_description(listing)
            if summary.name == code_tool_name
            else summary.description or ""
        )
        lines.extend(_format_wrapped_description(description))

        args_line = _format_args(summary.args)
        if args_line:
            lines.append(_format_tool_detail("Args", args_line))

        template = optional_string(summary.template)
        if template is not None:
            lines.append(_format_tool_detail("Template", markdown_code_span(template)))

        if summary.direct_reason is not None:
            lines.append(_format_tool_detail("Direct", escape_markdown_text(summary.direct_reason)))

        lines.append("")
    return lines


def _programmatic_tool_lines(listing: "ToolsListing", *, show_all: bool) -> list[str]:
    ptc = listing.ptc
    if ptc is None or not listing.programmatic:
        return []
    lines = [
        (
            f"## Callable from {escape_markdown_text(ptc.tool_name)} "
            f"({len(listing.programmatic)} · {format_token_estimate(listing.programmatic_tokens)} "
            f"inside {escape_markdown_text(ptc.tool_name)})"
        ),
        "",
    ]
    if show_all:
        for index, summary in enumerate(listing.programmatic, start=len(listing.model) + 1):
            lines.append(_format_function_header(index=index, summary=summary))
            if summary.python_name and summary.python_name != summary.name:
                lines.append(_format_tool_detail("Tool", markdown_code_span(summary.name)))
            lines.extend(_format_wrapped_description(summary.description or ""))
            lines.append("")
    else:
        for group, summaries in listing.programmatic_groups():
            names = ", ".join(
                markdown_code_span(summary.python_name or summary.name) for summary in summaries
            )
            lines.append(f"- **{escape_markdown_text(group)}** ({len(summaries)}): {names}")
        lines.append("")

    large = listing.large_programmatic
    if large:
        names = ", ".join(
            f"{markdown_code_span(summary.python_name or summary.name)} "
            f"{format_token_estimate(summary.context_tokens or 0)}"
            for summary in large
        )
        warning = f"> ⚠ Large stubs (≥ ~{LARGE_STUB_TOKENS} tokens each) inflate the code tool: "
        lines.extend([warning + names, ""])
    if not show_all:
        lines.extend([f"_{TOOLS_LIST_HINT}_", ""])
    return lines


def render_tools_markdown(
    listing: "ToolsListing",
    *,
    heading: str,
    show_all: bool = False,
) -> str:
    lines = [markdown_heading(heading), ""]

    if listing.ptc is not None:
        lines.extend([f"_{escape_markdown_text(ptc_settings_label(listing.ptc))}_", ""])

    if listing.model:
        lines.extend(_model_tool_lines(listing))

    lines.extend(_programmatic_tool_lines(listing, show_all=show_all))

    if listing.provider:
        if listing.model or listing.programmatic:
            lines.append("")
        provider_heading = "## Provider-managed / hosted tools"
        if listing.ptc is not None:
            provider_heading += " (not callable from code)"
        lines.extend([provider_heading, ""])
        lines.extend(_format_provider_tool(summary) for summary in listing.provider)

    return "\n".join(lines).rstrip()
