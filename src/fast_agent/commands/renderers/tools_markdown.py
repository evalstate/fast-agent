"""Markdown renderers for tool summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.renderers.markdown_blocks import (
    markdown_heading,
    wrapped_quote_lines,
)
from fast_agent.commands.summary_utils import optional_string
from fast_agent.utils.markdown import escape_markdown_text, markdown_code_span

if TYPE_CHECKING:
    from fast_agent.commands.tool_summaries import ProviderToolSummary, ToolSummary

from fast_agent.commands.tool_summaries import provider_tool_status_label


def _format_args(args: list[str] | None) -> str | None:
    if not args:
        return None
    normalized_args = [arg for value in args if (arg := optional_string(value)) is not None]
    if not normalized_args:
        return None
    return ", ".join(markdown_code_span(arg) for arg in normalized_args)


def _format_header(*, index: int, summary: "ToolSummary") -> str:
    header = f"{index}. **{escape_markdown_text(summary.name)}**"
    suffix = optional_string(summary.suffix)
    title = optional_string(summary.title)

    if suffix is not None:
        header = f"{header} _{escape_markdown_text(suffix)}_"
    if title is not None:
        header = f"{header} — {escape_markdown_text(title)}"

    return header


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


def render_tools_markdown(
    summaries: list["ToolSummary"],
    *,
    heading: str,
    provider_summaries: list["ProviderToolSummary"] | None = None,
) -> str:
    lines = [markdown_heading(heading), ""]

    if summaries:
        lines.extend(["## MCP / local tools", ""])

    for index, summary in enumerate(summaries, start=1):
        lines.append(_format_header(index=index, summary=summary))

        description = summary.description or ""
        lines.extend(_format_wrapped_description(description))

        args_line = _format_args(summary.args)
        if args_line:
            lines.append(_format_tool_detail("Args", args_line))

        template = optional_string(summary.template)
        if template is not None:
            lines.append(_format_tool_detail("Template", markdown_code_span(template)))

        lines.append("")

    if provider_summaries:
        if summaries:
            lines.append("")
        lines.extend(["## Provider-managed / hosted tools", ""])
        lines.extend(_format_provider_tool(summary) for summary in provider_summaries)

    return "\n".join(lines).rstrip()
