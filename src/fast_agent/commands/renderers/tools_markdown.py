"""Markdown renderers for tool summaries."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.commands.tool_summaries import ProviderToolSummary, ToolSummary


def _format_args(args: list[str] | None) -> str | None:
    if not args:
        return None
    return ", ".join(f"`{arg}`" for arg in args)


def _format_header(*, index: int, summary: "ToolSummary") -> str:
    header = f"{index}. **{summary.name}**"
    suffix = (summary.suffix or "").strip()
    title = (summary.title or "").strip()

    if suffix:
        header = f"{header} _{suffix}_"
    if title:
        header = f"{header} — {title}"

    return header


def _format_provider_tool(summary: "ProviderToolSummary") -> str:
    state = "enabled" if summary.enabled else "disabled"
    return f"- **{summary.name}** _(provider-hosted, {state})_ — {summary.description}"


def render_tools_markdown(
    summaries: list["ToolSummary"],
    *,
    heading: str,
    provider_summaries: list["ProviderToolSummary"] | None = None,
) -> str:
    lines = [f"# {heading}", ""]

    if summaries:
        lines.extend(["## MCP / local tools", ""])

    for index, summary in enumerate(summaries, start=1):
        lines.append(_format_header(index=index, summary=summary))

        description = summary.description or ""
        if description:
            wrapped = textwrap.wrap(description, width=88)
            lines.extend(f"    > {desc_line}" for desc_line in wrapped[:4])
            if len(wrapped) > 4:
                lines.append("    > …")

        args_line = _format_args(summary.args)
        if args_line:
            lines.append("    > ")
            lines.append(f"    > **Args:** {args_line}")

        if summary.template:
            lines.append(f"    > **Template:** `{summary.template}`")

        lines.append("")

    if provider_summaries:
        if summaries:
            lines.append("")
        lines.extend(["## Provider-hosted tools", ""])
        lines.extend(_format_provider_tool(summary) for summary in provider_summaries)

    return "\n".join(lines).rstrip()
