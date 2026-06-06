"""Markdown renderers for ACP status output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.renderers.markdown_blocks import markdown_heading
from fast_agent.utils.count_display import format_count_breakdown
from fast_agent.utils.markdown import escape_markdown_text, markdown_code_span
from fast_agent.utils.time import format_duration

if TYPE_CHECKING:
    from fast_agent.commands.status_summaries import (
        AgentModelSummary,
        ClientInfoSummary,
        ConversationStatsSummary,
        ErrorHandlingSummary,
        PermissionsSummary,
        StatusSummary,
        SystemPromptSummary,
    )
    from fast_agent.commands.summary_utils import JsonObject


def _format_provider_line(provider_display: str, provider: str, hf_provider: str | None) -> str:
    line = _markdown_value(provider)
    if provider_display != "unknown":
        line = f"{_markdown_value(provider_display)} ({_markdown_value(provider)})"
    if hf_provider:
        line = f"{line} / {_markdown_value(hf_provider)}"
    return line


def _markdown_value(value: object) -> str:
    return escape_markdown_text(str(value))


def _context_window_display(context_window: int | None) -> str:
    if context_window:
        return f"{context_window} tokens"
    return "unknown"


def _append_parallel_agent_model_lines(lines: list[str], agent: "AgentModelSummary") -> None:
    lines.append(f"  - Provider: {_markdown_value(agent.provider_display or agent.provider)}")
    lines.append(f"  - Model: {_markdown_value(agent.model_name)}")
    if agent.wire_model_name:
        lines.append(f"  - Wire Model: {_markdown_value(agent.wire_model_name)}")
    lines.append(f"  - Context Window: {_context_window_display(agent.context_window)}")
    lines.append("")


def _active_model_lines(
    model: "AgentModelSummary | None",
    *,
    model_source: str | None,
) -> list[str]:
    provider_line = "unknown"
    model_name = "unknown"
    wire_model_name = None
    context_window = "unknown"
    capabilities = "Capabilities: unknown"
    if model:
        provider_line = _format_provider_line(
            model.provider_display,
            model.provider,
            model.hf_provider,
        )
        model_name = _markdown_value(model.model_name)
        wire_model_name = model.wire_model_name
        context_window = _context_window_display(model.context_window)
        if model.capabilities:
            capabilities = (
                f"Capabilities: {', '.join(_markdown_value(item) for item in model.capabilities)}"
            )

    return [
        "## Active Model",
        f"- Provider: {provider_line}",
        f"- Model: {model_name}",
        *([f"- Model Source: {_markdown_value(model_source)}"] if model_source else []),
        *([f"- Wire Model: {_markdown_value(wire_model_name)}"] if wire_model_name else []),
        f"- Context Window: {context_window}",
        f"- {capabilities}",
        "",
    ]


def _append_object_section(lines: list[str], heading: str, values: JsonObject) -> None:
    if not values:
        return
    lines.append(heading)
    lines.extend(
        f"  - {_markdown_value(key)}: {_markdown_value(value)}" for key, value in values.items()
    )


def _append_client_info_lines(lines: list[str], client: "ClientInfoSummary") -> None:
    lines.extend(["## Client Information", ""])
    if client.name:
        if client.title:
            lines.append(
                f"Client: {_markdown_value(client.title)} ({_markdown_value(client.name)})"
            )
        else:
            lines.append(f"Client: {_markdown_value(client.name)}")
    if client.version:
        lines.append(f"Client Version: {_markdown_value(client.version)}")
    if client.protocol_version:
        lines.append(f"ACP Protocol Version: {_markdown_value(client.protocol_version)}")

    _append_object_section(lines, "Filesystem:", client.filesystem_caps)
    if client.terminal:
        lines.append(f"  - Terminal: {_markdown_value(client.terminal)}")
    _append_object_section(lines, "Meta:", client.meta_caps)

    lines.append("")


def _append_conversation_stats_lines(
    lines: list[str],
    stats: "ConversationStatsSummary",
) -> None:
    lines.append(f"## Conversation Statistics ({_markdown_value(stats.agent_name)})")
    lines.append(f"- Turns: {stats.turns}")
    lines.append(
        "- "
        + format_count_breakdown(
            "Messages",
            stats.message_count,
            user=stats.user_message_count,
            assistant=stats.assistant_message_count,
        )
    )
    lines.append(
        "- "
        + format_count_breakdown(
            "Tool Calls",
            stats.tool_calls,
            successes=stats.tool_successes,
            errors=stats.tool_errors,
        )
    )
    lines.append(f"- {stats.context_usage_line}")

    if stats.total_llm_time_seconds:
        lines.append(f"- Total LLM Time: {format_duration(stats.total_llm_time_seconds)}")
    if stats.conversation_runtime_seconds:
        lines.append(
            "- Conversation Runtime (LLM + tools): "
            f"{format_duration(stats.conversation_runtime_seconds)}"
        )

    if stats.tool_breakdown:
        lines.append("")
        lines.append("### Tool Usage Breakdown")
        lines.extend(
            f"  - {_markdown_value(entry.name)}: {entry.count}" for entry in stats.tool_breakdown
        )


def _append_error_handling_lines(
    lines: list[str],
    error_report: "ErrorHandlingSummary",
) -> None:
    lines.extend(["", "## Error Handling"])
    if not error_report.recent_entries:
        lines.append("_No errors recorded_")
        return
    lines.append(error_report.channel_label)
    lines.append("Recent Entries:")
    lines.extend(f"- {_markdown_value(entry)}" for entry in error_report.recent_entries)


def _append_warning_lines(lines: list[str], warnings: list[str]) -> None:
    if not warnings:
        return
    lines.append("")
    lines.append("## Warnings")
    lines.extend(f"- {_markdown_value(warning)}" for warning in warnings)


def render_status_markdown(summary: "StatusSummary", *, heading: str) -> str:
    lines = [markdown_heading(heading), "", "## Version"]
    lines.append(f"fast-agent-mcp: {summary.fast_agent_version} - https://fast-agent.ai/")
    lines.append("")

    if summary.client_info:
        _append_client_info_lines(lines, summary.client_info)

    if summary.parallel_summary:
        lines.append("## Active Models (Parallel Mode)")
        lines.append("")
        fan_out = summary.parallel_summary.fan_out_agents
        if fan_out:
            lines.append(f"### Fan-Out Agents ({len(fan_out)})")
            for index, agent in enumerate(fan_out, start=1):
                lines.append(f"**{index}. {_markdown_value(agent.agent_name)}**")
                _append_parallel_agent_model_lines(lines, agent)
        else:
            lines.extend(["Fan-Out Agents: none configured", ""])

        fan_in = summary.parallel_summary.fan_in_agent
        if fan_in:
            lines.append(f"### Fan-In Agent: {_markdown_value(fan_in.agent_name)}")
            _append_parallel_agent_model_lines(lines, fan_in)
        else:
            lines.extend(["Fan-In Agent: none configured", ""])
    else:
        lines.extend(
            _active_model_lines(
                summary.model_summary,
                model_source=summary.model_source,
            )
        )

    _append_conversation_stats_lines(lines, summary.conversation_stats)
    lines.extend(["", f"ACP Agent Uptime: {format_duration(summary.uptime_seconds)}"])
    _append_error_handling_lines(lines, summary.error_report)
    _append_warning_lines(lines, summary.warnings)

    return "\n".join(lines)


def render_system_prompt_markdown(
    summary: "SystemPromptSummary",
    *,
    heading: str,
) -> str:
    lines = [markdown_heading(heading), ""]
    if not summary.system_prompt:
        lines.append("No system prompt available for this agent.")
        return "\n".join(lines)

    lines.extend([f"**Agent:** {_markdown_value(summary.agent_name)}", "", summary.system_prompt])
    return "\n".join(lines)


def render_permissions_markdown(summary: "PermissionsSummary") -> str:
    lines = [
        markdown_heading(summary.heading),
        "",
        summary.message,
        "",
        f"Path: {markdown_code_span(summary.path)}",
    ]
    return "\n".join(lines)
