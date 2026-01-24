"""Markdown renderers for ACP status output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.utils.time import format_duration

if TYPE_CHECKING:
    from fast_agent.commands.status_summaries import (
        PermissionsSummary,
        StatusSummary,
        SystemPromptSummary,
    )


def _format_provider_line(provider_display: str, provider: str, hf_provider: str | None) -> str:
    line = provider
    if provider_display != "unknown":
        line = f"{provider_display} ({provider})"
    if hf_provider:
        line = f"{line} / {hf_provider}"
    return line


def render_status_markdown(summary: "StatusSummary", *, heading: str) -> str:
    lines = [f"# {heading}", "", "## Version"]
    lines.append(
        f"fast-agent-mcp: {summary.fast_agent_version} - https://fast-agent.ai/"
    )
    lines.append("")

    if summary.client_info:
        client = summary.client_info
        lines.extend(["## Client Information", ""])
        if client.name:
            if client.title:
                lines.append(f"Client: {client.title} ({client.name})")
            else:
                lines.append(f"Client: {client.name}")
        if client.version:
            lines.append(f"Client Version: {client.version}")
        if client.protocol_version:
            lines.append(f"ACP Protocol Version: {client.protocol_version}")

        if client.filesystem_caps:
            for key, value in client.filesystem_caps.items():
                lines.append(f"  - {key}: {value}")
        if client.terminal:
            lines.append(f"  - Terminal: {client.terminal}")
        if client.meta_caps:
            lines.append("Meta:")
            for key, value in client.meta_caps.items():
                lines.append(f"  - {key}: {value}")

        lines.append("")

    if summary.parallel_summary:
        lines.append("## Active Models (Parallel Mode)")
        lines.append("")
        fan_out = summary.parallel_summary.fan_out_agents
        if fan_out:
            lines.append(f"### Fan-Out Agents ({len(fan_out)})")
            for index, agent in enumerate(fan_out, start=1):
                lines.append(f"**{index}. {agent.agent_name}**")
                lines.append(
                    f"  - Provider: {agent.provider_display or agent.provider}"
                )
                lines.append(f"  - Model: {agent.model_name}")
                if agent.context_window:
                    lines.append(
                        f"  - Context Window: {agent.context_window} tokens"
                    )
                else:
                    lines.append("  - Context Window: unknown")
                lines.append("")
        else:
            lines.extend(["Fan-Out Agents: none configured", ""])

        fan_in = summary.parallel_summary.fan_in_agent
        if fan_in:
            lines.append(f"### Fan-In Agent: {fan_in.agent_name}")
            lines.append(f"  - Provider: {fan_in.provider_display or fan_in.provider}")
            lines.append(f"  - Model: {fan_in.model_name}")
            if fan_in.context_window:
                lines.append(
                    f"  - Context Window: {fan_in.context_window} tokens"
                )
            else:
                lines.append("  - Context Window: unknown")
            lines.append("")
        else:
            lines.extend(["Fan-In Agent: none configured", ""])
    else:
        model = summary.model_summary
        provider_line = "unknown"
        model_name = "unknown"
        context_window = "unknown"
        capabilities = "Capabilities: unknown"
        hf_provider = None
        provider_display = "unknown"
        provider = "unknown"
        if model:
            provider_display = model.provider_display
            provider = model.provider
            hf_provider = model.hf_provider
            provider_line = _format_provider_line(provider_display, provider, hf_provider)
            model_name = model.model_name
            if model.context_window:
                context_window = f"{model.context_window} tokens"
            cap_list = model.capabilities
            if cap_list:
                capabilities = f"Capabilities: {', '.join(cap_list)}"
        lines.extend(
            [
                "## Active Model",
                f"- Provider: {provider_line}",
                f"- Model: {model_name}",
                f"- Context Window: {context_window}",
                f"- {capabilities}",
                "",
            ]
        )

    stats = summary.conversation_stats
    lines.append(f"## Conversation Statistics ({stats.agent_name})")
    lines.append(f"- Turns: {stats.turns}")
    lines.append(
        "- Messages: "
        f"{stats.message_count} (user: {stats.user_message_count}, "
        f"assistant: {stats.assistant_message_count})"
    )
    lines.append(
        "- Tool Calls: "
        f"{stats.tool_calls} (successes: {stats.tool_successes}, "
        f"errors: {stats.tool_errors})"
    )
    lines.append(f"- {stats.context_usage_line}")

    if stats.total_llm_time_seconds:
        lines.append(
            f"- Total LLM Time: {format_duration(stats.total_llm_time_seconds)}"
        )
    if stats.conversation_runtime_seconds:
        lines.append(
            "- Conversation Runtime (LLM + tools): "
            f"{format_duration(stats.conversation_runtime_seconds)}"
        )

    if stats.tool_breakdown:
        lines.append("")
        lines.append("### Tool Usage Breakdown")
        for entry in stats.tool_breakdown:
            lines.append(f"  - {entry.name}: {entry.count}")

    lines.extend(["", f"ACP Agent Uptime: {format_duration(summary.uptime_seconds)}"])
    lines.extend(["", "## Error Handling"])

    if summary.error_report.recent_entries:
        lines.append(summary.error_report.channel_label)
        lines.append("Recent Entries:")
        lines.extend(f"- {entry}" for entry in summary.error_report.recent_entries)
    else:
        lines.append("_No errors recorded_")

    if summary.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in summary.warnings)

    return "\n".join(lines)


def render_system_prompt_markdown(
    summary: "SystemPromptSummary",
    *,
    heading: str,
) -> str:
    lines = [f"# {heading}", ""]
    if not summary.system_prompt:
        lines.append("No system prompt available for this agent.")
        return "\n".join(lines)

    lines.extend([f"**Agent:** {summary.agent_name}", "", summary.system_prompt])
    return "\n".join(lines)


def render_permissions_markdown(summary: "PermissionsSummary") -> str:
    lines = [f"# {summary.heading}", "", summary.message, "", f"Path: `{summary.path}`"]
    return "\n".join(lines)
