"""Status slash command handlers."""

from __future__ import annotations

import time
from importlib.metadata import version as get_version
from typing import TYPE_CHECKING

from fast_agent.commands.protocols import InstructionAwareAgent
from fast_agent.commands.renderers.status_markdown import (
    render_permissions_markdown,
    render_status_markdown,
    render_system_prompt_markdown,
)
from fast_agent.commands.status_summaries import (
    build_permissions_summary,
    build_status_summary,
    build_system_prompt_summary,
)
from fast_agent.paths import resolve_environment_paths

if TYPE_CHECKING:
    from fast_agent.acp.slash_commands import SlashCommandHandler


async def handle_status(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    normalized = (arguments or "").strip().lower()
    if normalized == "system":
        return await handle_status_system(handler)
    if normalized == "auth":
        return handle_status_auth(handler)
    if normalized == "authreset":
        return handle_status_authreset(handler)

    try:
        fa_version = get_version("fast-agent-mcp")
    except Exception:
        fa_version = "unknown"

    agent = handler._get_current_agent()
    uptime_seconds = max(time.time() - handler._created_at, 0.0)
    summary = build_status_summary(
        fast_agent_version=fa_version,
        agent=agent,
        client_info=handler.client_info,
        client_capabilities=handler.client_capabilities,
        protocol_version=str(handler.protocol_version) if handler.protocol_version is not None else None,
        uptime_seconds=uptime_seconds,
        instance=handler.instance,
    )
    return render_status_markdown(summary, heading="fast-agent ACP status")


async def handle_status_system(handler: "SlashCommandHandler") -> str:
    heading = "# system prompt"

    agent, error = handler._get_current_agent_or_error(heading)
    if error:
        return error

    if handler._instruction_resolver:
        try:
            refreshed = await handler._instruction_resolver(handler.current_agent_name)
        except Exception as exc:
            handler._logger.debug(
                "Failed to refresh session instruction",
                agent_name=handler.current_agent_name,
                error=str(exc),
            )
        else:
            if refreshed:
                handler.update_session_instruction(handler.current_agent_name, refreshed)
                if isinstance(agent, InstructionAwareAgent):
                    handler.update_session_instruction(agent.name, refreshed)

    summary = build_system_prompt_summary(
        agent=agent,
        session_instructions=handler._session_instructions,
        current_agent_name=handler.current_agent_name,
    )
    return render_system_prompt_markdown(summary, heading="system prompt")


def handle_status_auth(handler: "SlashCommandHandler") -> str:
    heading = "permissions"
    auths_path = resolve_environment_paths().permissions_file
    resolved_path = auths_path.resolve()

    if not auths_path.exists():
        summary = build_permissions_summary(
            heading=heading,
            message="No permissions set",
            path=str(resolved_path),
        )
        return render_permissions_markdown(summary)

    try:
        content = auths_path.read_text(encoding="utf-8")
        message = content.strip() if content.strip() else "No permissions set"
        summary = build_permissions_summary(
            heading=heading,
            message=message,
            path=str(resolved_path),
        )
        return render_permissions_markdown(summary)
    except Exception as exc:
        summary = build_permissions_summary(
            heading=heading,
            message=f"Failed to read permissions file: {exc}",
            path=str(resolved_path),
        )
        return render_permissions_markdown(summary)


def handle_status_authreset(handler: "SlashCommandHandler") -> str:
    heading = "reset permissions"
    auths_path = resolve_environment_paths().permissions_file
    resolved_path = auths_path.resolve()

    if not auths_path.exists():
        summary = build_permissions_summary(
            heading=heading,
            message="No permissions file exists.",
            path=str(resolved_path),
        )
        return render_permissions_markdown(summary)

    try:
        auths_path.unlink()
        summary = build_permissions_summary(
            heading=heading,
            message="Permissions file removed successfully.",
            path=str(resolved_path),
        )
        return render_permissions_markdown(summary)
    except Exception as exc:
        summary = build_permissions_summary(
            heading=heading,
            message=f"Failed to remove permissions file: {exc}",
            path=str(resolved_path),
        )
        return render_permissions_markdown(summary)
