"""Session resume startup helpers for CLI runtime requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import typer

from fast_agent.cli.constants import RESUME_LATEST_SENTINEL
from fast_agent.session.preview import find_last_assistant_preview_text

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session.hydrator import SessionHydrationWarning
    from fast_agent.session.session_manager import ResumeSessionAgentsResult

    from .run_request import AgentRunRequest


class StartupNotice(Protocol):
    def __call__(self, notice: object) -> None: ...


class StartupMarkdownNotice(Protocol):
    def __call__(
        self,
        text: str,
        *,
        title: str | None = None,
        style: str | None = None,
        right_info: str | None = None,
        agent_name: str | None = None,
    ) -> None: ...


DEFERRED_RESUME_WARNING_CODES = frozenset({"git-state-changed"})


def find_last_assistant_text(history: Sequence[object]) -> str | None:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

    typed_history = [message for message in history if isinstance(message, PromptMessageExtended)]
    return find_last_assistant_preview_text(typed_history)


async def resume_session_if_requested(
    agent_app: AgentApp,
    request: AgentRunRequest,
) -> None:
    validate_resume_request(request)
    if not request.resume or request.noenv:
        return

    from fast_agent.ui.enhanced_prompt import queue_startup_markdown_notice, queue_startup_notice

    session_id = resume_session_id(request)
    default_agent = agent_app.resolve_agent()
    result = agent_app.latest_session_restore_result()
    interactive_notice = request.is_repl
    if not result:
        emit_resume_not_found_notice(session_id, interactive_notice, queue_startup_notice)
        return

    emit_resume_status_notices(result, interactive_notice, queue_startup_notice)
    emit_resume_history_summary(result, interactive_notice, queue_startup_notice)

    if result.active_agent is not None:
        request.target_agent_name = result.active_agent

    preview_agent = resume_preview_agent(agent_app, request, default_agent, result.loaded)
    emit_resume_assistant_preview(
        preview_agent,
        interactive_notice,
        queue_startup_markdown_notice,
    )
    emit_deferred_resume_warnings(result.warnings, interactive_notice, queue_startup_notice)


def validate_resume_request(request: AgentRunRequest) -> None:
    if request.noenv and request.resume:
        typer.echo("Error: --resume cannot be used with --noenv.", err=True)
        raise typer.Exit(1)


def resume_session_id(request: AgentRunRequest) -> str | None:
    return None if request.resume in ("", RESUME_LATEST_SENTINEL) else request.resume


def plain_resume_notice(notice: str) -> str:
    return (
        notice.replace("[yellow]", "")
        .replace("[/yellow]", "")
        .replace("[dim]", "")
        .replace("[/dim]", "")
        .replace("[cyan]", "")
        .replace("[/cyan]", "")
    )


def emit_resume_notice(
    notice: str,
    *,
    interactive_notice: bool,
    queue_startup_notice: StartupNotice,
    plain_notice: str | None = None,
) -> None:
    if interactive_notice:
        queue_startup_notice(notice)
    else:
        typer.echo(
            plain_notice if plain_notice is not None else plain_resume_notice(notice),
            err=True,
        )


def emit_resume_not_found_notice(
    session_id: str | None,
    interactive_notice: bool,
    queue_startup_notice: StartupNotice,
) -> None:
    notice = (
        f"[yellow]Session not found:[/yellow] {session_id}"
        if session_id
        else "[yellow]No sessions found to resume.[/yellow]"
    )
    emit_resume_notice(
        notice,
        interactive_notice=interactive_notice,
        queue_startup_notice=queue_startup_notice,
    )


def emit_resume_status_notices(
    result: ResumeSessionAgentsResult,
    interactive_notice: bool,
    queue_startup_notice: StartupNotice,
) -> None:
    session = result.session
    session_time = session.info.last_activity.strftime("%y-%m-%d %H:%M")
    resume_notice = (
        f"[dim]Resumed session[/dim] [cyan]{session.info.name}[/cyan] [dim]({session_time})[/dim]"
    )
    emit_resume_notice(
        resume_notice,
        interactive_notice=interactive_notice,
        queue_startup_notice=queue_startup_notice,
        plain_notice=f"Resumed session {session.info.name} ({session_time})",
    )
    emit_missing_resume_agents_notice(
        result.missing_agents,
        interactive_notice,
        queue_startup_notice,
    )
    emit_resume_warnings(
        result.warnings,
        interactive_notice,
        queue_startup_notice,
        deferred=False,
    )
    emit_resume_usage_notices(result.usage_notices, interactive_notice, queue_startup_notice)


def emit_missing_resume_agents_notice(
    missing_agents: list[str],
    interactive_notice: bool,
    queue_startup_notice: StartupNotice,
) -> None:
    if not missing_agents:
        return
    missing_list = ", ".join(sorted(missing_agents))
    emit_resume_notice(
        f"[yellow]Missing agents from session:[/yellow] {missing_list}",
        interactive_notice=interactive_notice,
        queue_startup_notice=queue_startup_notice,
        plain_notice=f"Missing agents from session: {missing_list}",
    )


def emit_resume_warnings(
    warnings: list[SessionHydrationWarning],
    interactive_notice: bool,
    queue_startup_notice: StartupNotice,
    *,
    deferred: bool | None = None,
) -> None:
    for warning in warnings:
        if warning.code == "missing-agent":
            continue
        if deferred is not None and (warning.code in DEFERRED_RESUME_WARNING_CODES) != deferred:
            continue
        emit_resume_notice(
            f"[yellow]{warning.message}[/yellow]",
            interactive_notice=interactive_notice,
            queue_startup_notice=queue_startup_notice,
            plain_notice=warning.message,
        )


def emit_deferred_resume_warnings(
    warnings: list[SessionHydrationWarning],
    interactive_notice: bool,
    queue_startup_notice: StartupNotice,
) -> None:
    emit_resume_warnings(
        warnings,
        interactive_notice,
        queue_startup_notice,
        deferred=True,
    )


def emit_resume_usage_notices(
    usage_notices: list[str],
    interactive_notice: bool,
    queue_startup_notice: StartupNotice,
) -> None:
    for usage_notice in usage_notices:
        if not usage_notice:
            continue
        emit_resume_notice(
            usage_notice,
            interactive_notice=interactive_notice,
            queue_startup_notice=queue_startup_notice,
        )


def emit_resume_history_summary(
    result: ResumeSessionAgentsResult,
    interactive_notice: bool,
    queue_startup_notice: StartupNotice,
) -> None:
    if result.missing_agents or not result.loaded:
        from fast_agent.session import format_history_summary, summarize_session_histories

        summary = summarize_session_histories(result.session)
        summary_text = format_history_summary(summary)
        if summary_text:
            emit_resume_notice(
                f"[dim]Available histories:[/dim] {summary_text}",
                interactive_notice=interactive_notice,
                queue_startup_notice=queue_startup_notice,
                plain_notice=f"Available histories: {summary_text}",
            )


def resume_preview_agent(
    agent_app: AgentApp,
    request: AgentRunRequest,
    default_agent: AgentProtocol,
    loaded: Mapping[str, Path],
) -> AgentProtocol:
    preview_agent = default_agent
    default_name = default_agent.name
    preview_name = request.target_agent_name or default_name
    if loaded and preview_name not in loaded:
        first_loaded_name = next(iter(loaded))
        preview_agent = agent_app.get_agent(first_loaded_name) or default_agent
    elif preview_name is not None:
        preview_agent = agent_app.get_agent(preview_name) or default_agent
    return preview_agent


def emit_resume_assistant_preview(
    preview_agent: AgentProtocol,
    interactive_notice: bool,
    queue_startup_markdown_notice: StartupMarkdownNotice,
) -> None:
    preview_history = preview_agent.message_history
    assistant_text = find_last_assistant_text(list(preview_history))
    if assistant_text:
        if interactive_notice:
            queue_startup_markdown_notice(
                assistant_text,
                title="Last assistant message",
                right_info="session",
                agent_name=preview_agent.name,
            )
        else:
            typer.echo("Last assistant message:", err=True)
            typer.echo(assistant_text, err=True)
