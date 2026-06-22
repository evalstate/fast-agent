"""Shell cwd policy preflight for CLI runtime startup."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from .shell_cwd_policy import (
    can_prompt_for_missing_cwd,
    collect_shell_cwd_issues,
    create_missing_shell_cwd_directories,
    effective_missing_shell_cwd_policy,
    format_shell_cwd_issues,
    resolve_missing_shell_cwd_policy,
)

if TYPE_CHECKING:
    from .run_request import AgentRunRequest


def apply_shell_cwd_policy_preflight(fast: Any, request: AgentRunRequest) -> None:
    from fast_agent.config import Settings

    issues = collect_shell_cwd_issues(
        fast.agents,
        shell_runtime_requested=request.shell_runtime,
        no_shell=request.no_shell,
        cwd=Path.cwd(),
    )
    if not issues:
        return

    settings = fast.app.context.config
    configured_policy = (
        settings.shell_execution.missing_cwd_policy if isinstance(settings, Settings) else None
    )
    resolved_policy = resolve_missing_shell_cwd_policy(
        cli_override=request.missing_shell_cwd_policy,
        configured_policy=configured_policy,
    )
    interactive_startup_context = request.mode == "interactive" and request.is_repl
    can_prompt = can_prompt_for_missing_cwd(
        mode=request.mode,
        execution_mode=request.execution_mode or "repl",
        stdin_is_tty=sys.stdin.isatty(),
        tty_device_available=False,
    )
    policy = effective_missing_shell_cwd_policy(resolved_policy, can_prompt=can_prompt)
    issue_lines = format_shell_cwd_issues(issues)

    if policy == "warn":
        emit_startup_notice(
            request,
            format_shell_cwd_policy_message(policy=policy, lines=issue_lines),
        )
        return

    if policy == "error":
        typer.echo(format_shell_cwd_policy_message(policy=policy, lines=issue_lines), err=True)
        raise typer.Exit(1)

    if policy == "ask":
        if interactive_startup_context:
            # Keep interactive confirmation inside the prompt lifecycle for ask mode.
            return
        policy = "warn"

    if policy == "warn":
        emit_startup_notice(
            request,
            format_shell_cwd_policy_message(policy=policy, lines=issue_lines),
        )
        return

    creation_result = create_missing_shell_cwd_directories(issues)
    if creation_result.created_paths:
        created_lines = [
            "Created missing shell cwd directories:",
            *[f" - {path}" for path in creation_result.created_paths],
        ]
        emit_startup_notice(request, "\n".join(created_lines))

    if creation_result.errors:
        error_lines = ["Failed to create one or more shell cwd directories:"]
        error_lines.extend(f" - {item.path}: {item.message}" for item in creation_result.errors)
        typer.echo("\n".join(error_lines), err=True)
        raise typer.Exit(1)

    remaining_issues = collect_shell_cwd_issues(
        fast.agents,
        shell_runtime_requested=request.shell_runtime,
        no_shell=request.no_shell,
        cwd=Path.cwd(),
    )
    if remaining_issues:
        typer.echo(
            format_shell_cwd_policy_message(
                policy="error",
                lines=format_shell_cwd_issues(remaining_issues),
            ),
            err=True,
        )
        raise typer.Exit(1)


def emit_startup_notice(request: AgentRunRequest, message: str) -> None:
    if request.mode == "interactive" and request.is_repl:
        from fast_agent.ui.enhanced_prompt import queue_startup_notice

        queue_startup_notice(message)
        return

    typer.echo(message, err=True)


def format_shell_cwd_policy_message(
    *,
    policy: str,
    lines: list[str],
) -> str:
    title = f"Shell cwd policy ({policy}):"
    return "\n".join([title, *lines])


__all__ = [
    "apply_shell_cwd_policy_preflight",
    "format_shell_cwd_policy_message",
]
