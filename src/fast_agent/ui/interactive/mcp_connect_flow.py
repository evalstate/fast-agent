"""MCP connect execution flow for interactive prompt."""

from __future__ import annotations

import asyncio
import signal
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich import print as rich_print
from rich.text import Text

from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.mcp.connect_targets import ParsedMcpConnectRequest, infer_server_name
from fast_agent.ui.console import console, ensure_blocking_console

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.commands.results import CommandOutcome
    from fast_agent.core.agent_app import AgentApp


@dataclass(slots=True)
class _McpConnectProgressState:
    oauth_link_shown: bool = False


def _connect_label(request: ParsedMcpConnectRequest) -> str:
    return request.target.server_name or infer_server_name(request.target)


async def _attached_servers_before_connect(
    prompt_provider: "AgentApp",
    agent: str,
) -> set[str]:
    try:
        return set(await prompt_provider.list_attached_mcp_servers(agent))
    except Exception:
        return set()


async def _handle_mcp_connect_cancel(
    *,
    prompt_provider: "AgentApp",
    agent: str,
    request: ParsedMcpConnectRequest,
    attached_before_connect: set[str],
) -> None:
    cancel_server_name = _connect_label(request)
    should_detach_on_cancel = bool(cancel_server_name) and (
        cancel_server_name not in attached_before_connect
    )
    if should_detach_on_cancel and cancel_server_name:
        with suppress(Exception, asyncio.CancelledError):
            await prompt_provider.detach_mcp_server(agent, cancel_server_name)

    rich_print()
    rich_print("[yellow]MCP connect cancelled; returned to prompt.[/yellow]")


def _make_mcp_progress_emitter(
    *,
    mcp_connect_status: Any,
    progress_state: _McpConnectProgressState,
):
    async def _emit_mcp_progress(message: str) -> None:
        if message.startswith("Open this link to authorize:"):
            auth_url = message.split(":", 1)[1].strip()
            if auth_url:
                progress_state.oauth_link_shown = True
                rich_print("[bold]Open this link to authorize:[/bold]")
                ensure_blocking_console()
                console.print(
                    f"[link={auth_url}]{auth_url}[/link]",
                    style="bright_cyan",
                    soft_wrap=True,
                )
                return
        mcp_connect_status.update(status=Text(message, style="yellow"))

    return _emit_mcp_progress


def _install_sigint_cancel_handler(connect_task: asyncio.Task[Any]) -> Any | None:
    if threading.current_thread() is not threading.main_thread():
        return None

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def _sigint_cancel_connect(_signum: int, _frame: Any) -> None:
        if not connect_task.done():
            connect_task.cancel()

    signal.signal(signal.SIGINT, _sigint_cancel_connect)
    return previous_sigint_handler


def _restore_sigint_handler(previous_sigint_handler: Any | None) -> None:
    if previous_sigint_handler is not None:
        signal.signal(signal.SIGINT, previous_sigint_handler)


async def _cancel_connect_task(connect_task: asyncio.Task[Any]) -> None:
    if connect_task.done():
        return
    connect_task.cancel()
    with suppress(asyncio.CancelledError, asyncio.TimeoutError):
        await asyncio.wait_for(connect_task, timeout=1.0)


def _remove_duplicate_oauth_link_message(outcome: "CommandOutcome") -> None:
    outcome.messages = [
        message
        for message in outcome.messages
        if not str(message.text).startswith("OAuth authorization link:")
    ]


async def handle_mcp_connect(
    *,
    context: "CommandContext",
    prompt_provider: "AgentApp",
    agent: str,
    request: ParsedMcpConnectRequest,
) -> "CommandOutcome | None":
    label = _connect_label(request)
    attached_before_connect = await _attached_servers_before_connect(prompt_provider, agent)

    with console.status(
        f"[yellow]Starting MCP server '{label}'...[/yellow]",
        spinner="dots",
    ) as mcp_connect_status:
        progress_state = _McpConnectProgressState()
        emit_progress = _make_mcp_progress_emitter(
            mcp_connect_status=mcp_connect_status,
            progress_state=progress_state,
        )

        connect_task = asyncio.create_task(
            mcp_runtime_handlers.handle_mcp_connect(
                context,
                manager=prompt_provider,
                agent_name=agent,
                request=request,
                on_progress=emit_progress,
            )
        )

        previous_sigint_handler = _install_sigint_cancel_handler(connect_task)
        try:
            outcome = await connect_task
        except (KeyboardInterrupt, asyncio.CancelledError):
            await _cancel_connect_task(connect_task)
            await _handle_mcp_connect_cancel(
                prompt_provider=prompt_provider,
                agent=agent,
                request=request,
                attached_before_connect=attached_before_connect,
            )
            return None
        finally:
            _restore_sigint_handler(previous_sigint_handler)

    if progress_state.oauth_link_shown:
        _remove_duplicate_oauth_link_message(outcome)

    return outcome
