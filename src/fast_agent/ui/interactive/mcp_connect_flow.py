"""MCP connect execution flow for interactive prompt."""

from __future__ import annotations

import asyncio
import signal
import threading
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from rich import print as rich_print
from rich.text import Text

from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.ui.console import console, ensure_blocking_console

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.commands.results import CommandOutcome
    from fast_agent.core.agent_app import AgentApp


async def handle_mcp_connect(
    *,
    context: "CommandContext",
    prompt_provider: "AgentApp",
    agent: str,
    runtime_target: str,
    target_text: str,
    server_name: str | None,
) -> "CommandOutcome | None":
    label = server_name or target_text.split(maxsplit=1)[0]
    attached_before_connect: set[str] = set()
    try:
        attached_before_connect = set(await prompt_provider.list_attached_mcp_servers(agent))
    except Exception:
        attached_before_connect = set()

    async def _handle_mcp_connect_cancel() -> None:
        cancel_server_name = server_name
        if not cancel_server_name:
            try:
                parsed_runtime = mcp_runtime_handlers.parse_connect_input(runtime_target)
                cancel_server_name = parsed_runtime.server_name
                if not cancel_server_name:
                    mode = mcp_runtime_handlers.infer_connect_mode(parsed_runtime.target_text)
                    cancel_server_name = mcp_runtime_handlers._infer_server_name(
                        parsed_runtime.target_text,
                        mode,
                    )
            except Exception:
                cancel_server_name = None

        should_detach_on_cancel = bool(cancel_server_name) and (
            cancel_server_name not in attached_before_connect
        )
        if should_detach_on_cancel and cancel_server_name:
            try:
                await prompt_provider.detach_mcp_server(agent, cancel_server_name)
            except (Exception, asyncio.CancelledError):
                pass

        rich_print()
        rich_print("[yellow]MCP connect cancelled; returned to prompt.[/yellow]")

    with console.status(
        f"[yellow]Starting MCP server '{label}'...[/yellow]",
        spinner="dots",
    ) as mcp_connect_status:
        oauth_link_shown = False

        async def _emit_mcp_progress(message: str) -> None:
            nonlocal oauth_link_shown
            if message.startswith("Open this link to authorize:"):
                auth_url = message.split(":", 1)[1].strip()
                if auth_url:
                    oauth_link_shown = True
                    rich_print("[bold]Open this link to authorize:[/bold]")
                    ensure_blocking_console()
                    console.print(
                        f"[link={auth_url}]{auth_url}[/link]",
                        style="bright_cyan",
                        soft_wrap=True,
                    )
                    return
            mcp_connect_status.update(status=Text(message, style="yellow"))

        connect_task = asyncio.create_task(
            mcp_runtime_handlers.handle_mcp_connect(
                context,
                manager=prompt_provider,
                agent_name=agent,
                target_text=runtime_target,
                on_progress=_emit_mcp_progress,
            )
        )

        previous_sigint_handler: Any | None = None
        sigint_handler_installed = False

        if threading.current_thread() is threading.main_thread():
            previous_sigint_handler = signal.getsignal(signal.SIGINT)

            def _sigint_cancel_connect(_signum: int, _frame: Any) -> None:
                if not connect_task.done():
                    connect_task.cancel()

            signal.signal(signal.SIGINT, _sigint_cancel_connect)
            sigint_handler_installed = True

        try:
            outcome = await connect_task
        except (KeyboardInterrupt, asyncio.CancelledError):
            if not connect_task.done():
                connect_task.cancel()
                with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                    await asyncio.wait_for(connect_task, timeout=1.0)

            await _handle_mcp_connect_cancel()
            return None
        finally:
            if sigint_handler_installed and previous_sigint_handler is not None:
                signal.signal(signal.SIGINT, previous_sigint_handler)

    if oauth_link_shown:
        outcome.messages = [
            message
            for message in outcome.messages
            if not str(message.text).startswith("OAuth authorization link:")
        ]

    return outcome
