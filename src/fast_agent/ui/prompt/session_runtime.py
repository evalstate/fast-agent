"""Runtime helpers for interactive prompt session lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich import print as rich_print

from fast_agent.ui.command_payloads import CommandPayload, InterruptCommand
from fast_agent.ui.prompt_marks import emit_prompt_mark

if TYPE_CHECKING:
    from collections.abc import Callable


def build_prompt_style() -> Style:
    """Build the shared prompt-toolkit style used by enhanced input."""
    return Style.from_dict(
        {
            "completion-menu.completion": "bg:#ansiblack #ansigreen",
            "completion-menu.completion.current": "bg:#ansiblack bold #ansigreen",
            "completion-menu.meta.completion": "bg:#ansiblack #ansiblue",
            "completion-menu.meta.completion.current": "bg:#ansibrightblack #ansiblue",
            "bottom-toolbar": "#ansiblack bg:#ansigray",
            "shell-command": "#ansired",
            "comment-command": "#ansiblue",
        }
    )


def create_prompt_session(*, history, completer, lexer, multiline_filter, toolbar, style) -> PromptSession:
    """Create a configured PromptSession for enhanced input."""
    return PromptSession(
        history=history,
        completer=completer,
        lexer=lexer,
        complete_while_typing=True,
        multiline=multiline_filter,
        complete_in_thread=True,
        mouse_support=False,
        bottom_toolbar=toolbar,
        style=style,
        erase_when_done=True,
    )


async def run_prompt_once(
    *,
    session: PromptSession,
    agent_name: str,
    default_buffer: str,
    resolve_prompt_text: "Callable[[], object]",
    parse_special_input: "Callable[[str], str | CommandPayload]",
) -> str | CommandPayload:
    """Run a single prompt cycle and normalize command/signal outcomes."""
    prompt_mark_started = False
    accept_state: dict[str, Any] = {}
    prompt_shutdown_warn_seconds = 0.5
    buffer = session.default_buffer
    original_accept_handler = buffer.accept_handler

    def _track_accept(buffer_obj) -> bool:
        accept_state["accepted_at"] = time.perf_counter()
        accept_state["text"] = buffer_obj.text
        accept_state["completer"] = type(buffer_obj.completer).__name__
        accept_state["had_completions"] = buffer_obj.complete_state is not None
        if original_accept_handler is not None:
            return original_accept_handler(buffer_obj)
        return True

    buffer.accept_handler = _track_accept
    try:
        emit_prompt_mark("A")
        prompt_mark_started = True
        result = await session.prompt_async(
            resolve_prompt_text,
            default=default_buffer,
            set_exception_handler=False,
        )
        prompt_returned_at = time.perf_counter()
        emit_prompt_mark("B")

        stripped = result.lstrip()
        accepted_at = accept_state.get("accepted_at")
        if accepted_at:
            shutdown_delay = prompt_returned_at - accepted_at
            if shutdown_delay >= prompt_shutdown_warn_seconds and stripped.startswith("!"):
                text_preview = str(accept_state.get("text") or "").strip()
                if len(text_preview) > 80:
                    text_preview = text_preview[:77] + "..."
                rich_print(
                    "[yellow]Prompt shutdown delay[/yellow] "
                    f"{shutdown_delay:.2f}s | "
                    f"completer={accept_state.get('completer')} "
                    f"completions_active={accept_state.get('had_completions')} "
                    f"cwd={Path.cwd()} "
                    f"input={text_preview!r}"
                )

        if stripped.startswith("/"):
            rich_print(f"[dim]{agent_name} ❯ {stripped.splitlines()[0]}[/dim]")
        elif stripped.startswith("!"):
            rich_print(f"[dim]{agent_name} ❯ {stripped.splitlines()[0]}[/dim]")

        return parse_special_input(result)
    except KeyboardInterrupt:
        if prompt_mark_started:
            emit_prompt_mark("B")
        return InterruptCommand()
    except EOFError:
        if prompt_mark_started:
            emit_prompt_mark("B")
        return "STOP"
    except Exception as exc:
        if prompt_mark_started:
            emit_prompt_mark("B")
        print(f"\nInput error: {type(exc).__name__}: {exc}")
        return "STOP"


def start_toolbar_switch_task(session: PromptSession, delay_seconds: float) -> asyncio.Task[None]:
    """Start delayed toolbar invalidation task used by shell mode."""

    async def _invalidate_toolbar_on_switch() -> None:
        await asyncio.sleep(delay_seconds)
        if session.app and not session.app.is_done:
            session.app.invalidate()

    return asyncio.create_task(_invalidate_toolbar_on_switch())


async def cleanup_prompt_session(
    *,
    session: PromptSession,
    toolbar_switch_task: asyncio.Task[None] | None,
) -> None:
    """Cancel helper tasks and terminate active prompt app state."""
    if toolbar_switch_task and not toolbar_switch_task.done():
        toolbar_switch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await toolbar_switch_task

    if session.app.is_running:
        session.app.exit()
